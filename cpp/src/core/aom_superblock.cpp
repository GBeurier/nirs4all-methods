// SPDX-License-Identifier: CECILL-2.1
//
// Sums, means and the argmin follow the numpy order of the n4m reference
// (pairwise sums over contiguous buffers, sequential column means, first
// minimum); the Ridge, PLS and Ridge-PLS heads are the native kernels the
// reference calls.

#include "core/aom_superblock.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <numeric>
#include <utility>
#include <vector>

#include "core/aom_operators.hpp"
#include "core/config.hpp"
#include "core/extra_pls.hpp"
#include "core/ridge.hpp"
#include "core/sweep.hpp"

namespace n4m::core {
namespace {

using Vec = std::vector<double>;
using Index = std::vector<std::size_t>;

constexpr double kTiny = 1e-30;

n4m_matrix_view_t view_of(Vec& v, std::size_t rows, std::size_t cols) noexcept {
    n4m_matrix_view_t view{};
    view.data = v.data();
    view.rows = static_cast<std::int64_t>(rows);
    view.cols = static_cast<std::int64_t>(cols);
    view.row_stride = view.cols;
    view.col_stride = 1;
    view.dtype = N4M_DTYPE_F64;
    return view;
}

n4m_status_t dense(Context& ctx, const n4m_matrix_view_t& V, const char* name, Vec& out) {
    if (V.dtype != N4M_DTYPE_F64 || V.data == nullptr || V.rows <= 0 || V.cols <= 0) {
        ctx.set_errorf("%s must be a non-empty float64 matrix", name);
        return N4M_ERR_INVALID_ARGUMENT;
    }
    const auto* src = static_cast<const double*>(V.data);
    out.resize(static_cast<std::size_t>(V.rows * V.cols));
    for (std::int64_t i = 0; i < V.rows; ++i) {
        for (std::int64_t j = 0; j < V.cols; ++j) {
            out[static_cast<std::size_t>(i * V.cols + j)] =
                src[i * V.row_stride + j * V.col_stride];
        }
    }
    return N4M_OK;
}

// numpy's pairwise summation of a contiguous buffer (np.sum, np.mean).
double pairwise_sum(const double* a, std::size_t n) {
    if (n < 8) {
        double res = 0.0;
        for (std::size_t i = 0; i < n; ++i) res += a[i];
        return res;
    }
    if (n <= 128) {
        double r[8];
        for (std::size_t j = 0; j < 8; ++j) r[j] = a[j];
        std::size_t i = 8;
        for (; i < n - (n % 8); i += 8) {
            for (std::size_t j = 0; j < 8; ++j) r[j] += a[i + j];
        }
        double res = ((r[0] + r[1]) + (r[2] + r[3])) + ((r[4] + r[5]) + (r[6] + r[7]));
        for (; i < n; ++i) res += a[i];
        return res;
    }
    std::size_t half = n / 2;
    half -= half % 8;
    return pairwise_sum(a, half) + pairwise_sum(a + half, n - half);
}

double mean_of(const Vec& v) {
    return pairwise_sum(v.data(), v.size()) / static_cast<double>(v.size());
}

// Column means with rows accumulated in order (numpy mean over axis 0).
Vec column_means(const Vec& A, std::size_t rows, std::size_t cols) {
    Vec m(cols, 0.0);
    for (std::size_t i = 0; i < rows; ++i) {
        for (std::size_t j = 0; j < cols; ++j) m[j] += A[i * cols + j];
    }
    for (double& v : m) v /= static_cast<double>(rows);
    return m;
}

Vec centered(const Vec& A, std::size_t rows, std::size_t cols, bool center) {
    Vec out(A);
    if (!center) return out;
    const Vec m = column_means(A, rows, cols);
    for (std::size_t i = 0; i < rows; ++i) {
        for (std::size_t j = 0; j < cols; ++j) out[i * cols + j] -= m[j];
    }
    return out;
}

Vec take_rows(const Vec& A, std::size_t cols, const Index& rows) {
    Vec out(rows.size() * cols);
    for (std::size_t r = 0; r < rows.size(); ++r) {
        std::copy_n(A.begin() + static_cast<std::ptrdiff_t>(rows[r] * cols), cols,
                    out.begin() + static_cast<std::ptrdiff_t>(r * cols));
    }
    return out;
}

double frobenius(const Vec& A) {
    double acc = 0.0;
    for (double v : A) acc += v * v;
    return std::sqrt(acc);
}

// A A^T of a rows x cols matrix, exactly symmetric.
Vec gram(const Vec& A, std::size_t rows, std::size_t cols) {
    Vec K(rows * rows);
    for (std::size_t i = 0; i < rows; ++i) {
        for (std::size_t k = i; k < rows; ++k) {
            double acc = 0.0;
            for (std::size_t j = 0; j < cols; ++j) acc += A[i * cols + j] * A[k * cols + j];
            K[i * rows + k] = acc;
            K[k * rows + i] = acc;
        }
    }
    return K;
}

// <K, T>_F / (||K|| ||T||), zero when either norm vanishes.
double alignment(const Vec& K, double k_norm, const Vec& T, double t_norm) {
    if (!(k_norm > kTiny && t_norm > kTiny)) return 0.0;
    Vec prod(K.size());
    for (std::size_t i = 0; i < K.size(); ++i) prod[i] = K[i] * T[i];
    return pairwise_sum(prod.data(), prod.size()) / (k_norm * t_norm);
}

// Indices by decreasing score, ties in index order.
Index order_desc(const Vec& scores) {
    Index idx(scores.size());
    std::iota(idx.begin(), idx.end(), std::size_t{0});
    std::stable_sort(idx.begin(), idx.end(),
                     [&](std::size_t a, std::size_t b) { return scores[a] > scores[b]; });
    return idx;
}

struct Fold {
    Index train;
    Index valid;
};

n4m_status_t make_folds(Context& ctx, const std::vector<std::int32_t>& ids, std::size_t n,
                        std::vector<Fold>& out) {
    if (ids.size() != n) {
        ctx.set_error("fold_ids length must match X rows");
        return N4M_ERR_SHAPE_MISMATCH;
    }
    const std::int32_t k = 1 + *std::max_element(ids.begin(), ids.end());
    std::vector<Fold> folds(static_cast<std::size_t>(k));
    for (std::size_t f = 0; f < folds.size(); ++f) {
        for (std::size_t i = 0; i < n; ++i) {
            (static_cast<std::size_t>(ids[i]) == f ? folds[f].valid : folds[f].train).push_back(i);
        }
        if (folds[f].train.empty() || folds[f].valid.empty()) {
            ctx.set_error("every fold must contain train and validation rows");
            return N4M_ERR_INVALID_ARGUMENT;
        }
    }
    out = std::move(folds);
    return N4M_OK;
}

std::size_t min_train_rows(const std::vector<Fold>& folds) {
    std::size_t m = std::numeric_limits<std::size_t>::max();
    for (const Fold& f : folds) m = std::min(m, f.train.size());
    return m;
}

n4m_status_t check_components(Context& ctx, const std::vector<std::int32_t>& components,
                              std::size_t min_train, std::size_t width) {
    if (components.empty()) {
        ctx.set_error("pls_components must contain at least one value");
        return N4M_ERR_INVALID_ARGUMENT;
    }
    if (min_train < 2) {
        ctx.set_error("every train fold must contain at least 2 rows");
        return N4M_ERR_INVALID_ARGUMENT;
    }
    const std::size_t max_components = std::min(min_train - 1, width);
    for (std::int32_t c : components) {
        if (c < 1 || static_cast<std::size_t>(c) > max_components) {
            ctx.set_error("PLS components must be in [1, min(train_rows - 1, features)]");
            return N4M_ERR_INVALID_ARGUMENT;
        }
    }
    return N4M_OK;
}

n4m_status_t check_lambdas(Context& ctx, const std::vector<double>& lambdas, bool strict,
                           const char* message) {
    bool ok = !lambdas.empty();
    for (double v : lambdas) ok = ok && std::isfinite(v) && (strict ? v > 0.0 : v >= 0.0);
    if (!ok) {
        ctx.set_error(message);
        return N4M_ERR_INVALID_ARGUMENT;
    }
    return N4M_OK;
}

// Outputs of `ops[which[k]]` on A (rows x cols), one row-major block each.
n4m_status_t operator_outputs(Context& ctx, const std::vector<OperatorEntry>& ops,
                              const Index& which, Vec& A, std::size_t rows, std::size_t cols,
                              std::vector<Vec>& out) {
    std::vector<Vec> blocks(which.size());
    const n4m_matrix_view_t view = view_of(A, rows, cols);
    for (std::size_t k = 0; k < which.size(); ++k) {
        const n4m_status_t st = transform_aom_strict_operator(ctx, ops[which[k]], view, blocks[k]);
        if (st != N4M_OK) return st;
    }
    out = std::move(blocks);
    return N4M_OK;
}

// The operator matrix M (cols x cols) with op(x) = x M: the operator applied
// to the identity rows.
n4m_status_t operator_matrix(Context& ctx, const OperatorEntry& op, std::size_t cols, Vec& out) {
    Vec eye(cols * cols, 0.0);
    for (std::size_t i = 0; i < cols; ++i) eye[i * cols + i] = 1.0;
    return transform_aom_strict_operator(ctx, op, view_of(eye, cols, cols), out);
}

// Blocks side by side: row i of block k at columns k * p .. (k + 1) * p.
Vec superblock(const std::vector<Vec>& blocks, std::size_t rows, std::size_t p) {
    const std::size_t width = blocks.size() * p;
    Vec Z(rows * width);
    for (std::size_t k = 0; k < blocks.size(); ++k) {
        for (std::size_t i = 0; i < rows; ++i) {
            std::copy_n(blocks[k].begin() + static_cast<std::ptrdiff_t>(i * p), p,
                        Z.begin() + static_cast<std::ptrdiff_t>(i * width + k * p));
        }
    }
    return Z;
}

// Train-fold column means and per-block factors; a factor multiplies its block
// after centering.
struct Scaling {
    Vec mean;
    Vec factors;
};

void scale_blocks(Vec& Z, std::size_t rows, std::size_t p, const Vec& factors) {
    const std::size_t width = factors.size() * p;
    for (std::size_t i = 0; i < rows; ++i) {
        for (std::size_t k = 0; k < factors.size(); ++k) {
            for (std::size_t j = 0; j < p; ++j) Z[i * width + k * p + j] *= factors[k];
        }
    }
}

void subtract_mean(Vec& Z, std::size_t rows, const Vec& mean) {
    const std::size_t width = mean.size();
    for (std::size_t i = 0; i < rows; ++i) {
        for (std::size_t j = 0; j < width; ++j) Z[i * width + j] -= mean[j];
    }
}

// Centers Z in place (when center) and scales each block to unit RMS (when
// rms and the block RMS exceeds 1e-12).
Scaling center_scale(Vec& Z, std::size_t rows, std::size_t n_blocks, std::size_t p, bool center,
                     bool rms) {
    const std::size_t width = n_blocks * p;
    Scaling s;
    s.mean = center ? column_means(Z, rows, width) : Vec(width, 0.0);
    subtract_mean(Z, rows, s.mean);
    s.factors.assign(n_blocks, 1.0);
    if (rms) {
        Vec sq(rows * p);
        for (std::size_t k = 0; k < n_blocks; ++k) {
            for (std::size_t i = 0; i < rows; ++i) {
                for (std::size_t j = 0; j < p; ++j) {
                    const double v = Z[i * width + k * p + j];
                    sq[i * p + j] = v * v;
                }
            }
            const double r = std::sqrt(mean_of(sq));
            if (r > 1e-12) s.factors[k] = 1.0 / r;
        }
    }
    scale_blocks(Z, rows, p, s.factors);
    return s;
}

// Kernel-target alignment of every block and the non-negative simplex weights
// of the top_k blocks.
void mkl_weights(const std::vector<Vec>& blocks, std::size_t rows, std::size_t p, const Vec& Y,
                 std::size_t q, std::int32_t top_k, bool center_x, bool center_y, Vec& weights,
                 Vec& scores) {
    const std::size_t n_ops = blocks.size();
    const std::size_t keep = std::min(static_cast<std::size_t>(top_k), n_ops);
    const Vec YYt = gram(centered(Y, rows, q, center_y), rows, q);
    const double yy_norm = frobenius(YYt);
    scores.assign(n_ops, 0.0);
    for (std::size_t b = 0; b < n_ops; ++b) {
        const Vec K = gram(centered(blocks[b], rows, p, center_x), rows, p);
        scores[b] = alignment(K, frobenius(K), YYt, yy_norm);
    }
    const Index order = order_desc(scores);
    Vec positive(n_ops, 0.0);
    for (std::size_t r = 0; r < keep; ++r) positive[order[r]] = std::max(scores[order[r]], 0.0);
    const double total = pairwise_sum(positive.data(), n_ops);
    weights.assign(n_ops, 0.0);
    if (total > 0.0) {
        for (std::size_t b = 0; b < n_ops; ++b) weights[b] = positive[b] / total;
    } else {
        for (std::size_t r = 0; r < keep; ++r) weights[order[r]] = 1.0 / static_cast<double>(keep);
    }
}

// Operator family of a strict-linear kind (per-family cap of the screen).
int family_of(n4m_operator_kind_t kind) {
    switch (kind) {
        case N4M_OP_IDENTITY: return 0;
        case N4M_OP_SAVGOL_SMOOTH: return 1;
        case N4M_OP_SAVGOL_DERIVATIVE: return 2;
        case N4M_OP_NORRIS_WILLIAMS: return 3;
        case N4M_OP_FINITE_DIFFERENCE: return 4;
        case N4M_OP_DETREND_POLY: return 5;
        case N4M_OP_WHITTAKER: return 6;
        case N4M_OP_GAUSSIAN: return 7;
        case N4M_OP_FCK: return 8;
        default: return 9;
    }
}

Vec unit_interval(const Vec& v) {
    const auto [lo, hi] = std::minmax_element(v.begin(), v.end());
    Vec out(v.size(), 0.0);
    if (*hi <= *lo) return out;
    for (std::size_t i = 0; i < v.size(); ++i) out[i] = (v[i] - *lo) / (*hi - *lo);
    return out;
}

// Active screen: response signatures scale_b * Z_b' y_c scored by squared
// norm, kernel-target alignment or both, then a greedy pick by decreasing
// score that skips a block whose unit signature is too collinear with a kept
// one or whose family is full. Identity is kept first when requested.
Index active_screen(const std::vector<Vec>& blocks, const std::vector<OperatorEntry>& ops,
                    std::size_t rows, std::size_t p, const Vec& Y, std::size_t q,
                    const SuperblockOptions& o, Vec& scores) {
    const std::size_t n_ops = blocks.size();
    const std::size_t cap = std::min(static_cast<std::size_t>(o.active_top_m), n_ops);
    const Vec yc = centered(Y, rows, q, o.center_y);
    const Vec yy = gram(yc, rows, q);
    const double yy_norm = frobenius(yy);
    const bool with_kta = o.active_score != ActiveScore::kNorm;
    Vec norm_scores(n_ops, 0.0), kta_scores(n_ops, 0.0);
    std::vector<Vec> signatures(n_ops);
    for (std::size_t b = 0; b < n_ops; ++b) {
        const Vec block = centered(blocks[b], rows, p, o.center_x);
        double scale = 1.0;
        if (o.rms_scaling) {
            Vec sq(block.size());
            for (std::size_t i = 0; i < block.size(); ++i) sq[i] = block[i] * block[i];
            const double r = std::sqrt(mean_of(sq));
            if (r > 1e-12) scale = 1.0 / r;
        }
        Vec& sig = signatures[b];
        sig.assign(p * q, 0.0);
        for (std::size_t j = 0; j < p; ++j) {
            for (std::size_t t = 0; t < q; ++t) {
                double acc = 0.0;
                for (std::size_t i = 0; i < rows; ++i) acc += block[i * p + j] * yc[i * q + t];
                sig[j * q + t] = scale * acc;
            }
        }
        const double norm = frobenius(sig);
        norm_scores[b] = norm * norm;
        if (with_kta) {
            Vec K = gram(block, rows, p);
            for (double& v : K) v *= scale * scale;
            kta_scores[b] = alignment(K, frobenius(K), yy, yy_norm);
        }
    }
    if (o.active_score == ActiveScore::kNorm) {
        scores = norm_scores;
    } else if (o.active_score == ActiveScore::kKta) {
        scores = kta_scores;
    } else {
        const Vec a = unit_interval(norm_scores), b = unit_interval(kta_scores);
        scores.assign(n_ops, 0.0);
        for (std::size_t i = 0; i < n_ops; ++i) scores[i] = a[i] + b[i];
    }

    Index selected;
    std::vector<Vec> units;
    std::vector<std::int32_t> family_count(10, 0);
    auto add = [&](std::size_t idx) {
        if (std::find(selected.begin(), selected.end(), idx) != selected.end()) return;
        const Vec& sig = signatures[idx];
        const double norm = frobenius(sig);
        Vec unit(sig);
        if (norm > kTiny) {
            for (double& v : unit) v /= norm;
        }
        for (const Vec& prev : units) {
            double dot = 0.0;
            for (std::size_t i = 0; i < unit.size(); ++i) dot += unit[i] * prev[i];
            if (norm > kTiny && std::fabs(dot) >= o.active_diversity_threshold) return;
        }
        if (o.active_max_per_family > 0) {
            std::int32_t& count =
                family_count[static_cast<std::size_t>(family_of(ops[idx].kind))];
            if (count >= o.active_max_per_family) return;
            ++count;
        }
        selected.push_back(idx);
        units.push_back(std::move(unit));
    };
    if (o.keep_identity) {
        for (std::size_t b = 0; b < n_ops; ++b) {
            if (ops[b].kind == N4M_OP_IDENTITY) {
                add(b);
                break;
            }
        }
    }
    for (std::size_t idx : order_desc(scores)) {
        if (selected.size() >= cap) break;
        add(idx);
    }
    if (selected.empty()) {
        selected.push_back(static_cast<std::size_t>(
            std::max_element(scores.begin(), scores.end()) - scores.begin()));
    }
    selected.resize(std::min(selected.size(), cap));
    return selected;
}

// A fitted head: predictions on its design are Z beta + offset.
struct Head {
    Vec beta;
    Vec offset;
};

struct HeadSpec {
    SuperblockHead kind;
    bool center_x;  // Ridge-PLS
    bool center_y;
    std::int32_t components;
    double lambda;
};

n4m_status_t fit_head(Context& ctx, const HeadSpec& h, Vec& Z, std::size_t rows, std::size_t cols,
                      const Vec& Y, std::size_t q, Head& out) {
    Config cfg;
    cfg.scale_x = 0;
    cfg.scale_y = 0;
    const n4m_matrix_view_t Zv = view_of(Z, rows, cols);
    if (h.kind == SuperblockHead::kRidge) {
        out.offset = h.center_y ? column_means(Y, rows, q) : Vec(q, 0.0);
        Vec Yc(Y);
        for (std::size_t i = 0; i < rows; ++i) {
            for (std::size_t t = 0; t < q; ++t) Yc[i * q + t] -= out.offset[t];
        }
        cfg.center_x = 0;
        cfg.center_y = 0;
        RidgeResult r;
        const n4m_status_t st = fit_ridge(ctx, cfg, Zv, view_of(Yc, rows, q), h.lambda,
                                          RidgeSolver::kAuto, cfg.ridge_fit_intercept != 0, r);
        out.beta = std::move(r.coefficients);
        return st;
    }
    Vec Ym(Y);
    if (h.kind == SuperblockHead::kPls) {
        cfg.center_x = 0;
        cfg.center_y = h.center_y ? 1 : 0;
        SweepResult r;
        const n4m_status_t st = run_moment_final_fit(ctx, cfg, Zv, view_of(Ym, rows, q), 1,
                                                     static_cast<double>(h.components), r);
        out.beta = std::move(r.coefficients);
        out.offset = std::move(r.intercept);
        return st;
    }
    cfg.n_components = h.components;
    cfg.center_x = h.center_x ? 1 : 0;
    cfg.center_y = h.center_y ? 1 : 0;
    WeightedPlsResult r;
    const n4m_status_t st = fit_ridge_pls(ctx, cfg, Zv, view_of(Ym, rows, q), h.lambda, r);
    if (st != N4M_OK) return st;
    out.beta = std::move(r.coefficients);
    out.offset.assign(q, 0.0);
    for (std::size_t t = 0; t < q; ++t) {
        double dot = 0.0;
        for (std::size_t j = 0; j < cols; ++j) dot += r.x_mean[j] * out.beta[j * q + t];
        out.offset[t] = r.y_mean[t] - dot;
    }
    return N4M_OK;
}

// Z beta + offset for rows of Z, written at `rows_out` of `out` (n x q).
void predict_rows(const Vec& Z, std::size_t cols, const Head& h, std::size_t q,
                  const Index& rows_out, Vec& out) {
    for (std::size_t r = 0; r < rows_out.size(); ++r) {
        for (std::size_t t = 0; t < q; ++t) {
            double acc = 0.0;
            for (std::size_t j = 0; j < cols; ++j) acc += Z[r * cols + j] * h.beta[j * q + t];
            out[rows_out[r] * q + t] = acc + h.offset[t];
        }
    }
}

Index iota_index(std::size_t n) {
    Index idx(n);
    std::iota(idx.begin(), idx.end(), std::size_t{0});
    return idx;
}

double rmse(const Vec& Y, const Vec& P) {
    Vec sq(Y.size());
    for (std::size_t i = 0; i < Y.size(); ++i) {
        const double r = Y[i] - P[i];
        sq[i] = r * r;
    }
    return std::sqrt(mean_of(sq));
}

// First minimum, a NaN winning as in numpy.argmin.
std::size_t argmin(const Vec& v) {
    std::size_t best = 0;
    for (std::size_t i = 0; i < v.size(); ++i) {
        if (std::isnan(v[i])) return i;
        if (v[i] < v[best]) best = i;
    }
    return best;
}

// The superblock design of one set of rows: the selected operator outputs,
// centered and scaled (train rows fit the scaling, other rows reuse it), with
// the MKL weights folded into the block factors.
struct Design {
    Index ops;  // operator index of each block
    Scaling scaling;
    Vec weights;  // MKL weights of every operator (empty otherwise)
    Vec scores;   // MKL alignment or active scores of every operator
};

// Block factors of the MKL weights.
Vec weight_roots(const Vec& weights) {
    Vec roots(weights.size());
    for (std::size_t b = 0; b < roots.size(); ++b) roots[b] = std::sqrt(std::max(weights[b], 0.0));
    return roots;
}

n4m_status_t train_design(Context& ctx, const std::vector<OperatorEntry>& operators, Vec& Xr,
                          std::size_t rows, std::size_t p, const Vec& Y, std::size_t q,
                          const SuperblockOptions& o, Design& d, Vec& Z) {
    std::vector<Vec> outputs;
    const Index every = iota_index(operators.size());
    n4m_status_t st = operator_outputs(ctx, operators, every, Xr, rows, p, outputs);
    if (st != N4M_OK) return st;
    d.ops = every;
    d.weights.clear();
    d.scores.clear();
    if (o.selection == BlockSelection::kMkl) {
        mkl_weights(outputs, rows, p, Y, q, o.mkl_top_k, o.center_x, o.center_y, d.weights,
                    d.scores);
    } else if (o.selection == BlockSelection::kActive) {
        d.ops = active_screen(outputs, operators, rows, p, Y, q, o, d.scores);
        std::vector<Vec> kept;
        for (std::size_t b : d.ops) kept.push_back(std::move(outputs[b]));
        outputs = std::move(kept);
    }
    Z = superblock(outputs, rows, p);
    d.scaling = center_scale(Z, rows, d.ops.size(), p, o.center_x, o.rms_scaling);
    if (!d.weights.empty()) scale_blocks(Z, rows, p, weight_roots(d.weights));
    return N4M_OK;
}

n4m_status_t apply_design(Context& ctx, const std::vector<OperatorEntry>& operators, Vec& Xr,
                          std::size_t rows, std::size_t p, const Design& d, Vec& Z) {
    std::vector<Vec> outputs;
    const n4m_status_t st = operator_outputs(ctx, operators, d.ops, Xr, rows, p, outputs);
    if (st != N4M_OK) return st;
    Z = superblock(outputs, rows, p);
    subtract_mean(Z, rows, d.scaling.mean);
    scale_blocks(Z, rows, p, d.scaling.factors);
    if (!d.weights.empty()) scale_blocks(Z, rows, p, weight_roots(d.weights));
    return N4M_OK;
}

// Candidate heads of the grid: Ridge alphas, PLS component counts, or Ridge-PLS
// components x lambdas.
std::vector<HeadSpec> candidates(const SuperblockOptions& o) {
    std::vector<HeadSpec> out;
    const bool rpls = o.head == SuperblockHead::kRidgePls;
    if (o.head == SuperblockHead::kRidge) {
        for (double l : o.lambdas) out.push_back({o.head, true, o.center_y, 0, l});
    } else {
        for (std::int32_t c : o.components) {
            if (!rpls) {
                out.push_back({o.head, true, o.center_y, c, 0.0});
                continue;
            }
            // The reference fits Ridge-PLS on the superblock with centered X and Y.
            for (double l : o.lambdas) out.push_back({o.head, true, true, c, l});
        }
    }
    return out;
}

n4m_status_t validate_superblock(Context& ctx, const SuperblockOptions& o, std::size_t min_train,
                                 std::size_t width) {
    if (o.head == SuperblockHead::kRidge) {
        return check_lambdas(ctx, o.lambdas, true, "alphas must be finite and strictly positive");
    }
    n4m_status_t st = check_components(ctx, o.components, min_train, width);
    if (st == N4M_OK && o.head == SuperblockHead::kRidgePls) {
        st = check_lambdas(ctx, o.lambdas, false, "ridge_lambdas must be finite and non-negative");
    }
    return st;
}

std::vector<std::int64_t> to_int64(const Index& v) {
    std::vector<std::int64_t> out;
    for (std::size_t x : v) out.push_back(static_cast<std::int64_t>(x));
    return out;
}

// Predictions of the input-space predictor on X (n x p).
Vec predict_input(const Vec& X, std::size_t n, std::size_t p, const Vec& coef, const Vec& b0,
                  std::size_t q) {
    Vec out(n * q);
    for (std::size_t i = 0; i < n; ++i) {
        for (std::size_t t = 0; t < q; ++t) {
            double acc = 0.0;
            for (std::size_t j = 0; j < p; ++j) acc += X[i * p + j] * coef[j * q + t];
            out[i * q + t] = acc + b0[t];
        }
    }
    return out;
}

}  // namespace

n4m_status_t fit_aom_superblock(Context& ctx, const std::vector<OperatorEntry>& operators,
                                const n4m_matrix_view_t& X, const n4m_matrix_view_t& Y,
                                const std::vector<std::int32_t>& fold_ids,
                                const SuperblockOptions& options, MethodResult& out) {
    Vec Xd, Yd;
    n4m_status_t st = dense(ctx, X, "X", Xd);
    if (st == N4M_OK) st = dense(ctx, Y, "Y", Yd);
    if (st != N4M_OK) return st;
    if (operators.empty()) {
        ctx.set_error("operators must contain at least one strict AOM operator");
        return N4M_ERR_INVALID_ARGUMENT;
    }
    const auto n = static_cast<std::size_t>(X.rows);
    const auto p = static_cast<std::size_t>(X.cols);
    const auto q = static_cast<std::size_t>(Y.cols);
    std::vector<Fold> folds;
    st = make_folds(ctx, fold_ids, n, folds);
    if (st == N4M_OK) {
        st = validate_superblock(ctx, options, min_train_rows(folds), operators.size() * p);
    }
    if (st != N4M_OK) return st;

    const std::vector<HeadSpec> grid = candidates(options);
    std::vector<Vec> oof(grid.size(), Vec(n * q, 0.0));
    for (const Fold& fold : folds) {
        Vec Xtr = take_rows(Xd, p, fold.train), Xva = take_rows(Xd, p, fold.valid);
        const Vec Ytr = take_rows(Yd, q, fold.train);
        Design d;
        Vec Ztr, Zva;
        st = train_design(ctx, operators, Xtr, fold.train.size(), p, Ytr, q, options, d, Ztr);
        if (st == N4M_OK) st = apply_design(ctx, operators, Xva, fold.valid.size(), p, d, Zva);
        if (st != N4M_OK) return st;
        const std::size_t width = d.ops.size() * p;
        for (std::size_t c = 0; c < grid.size(); ++c) {
            Head h;
            st = fit_head(ctx, grid[c], Ztr, fold.train.size(), width, Ytr, q, h);
            if (st != N4M_OK) return st;
            predict_rows(Zva, width, h, q, fold.valid, oof[c]);
        }
    }
    const bool rpls = options.head == SuperblockHead::kRidgePls;
    const std::size_t score_cols = rpls ? 4 : 3;
    Vec scores(grid.size() * score_cols), rmses(grid.size());
    for (std::size_t c = 0; c < grid.size(); ++c) {
        rmses[c] = rmse(Yd, oof[c]);
        double* row = &scores[c * score_cols];
        row[0] = static_cast<double>(c);
        if (options.head == SuperblockHead::kRidge) {
            row[1] = grid[c].lambda;
        } else {
            row[1] = static_cast<double>(grid[c].components);
            if (rpls) row[2] = grid[c].lambda;
        }
        row[score_cols - 1] = rmses[c];
    }
    const std::size_t best = argmin(rmses);

    Design d;
    Vec Z;
    st = train_design(ctx, operators, Xd, n, p, Yd, q, options, d, Z);
    if (st != N4M_OK) return st;
    const std::size_t width = d.ops.size() * p;
    Head h;
    st = fit_head(ctx, grid[best], Z, n, width, Yd, q, h);
    if (st != N4M_OK) return st;

    // input_coefficients = sum_k M_k (f_k beta_k), intercept = offset - mean' (f beta).
    Vec factors(d.scaling.factors);
    if (!d.weights.empty()) {
        const Vec roots = weight_roots(d.weights);
        for (std::size_t k = 0; k < factors.size(); ++k) factors[k] *= roots[k];
    }
    Vec scaled(h.beta);
    for (std::size_t k = 0; k < factors.size(); ++k) {
        for (std::size_t j = 0; j < p; ++j) {
            for (std::size_t t = 0; t < q; ++t) scaled[(k * p + j) * q + t] *= factors[k];
        }
    }
    Vec coef(p * q, 0.0), M, part(p * q);
    for (std::size_t k = 0; k < d.ops.size(); ++k) {
        st = operator_matrix(ctx, operators[d.ops[k]], p, M);
        if (st != N4M_OK) return st;
        for (std::size_t i = 0; i < p; ++i) {
            for (std::size_t t = 0; t < q; ++t) {
                double acc = 0.0;
                for (std::size_t j = 0; j < p; ++j) {
                    acc += M[i * p + j] * scaled[(k * p + j) * q + t];
                }
                part[i * q + t] = acc;
            }
        }
        for (std::size_t i = 0; i < p * q; ++i) coef[i] += part[i];
    }
    Vec intercept(q);
    for (std::size_t t = 0; t < q; ++t) {
        double acc = 0.0;
        for (std::size_t j = 0; j < width; ++j) acc += d.scaling.mean[j] * scaled[j * q + t];
        intercept[t] = h.offset[t] - acc;
    }

    const auto n64 = static_cast<std::int64_t>(n);
    const auto p64 = static_cast<std::int64_t>(p);
    const auto q64 = static_cast<std::int64_t>(q);
    const auto nb64 = static_cast<std::int64_t>(d.ops.size());
    out.set_double_matrix("predictions", predict_input(Xd, n, p, coef, intercept, q), n64, q64);
    out.set_double_matrix("oof_predictions", std::move(oof[best]), n64, q64);
    out.set_double_matrix("coefficients", std::move(h.beta), static_cast<std::int64_t>(width), q64);
    out.set_double_matrix("input_coefficients", std::move(coef), p64, q64);
    out.set_double_matrix("intercept", std::move(intercept), 1, q64);
    out.set_double_matrix("candidate_scores", std::move(scores),
                          static_cast<std::int64_t>(grid.size()),
                          static_cast<std::int64_t>(score_cols));
    out.set_double_matrix("block_scales", d.scaling.factors, nb64, 1);
    out.set_double_matrix("superblock_mean", std::move(d.scaling.mean), 1,
                          static_cast<std::int64_t>(width));
    out.set_int_vector("fold_ids", fold_ids);
    out.set_scalar("selected_candidate_id", static_cast<double>(best));
    out.set_scalar("selected_cv_rmse", rmses[best]);
    out.set_scalar("n_operators", static_cast<double>(operators.size()));
    if (options.head == SuperblockHead::kRidge) {
        out.set_scalar("selected_alpha", grid[best].lambda);
    } else {
        out.set_scalar("selected_n_components", grid[best].components);
        if (rpls) out.set_scalar("selected_ridge_lambda", grid[best].lambda);
    }
    if (options.selection == BlockSelection::kMkl) {
        Index active;
        for (std::size_t b = 0; b < d.weights.size(); ++b) {
            if (d.weights[b] > 0.0) active.push_back(b);
        }
        out.set_int64_vector("selected_operator_indices", to_int64(active));
        out.set_double_matrix("mkl_weights", d.weights,
                              static_cast<std::int64_t>(d.weights.size()), 1);
        out.set_double_matrix("mkl_alignment_scores", d.scores,
                              static_cast<std::int64_t>(d.scores.size()), 1);
    } else if (options.selection == BlockSelection::kActive) {
        out.set_int64_vector("selected_operator_indices", to_int64(d.ops));
        out.set_double_matrix("active_scores", d.scores,
                              static_cast<std::int64_t>(d.scores.size()), 1);
    }
    return N4M_OK;
}

namespace {

n4m_status_t apply_chain(Context& ctx, const std::vector<OperatorEntry>& chain, const Vec& A,
                         std::size_t rows, std::size_t cols, Vec& out) {
    out = A;
    for (const OperatorEntry& op : chain) {
        Vec next;
        const n4m_status_t st =
            transform_aom_strict_operator(ctx, op, view_of(out, rows, cols), next);
        if (st != N4M_OK) return st;
        out = std::move(next);
    }
    return N4M_OK;
}

}  // namespace

n4m_status_t fit_aom_chain_ridge_pls(Context& ctx,
                                     const std::vector<std::vector<OperatorEntry>>& chains,
                                     const n4m_matrix_view_t& X, const n4m_matrix_view_t& Y,
                                     const std::vector<std::int32_t>& fold_ids,
                                     const ChainRidgePlsOptions& options, MethodResult& out) {
    Vec Xd, Yd;
    n4m_status_t st = dense(ctx, X, "X", Xd);
    if (st == N4M_OK) st = dense(ctx, Y, "Y", Yd);
    if (st != N4M_OK) return st;
    bool chains_ok = !chains.empty();
    for (const auto& chain : chains) chains_ok = chains_ok && !chain.empty();
    if (!chains_ok) {
        ctx.set_error("chains must contain at least one non-empty strict AOM chain");
        return N4M_ERR_INVALID_ARGUMENT;
    }
    const auto n = static_cast<std::size_t>(X.rows);
    const auto p = static_cast<std::size_t>(X.cols);
    const auto q = static_cast<std::size_t>(Y.cols);
    std::vector<Fold> folds;
    st = make_folds(ctx, fold_ids, n, folds);
    if (st == N4M_OK) st = check_components(ctx, options.components, min_train_rows(folds), p);
    if (st == N4M_OK) {
        st = check_lambdas(ctx, options.lambdas, false,
                           "ridge_lambdas must be finite and non-negative");
    }
    if (st != N4M_OK) return st;

    std::vector<HeadSpec> grid;
    for (std::int32_t c : options.components) {
        for (double l : options.lambdas) {
            grid.push_back({SuperblockHead::kRidgePls, options.center_x, options.center_y, c, l});
        }
    }
    const std::size_t per_chain = grid.size();
    Vec scores(chains.size() * per_chain * 5);
    std::size_t best = 0, best_chain = 0, best_head = 0;
    double best_rmse = std::numeric_limits<double>::infinity();
    Vec best_oof;
    bool found = false;
    for (std::size_t ci = 0; ci < chains.size(); ++ci) {
        std::vector<Vec> oof(per_chain, Vec(n * q, 0.0));
        for (const Fold& fold : folds) {
            const std::size_t ntr = fold.train.size(), nva = fold.valid.size();
            Vec Ztr, Zva;
            st = apply_chain(ctx, chains[ci], take_rows(Xd, p, fold.train), ntr, p, Ztr);
            if (st == N4M_OK) {
                st = apply_chain(ctx, chains[ci], take_rows(Xd, p, fold.valid), nva, p, Zva);
            }
            if (st != N4M_OK) return st;
            const Vec Ytr = take_rows(Yd, q, fold.train);
            for (std::size_t c = 0; c < per_chain; ++c) {
                Head h;
                st = fit_head(ctx, grid[c], Ztr, ntr, p, Ytr, q, h);
                if (st != N4M_OK) return st;
                predict_rows(Zva, p, h, q, fold.valid, oof[c]);
            }
        }
        for (std::size_t c = 0; c < per_chain; ++c) {
            const std::size_t id = ci * per_chain + c;
            const double r = rmse(Yd, oof[c]);
            double* row = &scores[id * 5];
            row[0] = static_cast<double>(id);
            row[1] = static_cast<double>(ci);
            row[2] = static_cast<double>(grid[c].components);
            row[3] = grid[c].lambda;
            row[4] = r;
            if (r < best_rmse) {
                best_rmse = r;
                best = id;
                best_chain = ci;
                best_head = c;
                best_oof = oof[c];
                found = true;
            }
        }
    }
    if (!found) {
        ctx.set_error("AOM chain Ridge-PLS scoring produced no finite candidate");
        return N4M_ERR_NUMERICAL_FAILURE;
    }

    Vec Z, T;
    st = apply_chain(ctx, chains[best_chain], Xd, n, p, Z);
    Head h;
    if (st == N4M_OK) st = fit_head(ctx, grid[best_head], Z, n, p, Yd, q, h);
    if (st == N4M_OK) {
        Vec eye(p * p, 0.0);
        for (std::size_t i = 0; i < p; ++i) eye[i * p + i] = 1.0;
        st = apply_chain(ctx, chains[best_chain], eye, p, p, T);
    }
    if (st != N4M_OK) return st;
    Vec coef(p * q);
    for (std::size_t i = 0; i < p; ++i) {
        for (std::size_t t = 0; t < q; ++t) {
            double acc = 0.0;
            for (std::size_t j = 0; j < p; ++j) acc += T[i * p + j] * h.beta[j * q + t];
            coef[i * q + t] = acc;
        }
    }

    const auto n64 = static_cast<std::int64_t>(n);
    const auto p64 = static_cast<std::int64_t>(p);
    const auto q64 = static_cast<std::int64_t>(q);
    out.set_double_matrix("predictions", predict_input(Xd, n, p, coef, h.offset, q), n64, q64);
    out.set_double_matrix("oof_predictions", std::move(best_oof), n64, q64);
    out.set_double_matrix("coefficients", std::move(h.beta), p64, q64);
    out.set_double_matrix("input_coefficients", std::move(coef), p64, q64);
    out.set_double_matrix("intercept", std::move(h.offset), 1, q64);
    out.set_double_matrix("candidate_scores", std::move(scores),
                          static_cast<std::int64_t>(chains.size() * per_chain), 5);
    out.set_int_vector("fold_ids", fold_ids);
    out.set_scalar("selected_candidate_id", static_cast<double>(best));
    out.set_scalar("selected_chain_id", static_cast<double>(best_chain));
    out.set_scalar("selected_n_components", grid[best_head].components);
    out.set_scalar("selected_ridge_lambda", grid[best_head].lambda);
    out.set_scalar("selected_cv_rmse", best_rmse);
    out.set_scalar("n_chains", static_cast<double>(chains.size()));
    return N4M_OK;
}

}  // namespace n4m::core
