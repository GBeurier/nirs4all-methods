// SPDX-License-Identifier: CECILL-2.1

#include "core/aom_operator_pls_stack.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <initializer_list>
#include <limits>
#include <numeric>
#include <utility>
#include <vector>

#include "core/aom_operators.hpp"
#include "core/aom_robust_hpo.hpp"
#include "core/common/matrix_view.hpp"
#include "core/operator_entry.hpp"
#include "core/ridge.hpp"

namespace n4m::core {

namespace {

struct ChainSpec {
    std::int32_t id{0};
    std::vector<OperatorEntry> ops;
};

struct Standardizer {
    std::vector<double> mean;
    std::vector<double> scale;
};

struct Projector {
    std::vector<double> x_mean;
    std::vector<double> rotations; // p x k
    std::int32_t n_components{0};
};

struct FittedView {
    Standardizer scaler;
    Projector projector;
};

struct FittedStack {
    std::vector<FittedView> views;
    RidgeResult ridge;
    std::int64_t n_features{0};
};

[[nodiscard]] std::size_t at(std::size_t row,
                             std::size_t cols,
                             std::size_t col) noexcept {
    return row * cols + col;
}

[[nodiscard]] n4m_matrix_view_t rowmajor_view(std::vector<double>& data,
                                              std::int64_t rows,
                                              std::int64_t cols) noexcept {
    n4m_matrix_view_t view{};
    view.data = data.empty() ? nullptr : data.data();
    view.rows = rows;
    view.cols = cols;
    view.row_stride = cols;
    view.col_stride = 1;
    view.dtype = N4M_DTYPE_F64;
    view.reserved0 = 0;
    return view;
}

OperatorEntry op(n4m_operator_kind_t kind,
                 std::initializer_list<double> params = {}) {
    const std::vector<double> owned(params);
    return OperatorEntry(kind,
                         owned.empty() ? nullptr : owned.data(),
                         static_cast<std::int32_t>(owned.size()));
}

void add_chain(std::vector<ChainSpec>& chains,
               std::vector<OperatorEntry> ops) {
    ChainSpec chain{};
    chain.id = static_cast<std::int32_t>(chains.size());
    chain.ops = std::move(ops);
    chains.push_back(std::move(chain));
}

std::vector<ChainSpec> build_bank(std::int32_t profile) {
    std::vector<ChainSpec> chains;
    add_chain(chains, {op(N4M_OP_IDENTITY)});
    add_chain(chains, {op(N4M_OP_DETREND_POLY, {1})});
    add_chain(chains, {op(N4M_OP_DETREND_POLY, {2})});
    add_chain(chains, {op(N4M_OP_SAVGOL_SMOOTH, {5, 2})});
    add_chain(chains, {op(N4M_OP_SAVGOL_SMOOTH, {7, 2})});
    add_chain(chains, {op(N4M_OP_SAVGOL_DERIVATIVE, {7, 2, 1})});
    add_chain(chains, {op(N4M_OP_SAVGOL_DERIVATIVE, {11, 2, 2})});
    add_chain(chains, {op(N4M_OP_NORRIS_WILLIAMS, {5, 5, 1})});
    add_chain(chains, {op(N4M_OP_FINITE_DIFFERENCE, {1})});
    add_chain(chains, {op(N4M_OP_DETREND_POLY, {1}),
                       op(N4M_OP_SAVGOL_DERIVATIVE, {7, 2, 1})});
    add_chain(chains, {op(N4M_OP_DETREND_POLY, {1}),
                       op(N4M_OP_NORRIS_WILLIAMS, {5, 5, 1})});
    add_chain(chains, {op(N4M_OP_SAVGOL_SMOOTH, {5, 2}),
                       op(N4M_OP_FINITE_DIFFERENCE, {1})});

    if (profile == static_cast<std::int32_t>(AomRobustHpoProfile::kWide)) {
        add_chain(chains, {op(N4M_OP_SAVGOL_SMOOTH, {9, 2})});
        add_chain(chains, {op(N4M_OP_SAVGOL_SMOOTH, {11, 3})});
        add_chain(chains, {op(N4M_OP_SAVGOL_SMOOTH, {15, 3})});
        add_chain(chains, {op(N4M_OP_SAVGOL_DERIVATIVE, {9, 2, 1})});
        add_chain(chains, {op(N4M_OP_SAVGOL_DERIVATIVE, {15, 3, 1})});
        add_chain(chains, {op(N4M_OP_SAVGOL_DERIVATIVE, {21, 3, 1})});
        add_chain(chains, {op(N4M_OP_SAVGOL_DERIVATIVE, {15, 3, 2})});
        add_chain(chains, {op(N4M_OP_NORRIS_WILLIAMS, {7, 7, 1})});
        add_chain(chains, {op(N4M_OP_NORRIS_WILLIAMS, {11, 11, 1})});
        add_chain(chains, {op(N4M_OP_NORRIS_WILLIAMS, {7, 7, 2})});
        add_chain(chains, {op(N4M_OP_FINITE_DIFFERENCE, {2})});
        add_chain(chains, {op(N4M_OP_GAUSSIAN, {1.0})});
        add_chain(chains, {op(N4M_OP_GAUSSIAN, {2.0})});
        add_chain(chains, {op(N4M_OP_FCK, {0.0})});
        add_chain(chains, {op(N4M_OP_FCK, {1.0})});
        add_chain(chains, {op(N4M_OP_WHITTAKER, {100.0})});
        add_chain(chains, {op(N4M_OP_WHITTAKER, {1000.0})});
        add_chain(chains, {op(N4M_OP_DETREND_POLY, {2}),
                           op(N4M_OP_SAVGOL_DERIVATIVE, {11, 2, 1})});
        add_chain(chains, {op(N4M_OP_WHITTAKER, {100.0}),
                           op(N4M_OP_SAVGOL_DERIVATIVE, {7, 2, 1})});
    }
    return chains;
}

[[nodiscard]] n4m_status_t copy_matrix(Context& ctx,
                                       const n4m_matrix_view_t& V,
                                       const char* name,
                                       std::vector<double>& out) {
    if (validate_nonnull_view(V) != N4M_OK ||
        (V.dtype != N4M_DTYPE_F64 && V.dtype != N4M_DTYPE_F32)) {
        ctx.set_errorf("%s must be a finite F64/F32 matrix", name);
        return N4M_ERR_INVALID_ARGUMENT;
    }
    if (V.rows <= 0 || V.cols <= 0) {
        ctx.set_errorf("%s must be non-empty", name);
        return N4M_ERR_INVALID_ARGUMENT;
    }
    const auto rows = static_cast<std::size_t>(V.rows);
    const auto cols = static_cast<std::size_t>(V.cols);
    const auto rs = static_cast<std::size_t>(V.row_stride);
    const auto cs = static_cast<std::size_t>(V.col_stride);
    out.assign(rows * cols, 0.0);
    if (V.dtype == N4M_DTYPE_F64) {
        const auto* data = static_cast<const double*>(V.data);
        for (std::size_t r = 0; r < rows; ++r) {
            for (std::size_t c = 0; c < cols; ++c) {
                const double value = data[r * rs + c * cs];
                if (!std::isfinite(value)) {
                    ctx.set_errorf("%s contains NaN or Inf", name);
                    return N4M_ERR_INVALID_ARGUMENT;
                }
                out[at(r, cols, c)] = value;
            }
        }
    } else {
        const auto* data = static_cast<const float*>(V.data);
        for (std::size_t r = 0; r < rows; ++r) {
            for (std::size_t c = 0; c < cols; ++c) {
                const double value = static_cast<double>(data[r * rs + c * cs]);
                if (!std::isfinite(value)) {
                    ctx.set_errorf("%s contains NaN or Inf", name);
                    return N4M_ERR_INVALID_ARGUMENT;
                }
                out[at(r, cols, c)] = value;
            }
        }
    }
    return N4M_OK;
}

[[nodiscard]] n4m_status_t make_fold_ids(Context& ctx,
                                         std::int64_t n,
                                         std::int32_t cv,
                                         const std::int32_t* fold_ids,
                                         std::int64_t n_fold_ids,
                                         std::vector<std::int32_t>& ids,
                                         std::int32_t& out_cv) {
    if (n <= 1) {
        ctx.set_error("AOM operator PLS stack requires at least two rows");
        return N4M_ERR_INVALID_ARGUMENT;
    }
    ids.assign(static_cast<std::size_t>(n), 0);
    if (fold_ids == nullptr || n_fold_ids == 0) {
        if (cv < 2 || cv > n) {
            ctx.set_error("AOM operator PLS stack cv must be in [2, n_samples]");
            return N4M_ERR_INVALID_ARGUMENT;
        }
        out_cv = cv;
        for (std::int64_t i = 0; i < n; ++i) {
            ids[static_cast<std::size_t>(i)] =
                static_cast<std::int32_t>((i * cv) / n);
        }
    } else {
        if (n_fold_ids != n) {
            ctx.set_error("AOM operator PLS stack fold_ids length must match n_samples");
            return N4M_ERR_INVALID_ARGUMENT;
        }
        std::int32_t max_id = -1;
        for (std::int64_t i = 0; i < n; ++i) {
            const auto id = fold_ids[i];
            if (id < 0) {
                ctx.set_error("AOM operator PLS stack fold_ids must be non-negative");
                return N4M_ERR_INVALID_ARGUMENT;
            }
            ids[static_cast<std::size_t>(i)] = id;
            max_id = std::max(max_id, id);
        }
        out_cv = cv > 0 ? cv : max_id + 1;
        if (out_cv < 2) {
            ctx.set_error("AOM operator PLS stack requires at least two folds");
            return N4M_ERR_INVALID_ARGUMENT;
        }
        for (const auto id : ids) {
            if (id >= out_cv) {
                ctx.set_error("AOM operator PLS stack fold id is outside cv range");
                return N4M_ERR_INVALID_ARGUMENT;
            }
        }
    }

    std::vector<std::int64_t> counts(static_cast<std::size_t>(out_cv), 0);
    for (const auto id : ids) {
        ++counts[static_cast<std::size_t>(id)];
    }
    for (std::int32_t f = 0; f < out_cv; ++f) {
        if (counts[static_cast<std::size_t>(f)] == 0) {
            ctx.set_error("AOM operator PLS stack requires every fold to contain rows");
            return N4M_ERR_INVALID_ARGUMENT;
        }
        if (counts[static_cast<std::size_t>(f)] == n) {
            ctx.set_error("AOM operator PLS stack fold would leave an empty train set");
            return N4M_ERR_INVALID_ARGUMENT;
        }
    }
    return N4M_OK;
}

std::vector<std::vector<std::int64_t>> fold_rows_from_ids(
    const std::vector<std::int32_t>& ids,
    std::int32_t cv) {
    std::vector<std::vector<std::int64_t>> rows(static_cast<std::size_t>(cv));
    for (std::size_t i = 0; i < ids.size(); ++i) {
        rows[static_cast<std::size_t>(ids[i])].push_back(
            static_cast<std::int64_t>(i));
    }
    return rows;
}

std::vector<std::vector<std::int64_t>> train_rows_from_folds(
    std::int64_t n,
    const std::vector<std::vector<std::int64_t>>& fold_rows) {
    std::vector<std::vector<std::int64_t>> out(fold_rows.size());
    std::vector<char> held(static_cast<std::size_t>(n), 0);
    for (std::size_t fold = 0; fold < fold_rows.size(); ++fold) {
        std::fill(held.begin(), held.end(), 0);
        for (const auto row : fold_rows[fold]) {
            held[static_cast<std::size_t>(row)] = 1;
        }
        auto& dst = out[fold];
        dst.reserve(static_cast<std::size_t>(n) - fold_rows[fold].size());
        for (std::int64_t row = 0; row < n; ++row) {
            if (!held[static_cast<std::size_t>(row)]) dst.push_back(row);
        }
    }
    return out;
}

void extract_rows(const std::vector<double>& src,
                  std::int64_t cols,
                  const std::vector<std::int64_t>& rows,
                  std::vector<double>& out) {
    const auto c = static_cast<std::size_t>(cols);
    out.assign(rows.size() * c, 0.0);
    for (std::size_t r = 0; r < rows.size(); ++r) {
        const auto src_row = static_cast<std::size_t>(rows[r]);
        std::copy_n(src.data() + src_row * c, c, out.data() + r * c);
    }
}

[[nodiscard]] n4m_status_t transform_chain(Context& ctx,
                                           const ChainSpec& chain,
                                           const n4m_matrix_view_t& X,
                                           std::vector<double>& out) {
    if (chain.ops.empty()) {
        ctx.set_error("AOM operator PLS stack chain must not be empty");
        return N4M_ERR_INVALID_ARGUMENT;
    }
    std::vector<double> current;
    n4m_matrix_view_t current_view = X;
    for (const auto& entry : chain.ops) {
        std::vector<double> next;
        const n4m_status_t status =
            transform_aom_strict_operator(ctx, entry, current_view, next);
        if (status != N4M_OK) return status;
        current = std::move(next);
        current_view = rowmajor_view(current, X.rows, X.cols);
    }
    out = std::move(current);
    return N4M_OK;
}

[[nodiscard]] n4m_status_t build_chain_operator_matrix(Context& ctx,
                                                       const ChainSpec& chain,
                                                       std::int64_t n_features,
                                                       std::vector<double>& out) {
    const auto p = static_cast<std::size_t>(n_features);
    if (p == 0 ||
        p > (std::numeric_limits<std::size_t>::max() / p)) {
        ctx.set_error("AOM operator PLS stack operator matrix dimensions overflow addressable memory");
        return N4M_ERR_INVALID_ARGUMENT;
    }
    std::vector<double> basis(p * p, 0.0);
    for (std::size_t i = 0; i < p; ++i) basis[at(i, p, i)] = 1.0;
    n4m_matrix_view_t basis_view =
        rowmajor_view(basis, n_features, n_features);
    return transform_chain(ctx, chain, basis_view, out);
}

[[nodiscard]] n4m_status_t fold_chain_coefficients_to_input_space(
    Context& ctx,
    const ChainSpec& chain,
    std::int64_t n_features,
    const std::vector<double>& transformed_coefficients,
    std::vector<double>& input_coefficients) {
    const auto p = static_cast<std::size_t>(n_features);
    if (p == 0 || transformed_coefficients.size() != p) {
        ctx.set_error("AOM operator PLS stack coefficient folding received inconsistent dimensions");
        return N4M_ERR_INVALID_ARGUMENT;
    }

    std::vector<double> op_matrix;
    n4m_status_t status = build_chain_operator_matrix(ctx, chain, n_features,
                                                      op_matrix);
    if (status != N4M_OK) return status;
    if (op_matrix.size() != p * p) {
        ctx.set_error("AOM operator PLS stack coefficient folding operator has invalid shape");
        return N4M_ERR_INVALID_ARGUMENT;
    }

    input_coefficients.assign(p, 0.0);
    for (std::size_t src = 0; src < p; ++src) {
        for (std::size_t mid = 0; mid < p; ++mid) {
            const double a = op_matrix[at(src, p, mid)];
            if (a == 0.0) continue;
            input_coefficients[src] += a * transformed_coefficients[mid];
        }
    }
    return N4M_OK;
}

void fit_standardizer(const std::vector<double>& X,
                      std::int64_t rows,
                      std::int64_t cols,
                      Standardizer& scaler,
                      std::vector<double>& scaled) {
    const auto n = static_cast<std::size_t>(rows);
    const auto p = static_cast<std::size_t>(cols);
    scaler.mean.assign(p, 0.0);
    scaler.scale.assign(p, 1.0);
    for (std::size_t r = 0; r < n; ++r) {
        for (std::size_t c = 0; c < p; ++c) {
            scaler.mean[c] += X[at(r, p, c)];
        }
    }
    const double inv_n = 1.0 / static_cast<double>(n);
    for (auto& value : scaler.mean) value *= inv_n;
    scaled.assign(n * p, 0.0);
    for (std::size_t r = 0; r < n; ++r) {
        for (std::size_t c = 0; c < p; ++c) {
            const double value = X[at(r, p, c)] - scaler.mean[c];
            scaled[at(r, p, c)] = value;
            scaler.scale[c] += value * value;
        }
    }
    for (std::size_t c = 0; c < p; ++c) {
        const double stddev = std::sqrt((scaler.scale[c] - 1.0) * inv_n);
        scaler.scale[c] =
            (stddev > 1e-12 && std::isfinite(stddev)) ? stddev : 1.0;
    }
    for (std::size_t r = 0; r < n; ++r) {
        for (std::size_t c = 0; c < p; ++c) {
            scaled[at(r, p, c)] /= scaler.scale[c];
        }
    }
}

void apply_standardizer(const std::vector<double>& X,
                        std::int64_t rows,
                        std::int64_t cols,
                        const Standardizer& scaler,
                        std::vector<double>& scaled) {
    const auto n = static_cast<std::size_t>(rows);
    const auto p = static_cast<std::size_t>(cols);
    scaled.assign(n * p, 0.0);
    for (std::size_t r = 0; r < n; ++r) {
        for (std::size_t c = 0; c < p; ++c) {
            scaled[at(r, p, c)] =
                (X[at(r, p, c)] - scaler.mean[c]) / scaler.scale[c];
        }
    }
}

[[nodiscard]] bool invert_square(std::vector<double> a,
                                 std::size_t n,
                                 std::vector<double>& inverse) {
    inverse.assign(n * n, 0.0);
    for (std::size_t i = 0; i < n; ++i) inverse[at(i, n, i)] = 1.0;
    for (std::size_t col = 0; col < n; ++col) {
        std::size_t pivot = col;
        double pivot_abs = std::fabs(a[at(col, n, col)]);
        for (std::size_t row = col + 1U; row < n; ++row) {
            const double candidate = std::fabs(a[at(row, n, col)]);
            if (candidate > pivot_abs) {
                pivot = row;
                pivot_abs = candidate;
            }
        }
        if (pivot_abs <= 1e-14) return false;
        if (pivot != col) {
            for (std::size_t j = 0; j < n; ++j) {
                std::swap(a[at(col, n, j)], a[at(pivot, n, j)]);
                std::swap(inverse[at(col, n, j)], inverse[at(pivot, n, j)]);
            }
        }
        const double diag = a[at(col, n, col)];
        for (std::size_t j = 0; j < n; ++j) {
            a[at(col, n, j)] /= diag;
            inverse[at(col, n, j)] /= diag;
        }
        for (std::size_t row = 0; row < n; ++row) {
            if (row == col) continue;
            const double factor = a[at(row, n, col)];
            for (std::size_t j = 0; j < n; ++j) {
                a[at(row, n, j)] -= factor * a[at(col, n, j)];
                inverse[at(row, n, j)] -= factor * inverse[at(col, n, j)];
            }
        }
    }
    return true;
}

[[nodiscard]] n4m_status_t fit_pls1_projector(Context& ctx,
                                              const std::vector<double>& X,
                                              const std::vector<double>& y,
                                              std::int64_t rows,
                                              std::int64_t cols,
                                              std::int32_t requested_components,
                                              Projector& projector) {
    const auto n = static_cast<std::size_t>(rows);
    const auto p = static_cast<std::size_t>(cols);
    const auto k_max = static_cast<std::size_t>(
        std::min<std::int64_t>({requested_components, cols, std::max<std::int64_t>(1, rows - 1)}));
    projector.x_mean.assign(p, 0.0);
    for (std::size_t r = 0; r < n; ++r) {
        for (std::size_t c = 0; c < p; ++c) {
            projector.x_mean[c] += X[at(r, p, c)];
        }
    }
    const double inv_n = 1.0 / static_cast<double>(n);
    for (auto& value : projector.x_mean) value *= inv_n;
    double y_mean = 0.0;
    for (const auto value : y) y_mean += value;
    y_mean *= inv_n;

    std::vector<double> Xr(n * p, 0.0);
    std::vector<double> yr(n, 0.0);
    for (std::size_t r = 0; r < n; ++r) {
        yr[r] = y[r] - y_mean;
        for (std::size_t c = 0; c < p; ++c) {
            Xr[at(r, p, c)] = X[at(r, p, c)] - projector.x_mean[c];
        }
    }

    std::vector<double> W;
    std::vector<double> P;
    std::size_t k_eff = 0;
    W.reserve(p * k_max);
    P.reserve(p * k_max);
    for (std::size_t comp = 0; comp < k_max; ++comp) {
        std::vector<double> w(p, 0.0);
        for (std::size_t c = 0; c < p; ++c) {
            for (std::size_t r = 0; r < n; ++r) {
                w[c] += Xr[at(r, p, c)] * yr[r];
            }
        }
        double w_norm = 0.0;
        for (const auto value : w) w_norm += value * value;
        w_norm = std::sqrt(w_norm);
        if (w_norm <= 1e-12) break;
        for (auto& value : w) value /= w_norm;

        std::vector<double> t(n, 0.0);
        for (std::size_t r = 0; r < n; ++r) {
            for (std::size_t c = 0; c < p; ++c) {
                t[r] += Xr[at(r, p, c)] * w[c];
            }
        }
        double denom = 0.0;
        for (const auto value : t) denom += value * value;
        if (denom <= 1e-12) break;

        std::vector<double> loading(p, 0.0);
        for (std::size_t c = 0; c < p; ++c) {
            for (std::size_t r = 0; r < n; ++r) {
                loading[c] += Xr[at(r, p, c)] * t[r];
            }
            loading[c] /= denom;
        }
        double q = 0.0;
        for (std::size_t r = 0; r < n; ++r) q += yr[r] * t[r];
        q /= denom;

        for (std::size_t r = 0; r < n; ++r) {
            for (std::size_t c = 0; c < p; ++c) {
                Xr[at(r, p, c)] -= t[r] * loading[c];
            }
            yr[r] -= q * t[r];
        }
        W.insert(W.end(), w.begin(), w.end());
        P.insert(P.end(), loading.begin(), loading.end());
        ++k_eff;
    }

    if (k_eff == 0) {
        projector.n_components = 1;
        projector.rotations.assign(p, 0.0);
        return N4M_OK;
    }

    std::vector<double> ptw(k_eff * k_eff, 0.0);
    for (std::size_t a = 0; a < k_eff; ++a) {
        for (std::size_t b = 0; b < k_eff; ++b) {
            double acc = 0.0;
            for (std::size_t c = 0; c < p; ++c) {
                acc += P[at(a, p, c)] * W[at(b, p, c)];
            }
            ptw[at(a, k_eff, b)] = acc;
        }
    }
    std::vector<double> inv;
    if (!invert_square(ptw, k_eff, inv)) {
        for (std::size_t i = 0; i < k_eff; ++i) {
            ptw[at(i, k_eff, i)] += 1e-10;
        }
        if (!invert_square(ptw, k_eff, inv)) {
            ctx.set_error("AOM operator PLS stack PLS rotation solve failed");
            return N4M_ERR_INTERNAL;
        }
    }

    projector.n_components = static_cast<std::int32_t>(k_eff);
    projector.rotations.assign(p * k_eff, 0.0);
    for (std::size_t c = 0; c < p; ++c) {
        for (std::size_t b = 0; b < k_eff; ++b) {
            double acc = 0.0;
            for (std::size_t a = 0; a < k_eff; ++a) {
                acc += W[at(a, p, c)] * inv[at(a, k_eff, b)];
            }
            projector.rotations[at(c, k_eff, b)] = acc;
        }
    }
    return N4M_OK;
}

void transform_projector(const std::vector<double>& X,
                         std::int64_t rows,
                         std::int64_t cols,
                         const Projector& projector,
                         std::vector<double>& scores) {
    const auto n = static_cast<std::size_t>(rows);
    const auto p = static_cast<std::size_t>(cols);
    const auto k = static_cast<std::size_t>(projector.n_components);
    scores.assign(n * k, 0.0);
    for (std::size_t r = 0; r < n; ++r) {
        for (std::size_t comp = 0; comp < k; ++comp) {
            double acc = 0.0;
            for (std::size_t c = 0; c < p; ++c) {
                acc += (X[at(r, p, c)] - projector.x_mean[c]) *
                       projector.rotations[at(c, k, comp)];
            }
            scores[at(r, k, comp)] = acc;
        }
    }
}

void append_block(std::vector<double>& dst,
                  std::int64_t rows,
                  std::int64_t old_cols,
                  const std::vector<double>& block,
                  std::int64_t block_cols) {
    const auto n = static_cast<std::size_t>(rows);
    const auto oldc = static_cast<std::size_t>(old_cols);
    const auto addc = static_cast<std::size_t>(block_cols);
    std::vector<double> out(n * (oldc + addc), 0.0);
    for (std::size_t r = 0; r < n; ++r) {
        if (oldc > 0) {
            std::copy_n(dst.data() + r * oldc, oldc, out.data() + r * (oldc + addc));
        }
        std::copy_n(block.data() + r * addc, addc,
                    out.data() + r * (oldc + addc) + oldc);
    }
    dst = std::move(out);
}

[[nodiscard]] n4m_status_t transform_stack_rows(
    const std::vector<std::vector<double>>& operator_views,
    std::int64_t p,
    const std::vector<std::int64_t>& rows,
    const std::vector<FittedView>& views,
    std::vector<double>& features) {
    features.clear();
    std::int64_t current_cols = 0;
    for (std::size_t view_ix = 0; view_ix < views.size(); ++view_ix) {
        std::vector<double> Xrows;
        extract_rows(operator_views[view_ix], p, rows, Xrows);
        std::vector<double> scaled;
        apply_standardizer(Xrows,
                           static_cast<std::int64_t>(rows.size()),
                           p,
                           views[view_ix].scaler,
                           scaled);
        std::vector<double> scores;
        transform_projector(scaled,
                            static_cast<std::int64_t>(rows.size()),
                            p,
                            views[view_ix].projector,
                            scores);
        append_block(features,
                     static_cast<std::int64_t>(rows.size()),
                     current_cols,
                     scores,
                     views[view_ix].projector.n_components);
        current_cols += views[view_ix].projector.n_components;
    }
    return N4M_OK;
}

[[nodiscard]] n4m_status_t fit_stack(
    Context& ctx,
    const Config& cfg,
    const std::vector<std::vector<double>>& operator_views,
    const std::vector<double>& Yall,
    std::int64_t p,
    const std::vector<std::int64_t>& train_rows,
    std::int32_t components,
    double alpha,
    FittedStack& fitted,
    std::vector<double>& train_features) {
    std::vector<double> ytrain;
    extract_rows(Yall, 1, train_rows, ytrain);
    fitted.views.clear();
    fitted.views.reserve(operator_views.size());
    train_features.clear();
    std::int64_t current_cols = 0;

    for (const auto& view : operator_views) {
        std::vector<double> Xtrain;
        extract_rows(view, p, train_rows, Xtrain);
        FittedView fitted_view;
        std::vector<double> scaled;
        fit_standardizer(Xtrain,
                         static_cast<std::int64_t>(train_rows.size()),
                         p,
                         fitted_view.scaler,
                         scaled);
        n4m_status_t status = fit_pls1_projector(ctx,
                                                 scaled,
                                                 ytrain,
                                                 static_cast<std::int64_t>(train_rows.size()),
                                                 p,
                                                 components,
                                                 fitted_view.projector);
        if (status != N4M_OK) return status;
        std::vector<double> scores;
        transform_projector(scaled,
                            static_cast<std::int64_t>(train_rows.size()),
                            p,
                            fitted_view.projector,
                            scores);
        append_block(train_features,
                     static_cast<std::int64_t>(train_rows.size()),
                     current_cols,
                     scores,
                     fitted_view.projector.n_components);
        current_cols += fitted_view.projector.n_components;
        fitted.views.push_back(std::move(fitted_view));
    }

    Config ridge_cfg = cfg;
    ridge_cfg.center_x = 1;
    ridge_cfg.center_y = 1;
    ridge_cfg.scale_x = 0;
    ridge_cfg.scale_y = 0;
    n4m_matrix_view_t Xfv = rowmajor_view(
        train_features,
        static_cast<std::int64_t>(train_rows.size()),
        current_cols);
    n4m_matrix_view_t Yv = rowmajor_view(
        ytrain,
        static_cast<std::int64_t>(train_rows.size()),
        1);
    n4m_status_t status = fit_ridge(ctx, ridge_cfg, Xfv, Yv, alpha,
                                    RidgeSolver::kAuto, true, fitted.ridge);
    if (status != N4M_OK) return status;
    fitted.n_features = current_cols;
    return N4M_OK;
}

void predict_ridge(const RidgeResult& ridge,
                   const std::vector<double>& X,
                   std::int64_t rows,
                   std::int64_t cols,
                   std::vector<double>& pred) {
    const auto n = static_cast<std::size_t>(rows);
    const auto p = static_cast<std::size_t>(cols);
    pred.assign(n, 0.0);
    for (std::size_t r = 0; r < n; ++r) {
        double acc = ridge.intercept.empty() ? 0.0 : ridge.intercept[0];
        for (std::size_t c = 0; c < p; ++c) {
            acc += X[at(r, p, c)] * ridge.coefficients[c];
        }
        pred[r] = acc;
    }
}

[[nodiscard]] n4m_status_t fold_stack_to_input_space(
    Context& ctx,
    const std::vector<ChainSpec>& chains,
    const FittedStack& stack,
    std::int64_t n_input_features,
    std::vector<double>& input_coefficients,
    std::vector<double>& input_intercept) {
    const auto p = static_cast<std::size_t>(n_input_features);
    if (p == 0 || chains.size() != stack.views.size()) {
        ctx.set_error("AOM operator PLS stack input folding received inconsistent dimensions");
        return N4M_ERR_INVALID_ARGUMENT;
    }
    if (stack.ridge.intercept.empty()) {
        ctx.set_error("AOM operator PLS stack input folding missing Ridge intercept");
        return N4M_ERR_INVALID_ARGUMENT;
    }

    input_coefficients.assign(p, 0.0);
    input_intercept.assign(1, stack.ridge.intercept[0]);

    std::int64_t feature_offset = 0;
    for (std::size_t view_ix = 0; view_ix < stack.views.size(); ++view_ix) {
        const auto& view = stack.views[view_ix];
        const auto k = static_cast<std::size_t>(view.projector.n_components);
        if (view.scaler.mean.size() != p || view.scaler.scale.size() != p ||
            view.projector.x_mean.size() != p ||
            view.projector.rotations.size() != p * k ||
            feature_offset + static_cast<std::int64_t>(k) >
                static_cast<std::int64_t>(stack.ridge.coefficients.size())) {
            ctx.set_error("AOM operator PLS stack input folding found invalid fitted view");
            return N4M_ERR_INVALID_ARGUMENT;
        }

        std::vector<double> transformed_coefficients(p, 0.0);
        for (std::size_t comp = 0; comp < k; ++comp) {
            const double beta =
                stack.ridge.coefficients[
                    static_cast<std::size_t>(feature_offset) + comp];
            for (std::size_t c = 0; c < p; ++c) {
                const double rotation = view.projector.rotations[at(c, k, comp)];
                const double scale = view.scaler.scale[c];
                if (scale == 0.0 || !std::isfinite(scale)) {
                    ctx.set_error("AOM operator PLS stack input folding found invalid scale");
                    return N4M_ERR_INVALID_ARGUMENT;
                }
                transformed_coefficients[c] += beta * rotation / scale;
                input_intercept[0] -=
                    beta * ((view.scaler.mean[c] / scale) +
                            view.projector.x_mean[c]) * rotation;
            }
        }

        std::vector<double> folded;
        n4m_status_t status = fold_chain_coefficients_to_input_space(
            ctx, chains[view_ix], n_input_features, transformed_coefficients,
            folded);
        if (status != N4M_OK) return status;
        for (std::size_t c = 0; c < p; ++c) {
            input_coefficients[c] += folded[c];
        }
        feature_offset += static_cast<std::int64_t>(k);
    }
    if (feature_offset !=
        static_cast<std::int64_t>(stack.ridge.coefficients.size())) {
        ctx.set_error("AOM operator PLS stack input folding did not consume all Ridge coefficients");
        return N4M_ERR_INVALID_ARGUMENT;
    }
    return N4M_OK;
}

[[nodiscard]] double rmse_rows(const std::vector<double>& yall,
                               const std::vector<std::int64_t>& rows,
                               const std::vector<double>& pred) {
    double sse = 0.0;
    for (std::size_t i = 0; i < rows.size(); ++i) {
        const double diff = yall[static_cast<std::size_t>(rows[i])] - pred[i];
        sse += diff * diff;
    }
    return std::sqrt(sse / static_cast<double>(rows.size()));
}

struct SpecScore {
    double mean_oof{0.0};
    double std_oof{0.0};
    double mean_train{0.0};
    double criterion{0.0};
    std::vector<double> fold_scores;
};

[[nodiscard]] n4m_status_t score_spec(
    Context& ctx,
    const Config& cfg,
    const std::vector<std::vector<double>>& operator_views,
    const std::vector<double>& Yall,
    std::int64_t p,
    const std::vector<std::vector<std::int64_t>>& valid_rows,
    const std::vector<std::vector<std::int64_t>>& train_rows,
    std::int32_t components,
    double alpha,
    double std_penalty,
    double gap_penalty,
    SpecScore& score,
    std::vector<double>* oof_out) {
    std::vector<double> fold_losses;
    std::vector<double> train_losses;
    fold_losses.reserve(valid_rows.size());
    train_losses.reserve(valid_rows.size());
    if (oof_out != nullptr) {
        oof_out->assign(Yall.size(), 0.0);
    }

    for (std::size_t fold = 0; fold < valid_rows.size(); ++fold) {
        FittedStack fitted;
        std::vector<double> train_features;
        n4m_status_t status = fit_stack(ctx,
                                        cfg,
                                        operator_views,
                                        Yall,
                                        p,
                                        train_rows[fold],
                                        components,
                                        alpha,
                                        fitted,
                                        train_features);
        if (status != N4M_OK) return status;
        std::vector<double> train_pred;
        predict_ridge(fitted.ridge,
                      train_features,
                      static_cast<std::int64_t>(train_rows[fold].size()),
                      fitted.n_features,
                      train_pred);
        train_losses.push_back(rmse_rows(Yall, train_rows[fold], train_pred));

        std::vector<double> valid_features;
        status = transform_stack_rows(operator_views,
                                      p,
                                      valid_rows[fold],
                                      fitted.views,
                                      valid_features);
        if (status != N4M_OK) return status;
        std::vector<double> valid_pred;
        predict_ridge(fitted.ridge,
                      valid_features,
                      static_cast<std::int64_t>(valid_rows[fold].size()),
                      fitted.n_features,
                      valid_pred);
        fold_losses.push_back(rmse_rows(Yall, valid_rows[fold], valid_pred));
        if (oof_out != nullptr) {
            for (std::size_t i = 0; i < valid_rows[fold].size(); ++i) {
                (*oof_out)[static_cast<std::size_t>(valid_rows[fold][i])] =
                    valid_pred[i];
            }
        }
    }

    const double inv_folds = 1.0 / static_cast<double>(fold_losses.size());
    score.mean_oof = std::accumulate(fold_losses.begin(), fold_losses.end(), 0.0) * inv_folds;
    score.mean_train = std::accumulate(train_losses.begin(), train_losses.end(), 0.0) * inv_folds;
    score.std_oof = 0.0;
    if (fold_losses.size() > 1U) {
        double ss = 0.0;
        for (const auto value : fold_losses) {
            const double diff = value - score.mean_oof;
            ss += diff * diff;
        }
        score.std_oof = std::sqrt(ss / static_cast<double>(fold_losses.size() - 1U));
    }
    const double gap = std::max(0.0, score.mean_oof - score.mean_train);
    score.criterion = score.mean_oof + std_penalty * score.std_oof + gap_penalty * gap;
    score.fold_scores = std::move(fold_losses);
    return N4M_OK;
}

}  // namespace

n4m_status_t fit_aom_operator_pls_stack(
    Context& ctx,
    const Config& cfg,
    const n4m_matrix_view_t& X,
    const n4m_matrix_view_t& Y,
    std::int32_t profile,
    std::int32_t cv,
    const std::int32_t* fold_ids,
    std::int64_t n_fold_ids,
    const std::int32_t* components,
    std::int64_t n_components,
    const double* alphas,
    std::int64_t n_alphas,
    double std_penalty,
    double gap_penalty,
    AomOperatorPlsStackResult& out) {
    if (profile != static_cast<std::int32_t>(AomRobustHpoProfile::kCompact) &&
        profile != static_cast<std::int32_t>(AomRobustHpoProfile::kWide)) {
        ctx.set_error("AOM operator PLS stack profile must be 0=compact or 1=wide");
        return N4M_ERR_INVALID_ARGUMENT;
    }
    if (n_components <= 0 || components == nullptr) {
        ctx.set_error("AOM operator PLS stack components must not be empty");
        return N4M_ERR_INVALID_ARGUMENT;
    }
    if (n_alphas <= 0 || alphas == nullptr) {
        ctx.set_error("AOM operator PLS stack alphas must not be empty");
        return N4M_ERR_INVALID_ARGUMENT;
    }
    if (!std::isfinite(std_penalty) || std_penalty < 0.0 ||
        !std::isfinite(gap_penalty) || gap_penalty < 0.0) {
        ctx.set_error("AOM operator PLS stack penalties must be finite and non-negative");
        return N4M_ERR_INVALID_ARGUMENT;
    }

    std::vector<double> Xall;
    std::vector<double> Yall;
    n4m_status_t status = copy_matrix(ctx, X, "X", Xall);
    if (status != N4M_OK) return status;
    status = copy_matrix(ctx, Y, "Y", Yall);
    if (status != N4M_OK) return status;
    if (X.rows != Y.rows) {
        ctx.set_error("AOM operator PLS stack X and Y rows must match");
        return N4M_ERR_SHAPE_MISMATCH;
    }
    if (Y.cols != 1) {
        ctx.set_error("AOM operator PLS stack currently supports one Y target");
        return N4M_ERR_INVALID_ARGUMENT;
    }
    for (std::int64_t i = 0; i < n_components; ++i) {
        if (components[i] < 1) {
            ctx.set_error("AOM operator PLS stack components must be positive");
            return N4M_ERR_INVALID_ARGUMENT;
        }
    }
    for (std::int64_t i = 0; i < n_alphas; ++i) {
        if (!std::isfinite(alphas[i]) || alphas[i] < 0.0) {
            ctx.set_error("AOM operator PLS stack alphas must be finite and >= 0");
            return N4M_ERR_INVALID_ARGUMENT;
        }
    }

    std::int32_t actual_cv = 0;
    std::vector<std::int32_t> ids;
    status = make_fold_ids(ctx, X.rows, cv, fold_ids, n_fold_ids, ids, actual_cv);
    if (status != N4M_OK) return status;
    const auto valid_rows = fold_rows_from_ids(ids, actual_cv);
    const auto train_rows = train_rows_from_folds(X.rows, valid_rows);

    const auto chains = build_bank(profile);
    std::vector<std::vector<double>> operator_views;
    operator_views.reserve(chains.size());
    n4m_matrix_view_t Xview = X;
    if (X.dtype != N4M_DTYPE_F64 || X.row_stride != X.cols || X.col_stride != 1) {
        Xview = rowmajor_view(Xall, X.rows, X.cols);
    }
    for (const auto& chain : chains) {
        std::vector<double> Xt;
        status = transform_chain(ctx, chain, Xview, Xt);
        if (status != N4M_OK) return status;
        operator_views.push_back(std::move(Xt));
    }

    const std::int64_t n_specs = n_components * n_alphas;
    out = AomOperatorPlsStackResult{};
    out.n_samples = X.rows;
    out.n_features = X.cols;
    out.n_targets = 1;
    out.profile = profile;
    out.cv = actual_cv;
    out.n_operators = static_cast<std::int64_t>(operator_views.size());
    out.n_specs = n_specs;
    out.std_penalty = std_penalty;
    out.gap_penalty = gap_penalty;
    out.fold_ids = ids;
    out.candidate_scores.assign(static_cast<std::size_t>(n_specs * 7), 0.0);
    out.fold_scores.assign(static_cast<std::size_t>(n_specs * actual_cv), 0.0);

    double best = std::numeric_limits<double>::infinity();
    std::int64_t spec_id = 0;
    for (std::int64_t ci = 0; ci < n_components; ++ci) {
        for (std::int64_t ai = 0; ai < n_alphas; ++ai) {
            SpecScore score;
            status = score_spec(ctx,
                                cfg,
                                operator_views,
                                Yall,
                                X.cols,
                                valid_rows,
                                train_rows,
                                components[ci],
                                alphas[ai],
                                std_penalty,
                                gap_penalty,
                                score,
                                nullptr);
            if (status != N4M_OK) return status;
            const auto off = static_cast<std::size_t>(spec_id * 7);
            out.candidate_scores[off + 0] = static_cast<double>(spec_id);
            out.candidate_scores[off + 1] = static_cast<double>(components[ci]);
            out.candidate_scores[off + 2] = alphas[ai];
            out.candidate_scores[off + 3] = score.mean_oof;
            out.candidate_scores[off + 4] = score.std_oof;
            out.candidate_scores[off + 5] = score.mean_train;
            out.candidate_scores[off + 6] = score.criterion;
            for (std::int32_t f = 0; f < actual_cv; ++f) {
                out.fold_scores[static_cast<std::size_t>(spec_id * actual_cv + f)] =
                    score.fold_scores[static_cast<std::size_t>(f)];
            }
            if (score.criterion < best) {
                best = score.criterion;
                out.selected_spec_id = spec_id;
                out.selected_components = components[ci];
                out.selected_alpha = alphas[ai];
                out.selected_oof_rmse = score.mean_oof;
                out.selected_train_rmse = score.mean_train;
                out.selected_criterion = score.criterion;
            }
            ++spec_id;
        }
    }

    SpecScore selected_score;
    status = score_spec(ctx,
                        cfg,
                        operator_views,
                        Yall,
                        X.cols,
                        valid_rows,
                        train_rows,
                        out.selected_components,
                        out.selected_alpha,
                        std_penalty,
                        gap_penalty,
                        selected_score,
                        &out.oof_predictions);
    if (status != N4M_OK) return status;

    std::vector<std::int64_t> all_rows(static_cast<std::size_t>(X.rows));
    std::iota(all_rows.begin(), all_rows.end(), 0);
    FittedStack final_stack;
    status = fit_stack(ctx,
                       cfg,
                       operator_views,
                       Yall,
                       X.cols,
                       all_rows,
                       out.selected_components,
                       out.selected_alpha,
                       final_stack,
                       out.stack_features);
    if (status != N4M_OK) return status;
    predict_ridge(final_stack.ridge,
                  out.stack_features,
                  X.rows,
                  final_stack.n_features,
                  out.predictions);
    out.n_operator_features = final_stack.n_features;
    out.coefficients = final_stack.ridge.coefficients;
    out.intercept = final_stack.ridge.intercept;
    status = fold_stack_to_input_space(ctx,
                                       chains,
                                       final_stack,
                                       X.cols,
                                       out.input_coefficients,
                                       out.input_intercept);
    if (status != N4M_OK) return status;
    out.operator_feature_offsets.assign(final_stack.views.size() + 1U, 0);
    std::int32_t offset = 0;
    for (std::size_t i = 0; i < final_stack.views.size(); ++i) {
        out.operator_feature_offsets[i] = offset;
        offset += final_stack.views[i].projector.n_components;
    }
    out.operator_feature_offsets[final_stack.views.size()] = offset;
    return N4M_OK;
}

}  // namespace n4m::core
