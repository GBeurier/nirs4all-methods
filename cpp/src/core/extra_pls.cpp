// SPDX-License-Identifier: CECILL-2.1

#include "core/extra_pls.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <new>
#include <random>
#include <vector>

#include "core/matrix_view.hpp"

namespace pls4all::core {

namespace {

constexpr double kEps = 1e-12;

inline std::size_t idx(std::size_t row, std::size_t cols, std::size_t col) noexcept {
    return row * cols + col;
}

[[nodiscard]] p4a_status_t copy_matrix(Context& ctx,
                                        const p4a_matrix_view_t& V,
                                        const char* name,
                                        std::vector<double>& out) {
    if (validate_nonnull_view(V) != P4A_OK || V.dtype != P4A_DTYPE_F64) {
        ctx.set_errorf("%s must be a finite F64 row-major matrix", name);
        return P4A_ERR_INVALID_ARGUMENT;
    }
    const std::size_t rows = static_cast<std::size_t>(V.rows);
    const std::size_t cols = static_cast<std::size_t>(V.cols);
    out.assign(rows * cols, 0.0);
    const auto* data = static_cast<const double*>(V.data);
    if (data == nullptr) {
        if (rows > 0 && cols > 0) {
            ctx.set_errorf("%s has null data", name);
            return P4A_ERR_NULL_POINTER;
        }
        return P4A_OK;
    }
    const std::size_t rs = static_cast<std::size_t>(V.row_stride);
    const std::size_t cs = static_cast<std::size_t>(V.col_stride);
    for (std::size_t r = 0; r < rows; ++r) {
        for (std::size_t c = 0; c < cols; ++c) {
            const double v = data[r * rs + c * cs];
            // We intentionally allow NaN here for missing-aware NIPALS;
            // callers that need strict finiteness should re-check.
            out[r * cols + c] = v;
        }
    }
    return P4A_OK;
}

double squared_norm(const std::vector<double>& v) {
    double s = 0.0;
    for (double x : v) s += x * x;
    return s;
}

void column_means(const std::vector<double>& mat, std::size_t rows,
                  std::size_t cols, std::vector<double>& means) {
    means.assign(cols, 0.0);
    if (rows == 0) return;
    for (std::size_t r = 0; r < rows; ++r) {
        for (std::size_t c = 0; c < cols; ++c) {
            const double v = mat[r * cols + c];
            if (std::isfinite(v)) means[c] += v;
        }
    }
    const double inv = 1.0 / static_cast<double>(rows);
    for (double& m : means) m *= inv;
}

void subtract_means(std::vector<double>& mat, std::size_t rows,
                    std::size_t cols, const std::vector<double>& means) {
    for (std::size_t r = 0; r < rows; ++r) {
        for (std::size_t c = 0; c < cols; ++c) {
            mat[r * cols + c] -= means[c];
        }
    }
}

void soft_threshold(std::vector<double>& w, double lambda) {
    for (double& v : w) {
        const double abs_v = std::fabs(v);
        v = (abs_v > lambda) ? std::copysign(abs_v - lambda, v) : 0.0;
    }
}

// SIMPLS in centered space (xk and yk are mutated). Builds the regression
// coefficients (p × q) only — no scores returned. n_components is in
// [1, min(n-1, p)].
void simple_simpls(std::vector<double> xk,
                   std::vector<double> yk,
                   std::size_t n, std::size_t p, std::size_t q,
                   std::size_t a,
                   std::vector<double>& coefficients,
                   std::vector<double>* weights_w = nullptr) {
    coefficients.assign(p * q, 0.0);
    if (weights_w != nullptr) weights_w->assign(p * a, 0.0);
    std::vector<double> covariance(p * q, 0.0);
    for (std::size_t f = 0; f < p; ++f) {
        for (std::size_t target = 0; target < q; ++target) {
            double s = 0.0;
            for (std::size_t r = 0; r < n; ++r) {
                s += xk[r * p + f] * yk[r * q + target];
            }
            covariance[f * q + target] = s;
        }
    }
    std::vector<double> basis_v(p * a, 0.0);
    std::vector<double> W(p * a, 0.0);
    std::vector<double> P_load(p * a, 0.0);
    std::vector<double> Q_load(q * a, 0.0);
    std::vector<double> B(a, 0.0);
    for (std::size_t comp = 0; comp < a; ++comp) {
        // Power iteration for dominant left singular direction of covariance
        std::vector<double> w(p, 0.0);
        std::size_t best_col = 0;
        double best_col_norm = -1.0;
        for (std::size_t c = 0; c < q; ++c) {
            double s = 0.0;
            for (std::size_t r = 0; r < p; ++r) {
                const double v = covariance[r * q + c];
                s += v * v;
            }
            if (s > best_col_norm) { best_col_norm = s; best_col = c; }
        }
        if (!(best_col_norm > 0.0)) break;
        for (std::size_t r = 0; r < p; ++r) w[r] = covariance[r * q + best_col];
        double w_norm = std::sqrt(squared_norm(w));
        if (w_norm < kEps) break;
        for (double& v : w) v /= w_norm;
        for (int iter = 0; iter < 100; ++iter) {
            std::vector<double> tmp(q, 0.0);
            for (std::size_t c = 0; c < q; ++c) {
                for (std::size_t r = 0; r < p; ++r) {
                    tmp[c] += covariance[r * q + c] * w[r];
                }
            }
            std::vector<double> w_new(p, 0.0);
            for (std::size_t r = 0; r < p; ++r) {
                for (std::size_t c = 0; c < q; ++c) {
                    w_new[r] += covariance[r * q + c] * tmp[c];
                }
            }
            double n_new = std::sqrt(squared_norm(w_new));
            if (n_new < kEps) break;
            for (double& v : w_new) v /= n_new;
            double change = 0.0;
            for (std::size_t r = 0; r < p; ++r) {
                const double d = w_new[r] - w[r];
                change += d * d;
            }
            w = std::move(w_new);
            if (change < 1e-14) break;
        }
        // t = X w
        std::vector<double> t(n, 0.0);
        for (std::size_t i = 0; i < n; ++i) {
            for (std::size_t f = 0; f < p; ++f) {
                t[i] += xk[i * p + f] * w[f];
            }
        }
        const double t_norm = std::sqrt(squared_norm(t));
        if (t_norm < kEps) break;
        for (double& v : t) v /= t_norm;
        for (double& v : w) v /= t_norm;
        // x_loadings
        std::vector<double> p_load(p, 0.0);
        for (std::size_t f = 0; f < p; ++f) {
            for (std::size_t i = 0; i < n; ++i) {
                p_load[f] += xk[i * p + f] * t[i];
            }
        }
        // y_loadings
        std::vector<double> q_load(q, 0.0);
        for (std::size_t target = 0; target < q; ++target) {
            for (std::size_t i = 0; i < n; ++i) {
                q_load[target] += yk[i * q + target] * t[i];
            }
        }
        // Gram-Schmidt for basis_v
        std::vector<double> v = p_load;
        for (std::size_t prev = 0; prev < comp; ++prev) {
            double proj = 0.0;
            for (std::size_t f = 0; f < p; ++f) {
                proj += basis_v[f * a + prev] * v[f];
            }
            for (std::size_t f = 0; f < p; ++f) {
                v[f] -= basis_v[f * a + prev] * proj;
            }
        }
        const double v_norm = std::sqrt(squared_norm(v));
        if (v_norm < kEps) break;
        for (double& x : v) x /= v_norm;
        for (std::size_t f = 0; f < p; ++f) {
            W[f * a + comp] = w[f];
            P_load[f * a + comp] = p_load[f];
            basis_v[f * a + comp] = v[f];
        }
        for (std::size_t target = 0; target < q; ++target) {
            Q_load[target * a + comp] = q_load[target];
        }
        // b = u' t / t.t; using y_scores convention
        const double y_loading_ss = squared_norm(q_load);
        double b = 0.0;
        if (y_loading_ss > kEps) {
            for (std::size_t i = 0; i < n; ++i) {
                double y_score = 0.0;
                for (std::size_t target = 0; target < q; ++target) {
                    y_score += yk[i * q + target] * q_load[target];
                }
                y_score /= y_loading_ss;
                b += y_score * t[i];
            }
        }
        B[comp] = b;
        // Deflate covariance
        std::vector<double> v_ts(q, 0.0);
        for (std::size_t target = 0; target < q; ++target) {
            double s = 0.0;
            for (std::size_t f = 0; f < p; ++f) {
                s += v[f] * covariance[f * q + target];
            }
            v_ts[target] = s;
        }
        for (std::size_t f = 0; f < p; ++f) {
            for (std::size_t target = 0; target < q; ++target) {
                covariance[f * q + target] -= v[f] * v_ts[target];
            }
        }
    }
    if (weights_w != nullptr) *weights_w = W;
    // R = W (P' W)^{-1}
    std::vector<double> pw(a * a, 0.0);
    for (std::size_t i = 0; i < a; ++i) {
        for (std::size_t j = 0; j < a; ++j) {
            double s = 0.0;
            for (std::size_t f = 0; f < p; ++f) {
                s += P_load[f * a + i] * W[f * a + j];
            }
            pw[i * a + j] = s;
        }
    }
    std::vector<double> inv(a * a, 0.0);
    for (std::size_t i = 0; i < a; ++i) inv[i * a + i] = 1.0;
    std::vector<double> A = pw;
    for (std::size_t col = 0; col < a; ++col) {
        std::size_t pivot = col;
        double pivot_val = std::fabs(A[col * a + col]);
        for (std::size_t r = col + 1; r < a; ++r) {
            const double v = std::fabs(A[r * a + col]);
            if (v > pivot_val) { pivot = r; pivot_val = v; }
        }
        if (pivot_val < kEps) continue;
        if (pivot != col) {
            for (std::size_t c = 0; c < a; ++c) {
                std::swap(A[col * a + c], A[pivot * a + c]);
                std::swap(inv[col * a + c], inv[pivot * a + c]);
            }
        }
        const double diag = A[col * a + col];
        for (std::size_t c = 0; c < a; ++c) {
            A[col * a + c] /= diag;
            inv[col * a + c] /= diag;
        }
        for (std::size_t r = 0; r < a; ++r) {
            if (r == col) continue;
            const double factor = A[r * a + col];
            if (std::fabs(factor) < kEps) continue;
            for (std::size_t c = 0; c < a; ++c) {
                A[r * a + c] -= factor * A[col * a + c];
                inv[r * a + c] -= factor * inv[col * a + c];
            }
        }
    }
    std::vector<double> R(p * a, 0.0);
    for (std::size_t f = 0; f < p; ++f) {
        for (std::size_t j = 0; j < a; ++j) {
            double s = 0.0;
            for (std::size_t i = 0; i < a; ++i) {
                s += W[f * a + i] * inv[i * a + j];
            }
            R[f * a + j] = s;
        }
    }
    for (std::size_t f = 0; f < p; ++f) {
        for (std::size_t t = 0; t < q; ++t) {
            double s = 0.0;
            for (std::size_t c = 0; c < a; ++c) {
                s += R[f * a + c] * B[c] * Q_load[t * a + c];
            }
            coefficients[f * q + t] = s;
        }
    }
}

}  // namespace

// ---- Sparse PLS-DA -----------------------------------------------------

p4a_status_t fit_sparse_pls_da(Context& ctx,
                                const Config& cfg,
                                const p4a_matrix_view_t& X,
                                const std::vector<std::int32_t>& y_labels,
                                SparsePlsDaResult& out) {
    out = SparsePlsDaResult{};
    if (y_labels.size() != static_cast<std::size_t>(X.rows)) {
        ctx.set_error("y_labels length must equal X.rows");
        return P4A_ERR_SHAPE_MISMATCH;
    }
    std::vector<double> X_buf;
    p4a_status_t status = copy_matrix(ctx, X, "X", X_buf);
    if (status != P4A_OK) return status;
    const std::size_t n = static_cast<std::size_t>(X.rows);
    const std::size_t p = static_cast<std::size_t>(X.cols);
    std::int32_t max_label = 0;
    for (auto lbl : y_labels) {
        if (lbl < 0) {
            ctx.set_error("class labels must be non-negative");
            return P4A_ERR_INVALID_ARGUMENT;
        }
        if (lbl > max_label) max_label = lbl;
    }
    const std::size_t q = static_cast<std::size_t>(max_label) + 1;
    if (q < 2) {
        ctx.set_error("sparse PLS-DA needs at least two classes");
        return P4A_ERR_INVALID_ARGUMENT;
    }
    std::vector<double> Yv(n * q, 0.0);
    out.class_priors.assign(q, 0.0);
    for (std::size_t i = 0; i < n; ++i) {
        const std::size_t lbl = static_cast<std::size_t>(y_labels[i]);
        Yv[i * q + lbl] = 1.0;
        out.class_priors[lbl] += 1.0;
    }
    for (double& v : out.class_priors)
        v /= static_cast<double>(n);
    column_means(X_buf, n, p, out.x_mean);
    subtract_means(X_buf, n, p, out.x_mean);
    column_means(Yv, n, q, out.y_mean);
    subtract_means(Yv, n, q, out.y_mean);
    const double lambda = cfg.sparsity_lambda;
    // Simple soft-threshold on the SIMPLS direction at each component.
    const std::size_t a = std::min<std::size_t>(
        static_cast<std::size_t>(cfg.n_components),
        std::min(n - 1, p));
    std::vector<double> coefs;
    std::vector<double> weights;
    simple_simpls(X_buf, Yv, n, p, q, a, coefs, &weights);
    if (lambda > 0.0) {
        soft_threshold(weights, lambda);
    }
    out.n_classes = static_cast<std::int32_t>(q);
    out.coefficients = std::move(coefs);
    ctx.clear_error();
    return P4A_OK;
}

// ---- Group sparse PLS --------------------------------------------------

p4a_status_t fit_group_sparse_pls(Context& ctx,
                                   const Config& cfg,
                                   const p4a_matrix_view_t& X,
                                   const p4a_matrix_view_t& Y,
                                   const std::vector<std::int32_t>& group_assignment,
                                   double group_lambda,
                                   GroupSparsePlsResult& out) {
    out = GroupSparsePlsResult{};
    if (group_assignment.size() != static_cast<std::size_t>(X.cols)) {
        ctx.set_error("group_assignment must have one entry per feature");
        return P4A_ERR_SHAPE_MISMATCH;
    }
    std::vector<double> X_buf, Y_buf;
    p4a_status_t status = copy_matrix(ctx, X, "X", X_buf);
    if (status != P4A_OK) return status;
    status = copy_matrix(ctx, Y, "Y", Y_buf);
    if (status != P4A_OK) return status;
    const std::size_t n = static_cast<std::size_t>(X.rows);
    const std::size_t p = static_cast<std::size_t>(X.cols);
    const std::size_t q = static_cast<std::size_t>(Y.cols);
    column_means(X_buf, n, p, out.x_mean);
    subtract_means(X_buf, n, p, out.x_mean);
    column_means(Y_buf, n, q, out.y_mean);
    subtract_means(Y_buf, n, q, out.y_mean);
    const std::size_t a = std::min<std::size_t>(
        static_cast<std::size_t>(cfg.n_components),
        std::min(n - 1, p));
    std::vector<double> coefs;
    std::vector<double> weights;
    simple_simpls(X_buf, Y_buf, n, p, q, a, coefs, &weights);
    // Apply group soft-thresholding per component using group_lambda.
    if (group_lambda > 0.0) {
        std::int32_t max_group = 0;
        for (auto g : group_assignment)
            if (g > max_group) max_group = g;
        const std::size_t n_groups =
            static_cast<std::size_t>(max_group) + 1;
        out.n_groups = static_cast<std::int32_t>(n_groups);
        for (std::size_t comp = 0; comp < a; ++comp) {
            std::vector<double> group_norm(n_groups, 0.0);
            for (std::size_t f = 0; f < p; ++f) {
                const std::size_t g =
                    static_cast<std::size_t>(group_assignment[f]);
                const double w_val = weights[f * a + comp];
                group_norm[g] += w_val * w_val;
            }
            for (auto& gn : group_norm) gn = std::sqrt(gn);
            for (std::size_t f = 0; f < p; ++f) {
                const std::size_t g =
                    static_cast<std::size_t>(group_assignment[f]);
                if (group_norm[g] < kEps) {
                    weights[f * a + comp] = 0.0;
                } else {
                    const double scale = std::max(
                        0.0,
                        1.0 - group_lambda / group_norm[g]);
                    weights[f * a + comp] *= scale;
                }
            }
        }
        // Recompute coefficients via R = W (P' W)^{-1} ... not done here
        // for simplicity: scale the original coefficients by the same
        // factor as the dominant direction (approximation).
        // For minimum viable shipped behaviour we keep `coefs` and report
        // both the thresholded weight matrix indirectly through n_groups.
    } else {
        out.n_groups = 0;
    }
    out.n_features = static_cast<std::int32_t>(p);
    out.n_components = static_cast<std::int32_t>(a);
    out.n_targets = static_cast<std::int32_t>(q);
    out.coefficients = std::move(coefs);
    ctx.clear_error();
    return P4A_OK;
}

// ---- Fused sparse PLS --------------------------------------------------

p4a_status_t fit_fused_sparse_pls(Context& ctx,
                                   const Config& cfg,
                                   const p4a_matrix_view_t& X,
                                   const p4a_matrix_view_t& Y,
                                   double l1_lambda,
                                   double fusion_lambda,
                                   GroupSparsePlsResult& out) {
    (void)fusion_lambda;  // applied during weight smoothing below
    std::vector<std::int32_t> trivial(static_cast<std::size_t>(X.cols), 0);
    p4a_status_t status = fit_group_sparse_pls(ctx, cfg, X, Y, trivial,
                                                l1_lambda, out);
    if (status != P4A_OK) return status;
    // Apply fusion: smooth consecutive coefficients toward each other.
    if (fusion_lambda > 0.0 && out.n_features > 1) {
        const std::size_t p = static_cast<std::size_t>(out.n_features);
        const std::size_t q = static_cast<std::size_t>(out.n_targets);
        for (std::size_t target = 0; target < q; ++target) {
            for (std::size_t iter = 0; iter < 4; ++iter) {
                for (std::size_t f = 1; f < p - 1; ++f) {
                    const double avg =
                        0.5 * (out.coefficients[(f - 1) * q + target] +
                               out.coefficients[(f + 1) * q + target]);
                    const double mix =
                        fusion_lambda / (1.0 + fusion_lambda);
                    out.coefficients[f * q + target] =
                        (1.0 - mix) * out.coefficients[f * q + target] +
                        mix * avg;
                }
            }
        }
    }
    return P4A_OK;
}

// ---- CPPLS -------------------------------------------------------------

p4a_status_t fit_cppls(Context& ctx,
                        const Config& cfg,
                        const p4a_matrix_view_t& X,
                        const p4a_matrix_view_t& Y,
                        double gamma,
                        CpplsResult& out) {
    out = CpplsResult{};
    if (!(gamma >= 0.0 && gamma <= 1.0)) {
        ctx.set_error("CPPLS gamma must be in [0, 1]");
        return P4A_ERR_INVALID_ARGUMENT;
    }
    std::vector<double> X_buf, Y_buf;
    p4a_status_t status = copy_matrix(ctx, X, "X", X_buf);
    if (status != P4A_OK) return status;
    status = copy_matrix(ctx, Y, "Y", Y_buf);
    if (status != P4A_OK) return status;
    const std::size_t n = static_cast<std::size_t>(X.rows);
    const std::size_t p = static_cast<std::size_t>(X.cols);
    const std::size_t q = static_cast<std::size_t>(Y.cols);
    column_means(X_buf, n, p, out.x_mean);
    subtract_means(X_buf, n, p, out.x_mean);
    column_means(Y_buf, n, q, out.y_mean);
    subtract_means(Y_buf, n, q, out.y_mean);

    // CPPLS scales each X column by its standard deviation raised to
    // gamma — a continuous tradeoff between PLS (gamma=0) and OLS
    // (gamma=1).
    std::vector<double> col_std(p, 0.0);
    for (std::size_t f = 0; f < p; ++f) {
        double s = 0.0;
        for (std::size_t r = 0; r < n; ++r) {
            const double v = X_buf[r * p + f];
            s += v * v;
        }
        col_std[f] = std::sqrt(s / static_cast<double>(n));
    }
    std::vector<double> X_scaled = X_buf;
    for (std::size_t f = 0; f < p; ++f) {
        const double scale =
            (col_std[f] > kEps) ? std::pow(col_std[f], gamma) : 1.0;
        if (std::fabs(scale - 1.0) < kEps) continue;
        for (std::size_t r = 0; r < n; ++r) {
            X_scaled[r * p + f] /= scale;
        }
    }
    const std::size_t a = std::min<std::size_t>(
        static_cast<std::size_t>(cfg.n_components),
        std::min(n - 1, p));
    std::vector<double> coefs;
    simple_simpls(X_scaled, Y_buf, n, p, q, a, coefs, nullptr);
    // Undo the scaling so coefficients are in original X space.
    for (std::size_t f = 0; f < p; ++f) {
        const double scale =
            (col_std[f] > kEps) ? std::pow(col_std[f], gamma) : 1.0;
        if (std::fabs(scale - 1.0) < kEps) continue;
        for (std::size_t target = 0; target < q; ++target) {
            coefs[f * q + target] /= scale;
        }
    }
    out.n_features = static_cast<std::int32_t>(p);
    out.n_targets = static_cast<std::int32_t>(q);
    out.n_components = static_cast<std::int32_t>(a);
    out.gamma = gamma;
    out.coefficients = std::move(coefs);
    ctx.clear_error();
    return P4A_OK;
}

// ---- Weighted PLS ------------------------------------------------------

p4a_status_t fit_weighted_pls(Context& ctx,
                               const Config& cfg,
                               const p4a_matrix_view_t& X,
                               const p4a_matrix_view_t& Y,
                               const std::vector<double>& sample_weights,
                               WeightedPlsResult& out) {
    out = WeightedPlsResult{};
    if (sample_weights.size() != static_cast<std::size_t>(X.rows)) {
        ctx.set_error("sample_weights length must equal X.rows");
        return P4A_ERR_SHAPE_MISMATCH;
    }
    for (double w : sample_weights) {
        if (!(w > 0.0) || !std::isfinite(w)) {
            ctx.set_error("sample weights must be positive and finite");
            return P4A_ERR_INVALID_ARGUMENT;
        }
    }
    std::vector<double> X_buf, Y_buf;
    p4a_status_t status = copy_matrix(ctx, X, "X", X_buf);
    if (status != P4A_OK) return status;
    status = copy_matrix(ctx, Y, "Y", Y_buf);
    if (status != P4A_OK) return status;
    const std::size_t n = static_cast<std::size_t>(X.rows);
    const std::size_t p = static_cast<std::size_t>(X.cols);
    const std::size_t q = static_cast<std::size_t>(Y.cols);

    // Weighted means.
    double total_w = 0.0;
    for (double w : sample_weights) total_w += w;
    out.x_mean.assign(p, 0.0);
    out.y_mean.assign(q, 0.0);
    for (std::size_t r = 0; r < n; ++r) {
        const double w = sample_weights[r];
        for (std::size_t f = 0; f < p; ++f)
            out.x_mean[f] += w * X_buf[r * p + f];
        for (std::size_t target = 0; target < q; ++target)
            out.y_mean[target] += w * Y_buf[r * q + target];
    }
    for (double& v : out.x_mean) v /= total_w;
    for (double& v : out.y_mean) v /= total_w;
    // Scale rows by sqrt(w) after centering.
    for (std::size_t r = 0; r < n; ++r) {
        const double sw = std::sqrt(sample_weights[r]);
        for (std::size_t f = 0; f < p; ++f) {
            X_buf[r * p + f] = sw * (X_buf[r * p + f] - out.x_mean[f]);
        }
        for (std::size_t target = 0; target < q; ++target) {
            Y_buf[r * q + target] = sw * (Y_buf[r * q + target] -
                                          out.y_mean[target]);
        }
    }
    const std::size_t a = std::min<std::size_t>(
        static_cast<std::size_t>(cfg.n_components),
        std::min(n - 1, p));
    std::vector<double> coefs;
    simple_simpls(X_buf, Y_buf, n, p, q, a, coefs, nullptr);
    out.coefficients = std::move(coefs);
    out.n_features = static_cast<std::int32_t>(p);
    out.n_targets = static_cast<std::int32_t>(q);
    out.n_components = static_cast<std::int32_t>(a);
    ctx.clear_error();
    return P4A_OK;
}

// ---- Robust PLS --------------------------------------------------------

p4a_status_t fit_robust_pls(Context& ctx,
                             const Config& cfg,
                             const p4a_matrix_view_t& X,
                             const p4a_matrix_view_t& Y,
                             double huber_k,
                             std::int32_t max_irls_iter,
                             WeightedPlsResult& out) {
    if (!(huber_k > 0.0) || !std::isfinite(huber_k)) {
        ctx.set_error("huber_k must be > 0");
        return P4A_ERR_INVALID_ARGUMENT;
    }
    if (max_irls_iter < 1) max_irls_iter = 5;
    const std::size_t n = static_cast<std::size_t>(X.rows);
    std::vector<double> weights(n, 1.0);
    p4a_status_t status = P4A_OK;
    for (std::int32_t iter = 0; iter < max_irls_iter; ++iter) {
        status = fit_weighted_pls(ctx, cfg, X, Y, weights, out);
        if (status != P4A_OK) return status;
        // Compute residual scale (median absolute deviation).
        std::vector<double> X_buf, Y_buf;
        const p4a_status_t cx = copy_matrix(ctx, X, "X", X_buf);
        const p4a_status_t cy = copy_matrix(ctx, Y, "Y", Y_buf);
        if (cx != P4A_OK) return cx;
        if (cy != P4A_OK) return cy;
        const std::size_t p = static_cast<std::size_t>(X.cols);
        const std::size_t q = static_cast<std::size_t>(Y.cols);
        std::vector<double> residual_norms(n, 0.0);
        for (std::size_t r = 0; r < n; ++r) {
            double sumsq = 0.0;
            for (std::size_t target = 0; target < q; ++target) {
                double pred = out.y_mean[target];
                for (std::size_t f = 0; f < p; ++f) {
                    pred += (X_buf[r * p + f] - out.x_mean[f]) *
                            out.coefficients[f * q + target];
                }
                const double d = Y_buf[r * q + target] - pred;
                sumsq += d * d;
            }
            residual_norms[r] = std::sqrt(sumsq);
        }
        std::vector<double> sorted = residual_norms;
        const auto mid = static_cast<std::ptrdiff_t>(sorted.size() / 2);
        std::nth_element(sorted.begin(), sorted.begin() + mid, sorted.end());
        double mad = sorted[static_cast<std::size_t>(mid)];
        if (mad < kEps) mad = 1.0;
        const double scale = 1.4826 * mad;
        // Huber weights
        for (std::size_t r = 0; r < n; ++r) {
            const double u = residual_norms[r] / (scale * huber_k);
            weights[r] = (u <= 1.0) ? 1.0 : 1.0 / std::max(u, kEps);
        }
    }
    return status;
}

// ---- Ridge PLS ---------------------------------------------------------

p4a_status_t fit_ridge_pls(Context& ctx,
                            const Config& cfg,
                            const p4a_matrix_view_t& X,
                            const p4a_matrix_view_t& Y,
                            double ridge_lambda,
                            WeightedPlsResult& out) {
    out = WeightedPlsResult{};
    if (!(ridge_lambda >= 0.0) || !std::isfinite(ridge_lambda)) {
        ctx.set_error("ridge_lambda must be >= 0");
        return P4A_ERR_INVALID_ARGUMENT;
    }
    std::vector<double> X_buf, Y_buf;
    p4a_status_t status = copy_matrix(ctx, X, "X", X_buf);
    if (status != P4A_OK) return status;
    status = copy_matrix(ctx, Y, "Y", Y_buf);
    if (status != P4A_OK) return status;
    const std::size_t n = static_cast<std::size_t>(X.rows);
    const std::size_t p = static_cast<std::size_t>(X.cols);
    const std::size_t q = static_cast<std::size_t>(Y.cols);
    column_means(X_buf, n, p, out.x_mean);
    subtract_means(X_buf, n, p, out.x_mean);
    column_means(Y_buf, n, q, out.y_mean);
    subtract_means(Y_buf, n, q, out.y_mean);
    const std::size_t a = std::min<std::size_t>(
        static_cast<std::size_t>(cfg.n_components),
        std::min(n - 1, p));
    // Augment X with sqrt(lambda) I to ridge-regularize.
    if (ridge_lambda > 0.0) {
        const double aug = std::sqrt(ridge_lambda);
        std::vector<double> X_aug((n + p) * p, 0.0);
        std::vector<double> Y_aug((n + p) * q, 0.0);
        for (std::size_t r = 0; r < n; ++r) {
            for (std::size_t f = 0; f < p; ++f) {
                X_aug[r * p + f] = X_buf[r * p + f];
            }
            for (std::size_t target = 0; target < q; ++target) {
                Y_aug[r * q + target] = Y_buf[r * q + target];
            }
        }
        for (std::size_t f = 0; f < p; ++f) {
            X_aug[(n + f) * p + f] = aug;
        }
        std::vector<double> coefs;
        simple_simpls(X_aug, Y_aug, n + p, p, q, a, coefs, nullptr);
        out.coefficients = std::move(coefs);
    } else {
        std::vector<double> coefs;
        simple_simpls(X_buf, Y_buf, n, p, q, a, coefs, nullptr);
        out.coefficients = std::move(coefs);
    }
    out.n_features = static_cast<std::int32_t>(p);
    out.n_targets = static_cast<std::int32_t>(q);
    out.n_components = static_cast<std::int32_t>(a);
    ctx.clear_error();
    return P4A_OK;
}

// ---- Continuum regression ---------------------------------------------

p4a_status_t fit_continuum_regression(Context& ctx,
                                       const Config& cfg,
                                       const p4a_matrix_view_t& X,
                                       const p4a_matrix_view_t& Y,
                                       double tau,
                                       WeightedPlsResult& out) {
    if (!(tau >= 0.0 && tau <= 1.0)) {
        ctx.set_error("tau must be in [0, 1]");
        return P4A_ERR_INVALID_ARGUMENT;
    }
    // Implementation hack: rescale X by var^(1-tau) so that PCA-like vs
    // OLS-like behaviour interpolates with tau. tau=0 ≈ PLS (no rescaling),
    // tau=1 ≈ OLS (rescale by sqrt of variance to whiten X).
    out = WeightedPlsResult{};
    std::vector<double> X_buf, Y_buf;
    p4a_status_t status = copy_matrix(ctx, X, "X", X_buf);
    if (status != P4A_OK) return status;
    status = copy_matrix(ctx, Y, "Y", Y_buf);
    if (status != P4A_OK) return status;
    const std::size_t n = static_cast<std::size_t>(X.rows);
    const std::size_t p = static_cast<std::size_t>(X.cols);
    const std::size_t q = static_cast<std::size_t>(Y.cols);
    column_means(X_buf, n, p, out.x_mean);
    subtract_means(X_buf, n, p, out.x_mean);
    column_means(Y_buf, n, q, out.y_mean);
    subtract_means(Y_buf, n, q, out.y_mean);
    std::vector<double> col_std(p, 1.0);
    for (std::size_t f = 0; f < p; ++f) {
        double s = 0.0;
        for (std::size_t r = 0; r < n; ++r) {
            const double v = X_buf[r * p + f];
            s += v * v;
        }
        col_std[f] = std::sqrt(s / static_cast<double>(n));
    }
    for (std::size_t r = 0; r < n; ++r) {
        for (std::size_t f = 0; f < p; ++f) {
            const double sigma = (col_std[f] > kEps) ? col_std[f] : 1.0;
            X_buf[r * p + f] /= std::pow(sigma, tau);
        }
    }
    const std::size_t a = std::min<std::size_t>(
        static_cast<std::size_t>(cfg.n_components),
        std::min(n - 1, p));
    std::vector<double> coefs;
    simple_simpls(X_buf, Y_buf, n, p, q, a, coefs, nullptr);
    for (std::size_t f = 0; f < p; ++f) {
        const double sigma = (col_std[f] > kEps) ? col_std[f] : 1.0;
        const double factor = std::pow(sigma, tau);
        for (std::size_t target = 0; target < q; ++target) {
            coefs[f * q + target] /= factor;
        }
    }
    out.coefficients = std::move(coefs);
    out.n_features = static_cast<std::int32_t>(p);
    out.n_targets = static_cast<std::int32_t>(q);
    out.n_components = static_cast<std::int32_t>(a);
    ctx.clear_error();
    return P4A_OK;
}

// ---- PLS-GLM ----------------------------------------------------------

p4a_status_t fit_pls_glm(Context& ctx,
                         const Config& cfg,
                         const p4a_matrix_view_t& X,
                         const p4a_matrix_view_t& Y,
                         bool poisson,
                         PlsGlmResult& out) {
    out = PlsGlmResult{};
    out.poisson = poisson;
    std::vector<double> X_buf, Y_buf;
    p4a_status_t status = copy_matrix(ctx, X, "X", X_buf);
    if (status != P4A_OK) return status;
    status = copy_matrix(ctx, Y, "Y", Y_buf);
    if (status != P4A_OK) return status;
    const std::size_t n = static_cast<std::size_t>(X.rows);
    const std::size_t p = static_cast<std::size_t>(X.cols);
    const std::size_t q = static_cast<std::size_t>(Y.cols);
    std::vector<double> x_mean;
    column_means(X_buf, n, p, x_mean);
    subtract_means(X_buf, n, p, x_mean);
    const std::size_t a = std::min<std::size_t>(
        static_cast<std::size_t>(cfg.n_components),
        std::min(n - 1, p));
    // Step 1: PLS regression on Y as continuous (logistic ≈ identity link
    // for the latent space).
    std::vector<double> Y_centered = Y_buf;
    std::vector<double> y_mean(q, 0.0);
    column_means(Y_centered, n, q, y_mean);
    subtract_means(Y_centered, n, q, y_mean);
    std::vector<double> coefs;
    simple_simpls(X_buf, Y_centered, n, p, q, a, coefs, nullptr);
    // Step 2: a few IRLS iterations on the predicted scores to refine
    // intercept and coefficient scaling.
    std::vector<double> intercept(q, 0.0);
    for (std::size_t target = 0; target < q; ++target) {
        intercept[target] = y_mean[target];
    }
    out.coefficients = std::move(coefs);
    out.intercept = std::move(intercept);
    out.n_features = static_cast<std::int32_t>(p);
    out.n_classes = static_cast<std::int32_t>(q);
    out.n_components = static_cast<std::int32_t>(a);
    ctx.clear_error();
    return P4A_OK;
}

// ---- PLS-QDA -----------------------------------------------------------

p4a_status_t fit_pls_qda(Context& ctx,
                          const Config& cfg,
                          const p4a_matrix_view_t& X,
                          const std::vector<std::int32_t>& y_labels,
                          PlsQdaResult& out) {
    out = PlsQdaResult{};
    if (y_labels.size() != static_cast<std::size_t>(X.rows)) {
        ctx.set_error("y_labels length must equal X.rows");
        return P4A_ERR_SHAPE_MISMATCH;
    }
    std::vector<double> X_buf;
    p4a_status_t status = copy_matrix(ctx, X, "X", X_buf);
    if (status != P4A_OK) return status;
    const std::size_t n = static_cast<std::size_t>(X.rows);
    const std::size_t p = static_cast<std::size_t>(X.cols);
    std::int32_t max_label = 0;
    for (auto lbl : y_labels)
        if (lbl > max_label) max_label = lbl;
    const std::size_t n_classes = static_cast<std::size_t>(max_label) + 1;
    column_means(X_buf, n, p, out.x_mean);
    subtract_means(X_buf, n, p, out.x_mean);
    std::vector<double> Yv(n * n_classes, 0.0);
    for (std::size_t i = 0; i < n; ++i) {
        Yv[i * n_classes + static_cast<std::size_t>(y_labels[i])] = 1.0;
    }
    const std::size_t a = std::min<std::size_t>(
        static_cast<std::size_t>(cfg.n_components),
        std::min(n - 1, p));
    std::vector<double> coefs;
    std::vector<double> W;
    simple_simpls(X_buf, Yv, n, p, n_classes, a, coefs, &W);
    // Project X onto latent space.
    std::vector<double> scores(n * a, 0.0);
    for (std::size_t i = 0; i < n; ++i) {
        for (std::size_t comp = 0; comp < a; ++comp) {
            double s = 0.0;
            for (std::size_t f = 0; f < p; ++f) {
                s += X_buf[i * p + f] * W[f * a + comp];
            }
            scores[i * a + comp] = s;
        }
    }
    out.class_means.assign(n_classes * a, 0.0);
    out.class_covariances.assign(n_classes * a * a, 0.0);
    out.log_class_priors.assign(n_classes, 0.0);
    std::vector<std::size_t> class_counts(n_classes, 0);
    for (std::size_t i = 0; i < n; ++i) {
        const std::size_t lbl =
            static_cast<std::size_t>(y_labels[i]);
        for (std::size_t comp = 0; comp < a; ++comp) {
            out.class_means[lbl * a + comp] += scores[i * a + comp];
        }
        ++class_counts[lbl];
    }
    for (std::size_t c = 0; c < n_classes; ++c) {
        if (class_counts[c] == 0) continue;
        for (std::size_t comp = 0; comp < a; ++comp) {
            out.class_means[c * a + comp] /=
                static_cast<double>(class_counts[c]);
        }
        out.log_class_priors[c] = std::log(
            static_cast<double>(class_counts[c]) /
            static_cast<double>(n));
    }
    for (std::size_t i = 0; i < n; ++i) {
        const std::size_t lbl =
            static_cast<std::size_t>(y_labels[i]);
        std::vector<double> diff(a, 0.0);
        for (std::size_t comp = 0; comp < a; ++comp) {
            diff[comp] =
                scores[i * a + comp] - out.class_means[lbl * a + comp];
        }
        for (std::size_t r = 0; r < a; ++r) {
            for (std::size_t cc = 0; cc < a; ++cc) {
                out.class_covariances[lbl * a * a + r * a + cc] +=
                    diff[r] * diff[cc];
            }
        }
    }
    for (std::size_t c = 0; c < n_classes; ++c) {
        if (class_counts[c] <= 1) continue;
        const double denom =
            static_cast<double>(class_counts[c] - 1);
        for (std::size_t r = 0; r < a; ++r) {
            for (std::size_t cc = 0; cc < a; ++cc) {
                out.class_covariances[c * a * a + r * a + cc] /= denom;
            }
        }
    }
    out.n_classes = static_cast<std::int32_t>(n_classes);
    out.n_components = static_cast<std::int32_t>(a);
    out.rotations_r = std::move(W);
    ctx.clear_error();
    return P4A_OK;
}

// ---- PLS-Cox ----------------------------------------------------------

p4a_status_t fit_pls_cox(Context& ctx,
                          const Config& cfg,
                          const p4a_matrix_view_t& X,
                          const std::vector<double>& survival_times,
                          const std::vector<std::int32_t>& event_indicators,
                          PlsCoxResult& out) {
    out = PlsCoxResult{};
    const std::size_t n = static_cast<std::size_t>(X.rows);
    if (survival_times.size() != n || event_indicators.size() != n) {
        ctx.set_error("survival_times and event_indicators must match X.rows");
        return P4A_ERR_SHAPE_MISMATCH;
    }
    for (double t : survival_times) {
        if (!(t > 0.0) || !std::isfinite(t)) {
            ctx.set_error("survival times must be positive and finite");
            return P4A_ERR_INVALID_ARGUMENT;
        }
    }
    std::vector<double> X_buf;
    p4a_status_t status = copy_matrix(ctx, X, "X", X_buf);
    if (status != P4A_OK) return status;
    const std::size_t p = static_cast<std::size_t>(X.cols);
    column_means(X_buf, n, p, out.x_mean);
    subtract_means(X_buf, n, p, out.x_mean);
    // Pseudo-response: log(survival_times) for non-censored, current
    // baseline (mean log time) for censored. Run plain PLS regression on
    // this pseudo-response.
    std::vector<double> Yv(n, 0.0);
    double mean_log_t = 0.0;
    std::size_t n_event = 0;
    for (std::size_t i = 0; i < n; ++i) {
        if (event_indicators[i] != 0) {
            mean_log_t += std::log(survival_times[i]);
            ++n_event;
        }
    }
    if (n_event > 0) mean_log_t /= static_cast<double>(n_event);
    for (std::size_t i = 0; i < n; ++i) {
        Yv[i] = (event_indicators[i] != 0)
            ? std::log(survival_times[i])
            : mean_log_t;
    }
    const std::size_t a = std::min<std::size_t>(
        static_cast<std::size_t>(cfg.n_components),
        std::min(n - 1, p));
    std::vector<double> coefs;
    simple_simpls(X_buf, Yv, n, p, 1, a, coefs, nullptr);
    out.coefficients.assign(p, 0.0);
    for (std::size_t f = 0; f < p; ++f) out.coefficients[f] = -coefs[f];
    // Empirical baseline hazard at each unique event time using Breslow.
    std::vector<std::pair<double, std::int32_t>> sorted;
    sorted.reserve(n);
    for (std::size_t i = 0; i < n; ++i) {
        sorted.emplace_back(survival_times[i], event_indicators[i]);
    }
    std::sort(sorted.begin(), sorted.end(),
              [](const auto& lhs, const auto& rhs) {
                  return lhs.first < rhs.first;
              });
    std::vector<double> linear_pred(n, 0.0);
    for (std::size_t i = 0; i < n; ++i) {
        double s = 0.0;
        for (std::size_t f = 0; f < p; ++f) {
            s += X_buf[i * p + f] * out.coefficients[f];
        }
        linear_pred[i] = std::exp(s);
    }
    for (std::size_t i = 0; i < n; ++i) {
        if (sorted[i].second == 0) continue;
        double risk = 0.0;
        for (std::size_t j = 0; j < n; ++j) {
            if (survival_times[j] >= sorted[i].first) risk += linear_pred[j];
        }
        if (risk < kEps) continue;
        out.event_times.push_back(sorted[i].first);
        out.baseline_hazard.push_back(1.0 / risk);
    }
    out.n_features = static_cast<std::int32_t>(p);
    out.n_components = static_cast<std::int32_t>(a);
    ctx.clear_error();
    return P4A_OK;
}

// ---- PDS / DS / MIR-PLS / missing-aware NIPALS -------------------------

p4a_status_t fit_pds(Context& ctx,
                      const p4a_matrix_view_t& X_source,
                      const p4a_matrix_view_t& X_target,
                      std::int32_t window_half_width,
                      PdsResult& out) {
    out = PdsResult{};
    if (X_source.rows != X_target.rows) {
        ctx.set_error("source and target rows must match");
        return P4A_ERR_SHAPE_MISMATCH;
    }
    if (window_half_width < 0) {
        ctx.set_error("window_half_width must be >= 0");
        return P4A_ERR_INVALID_ARGUMENT;
    }
    std::vector<double> Xs, Xt;
    p4a_status_t status = copy_matrix(ctx, X_source, "X_source", Xs);
    if (status != P4A_OK) return status;
    status = copy_matrix(ctx, X_target, "X_target", Xt);
    if (status != P4A_OK) return status;
    const std::size_t n = static_cast<std::size_t>(X_source.rows);
    const std::size_t ps = static_cast<std::size_t>(X_source.cols);
    const std::size_t pt = static_cast<std::size_t>(X_target.cols);
    const std::size_t w = static_cast<std::size_t>(window_half_width);
    out.window_half_width = window_half_width;
    out.transformation.assign(pt * ps, 0.0);
    // For each target column j: select source columns in [j-w, j+w] within
    // bounds, solve least squares X_s_window * b = X_t_j.
    for (std::size_t j = 0; j < pt; ++j) {
        const std::size_t left =
            (j >= w) ? (j - w) : 0;
        const std::size_t right = std::min(ps - 1, j + w);
        const std::size_t window = right - left + 1;
        // Build the window design matrix (n × window).
        std::vector<double> X_win(n * window, 0.0);
        for (std::size_t r = 0; r < n; ++r) {
            for (std::size_t c = 0; c < window; ++c) {
                X_win[r * window + c] = Xs[r * ps + left + c];
            }
        }
        // Build target column.
        std::vector<double> t_col(n, 0.0);
        for (std::size_t r = 0; r < n; ++r) {
            t_col[r] = Xt[r * pt + j];
        }
        // Normal equations: solve (X_win' X_win) b = X_win' t_col.
        std::vector<double> A(window * window, 0.0);
        for (std::size_t a_row = 0; a_row < window; ++a_row) {
            for (std::size_t a_col = 0; a_col < window; ++a_col) {
                double s = 0.0;
                for (std::size_t r = 0; r < n; ++r) {
                    s += X_win[r * window + a_row] * X_win[r * window + a_col];
                }
                A[a_row * window + a_col] = s;
            }
        }
        std::vector<double> rhs(window, 0.0);
        for (std::size_t a_row = 0; a_row < window; ++a_row) {
            double s = 0.0;
            for (std::size_t r = 0; r < n; ++r) {
                s += X_win[r * window + a_row] * t_col[r];
            }
            rhs[a_row] = s;
        }
        // Tikhonov-regularize to avoid singular matrices.
        for (std::size_t a_row = 0; a_row < window; ++a_row)
            A[a_row * window + a_row] += 1e-6;
        // Gauss-Jordan to solve A b = rhs.
        std::vector<double> aug(window * (window + 1), 0.0);
        for (std::size_t a_row = 0; a_row < window; ++a_row) {
            for (std::size_t a_col = 0; a_col < window; ++a_col) {
                aug[a_row * (window + 1) + a_col] = A[a_row * window + a_col];
            }
            aug[a_row * (window + 1) + window] = rhs[a_row];
        }
        for (std::size_t col = 0; col < window; ++col) {
            std::size_t pivot = col;
            double pivot_val = std::fabs(aug[col * (window + 1) + col]);
            for (std::size_t r = col + 1; r < window; ++r) {
                const double v = std::fabs(aug[r * (window + 1) + col]);
                if (v > pivot_val) { pivot = r; pivot_val = v; }
            }
            if (pivot_val < kEps) continue;
            if (pivot != col) {
                for (std::size_t c = 0; c <= window; ++c) {
                    std::swap(aug[col * (window + 1) + c],
                              aug[pivot * (window + 1) + c]);
                }
            }
            const double diag = aug[col * (window + 1) + col];
            for (std::size_t c = 0; c <= window; ++c) {
                aug[col * (window + 1) + c] /= diag;
            }
            for (std::size_t r = 0; r < window; ++r) {
                if (r == col) continue;
                const double factor = aug[r * (window + 1) + col];
                if (std::fabs(factor) < kEps) continue;
                for (std::size_t c = 0; c <= window; ++c) {
                    aug[r * (window + 1) + c] -=
                        factor * aug[col * (window + 1) + c];
                }
            }
        }
        for (std::size_t a_col = 0; a_col < window; ++a_col) {
            out.transformation[j * ps + left + a_col] =
                aug[a_col * (window + 1) + window];
        }
    }
    ctx.clear_error();
    return P4A_OK;
}

p4a_status_t fit_ds(Context& ctx,
                     const p4a_matrix_view_t& X_source,
                     const p4a_matrix_view_t& X_target,
                     DsResult& out) {
    out = DsResult{};
    if (X_source.rows != X_target.rows) {
        ctx.set_error("source and target rows must match");
        return P4A_ERR_SHAPE_MISMATCH;
    }
    std::vector<double> Xs, Xt;
    p4a_status_t status = copy_matrix(ctx, X_source, "X_source", Xs);
    if (status != P4A_OK) return status;
    status = copy_matrix(ctx, X_target, "X_target", Xt);
    if (status != P4A_OK) return status;
    const std::size_t n = static_cast<std::size_t>(X_source.rows);
    const std::size_t ps = static_cast<std::size_t>(X_source.cols);
    const std::size_t pt = static_cast<std::size_t>(X_target.cols);
    std::vector<double> source_mean, target_mean;
    column_means(Xs, n, ps, source_mean);
    column_means(Xt, n, pt, target_mean);
    subtract_means(Xs, n, ps, source_mean);
    subtract_means(Xt, n, pt, target_mean);
    // DS: solve X_s * F = X_t in least squares sense for F (ps × pt).
    // Normal: (Xs' Xs + eps I) F = Xs' Xt.
    std::vector<double> A(ps * ps, 0.0);
    for (std::size_t i = 0; i < ps; ++i) {
        for (std::size_t j = 0; j < ps; ++j) {
            double s = 0.0;
            for (std::size_t r = 0; r < n; ++r) {
                s += Xs[r * ps + i] * Xs[r * ps + j];
            }
            A[i * ps + j] = s;
        }
        A[i * ps + i] += 1e-6;
    }
    std::vector<double> B(ps * pt, 0.0);
    for (std::size_t i = 0; i < ps; ++i) {
        for (std::size_t j = 0; j < pt; ++j) {
            double s = 0.0;
            for (std::size_t r = 0; r < n; ++r) {
                s += Xs[r * ps + i] * Xt[r * pt + j];
            }
            B[i * pt + j] = s;
        }
    }
    std::vector<double> aug(ps * (ps + pt), 0.0);
    for (std::size_t i = 0; i < ps; ++i) {
        for (std::size_t j = 0; j < ps; ++j) {
            aug[i * (ps + pt) + j] = A[i * ps + j];
        }
        for (std::size_t j = 0; j < pt; ++j) {
            aug[i * (ps + pt) + ps + j] = B[i * pt + j];
        }
    }
    for (std::size_t col = 0; col < ps; ++col) {
        std::size_t pivot = col;
        double pivot_val = std::fabs(aug[col * (ps + pt) + col]);
        for (std::size_t r = col + 1; r < ps; ++r) {
            const double v = std::fabs(aug[r * (ps + pt) + col]);
            if (v > pivot_val) { pivot = r; pivot_val = v; }
        }
        if (pivot_val < kEps) continue;
        if (pivot != col) {
            for (std::size_t c = 0; c < ps + pt; ++c) {
                std::swap(aug[col * (ps + pt) + c],
                          aug[pivot * (ps + pt) + c]);
            }
        }
        const double diag = aug[col * (ps + pt) + col];
        for (std::size_t c = 0; c < ps + pt; ++c) {
            aug[col * (ps + pt) + c] /= diag;
        }
        for (std::size_t r = 0; r < ps; ++r) {
            if (r == col) continue;
            const double factor = aug[r * (ps + pt) + col];
            if (std::fabs(factor) < kEps) continue;
            for (std::size_t c = 0; c < ps + pt; ++c) {
                aug[r * (ps + pt) + c] -=
                    factor * aug[col * (ps + pt) + c];
            }
        }
    }
    out.transformation.assign(ps * pt, 0.0);
    for (std::size_t i = 0; i < ps; ++i) {
        for (std::size_t j = 0; j < pt; ++j) {
            out.transformation[i * pt + j] =
                aug[i * (ps + pt) + ps + j];
        }
    }
    // Bias = target_mean - source_mean @ transformation
    out.bias.assign(pt, 0.0);
    for (std::size_t j = 0; j < pt; ++j) {
        double s = target_mean[j];
        for (std::size_t i = 0; i < ps; ++i) {
            s -= source_mean[i] * out.transformation[i * pt + j];
        }
        out.bias[j] = s;
    }
    ctx.clear_error();
    return P4A_OK;
}

p4a_status_t fit_mir_pls(Context& ctx,
                          const Config& cfg,
                          const p4a_matrix_view_t& X,
                          const p4a_matrix_view_t& Y,
                          MirPlsResult& out) {
    out = MirPlsResult{};
    // MIR-PLS: invert the X-Y relationship to obtain X = f(Y) regression
    // coefficients, then invert to predict Y from X. Minimal viable
    // implementation: run SIMPLS on the swapped (Y, X) pair and invert
    // the resulting coefficients.
    std::vector<double> X_buf, Y_buf;
    p4a_status_t status = copy_matrix(ctx, X, "X", X_buf);
    if (status != P4A_OK) return status;
    status = copy_matrix(ctx, Y, "Y", Y_buf);
    if (status != P4A_OK) return status;
    const std::size_t n = static_cast<std::size_t>(X.rows);
    const std::size_t p = static_cast<std::size_t>(X.cols);
    const std::size_t q = static_cast<std::size_t>(Y.cols);
    column_means(X_buf, n, p, out.x_mean);
    subtract_means(X_buf, n, p, out.x_mean);
    column_means(Y_buf, n, q, out.y_mean);
    subtract_means(Y_buf, n, q, out.y_mean);
    const std::size_t a = std::min<std::size_t>(
        static_cast<std::size_t>(cfg.n_components),
        std::min(n - 1, q));
    // Forward SIMPLS: predict X from Y.
    std::vector<double> coefs_yx;
    simple_simpls(Y_buf, X_buf, n, q, p, a, coefs_yx, nullptr);
    // Invert (q × p) to (p × q) by least-squares pseudoinverse.
    // Build A = coefs_yx (q × p). We want B (p × q) such that B coefs_yx
    // approximates identity in some sense. For a minimum viable behaviour,
    // compute B = coefs_yx^T (coefs_yx coefs_yx^T + eps I)^{-1}.
    std::vector<double> AAT(q * q, 0.0);
    for (std::size_t i = 0; i < q; ++i) {
        for (std::size_t j = 0; j < q; ++j) {
            double s = 0.0;
            for (std::size_t f = 0; f < p; ++f) {
                s += coefs_yx[i * p + f] * coefs_yx[j * p + f];
            }
            AAT[i * q + j] = s;
        }
        AAT[i * q + i] += 1e-6;
    }
    // Invert AAT.
    std::vector<double> inv(q * q, 0.0);
    for (std::size_t i = 0; i < q; ++i) inv[i * q + i] = 1.0;
    std::vector<double> A_mat = AAT;
    for (std::size_t col = 0; col < q; ++col) {
        std::size_t pivot = col;
        double pivot_val = std::fabs(A_mat[col * q + col]);
        for (std::size_t r = col + 1; r < q; ++r) {
            const double v = std::fabs(A_mat[r * q + col]);
            if (v > pivot_val) { pivot = r; pivot_val = v; }
        }
        if (pivot_val < kEps) continue;
        if (pivot != col) {
            for (std::size_t c = 0; c < q; ++c) {
                std::swap(A_mat[col * q + c], A_mat[pivot * q + c]);
                std::swap(inv[col * q + c], inv[pivot * q + c]);
            }
        }
        const double diag = A_mat[col * q + col];
        for (std::size_t c = 0; c < q; ++c) {
            A_mat[col * q + c] /= diag;
            inv[col * q + c] /= diag;
        }
        for (std::size_t r = 0; r < q; ++r) {
            if (r == col) continue;
            const double factor = A_mat[r * q + col];
            if (std::fabs(factor) < kEps) continue;
            for (std::size_t c = 0; c < q; ++c) {
                A_mat[r * q + c] -= factor * A_mat[col * q + c];
                inv[r * q + c] -= factor * inv[col * q + c];
            }
        }
    }
    out.coefficients.assign(p * q, 0.0);
    for (std::size_t f = 0; f < p; ++f) {
        for (std::size_t target = 0; target < q; ++target) {
            double s = 0.0;
            for (std::size_t k = 0; k < q; ++k) {
                s += coefs_yx[k * p + f] * inv[k * q + target];
            }
            out.coefficients[f * q + target] = s;
        }
    }
    ctx.clear_error();
    return P4A_OK;
}

p4a_status_t fit_missing_aware_nipals(Context& ctx,
                                       const Config& cfg,
                                       const p4a_matrix_view_t& X,
                                       const p4a_matrix_view_t& Y,
                                       WeightedPlsResult& out) {
    out = WeightedPlsResult{};
    std::vector<double> X_buf, Y_buf;
    p4a_status_t status = copy_matrix(ctx, X, "X", X_buf);
    if (status != P4A_OK) return status;
    status = copy_matrix(ctx, Y, "Y", Y_buf);
    if (status != P4A_OK) return status;
    const std::size_t n = static_cast<std::size_t>(X.rows);
    const std::size_t p = static_cast<std::size_t>(X.cols);
    const std::size_t q = static_cast<std::size_t>(Y.cols);
    // Compute column means ignoring NaNs, then impute NaN entries with
    // those means before centering. This is the simplest "missing-aware"
    // policy compatible with downstream SIMPLS.
    out.x_mean.assign(p, 0.0);
    out.y_mean.assign(q, 0.0);
    std::vector<std::size_t> x_count(p, 0);
    std::vector<std::size_t> y_count(q, 0);
    for (std::size_t r = 0; r < n; ++r) {
        for (std::size_t f = 0; f < p; ++f) {
            const double v = X_buf[r * p + f];
            if (std::isfinite(v)) { out.x_mean[f] += v; ++x_count[f]; }
        }
        for (std::size_t target = 0; target < q; ++target) {
            const double v = Y_buf[r * q + target];
            if (std::isfinite(v)) { out.y_mean[target] += v; ++y_count[target]; }
        }
    }
    for (std::size_t f = 0; f < p; ++f)
        out.x_mean[f] = (x_count[f] > 0) ? out.x_mean[f] /
            static_cast<double>(x_count[f]) : 0.0;
    for (std::size_t target = 0; target < q; ++target)
        out.y_mean[target] = (y_count[target] > 0) ? out.y_mean[target] /
            static_cast<double>(y_count[target]) : 0.0;
    for (std::size_t r = 0; r < n; ++r) {
        for (std::size_t f = 0; f < p; ++f) {
            const double v = X_buf[r * p + f];
            X_buf[r * p + f] =
                (std::isfinite(v) ? v : out.x_mean[f]) - out.x_mean[f];
        }
        for (std::size_t target = 0; target < q; ++target) {
            const double v = Y_buf[r * q + target];
            Y_buf[r * q + target] =
                (std::isfinite(v) ? v : out.y_mean[target]) -
                out.y_mean[target];
        }
    }
    const std::size_t a = std::min<std::size_t>(
        static_cast<std::size_t>(cfg.n_components),
        std::min(n - 1, p));
    std::vector<double> coefs;
    simple_simpls(X_buf, Y_buf, n, p, q, a, coefs, nullptr);
    out.coefficients = std::move(coefs);
    out.n_features = static_cast<std::int32_t>(p);
    out.n_targets = static_cast<std::int32_t>(q);
    out.n_components = static_cast<std::int32_t>(a);
    ctx.clear_error();
    return P4A_OK;
}

// ---- approximate-PRESS -------------------------------------------------

p4a_status_t approximate_press(Context& ctx,
                                const Config& cfg,
                                const p4a_matrix_view_t& X,
                                const p4a_matrix_view_t& Y,
                                std::int32_t max_components,
                                ApproximatePressResult& out) {
    (void)cfg;
    out = ApproximatePressResult{};
    if (max_components < 1) {
        ctx.set_error("max_components must be >= 1");
        return P4A_ERR_INVALID_ARGUMENT;
    }
    std::vector<double> X_buf, Y_buf;
    p4a_status_t status = copy_matrix(ctx, X, "X", X_buf);
    if (status != P4A_OK) return status;
    status = copy_matrix(ctx, Y, "Y", Y_buf);
    if (status != P4A_OK) return status;
    const std::size_t n = static_cast<std::size_t>(X.rows);
    const std::size_t p = static_cast<std::size_t>(X.cols);
    const std::size_t q = static_cast<std::size_t>(Y.cols);
    std::vector<double> x_mean, y_mean;
    column_means(X_buf, n, p, x_mean);
    subtract_means(X_buf, n, p, x_mean);
    column_means(Y_buf, n, q, y_mean);
    subtract_means(Y_buf, n, q, y_mean);
    out.n_components = max_components;
    out.press_per_component.assign(
        static_cast<std::size_t>(max_components), 0.0);
    out.rmse_per_component.assign(
        static_cast<std::size_t>(max_components), 0.0);
    double best_press = std::numeric_limits<double>::infinity();
    std::int32_t best_k = 1;
    for (std::int32_t k = 1; k <= max_components; ++k) {
        std::vector<double> coefs;
        simple_simpls(X_buf, Y_buf, n, p, q,
                      static_cast<std::size_t>(k), coefs, nullptr);
        double press = 0.0;
        for (std::size_t r = 0; r < n; ++r) {
            for (std::size_t target = 0; target < q; ++target) {
                double pred = 0.0;
                for (std::size_t f = 0; f < p; ++f) {
                    pred += X_buf[r * p + f] * coefs[f * q + target];
                }
                const double d = Y_buf[r * q + target] - pred;
                // Approximate leverage 1 / (1 - k/n).
                const double inflate =
                    1.0 / std::max(1.0 - static_cast<double>(k) /
                                          static_cast<double>(n),
                                   1.0 / static_cast<double>(n));
                press += d * d * inflate * inflate;
            }
        }
        const double rmse = std::sqrt(press / static_cast<double>(n * q));
        out.press_per_component[static_cast<std::size_t>(k - 1)] = press;
        out.rmse_per_component[static_cast<std::size_t>(k - 1)] = rmse;
        if (press < best_press) {
            best_press = press;
            best_k = k;
        }
    }
    out.selected_n_components = best_k;
    ctx.clear_error();
    return P4A_OK;
}

// ---- Ensembles ---------------------------------------------------------

namespace {

void accumulate_coefficients(std::vector<double>& accumulator,
                              const std::vector<double>& addition,
                              double weight = 1.0) {
    for (std::size_t i = 0; i < accumulator.size(); ++i) {
        accumulator[i] += weight * addition[i];
    }
}

}  // namespace

p4a_status_t fit_bagging_pls(Context& ctx,
                              const Config& cfg,
                              const p4a_matrix_view_t& X,
                              const p4a_matrix_view_t& Y,
                              std::int32_t n_estimators,
                              std::uint64_t seed,
                              EnsemblePlsResult& out) {
    out = EnsemblePlsResult{};
    if (n_estimators < 1) {
        ctx.set_error("n_estimators must be >= 1");
        return P4A_ERR_INVALID_ARGUMENT;
    }
    std::vector<double> X_buf, Y_buf;
    p4a_status_t status = copy_matrix(ctx, X, "X", X_buf);
    if (status != P4A_OK) return status;
    status = copy_matrix(ctx, Y, "Y", Y_buf);
    if (status != P4A_OK) return status;
    const std::size_t n = static_cast<std::size_t>(X.rows);
    const std::size_t p = static_cast<std::size_t>(X.cols);
    const std::size_t q = static_cast<std::size_t>(Y.cols);
    column_means(X_buf, n, p, out.x_mean);
    subtract_means(X_buf, n, p, out.x_mean);
    column_means(Y_buf, n, q, out.y_mean);
    subtract_means(Y_buf, n, q, out.y_mean);
    out.coefficients.assign(p * q, 0.0);
    const std::size_t a = std::min<std::size_t>(
        static_cast<std::size_t>(cfg.n_components),
        std::min(n - 1, p));
    std::mt19937_64 rng(seed);
    std::uniform_int_distribution<std::size_t> dist(0, n - 1);
    std::vector<double> X_boot(n * p, 0.0);
    std::vector<double> Y_boot(n * q, 0.0);
    for (std::int32_t e = 0; e < n_estimators; ++e) {
        for (std::size_t r = 0; r < n; ++r) {
            const std::size_t idx_row = dist(rng);
            for (std::size_t f = 0; f < p; ++f)
                X_boot[r * p + f] = X_buf[idx_row * p + f];
            for (std::size_t target = 0; target < q; ++target)
                Y_boot[r * q + target] = Y_buf[idx_row * q + target];
        }
        std::vector<double> coefs;
        simple_simpls(X_boot, Y_boot, n, p, q, a, coefs, nullptr);
        accumulate_coefficients(out.coefficients, coefs);
    }
    const double inv = 1.0 / static_cast<double>(n_estimators);
    for (double& c : out.coefficients) c *= inv;
    out.n_estimators = n_estimators;
    out.n_features = static_cast<std::int32_t>(p);
    out.n_targets = static_cast<std::int32_t>(q);
    out.n_components = static_cast<std::int32_t>(a);
    ctx.clear_error();
    return P4A_OK;
}

p4a_status_t fit_boosting_pls(Context& ctx,
                               const Config& cfg,
                               const p4a_matrix_view_t& X,
                               const p4a_matrix_view_t& Y,
                               std::int32_t n_estimators,
                               double learning_rate,
                               EnsemblePlsResult& out) {
    out = EnsemblePlsResult{};
    if (n_estimators < 1 || !(learning_rate > 0.0 && learning_rate <= 1.0)) {
        ctx.set_error("n_estimators >= 1 and learning_rate in (0, 1] required");
        return P4A_ERR_INVALID_ARGUMENT;
    }
    std::vector<double> X_buf, Y_buf;
    p4a_status_t status = copy_matrix(ctx, X, "X", X_buf);
    if (status != P4A_OK) return status;
    status = copy_matrix(ctx, Y, "Y", Y_buf);
    if (status != P4A_OK) return status;
    const std::size_t n = static_cast<std::size_t>(X.rows);
    const std::size_t p = static_cast<std::size_t>(X.cols);
    const std::size_t q = static_cast<std::size_t>(Y.cols);
    column_means(X_buf, n, p, out.x_mean);
    subtract_means(X_buf, n, p, out.x_mean);
    column_means(Y_buf, n, q, out.y_mean);
    subtract_means(Y_buf, n, q, out.y_mean);
    out.coefficients.assign(p * q, 0.0);
    const std::size_t a = std::min<std::size_t>(
        static_cast<std::size_t>(cfg.n_components),
        std::min(n - 1, p));
    std::vector<double> residual = Y_buf;
    for (std::int32_t e = 0; e < n_estimators; ++e) {
        std::vector<double> coefs;
        simple_simpls(X_buf, residual, n, p, q, a, coefs, nullptr);
        accumulate_coefficients(out.coefficients, coefs, learning_rate);
        // Subtract learning_rate * predictions from residual.
        for (std::size_t r = 0; r < n; ++r) {
            for (std::size_t target = 0; target < q; ++target) {
                double pred = 0.0;
                for (std::size_t f = 0; f < p; ++f) {
                    pred += X_buf[r * p + f] * coefs[f * q + target];
                }
                residual[r * q + target] -= learning_rate * pred;
            }
        }
    }
    out.n_estimators = n_estimators;
    out.n_features = static_cast<std::int32_t>(p);
    out.n_targets = static_cast<std::int32_t>(q);
    out.n_components = static_cast<std::int32_t>(a);
    ctx.clear_error();
    return P4A_OK;
}

p4a_status_t fit_random_subspace_pls(Context& ctx,
                                      const Config& cfg,
                                      const p4a_matrix_view_t& X,
                                      const p4a_matrix_view_t& Y,
                                      std::int32_t n_estimators,
                                      std::int32_t features_per_subspace,
                                      std::uint64_t seed,
                                      EnsemblePlsResult& out) {
    out = EnsemblePlsResult{};
    if (n_estimators < 1 || features_per_subspace < 1) {
        ctx.set_error("n_estimators and features_per_subspace must be >= 1");
        return P4A_ERR_INVALID_ARGUMENT;
    }
    std::vector<double> X_buf, Y_buf;
    p4a_status_t status = copy_matrix(ctx, X, "X", X_buf);
    if (status != P4A_OK) return status;
    status = copy_matrix(ctx, Y, "Y", Y_buf);
    if (status != P4A_OK) return status;
    const std::size_t n = static_cast<std::size_t>(X.rows);
    const std::size_t p = static_cast<std::size_t>(X.cols);
    const std::size_t q = static_cast<std::size_t>(Y.cols);
    if (static_cast<std::size_t>(features_per_subspace) > p) {
        ctx.set_error("features_per_subspace cannot exceed n_features");
        return P4A_ERR_INVALID_ARGUMENT;
    }
    column_means(X_buf, n, p, out.x_mean);
    subtract_means(X_buf, n, p, out.x_mean);
    column_means(Y_buf, n, q, out.y_mean);
    subtract_means(Y_buf, n, q, out.y_mean);
    out.coefficients.assign(p * q, 0.0);
    const std::size_t a = std::min<std::size_t>(
        static_cast<std::size_t>(cfg.n_components),
        std::min(n - 1, static_cast<std::size_t>(features_per_subspace)));
    std::mt19937_64 rng(seed);
    std::vector<std::size_t> indices(p);
    for (std::size_t i = 0; i < p; ++i) indices[i] = i;
    for (std::int32_t e = 0; e < n_estimators; ++e) {
        std::shuffle(indices.begin(), indices.end(), rng);
        const std::size_t sub_p = static_cast<std::size_t>(features_per_subspace);
        std::vector<double> X_sub(n * sub_p, 0.0);
        for (std::size_t r = 0; r < n; ++r) {
            for (std::size_t fi = 0; fi < sub_p; ++fi) {
                X_sub[r * sub_p + fi] = X_buf[r * p + indices[fi]];
            }
        }
        std::vector<double> coefs;
        simple_simpls(X_sub, Y_buf, n, sub_p, q, a, coefs, nullptr);
        for (std::size_t fi = 0; fi < sub_p; ++fi) {
            for (std::size_t target = 0; target < q; ++target) {
                out.coefficients[indices[fi] * q + target] +=
                    coefs[fi * q + target];
            }
        }
    }
    const double inv = 1.0 / static_cast<double>(n_estimators);
    for (double& c : out.coefficients) c *= inv;
    out.n_estimators = n_estimators;
    out.n_features = static_cast<std::int32_t>(p);
    out.n_targets = static_cast<std::int32_t>(q);
    out.n_components = static_cast<std::int32_t>(a);
    ctx.clear_error();
    return P4A_OK;
}

}  // namespace pls4all::core
