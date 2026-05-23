// SPDX-License-Identifier: CECILL-2.1

#include "core/kernel_pls.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <new>
#include <vector>

#include "core/common/linalg.hpp"
#include "core/common/matrix_view.hpp"

namespace n4m::core {

namespace {

constexpr double kEps = 1e-12;

[[nodiscard]] n4m_status_t copy_matrix(Context& ctx,
                                        const n4m_matrix_view_t& V,
                                        const char* name,
                                        std::vector<double>& out) {
    if (validate_nonnull_view(V) != N4M_OK || V.dtype != N4M_DTYPE_F64) {
        ctx.set_errorf("%s must be a finite F64 row-major matrix", name);
        return N4M_ERR_INVALID_ARGUMENT;
    }
    const std::size_t rows = static_cast<std::size_t>(V.rows);
    const std::size_t cols = static_cast<std::size_t>(V.cols);
    out.assign(rows * cols, 0.0);
    const auto* data = static_cast<const double*>(V.data);
    if (data == nullptr) {
        if (rows > 0 && cols > 0) {
            ctx.set_errorf("%s has null data", name);
            return N4M_ERR_NULL_POINTER;
        }
        return N4M_OK;
    }
    const std::size_t rs = static_cast<std::size_t>(V.row_stride);
    const std::size_t cs = static_cast<std::size_t>(V.col_stride);
    for (std::size_t r = 0; r < rows; ++r) {
        for (std::size_t c = 0; c < cols; ++c) {
            const double v = data[r * rs + c * cs];
            if (!std::isfinite(v)) {
                ctx.set_errorf("%s contains NaN or Inf", name);
                return N4M_ERR_INVALID_ARGUMENT;
            }
            out[r * cols + c] = v;
        }
    }
    return N4M_OK;
}

double kernel_value(KernelType kernel, double gamma, double coef0,
                    std::int32_t degree,
                    const double* a, const double* b, std::size_t p) {
    if (kernel == KernelType::LINEAR) {
        double s = 0.0;
        for (std::size_t i = 0; i < p; ++i) s += a[i] * b[i];
        return s;
    }
    if (kernel == KernelType::RBF) {
        double sumsq = 0.0;
        for (std::size_t i = 0; i < p; ++i) {
            const double d = a[i] - b[i];
            sumsq += d * d;
        }
        return std::exp(-gamma * sumsq);
    }
    if (kernel == KernelType::POLYNOMIAL) {
        double dot_val = 0.0;
        for (std::size_t i = 0; i < p; ++i) dot_val += a[i] * b[i];
        return std::pow(gamma * dot_val + coef0,
                        static_cast<double>(degree));
    }
    // SIGMOID
    double dot_val = 0.0;
    for (std::size_t i = 0; i < p; ++i) dot_val += a[i] * b[i];
    return std::tanh(gamma * dot_val + coef0);
}

void compute_gram(KernelType kernel, double gamma, double coef0,
                  std::int32_t degree,
                  const std::vector<double>& X1, std::size_t n1,
                  const std::vector<double>& X2, std::size_t n2,
                  std::size_t p, std::vector<double>& K) {
    K.assign(n1 * n2, 0.0);
    for (std::size_t i = 0; i < n1; ++i) {
        for (std::size_t j = 0; j < n2; ++j) {
            K[i * n2 + j] = kernel_value(kernel, gamma, coef0, degree,
                                         X1.data() + i * p,
                                         X2.data() + j * p, p);
        }
    }
}

double squared_norm(const std::vector<double>& v) {
    double s = 0.0;
    for (double x : v) s += x * x;
    return s;
}

// SIMPLS regression on the centered Gram matrix `K_c` (n x n, symmetric,
// row & column means already zero) against centered target `Y_c` (n x q).
//
// Matches `pls::plsr(Y ~ K_c, ncomp=a, method="simpls", scale=FALSE)` from
// the R `pls` package — the canonical Kernel PLS reference of Rosipal & Trejo
// (2001) expressed via the SIMPLS deflation order. The R `plsr` default
// (`kernelpls` = Dayal & MacGregor 1997) is numerically equivalent to SIMPLS
// on the same input (verified to ~1e-15 in R), so this single SIMPLS-on-K_c
// path matches both reference algorithms.
//
// Returns regression coefficients B (n x q row-major). In-sample predictions
// follow `Yhat = K_c @ B + y_mean`; out-of-sample predictions use the
// centered test-vs-train Gram matrix K_test_c.
//
// The implementation gracefully stops if the residual covariance, the X
// score, or the deflation basis collapses to (near) zero — relevant when
// K_c is rank-deficient (e.g. when the RBF kernel saturates to ~I for
// high-dimensional inputs).
[[nodiscard]] n4m_status_t simpls_kc_coefficients(
    Context& ctx,
    const std::vector<double>& K_c,   // n x n row-major
    const std::vector<double>& Y_c,   // n x q row-major (already y-centered)
    std::size_t n, std::size_t q,
    std::size_t a_requested,
    std::vector<double>& coefficients) {  // n x q row-major
    (void)ctx;  // currently no error-message channel used inside SIMPLS
    coefficients.assign(n * q, 0.0);
    if (a_requested == 0U) {
        return N4M_OK;
    }

    // S = K_c^T @ Y_c  (n x q). K_c is symmetric, so K_c^T == K_c. We use
    // gemm with Trans_Yes so the math reads identical to the canonical
    // SIMPLS in core/model.cpp::fit_pls_regression_simpls.
    std::vector<double> S(n * q, 0.0);
    n4m::linalg::gemm(n4m::linalg::Trans_Yes, n4m::linalg::Trans_No,
                       n, q, n,
                       1.0,
                       K_c.data(), n,
                       Y_c.data(), q,
                       0.0,
                       S.data(), q);

    // Working buffers for components actually retained.
    std::vector<double> R_mat(n * a_requested, 0.0);   // rotations (one column per comp)
    std::vector<double> Q_mat(q * a_requested, 0.0);   // y_loadings
    std::vector<double> V_mat(n * a_requested, 0.0);   // deflation basis

    const double tiny = std::numeric_limits<double>::epsilon();
    std::size_t a_used = 0;

    for (std::size_t comp = 0; comp < a_requested; ++comp) {
        // 1. Dominant left singular vector of S → `r` of length n.
        std::vector<double> r(n, 0.0);
        if (q == 1U) {
            // Single-target shortcut: r = S[:, 0] / ||S[:, 0]||.
            double rn = 0.0;
            for (std::size_t i = 0; i < n; ++i) {
                r[i] = S[i * q + 0];
                rn += r[i] * r[i];
            }
            if (rn <= tiny) break;  // residual covariance vanished
            const double inv = 1.0 / std::sqrt(rn);
            for (double& v : r) v *= inv;
        } else {
            // Multi-target: power-iteration on S^T S to extract the
            // dominant left singular direction. Match the convention of
            // core/model.cpp::dominant_left_singular_direction so signs and
            // iteration ordering line up with R's SVD-based simpls.fit.
            std::size_t best_target = 0;
            double best_norm = 0.0;
            for (std::size_t t = 0; t < q; ++t) {
                double nrm = 0.0;
                for (std::size_t i = 0; i < n; ++i) {
                    const double v = S[i * q + t];
                    nrm += v * v;
                }
                if (nrm > best_norm) {
                    best_norm = nrm;
                    best_target = t;
                }
            }
            if (best_norm <= tiny) break;
            const double inv = 1.0 / std::sqrt(best_norm);
            for (std::size_t i = 0; i < n; ++i) {
                r[i] = S[i * q + best_target] * inv;
            }
            std::vector<double> right(q, 0.0);
            std::vector<double> next_r(n, 0.0);
            constexpr int kMaxIter = 500;
            const double stop_tol = 1e-13;
            bool converged = false;
            for (int it = 0; it < kMaxIter; ++it) {
                // right = S^T r
                for (std::size_t t = 0; t < q; ++t) {
                    double s = 0.0;
                    for (std::size_t i = 0; i < n; ++i) {
                        s += S[i * q + t] * r[i];
                    }
                    right[t] = s;
                }
                const double rn = std::sqrt(squared_norm(right));
                if (rn <= tiny) { converged = false; break; }
                for (double& v : right) v /= rn;
                // next_r = S right
                for (std::size_t i = 0; i < n; ++i) {
                    double s = 0.0;
                    for (std::size_t t = 0; t < q; ++t) {
                        s += S[i * q + t] * right[t];
                    }
                    next_r[i] = s;
                }
                const double ln = std::sqrt(squared_norm(next_r));
                if (ln <= tiny) { converged = false; break; }
                for (double& v : next_r) v /= ln;
                double dotrr = 0.0;
                for (std::size_t i = 0; i < n; ++i) dotrr += next_r[i] * r[i];
                if (dotrr < 0.0) for (double& v : next_r) v = -v;
                double diff = 0.0;
                for (std::size_t i = 0; i < n; ++i) {
                    const double d = next_r[i] - r[i];
                    diff += d * d;
                }
                r = next_r;
                if (diff < stop_tol * stop_tol) { converged = true; break; }
            }
            if (!converged) break;
        }

        // 2. x_score t = K_c r ; t /= ||t|| ; r /= ||t||.
        std::vector<double> t(n, 0.0);
        n4m::linalg::gemv(n4m::linalg::Trans_No, n, n, 1.0,
                          K_c.data(), r.data(), 0.0, t.data());
        const double t_norm = std::sqrt(squared_norm(t));
        if (t_norm <= tiny) break;
        const double inv_t = 1.0 / t_norm;
        for (double& v : t) v *= inv_t;
        for (double& v : r) v *= inv_t;

        // 3. p_loading = K_c^T t  (length n) ; q_loading = Y_c^T t (length q).
        std::vector<double> p_load(n, 0.0);
        n4m::linalg::gemv(n4m::linalg::Trans_Yes, n, n, 1.0,
                          K_c.data(), t.data(), 0.0, p_load.data());
        std::vector<double> q_load(q, 0.0);
        n4m::linalg::gemv(n4m::linalg::Trans_Yes, n, q, 1.0,
                          Y_c.data(), t.data(), 0.0, q_load.data());

        // 4. Deflation basis v_k = orth(p_load against V_{1..k-1}).
        std::vector<double> v(p_load);
        for (std::size_t prev = 0; prev < comp; ++prev) {
            double dotp = 0.0;
            for (std::size_t i = 0; i < n; ++i) {
                dotp += V_mat[i * a_requested + prev] * v[i];
            }
            for (std::size_t i = 0; i < n; ++i) {
                v[i] -= V_mat[i * a_requested + prev] * dotp;
            }
        }
        const double v_norm = std::sqrt(squared_norm(v));
        if (v_norm <= tiny) break;
        const double inv_v = 1.0 / v_norm;
        for (double& x : v) x *= inv_v;

        // 5. Store r, q_load, v in the kth column.
        for (std::size_t i = 0; i < n; ++i) {
            R_mat[i * a_requested + comp] = r[i];
            V_mat[i * a_requested + comp] = v[i];
        }
        for (std::size_t target = 0; target < q; ++target) {
            Q_mat[target * a_requested + comp] = q_load[target];
        }

        // 6. Deflate S := S - v (v^T S).
        std::vector<double> vts(q, 0.0);
        for (std::size_t target = 0; target < q; ++target) {
            double s = 0.0;
            for (std::size_t i = 0; i < n; ++i) {
                s += v[i] * S[i * q + target];
            }
            vts[target] = s;
        }
        n4m::linalg::ger(n, q, -1.0, v.data(), vts.data(), S.data(), q);

        a_used = comp + 1;
    }

    if (a_used == 0U) {
        // Nothing useful extracted — coefficients stay zero, predictions
        // collapse to y_mean (the analytic best when no signal is found).
        return N4M_OK;
    }

    // 7. coefficients = R @ Q^T  with the columns 0..a_used-1.
    //    Express as a single GEMM by packing the active columns.
    std::vector<double> R_packed(n * a_used, 0.0);
    std::vector<double> Q_packed(q * a_used, 0.0);
    for (std::size_t i = 0; i < n; ++i) {
        for (std::size_t k = 0; k < a_used; ++k) {
            R_packed[i * a_used + k] = R_mat[i * a_requested + k];
        }
    }
    for (std::size_t target = 0; target < q; ++target) {
        for (std::size_t k = 0; k < a_used; ++k) {
            Q_packed[target * a_used + k] = Q_mat[target * a_requested + k];
        }
    }
    // B (n x q) = R_packed (n x a_used) * Q_packed^T (a_used x q).
    n4m::linalg::gemm(n4m::linalg::Trans_No, n4m::linalg::Trans_Yes,
                       n, q, a_used,
                       1.0,
                       R_packed.data(), a_used,
                       Q_packed.data(), a_used,
                       0.0,
                       coefficients.data(), q);
    return N4M_OK;
}

}  // namespace

n4m_status_t fit_kernel_pls(Context& ctx,
                             const Config& cfg,
                             KernelType kernel,
                             double gamma,
                             double coef0,
                             std::int32_t degree,
                             const n4m_matrix_view_t& X,
                             const n4m_matrix_view_t& Y,
                             KernelPlsResult& out) {
    out = KernelPlsResult{};
    if (X.rows != Y.rows) {
        ctx.set_error("X and Y must share row count");
        return N4M_ERR_SHAPE_MISMATCH;
    }
    if (cfg.n_components < 1) {
        ctx.set_error("n_components must be >= 1");
        return N4M_ERR_INVALID_ARGUMENT;
    }
    if (degree < 1) {
        ctx.set_error("polynomial degree must be >= 1");
        return N4M_ERR_INVALID_ARGUMENT;
    }

    std::vector<double> X_buf, Y_buf;
    n4m_status_t status = copy_matrix(ctx, X, "X", X_buf);
    if (status != N4M_OK) return status;
    status = copy_matrix(ctx, Y, "Y", Y_buf);
    if (status != N4M_OK) return status;

    const std::size_t n = static_cast<std::size_t>(X.rows);
    const std::size_t p = static_cast<std::size_t>(X.cols);
    const std::size_t q = static_cast<std::size_t>(Y.cols);
    if (n < 2 || p < 1 || q < 1) {
        ctx.set_error("kernel PLS requires at least 2 rows and 1 feature");
        return N4M_ERR_INVALID_ARGUMENT;
    }

    if (gamma <= 0.0) {
        gamma = 1.0 / static_cast<double>(p);
    }

    out.n_train = static_cast<std::int64_t>(n);
    out.n_features = static_cast<std::int32_t>(p);
    out.n_targets = static_cast<std::int32_t>(q);
    out.n_components = cfg.n_components;
    out.kernel_type = kernel;
    out.gamma = gamma;
    out.coef0 = coef0;
    out.degree = degree;
    out.x_train = X_buf;

    // Center Y.
    out.y_mean.assign(q, 0.0);
    if (cfg.center_y != 0) {
        for (std::size_t r = 0; r < n; ++r) {
            for (std::size_t target = 0; target < q; ++target) {
                out.y_mean[target] += Y_buf[r * q + target];
            }
        }
        for (std::size_t target = 0; target < q; ++target)
            out.y_mean[target] /= static_cast<double>(n);
        for (std::size_t r = 0; r < n; ++r) {
            for (std::size_t target = 0; target < q; ++target) {
                Y_buf[r * q + target] -= out.y_mean[target];
            }
        }
    }

    // Build Gram K_train.
    std::vector<double> K;
    compute_gram(kernel, gamma, coef0, degree, X_buf, n, X_buf, n, p, K);
    // Centering of K: K_c = K - 1_n K - K 1_n + 1_n K 1_n  (== H K H with
    // H = I - 11'/n). Row & column means of K_c are zero by construction.
    out.K_train_row_means.assign(n, 0.0);
    double K_global = 0.0;
    for (std::size_t i = 0; i < n; ++i) {
        for (std::size_t j = 0; j < n; ++j) {
            out.K_train_row_means[i] += K[i * n + j];
        }
        K_global += out.K_train_row_means[i];
        out.K_train_row_means[i] /= static_cast<double>(n);
    }
    out.K_train_global_mean = K_global / static_cast<double>(n * n);
    for (std::size_t i = 0; i < n; ++i) {
        for (std::size_t j = 0; j < n; ++j) {
            K[i * n + j] = K[i * n + j] - out.K_train_row_means[i] -
                            out.K_train_row_means[j] +
                            out.K_train_global_mean;
        }
    }

    // SIMPLS on the centered Gram matrix — canonical kernel PLS reference
    // matching R `pls::plsr(method="simpls"|"kernelpls", scale=FALSE)` on
    // `kernlab::kernelMatrix(...)`. The returned coefficients are the
    // dual-space regression weights `alpha` (n x q) such that
    //   y_hat = K_test_c @ alpha + y_mean.
    out.alpha.assign(n * q, 0.0);
    const std::size_t a_req = static_cast<std::size_t>(cfg.n_components);
    status = simpls_kc_coefficients(ctx, K, Y_buf, n, q, a_req, out.alpha);
    if (status != N4M_OK) return status;

    ctx.clear_error();
    return N4M_OK;
}

n4m_status_t predict_kernel_pls(Context& ctx,
                                 const KernelPlsResult& model,
                                 const n4m_matrix_view_t& X,
                                 std::vector<double>& out_predictions) {
    out_predictions.clear();
    if (model.x_train.empty()) {
        ctx.set_error("kernel PLS model is not fitted");
        return N4M_ERR_NOT_FITTED;
    }
    if (X.cols != model.n_features) {
        ctx.set_error("X cols must match training n_features");
        return N4M_ERR_SHAPE_MISMATCH;
    }
    std::vector<double> X_buf;
    n4m_status_t status = copy_matrix(ctx, X, "X", X_buf);
    if (status != N4M_OK) return status;
    const std::size_t n_test = static_cast<std::size_t>(X.rows);
    const std::size_t n_train = static_cast<std::size_t>(model.n_train);
    const std::size_t p = static_cast<std::size_t>(model.n_features);
    const std::size_t q = static_cast<std::size_t>(model.n_targets);

    std::vector<double> K_test;
    compute_gram(model.kernel_type, model.gamma, model.coef0, model.degree,
                 X_buf, n_test, model.x_train, n_train, p, K_test);
    // Center K_test using training means:
    //   K_c[i,j] = K_test[i,j] - row_mean_test[i] - col_mean_train[j] +
    //              global_mean_train
    // where row_mean_test[i] = mean_j K_test[i,j]
    std::vector<double> row_mean_test(n_test, 0.0);
    for (std::size_t i = 0; i < n_test; ++i) {
        double s = 0.0;
        for (std::size_t j = 0; j < n_train; ++j) s += K_test[i * n_train + j];
        row_mean_test[i] = s / static_cast<double>(n_train);
    }
    for (std::size_t i = 0; i < n_test; ++i) {
        for (std::size_t j = 0; j < n_train; ++j) {
            K_test[i * n_train + j] =
                K_test[i * n_train + j] - row_mean_test[i] -
                model.K_train_row_means[j] + model.K_train_global_mean;
        }
    }
    // Predictions = K_test_c @ alpha + y_mean.
    out_predictions.assign(n_test * q, 0.0);
    for (std::size_t i = 0; i < n_test; ++i) {
        for (std::size_t target = 0; target < q; ++target) {
            double s = model.y_mean[target];
            for (std::size_t j = 0; j < n_train; ++j) {
                s += K_test[i * n_train + j] * model.alpha[j * q + target];
            }
            out_predictions[i * q + target] = s;
        }
    }
    ctx.clear_error();
    return N4M_OK;
}

}  // namespace n4m::core
