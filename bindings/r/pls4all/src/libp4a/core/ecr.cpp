// SPDX-License-Identifier: CECILL-2.1
//
// Elastic Component Regression — closely mirrors libPLS 1.95 `ecr.m`.

#include "core/ecr.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <new>
#include <vector>

#include "core/matrix_view.hpp"

namespace n4m::core {

namespace {

constexpr double kEps = 1e-12;
constexpr std::int32_t kPowerMaxIter = 500;
constexpr double kPowerTol = 1e-9;

[[nodiscard]] n4m_status_t copy_matrix(Context& ctx,
                                        const n4m_matrix_view_t& V,
                                        const char* name,
                                        std::vector<double>& out) {
    if (validate_nonnull_view(V) != N4M_OK ||
        (V.dtype != N4M_DTYPE_F64 && V.dtype != N4M_DTYPE_F32)) {
        ctx.set_errorf("%s must be a finite row-major matrix", name);
        return N4M_ERR_INVALID_ARGUMENT;
    }
    const auto rows = static_cast<std::size_t>(V.rows);
    const auto cols = static_cast<std::size_t>(V.cols);
    out.assign(rows * cols, 0.0);
    if (rows == 0 || cols == 0) {
        return N4M_OK;
    }
    const auto rs = static_cast<std::size_t>(V.row_stride);
    const auto cs = static_cast<std::size_t>(V.col_stride);
    if (V.dtype == N4M_DTYPE_F64) {
        const auto* data = static_cast<const double*>(V.data);
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
    } else {
        const auto* data = static_cast<const float*>(V.data);
        for (std::size_t r = 0; r < rows; ++r) {
            for (std::size_t c = 0; c < cols; ++c) {
                const double v = static_cast<double>(data[r * rs + c * cs]);
                if (!std::isfinite(v)) {
                    ctx.set_errorf("%s contains NaN or Inf", name);
                    return N4M_ERR_INVALID_ARGUMENT;
                }
                out[r * cols + c] = v;
            }
        }
    }
    return N4M_OK;
}

void column_means(const std::vector<double>& mat, std::size_t rows,
                  std::size_t cols, std::vector<double>& means) {
    means.assign(cols, 0.0);
    if (rows == 0) return;
    for (std::size_t r = 0; r < rows; ++r) {
        for (std::size_t c = 0; c < cols; ++c) {
            means[c] += mat[r * cols + c];
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

// Power-method dominant eigenvector of symmetric H (size p × p).
// Returns w (length p) normalized to unit l2 norm.
void power_method(const std::vector<double>& H, std::size_t p,
                  std::vector<double>& w) {
    w.assign(p, 0.0);
    if (p == 0) return;
    // Initialize with the first column of H — matches the MATLAB
    // `powermethod` helper used by libPLS, which seeds with H[:,1].
    double seed_norm_sq = 0.0;
    for (std::size_t i = 0; i < p; ++i) {
        const double v = H[i * p + 0];
        w[i] = v;
        seed_norm_sq += v * v;
    }
    if (seed_norm_sq < kEps) {
        // Fall back to e_0.
        std::fill(w.begin(), w.end(), 0.0);
        w[0] = 1.0;
        seed_norm_sq = 1.0;
    }
    {
        const double s = 1.0 / std::sqrt(seed_norm_sq);
        for (double& v : w) v *= s;
    }

    std::vector<double> w_new(p, 0.0);
    double prev_eig = 0.0;
    for (std::int32_t it = 0; it < kPowerMaxIter; ++it) {
        // w_new = H * w
        for (std::size_t i = 0; i < p; ++i) {
            double s = 0.0;
            for (std::size_t j = 0; j < p; ++j) {
                s += H[i * p + j] * w[j];
            }
            w_new[i] = s;
        }
        double norm_sq = 0.0;
        for (double v : w_new) norm_sq += v * v;
        if (norm_sq < kEps) {
            // Degenerate; return current w.
            return;
        }
        const double norm = std::sqrt(norm_sq);
        const double inv = 1.0 / norm;
        for (double& v : w_new) v *= inv;
        // Align sign with previous iterate to keep deterministic output.
        double dot = 0.0;
        for (std::size_t i = 0; i < p; ++i) dot += w_new[i] * w[i];
        if (dot < 0.0) {
            for (double& v : w_new) v = -v;
        }
        const double eig = norm;  // Rayleigh-ish proxy after normalisation.
        if (it > 0 && std::fabs(eig - prev_eig) < kPowerTol * std::fabs(prev_eig + kEps)) {
            w = w_new;
            return;
        }
        prev_eig = eig;
        w = w_new;
    }
}

// Solve a linear system A * x = b for a small dense square A using
// Gaussian elimination with partial pivoting. A is m×m row-major, x and
// b are length m. Returns true on success, false if singular.
bool solve_dense(std::vector<double> A, std::vector<double>& b,
                 std::vector<double>& x, std::size_t m) {
    x.assign(m, 0.0);
    if (m == 0) return true;
    // Forward elimination.
    std::vector<std::size_t> perm(m, 0);
    for (std::size_t i = 0; i < m; ++i) perm[i] = i;
    for (std::size_t k = 0; k < m; ++k) {
        // Pivot search.
        std::size_t pivot = k;
        double max_abs = std::fabs(A[perm[k] * m + k]);
        for (std::size_t i = k + 1; i < m; ++i) {
            const double v = std::fabs(A[perm[i] * m + k]);
            if (v > max_abs) {
                max_abs = v;
                pivot = i;
            }
        }
        if (max_abs < kEps) return false;
        std::swap(perm[k], perm[pivot]);
        const double pivot_val = A[perm[k] * m + k];
        for (std::size_t i = k + 1; i < m; ++i) {
            const double factor = A[perm[i] * m + k] / pivot_val;
            for (std::size_t j = k; j < m; ++j) {
                A[perm[i] * m + j] -= factor * A[perm[k] * m + j];
            }
            b[perm[i]] -= factor * b[perm[k]];
        }
    }
    // Back substitution.
    for (std::size_t step = 0; step < m; ++step) {
        const std::size_t k = m - 1U - step;
        const std::size_t pk = perm[k];
        double s = b[pk];
        for (std::size_t j = k + 1U; j < m; ++j) {
            s -= A[pk * m + j] * x[j];
        }
        const double piv = A[pk * m + k];
        if (std::fabs(piv) < kEps) return false;
        x[k] = s / piv;
    }
    return true;
}

}  // namespace

n4m_status_t fit_ecr(Context& ctx,
                     const Config& cfg,
                     const n4m_matrix_view_t& X,
                     const n4m_matrix_view_t& Y,
                     double alpha,
                     EcrResult& out) {
    out = EcrResult{};
    if (!std::isfinite(alpha)) {
        ctx.set_error("alpha must be a finite real value");
        return N4M_ERR_INVALID_ARGUMENT;
    }
    if (alpha < 0.0) alpha = 0.0;
    if (alpha > 1.0) alpha = 1.0;
    out.alpha = alpha;

    std::vector<double> X_buf;
    n4m_status_t status = copy_matrix(ctx, X, "X", X_buf);
    if (status != N4M_OK) return status;
    std::vector<double> Y_buf;
    status = copy_matrix(ctx, Y, "Y", Y_buf);
    if (status != N4M_OK) return status;

    const auto n = static_cast<std::size_t>(X.rows);
    const auto p = static_cast<std::size_t>(X.cols);
    const auto q = static_cast<std::size_t>(Y.cols);
    if (n == 0 || p == 0 || q == 0) {
        ctx.set_error("ECR matrices must be non-empty");
        return N4M_ERR_INVALID_ARGUMENT;
    }
    if (X.rows != Y.rows) {
        ctx.set_errorf("X rows (%lld) must match Y rows (%lld)",
                       static_cast<long long>(X.rows),
                       static_cast<long long>(Y.rows));
        return N4M_ERR_SHAPE_MISMATCH;
    }
    if (cfg.n_components < 1) {
        ctx.set_error("cfg.n_components must be >= 1");
        return N4M_ERR_INVALID_ARGUMENT;
    }

    // Center X and Y. libPLS 'center' method = mean only (no scaling).
    out.x_mean.assign(p, 0.0);
    out.x_scale.assign(p, 1.0);
    out.y_mean.assign(q, 0.0);
    out.y_scale.assign(q, 1.0);
    column_means(X_buf, n, p, out.x_mean);
    column_means(Y_buf, n, q, out.y_mean);
    subtract_means(X_buf, n, p, out.x_mean);
    subtract_means(Y_buf, n, q, out.y_mean);

    // Keep the original centered X for prediction (B is built in deflated
    // space but maps the centered X back to centered Y).
    const std::vector<double> X_orig = X_buf;

    // Total variances (after centering) for R2 reporting.
    double ssq_x = 0.0;
    for (double v : X_buf) ssq_x += v * v;
    double ssq_y = 0.0;
    for (double v : Y_buf) ssq_y += v * v;
    if (ssq_x < kEps) ssq_x = kEps;
    if (ssq_y < kEps) ssq_y = kEps;

    // Effective number of components: min(cfg.n_components, n-1, p-1) — but
    // when p == 1 the libPLS cap is `min(n-1, p-1) = 0`. Match the MATLAB
    // formula but clamp to >= 1 when both n > 1 and p >= 1 to allow the
    // degenerate single-feature single-target case.
    std::size_t a = static_cast<std::size_t>(cfg.n_components);
    if (n > 0) a = std::min(a, n - 1);
    if (p > 1) a = std::min(a, p - 1);
    if (a == 0) {
        // libPLS cap would be 0 here, but a 0-component fit is useless.
        // Mirror the cap and produce a zero-coefficient predictor that
        // still returns y_mean.
        a = 0;
    }
    out.n_samples = static_cast<std::int64_t>(n);
    out.n_features = static_cast<std::int32_t>(p);
    out.n_targets = static_cast<std::int32_t>(q);
    out.n_components = static_cast<std::int32_t>(a);

    out.weights_w.assign(p * a, 0.0);
    out.loadings_p.assign(p * a, 0.0);
    out.y_loadings.assign(q * a, 0.0);
    out.r2x.assign(a, 0.0);
    out.r2y.assign(a, 0.0);

    // Per-component scores t (n).
    std::vector<double> t_vec(n, 0.0);

    // ECR loop. Stop early if the deflated X collapses to zero.
    for (std::size_t i = 0; i < a; ++i) {
        // Build H = (1-alpha) * X'X + alpha * X' y y' X.
        // y here is the (possibly multi-target) Y treated jointly: the
        // libPLS reference only supports q=1 explicitly; for q>1 we
        // generalize by replacing y y' with the (n × n) cross-Gram
        // Y Y'. Then X' y y' X becomes X' (YY') X = (X'Y)(X'Y)'.
        std::vector<double> XtX(p * p, 0.0);
        std::vector<double> XtY(p * q, 0.0);
        for (std::size_t r = 0; r < n; ++r) {
            for (std::size_t f = 0; f < p; ++f) {
                const double xf = X_buf[r * p + f];
                for (std::size_t g = 0; g < p; ++g) {
                    XtX[f * p + g] += xf * X_buf[r * p + g];
                }
                for (std::size_t target = 0; target < q; ++target) {
                    XtY[f * q + target] += xf * Y_buf[r * q + target];
                }
            }
        }
        // M = XtY * XtY' (p × p)
        std::vector<double> H(p * p, 0.0);
        for (std::size_t f = 0; f < p; ++f) {
            for (std::size_t g = 0; g < p; ++g) {
                double s = 0.0;
                for (std::size_t target = 0; target < q; ++target) {
                    s += XtY[f * q + target] * XtY[g * q + target];
                }
                H[f * p + g] = (1.0 - alpha) * XtX[f * p + g] + alpha * s;
            }
        }

        std::vector<double> w_vec;
        power_method(H, p, w_vec);

        // t = X * w
        for (std::size_t r = 0; r < n; ++r) {
            double s = 0.0;
            for (std::size_t f = 0; f < p; ++f) {
                s += X_buf[r * p + f] * w_vec[f];
            }
            t_vec[r] = s;
        }
        double tt = 0.0;
        for (std::size_t r = 0; r < n; ++r) tt += t_vec[r] * t_vec[r];
        if (tt < kEps) {
            // X has been fully deflated; record zero loadings + bail.
            for (std::size_t k = i; k < a; ++k) {
                for (std::size_t f = 0; f < p; ++f) {
                    out.weights_w[f * a + k] = 0.0;
                    out.loadings_p[f * a + k] = 0.0;
                }
                for (std::size_t target = 0; target < q; ++target) {
                    out.y_loadings[target * a + k] = 0.0;
                }
            }
            break;
        }
        // p_load = X' t / (t' t)
        std::vector<double> p_load(p, 0.0);
        for (std::size_t f = 0; f < p; ++f) {
            double s = 0.0;
            for (std::size_t r = 0; r < n; ++r) {
                s += X_buf[r * p + f] * t_vec[r];
            }
            p_load[f] = s / tt;
        }
        // r_load = Y' t / (t' t)   (length q)
        std::vector<double> r_load(q, 0.0);
        for (std::size_t target = 0; target < q; ++target) {
            double s = 0.0;
            for (std::size_t r = 0; r < n; ++r) {
                s += Y_buf[r * q + target] * t_vec[r];
            }
            r_load[target] = s / tt;
        }

        // Store column i in the (p × a) and (q × a) buffers.
        for (std::size_t f = 0; f < p; ++f) {
            out.weights_w[f * a + i] = w_vec[f];
            out.loadings_p[f * a + i] = p_load[f];
        }
        for (std::size_t target = 0; target < q; ++target) {
            out.y_loadings[target * a + i] = r_load[target];
        }

        // R2 (libPLS): % variance captured by component i.
        const std::size_t cols_a = a;
        const std::size_t col_i = i;
        double tnorm = tt;
        double pnorm = 0.0;
        for (std::size_t f = 0; f < p; ++f) {
            pnorm += out.loadings_p[f * cols_a + col_i] *
                     out.loadings_p[f * cols_a + col_i];
        }
        double rnorm = 0.0;
        for (std::size_t target = 0; target < q; ++target) {
            rnorm += out.y_loadings[target * cols_a + col_i] *
                     out.y_loadings[target * cols_a + col_i];
        }
        out.r2x[i] = (tnorm * pnorm) / ssq_x * 100.0;
        out.r2y[i] = (tnorm * rnorm) / ssq_y * 100.0;

        // Deflate: X <- X - t p',  Y <- Y - t r'.
        for (std::size_t r = 0; r < n; ++r) {
            const double tr = t_vec[r];
            for (std::size_t f = 0; f < p; ++f) {
                X_buf[r * p + f] -= tr * p_load[f];
            }
            for (std::size_t target = 0; target < q; ++target) {
                Y_buf[r * q + target] -= tr * r_load[target];
            }
        }
    }

    // Build Wstar = W (P' W)^{-1} (so that X · Wstar = T).
    // P' W is (a × a) — solve via Gaussian elimination on a small dense
    // system, one right-hand-side column of identity at a time.
    out.wstar.assign(p * a, 0.0);
    if (a > 0) {
        std::vector<double> PtW(a * a, 0.0);
        for (std::size_t i = 0; i < a; ++i) {
            for (std::size_t j = 0; j < a; ++j) {
                double s = 0.0;
                for (std::size_t f = 0; f < p; ++f) {
                    s += out.loadings_p[f * a + i] * out.weights_w[f * a + j];
                }
                PtW[i * a + j] = s;
            }
        }
        // Compute (PtW)^{-1} columnwise.
        std::vector<double> inv(a * a, 0.0);
        for (std::size_t col = 0; col < a; ++col) {
            std::vector<double> b(a, 0.0);
            b[col] = 1.0;
            std::vector<double> x;
            if (!solve_dense(PtW, b, x, a)) {
                ctx.set_error("ECR: P' W is singular; cannot compute Wstar / B");
                return N4M_ERR_INTERNAL;
            }
            for (std::size_t i = 0; i < a; ++i) inv[i * a + col] = x[i];
        }
        // Wstar = W * inv  (p × a) = (p × a) * (a × a)
        for (std::size_t f = 0; f < p; ++f) {
            for (std::size_t col = 0; col < a; ++col) {
                double s = 0.0;
                for (std::size_t k = 0; k < a; ++k) {
                    s += out.weights_w[f * a + k] * inv[k * a + col];
                }
                out.wstar[f * a + col] = s;
            }
        }
    }

    // Regression coefficients B (p × q) = Wstar * R'.
    out.coefficients.assign(p * q, 0.0);
    if (a > 0) {
        for (std::size_t f = 0; f < p; ++f) {
            for (std::size_t target = 0; target < q; ++target) {
                double s = 0.0;
                for (std::size_t k = 0; k < a; ++k) {
                    s += out.wstar[f * a + k] * out.y_loadings[target * a + k];
                }
                out.coefficients[f * q + target] = s;
            }
        }
    }

    // Predictions on the original centered X, then add y_mean back.
    out.predictions.assign(n * q, 0.0);
    double ss_res = 0.0;
    std::size_t n_finite = 0;
    for (std::size_t r = 0; r < n; ++r) {
        for (std::size_t target = 0; target < q; ++target) {
            double pred = 0.0;
            for (std::size_t f = 0; f < p; ++f) {
                pred += X_orig[r * p + f] * out.coefficients[f * q + target];
            }
            pred += out.y_mean[target];
            out.predictions[r * q + target] = pred;
            // Re-read centered Y for residuals.
            // Y_buf has been deflated; reconstruct original y via y_mean.
            const double orig_y = [&] {
                if (Y.dtype == N4M_DTYPE_F64) {
                    const auto* data = static_cast<const double*>(Y.data);
                    const auto rs = static_cast<std::size_t>(Y.row_stride);
                    const auto cs = static_cast<std::size_t>(Y.col_stride);
                    return data[r * rs + target * cs];
                }
                const auto* data = static_cast<const float*>(Y.data);
                const auto rs = static_cast<std::size_t>(Y.row_stride);
                const auto cs = static_cast<std::size_t>(Y.col_stride);
                return static_cast<double>(data[r * rs + target * cs]);
            }();
            const double diff = pred - orig_y;
            ss_res += diff * diff;
            ++n_finite;
        }
    }
    out.rmse = n_finite > 0 ? std::sqrt(ss_res / static_cast<double>(n_finite))
                            : 0.0;
    ctx.clear_error();
    return N4M_OK;
}

}  // namespace n4m::core
