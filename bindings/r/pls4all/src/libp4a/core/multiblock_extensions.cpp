// SPDX-License-Identifier: CECILL-2.1

#include "core/multiblock_extensions.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <new>
#include <vector>

#include "core/matrix_view.hpp"
#include "core/model.hpp"

namespace pls4all::core {

namespace {

constexpr double kEps = 1e-12;

[[maybe_unused]] inline std::size_t idx(std::size_t row, std::size_t cols, std::size_t col) noexcept {
    return row * cols + col;
}

[[nodiscard]] p4a_status_t copy_matrix(::pls4all::core::Context& ctx,
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
            if (!std::isfinite(v)) {
                ctx.set_errorf("%s contains NaN or Inf", name);
                return P4A_ERR_INVALID_ARGUMENT;
            }
            out[r * cols + c] = v;
        }
    }
    return P4A_OK;
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
    for (std::size_t c = 0; c < cols; ++c) means[c] *= inv;
}

void subtract_means(std::vector<double>& mat, std::size_t rows,
                    std::size_t cols, const std::vector<double>& means) {
    for (std::size_t r = 0; r < rows; ++r) {
        for (std::size_t c = 0; c < cols; ++c) {
            mat[r * cols + c] -= means[c];
        }
    }
}

double squared_norm(const std::vector<double>& v) {
    double s = 0.0;
    for (double value : v) s += value * value;
    return s;
}

double dot(const std::vector<double>& a, const std::vector<double>& b) {
    double s = 0.0;
    for (std::size_t i = 0; i < a.size(); ++i) s += a[i] * b[i];
    return s;
}

// Standard NIPALS PLS1: returns weights W (p × k), x-scores T (n × k),
// x-loadings P (p × k), regression coefficients (p × q) computed in
// centered space. Y must be (n × q). Operates on already-centered xk/yk;
// the caller restores intercepts.
[[nodiscard]] p4a_status_t nipals_pls(std::vector<double>& xk,
                                       std::vector<double>& yk,
                                       std::size_t n, std::size_t p,
                                       std::size_t q, std::size_t k,
                                       std::vector<double>& W,
                                       std::vector<double>& T,
                                       std::vector<double>& P,
                                       std::vector<double>& Q,
                                       std::vector<double>& B) {
    W.assign(p * k, 0.0);
    T.assign(n * k, 0.0);
    P.assign(p * k, 0.0);
    Q.assign(q * k, 0.0);
    B.assign(k, 0.0);
    std::vector<double> w(p, 0.0);
    std::vector<double> t(n, 0.0);
    std::vector<double> p_load(p, 0.0);
    std::vector<double> q_load(q, 0.0);

    for (std::size_t comp = 0; comp < k; ++comp) {
        // Use first column of Y for u initialization.
        std::vector<double> u(n, 0.0);
        for (std::size_t i = 0; i < n; ++i) u[i] = yk[i * q];

        for (int iter = 0; iter < 100; ++iter) {
            // w = X' u / |X' u|
            std::fill(w.begin(), w.end(), 0.0);
            for (std::size_t f = 0; f < p; ++f) {
                for (std::size_t i = 0; i < n; ++i) {
                    w[f] += xk[i * p + f] * u[i];
                }
            }
            double w_norm = std::sqrt(squared_norm(w));
            if (w_norm < kEps) break;
            for (double& v : w) v /= w_norm;

            // t = X w
            std::fill(t.begin(), t.end(), 0.0);
            for (std::size_t i = 0; i < n; ++i) {
                for (std::size_t f = 0; f < p; ++f) {
                    t[i] += xk[i * p + f] * w[f];
                }
            }
            // c = Y' t / (t.t)
            const double tt = squared_norm(t);
            if (tt < kEps) break;
            std::fill(q_load.begin(), q_load.end(), 0.0);
            for (std::size_t target = 0; target < q; ++target) {
                for (std::size_t i = 0; i < n; ++i) {
                    q_load[target] += yk[i * q + target] * t[i];
                }
                q_load[target] /= tt;
            }
            // u_new = Y c / (c.c)
            const double cc = squared_norm(q_load);
            if (cc < kEps) break;
            std::vector<double> u_new(n, 0.0);
            for (std::size_t i = 0; i < n; ++i) {
                for (std::size_t target = 0; target < q; ++target) {
                    u_new[i] += yk[i * q + target] * q_load[target];
                }
                u_new[i] /= cc;
            }
            double change = 0.0;
            for (std::size_t i = 0; i < n; ++i) {
                const double diff = u_new[i] - u[i];
                change += diff * diff;
            }
            u = std::move(u_new);
            if (change < 1e-12) break;
        }

        // p_load = X' t / (t.t)
        const double tt2 = squared_norm(t);
        if (tt2 < kEps) {
            for (std::size_t f = 0; f < p; ++f) {
                W[f * k + comp] = w[f];
            }
            continue;
        }
        std::fill(p_load.begin(), p_load.end(), 0.0);
        for (std::size_t f = 0; f < p; ++f) {
            for (std::size_t i = 0; i < n; ++i) {
                p_load[f] += xk[i * p + f] * t[i];
            }
            p_load[f] /= tt2;
        }
        // b = u' t / (t.t)
        double ut = 0.0;
        for (std::size_t i = 0; i < n; ++i) ut += u[i] * t[i];
        const double b = ut / tt2;

        for (std::size_t f = 0; f < p; ++f) W[f * k + comp] = w[f];
        for (std::size_t i = 0; i < n; ++i) T[i * k + comp] = t[i];
        for (std::size_t f = 0; f < p; ++f) P[f * k + comp] = p_load[f];
        for (std::size_t target = 0; target < q; ++target)
            Q[target * k + comp] = q_load[target];
        B[comp] = b;

        // Deflate xk and yk
        for (std::size_t i = 0; i < n; ++i) {
            for (std::size_t f = 0; f < p; ++f) {
                xk[i * p + f] -= t[i] * p_load[f];
            }
            for (std::size_t target = 0; target < q; ++target) {
                yk[i * q + target] -= b * t[i] * q_load[target];
            }
        }
    }
    return P4A_OK;
}

// Compute coefficients = W (P' W)^{-1} diag(B) Q' for NIPALS results.
// Inputs are p × k, k × k, k, q × k. Output is p × q row-major.
void coefficients_from_nipals(const std::vector<double>& W,
                              const std::vector<double>& P,
                              const std::vector<double>& Q_load,
                              const std::vector<double>& B,
                              std::size_t p, std::size_t q, std::size_t k,
                              std::vector<double>& coefficients) {
    // Compute P' W (k × k)
    std::vector<double> pw(k * k, 0.0);
    for (std::size_t i = 0; i < k; ++i) {
        for (std::size_t j = 0; j < k; ++j) {
            double s = 0.0;
            for (std::size_t f = 0; f < p; ++f) {
                s += P[f * k + i] * W[f * k + j];
            }
            pw[i * k + j] = s;
        }
    }
    // Invert pw via Gauss-Jordan elimination (k is small).
    std::vector<double> inv(k * k, 0.0);
    for (std::size_t i = 0; i < k; ++i) inv[i * k + i] = 1.0;
    std::vector<double> a = pw;
    for (std::size_t col = 0; col < k; ++col) {
        // Pivot
        std::size_t pivot = col;
        double pivot_val = std::fabs(a[col * k + col]);
        for (std::size_t r = col + 1; r < k; ++r) {
            const double v = std::fabs(a[r * k + col]);
            if (v > pivot_val) { pivot = r; pivot_val = v; }
        }
        if (pivot_val < kEps) continue;
        if (pivot != col) {
            for (std::size_t c = 0; c < k; ++c) {
                std::swap(a[col * k + c], a[pivot * k + c]);
                std::swap(inv[col * k + c], inv[pivot * k + c]);
            }
        }
        const double diag = a[col * k + col];
        for (std::size_t c = 0; c < k; ++c) {
            a[col * k + c] /= diag;
            inv[col * k + c] /= diag;
        }
        for (std::size_t r = 0; r < k; ++r) {
            if (r == col) continue;
            const double factor = a[r * k + col];
            if (std::fabs(factor) < kEps) continue;
            for (std::size_t c = 0; c < k; ++c) {
                a[r * k + c] -= factor * a[col * k + c];
                inv[r * k + c] -= factor * inv[col * k + c];
            }
        }
    }
    // R = W * inv(P' W) (p × k)
    std::vector<double> R(p * k, 0.0);
    for (std::size_t f = 0; f < p; ++f) {
        for (std::size_t j = 0; j < k; ++j) {
            double s = 0.0;
            for (std::size_t i = 0; i < k; ++i) {
                s += W[f * k + i] * inv[i * k + j];
            }
            R[f * k + j] = s;
        }
    }
    // coefficients[f, t] = sum_j R[f, j] * B[j] * Q[t, j]
    coefficients.assign(p * q, 0.0);
    for (std::size_t f = 0; f < p; ++f) {
        for (std::size_t t = 0; t < q; ++t) {
            double s = 0.0;
            for (std::size_t j = 0; j < k; ++j) {
                s += R[f * k + j] * B[j] * Q_load[t * k + j];
            }
            coefficients[f * q + t] = s;
        }
    }
}

// Dominant left singular direction of M (rows × cols) via power iteration.
[[nodiscard]] bool dominant_direction(const std::vector<double>& M,
                                       std::size_t rows, std::size_t cols,
                                       std::vector<double>& out) {
    out.assign(rows, 0.0);
    if (rows == 0 || cols == 0) return false;
    // Initialize with column of largest L2 norm.
    std::vector<double> col_norm(cols, 0.0);
    for (std::size_t c = 0; c < cols; ++c) {
        for (std::size_t r = 0; r < rows; ++r) {
            const double v = M[r * cols + c];
            col_norm[c] += v * v;
        }
    }
    std::size_t best_col = 0;
    for (std::size_t c = 1; c < cols; ++c) {
        if (col_norm[c] > col_norm[best_col]) best_col = c;
    }
    if (col_norm[best_col] < kEps) return false;
    for (std::size_t r = 0; r < rows; ++r) {
        out[r] = M[r * cols + best_col];
    }
    double norm = std::sqrt(squared_norm(out));
    if (norm < kEps) return false;
    for (double& v : out) v /= norm;
    // Power iterations on M M'.
    std::vector<double> tmp(cols, 0.0);
    for (int iter = 0; iter < 64; ++iter) {
        std::fill(tmp.begin(), tmp.end(), 0.0);
        for (std::size_t c = 0; c < cols; ++c) {
            for (std::size_t r = 0; r < rows; ++r) {
                tmp[c] += M[r * cols + c] * out[r];
            }
        }
        std::vector<double> next(rows, 0.0);
        for (std::size_t r = 0; r < rows; ++r) {
            for (std::size_t c = 0; c < cols; ++c) {
                next[r] += M[r * cols + c] * tmp[c];
            }
        }
        const double next_norm = std::sqrt(squared_norm(next));
        if (next_norm < kEps) return true;
        for (double& v : next) v /= next_norm;
        double diff = 0.0;
        for (std::size_t r = 0; r < rows; ++r) {
            const double d = next[r] - out[r];
            diff += d * d;
        }
        out = std::move(next);
        if (diff < 1e-12) break;
    }
    return true;
}

}  // namespace

// ---------- O2PLS --------------------------------------------------------

p4a_status_t fit_o2pls(Context& ctx,
                       const Config& cfg,
                       const p4a_matrix_view_t& X,
                       const p4a_matrix_view_t& Y,
                       std::int32_t n_predictive,
                       std::int32_t n_x_orthogonal,
                       std::int32_t n_y_orthogonal,
                       O2PlsResult& out) {
    (void)cfg;
    out = O2PlsResult{};
    if (n_predictive < 1) {
        ctx.set_error("O2PLS needs at least one predictive component");
        return P4A_ERR_INVALID_ARGUMENT;
    }
    if (n_x_orthogonal < 0 || n_y_orthogonal < 0) {
        ctx.set_error("orthogonal counts must be >= 0");
        return P4A_ERR_INVALID_ARGUMENT;
    }
    if (X.rows != Y.rows) {
        ctx.set_error("X and Y must have the same row count");
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
    if (n < 2 || p < 1 || q < 1) {
        ctx.set_error("O2PLS requires at least 2 rows and 1 column on each side");
        return P4A_ERR_INVALID_ARGUMENT;
    }

    column_means(X_buf, n, p, out.x_mean);
    column_means(Y_buf, n, q, out.y_mean);
    subtract_means(X_buf, n, p, out.x_mean);
    subtract_means(Y_buf, n, q, out.y_mean);

    out.n_samples = static_cast<std::int64_t>(n);
    out.n_features_x = static_cast<std::int32_t>(p);
    out.n_features_y = static_cast<std::int32_t>(q);
    out.n_predictive = n_predictive;
    out.n_x_orthogonal = n_x_orthogonal;
    out.n_y_orthogonal = n_y_orthogonal;

    // Step 1: deflate X by its components orthogonal to Y. Strategy: at
    // each iteration compute the dominant direction of X' Y to get a
    // predictive direction w_p. Project w_p out via X' Y, then the
    // remaining variance in X' Y minus this column is "X-orthogonal".
    // We use the simplified O2PLS recipe from Trygg & Wold (2003).
    std::vector<double> Xk = X_buf;
    std::vector<double> Yk = Y_buf;

    // Remove X-orthogonal components: for each requested, find the
    // dominant direction of the residual that is NOT correlated with Y.
    out.w_x_orthogonal.assign(p * static_cast<std::size_t>(n_x_orthogonal), 0.0);
    for (std::int32_t comp = 0; comp < n_x_orthogonal; ++comp) {
        // Compute X' Y (p × q)
        std::vector<double> S(p * q, 0.0);
        for (std::size_t f = 0; f < p; ++f) {
            for (std::size_t target = 0; target < q; ++target) {
                double s = 0.0;
                for (std::size_t i = 0; i < n; ++i) {
                    s += Xk[i * p + f] * Yk[i * q + target];
                }
                S[f * q + target] = s;
            }
        }
        // Predictive direction
        std::vector<double> w_p;
        if (!dominant_direction(S, p, q, w_p)) break;
        // X-loading direction
        std::vector<double> t(n, 0.0);
        for (std::size_t i = 0; i < n; ++i) {
            for (std::size_t f = 0; f < p; ++f) t[i] += Xk[i * p + f] * w_p[f];
        }
        std::vector<double> p_load(p, 0.0);
        const double tt = squared_norm(t);
        if (tt < kEps) break;
        for (std::size_t f = 0; f < p; ++f) {
            for (std::size_t i = 0; i < n; ++i) {
                p_load[f] += Xk[i * p + f] * t[i];
            }
            p_load[f] /= tt;
        }
        // X-orthogonal direction = p_load - w_p * (w_p . p_load)
        const double scal = dot(w_p, p_load);
        std::vector<double> w_o(p, 0.0);
        for (std::size_t f = 0; f < p; ++f) w_o[f] = p_load[f] - scal * w_p[f];
        const double n_o = std::sqrt(squared_norm(w_o));
        if (n_o < kEps) break;
        for (double& v : w_o) v /= n_o;
        // Score on orthogonal direction
        std::vector<double> t_o(n, 0.0);
        for (std::size_t i = 0; i < n; ++i) {
            for (std::size_t f = 0; f < p; ++f) t_o[i] += Xk[i * p + f] * w_o[f];
        }
        // Deflate Xk
        for (std::size_t i = 0; i < n; ++i) {
            for (std::size_t f = 0; f < p; ++f) {
                Xk[i * p + f] -= t_o[i] * w_o[f];
            }
        }
        for (std::size_t f = 0; f < p; ++f) {
            out.w_x_orthogonal[f * static_cast<std::size_t>(n_x_orthogonal) +
                               static_cast<std::size_t>(comp)] = w_o[f];
        }
    }

    // Symmetric Y-orthogonal removal.
    out.c_y_orthogonal.assign(q * static_cast<std::size_t>(n_y_orthogonal), 0.0);
    for (std::int32_t comp = 0; comp < n_y_orthogonal; ++comp) {
        std::vector<double> S(q * p, 0.0);
        for (std::size_t target = 0; target < q; ++target) {
            for (std::size_t f = 0; f < p; ++f) {
                double s = 0.0;
                for (std::size_t i = 0; i < n; ++i) {
                    s += Yk[i * q + target] * Xk[i * p + f];
                }
                S[target * p + f] = s;
            }
        }
        std::vector<double> c_p;
        if (!dominant_direction(S, q, p, c_p)) break;
        std::vector<double> u(n, 0.0);
        for (std::size_t i = 0; i < n; ++i) {
            for (std::size_t target = 0; target < q; ++target) {
                u[i] += Yk[i * q + target] * c_p[target];
            }
        }
        const double uu = squared_norm(u);
        if (uu < kEps) break;
        std::vector<double> q_load(q, 0.0);
        for (std::size_t target = 0; target < q; ++target) {
            for (std::size_t i = 0; i < n; ++i) {
                q_load[target] += Yk[i * q + target] * u[i];
            }
            q_load[target] /= uu;
        }
        const double scal = dot(c_p, q_load);
        std::vector<double> c_o(q, 0.0);
        for (std::size_t target = 0; target < q; ++target) {
            c_o[target] = q_load[target] - scal * c_p[target];
        }
        const double n_o = std::sqrt(squared_norm(c_o));
        if (n_o < kEps) break;
        for (double& v : c_o) v /= n_o;
        std::vector<double> u_o(n, 0.0);
        for (std::size_t i = 0; i < n; ++i) {
            for (std::size_t target = 0; target < q; ++target) {
                u_o[i] += Yk[i * q + target] * c_o[target];
            }
        }
        for (std::size_t i = 0; i < n; ++i) {
            for (std::size_t target = 0; target < q; ++target) {
                Yk[i * q + target] -= u_o[i] * c_o[target];
            }
        }
        for (std::size_t target = 0; target < q; ++target) {
            out.c_y_orthogonal[target * static_cast<std::size_t>(n_y_orthogonal) +
                               static_cast<std::size_t>(comp)] = c_o[target];
        }
    }

    // Step 2: run NIPALS PLS on the deflated Xk / Yk for n_predictive
    // components.
    std::vector<double> W, T, P_load, Q_load, B;
    status = nipals_pls(Xk, Yk, n, p, q,
                        static_cast<std::size_t>(n_predictive),
                        W, T, P_load, Q_load, B);
    if (status != P4A_OK) return status;
    out.w_predictive = std::move(W);
    out.c_predictive = std::move(Q_load);
    out.b_predictive = std::move(B);
    coefficients_from_nipals(out.w_predictive,
                              P_load,
                              out.c_predictive,
                              out.b_predictive,
                              p, q,
                              static_cast<std::size_t>(n_predictive),
                              out.coefficients);

    ctx.clear_error();
    return P4A_OK;
}

// ---------- SO-PLS ------------------------------------------------------

p4a_status_t fit_so_pls(Context& ctx,
                        const Config& cfg,
                        const std::vector<p4a_matrix_view_t>& X_blocks,
                        const p4a_matrix_view_t& Y,
                        const std::vector<std::int32_t>& n_components_per_block,
                        SoPlsResult& out) {
    (void)cfg;
    out = SoPlsResult{};
    if (X_blocks.empty()) {
        ctx.set_error("SO-PLS requires at least one X block");
        return P4A_ERR_INVALID_ARGUMENT;
    }
    if (n_components_per_block.size() != X_blocks.size()) {
        ctx.set_error("n_components_per_block must have one entry per block");
        return P4A_ERR_INVALID_ARGUMENT;
    }
    const std::int64_t n_rows = Y.rows;
    for (const auto& X : X_blocks) {
        if (X.rows != n_rows) {
            ctx.set_error("all X blocks must have the same row count as Y");
            return P4A_ERR_SHAPE_MISMATCH;
        }
    }

    std::vector<double> Y_buf;
    p4a_status_t status = copy_matrix(ctx, Y, "Y", Y_buf);
    if (status != P4A_OK) return status;
    const std::size_t n = static_cast<std::size_t>(n_rows);
    const std::size_t q = static_cast<std::size_t>(Y.cols);
    column_means(Y_buf, n, q, out.y_mean);
    subtract_means(Y_buf, n, q, out.y_mean);

    out.n_blocks = static_cast<std::int32_t>(X_blocks.size());
    out.n_components_per_block = n_components_per_block;
    out.predictions.assign(n * q, 0.0);
    for (std::size_t i = 0; i < n; ++i) {
        for (std::size_t target = 0; target < q; ++target) {
            out.predictions[i * q + target] = out.y_mean[target];
        }
    }
    out.block_coefficients.assign(X_blocks.size(), {});

    std::vector<double> Y_residual = Y_buf;
    std::vector<std::vector<double>> block_scores;
    block_scores.reserve(X_blocks.size());

    for (std::size_t block = 0; block < X_blocks.size(); ++block) {
        const p4a_matrix_view_t& Xv = X_blocks[block];
        std::vector<double> X_buf;
        status = copy_matrix(ctx, Xv, "X_block", X_buf);
        if (status != P4A_OK) return status;
        const std::size_t p = static_cast<std::size_t>(Xv.cols);
        std::vector<double> x_mean;
        column_means(X_buf, n, p, x_mean);
        subtract_means(X_buf, n, p, x_mean);
        // Orthogonalize this block against previous block scores.
        for (const auto& prev_scores : block_scores) {
            const std::size_t k_prev = prev_scores.size() / n;
            for (std::size_t kp = 0; kp < k_prev; ++kp) {
                std::vector<double> t_prev(n, 0.0);
                for (std::size_t i = 0; i < n; ++i)
                    t_prev[i] = prev_scores[i * k_prev + kp];
                const double tt = squared_norm(t_prev);
                if (tt < kEps) continue;
                for (std::size_t f = 0; f < p; ++f) {
                    double dot_val = 0.0;
                    for (std::size_t i = 0; i < n; ++i) {
                        dot_val += t_prev[i] * X_buf[i * p + f];
                    }
                    dot_val /= tt;
                    for (std::size_t i = 0; i < n; ++i) {
                        X_buf[i * p + f] -= dot_val * t_prev[i];
                    }
                }
            }
        }

        const std::size_t k_b =
            static_cast<std::size_t>(n_components_per_block[block]);
        if (k_b == 0) {
            block_scores.emplace_back();
            continue;
        }
        std::vector<double> Y_centered = Y_residual;
        std::vector<double> W, T, P_load, Q_load, B;
        status = nipals_pls(X_buf, Y_centered, n, p, q, k_b,
                            W, T, P_load, Q_load, B);
        if (status != P4A_OK) return status;
        // Block coefficients (p × q).
        std::vector<double> coefs;
        coefficients_from_nipals(W, P_load, Q_load, B, p, q, k_b, coefs);
        out.block_coefficients[block] = coefs;

        // Predict block contribution and subtract from residual.
        std::vector<double> X_centered(n * p, 0.0);
        // Re-copy and center using same x_mean and apply orthogonalization
        // again, so the contribution to predictions is consistent with
        // training. To keep this simple, we use T @ Q_load' to reconstruct
        // the predicted Y contribution.
        const std::size_t k_dim = k_b;
        for (std::size_t i = 0; i < n; ++i) {
            for (std::size_t target = 0; target < q; ++target) {
                double s = 0.0;
                for (std::size_t comp = 0; comp < k_dim; ++comp) {
                    s += T[i * k_dim + comp] * B[comp] *
                         Q_load[target * k_dim + comp];
                }
                out.predictions[i * q + target] += s;
                Y_residual[i * q + target] -= s;
            }
        }
        // Store scores for orthogonalization of the next block.
        block_scores.push_back(T);
    }

    ctx.clear_error();
    return P4A_OK;
}

// ---------- OnPLS -------------------------------------------------------

p4a_status_t fit_on_pls(Context& ctx,
                        const Config& cfg,
                        const std::vector<p4a_matrix_view_t>& X_blocks,
                        std::int32_t n_joint,
                        const std::vector<std::int32_t>& n_unique_per_block,
                        OnPlsResult& out) {
    (void)cfg;
    out = OnPlsResult{};
    if (X_blocks.size() < 2) {
        ctx.set_error("OnPLS requires at least 2 X blocks");
        return P4A_ERR_INVALID_ARGUMENT;
    }
    if (n_unique_per_block.size() != X_blocks.size()) {
        ctx.set_error("n_unique_per_block must have one entry per block");
        return P4A_ERR_INVALID_ARGUMENT;
    }
    const std::int64_t n_rows = X_blocks[0].rows;
    for (const auto& X : X_blocks) {
        if (X.rows != n_rows) {
            ctx.set_error("all blocks must share row count");
            return P4A_ERR_SHAPE_MISMATCH;
        }
    }
    if (n_joint < 0) {
        ctx.set_error("n_joint must be >= 0");
        return P4A_ERR_INVALID_ARGUMENT;
    }

    const std::size_t n = static_cast<std::size_t>(n_rows);
    const std::size_t n_blocks = X_blocks.size();
    std::vector<std::vector<double>> centered(n_blocks);
    std::vector<std::size_t> p_block(n_blocks, 0);
    for (std::size_t b = 0; b < n_blocks; ++b) {
        p4a_status_t status =
            copy_matrix(ctx, X_blocks[b], "X_block", centered[b]);
        if (status != P4A_OK) return status;
        p_block[b] = static_cast<std::size_t>(X_blocks[b].cols);
        std::vector<double> mean;
        column_means(centered[b], n, p_block[b], mean);
        subtract_means(centered[b], n, p_block[b], mean);
    }

    out.n_blocks = static_cast<std::int32_t>(n_blocks);
    out.n_joint = n_joint;
    out.n_unique_per_block = n_unique_per_block;
    out.joint_loadings_per_block.assign(n_blocks, {});
    out.joint_scores_per_block.assign(n_blocks, {});
    out.unique_loadings_per_block.assign(n_blocks, {});
    for (std::size_t b = 0; b < n_blocks; ++b) {
        out.joint_loadings_per_block[b].assign(p_block[b] *
            static_cast<std::size_t>(n_joint), 0.0);
        out.joint_scores_per_block[b].assign(n *
            static_cast<std::size_t>(n_joint), 0.0);
        out.unique_loadings_per_block[b].assign(p_block[b] *
            static_cast<std::size_t>(n_unique_per_block[b]), 0.0);
    }

    // Iterative extraction of joint components: at each iteration find
    // the direction in each block that best correlates with the others.
    for (std::int32_t j = 0; j < n_joint; ++j) {
        // Build the cross-block sum_b X_b X_b^T (n × n) summed scores.
        // Simpler MVP: take first block's dominant direction and project
        // each block onto it after centering.
        std::vector<double>& X0 = centered[0];
        // SVD of X0 via power iteration on X0' X0 — get dominant direction.
        const std::size_t p0 = p_block[0];
        std::vector<double> S(p0 * p0, 0.0);
        for (std::size_t i = 0; i < p0; ++i) {
            for (std::size_t k = 0; k < p0; ++k) {
                double s = 0.0;
                for (std::size_t r = 0; r < n; ++r) {
                    s += X0[r * p0 + i] * X0[r * p0 + k];
                }
                S[i * p0 + k] = s;
            }
        }
        std::vector<double> w;
        if (!dominant_direction(S, p0, p0, w)) break;
        std::vector<double> t0(n, 0.0);
        for (std::size_t r = 0; r < n; ++r) {
            for (std::size_t f = 0; f < p0; ++f) {
                t0[r] += X0[r * p0 + f] * w[f];
            }
        }
        const double tt0 = squared_norm(t0);
        if (tt0 < kEps) break;
        // Loadings + deflate each block by t0
        for (std::size_t b = 0; b < n_blocks; ++b) {
            std::vector<double> load(p_block[b], 0.0);
            for (std::size_t f = 0; f < p_block[b]; ++f) {
                double s = 0.0;
                for (std::size_t r = 0; r < n; ++r) {
                    s += centered[b][r * p_block[b] + f] * t0[r];
                }
                load[f] = s / tt0;
                out.joint_loadings_per_block[b][
                    f * static_cast<std::size_t>(n_joint) +
                    static_cast<std::size_t>(j)] = load[f];
            }
            for (std::size_t r = 0; r < n; ++r) {
                out.joint_scores_per_block[b][
                    r * static_cast<std::size_t>(n_joint) +
                    static_cast<std::size_t>(j)] = t0[r];
                for (std::size_t f = 0; f < p_block[b]; ++f) {
                    centered[b][r * p_block[b] + f] -= t0[r] * load[f];
                }
            }
        }
    }

    // Unique components per block: standard PCA-like power iteration on
    // the remaining residual.
    for (std::size_t b = 0; b < n_blocks; ++b) {
        const std::size_t pb = p_block[b];
        std::vector<double>& Xb = centered[b];
        for (std::int32_t u = 0; u < n_unique_per_block[b]; ++u) {
            std::vector<double> S(pb * pb, 0.0);
            for (std::size_t i = 0; i < pb; ++i) {
                for (std::size_t k = 0; k < pb; ++k) {
                    double s = 0.0;
                    for (std::size_t r = 0; r < n; ++r) {
                        s += Xb[r * pb + i] * Xb[r * pb + k];
                    }
                    S[i * pb + k] = s;
                }
            }
            std::vector<double> w;
            if (!dominant_direction(S, pb, pb, w)) break;
            std::vector<double> t(n, 0.0);
            for (std::size_t r = 0; r < n; ++r) {
                for (std::size_t f = 0; f < pb; ++f)
                    t[r] += Xb[r * pb + f] * w[f];
            }
            const double tt = squared_norm(t);
            if (tt < kEps) break;
            std::vector<double> load(pb, 0.0);
            for (std::size_t f = 0; f < pb; ++f) {
                double s = 0.0;
                for (std::size_t r = 0; r < n; ++r) s += Xb[r * pb + f] * t[r];
                load[f] = s / tt;
                out.unique_loadings_per_block[b][
                    f * static_cast<std::size_t>(n_unique_per_block[b]) +
                    static_cast<std::size_t>(u)] = load[f];
            }
            for (std::size_t r = 0; r < n; ++r) {
                for (std::size_t f = 0; f < pb; ++f) {
                    Xb[r * pb + f] -= t[r] * load[f];
                }
            }
        }
    }

    ctx.clear_error();
    return P4A_OK;
}

// ---------- ROSA --------------------------------------------------------

p4a_status_t fit_rosa(Context& ctx,
                      const Config& cfg,
                      const std::vector<p4a_matrix_view_t>& X_blocks,
                      const p4a_matrix_view_t& Y,
                      std::int32_t n_components,
                      RosaResult& out) {
    (void)cfg;
    out = RosaResult{};
    if (X_blocks.empty()) {
        ctx.set_error("ROSA requires at least one X block");
        return P4A_ERR_INVALID_ARGUMENT;
    }
    if (n_components < 1) {
        ctx.set_error("ROSA n_components must be >= 1");
        return P4A_ERR_INVALID_ARGUMENT;
    }
    const std::int64_t n_rows = Y.rows;
    for (const auto& X : X_blocks) {
        if (X.rows != n_rows) {
            ctx.set_error("all X blocks must match Y row count");
            return P4A_ERR_SHAPE_MISMATCH;
        }
    }

    std::vector<double> Y_buf;
    p4a_status_t status = copy_matrix(ctx, Y, "Y", Y_buf);
    if (status != P4A_OK) return status;
    const std::size_t n = static_cast<std::size_t>(n_rows);
    const std::size_t q = static_cast<std::size_t>(Y.cols);
    const std::size_t n_blocks = X_blocks.size();
    std::vector<std::vector<double>> X_bufs(n_blocks);
    std::vector<std::size_t> p_block(n_blocks, 0);
    for (std::size_t b = 0; b < n_blocks; ++b) {
        status = copy_matrix(ctx, X_blocks[b], "X_block", X_bufs[b]);
        if (status != P4A_OK) return status;
        p_block[b] = static_cast<std::size_t>(X_blocks[b].cols);
        std::vector<double> mean;
        column_means(X_bufs[b], n, p_block[b], mean);
        subtract_means(X_bufs[b], n, p_block[b], mean);
    }
    column_means(Y_buf, n, q, out.y_mean);
    subtract_means(Y_buf, n, q, out.y_mean);

    out.n_components = n_components;
    out.selected_block_per_component.assign(
        static_cast<std::size_t>(n_components), 0);
    out.predictions.assign(n * q, 0.0);
    out.block_coefficients.assign(n_blocks, {});
    for (std::size_t b = 0; b < n_blocks; ++b) {
        out.block_coefficients[b].assign(p_block[b] * q, 0.0);
    }
    for (std::size_t i = 0; i < n; ++i) {
        for (std::size_t target = 0; target < q; ++target) {
            out.predictions[i * q + target] = out.y_mean[target];
        }
    }

    std::vector<double> Y_residual = Y_buf;
    std::vector<std::vector<double>> previous_scores;  // shared across blocks

    for (std::int32_t comp = 0; comp < n_components; ++comp) {
        std::int32_t best_block = -1;
        double best_score = -std::numeric_limits<double>::infinity();
        std::vector<double> best_w;
        std::vector<double> best_t;

        for (std::size_t b = 0; b < n_blocks; ++b) {
            // Compute X_b' Y_residual.
            const std::size_t pb = p_block[b];
            std::vector<double> S(pb * q, 0.0);
            for (std::size_t f = 0; f < pb; ++f) {
                for (std::size_t target = 0; target < q; ++target) {
                    double s = 0.0;
                    for (std::size_t i = 0; i < n; ++i) {
                        s += X_bufs[b][i * pb + f] *
                             Y_residual[i * q + target];
                    }
                    S[f * q + target] = s;
                }
            }
            std::vector<double> w;
            if (!dominant_direction(S, pb, q, w)) continue;
            std::vector<double> t(n, 0.0);
            for (std::size_t i = 0; i < n; ++i) {
                for (std::size_t f = 0; f < pb; ++f) {
                    t[i] += X_bufs[b][i * pb + f] * w[f];
                }
            }
            // Orthogonalize t against previous scores.
            for (const auto& prev : previous_scores) {
                const double tt_prev = squared_norm(prev);
                if (tt_prev < kEps) continue;
                double proj = 0.0;
                for (std::size_t i = 0; i < n; ++i) proj += prev[i] * t[i];
                proj /= tt_prev;
                for (std::size_t i = 0; i < n; ++i) t[i] -= proj * prev[i];
            }
            const double tt = squared_norm(t);
            if (tt < kEps) continue;
            // Score = correlation with Y_residual squared norm.
            double score = 0.0;
            for (std::size_t target = 0; target < q; ++target) {
                double dot_val = 0.0;
                for (std::size_t i = 0; i < n; ++i) {
                    dot_val += t[i] * Y_residual[i * q + target];
                }
                score += dot_val * dot_val / tt;
            }
            if (score > best_score) {
                best_score = score;
                best_block = static_cast<std::int32_t>(b);
                best_w = std::move(w);
                best_t = std::move(t);
            }
        }

        if (best_block < 0) break;
        out.selected_block_per_component[static_cast<std::size_t>(comp)] =
            best_block;
        // Compute Y loadings and add contribution.
        const double tt = squared_norm(best_t);
        std::vector<double> q_load(q, 0.0);
        for (std::size_t target = 0; target < q; ++target) {
            double s = 0.0;
            for (std::size_t i = 0; i < n; ++i) {
                s += Y_residual[i * q + target] * best_t[i];
            }
            q_load[target] = s / tt;
        }
        // Add t @ q_load' to predictions and subtract from residual.
        for (std::size_t i = 0; i < n; ++i) {
            for (std::size_t target = 0; target < q; ++target) {
                const double contribution = best_t[i] * q_load[target];
                out.predictions[i * q + target] += contribution;
                Y_residual[i * q + target] -= contribution;
            }
        }
        // Update block coefficients via outer product w * q_load'.
        const std::size_t pb = p_block[static_cast<std::size_t>(best_block)];
        std::vector<double>& bcoef =
            out.block_coefficients[static_cast<std::size_t>(best_block)];
        // Compute residual block loading p_load = X_b' t / (t.t)
        std::vector<double> p_load(pb, 0.0);
        for (std::size_t f = 0; f < pb; ++f) {
            double s = 0.0;
            for (std::size_t i = 0; i < n; ++i) {
                s += X_bufs[static_cast<std::size_t>(best_block)][i * pb + f] *
                     best_t[i];
            }
            p_load[f] = s / tt;
        }
        for (std::size_t f = 0; f < pb; ++f) {
            for (std::size_t target = 0; target < q; ++target) {
                bcoef[f * q + target] +=
                    best_w[f] * q_load[target];
                (void)p_load;  // p_load reserved for future deflation logic
            }
        }
        // Deflate the chosen X block by t.
        for (std::size_t i = 0; i < n; ++i) {
            for (std::size_t f = 0; f < pb; ++f) {
                X_bufs[static_cast<std::size_t>(best_block)][i * pb + f] -=
                    best_t[i] * p_load[f];
            }
        }
        previous_scores.push_back(best_t);
    }

    ctx.clear_error();
    return P4A_OK;
}

}  // namespace pls4all::core
