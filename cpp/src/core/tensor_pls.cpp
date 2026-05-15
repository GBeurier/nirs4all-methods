// SPDX-License-Identifier: CeCILL-2.1

#include "core/tensor_pls.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <new>
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
            if (!std::isfinite(v)) {
                ctx.set_errorf("%s contains NaN or Inf", name);
                return P4A_ERR_INVALID_ARGUMENT;
            }
            out[r * cols + c] = v;
        }
    }
    return P4A_OK;
}

double squared_norm(const std::vector<double>& v) {
    double s = 0.0;
    for (double value : v) s += value * value;
    return s;
}

// Rank-1 decomposition of an (rows × cols) matrix via power iteration:
// returns u (rows), v (cols) such that u v' approximates M and ||u|| = 1
// and ||v|| = 1, scaled so that u v' has the correct singular value
// embedded in v (or, equivalently, we keep both unit-normalized and
// return the singular value separately).
void rank_one_decomposition(const std::vector<double>& M,
                            std::size_t rows, std::size_t cols,
                            std::vector<double>& u,
                            std::vector<double>& v,
                            double& sigma) {
    u.assign(rows, 0.0);
    v.assign(cols, 0.0);
    sigma = 0.0;
    if (rows == 0 || cols == 0) return;

    // Initialize v with the column of largest L2 norm.
    std::size_t best_col = 0;
    double best_col_norm = -1.0;
    for (std::size_t c = 0; c < cols; ++c) {
        double s = 0.0;
        for (std::size_t r = 0; r < rows; ++r) {
            const double mv = M[r * cols + c];
            s += mv * mv;
        }
        if (s > best_col_norm) { best_col_norm = s; best_col = c; }
    }
    if (!(best_col_norm > 0.0)) return;
    for (std::size_t r = 0; r < rows; ++r) u[r] = M[r * cols + best_col];
    double u_norm = std::sqrt(squared_norm(u));
    if (u_norm < kEps) return;
    for (double& x : u) x /= u_norm;

    for (int iter = 0; iter < 128; ++iter) {
        // v = M' u
        std::fill(v.begin(), v.end(), 0.0);
        for (std::size_t c = 0; c < cols; ++c) {
            for (std::size_t r = 0; r < rows; ++r) {
                v[c] += M[r * cols + c] * u[r];
            }
        }
        const double v_norm = std::sqrt(squared_norm(v));
        if (v_norm < kEps) return;
        for (double& x : v) x /= v_norm;
        // u_new = M v
        std::vector<double> u_new(rows, 0.0);
        for (std::size_t r = 0; r < rows; ++r) {
            for (std::size_t c = 0; c < cols; ++c) {
                u_new[r] += M[r * cols + c] * v[c];
            }
        }
        const double u_new_norm = std::sqrt(squared_norm(u_new));
        if (u_new_norm < kEps) return;
        for (double& x : u_new) x /= u_new_norm;
        double change = 0.0;
        for (std::size_t r = 0; r < rows; ++r) {
            const double d = u_new[r] - u[r];
            change += d * d;
        }
        u = std::move(u_new);
        if (change < 1e-14) break;
    }

    // Recover sigma = u' M v
    sigma = 0.0;
    for (std::size_t r = 0; r < rows; ++r) {
        for (std::size_t c = 0; c < cols; ++c) {
            sigma += u[r] * M[r * cols + c] * v[c];
        }
    }
}

}  // namespace

p4a_status_t fit_n_pls(Context& ctx,
                       const Config& cfg,
                       const p4a_matrix_view_t& X_flat,
                       std::int32_t mode_j,
                       std::int32_t mode_k,
                       const p4a_matrix_view_t& Y,
                       NPlsResult& out) {
    out = NPlsResult{};
    if (mode_j < 1 || mode_k < 1) {
        ctx.set_error("mode_j and mode_k must be >= 1");
        return P4A_ERR_INVALID_ARGUMENT;
    }
    if (X_flat.cols != static_cast<std::int64_t>(mode_j) * mode_k) {
        ctx.set_errorf("X cols (%lld) must equal mode_j * mode_k (%lld)",
                       static_cast<long long>(X_flat.cols),
                       static_cast<long long>(mode_j) * mode_k);
        return P4A_ERR_SHAPE_MISMATCH;
    }
    if (X_flat.rows != Y.rows) {
        ctx.set_error("X.rows must equal Y.rows");
        return P4A_ERR_SHAPE_MISMATCH;
    }
    if (cfg.n_components < 1) {
        ctx.set_error("n_components must be >= 1");
        return P4A_ERR_INVALID_ARGUMENT;
    }

    std::vector<double> X_buf, Y_buf;
    p4a_status_t status = copy_matrix(ctx, X_flat, "X", X_buf);
    if (status != P4A_OK) return status;
    status = copy_matrix(ctx, Y, "Y", Y_buf);
    if (status != P4A_OK) return status;

    const std::size_t n = static_cast<std::size_t>(X_flat.rows);
    const std::size_t J = static_cast<std::size_t>(mode_j);
    const std::size_t K = static_cast<std::size_t>(mode_k);
    const std::size_t JK = J * K;
    const std::size_t q = static_cast<std::size_t>(Y.cols);
    const std::size_t a = static_cast<std::size_t>(cfg.n_components);

    out.n_samples = static_cast<std::int64_t>(n);
    out.mode_j = mode_j;
    out.mode_k = mode_k;
    out.n_components = cfg.n_components;
    out.n_targets = static_cast<std::int32_t>(q);
    out.w_j_per_component.assign(J * a, 0.0);
    out.w_k_per_component.assign(K * a, 0.0);
    out.scores_t.assign(n * a, 0.0);
    out.y_loadings_q.assign(q * a, 0.0);
    out.regression_b.assign(a, 0.0);
    out.coefficients.assign(JK * q, 0.0);
    out.x_mean.assign(JK, 0.0);
    out.y_mean.assign(q, 0.0);

    if (cfg.center_x != 0) {
        for (std::size_t r = 0; r < n; ++r) {
            for (std::size_t c = 0; c < JK; ++c) {
                out.x_mean[c] += X_buf[r * JK + c];
            }
        }
        for (std::size_t c = 0; c < JK; ++c)
            out.x_mean[c] /= static_cast<double>(n);
        for (std::size_t r = 0; r < n; ++r) {
            for (std::size_t c = 0; c < JK; ++c) {
                X_buf[r * JK + c] -= out.x_mean[c];
            }
        }
    }
    if (cfg.center_y != 0) {
        for (std::size_t r = 0; r < n; ++r) {
            for (std::size_t c = 0; c < q; ++c) {
                out.y_mean[c] += Y_buf[r * q + c];
            }
        }
        for (std::size_t c = 0; c < q; ++c)
            out.y_mean[c] /= static_cast<double>(n);
        for (std::size_t r = 0; r < n; ++r) {
            for (std::size_t c = 0; c < q; ++c) {
                Y_buf[r * q + c] -= out.y_mean[c];
            }
        }
    }

    std::vector<double> X_residual = X_buf;
    std::vector<double> Y_residual = Y_buf;

    // P_loadings store rank-1 X loading vectors (JK × a) for deflation
    // tracking.
    std::vector<double> p_loadings(JK * a, 0.0);

    for (std::size_t comp = 0; comp < a; ++comp) {
        // Z = X_residual' Y_residual / N is JK × q. Form Z as JK × q.
        std::vector<double> Z(JK * q, 0.0);
        for (std::size_t i = 0; i < n; ++i) {
            for (std::size_t f = 0; f < JK; ++f) {
                for (std::size_t target = 0; target < q; ++target) {
                    Z[f * q + target] += X_residual[i * JK + f] *
                                          Y_residual[i * q + target];
                }
            }
        }
        // Dominant left singular direction w_full (JK).
        std::vector<double> w_full, dummy_v;
        double sigma = 0.0;
        rank_one_decomposition(Z, JK, q, w_full, dummy_v, sigma);
        if (squared_norm(w_full) < kEps) break;
        // Reshape w_full as J × K and do rank-1 decomposition → w_J, w_K.
        std::vector<double> W_mat(J * K, 0.0);
        for (std::size_t j = 0; j < J; ++j) {
            for (std::size_t k = 0; k < K; ++k) {
                W_mat[j * K + k] = w_full[j * K + k];
            }
        }
        std::vector<double> w_j, w_k;
        double sigma_jk = 0.0;
        rank_one_decomposition(W_mat, J, K, w_j, w_k, sigma_jk);
        if (squared_norm(w_j) < kEps || squared_norm(w_k) < kEps) break;

        // t = X * (w_J ⊗ w_K)
        std::vector<double> t(n, 0.0);
        for (std::size_t i = 0; i < n; ++i) {
            double s = 0.0;
            for (std::size_t j = 0; j < J; ++j) {
                for (std::size_t k = 0; k < K; ++k) {
                    s += X_residual[i * JK + j * K + k] * w_j[j] * w_k[k];
                }
            }
            t[i] = s;
        }
        const double tt = squared_norm(t);
        if (tt < kEps) break;

        // q_load = Y' t / tt
        std::vector<double> q_load(q, 0.0);
        for (std::size_t target = 0; target < q; ++target) {
            double s = 0.0;
            for (std::size_t i = 0; i < n; ++i) {
                s += Y_residual[i * q + target] * t[i];
            }
            q_load[target] = s / tt;
        }

        // u = Y q_load / (q_load.q_load), b = u'.t/tt
        const double cc = squared_norm(q_load);
        double b = 0.0;
        if (cc > kEps) {
            std::vector<double> u(n, 0.0);
            for (std::size_t i = 0; i < n; ++i) {
                double s = 0.0;
                for (std::size_t target = 0; target < q; ++target) {
                    s += Y_residual[i * q + target] * q_load[target];
                }
                u[i] = s / cc;
            }
            double ut = 0.0;
            for (std::size_t i = 0; i < n; ++i) ut += u[i] * t[i];
            b = ut / tt;
        }

        // X-loading p_load = X' t / tt
        std::vector<double> p_load(JK, 0.0);
        for (std::size_t f = 0; f < JK; ++f) {
            double s = 0.0;
            for (std::size_t i = 0; i < n; ++i) {
                s += X_residual[i * JK + f] * t[i];
            }
            p_load[f] = s / tt;
        }

        // Store and deflate.
        for (std::size_t j = 0; j < J; ++j)
            out.w_j_per_component[j * a + comp] = w_j[j];
        for (std::size_t k = 0; k < K; ++k)
            out.w_k_per_component[k * a + comp] = w_k[k];
        for (std::size_t i = 0; i < n; ++i)
            out.scores_t[i * a + comp] = t[i];
        for (std::size_t target = 0; target < q; ++target)
            out.y_loadings_q[target * a + comp] = q_load[target];
        out.regression_b[comp] = b;
        for (std::size_t f = 0; f < JK; ++f) p_loadings[f * a + comp] = p_load[f];

        for (std::size_t i = 0; i < n; ++i) {
            for (std::size_t f = 0; f < JK; ++f) {
                X_residual[i * JK + f] -= t[i] * p_load[f];
            }
            for (std::size_t target = 0; target < q; ++target) {
                Y_residual[i * q + target] -=
                    b * t[i] * q_load[target];
            }
        }
    }

    // Coefficients: project the rank-1 (w_J ⊗ w_K) directions back to
    // JK × q via W (P' W)^{-1} diag(b) Q'. Build W (JK × a) from w_j and
    // w_k.
    std::vector<double> W(JK * a, 0.0);
    for (std::size_t comp = 0; comp < a; ++comp) {
        for (std::size_t j = 0; j < J; ++j) {
            for (std::size_t k = 0; k < K; ++k) {
                W[(j * K + k) * a + comp] =
                    out.w_j_per_component[j * a + comp] *
                    out.w_k_per_component[k * a + comp];
            }
        }
    }
    // Solve (P' W) coefficients block: pw (a × a)
    std::vector<double> pw(a * a, 0.0);
    for (std::size_t i = 0; i < a; ++i) {
        for (std::size_t j = 0; j < a; ++j) {
            double s = 0.0;
            for (std::size_t f = 0; f < JK; ++f) {
                s += p_loadings[f * a + i] * W[f * a + j];
            }
            pw[i * a + j] = s;
        }
    }
    // Invert pw via Gauss-Jordan.
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
    // R = W * inv(P' W) (JK × a)
    std::vector<double> R(JK * a, 0.0);
    for (std::size_t f = 0; f < JK; ++f) {
        for (std::size_t j = 0; j < a; ++j) {
            double s = 0.0;
            for (std::size_t i = 0; i < a; ++i) {
                s += W[f * a + i] * inv[i * a + j];
            }
            R[f * a + j] = s;
        }
    }
    // coefficients[f, t] = sum_c R[f, c] * b[c] * q_load[t, c]
    for (std::size_t f = 0; f < JK; ++f) {
        for (std::size_t t = 0; t < q; ++t) {
            double s = 0.0;
            for (std::size_t c = 0; c < a; ++c) {
                s += R[f * a + c] * out.regression_b[c] *
                     out.y_loadings_q[t * a + c];
            }
            out.coefficients[f * q + t] = s;
        }
    }

    ctx.clear_error();
    return P4A_OK;
}

p4a_status_t predict_n_pls(Context& ctx,
                            const NPlsResult& model,
                            const p4a_matrix_view_t& X_flat,
                            std::vector<double>& out_predictions) {
    out_predictions.clear();
    if (model.coefficients.empty()) {
        ctx.set_error("N-PLS model is not fitted");
        return P4A_ERR_NOT_FITTED;
    }
    if (X_flat.cols !=
        static_cast<std::int64_t>(model.mode_j) * model.mode_k) {
        ctx.set_error("X cols must equal mode_j * mode_k");
        return P4A_ERR_SHAPE_MISMATCH;
    }
    std::vector<double> X_buf;
    p4a_status_t status = copy_matrix(ctx, X_flat, "X", X_buf);
    if (status != P4A_OK) return status;
    const std::size_t n = static_cast<std::size_t>(X_flat.rows);
    const std::size_t JK = X_buf.size() == 0 ? 0 : X_buf.size() / n;
    const std::size_t q = static_cast<std::size_t>(model.n_targets);

    out_predictions.assign(n * q, 0.0);
    for (std::size_t i = 0; i < n; ++i) {
        for (std::size_t target = 0; target < q; ++target) {
            double s = model.y_mean[target];
            for (std::size_t f = 0; f < JK; ++f) {
                const double centered = X_buf[i * JK + f] - model.x_mean[f];
                s += centered * model.coefficients[f * q + target];
            }
            out_predictions[i * q + target] = s;
        }
    }
    ctx.clear_error();
    return P4A_OK;
}

}  // namespace pls4all::core
