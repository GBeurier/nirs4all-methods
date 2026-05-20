// SPDX-License-Identifier: CECILL-2.1

#include "core/multiblock_extensions.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <new>
#include <numeric>
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

[[nodiscard]] bool invert_square(const std::vector<double>& input,
                                  std::size_t n,
                                  std::vector<double>& inv) {
    inv.assign(n * n, 0.0);
    for (std::size_t i = 0; i < n; ++i) inv[i * n + i] = 1.0;
    std::vector<double> a = input;
    for (std::size_t col = 0; col < n; ++col) {
        std::size_t pivot = col;
        double pivot_val = std::fabs(a[col * n + col]);
        for (std::size_t r = col + 1; r < n; ++r) {
            const double v = std::fabs(a[r * n + col]);
            if (v > pivot_val) {
                pivot = r;
                pivot_val = v;
            }
        }
        if (pivot_val < kEps) return false;
        if (pivot != col) {
            for (std::size_t c = 0; c < n; ++c) {
                std::swap(a[col * n + c], a[pivot * n + c]);
                std::swap(inv[col * n + c], inv[pivot * n + c]);
            }
        }
        const double diag = a[col * n + col];
        for (std::size_t c = 0; c < n; ++c) {
            a[col * n + c] /= diag;
            inv[col * n + c] /= diag;
        }
        for (std::size_t r = 0; r < n; ++r) {
            if (r == col) continue;
            const double factor = a[r * n + col];
            if (std::fabs(factor) < kEps) continue;
            for (std::size_t c = 0; c < n; ++c) {
                a[r * n + c] -= factor * a[col * n + c];
                inv[r * n + c] -= factor * inv[col * n + c];
            }
        }
    }
    return true;
}

[[nodiscard]] bool solve_square_right(const std::vector<double>& left,
                                       const std::vector<double>& right,
                                       std::size_t rows,
                                       std::size_t n,
                                       std::vector<double>& out) {
    std::vector<double> inv;
    if (!invert_square(right, n, inv)) return false;
    out.assign(rows * n, 0.0);
    for (std::size_t r = 0; r < rows; ++r) {
        for (std::size_t c = 0; c < n; ++c) {
            double s = 0.0;
            for (std::size_t k = 0; k < n; ++k) {
                s += left[r * n + k] * inv[k * n + c];
            }
            out[r * n + c] = s;
        }
    }
    return true;
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

// Right singular loading used by the vendored OnPLS PCA implementation.
[[nodiscard]] bool pca_loading_nipals(const std::vector<double>& X,
                                      std::size_t rows,
                                      std::size_t cols,
                                      std::vector<double>& loading,
                                      int max_iter = 200,
                                      double eps = 1e-6) {
    loading.assign(cols, 0.0);
    if (rows == 0 || cols == 0) return false;
    std::size_t best_col = 0;
    double best_ss = -1.0;
    for (std::size_t c = 0; c < cols; ++c) {
        double ss = 0.0;
        for (std::size_t r = 0; r < rows; ++r) {
            const double v = X[r * cols + c];
            ss += v * v;
        }
        if (ss > best_ss) {
            best_ss = ss;
            best_col = c;
        }
    }
    if (best_ss < kEps) return false;
    std::vector<double> t(rows, 0.0);
    for (std::size_t r = 0; r < rows; ++r) t[r] = X[r * cols + best_col];
    for (std::size_t c = 0; c < cols; ++c) {
        for (std::size_t r = 0; r < rows; ++r) {
            loading[c] += X[r * cols + c] * t[r];
        }
    }
    double loading_norm = std::sqrt(squared_norm(loading));
    if (loading_norm < kEps) return false;
    for (double& v : loading) v /= loading_norm;
    for (int iter = 0; iter < max_iter; ++iter) {
        std::fill(t.begin(), t.end(), 0.0);
        for (std::size_t r = 0; r < rows; ++r) {
            for (std::size_t c = 0; c < cols; ++c) {
                t[r] += X[r * cols + c] * loading[c];
            }
        }
        std::vector<double> next(cols, 0.0);
        for (std::size_t c = 0; c < cols; ++c) {
            for (std::size_t r = 0; r < rows; ++r) {
                next[c] += X[r * cols + c] * t[r];
            }
        }
        const double norm = std::sqrt(squared_norm(next));
        if (norm < kEps) return false;
        for (double& v : next) v /= norm;
        double diff = 0.0;
        for (std::size_t c = 0; c < cols; ++c) {
            const double d = next[c] - loading[c];
            diff += d * d;
        }
        loading = std::move(next);
        if (diff < eps * eps) break;
    }
    return true;
}

void normalise_columns(std::vector<double>& M, std::size_t rows,
                       std::size_t cols) {
    for (std::size_t c = 0; c < cols; ++c) {
        double ss = 0.0;
        for (std::size_t r = 0; r < rows; ++r) {
            const double v = M[r * cols + c];
            ss += v * v;
        }
        const double norm = std::sqrt(ss);
        if (norm < kEps) continue;
        for (std::size_t r = 0; r < rows; ++r) M[r * cols + c] /= norm;
    }
}

void component_combos(const std::vector<std::int32_t>& comps,
                      std::int32_t max_comps,
                      std::vector<std::vector<std::int32_t>>& comp_list,
                      std::vector<std::int32_t>& change_block) {
    const std::size_t n_blocks = comps.size();
    std::vector<std::int32_t> current(n_blocks, 0);
    const std::size_t total = n_blocks == 0 ? 0 : n_blocks - 1;
    while (true) {
        std::int32_t sum = 0;
        for (const auto v : current) sum += v;
        if (sum <= max_comps) comp_list.push_back(current);

        std::size_t pos = total;
        while (true) {
            ++current[pos];
            if (current[pos] <= comps[pos]) break;
            current[pos] = 0;
            if (pos == 0) {
                change_block.assign(comp_list.size(),
                                    static_cast<std::int32_t>(n_blocks));
                for (std::size_t i = 1; i < comp_list.size(); ++i) {
                    for (std::size_t b = 0; b < n_blocks; ++b) {
                        if (comp_list[i][b] > comp_list[i - 1][b]) {
                            change_block[i] = static_cast<std::int32_t>(b + 1);
                            break;
                        }
                    }
                }
                return;
            }
            --pos;
        }
    }
}

void kernel_tcrossprod(const std::vector<double>& X,
                       std::size_t n,
                       std::size_t p,
                       std::vector<double>& C) {
    C.assign(n * n, 0.0);
    for (std::size_t i = 0; i < n; ++i) {
        for (std::size_t j = 0; j < n; ++j) {
            double s = 0.0;
            for (std::size_t f = 0; f < p; ++f) {
                s += X[i * p + f] * X[j * p + f];
            }
            C[i * n + j] = s;
        }
    }
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

    std::vector<double> Y_raw;
    p4a_status_t status = copy_matrix(ctx, Y, "Y", Y_raw);
    if (status != P4A_OK) return status;
    const std::size_t n = static_cast<std::size_t>(n_rows);
    const std::size_t q = static_cast<std::size_t>(Y.cols);
    column_means(Y_raw, n, q, out.y_mean);

    std::vector<double> Y_centered = Y_raw;
    subtract_means(Y_centered, n, q, out.y_mean);

    const std::int32_t total_requested_components = std::accumulate(
        n_components_per_block.begin(), n_components_per_block.end(),
        std::int32_t{0});
    const std::int32_t max_components =
        std::min<std::int32_t>(20, std::max<std::int32_t>(
                                       1, total_requested_components));
    std::vector<std::vector<std::int32_t>> comp_list;
    std::vector<std::int32_t> change_block;
    component_combos(n_components_per_block, max_components,
                     comp_list, change_block);
    if (comp_list.empty()) {
        ctx.set_error("SO-PLS component grid is empty");
        return P4A_ERR_INVALID_ARGUMENT;
    }

    std::vector<std::vector<double>> kernels(X_blocks.size());
    for (std::size_t block = 0; block < X_blocks.size(); ++block) {
        const p4a_matrix_view_t& Xv = X_blocks[block];
        std::vector<double> X_buf;
        status = copy_matrix(ctx, Xv, "X_block", X_buf);
        if (status != P4A_OK) return status;
        const std::size_t p = static_cast<std::size_t>(Xv.cols);
        std::vector<double> x_mean;
        column_means(X_buf, n, p, x_mean);
        subtract_means(X_buf, n, p, x_mean);
        kernel_tcrossprod(X_buf, n, p, kernels[block]);
    }

    const std::size_t total_components = comp_list.size();
    std::vector<double> Ry(n * total_components, 0.0);
    std::vector<double> T(n * total_components, 0.0);
    std::vector<double> Q(q * total_components, 0.0);
    std::vector<double> Ry_curr(n * static_cast<std::size_t>(max_components), 0.0);
    std::vector<double> T_curr(n * static_cast<std::size_t>(max_components), 0.0);
    std::vector<double> Q_curr(q * static_cast<std::size_t>(max_components), 0.0);
    std::vector<std::vector<double>> Y_currB(X_blocks.size(), Y_centered);

    for (std::size_t comp = 1; comp < total_components; ++comp) {
        const std::size_t cb = static_cast<std::size_t>(change_block[comp] - 1);
        const std::size_t comp_curr = static_cast<std::size_t>(
            std::accumulate(comp_list[comp].begin(), comp_list[comp].end(),
                            std::int32_t{0}));
        std::vector<double> Y_curr = Y_currB[cb];
        std::vector<double> t(n * q, 0.0);
        for (std::size_t i = 0; i < n; ++i) {
            for (std::size_t target = 0; target < q; ++target) {
                double s = 0.0;
                for (std::size_t j = 0; j < n; ++j) {
                    s += kernels[cb][i * n + j] * Y_curr[j * q + target];
                }
                t[i * q + target] = s;
            }
        }
        std::vector<double> t_vec(n, 0.0);
        std::vector<double> ry(n, 0.0);
        if (q > 1) {
            std::vector<double> cp(q * q, 0.0);
            for (std::size_t a = 0; a < q; ++a) {
                for (std::size_t b = 0; b < q; ++b) {
                    double s = 0.0;
                    for (std::size_t i = 0; i < n; ++i) {
                        s += Y_curr[i * q + a] * t[i * q + b];
                    }
                    cp[b * q + a] = s;  // transpose for right singular vector
                }
            }
            std::vector<double> w;
            if (!dominant_direction(cp, q, q, w)) continue;
            for (std::size_t i = 0; i < n; ++i) {
                for (std::size_t target = 0; target < q; ++target) {
                    t_vec[i] += t[i * q + target] * w[target];
                    ry[i] += Y_curr[i * q + target] * w[target];
                }
            }
        } else {
            for (std::size_t i = 0; i < n; ++i) {
                t_vec[i] = t[i];
                ry[i] = Y_curr[i];
            }
        }
        if (comp_curr > 1) {
            for (std::size_t prev = 0; prev < comp_curr - 1; ++prev) {
                double proj = 0.0;
                for (std::size_t i = 0; i < n; ++i) {
                    proj += T_curr[i * static_cast<std::size_t>(max_components) + prev] *
                            t_vec[i];
                }
                for (std::size_t i = 0; i < n; ++i) {
                    t_vec[i] -= T_curr[i * static_cast<std::size_t>(max_components) + prev] *
                                proj;
                }
            }
        }
        const double t_norm = std::sqrt(squared_norm(t_vec));
        if (t_norm < kEps) continue;
        for (double& v : t_vec) v /= t_norm;

        std::vector<double> q_load(q, 0.0);
        for (std::size_t target = 0; target < q; ++target) {
            for (std::size_t i = 0; i < n; ++i) {
                q_load[target] += t_vec[i] * Y_curr[i * q + target];
            }
        }
        for (std::size_t i = 0; i < n; ++i) {
            for (std::size_t target = 0; target < q; ++target) {
                Y_curr[i * q + target] -= t_vec[i] * q_load[target];
            }
        }
        for (std::size_t b = cb; b < X_blocks.size(); ++b) {
            Y_currB[b] = Y_curr;
        }
        for (std::size_t i = 0; i < n; ++i) {
            T_curr[i * static_cast<std::size_t>(max_components) +
                   (comp_curr - 1)] = t_vec[i];
            Ry_curr[i * static_cast<std::size_t>(max_components) +
                    (comp_curr - 1)] = ry[i];
            T[i * total_components + comp] = t_vec[i];
            Ry[i * total_components + comp] = ry[i];
        }
        for (std::size_t target = 0; target < q; ++target) {
            Q_curr[target * static_cast<std::size_t>(max_components) +
                   (comp_curr - 1)] = q_load[target];
            Q[target * total_components + comp] = q_load[target];
        }
    }

    std::vector<std::size_t> hits;
    hits.reserve(static_cast<std::size_t>(
        std::accumulate(n_components_per_block.begin(),
                        n_components_per_block.end(), std::int32_t{0})));
    for (std::size_t b = 0; b < X_blocks.size(); ++b) {
        for (std::int32_t c = 1; c <= n_components_per_block[b]; ++c) {
            std::vector<std::int32_t> path(X_blocks.size(), 0);
            for (std::size_t prev = 0; prev < b; ++prev) {
                path[prev] = n_components_per_block[prev];
            }
            path[b] = c;
            auto it = std::find(comp_list.begin(), comp_list.end(), path);
            if (it != comp_list.end()) {
                hits.push_back(static_cast<std::size_t>(
                    std::distance(comp_list.begin(), it)));
            }
        }
    }
    const std::size_t k = hits.size();
    std::vector<double> Cr(n * k, 0.0);
    for (std::size_t h = 0; h < k; ++h) {
        const std::size_t comp = hits[h];
        for (std::size_t b = 0; b < X_blocks.size(); ++b) {
            if (n_components_per_block[b] <= 0) continue;
            for (std::size_t i = 0; i < n; ++i) {
                double s = 0.0;
                for (std::size_t j = 0; j < n; ++j) {
                    s += kernels[b][i * n + j] * Ry[j * total_components + comp];
                }
                Cr[i * k + h] += s;
            }
        }
    }
    std::vector<double> T_sel(n * k, 0.0);
    for (std::size_t i = 0; i < n; ++i) {
        for (std::size_t h = 0; h < k; ++h) {
            T_sel[i * k + h] = T[i * total_components + hits[h]];
        }
    }
    std::vector<double> right(k * k, 0.0);
    for (std::size_t a = 0; a < k; ++a) {
        for (std::size_t b = 0; b < k; ++b) {
            double s = 0.0;
            for (std::size_t i = 0; i < n; ++i) s += T_sel[i * k + a] * Cr[i * k + b];
            right[a * k + b] = s;
        }
    }
    std::vector<double> no_Q;
    if (!solve_square_right(Cr, right, n, k, no_Q)) {
        ctx.set_error("SO-PLS prediction system is singular");
        return P4A_ERR_INVALID_ARGUMENT;
    }

    out.n_blocks = static_cast<std::int32_t>(X_blocks.size());
    out.n_components_per_block = n_components_per_block;
    out.predictions.assign(n * q, 0.0);
    for (std::size_t i = 0; i < n; ++i) {
        for (std::size_t target = 0; target < q; ++target) {
            double yhat = out.y_mean[target];
            for (std::size_t h = 0; h < k; ++h) {
                yhat += no_Q[i * k + h] * Q[target * total_components + hits[h]];
            }
            out.predictions[i * q + target] = yhat;
        }
    }
    out.block_coefficients.assign(X_blocks.size(), {});
    for (std::size_t b = 0; b < X_blocks.size(); ++b) {
        out.block_coefficients[b].assign(
            static_cast<std::size_t>(X_blocks[b].cols) * q, 0.0);
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

    // Match the vendored Python OnPLS adapter used as executable reference:
    // raw blocks, one orthogonal filtering pass, then nPLS joint components.
    std::vector<std::vector<double>> precomputed_w(n_blocks);
    for (std::size_t i = 0; i < n_blocks; ++i) {
        const std::size_t pi = p_block[i];
        precomputed_w[i].assign(pi * static_cast<std::size_t>(n_joint), 0.0);
        std::size_t written = 0;
        for (std::size_t j = 0; j < n_blocks && written < static_cast<std::size_t>(n_joint); ++j) {
            if (i == j) continue;
            const std::size_t pj = p_block[j];
            std::vector<double> cross(pj * pi, 0.0);
            for (std::size_t rj = 0; rj < pj; ++rj) {
                for (std::size_t ci = 0; ci < pi; ++ci) {
                    double s = 0.0;
                    for (std::size_t row = 0; row < n; ++row) {
                        s += centered[j][row * pj + rj] *
                             centered[i][row * pi + ci];
                    }
                    cross[rj * pi + ci] = s;
                }
            }
            std::vector<double> Xp = cross;
            for (std::int32_t comp = 0;
                 comp < n_joint && written < static_cast<std::size_t>(n_joint);
                 ++comp) {
                std::vector<double> load;
                if (!pca_loading_nipals(Xp, pj, pi, load)) break;
                for (std::size_t f = 0; f < pi; ++f) {
                    precomputed_w[i][f * static_cast<std::size_t>(n_joint) + written] =
                        load[f];
                }
                std::vector<double> score(pj, 0.0);
                for (std::size_t r = 0; r < pj; ++r) {
                    for (std::size_t f = 0; f < pi; ++f) {
                        score[r] += Xp[r * pi + f] * load[f];
                    }
                }
                const double tt = squared_norm(score);
                if (tt < kEps) break;
                std::vector<double> pload(pi, 0.0);
                for (std::size_t f = 0; f < pi; ++f) {
                    for (std::size_t r = 0; r < pj; ++r) {
                        pload[f] += Xp[r * pi + f] * score[r];
                    }
                    pload[f] /= tt;
                }
                for (std::size_t r = 0; r < pj; ++r) {
                    for (std::size_t f = 0; f < pi; ++f) {
                        Xp[r * pi + f] -= score[r] * pload[f];
                    }
                }
                ++written;
            }
        }
        normalise_columns(precomputed_w[i], pi, static_cast<std::size_t>(n_joint));
    }

    std::vector<std::vector<double>> filtered = centered;
    for (std::size_t b = 0; b < n_blocks; ++b) {
        for (std::int32_t oc = 0; oc < n_unique_per_block[b]; ++oc) {
            const std::size_t pb = p_block[b];
            std::vector<double> Ti(n * static_cast<std::size_t>(n_joint), 0.0);
            for (std::size_t row = 0; row < n; ++row) {
                for (std::int32_t c = 0; c < n_joint; ++c) {
                    double s = 0.0;
                    for (std::size_t f = 0; f < pb; ++f) {
                        s += filtered[b][row * pb + f] *
                             precomputed_w[b][f * static_cast<std::size_t>(n_joint) +
                                              static_cast<std::size_t>(c)];
                    }
                    Ti[row * static_cast<std::size_t>(n_joint) +
                       static_cast<std::size_t>(c)] = s;
                }
            }
            std::vector<double> Ei = filtered[b];
            for (std::size_t row = 0; row < n; ++row) {
                for (std::size_t f = 0; f < pb; ++f) {
                    double s = 0.0;
                    for (std::int32_t c = 0; c < n_joint; ++c) {
                        s += Ti[row * static_cast<std::size_t>(n_joint) +
                                static_cast<std::size_t>(c)] *
                             precomputed_w[b][f * static_cast<std::size_t>(n_joint) +
                                              static_cast<std::size_t>(c)];
                    }
                    Ei[row * pb + f] -= s;
                }
            }
            std::vector<double> Wortho(static_cast<std::size_t>(n_joint) * pb, 0.0);
            for (std::int32_t c = 0; c < n_joint; ++c) {
                for (std::size_t f = 0; f < pb; ++f) {
                    double s = 0.0;
                    for (std::size_t row = 0; row < n; ++row) {
                        s += Ti[row * static_cast<std::size_t>(n_joint) +
                                static_cast<std::size_t>(c)] *
                             Ei[row * pb + f];
                    }
                    Wortho[static_cast<std::size_t>(c) * pb + f] = s;
                }
            }
            std::vector<double> wortho;
            if (!pca_loading_nipals(Wortho, static_cast<std::size_t>(n_joint),
                                    pb, wortho)) break;
            std::vector<double> tortho(n, 0.0);
            for (std::size_t row = 0; row < n; ++row) {
                for (std::size_t f = 0; f < pb; ++f) {
                    tortho[row] += filtered[b][row * pb + f] * wortho[f];
                }
            }
            const double tt = squared_norm(tortho);
            if (tt < kEps) break;
            std::vector<double> portho(pb, 0.0);
            for (std::size_t f = 0; f < pb; ++f) {
                for (std::size_t row = 0; row < n; ++row) {
                    portho[f] += filtered[b][row * pb + f] * tortho[row];
                }
                portho[f] /= tt;
                out.unique_loadings_per_block[b][
                    f * static_cast<std::size_t>(n_unique_per_block[b]) +
                    static_cast<std::size_t>(oc)] = wortho[f];
            }
            for (std::size_t row = 0; row < n; ++row) {
                for (std::size_t f = 0; f < pb; ++f) {
                    filtered[b][row * pb + f] -= tortho[row] * portho[f];
                }
            }
        }
    }

    for (std::int32_t comp = 0; comp < n_joint; ++comp) {
        std::vector<std::vector<double>> w(n_blocks);
        for (std::size_t b = 0; b < n_blocks; ++b) {
            w[b].assign(p_block[b], 1.0 / std::sqrt(static_cast<double>(p_block[b])));
        }
        auto npls_objective = [&filtered, &p_block, n, n_blocks](
                                  const std::vector<std::vector<double>>& weights) {
            std::vector<std::vector<double>> scores(n_blocks);
            for (std::size_t b = 0; b < n_blocks; ++b) {
                scores[b].assign(n, 0.0);
                for (std::size_t row = 0; row < n; ++row) {
                    for (std::size_t f = 0; f < p_block[b]; ++f) {
                        scores[b][row] += filtered[b][row * p_block[b] + f] *
                                          weights[b][f];
                    }
                }
            }
            double value = 0.0;
            for (std::size_t i = 0; i < n_blocks; ++i) {
                for (std::size_t j = 0; j < n_blocks; ++j) {
                    if (i == j) continue;
                    value += dot(scores[i], scores[j]);
                }
            }
            return value;
        };
        double previous_objective = npls_objective(w);
        for (int iter = 0; iter < 1000; ++iter) {
            std::vector<std::vector<double>> next = w;
            for (std::size_t i = 0; i < n_blocks; ++i) {
                std::fill(next[i].begin(), next[i].end(), 0.0);
                const std::size_t pi = p_block[i];
                for (std::size_t j = 0; j < n_blocks; ++j) {
                    if (i == j) continue;
                    const std::size_t pj = p_block[j];
                    std::vector<double> xjw(n, 0.0);
                    for (std::size_t row = 0; row < n; ++row) {
                        for (std::size_t f = 0; f < pj; ++f) {
                            xjw[row] += filtered[j][row * pj + f] * w[j][f];
                        }
                    }
                    for (std::size_t f = 0; f < pi; ++f) {
                        for (std::size_t row = 0; row < n; ++row) {
                            next[i][f] += filtered[i][row * pi + f] * xjw[row];
                        }
                    }
                }
                const double norm = std::sqrt(squared_norm(next[i]));
                if (norm > kEps) {
                    for (double& v : next[i]) v /= norm;
                }
            }
            w = std::move(next);
            const double objective = npls_objective(w);
            const double err = objective - previous_objective;
            previous_objective = objective;
            if (std::fabs(err) < 5e-8) break;
        }
        for (std::size_t b = 0; b < n_blocks; ++b) {
            const std::size_t pb = p_block[b];
            std::vector<double> t(n, 0.0);
            for (std::size_t row = 0; row < n; ++row) {
                for (std::size_t f = 0; f < pb; ++f) {
                    t[row] += filtered[b][row * pb + f] * w[b][f];
                }
            }
            const double tt = squared_norm(t);
            if (tt < kEps) continue;
            std::vector<double> p_load(pb, 0.0);
            for (std::size_t f = 0; f < pb; ++f) {
                for (std::size_t row = 0; row < n; ++row) {
                    p_load[f] += filtered[b][row * pb + f] * t[row];
                }
                p_load[f] /= tt;
                out.joint_loadings_per_block[b][
                    f * static_cast<std::size_t>(n_joint) +
                    static_cast<std::size_t>(comp)] = p_load[f];
            }
            for (std::size_t row = 0; row < n; ++row) {
                out.joint_scores_per_block[b][
                    row * static_cast<std::size_t>(n_joint) +
                    static_cast<std::size_t>(comp)] = t[row];
                for (std::size_t f = 0; f < pb; ++f) {
                    filtered[b][row * pb + f] -= t[row] * p_load[f];
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
