// SPDX-License-Identifier: CECILL-2.1

#include "core/multiblock_extensions.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <new>
#include <vector>

#include "core/common/matrix_view.hpp"
#include "core/common/svd.h"
#include "core/model.hpp"

namespace n4m::core {

namespace {

constexpr double kEps = 1e-12;

[[maybe_unused]] inline std::size_t idx(std::size_t row, std::size_t cols, std::size_t col) noexcept {
    return row * cols + col;
}

[[nodiscard]] n4m_status_t copy_matrix(::n4m::core::Context& ctx,
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
[[nodiscard]] n4m_status_t nipals_pls(std::vector<double>& xk,
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
    return N4M_OK;
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

// ---------- OmicsPLS::o2m helpers ---------------------------------------

// Invert a square (k × k) matrix in place via Gauss-Jordan. Returns false
// if the matrix is (numerically) singular.
[[nodiscard]] bool invert_square_gj(std::vector<double>& a, std::size_t k,
                                     std::vector<double>& inv) {
    inv.assign(k * k, 0.0);
    for (std::size_t i = 0; i < k; ++i) inv[i * k + i] = 1.0;
    for (std::size_t col = 0; col < k; ++col) {
        std::size_t pivot = col;
        double pivot_val = std::fabs(a[col * k + col]);
        for (std::size_t r = col + 1; r < k; ++r) {
            const double v = std::fabs(a[r * k + col]);
            if (v > pivot_val) { pivot = r; pivot_val = v; }
        }
        if (pivot_val < kEps) return false;
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
    return true;
}

// General row-major matmul: C (m × n) = A (m × k) @ B (k × n).
void matmul(const std::vector<double>& A, const std::vector<double>& B,
            std::size_t m, std::size_t k, std::size_t n,
            std::vector<double>& C) {
    C.assign(m * n, 0.0);
    for (std::size_t i = 0; i < m; ++i) {
        for (std::size_t l = 0; l < k; ++l) {
            const double a = A[i * k + l];
            if (a == 0.0) continue;
            for (std::size_t j = 0; j < n; ++j) {
                C[i * n + j] += a * B[l * n + j];
            }
        }
    }
}

// matmul_TN: C (m × n) = A^T (m × k) @ B (k × n). A is stored as (k × m).
void matmul_TN(const std::vector<double>& A, const std::vector<double>& B,
               std::size_t m, std::size_t k, std::size_t n,
               std::vector<double>& C) {
    C.assign(m * n, 0.0);
    for (std::size_t i = 0; i < m; ++i) {
        for (std::size_t l = 0; l < k; ++l) {
            const double a = A[l * m + i];   // A^T[i, l] = A[l, i]
            if (a == 0.0) continue;
            for (std::size_t j = 0; j < n; ++j) {
                C[i * n + j] += a * B[l * n + j];
            }
        }
    }
}

// matmul_NT: C (m × n) = A (m × k) @ B^T (k × n). B is stored as (n × k).
void matmul_NT(const std::vector<double>& A, const std::vector<double>& B,
               std::size_t m, std::size_t k, std::size_t n,
               std::vector<double>& C) {
    C.assign(m * n, 0.0);
    for (std::size_t i = 0; i < m; ++i) {
        for (std::size_t l = 0; l < k; ++l) {
            const double a = A[i * k + l];
            if (a == 0.0) continue;
            for (std::size_t j = 0; j < n; ++j) {
                C[i * n + j] += a * B[j * k + l];   // B^T[l, j] = B[j, l]
            }
        }
    }
}

// Copy the leading `k` columns of a (rows × full_k) row-major matrix into
// `out` shaped (rows × k).
void copy_leading_cols(const std::vector<double>& src, std::size_t rows,
                       std::size_t full_k, std::size_t k,
                       std::vector<double>& out) {
    out.assign(rows * k, 0.0);
    for (std::size_t r = 0; r < rows; ++r) {
        for (std::size_t c = 0; c < k; ++c) {
            out[r * k + c] = src[r * full_k + c];
        }
    }
}

// Transpose helper: B (cols × rows) = A^T where A is (rows × cols).
void transpose(const std::vector<double>& A, std::size_t rows,
               std::size_t cols, std::vector<double>& B) {
    B.assign(cols * rows, 0.0);
    for (std::size_t r = 0; r < rows; ++r) {
        for (std::size_t c = 0; c < cols; ++c) {
            B[c * rows + r] = A[r * cols + c];
        }
    }
}

// Compute OmicsPLS::o2m_stripped + predict.o2m(XorY='X'), collapsed into a
// single (p × q) coefficient matrix `G` such that `X_new @ G` reproduces
// the OmicsPLS prediction. X / Y must be the raw uncentered matrices (R
// `OmicsPLS::o2m` does not center internally).
//
// Sign convention: the W and C blocks come from SVD of (Y^T X). The predict
// formula `(I - W_Yosc P_Yosc.T) W B_T C^T` is invariant under column-wise
// sign flips of W, C, and W_Yosc, so the LAPACK-vs-n4m_svd_compact sign
// difference is benign.
[[nodiscard]] n4m_status_t omicspls_o2m_coefficients(
    ::n4m::core::Context& ctx,
    const std::vector<double>& X_in, const std::vector<double>& Y_in,
    std::size_t n, std::size_t p, std::size_t q,
    std::size_t n_pred, std::size_t nx, std::size_t ny,
    std::vector<double>& coefficients) {
    // Working copies (mutated by orthogonal deflation).
    std::vector<double> X = X_in;
    std::vector<double> Y = Y_in;

    std::vector<double> W_Yosc;   // p × nx
    std::vector<double> P_Yosc;   // p × nx

    // Reusable SVD scratch.
    auto svd = [&ctx](const std::vector<double>& A_in,
                       std::size_t rows, std::size_t cols,
                       std::vector<double>& U_out, std::vector<double>& S_out,
                       std::vector<double>& Vt_out) -> n4m_status_t {
        if (rows == 0 || cols == 0) {
            ctx.set_error("o2pls: empty matrix in SVD");
            return N4M_ERR_INVALID_ARGUMENT;
        }
        std::vector<double> work = A_in;
        const std::size_t k = std::min(rows, cols);
        U_out.assign(rows * k, 0.0);
        S_out.assign(k, 0.0);
        Vt_out.assign(k * cols, 0.0);
        const n4m_status_t st = n4m_svd_compact(work.data(),
            static_cast<std::int64_t>(rows),
            static_cast<std::int64_t>(cols),
            U_out.data(), S_out.data(), Vt_out.data());
        if (st != N4M_OK) ctx.set_error("o2pls: SVD failed");
        return st;
    };

    // Step 1: orthogonal-component removal using the bootstrap SVD.
    // SVD(Y^T X) with n2 = n_pred + max(nx, ny).
    std::vector<double> C_full;          // q × n2
    std::vector<double> W_full;          // p × n2
    if (nx + ny > 0) {
        const std::size_t n2 = n_pred + std::max(nx, ny);
        if (n2 > std::min(q, p)) {
            ctx.set_error("o2pls: n + max(nx, ny) exceeds rank of (Y^T X)");
            return N4M_ERR_INVALID_ARGUMENT;
        }
        // Build Y^T X (q × p).
        std::vector<double> YtX(q * p, 0.0);
        for (std::size_t i = 0; i < n; ++i) {
            for (std::size_t r = 0; r < q; ++r) {
                const double yir = Y[i * q + r];
                if (yir == 0.0) continue;
                for (std::size_t c = 0; c < p; ++c) {
                    YtX[r * p + c] += yir * X[i * p + c];
                }
            }
        }
        std::vector<double> U_svd, S_svd, Vt_svd;
        n4m_status_t st = svd(YtX, q, p, U_svd, S_svd, Vt_svd);
        if (st != N4M_OK) return st;
        // C = U_svd[:, :n2]   (q × n2)
        const std::size_t k_full = std::min(q, p);
        copy_leading_cols(U_svd, q, k_full, n2, C_full);
        // W = Vt_svd^T[:, :n2]  (p × n2)
        // Vt_svd is (k_full × p); transpose then take leading n2 cols.
        std::vector<double> V_full;     // p × k_full
        transpose(Vt_svd, k_full, p, V_full);
        copy_leading_cols(V_full, p, k_full, n2, W_full);

        // Tt = X W  (n × n2)
        std::vector<double> Tt;
        matmul(X, W_full, n, p, n2, Tt);

        if (nx > 0) {
            // E_XY = X - Tt W^T  (n × p)
            std::vector<double> Tt_Wt;
            matmul_NT(Tt, W_full, n, n2, p, Tt_Wt);   // Tt @ W^T
            std::vector<double> E_XY(n * p, 0.0);
            for (std::size_t i = 0; i < n * p; ++i) {
                E_XY[i] = X[i] - Tt_Wt[i];
            }
            // svd(E_XY^T Tt) — (p × n2)
            std::vector<double> EXY_T_Tt(p * n2, 0.0);
            for (std::size_t i = 0; i < n; ++i) {
                for (std::size_t c = 0; c < p; ++c) {
                    const double exyic = E_XY[i * p + c];
                    if (exyic == 0.0) continue;
                    for (std::size_t j = 0; j < n2; ++j) {
                        EXY_T_Tt[c * n2 + j] += exyic * Tt[i * n2 + j];
                    }
                }
            }
            std::vector<double> Uo, So, Vto;
            st = svd(EXY_T_Tt, p, n2, Uo, So, Vto);
            if (st != N4M_OK) return st;
            // W_Yosc = Uo[:, :nx]  (p × nx)
            const std::size_t k_o = std::min(p, n2);
            copy_leading_cols(Uo, p, k_o, nx, W_Yosc);
            // T_Yosc = X W_Yosc  (n × nx)
            std::vector<double> T_Yosc;
            matmul(X, W_Yosc, n, p, nx, T_Yosc);
            // M = T_Yosc^T T_Yosc  (nx × nx)
            std::vector<double> M(nx * nx, 0.0);
            for (std::size_t i = 0; i < n; ++i) {
                for (std::size_t r = 0; r < nx; ++r) {
                    const double tir = T_Yosc[i * nx + r];
                    if (tir == 0.0) continue;
                    for (std::size_t c = 0; c < nx; ++c) {
                        M[r * nx + c] += tir * T_Yosc[i * nx + c];
                    }
                }
            }
            std::vector<double> M_inv, M_copy = M;
            if (!invert_square_gj(M_copy, nx, M_inv)) {
                ctx.set_error("o2pls: T_Yosc^T T_Yosc is singular");
                return N4M_ERR_INVALID_ARGUMENT;
            }
            // P_Yosc = X^T T_Yosc @ inv(M)   (p × nx)
            std::vector<double> XtT(p * nx, 0.0);
            for (std::size_t i = 0; i < n; ++i) {
                for (std::size_t c = 0; c < p; ++c) {
                    const double xic = X[i * p + c];
                    if (xic == 0.0) continue;
                    for (std::size_t r = 0; r < nx; ++r) {
                        XtT[c * nx + r] += xic * T_Yosc[i * nx + r];
                    }
                }
            }
            matmul(XtT, M_inv, p, nx, nx, P_Yosc);
            // X = X - T_Yosc P_Yosc^T
            std::vector<double> def;
            matmul_NT(T_Yosc, P_Yosc, n, nx, p, def);
            for (std::size_t i = 0; i < n * p; ++i) X[i] -= def[i];
        }

        if (ny > 0) {
            // U = Y C  (n × n2)
            std::vector<double> U_n;
            matmul(Y, C_full, n, q, n2, U_n);
            // F_XY = Y - U C^T  (n × q)
            std::vector<double> U_Ct;
            matmul_NT(U_n, C_full, n, n2, q, U_Ct);
            std::vector<double> F_XY(n * q, 0.0);
            for (std::size_t i = 0; i < n * q; ++i) {
                F_XY[i] = Y[i] - U_Ct[i];
            }
            // svd(F_XY^T U)  (q × n2)
            std::vector<double> FXY_T_U(q * n2, 0.0);
            for (std::size_t i = 0; i < n; ++i) {
                for (std::size_t r = 0; r < q; ++r) {
                    const double fir = F_XY[i * q + r];
                    if (fir == 0.0) continue;
                    for (std::size_t c = 0; c < n2; ++c) {
                        FXY_T_U[r * n2 + c] += fir * U_n[i * n2 + c];
                    }
                }
            }
            std::vector<double> Uy, Sy, Vty;
            n4m_status_t st2 = svd(FXY_T_U, q, n2, Uy, Sy, Vty);
            if (st2 != N4M_OK) return st2;
            const std::size_t k_y = std::min(q, n2);
            std::vector<double> C_Xosc;   // q × ny
            copy_leading_cols(Uy, q, k_y, ny, C_Xosc);
            std::vector<double> U_Xosc;
            matmul(Y, C_Xosc, n, q, ny, U_Xosc);
            std::vector<double> M(ny * ny, 0.0);
            for (std::size_t i = 0; i < n; ++i) {
                for (std::size_t r = 0; r < ny; ++r) {
                    const double uir = U_Xosc[i * ny + r];
                    if (uir == 0.0) continue;
                    for (std::size_t c = 0; c < ny; ++c) {
                        M[r * ny + c] += uir * U_Xosc[i * ny + c];
                    }
                }
            }
            std::vector<double> M_inv, M_copy = M;
            if (!invert_square_gj(M_copy, ny, M_inv)) {
                ctx.set_error("o2pls: U_Xosc^T U_Xosc is singular");
                return N4M_ERR_INVALID_ARGUMENT;
            }
            std::vector<double> YtU(q * ny, 0.0);
            for (std::size_t i = 0; i < n; ++i) {
                for (std::size_t c = 0; c < q; ++c) {
                    const double yic = Y[i * q + c];
                    if (yic == 0.0) continue;
                    for (std::size_t r = 0; r < ny; ++r) {
                        YtU[c * ny + r] += yic * U_Xosc[i * ny + r];
                    }
                }
            }
            std::vector<double> P_Xosc;     // q × ny
            matmul(YtU, M_inv, q, ny, ny, P_Xosc);
            std::vector<double> def;
            matmul_NT(U_Xosc, P_Xosc, n, ny, q, def);
            for (std::size_t i = 0; i < n * q; ++i) Y[i] -= def[i];
        }
    }

    // Step 2: final SVD on the deflated (Y^T X), keep the first n_pred
    // components.
    std::vector<double> YtX_final(q * p, 0.0);
    for (std::size_t i = 0; i < n; ++i) {
        for (std::size_t r = 0; r < q; ++r) {
            const double yir = Y[i * q + r];
            if (yir == 0.0) continue;
            for (std::size_t c = 0; c < p; ++c) {
                YtX_final[r * p + c] += yir * X[i * p + c];
            }
        }
    }
    std::vector<double> U_fin, S_fin, Vt_fin;
    n4m_status_t st = svd(YtX_final, q, p, U_fin, S_fin, Vt_fin);
    if (st != N4M_OK) return st;
    const std::size_t k_fin = std::min(q, p);
    std::vector<double> C(q * n_pred);          // q × n_pred
    copy_leading_cols(U_fin, q, k_fin, n_pred, C);
    std::vector<double> V_fin;                  // p × k_fin
    transpose(Vt_fin, k_fin, p, V_fin);
    std::vector<double> W(p * n_pred);          // p × n_pred
    copy_leading_cols(V_fin, p, k_fin, n_pred, W);

    // Tt = X W (deflated X)
    std::vector<double> Tt;
    matmul(X, W, n, p, n_pred, Tt);
    // U = Y C (deflated Y)
    std::vector<double> U;
    matmul(Y, C, n, q, n_pred, U);
    // B_T = inv(Tt^T Tt) Tt^T U  (n_pred × n_pred)
    std::vector<double> TtTt(n_pred * n_pred, 0.0);
    for (std::size_t i = 0; i < n; ++i) {
        for (std::size_t r = 0; r < n_pred; ++r) {
            const double tir = Tt[i * n_pred + r];
            if (tir == 0.0) continue;
            for (std::size_t c = 0; c < n_pred; ++c) {
                TtTt[r * n_pred + c] += tir * Tt[i * n_pred + c];
            }
        }
    }
    std::vector<double> TtTt_inv, TtTt_copy = TtTt;
    if (!invert_square_gj(TtTt_copy, n_pred, TtTt_inv)) {
        ctx.set_error("o2pls: predictive Tt^T Tt is singular");
        return N4M_ERR_INVALID_ARGUMENT;
    }
    std::vector<double> TtU(n_pred * n_pred, 0.0);
    for (std::size_t i = 0; i < n; ++i) {
        for (std::size_t r = 0; r < n_pred; ++r) {
            const double tir = Tt[i * n_pred + r];
            if (tir == 0.0) continue;
            for (std::size_t c = 0; c < n_pred; ++c) {
                TtU[r * n_pred + c] += tir * U[i * n_pred + c];
            }
        }
    }
    std::vector<double> B_T;
    matmul(TtTt_inv, TtU, n_pred, n_pred, n_pred, B_T);

    // Build coefficient matrix G (p × q) so that the OmicsPLS prediction
    // for a new row X_new (raw, uncentered) is X_new @ G.
    //
    //   G = (I - W_Yosc P_Yosc^T) W B_T C^T
    //     = M W B_T C^T,  where M = (I - W_Yosc P_Yosc^T)  (p × p)
    //
    // We avoid materialising the (p × p) projector and instead compute
    // sequentially:
    //   WBT  = W @ B_T        (p × n_pred)
    //   tmp  = M @ WBT        (p × n_pred)  = WBT - W_Yosc @ (P_Yosc^T WBT)
    //   G    = tmp @ C^T      (p × q)
    std::vector<double> WBT;
    matmul(W, B_T, p, n_pred, n_pred, WBT);
    std::vector<double> tmp = WBT;
    if (nx > 0) {
        // P_Yosc^T WBT  (nx × n_pred)
        std::vector<double> PtWBT(nx * n_pred, 0.0);
        for (std::size_t r = 0; r < nx; ++r) {
            for (std::size_t c = 0; c < p; ++c) {
                const double prc = P_Yosc[c * nx + r];  // P_Yosc is (p × nx)
                if (prc == 0.0) continue;
                for (std::size_t j = 0; j < n_pred; ++j) {
                    PtWBT[r * n_pred + j] += prc * WBT[c * n_pred + j];
                }
            }
        }
        std::vector<double> sub;
        matmul(W_Yosc, PtWBT, p, nx, n_pred, sub);
        for (std::size_t i = 0; i < p * n_pred; ++i) tmp[i] -= sub[i];
    }
    matmul_NT(tmp, C, p, n_pred, q, coefficients);  // G = tmp @ C^T
    return N4M_OK;
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

n4m_status_t fit_o2pls(Context& ctx,
                       const Config& cfg,
                       const n4m_matrix_view_t& X,
                       const n4m_matrix_view_t& Y,
                       std::int32_t n_predictive,
                       std::int32_t n_x_orthogonal,
                       std::int32_t n_y_orthogonal,
                       O2PlsResult& out) {
    out = O2PlsResult{};
    if (n_predictive < 1) {
        ctx.set_error("O2PLS needs at least one predictive component");
        return N4M_ERR_INVALID_ARGUMENT;
    }
    if (n_x_orthogonal < 0 || n_y_orthogonal < 0) {
        ctx.set_error("orthogonal counts must be >= 0");
        return N4M_ERR_INVALID_ARGUMENT;
    }
    if (X.rows != Y.rows) {
        ctx.set_error("X and Y must have the same row count");
        return N4M_ERR_SHAPE_MISMATCH;
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
        ctx.set_error("O2PLS requires at least 2 rows and 1 column on each side");
        return N4M_ERR_INVALID_ARGUMENT;
    }

    out.n_samples = static_cast<std::int64_t>(n);
    out.n_features_x = static_cast<std::int32_t>(p);
    out.n_features_y = static_cast<std::int32_t>(q);
    out.n_predictive = n_predictive;
    out.n_x_orthogonal = n_x_orthogonal;
    out.n_y_orthogonal = n_y_orthogonal;

    // Dispatch: default = OmicsPLS::o2m (joint-SVD), opt-in legacy =
    // peel-then-PLS (Trygg & Wold 2003 §3.2 power-iteration recipe). The
    // legacy path is selected by setting `cfg.solver = N4M_SOLVER_NIPALS`;
    // every other solver (SVD, default) takes the OmicsPLS branch.
    const bool legacy = (cfg.solver == N4M_SOLVER_NIPALS);

    if (!legacy) {
        // OmicsPLS::o2m matches R bit-for-bit on its native (uncentered)
        // inputs. Force-zero the saved means so the C-API
        // `predict_from_coefficients` path collapses to `X_new @ G`.
        out.x_mean.assign(p, 0.0);
        out.y_mean.assign(q, 0.0);
        status = omicspls_o2m_coefficients(
            ctx, X_buf, Y_buf, n, p, q,
            static_cast<std::size_t>(n_predictive),
            static_cast<std::size_t>(n_x_orthogonal),
            static_cast<std::size_t>(n_y_orthogonal),
            out.coefficients);
        if (status != N4M_OK) return status;
        // The OmicsPLS path collapses all loadings into a single (p × q)
        // coefficient matrix; leave the per-component fields empty (they
        // describe the legacy decomposition only).
        out.w_predictive.clear();
        out.c_predictive.clear();
        out.w_x_orthogonal.clear();
        out.c_y_orthogonal.clear();
        out.b_predictive.clear();
        ctx.clear_error();
        return N4M_OK;
    }

    // ---- Legacy peel-then-PLS path (cfg.solver == N4M_SOLVER_NIPALS) ----
    column_means(X_buf, n, p, out.x_mean);
    column_means(Y_buf, n, q, out.y_mean);
    subtract_means(X_buf, n, p, out.x_mean);
    subtract_means(Y_buf, n, q, out.y_mean);

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
    if (status != N4M_OK) return status;
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
    return N4M_OK;
}

// ---------- SO-PLS ------------------------------------------------------

n4m_status_t fit_so_pls(Context& ctx,
                        const Config& cfg,
                        const std::vector<n4m_matrix_view_t>& X_blocks,
                        const n4m_matrix_view_t& Y,
                        const std::vector<std::int32_t>& n_components_per_block,
                        SoPlsResult& out) {
    (void)cfg;
    out = SoPlsResult{};
    if (X_blocks.empty()) {
        ctx.set_error("SO-PLS requires at least one X block");
        return N4M_ERR_INVALID_ARGUMENT;
    }
    if (n_components_per_block.size() != X_blocks.size()) {
        ctx.set_error("n_components_per_block must have one entry per block");
        return N4M_ERR_INVALID_ARGUMENT;
    }
    const std::int64_t n_rows = Y.rows;
    for (const auto& X : X_blocks) {
        if (X.rows != n_rows) {
            ctx.set_error("all X blocks must have the same row count as Y");
            return N4M_ERR_SHAPE_MISMATCH;
        }
    }

    std::vector<double> Y_buf;
    n4m_status_t status = copy_matrix(ctx, Y, "Y", Y_buf);
    if (status != N4M_OK) return status;
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
        const n4m_matrix_view_t& Xv = X_blocks[block];
        std::vector<double> X_buf;
        status = copy_matrix(ctx, Xv, "X_block", X_buf);
        if (status != N4M_OK) return status;
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
        if (status != N4M_OK) return status;
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
    return N4M_OK;
}

// ---------- OnPLS -------------------------------------------------------
//
// Faithful port of OnPLS (Löfstedt & Trygg 2011) matching the canonical
// `tomlof/OnPLS` Python reference. The algorithm:
//
//   1. Center each block per column.
//   2. Compute the globally-joint loading bank gjW[i] per block: stack
//      NIPALS-PCA loadings of X_j' X_i over all j with pred_comp[i][j] >
//      0, then truncate to the smallest non-zero pred_comp[i][:] and
//      normalize columns.
//   3. Filter out per-block orthogonal (unique) components: at each step
//      compute T_i = X_i gjW_i, residual E_i = X_i - T_i gjW_i', and run
//      NIPALS-PCA(1) on (T_i' E_i). Then t_o = X_i w_o,
//      p_o = X_i' t_o / (t_o' t_o); X_i ← X_i - t_o p_o'.
//   4. Build joint predictive components: for k = 0..n_joint-1 run a
//      single-component multiblock nPLS (Gauss-Seidel iteration on the
//      sum-of-pairwise-scores objective), save W/T/P, deflate.
//   5. Per block w accumulate X̂_w from the predictive components. For
//      each k, stack the connected-block scores Tiks = [T_i[:,k] for i
//      with pred_comp[i][w] > 0], solve beta = pinv(Tiks) @ T_w[:,k], and
//      add (Tiks beta) p_w[:,k]' to X̂_w.
//
// X̂_w is invariant under sign flips and rotational gauge freedom of the
// (W, T, P) triple, so it matches the Python reference exactly when both
// follow the same deterministic init (uniform ones, normalised) and
// tolerance.

namespace {

// Matches the Python reference's two-tier convergence:
//   * `kOnPlsNplsTol` (= OnPLS.consts.TOLERANCE = 5e-8) controls the
//     nPLS Gauss-Seidel break and the "non-trivial" sanity thresholds.
//   * `kOnPlsPcaEps` (= OnPLS init's `eps`) controls the NIPALS-PCA
//     `||p - p_old||` break. We pass 1e-14 to match
//     `_OnPlsPythonReference` (`eps=1e-14`).
constexpr double kOnPlsNplsTol = 5e-8;
constexpr double kOnPlsPcaEps = 1e-14;
constexpr int kOnPlsMaxIter = 1000;

// NIPALS-PCA single-component on M (rows × cols). Mirrors
// `OnPLS.estimators.PCA._compute_component`: start vector = column of
// largest squared norm; power-iterate p ← M' (M p), normalise p.
void on_pls_pca1(const std::vector<double>& M,
                 std::size_t rows, std::size_t cols,
                 std::vector<double>& p_out,
                 std::vector<double>& t_out) {
    p_out.assign(cols, 0.0);
    t_out.assign(rows, 0.0);
    if (rows == 0 || cols == 0) return;
    std::vector<double> col_ss(cols, 0.0);
    for (std::size_t r = 0; r < rows; ++r) {
        for (std::size_t c = 0; c < cols; ++c) {
            const double v = M[r * cols + c];
            col_ss[c] += v * v;
        }
    }
    std::size_t best = 0;
    for (std::size_t c = 1; c < cols; ++c) {
        if (col_ss[c] > col_ss[best]) best = c;
    }
    std::vector<double> t(rows, 0.0);
    for (std::size_t r = 0; r < rows; ++r) t[r] = M[r * cols + best];
    std::vector<double> p(cols, 0.0);
    for (std::size_t c = 0; c < cols; ++c) {
        double s = 0.0;
        for (std::size_t r = 0; r < rows; ++r) {
            s += M[r * cols + c] * t[r];
        }
        p[c] = s;
    }
    double pn = std::sqrt(squared_norm(p));
    if (pn > kOnPlsNplsTol) {  // norm guard uses TOLERANCE (5e-8)
        for (double& v : p) v /= pn;
    }
    std::vector<double> p_old(cols, 0.0);
    for (int it = 0; it < kOnPlsMaxIter; ++it) {
        std::fill(t.begin(), t.end(), 0.0);
        for (std::size_t r = 0; r < rows; ++r) {
            double s = 0.0;
            for (std::size_t c = 0; c < cols; ++c) {
                s += M[r * cols + c] * p[c];
            }
            t[r] = s;
        }
        p_old = p;
        std::fill(p.begin(), p.end(), 0.0);
        for (std::size_t c = 0; c < cols; ++c) {
            double s = 0.0;
            for (std::size_t r = 0; r < rows; ++r) {
                s += M[r * cols + c] * t[r];
            }
            p[c] = s;
        }
        pn = std::sqrt(squared_norm(p));
        if (pn > kOnPlsNplsTol) {  // norm guard uses TOLERANCE (5e-8)
            for (double& v : p) v /= pn;
        }
        double change = 0.0;
        for (std::size_t c = 0; c < cols; ++c) {
            const double d = p[c] - p_old[c];
            change += d * d;
        }
        if (std::sqrt(change) < kOnPlsPcaEps) break;  // convergence uses eps
    }
    // Final t = M p (matches the Python reference: t is recomputed at
    // the top of every iteration, so the final value uses the final p).
    std::fill(t.begin(), t.end(), 0.0);
    for (std::size_t r = 0; r < rows; ++r) {
        double s = 0.0;
        for (std::size_t c = 0; c < cols; ++c) {
            s += M[r * cols + c] * p[c];
        }
        t[r] = s;
    }
    p_out = std::move(p);
    t_out = std::move(t);
}

// Multi-component NIPALS-PCA on M (rows × cols), returning P (cols × K)
// row-major. Deflates M between components.
void on_pls_pca_k(std::vector<double> M, std::size_t rows, std::size_t cols,
                  std::size_t K, std::vector<double>& P_out) {
    P_out.assign(cols * K, 0.0);
    if (K == 0) return;
    std::vector<double> p_vec, t_vec;
    for (std::size_t k = 0; k < K; ++k) {
        on_pls_pca1(M, rows, cols, p_vec, t_vec);
        for (std::size_t c = 0; c < cols; ++c) P_out[c * K + k] = p_vec[c];
        for (std::size_t r = 0; r < rows; ++r) {
            for (std::size_t c = 0; c < cols; ++c) {
                M[r * cols + c] -= t_vec[r] * p_vec[c];
            }
        }
    }
}

// Minimum strictly-positive entry of row. Returns 0 if no positive.
std::int32_t on_pls_least_non_zero(const std::vector<std::int32_t>& row) {
    std::int32_t mn = 0;
    bool any = false;
    for (std::int32_t v : row) {
        if (v > 0) {
            if (!any || v < mn) { mn = v; any = true; }
        }
    }
    return any ? mn : 0;
}

// Numerical rank of an (rows × cols) matrix. Implementation: compute
// singular values^2 of M as eigenvalues of M'M via deflated power
// iteration, then count those above the relative numpy threshold
// `max(rows, cols) * eps * sigma_max`.
std::size_t on_pls_column_rank(const std::vector<double>& M,
                                std::size_t rows, std::size_t cols) {
    if (rows == 0 || cols == 0) return 0;
    std::vector<double> Gram(cols * cols, 0.0);
    for (std::size_t i = 0; i < cols; ++i) {
        for (std::size_t j = i; j < cols; ++j) {
            double s = 0.0;
            for (std::size_t r = 0; r < rows; ++r) {
                s += M[r * cols + i] * M[r * cols + j];
            }
            Gram[i * cols + j] = s;
            Gram[j * cols + i] = s;
        }
    }
    std::vector<double> sigmas;
    sigmas.reserve(cols);
    std::vector<double> v(cols, 0.0);
    std::vector<double> Gv(cols, 0.0);
    for (std::size_t k = 0; k < cols; ++k) {
        std::fill(v.begin(), v.end(),
                  1.0 / std::sqrt(static_cast<double>(cols)));
        double lambda = 0.0;
        for (int it = 0; it < 400; ++it) {
            std::fill(Gv.begin(), Gv.end(), 0.0);
            for (std::size_t i = 0; i < cols; ++i) {
                double s = 0.0;
                for (std::size_t j = 0; j < cols; ++j) {
                    s += Gram[i * cols + j] * v[j];
                }
                Gv[i] = s;
            }
            const double nrm = std::sqrt(squared_norm(Gv));
            if (nrm < 1e-300) { lambda = 0.0; break; }
            for (std::size_t i = 0; i < cols; ++i) Gv[i] /= nrm;
            double new_lambda = 0.0;
            for (std::size_t i = 0; i < cols; ++i) {
                double s = 0.0;
                for (std::size_t j = 0; j < cols; ++j) {
                    s += Gram[i * cols + j] * Gv[j];
                }
                new_lambda += Gv[i] * s;
            }
            const double diff = std::fabs(new_lambda - lambda);
            lambda = new_lambda;
            v = Gv;
            if (diff < 1e-14 * (std::fabs(lambda) + 1e-30)) break;
        }
        if (lambda <= 0.0) break;
        sigmas.push_back(std::sqrt(std::max(0.0, lambda)));
        for (std::size_t i = 0; i < cols; ++i) {
            for (std::size_t j = 0; j < cols; ++j) {
                Gram[i * cols + j] -= lambda * v[i] * v[j];
            }
        }
    }
    if (sigmas.empty()) return 0;
    double smax = 0.0;
    for (double s : sigmas) if (s > smax) smax = s;
    const double thresh = smax *
        static_cast<double>(std::max(rows, cols)) *
        std::numeric_limits<double>::epsilon();
    std::size_t rank = 0;
    for (double s : sigmas) if (s > thresh) ++rank;
    return rank;
}

// Single-component nPLS via Gauss-Seidel iteration on the
// sum-of-pairwise-scores objective. Initial w_i = ones(p_i)/sqrt(p_i);
// update w_i ← X_i' Σ_{j : pred_comp[i][j]>0} X_j w_j and normalize.
// Matches `OnPLS.estimators.nPLS._fit` exactly.
void on_pls_npls_1(const std::vector<std::vector<double>>& Xb,
                   const std::vector<std::size_t>& pb,
                   std::size_t n,
                   const std::vector<std::vector<std::int32_t>>& pred_comp,
                   std::vector<std::vector<double>>& w_out,
                   std::vector<std::vector<double>>& t_out,
                   std::vector<std::vector<double>>& p_out) {
    const std::size_t nb = Xb.size();
    w_out.assign(nb, {});
    t_out.assign(nb, {});
    p_out.assign(nb, {});
    for (std::size_t i = 0; i < nb; ++i) {
        w_out[i].assign(pb[i], 1.0 / std::sqrt(static_cast<double>(pb[i])));
    }
    std::vector<std::vector<double>> Xw(nb);
    for (std::size_t i = 0; i < nb; ++i) Xw[i].assign(n, 0.0);
    auto recompute_Xw = [&](std::size_t i) {
        std::fill(Xw[i].begin(), Xw[i].end(), 0.0);
        for (std::size_t r = 0; r < n; ++r) {
            double s = 0.0;
            for (std::size_t f = 0; f < pb[i]; ++f) {
                s += Xb[i][r * pb[i] + f] * w_out[i][f];
            }
            Xw[i][r] = s;
        }
    };
    for (std::size_t i = 0; i < nb; ++i) recompute_Xw(i);
    auto func_value = [&]() {
        double val = 0.0;
        for (std::size_t i = 0; i < nb; ++i) {
            for (std::size_t j = 0; j < nb; ++j) {
                if (pred_comp[i][j] > 0) {
                    double s = 0.0;
                    for (std::size_t r = 0; r < n; ++r) {
                        s += Xw[i][r] * Xw[j][r];
                    }
                    val += s;
                }
            }
        }
        return val;
    };
    double prev_f = func_value();
    std::vector<double> rhs;
    for (int it = 0; it < kOnPlsMaxIter; ++it) {
        for (std::size_t i = 0; i < nb; ++i) {
            std::vector<double> accum(n, 0.0);
            for (std::size_t j = 0; j < nb; ++j) {
                if (pred_comp[i][j] > 0) {
                    for (std::size_t r = 0; r < n; ++r) accum[r] += Xw[j][r];
                }
            }
            rhs.assign(pb[i], 0.0);
            for (std::size_t f = 0; f < pb[i]; ++f) {
                double s = 0.0;
                for (std::size_t r = 0; r < n; ++r) {
                    s += Xb[i][r * pb[i] + f] * accum[r];
                }
                rhs[f] = s;
            }
            const double nrm = std::sqrt(squared_norm(rhs));
            if (nrm > kOnPlsNplsTol) {
                for (double& v : rhs) v /= nrm;
            }
            w_out[i] = rhs;
            recompute_Xw(i);
        }
        const double new_f = func_value();
        const double err = (it >= 1) ? (new_f - prev_f) : new_f;
        prev_f = new_f;
        if (std::fabs(err) < kOnPlsNplsTol) break;
    }
    for (std::size_t i = 0; i < nb; ++i) {
        t_out[i].assign(n, 0.0);
        for (std::size_t r = 0; r < n; ++r) {
            double s = 0.0;
            for (std::size_t f = 0; f < pb[i]; ++f) {
                s += Xb[i][r * pb[i] + f] * w_out[i][f];
            }
            t_out[i][r] = s;
        }
        p_out[i].assign(pb[i], 0.0);
        double tt = 0.0;
        for (double v : t_out[i]) tt += v * v;
        if (tt > kOnPlsNplsTol) {
            for (std::size_t f = 0; f < pb[i]; ++f) {
                double s = 0.0;
                for (std::size_t r = 0; r < n; ++r) {
                    s += Xb[i][r * pb[i] + f] * t_out[i][r];
                }
                p_out[i][f] = s / tt;
            }
        }
    }
}

// Moore-Penrose pseudo-inverse of A (n × k) → A_pinv (k × n) via Jacobi
// eigendecomposition of A'A.
void on_pls_pinv(const std::vector<double>& A,
                  std::size_t n, std::size_t k,
                  std::vector<double>& A_pinv) {
    if (n == 0 || k == 0) {
        A_pinv.assign(k * n, 0.0);
        return;
    }
    std::vector<double> AtA(k * k, 0.0);
    for (std::size_t i = 0; i < k; ++i) {
        for (std::size_t j = i; j < k; ++j) {
            double s = 0.0;
            for (std::size_t r = 0; r < n; ++r) {
                s += A[r * k + i] * A[r * k + j];
            }
            AtA[i * k + j] = s;
            AtA[j * k + i] = s;
        }
    }
    std::vector<double> V(k * k, 0.0);
    for (std::size_t i = 0; i < k; ++i) V[i * k + i] = 1.0;
    for (int sweep = 0; sweep < 80; ++sweep) {
        double off = 0.0;
        for (std::size_t i = 0; i < k; ++i) {
            for (std::size_t j = 0; j < k; ++j) {
                if (i != j) off += AtA[i * k + j] * AtA[i * k + j];
            }
        }
        if (off < 1e-28) break;
        for (std::size_t p_idx = 0; p_idx + 1 < k; ++p_idx) {
            for (std::size_t q_idx = p_idx + 1; q_idx < k; ++q_idx) {
                const double apq = AtA[p_idx * k + q_idx];
                if (std::fabs(apq) < 1e-30) continue;
                const double app = AtA[p_idx * k + p_idx];
                const double aqq = AtA[q_idx * k + q_idx];
                const double tau = (aqq - app) / (2.0 * apq);
                double t_val;
                if (tau >= 0.0) {
                    t_val = 1.0 / (tau + std::sqrt(1.0 + tau * tau));
                } else {
                    t_val = 1.0 / (tau - std::sqrt(1.0 + tau * tau));
                }
                const double cc = 1.0 / std::sqrt(1.0 + t_val * t_val);
                const double ss = t_val * cc;
                for (std::size_t r = 0; r < k; ++r) {
                    const double a_rp = AtA[r * k + p_idx];
                    const double a_rq = AtA[r * k + q_idx];
                    AtA[r * k + p_idx] = cc * a_rp - ss * a_rq;
                    AtA[r * k + q_idx] = ss * a_rp + cc * a_rq;
                }
                for (std::size_t r = 0; r < k; ++r) {
                    const double a_pr = AtA[p_idx * k + r];
                    const double a_qr = AtA[q_idx * k + r];
                    AtA[p_idx * k + r] = cc * a_pr - ss * a_qr;
                    AtA[q_idx * k + r] = ss * a_pr + cc * a_qr;
                }
                for (std::size_t r = 0; r < k; ++r) {
                    const double v_rp = V[r * k + p_idx];
                    const double v_rq = V[r * k + q_idx];
                    V[r * k + p_idx] = cc * v_rp - ss * v_rq;
                    V[r * k + q_idx] = ss * v_rp + cc * v_rq;
                }
            }
        }
    }
    double max_lambda = 0.0;
    for (std::size_t i = 0; i < k; ++i) {
        const double v = AtA[i * k + i];
        if (v > max_lambda) max_lambda = v;
    }
    const double thresh = max_lambda *
        static_cast<double>(std::max(n, k)) *
        std::numeric_limits<double>::epsilon();
    std::vector<double> inv_lambda(k, 0.0);
    for (std::size_t i = 0; i < k; ++i) {
        const double v = AtA[i * k + i];
        if (v > thresh) inv_lambda[i] = 1.0 / v;
    }
    std::vector<double> M(k * k, 0.0);
    for (std::size_t i = 0; i < k; ++i) {
        for (std::size_t j = 0; j < k; ++j) {
            double s = 0.0;
            for (std::size_t r = 0; r < k; ++r) {
                s += V[i * k + r] * inv_lambda[r] * V[j * k + r];
            }
            M[i * k + j] = s;
        }
    }
    A_pinv.assign(k * n, 0.0);
    for (std::size_t i = 0; i < k; ++i) {
        for (std::size_t r = 0; r < n; ++r) {
            double s = 0.0;
            for (std::size_t j = 0; j < k; ++j) {
                s += M[i * k + j] * A[r * k + j];
            }
            A_pinv[i * n + r] = s;
        }
    }
}

}  // namespace (OnPLS helpers)

n4m_status_t fit_on_pls(Context& ctx,
                        const Config& cfg,
                        const std::vector<n4m_matrix_view_t>& X_blocks,
                        std::int32_t n_joint,
                        const std::vector<std::int32_t>& n_unique_per_block,
                        OnPlsResult& out) {
    (void)cfg;
    out = OnPlsResult{};
    if (X_blocks.size() < 2) {
        ctx.set_error("OnPLS requires at least 2 X blocks");
        return N4M_ERR_INVALID_ARGUMENT;
    }
    if (n_unique_per_block.size() != X_blocks.size()) {
        ctx.set_error("n_unique_per_block must have one entry per block");
        return N4M_ERR_INVALID_ARGUMENT;
    }
    const std::int64_t n_rows = X_blocks[0].rows;
    for (const auto& X : X_blocks) {
        if (X.rows != n_rows) {
            ctx.set_error("all blocks must share row count");
            return N4M_ERR_SHAPE_MISMATCH;
        }
    }
    if (n_joint < 0) {
        ctx.set_error("n_joint must be >= 0");
        return N4M_ERR_INVALID_ARGUMENT;
    }

    const std::size_t n = static_cast<std::size_t>(n_rows);
    const std::size_t nb = X_blocks.size();

    // Step 1: copy + per-block centering.
    std::vector<std::vector<double>> Xc(nb);     // mutable centered blocks
    std::vector<std::vector<double>> Xfresh(nb); // pristine centered copy
    std::vector<std::size_t> pb(nb, 0);
    for (std::size_t b = 0; b < nb; ++b) {
        n4m_status_t status = copy_matrix(ctx, X_blocks[b], "X_block", Xc[b]);
        if (status != N4M_OK) return status;
        pb[b] = static_cast<std::size_t>(X_blocks[b].cols);
        std::vector<double> mean;
        column_means(Xc[b], n, pb[b], mean);
        subtract_means(Xc[b], n, pb[b], mean);
        Xfresh[b] = Xc[b];
    }

    // pred_comp[i][j] = n_joint for i ≠ j, 0 on diagonal (matches the
    // wiring used by `_OnPlsPythonReference`).
    std::vector<std::vector<std::int32_t>> pred_comp(
        nb, std::vector<std::int32_t>(nb, 0));
    for (std::size_t i = 0; i < nb; ++i) {
        for (std::size_t j = 0; j < nb; ++j) {
            if (i != j) pred_comp[i][j] = n_joint;
        }
    }

    // Step 2: globally-joint loading bank gjW per block.
    std::vector<std::vector<double>> gjW(nb);
    std::vector<std::size_t> gjW_cols(nb, 0);
    for (std::size_t i = 0; i < nb; ++i) {
        std::vector<std::vector<double>> chunks;
        std::vector<std::size_t> chunk_cols;
        for (std::size_t j = 0; j < nb; ++j) {
            if (pred_comp[i][j] > 0) {
                std::vector<double> Mat(pb[j] * pb[i], 0.0);
                for (std::size_t a = 0; a < pb[j]; ++a) {
                    for (std::size_t b2 = 0; b2 < pb[i]; ++b2) {
                        double s = 0.0;
                        for (std::size_t r = 0; r < n; ++r) {
                            s += Xc[j][r * pb[j] + a] *
                                 Xc[i][r * pb[i] + b2];
                        }
                        Mat[a * pb[i] + b2] = s;
                    }
                }
                const std::size_t K =
                    static_cast<std::size_t>(pred_comp[i][j]);
                std::vector<double> P_chunk;
                on_pls_pca_k(std::move(Mat), pb[j], pb[i], K, P_chunk);
                chunks.push_back(std::move(P_chunk));
                chunk_cols.push_back(K);
            }
        }
        std::size_t total = 0;
        for (std::size_t c : chunk_cols) total += c;
        std::vector<double> W(pb[i] * total, 0.0);
        std::size_t col_off = 0;
        for (std::size_t cidx = 0; cidx < chunks.size(); ++cidx) {
            const std::size_t kk = chunk_cols[cidx];
            for (std::size_t r = 0; r < pb[i]; ++r) {
                for (std::size_t c = 0; c < kk; ++c) {
                    W[r * total + col_off + c] = chunks[cidx][r * kk + c];
                }
            }
            col_off += kk;
        }
        const std::size_t rank_W = on_pls_column_rank(W, pb[i], total);
        std::int32_t comps_pos = on_pls_least_non_zero(pred_comp[i]);
        if (comps_pos < 1) comps_pos = 1;
        std::size_t comps = std::min(rank_W,
            static_cast<std::size_t>(std::max(1, comps_pos)));
        std::vector<double> Wtrim(pb[i] * comps, 0.0);
        for (std::size_t r = 0; r < pb[i]; ++r) {
            for (std::size_t c = 0; c < comps; ++c) {
                Wtrim[r * comps + c] = W[r * total + c];
            }
        }
        for (std::size_t c = 0; c < comps; ++c) {
            double s = 0.0;
            for (std::size_t r = 0; r < pb[i]; ++r) {
                const double v = Wtrim[r * comps + c];
                s += v * v;
            }
            const double nrm = std::sqrt(s);
            const double denom = (nrm < kOnPlsNplsTol) ? 1.0 : nrm;
            for (std::size_t r = 0; r < pb[i]; ++r) {
                Wtrim[r * comps + c] /= denom;
            }
        }
        gjW[i] = std::move(Wtrim);
        gjW_cols[i] = comps;
    }

    // Step 3: per-block orthogonal (unique) component filtering.
    std::vector<std::vector<double>> Wo(nb), Po(nb);
    std::vector<std::size_t> ortho_cols(nb, 0);
    for (std::size_t i = 0; i < nb; ++i) {
        if (gjW_cols[i] < 1) continue;
        std::vector<double>& Xi = Xc[i];
        const std::size_t pi = pb[i];
        const std::size_t k_gj = gjW_cols[i];
        for (std::int32_t c = 0; c < n_unique_per_block[i]; ++c) {
            std::vector<double> Ti(n * k_gj, 0.0);
            for (std::size_t r = 0; r < n; ++r) {
                for (std::size_t kk = 0; kk < k_gj; ++kk) {
                    double s = 0.0;
                    for (std::size_t f = 0; f < pi; ++f) {
                        s += Xi[r * pi + f] * gjW[i][f * k_gj + kk];
                    }
                    Ti[r * k_gj + kk] = s;
                }
            }
            std::vector<double> Ei(n * pi, 0.0);
            for (std::size_t r = 0; r < n; ++r) {
                for (std::size_t f = 0; f < pi; ++f) {
                    double s = 0.0;
                    for (std::size_t kk = 0; kk < k_gj; ++kk) {
                        s += Ti[r * k_gj + kk] * gjW[i][f * k_gj + kk];
                    }
                    Ei[r * pi + f] = Xi[r * pi + f] - s;
                }
            }
            std::vector<double> Wortho(k_gj * pi, 0.0);
            for (std::size_t kk = 0; kk < k_gj; ++kk) {
                for (std::size_t f = 0; f < pi; ++f) {
                    double s = 0.0;
                    for (std::size_t r = 0; r < n; ++r) {
                        s += Ti[r * k_gj + kk] * Ei[r * pi + f];
                    }
                    Wortho[kk * pi + f] = s;
                }
            }
            std::vector<double> wortho, t_unused;
            on_pls_pca1(Wortho, k_gj, pi, wortho, t_unused);
            std::vector<double> tortho(n, 0.0);
            for (std::size_t r = 0; r < n; ++r) {
                double s = 0.0;
                for (std::size_t f = 0; f < pi; ++f) {
                    s += Xi[r * pi + f] * wortho[f];
                }
                tortho[r] = s;
            }
            double toto = 0.0;
            for (double v : tortho) toto += v * v;
            if (toto <= kOnPlsNplsTol) break;
            std::vector<double> portho(pi, 0.0);
            for (std::size_t f = 0; f < pi; ++f) {
                double s = 0.0;
                for (std::size_t r = 0; r < n; ++r) {
                    s += Xi[r * pi + f] * tortho[r];
                }
                portho[f] = s / toto;
            }
            Wo[i].insert(Wo[i].end(), wortho.begin(), wortho.end());
            Po[i].insert(Po[i].end(), portho.begin(), portho.end());
            ++ortho_cols[i];
            for (std::size_t r = 0; r < n; ++r) {
                for (std::size_t f = 0; f < pi; ++f) {
                    Xi[r * pi + f] -= tortho[r] * portho[f];
                }
            }
        }
    }

    // Step 4: predictive joint components via per-component nPLS.
    const std::size_t kk = static_cast<std::size_t>(n_joint);
    std::vector<std::vector<double>> W_pred(nb), T_pred(nb), P_pred(nb);
    for (std::size_t i = 0; i < nb; ++i) {
        W_pred[i].assign(pb[i] * kk, 0.0);
        T_pred[i].assign(n * kk, 0.0);
        P_pred[i].assign(pb[i] * kk, 0.0);
    }
    for (std::int32_t k = 0; k < n_joint; ++k) {
        std::vector<std::vector<double>> wk, tk, pk;
        on_pls_npls_1(Xc, pb, n, pred_comp, wk, tk, pk);
        for (std::size_t i = 0; i < nb; ++i) {
            for (std::size_t f = 0; f < pb[i]; ++f) {
                W_pred[i][f * kk + static_cast<std::size_t>(k)] = wk[i][f];
                P_pred[i][f * kk + static_cast<std::size_t>(k)] = pk[i][f];
            }
            for (std::size_t r = 0; r < n; ++r) {
                T_pred[i][r * kk + static_cast<std::size_t>(k)] = tk[i][r];
            }
            for (std::size_t r = 0; r < n; ++r) {
                for (std::size_t f = 0; f < pb[i]; ++f) {
                    Xc[i][r * pb[i] + f] -= tk[i][r] * pk[i][f];
                }
            }
        }
    }

    // Step 5: in-sample reconstruction X̂_b for every block, matching
    // `OnPLS.predict(blocks)[b]`. First re-filter fresh blocks with Wo/Po.
    std::vector<std::vector<double>> Xpred(nb);
    for (std::size_t i = 0; i < nb; ++i) Xpred[i] = Xfresh[i];
    for (std::size_t i = 0; i < nb; ++i) {
        if (ortho_cols[i] == 0) continue;
        const std::size_t pi = pb[i];
        for (std::size_t kk_o = 0; kk_o < ortho_cols[i]; ++kk_o) {
            std::vector<double> t(n, 0.0);
            for (std::size_t r = 0; r < n; ++r) {
                double s = 0.0;
                for (std::size_t f = 0; f < pi; ++f) {
                    s += Xpred[i][r * pi + f] * Wo[i][kk_o * pi + f];
                }
                t[r] = s;
            }
            for (std::size_t r = 0; r < n; ++r) {
                for (std::size_t f = 0; f < pi; ++f) {
                    Xpred[i][r * pi + f] -= t[r] * Po[i][kk_o * pi + f];
                }
            }
        }
    }
    // Per-component scores via sequential deflation by W_pred / P_pred.
    std::vector<std::vector<double>> Tpred(nb);
    for (std::size_t i = 0; i < nb; ++i) Tpred[i].assign(n * kk, 0.0);
    {
        std::vector<std::vector<double>> Xtmp = Xpred;
        for (std::int32_t k = 0; k < n_joint; ++k) {
            for (std::size_t i = 0; i < nb; ++i) {
                const std::size_t pi = pb[i];
                std::vector<double> tcol(n, 0.0);
                for (std::size_t r = 0; r < n; ++r) {
                    double s = 0.0;
                    for (std::size_t f = 0; f < pi; ++f) {
                        s += Xtmp[i][r * pi + f] *
                             W_pred[i][f * kk + static_cast<std::size_t>(k)];
                    }
                    tcol[r] = s;
                }
                for (std::size_t r = 0; r < n; ++r) {
                    Tpred[i][r * kk + static_cast<std::size_t>(k)] = tcol[r];
                }
                for (std::size_t r = 0; r < n; ++r) {
                    for (std::size_t f = 0; f < pi; ++f) {
                        Xtmp[i][r * pi + f] -= tcol[r] *
                            P_pred[i][f * kk + static_cast<std::size_t>(k)];
                    }
                }
            }
        }
    }
    // Accumulate X̂_w from connected-block scores per component.
    std::vector<std::vector<double>> Xhat(nb);
    for (std::size_t w = 0; w < nb; ++w) {
        const std::size_t pw = pb[w];
        Xhat[w].assign(n * pw, 0.0);
        for (std::int32_t k = 0; k < n_joint; ++k) {
            std::vector<std::size_t> contrib;
            for (std::size_t i = 0; i < nb; ++i) {
                if (pred_comp[i][w] > 0) contrib.push_back(i);
            }
            if (contrib.empty()) continue;
            const std::size_t ndim = contrib.size();
            std::vector<double> Tiks(n * ndim, 0.0);
            for (std::size_t r = 0; r < n; ++r) {
                for (std::size_t c = 0; c < ndim; ++c) {
                    Tiks[r * ndim + c] = Tpred[contrib[c]][r * kk +
                        static_cast<std::size_t>(k)];
                }
            }
            std::vector<double> Tiks_pinv;
            on_pls_pinv(Tiks, n, ndim, Tiks_pinv);
            std::vector<double> Twk(n, 0.0);
            for (std::size_t r = 0; r < n; ++r) {
                Twk[r] = Tpred[w][r * kk + static_cast<std::size_t>(k)];
            }
            std::vector<double> beta(ndim, 0.0);
            for (std::size_t c = 0; c < ndim; ++c) {
                double s = 0.0;
                for (std::size_t r = 0; r < n; ++r) {
                    s += Tiks_pinv[c * n + r] * Twk[r];
                }
                beta[c] = s;
            }
            std::vector<double> Thatwk(n, 0.0);
            for (std::size_t r = 0; r < n; ++r) {
                double s = 0.0;
                for (std::size_t c = 0; c < ndim; ++c) {
                    s += Tiks[r * ndim + c] * beta[c];
                }
                Thatwk[r] = s;
            }
            for (std::size_t r = 0; r < n; ++r) {
                for (std::size_t f = 0; f < pw; ++f) {
                    Xhat[w][r * pw + f] += Thatwk[r] *
                        P_pred[w][f * kk + static_cast<std::size_t>(k)];
                }
            }
        }
    }

    // Pack the result. joint_loadings/scores/unique_loadings now hold
    // the canonical OnPLS quantities; block_reconstruction_per_block is
    // the sign- and rotation-invariant prediction used by the parity
    // gate.
    out.n_blocks = static_cast<std::int32_t>(nb);
    out.n_joint = n_joint;
    out.n_unique_per_block = n_unique_per_block;
    out.joint_loadings_per_block.assign(nb, {});
    out.joint_scores_per_block.assign(nb, {});
    out.unique_loadings_per_block.assign(nb, {});
    out.block_reconstruction_per_block.assign(nb, {});
    for (std::size_t b = 0; b < nb; ++b) {
        out.joint_loadings_per_block[b] = P_pred[b];
        out.joint_scores_per_block[b] = T_pred[b];
        const std::size_t pi = pb[b];
        const std::size_t nu =
            static_cast<std::size_t>(n_unique_per_block[b]);
        const std::size_t nu_real = ortho_cols[b];
        std::vector<double> Po_packed(pi * nu, 0.0);
        for (std::size_t kc = 0; kc < std::min(nu, nu_real); ++kc) {
            for (std::size_t f = 0; f < pi; ++f) {
                Po_packed[f * nu + kc] = Po[b][kc * pi + f];
            }
        }
        out.unique_loadings_per_block[b] = std::move(Po_packed);
        out.block_reconstruction_per_block[b] = Xhat[b];
    }

    ctx.clear_error();
    return N4M_OK;
}

// ---------- ROSA --------------------------------------------------------

n4m_status_t fit_rosa(Context& ctx,
                      const Config& cfg,
                      const std::vector<n4m_matrix_view_t>& X_blocks,
                      const n4m_matrix_view_t& Y,
                      std::int32_t n_components,
                      RosaResult& out) {
    (void)cfg;
    out = RosaResult{};
    if (X_blocks.empty()) {
        ctx.set_error("ROSA requires at least one X block");
        return N4M_ERR_INVALID_ARGUMENT;
    }
    if (n_components < 1) {
        ctx.set_error("ROSA n_components must be >= 1");
        return N4M_ERR_INVALID_ARGUMENT;
    }
    const std::int64_t n_rows = Y.rows;
    for (const auto& X : X_blocks) {
        if (X.rows != n_rows) {
            ctx.set_error("all X blocks must match Y row count");
            return N4M_ERR_SHAPE_MISMATCH;
        }
    }

    std::vector<double> Y_buf;
    n4m_status_t status = copy_matrix(ctx, Y, "Y", Y_buf);
    if (status != N4M_OK) return status;
    const std::size_t n = static_cast<std::size_t>(n_rows);
    const std::size_t q = static_cast<std::size_t>(Y.cols);
    const std::size_t n_blocks = X_blocks.size();
    std::vector<std::vector<double>> X_bufs(n_blocks);
    std::vector<std::size_t> p_block(n_blocks, 0);
    for (std::size_t b = 0; b < n_blocks; ++b) {
        status = copy_matrix(ctx, X_blocks[b], "X_block", X_bufs[b]);
        if (status != N4M_OK) return status;
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
    return N4M_OK;
}

}  // namespace n4m::core
