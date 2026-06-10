// SPDX-License-Identifier: CECILL-2.1
//
// Direct (closed-form) multi-output Ridge regression. See ridge.hpp.

#include "core/ridge.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <vector>

#include "core/common/linalg.h"
#include "core/common/matrix_view.hpp"

namespace n4m::core {

namespace {

// Copy a (rows x cols) row-major matrix out of a (possibly strided) view,
// rejecting NaN/Inf. Mirrors the convention in extra_pls.cpp / ecr.cpp.
[[nodiscard]] n4m_status_t copy_matrix(Context& ctx, const n4m_matrix_view_t& V,
                                       const char* name,
                                       std::vector<double>& out,
                                       std::size_t& rows, std::size_t& cols) {
    if (validate_nonnull_view(V) != N4M_OK ||
        (V.dtype != N4M_DTYPE_F64 && V.dtype != N4M_DTYPE_F32)) {
        ctx.set_errorf("%s must be a finite row-major F64/F32 matrix", name);
        return N4M_ERR_INVALID_ARGUMENT;
    }
    rows = static_cast<std::size_t>(V.rows);
    cols = static_cast<std::size_t>(V.cols);
    out.assign(rows * cols, 0.0);
    if (rows == 0 || cols == 0) return N4M_OK;
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
        for (std::size_t c = 0; c < cols; ++c) means[c] += mat[r * cols + c];
    }
    const double inv = 1.0 / static_cast<double>(rows);
    for (double& m : means) m *= inv;
}

// Population std-dev per column (sklearn StandardScaler convention: ddof=0,
// zero-variance -> scale 1.0). Operates on already-centered data.
void column_scales(const std::vector<double>& centered, std::size_t rows,
                   std::size_t cols, std::vector<double>& scales) {
    scales.assign(cols, 1.0);
    if (rows == 0) return;
    for (std::size_t c = 0; c < cols; ++c) {
        double ss = 0.0;
        for (std::size_t r = 0; r < rows; ++r) {
            const double v = centered[r * cols + c];
            ss += v * v;
        }
        const double std_dev = std::sqrt(ss / static_cast<double>(rows));
        scales[c] = (std_dev > 0.0) ? std_dev : 1.0;
    }
}

void apply_column_scale(std::vector<double>& mat, std::size_t rows,
                        std::size_t cols, const std::vector<double>& scales) {
    for (std::size_t r = 0; r < rows; ++r) {
        for (std::size_t c = 0; c < cols; ++c) mat[r * cols + c] /= scales[c];
    }
}

// Solve A x = b for x where A is (m x m) row-major (square, full rank under
// the ridge penalty). Reuses the shipped Householder QR primitives. A is
// copied internally (the factorisation overwrites it). Returns status.
[[nodiscard]] n4m_status_t solve_square_qr(const std::vector<double>& A,
                                           std::size_t m,
                                           const std::vector<double>& b,
                                           std::vector<double>& x) {
    std::vector<double> Af = A;        // QR overwrites
    std::vector<double> tau(m, 0.0);
    n4m_status_t st = n4m_householder_qr(
        Af.data(), static_cast<std::int64_t>(m),
        static_cast<std::int64_t>(m), tau.data());
    if (st != N4M_OK) return st;
    std::vector<double> rhs = b;       // Q^T applied in place
    st = n4m_apply_qt(Af.data(), static_cast<std::int64_t>(m),
                      static_cast<std::int64_t>(m), tau.data(), rhs.data());
    if (st != N4M_OK) return st;
    x.assign(m, 0.0);
    return n4m_back_solve_R(Af.data(), static_cast<std::int64_t>(m),
                            static_cast<std::int64_t>(m), rhs.data(), x.data());
}

}  // namespace

n4m_status_t fit_ridge(Context& ctx, const Config& cfg,
                       const n4m_matrix_view_t& X, const n4m_matrix_view_t& Y,
                       double lambda, RidgeSolver solver, bool fit_intercept,
                       RidgeResult& out) {
    if (!std::isfinite(lambda) || lambda < 0.0) {
        ctx.set_errorf("ridge lambda must be finite and >= 0 (got %g)", lambda);
        return N4M_ERR_INVALID_ARGUMENT;
    }

    std::vector<double> Xc, Yc;
    std::size_t n = 0, p = 0, ny = 0, q = 0;
    n4m_status_t st = copy_matrix(ctx, X, "X", Xc, n, p);
    if (st != N4M_OK) return st;
    st = copy_matrix(ctx, Y, "Y", Yc, ny, q);
    if (st != N4M_OK) return st;
    if (n == 0 || p == 0 || q == 0) {
        ctx.set_error("ridge requires non-empty X and Y");
        return N4M_ERR_INVALID_ARGUMENT;
    }
    if (ny != n) {
        ctx.set_error("ridge: X and Y must have the same number of rows");
        return N4M_ERR_SHAPE_MISMATCH;
    }

    // --- standardize ---------------------------------------------------------
    out.x_mean.assign(p, 0.0);
    out.y_mean.assign(q, 0.0);
    out.x_scale.assign(p, 1.0);
    const bool center = fit_intercept;  // centering is what makes the intercept
    if (center && cfg.center_x != 0) {
        column_means(Xc, n, p, out.x_mean);
        for (std::size_t r = 0; r < n; ++r)
            for (std::size_t c = 0; c < p; ++c) Xc[r * p + c] -= out.x_mean[c];
    }
    if (center && cfg.center_y != 0) {
        column_means(Yc, n, q, out.y_mean);
        for (std::size_t r = 0; r < n; ++r)
            for (std::size_t c = 0; c < q; ++c) Yc[r * q + c] -= out.y_mean[c];
    }
    if (cfg.scale_x != 0) {
        column_scales(Xc, n, p, out.x_scale);
        apply_column_scale(Xc, n, p, out.x_scale);
    }

    // --- solver selection ----------------------------------------------------
    RidgeSolver chosen = solver;
    if (chosen == RidgeSolver::kAuto)
        chosen = (p > n) ? RidgeSolver::kDual : RidgeSolver::kPrimal;

    // Coefficients on the *scaled, centered* X. B is p x q row-major.
    std::vector<double> B(p * q, 0.0);

    if (chosen == RidgeSolver::kPrimal) {
        // Augmented QR: stack [Xc; sqrt(lambda) I_p] (rows n+p, cols p) and
        // [Yc; 0] per column, then least-squares solve. The normal equations
        // of this augmented system are exactly (Xc'Xc + lambda I) B = Xc'Yc.
        const std::size_t m = n + p;
        const double sl = std::sqrt(lambda);
        std::vector<double> Aug(m * p, 0.0);
        for (std::size_t r = 0; r < n; ++r)
            for (std::size_t c = 0; c < p; ++c) Aug[r * p + c] = Xc[r * p + c];
        for (std::size_t c = 0; c < p; ++c) Aug[(n + c) * p + c] = sl;

        std::vector<double> tau(p, 0.0);
        st = n4m_householder_qr(Aug.data(), static_cast<std::int64_t>(m),
                                static_cast<std::int64_t>(p), tau.data());
        if (st != N4M_OK) {
            ctx.set_error("ridge primal QR factorisation failed");
            return st;
        }
        std::vector<double> rhs(m, 0.0);
        std::vector<double> sol(p, 0.0);
        for (std::size_t t = 0; t < q; ++t) {
            for (std::size_t r = 0; r < n; ++r) rhs[r] = Yc[r * q + t];
            for (std::size_t r = n; r < m; ++r) rhs[r] = 0.0;
            st = n4m_apply_qt(Aug.data(), static_cast<std::int64_t>(m),
                              static_cast<std::int64_t>(p), tau.data(),
                              rhs.data());
            if (st != N4M_OK) return st;
            st = n4m_back_solve_R(Aug.data(), static_cast<std::int64_t>(m),
                                  static_cast<std::int64_t>(p), rhs.data(),
                                  sol.data());
            if (st != N4M_OK) {
                ctx.set_error("ridge primal back-substitution failed "
                              "(singular system; lambda=0 with rank-deficient X?)");
                return st;
            }
            for (std::size_t c = 0; c < p; ++c) B[c * q + t] = sol[c];
        }
    } else {
        // Dual: K = Xc Xc' (n x n), solve (K + lambda I_n) A = Yc, B = Xc' A.
        std::vector<double> K(n * n, 0.0);
        for (std::size_t i = 0; i < n; ++i) {
            for (std::size_t j = i; j < n; ++j) {
                double acc = 0.0;
                for (std::size_t c = 0; c < p; ++c)
                    acc += Xc[i * p + c] * Xc[j * p + c];
                K[i * n + j] = acc;
                K[j * n + i] = acc;
            }
        }
        for (std::size_t i = 0; i < n; ++i) K[i * n + i] += lambda;

        std::vector<double> A(n * q, 0.0);
        std::vector<double> rhs(n, 0.0);
        std::vector<double> sol;
        for (std::size_t t = 0; t < q; ++t) {
            for (std::size_t i = 0; i < n; ++i) rhs[i] = Yc[i * q + t];
            st = solve_square_qr(K, n, rhs, sol);
            if (st != N4M_OK) {
                ctx.set_error("ridge dual solve failed "
                              "(singular K+lambda*I; lambda=0 with rank-deficient X?)");
                return st;
            }
            for (std::size_t i = 0; i < n; ++i) A[i * q + t] = sol[i];
        }
        // B = Xc' A  -> B[c,t] = sum_i Xc[i,c] * A[i,t]
        for (std::size_t c = 0; c < p; ++c) {
            for (std::size_t t = 0; t < q; ++t) {
                double acc = 0.0;
                for (std::size_t i = 0; i < n; ++i)
                    acc += Xc[i * p + c] * A[i * q + t];
                B[c * q + t] = acc;
            }
        }
    }

    // --- de-scale coefficients back to the original X scale ------------------
    // B was fit on Xc/x_scale; B_orig[c,t] = B[c,t] / x_scale[c].
    if (cfg.scale_x != 0) {
        for (std::size_t c = 0; c < p; ++c) {
            const double inv = 1.0 / out.x_scale[c];
            for (std::size_t t = 0; t < q; ++t) B[c * q + t] *= inv;
        }
    }

    // --- intercept = y_mean - x_mean . B (penalty not applied to intercept) --
    out.intercept.assign(q, 0.0);
    if (fit_intercept) {
        for (std::size_t t = 0; t < q; ++t) {
            double acc = out.y_mean[t];
            for (std::size_t c = 0; c < p; ++c) acc -= out.x_mean[c] * B[c * q + t];
            out.intercept[t] = acc;
        }
    }

    out.coefficients = std::move(B);
    out.n_samples = static_cast<std::int64_t>(n);
    out.n_features = static_cast<std::int32_t>(p);
    out.n_targets = static_cast<std::int32_t>(q);
    out.lambda = lambda;
    out.solver_used = chosen;

    // --- predictions on ORIGINAL X + intercept, and in-sample RMSE -----------
    out.predictions.assign(n * q, 0.0);
    const auto* x_data = static_cast<const double*>(X.data);
    const auto x_rs = static_cast<std::size_t>(X.row_stride);
    const auto x_cs = static_cast<std::size_t>(X.col_stride);
    const auto* y_data = static_cast<const double*>(Y.data);
    const auto y_rs = static_cast<std::size_t>(Y.row_stride);
    const auto y_cs = static_cast<std::size_t>(Y.col_stride);
    const bool y_f64 = (Y.dtype == N4M_DTYPE_F64);
    const auto* y_f32 = static_cast<const float*>(Y.data);
    double sumsq = 0.0;
    for (std::size_t i = 0; i < n; ++i) {
        for (std::size_t t = 0; t < q; ++t) {
            double acc = out.intercept[t];
            for (std::size_t c = 0; c < p; ++c) {
                const double xv = (X.dtype == N4M_DTYPE_F64)
                    ? x_data[i * x_rs + c * x_cs]
                    : static_cast<double>(
                          static_cast<const float*>(X.data)[i * x_rs + c * x_cs]);
                acc += xv * out.coefficients[c * q + t];
            }
            out.predictions[i * q + t] = acc;
            const double truth = y_f64 ? y_data[i * y_rs + t * y_cs]
                                       : static_cast<double>(y_f32[i * y_rs + t * y_cs]);
            const double d = truth - acc;
            sumsq += d * d;
        }
    }
    out.rmse = std::sqrt(sumsq / static_cast<double>(n * q));

    return N4M_OK;
}

}  // namespace n4m::core
