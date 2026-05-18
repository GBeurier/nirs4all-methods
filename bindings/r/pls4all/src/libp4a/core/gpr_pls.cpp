// SPDX-License-Identifier: CECILL-2.1
//
// GPR-on-PLS (§47) implementation.
//
// The GP head (`fit_gp_on_scores`) is intentionally decoupled from the
// PLS stage so a future GPR-on-AOMPLS can hand it pre-computed scores
// from any score-generating frontend without re-deriving the kernel
// math.

#include "core/gpr_pls.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>
#include <memory>
#include <vector>

#include "core/linalg.hpp"
#include "core/model.hpp"

namespace pls4all::core {

namespace {

// In-place lower-triangular Cholesky of an n x n row-major SPD matrix.
// Zeros out the upper triangle on success. Returns false on a non-
// positive pivot (matrix is not positive-definite).
[[nodiscard]] bool cholesky_lower(double* A, std::size_t n) noexcept {
    for (std::size_t j = 0; j < n; ++j) {
        double s = A[j * n + j];
        for (std::size_t k = 0; k < j; ++k) {
            s -= A[j * n + k] * A[j * n + k];
        }
        if (!(s > 0.0)) {
            return false;
        }
        A[j * n + j] = std::sqrt(s);
        const double inv = 1.0 / A[j * n + j];
        for (std::size_t i = j + 1; i < n; ++i) {
            double t = A[i * n + j];
            for (std::size_t k = 0; k < j; ++k) {
                t -= A[i * n + k] * A[j * n + k];
            }
            A[i * n + j] = t * inv;
            A[j * n + i] = 0.0;  // zero upper triangle
        }
    }
    return true;
}

// Forward substitution: solve L @ x = b. L is n x n lower-triangular
// row-major; x and b are length n.
void trsv_lower(const double* L, const double* b, double* x,
                std::size_t n) noexcept {
    for (std::size_t i = 0; i < n; ++i) {
        double s = b[i];
        for (std::size_t k = 0; k < i; ++k) {
            s -= L[i * n + k] * x[k];
        }
        x[i] = s / L[i * n + i];
    }
}

// Back substitution: solve L^T @ x = b. Reads only the lower triangle
// of L (column k for the inner loop corresponds to L^T row k).
void trsv_upper(const double* L, const double* b, double* x,
                std::size_t n) noexcept {
    for (std::size_t i = n; i-- > 0; ) {
        double s = b[i];
        for (std::size_t k = i + 1; k < n; ++k) {
            s -= L[k * n + i] * x[k];
        }
        x[i] = s / L[i * n + i];
    }
}

// Read a row-major dense double matrix from a p4a_matrix_view_t. Local
// helper — extra_pls.cpp has the same one in its anonymous namespace.
[[nodiscard]] p4a_status_t copy_view(Context& ctx,
                                      const p4a_matrix_view_t& V,
                                      const char* name,
                                      std::vector<double>& out) {
    if (V.dtype != P4A_DTYPE_F64) {
        ctx.set_errorf("%s must be F64", name);
        return P4A_ERR_INVALID_ARGUMENT;
    }
    const std::size_t rows = static_cast<std::size_t>(V.rows);
    const std::size_t cols = static_cast<std::size_t>(V.cols);
    out.assign(rows * cols, 0.0);
    if (rows == 0 || cols == 0) {
        return P4A_OK;
    }
    if (V.data == nullptr) {
        ctx.set_errorf("%s has null data", name);
        return P4A_ERR_NULL_POINTER;
    }
    const auto* data = static_cast<const double*>(V.data);
    const std::size_t rs = static_cast<std::size_t>(V.row_stride);
    const std::size_t cs = static_cast<std::size_t>(V.col_stride);
    for (std::size_t r = 0; r < rows; ++r) {
        for (std::size_t c = 0; c < cols; ++c) {
            out[r * cols + c] = data[r * rs + c * cs];
        }
    }
    return P4A_OK;
}

}  // namespace

// ---------------------------------------------------------------------------
// GP head — train (K + σ²I) α = y_centered via Cholesky.
// Stores L, alpha, T_train so prediction can be replayed without retraining.
// ---------------------------------------------------------------------------
p4a_status_t fit_gp_on_scores(
        Context& ctx,
        const std::vector<double>& T,
        std::int32_t n_in,
        std::int32_t k_in,
        const std::vector<double>& y_centered,
        double length_scale,
        double noise_level,
        GpHeadResult& out) {
    out = GpHeadResult{};
    if (n_in < 1 || k_in < 1) {
        ctx.set_errorf("fit_gp_on_scores: n=%d k=%d must be >= 1",
                        static_cast<int>(n_in), static_cast<int>(k_in));
        return P4A_ERR_INVALID_ARGUMENT;
    }
    if (!(length_scale > 0.0)) {
        ctx.set_error("length_scale must be strictly positive");
        return P4A_ERR_INVALID_ARGUMENT;
    }
    if (!(noise_level > 0.0)) {
        ctx.set_error("noise_level must be strictly positive");
        return P4A_ERR_INVALID_ARGUMENT;
    }
    const std::size_t n = static_cast<std::size_t>(n_in);
    const std::size_t k = static_cast<std::size_t>(k_in);
    if (T.size() != n * k) {
        ctx.set_errorf("fit_gp_on_scores: T.size()=%zu but n*k=%zu",
                        T.size(), n * k);
        return P4A_ERR_SHAPE_MISMATCH;
    }
    if (y_centered.size() != n) {
        ctx.set_errorf("fit_gp_on_scores: y_centered.size()=%zu but n=%zu",
                        y_centered.size(), n);
        return P4A_ERR_SHAPE_MISMATCH;
    }

    out.n_train = n_in;
    out.n_components = k_in;
    out.length_scale = length_scale;
    out.noise_level = noise_level;
    out.T_train = T;
    out.alpha.assign(n, 0.0);
    out.L_lower.assign(n * n, 0.0);

    // Build K (with noise on diagonal) into out.L_lower; Cholesky factors
    // it in place. RBF: K[i,j] = exp(-||t_i - t_j||² / (2 * ℓ²)).
    const double inv_2sq = 1.0 / (2.0 * length_scale * length_scale);
    for (std::size_t i = 0; i < n; ++i) {
        for (std::size_t j = i; j < n; ++j) {
            double sq = 0.0;
            for (std::size_t c = 0; c < k; ++c) {
                const double d = T[i * k + c] - T[j * k + c];
                sq += d * d;
            }
            const double kij = std::exp(-sq * inv_2sq);
            out.L_lower[i * n + j] = kij;
            out.L_lower[j * n + i] = kij;
        }
        out.L_lower[i * n + i] += noise_level;  // diagonal noise term
    }

    if (!cholesky_lower(out.L_lower.data(), n)) {
        ctx.set_error(
            "fit_gp_on_scores: Cholesky of (K + noise·I) failed — "
            "try a larger noise_level or fewer components");
        return P4A_ERR_NUMERICAL_FAILURE;
    }

    // Solve L @ z = y_centered, then L^T @ alpha = z.
    std::vector<double> z(n, 0.0);
    trsv_lower(out.L_lower.data(), y_centered.data(), z.data(), n);
    trsv_upper(out.L_lower.data(), z.data(), out.alpha.data(), n);
    return P4A_OK;
}

// ---------------------------------------------------------------------------
// Top-level GPR-on-PLS fit.
// ---------------------------------------------------------------------------
p4a_status_t fit_gpr_pls(
        Context& ctx,
        const Config& cfg,
        const p4a_matrix_view_t& X,
        const p4a_matrix_view_t& Y,
        double length_scale,
        double noise_level,
        std::uint64_t seed,
        GprPlsResult& out) {
    (void)seed;  // reserved for ABI symmetry; the fit is deterministic
    out = GprPlsResult{};

    if (!(length_scale > 0.0)) {
        ctx.set_error("length_scale must be strictly positive");
        return P4A_ERR_INVALID_ARGUMENT;
    }
    if (!(noise_level > 0.0)) {
        ctx.set_error("noise_level must be strictly positive");
        return P4A_ERR_INVALID_ARGUMENT;
    }
    if (Y.cols != 1) {
        ctx.set_error("fit_gpr_pls: Y must be n x 1 (single target only)");
        return P4A_ERR_SHAPE_MISMATCH;
    }
    if (X.rows != Y.rows) {
        ctx.set_errorf("fit_gpr_pls: X.rows=%lld != Y.rows=%lld",
                        static_cast<long long>(X.rows),
                        static_cast<long long>(Y.rows));
        return P4A_ERR_SHAPE_MISMATCH;
    }
    const auto n_in = static_cast<std::int32_t>(X.rows);
    const auto p_in = static_cast<std::int32_t>(X.cols);
    const std::int32_t k_in = cfg.n_components;
    if (k_in < 1 || k_in > std::min(n_in, p_in)) {
        ctx.set_errorf(
            "fit_gpr_pls: n_components=%d out of range [1, min(%d,%d)]",
            static_cast<int>(k_in), static_cast<int>(n_in),
            static_cast<int>(p_in));
        return P4A_ERR_INVALID_ARGUMENT;
    }

    // Stage 1: fit PLS to obtain the rotation R (p x k).
    Config pls_cfg = cfg;
    pls_cfg.algorithm = P4A_ALGO_PLS_REGRESSION;
    pls_cfg.solver = P4A_SOLVER_SIMPLS;
    pls_cfg.deflation = P4A_DEFLATION_REGRESSION;
    pls_cfg.center_x = 1;
    pls_cfg.center_y = 1;
    // Force unit scaling so the rotation matches sklearn PLSRegression(scale=False).
    pls_cfg.scale_x = 0;
    pls_cfg.scale_y = 0;

    std::unique_ptr<Model> model;
    const p4a_status_t pls_status = fit_model(ctx, pls_cfg, X, Y, model);
    if (pls_status != P4A_OK) {
        return pls_status;
    }
    if (model->rotations_r.size() !=
        static_cast<std::size_t>(p_in) * static_cast<std::size_t>(k_in)) {
        ctx.set_error("fit_gpr_pls: rotation matrix has unexpected size");
        return P4A_ERR_INTERNAL;
    }

    const std::size_t n = static_cast<std::size_t>(n_in);
    const std::size_t p = static_cast<std::size_t>(p_in);
    const std::size_t k = static_cast<std::size_t>(k_in);

    out.n_features = p_in;
    out.n_samples = n_in;
    out.n_components = k_in;
    out.length_scale = length_scale;
    out.noise_level = noise_level;
    out.rotation_r = model->rotations_r;
    out.x_mean = model->x_mean;

    // Stage 2a: compute training scores T = (X - x_mean) @ R (n x k).
    std::vector<double> X_buf;
    p4a_status_t st = copy_view(ctx, X, "X", X_buf);
    if (st != P4A_OK) {
        return st;
    }
    for (std::size_t i = 0; i < n; ++i) {
        for (std::size_t j = 0; j < p; ++j) {
            X_buf[i * p + j] -= out.x_mean[j];
        }
    }
    std::vector<double> T(n * k, 0.0);
    pls4all::linalg::gemm(
        pls4all::linalg::Trans_No, pls4all::linalg::Trans_No,
        n, k, p,
        1.0,
        X_buf.data(), p,
        out.rotation_r.data(), k,
        0.0,
        T.data(), k);

    // Stage 2b: centre y. The PLS model has y_mean[0]; use it for consistency
    // with sklearn `Y.ravel().mean()`.
    if (model->y_mean.empty()) {
        ctx.set_error("fit_gpr_pls: PLS y_mean missing");
        return P4A_ERR_INTERNAL;
    }
    const double y_mean_scalar = model->y_mean[0];
    std::vector<double> y_centered(n, 0.0);
    {
        const auto* y_data = static_cast<const double*>(Y.data);
        const std::size_t y_rs = static_cast<std::size_t>(Y.row_stride);
        const std::size_t y_cs = static_cast<std::size_t>(Y.col_stride);
        for (std::size_t i = 0; i < n; ++i) {
            y_centered[i] = y_data[i * y_rs + 0 * y_cs] - y_mean_scalar;
        }
    }

    // Stage 2c: fit GP head.
    st = fit_gp_on_scores(ctx, T, n_in, k_in, y_centered,
                           length_scale, noise_level, out.gp);
    if (st != P4A_OK) {
        return st;
    }
    out.gp.y_mean_scalar = y_mean_scalar;

    // Stage 3: training predictions. Identity (K + σ²I) α = y_c implies
    // K @ α = y_c - σ² · α. So training mean prediction = y_c - σ² · α + y_mean.
    out.predictions.assign(n, 0.0);
    for (std::size_t i = 0; i < n; ++i) {
        out.predictions[i] = y_centered[i] - noise_level * out.gp.alpha[i]
                              + y_mean_scalar;
    }

    // Stage 4: training predictive variance.
    //   σ²*[i] = K(t_i, t_i) - k_i^T (K + σ²I)^{-1} k_i
    //          = 1 - v_i^T v_i, where L v_i = k_i (K[:,i] WITHOUT noise term).
    out.predictive_variance.assign(n, 0.0);
    std::vector<double> kcol(n, 0.0);
    std::vector<double> v(n, 0.0);
    const double inv_2sq = 1.0 / (2.0 * length_scale * length_scale);
    for (std::size_t i = 0; i < n; ++i) {
        for (std::size_t j = 0; j < n; ++j) {
            double sq = 0.0;
            for (std::size_t c = 0; c < k; ++c) {
                const double d = T[i * k + c] - T[j * k + c];
                sq += d * d;
            }
            kcol[j] = std::exp(-sq * inv_2sq);
        }
        trsv_lower(out.gp.L_lower.data(), kcol.data(), v.data(), n);
        double dot = 0.0;
        for (double x : v) {
            dot += x * x;
        }
        const double var = 1.0 - dot;
        out.predictive_variance[i] = (var > 0.0) ? var : 0.0;
    }

    return P4A_OK;
}

}  // namespace pls4all::core
