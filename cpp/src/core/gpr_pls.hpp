// SPDX-License-Identifier: CeCILL-2.1
//
// GPR-on-PLS (§47) — Gaussian Process Regression head on PLS scores.
//
// Two stages:
//   1. Fit a PLS regression on (X, Y) → rotation R (p × k).
//      Compute training scores T = (X − x_mean) @ R (n × k).
//   2. Fit a Gaussian Process with RBF kernel
//        K(t_i, t_j) = exp(-||t_i - t_j||² / (2·ℓ²))
//      and diagonal noise σ²·I on (T, y_centered):
//        (K + σ²·I) α = y_centered (Cholesky-based)
//
// The GP head is exposed as a separate callable (`fit_gp_on_scores`)
// so a future GPR-on-AOMPLS module reuses it as-is — only the
// score-generating stage changes.

#ifndef PLS4ALL_CORE_GPR_PLS_HPP
#define PLS4ALL_CORE_GPR_PLS_HPP

#include <cstdint>
#include <vector>

#include "pls4all/p4a.h"

#include "core/config.hpp"
#include "core/context.hpp"

namespace pls4all::core {

// ----------------------------------------------------------------------------
// GP head — standalone, reusable from any future module that produces
// training scores T (e.g. GPR-on-AOMPLS).
// ----------------------------------------------------------------------------
struct GpHeadResult {
    std::int32_t n_train{0};        // n
    std::int32_t n_components{0};   // k
    double length_scale{1.0};
    double noise_level{1e-3};
    std::vector<double> alpha;      // length n — dual weights
    std::vector<double> L_lower;    // n × n row-major Cholesky factor
    std::vector<double> T_train;    // n × k row-major training scores
    double y_mean_scalar{0.0};      // centring offset for y
};

// Fits the GP head on pre-computed training scores T (n × k row-major)
// and centred targets y_centered (length n). On success, fills every
// field of `out`. Returns P4A_ERR_NUMERICAL_FAILURE if the Cholesky of
// (K + noise²·I) fails (ill-conditioned at very low noise or redundant
// score directions).
[[nodiscard]] p4a_status_t fit_gp_on_scores(
    Context& ctx,
    const std::vector<double>& T,
    std::int32_t n,
    std::int32_t k,
    const std::vector<double>& y_centered,
    double length_scale,
    double noise_level,
    GpHeadResult& out);

// ----------------------------------------------------------------------------
// Top-level GPR-on-PLS result.
// ----------------------------------------------------------------------------
struct GprPlsResult {
    std::int32_t n_features{0};      // p
    std::int32_t n_samples{0};       // n
    std::int32_t n_components{0};    // k
    double length_scale{0.0};
    double noise_level{0.0};
    std::vector<double> rotation_r;          // p × k row-major
    std::vector<double> x_mean;              // length p
    GpHeadResult gp;                          // full GP head state
    std::vector<double> predictions;          // length n — training posterior mean
    std::vector<double> predictive_variance;  // length n — noisy posterior variance
};

// Fits GPR-on-PLS on (X, Y) with the given hyperparameters. `cfg.n_components`
// controls the PLS stage; `cfg.algorithm/solver/deflation` are forced to
// PLS_REGRESSION / SIMPLS / REGRESSION internally. `cfg.center_x` and
// `cfg.center_y` are forced to true. `seed` is reserved for ABI symmetry
// with ensemble methods — the fit is fully deterministic.
//
// Y must be n × 1 (single-target only in Phase 47).
[[nodiscard]] p4a_status_t fit_gpr_pls(
    Context& ctx,
    const Config& cfg,
    const p4a_matrix_view_t& X,
    const p4a_matrix_view_t& Y,
    double length_scale,
    double noise_level,
    std::uint64_t seed,
    GprPlsResult& out);

}  // namespace pls4all::core

#endif  // PLS4ALL_CORE_GPR_PLS_HPP
