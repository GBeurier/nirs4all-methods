// SPDX-License-Identifier: CECILL-2.1
//
// Internal deterministic Interval Random Frog (IRF) primitive. Reference:
//   Yun Y.-H., Wang W.-T., Deng B.-C., Lai G.-B., Liu X.-B., Ren D.-B.,
//   Liang Y.-Z., Fan W., Xu Q.-S. (2013). "An efficient method of
//   wavelength interval selection based on random frog for multivariate
//   spectral calibration." Spectrochimica Acta Part A 111, 31–36.
//   Implemented in libPLS 1.95 `irf.m`.

#pragma once

#include <cstdint>
#include <vector>

#include "n4m/n4m.h"

#include "core/config.hpp"
#include "core/context.hpp"
#include "core/validation.hpp"

namespace n4m::core {

struct IrfSelectionResult {
    std::int32_t n_features{0};
    std::int32_t n_targets{0};
    std::int32_t n_components{0};
    std::int32_t n_iterations{0};
    std::int32_t window_size{0};
    std::int32_t initial_intervals{0};
    std::int32_t n_intervals{0};
    std::int32_t top_k{0};
    std::uint64_t seed{0};
    double best_rmse{0.0};

    // Per-interval inclusion frequency over the chain.
    std::vector<double> probability;          // size n_intervals
    // Iteration trace (RMSEs) and accepted subset size.
    std::vector<double> rmse_by_iteration;    // size n_iterations
    std::vector<std::int64_t> subset_sizes;   // size n_iterations
    // Top-K interval indices (ranked by inclusion probability).
    std::vector<std::int64_t> top_k_intervals;
    // Final selected feature indices (union of best-scoring intervals).
    std::vector<std::int64_t> selected_indices;
};

[[nodiscard]] n4m_status_t select_by_irf(
    Context& ctx,
    const Config& cfg,
    const n4m_matrix_view_t& X,
    const n4m_matrix_view_t& Y,
    const ValidationPlan& plan,
    std::int32_t n_iterations,
    std::int32_t window_size,
    std::int32_t initial_intervals,
    std::int32_t top_k,
    std::uint64_t seed,
    IrfSelectionResult& out);

}  // namespace n4m::core
