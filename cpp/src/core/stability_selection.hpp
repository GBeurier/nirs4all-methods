// SPDX-License-Identifier: CECILL-2.1
//
// Internal Monte-Carlo coefficient-stability selectors.

#pragma once

#include <cstdint>
#include <vector>

#include "n4m/n4m.h"

#include "core/config.hpp"
#include "core/common/context.hpp"
#include "core/validation.hpp"

namespace n4m::core {

struct StabilitySelectionResult {
    std::int32_t n_features{0};
    std::int32_t n_targets{0};
    std::int32_t n_repeats{0};
    std::int32_t top_k{0};

    // Row-major n_features x n_targets.
    std::vector<double> mean_coefficients;
    std::vector<double> std_coefficients;

    // Per-feature max abs(mean/std) across targets, then exact top-k indices.
    std::vector<double> stability_scores;
    std::vector<std::int64_t> selected_indices;
};

[[nodiscard]] n4m_status_t select_by_coefficient_stability(
    Context& ctx,
    const Config& cfg,
    const n4m_matrix_view_t& X,
    const n4m_matrix_view_t& Y,
    const ValidationPlan& plan,
    std::int32_t top_k,
    StabilitySelectionResult& out);

}  // namespace n4m::core
