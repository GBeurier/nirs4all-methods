// SPDX-License-Identifier: CECILL-2.1
//
// Internal UVE / MCUVE-style variable selectors.

#pragma once

#include <cstdint>
#include <vector>

#include "n4m/n4m.h"

#include "core/config.hpp"
#include "core/context.hpp"
#include "core/validation.hpp"

namespace n4m::core {

struct UveSelectionResult {
    std::int32_t n_features{0};
    std::int32_t n_targets{0};
    std::int32_t n_repeats{0};
    std::int32_t n_noise_features{0};
    std::int32_t selected_count{0};
    std::uint64_t noise_seed{0};
    double noise_threshold{0.0};

    std::vector<double> real_stability_scores;
    std::vector<double> noise_stability_scores;
    std::vector<std::int64_t> selected_indices;
};

[[nodiscard]] n4m_status_t select_by_uve(
    Context& ctx,
    const Config& cfg,
    const n4m_matrix_view_t& X,
    const n4m_matrix_view_t& Y,
    const ValidationPlan& plan,
    std::int32_t noise_features,
    std::uint64_t noise_seed,
    UveSelectionResult& out);

}  // namespace n4m::core
