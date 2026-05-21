// SPDX-License-Identifier: CECILL-2.1
//
// Internal deterministic ensemble MC-UVE selector.

#pragma once

#include <cstdint>
#include <vector>

#include "n4m/n4m.h"

#include "core/config.hpp"
#include "core/context.hpp"
#include "core/validation.hpp"

namespace n4m::core {

struct EmcuveSelectionResult {
    std::int32_t n_features{0};
    std::int32_t n_targets{0};
    std::int32_t n_repeats{0};
    std::int32_t n_noise_features{0};
    std::int32_t n_ensembles{0};
    std::int32_t selected_count{0};
    std::uint64_t noise_seed{0};
    double vote_threshold{0.0};

    std::vector<double> mean_real_stability_scores;
    std::vector<double> vote_frequencies;
    std::vector<double> noise_thresholds;
    std::vector<std::int64_t> selected_indices;
};

[[nodiscard]] n4m_status_t select_by_emcuve(
    Context& ctx,
    const Config& cfg,
    const n4m_matrix_view_t& X,
    const n4m_matrix_view_t& Y,
    const ValidationPlan& plan,
    std::int32_t noise_features,
    std::uint64_t noise_seed,
    std::int32_t n_ensembles,
    double vote_threshold,
    EmcuveSelectionResult& out);

}  // namespace n4m::core

