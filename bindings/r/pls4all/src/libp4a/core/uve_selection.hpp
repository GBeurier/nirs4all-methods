// SPDX-License-Identifier: CECILL-2.1
//
// Internal UVE / MCUVE-style variable selectors.

#pragma once

#include <cstdint>
#include <vector>

#include "pls4all/p4a.h"

#include "core/config.hpp"
#include "core/context.hpp"
#include "core/validation.hpp"

namespace pls4all::core {

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

[[nodiscard]] p4a_status_t select_by_uve(
    Context& ctx,
    const Config& cfg,
    const p4a_matrix_view_t& X,
    const p4a_matrix_view_t& Y,
    const ValidationPlan& plan,
    std::int32_t noise_features,
    std::uint64_t noise_seed,
    UveSelectionResult& out);

}  // namespace pls4all::core
