// SPDX-License-Identifier: CECILL-2.1
//
// Internal deterministic PLS randomisation-test selector.

#pragma once

#include <cstdint>
#include <vector>

#include "pls4all/p4a.h"

#include "core/config.hpp"
#include "core/common/context.hpp"

namespace n4m::core {

struct RandomizationSelectionResult {
    std::int32_t n_features{0};
    std::int32_t n_targets{0};
    std::int32_t n_permutations{0};
    std::int32_t selected_count{0};
    std::uint64_t randomization_seed{0};
    double alpha{0.0};

    std::vector<double> observed_scores;
    std::vector<double> p_values;
    std::vector<std::int64_t> exceedance_counts;
    std::vector<std::int64_t> selected_indices;
};

[[nodiscard]] n4m_status_t select_by_randomization_test(
    Context& ctx,
    const Config& cfg,
    const n4m_matrix_view_t& X,
    const n4m_matrix_view_t& Y,
    std::int32_t n_permutations,
    std::uint64_t randomization_seed,
    double alpha,
    RandomizationSelectionResult& out);

}  // namespace n4m::core

