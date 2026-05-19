// SPDX-License-Identifier: CECILL-2.1
//
// Internal deterministic synergy interval PLS selector.

#pragma once

#include <cstdint>
#include <vector>

#include "pls4all/p4a.h"

#include "core/config.hpp"
#include "core/context.hpp"
#include "core/validation.hpp"

namespace pls4all::core {

struct SiplsSelectionResult {
    std::int32_t n_intervals{0};
    std::int32_t interval_width{0};
    std::int32_t combination_size{0};
    std::int32_t n_combinations{0};
    std::int32_t best_combination_index{-1};

    std::vector<std::int64_t> intervals;
    std::vector<std::int64_t> candidate_interval_indices;
    std::vector<double> rmse_by_combination;
    std::vector<std::int64_t> selected_interval_indices;
    std::vector<std::int64_t> selected_feature_indices;
};

[[nodiscard]] p4a_status_t select_by_sipls(
    Context& ctx,
    const Config& cfg,
    const p4a_matrix_view_t& X,
    const p4a_matrix_view_t& Y,
    const ValidationPlan& plan,
    std::int32_t interval_width,
    std::int32_t combination_size,
    SiplsSelectionResult& out);

}  // namespace pls4all::core
