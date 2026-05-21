// SPDX-License-Identifier: CECILL-2.1
//
// Internal deterministic backward interval PLS selector.

#pragma once

#include <cstdint>
#include <vector>

#include "n4m/n4m.h"

#include "core/config.hpp"
#include "core/context.hpp"
#include "core/validation.hpp"

namespace n4m::core {

struct BiplsSelectionResult {
    std::int32_t n_intervals{0};
    std::int32_t interval_width{0};
    std::int32_t min_intervals{0};
    std::int32_t best_step{0};

    std::vector<std::int64_t> intervals;
    std::vector<std::int64_t> removed_intervals;
    std::vector<std::int64_t> active_counts;
    std::vector<double> rmse_path;
    std::vector<std::int64_t> selected_interval_indices;
    std::vector<std::int64_t> selected_feature_indices;
};

[[nodiscard]] n4m_status_t select_by_bipls(
    Context& ctx,
    const Config& cfg,
    const n4m_matrix_view_t& X,
    const n4m_matrix_view_t& Y,
    const ValidationPlan& plan,
    std::int32_t interval_width,
    std::int32_t min_intervals,
    BiplsSelectionResult& out);

}  // namespace n4m::core

