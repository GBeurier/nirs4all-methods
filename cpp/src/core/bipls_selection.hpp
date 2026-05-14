// SPDX-License-Identifier: CeCILL-2.1
//
// Internal deterministic backward interval PLS selector.

#pragma once

#include <cstdint>
#include <vector>

#include "pls4all/p4a.h"

#include "core/config.hpp"
#include "core/context.hpp"
#include "core/validation.hpp"

namespace pls4all::core {

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

[[nodiscard]] p4a_status_t select_by_bipls(
    Context& ctx,
    const Config& cfg,
    const p4a_matrix_view_t& X,
    const p4a_matrix_view_t& Y,
    const ValidationPlan& plan,
    std::int32_t interval_width,
    std::int32_t min_intervals,
    BiplsSelectionResult& out);

}  // namespace pls4all::core

