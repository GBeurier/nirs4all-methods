// SPDX-License-Identifier: CECILL-2.1
//
// Internal deterministic REP-PLS variable-selection primitive.

#pragma once

#include <cstdint>
#include <vector>

#include "pls4all/p4a.h"

#include "core/config.hpp"
#include "core/common/context.hpp"
#include "core/validation.hpp"

namespace n4m::core {

struct RepSelectionResult {
    std::int32_t n_features{0};
    std::int32_t n_targets{0};
    std::int32_t n_components{0};
    std::int32_t n_steps{0};
    std::int32_t min_features{0};
    std::int32_t remove_count{0};
    std::int32_t best_step{-1};
    double best_rmse{0.0};

    // Row-major n_steps x n_features, zeros for inactive variables.
    std::vector<double> coefficient_scores;
    std::vector<double> rmse_by_step;
    std::vector<std::int64_t> retained_counts;
    std::vector<std::int64_t> removed_counts;
    std::vector<std::int64_t> removed_indices;
    std::vector<std::int64_t> selected_indices;
};

[[nodiscard]] n4m_status_t select_by_rep(Context& ctx,
                                         const Config& cfg,
                                         const n4m_matrix_view_t& X,
                                         const n4m_matrix_view_t& Y,
                                         const ValidationPlan& plan,
                                         std::int32_t n_steps,
                                         std::int32_t min_features,
                                         std::int32_t remove_count,
                                         RepSelectionResult& out);

}  // namespace n4m::core
