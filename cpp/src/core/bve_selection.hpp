// SPDX-License-Identifier: CECILL-2.1
//
// Internal deterministic BVE-PLS variable-selection primitive.

#pragma once

#include <cstdint>
#include <vector>

#include "n4m/n4m.h"

#include "core/config.hpp"
#include "core/common/context.hpp"
#include "core/validation.hpp"

namespace n4m::core {

struct BveSelectionResult {
    std::int32_t n_features{0};
    std::int32_t n_targets{0};
    std::int32_t n_components{0};
    std::int32_t n_steps{0};
    std::int32_t min_features{0};
    std::int32_t best_step{-1};
    double best_rmse{0.0};

    // Row-major n_steps x n_features, zeros for variables not removable at a step.
    std::vector<double> candidate_rmse;
    std::vector<double> rmse_by_step;
    std::vector<std::int64_t> retained_counts;
    std::vector<std::int64_t> removed_indices;
    std::vector<std::int64_t> selected_indices;
};

[[nodiscard]] n4m_status_t select_by_bve(Context& ctx,
                                         const Config& cfg,
                                         const n4m_matrix_view_t& X,
                                         const n4m_matrix_view_t& Y,
                                         const ValidationPlan& plan,
                                         std::int32_t n_steps,
                                         std::int32_t min_features,
                                         BveSelectionResult& out);

}  // namespace n4m::core
