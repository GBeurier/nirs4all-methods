// SPDX-License-Identifier: CECILL-2.1
//
// Internal deterministic ST-PLS score-threshold variable-selection primitive.

#pragma once

#include <cstdint>
#include <vector>

#include "pls4all/p4a.h"

#include "core/config.hpp"
#include "core/common/context.hpp"
#include "core/validation.hpp"

namespace n4m::core {

struct StSelectionResult {
    std::int32_t n_features{0};
    std::int32_t n_targets{0};
    std::int32_t n_components{0};
    std::int32_t n_thresholds{0};
    std::int32_t min_selected{0};
    std::int32_t best_threshold_index{-1};
    double best_threshold{0.0};
    double best_rmse{0.0};

    std::vector<double> coefficient_scores;
    std::vector<double> thresholds;
    std::vector<double> rmse_by_threshold;
    std::vector<std::int64_t> selected_counts;
    std::vector<std::int64_t> selected_masks;  // row-major n_thresholds x n_features
    std::vector<std::int64_t> ranking_indices;
    std::vector<std::int64_t> selected_indices;
};

[[nodiscard]] n4m_status_t select_by_st(Context& ctx,
                                        const Config& cfg,
                                        const n4m_matrix_view_t& X,
                                        const n4m_matrix_view_t& Y,
                                        const ValidationPlan& plan,
                                        const std::vector<double>& thresholds,
                                        std::int32_t min_selected,
                                        StSelectionResult& out);

}  // namespace n4m::core
