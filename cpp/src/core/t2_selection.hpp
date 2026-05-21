// SPDX-License-Identifier: CECILL-2.1
//
// Internal deterministic Hotelling T2-PLS variable-selection primitive.

#pragma once

#include <cstdint>
#include <vector>

#include "pls4all/p4a.h"

#include "core/config.hpp"
#include "core/common/context.hpp"
#include "core/validation.hpp"

namespace n4m::core {

struct T2SelectionResult {
    std::int32_t n_features{0};
    std::int32_t n_targets{0};
    std::int32_t n_components{0};
    std::int32_t n_alphas{0};
    std::int32_t min_selected{0};
    std::int32_t best_error_index{-1};
    std::int32_t min_set_index{-1};
    double best_rmse{0.0};
    double min_set_rmse{0.0};

    std::vector<double> t2_scores;
    std::vector<double> ucl_by_alpha;
    std::vector<double> rmse_by_alpha;
    std::vector<std::int64_t> selected_counts;
    std::vector<std::int64_t> selected_mask; // row-major n_alphas x n_features
    std::vector<std::int64_t> selected_indices_best_error;
    std::vector<std::int64_t> selected_indices_min_set;
};

[[nodiscard]] n4m_status_t select_by_t2(Context& ctx,
                                        const Config& cfg,
                                        const n4m_matrix_view_t& X,
                                        const n4m_matrix_view_t& Y,
                                        const ValidationPlan& plan,
                                        const std::vector<double>& alpha,
                                        std::int32_t min_selected,
                                        T2SelectionResult& out);

}  // namespace n4m::core
