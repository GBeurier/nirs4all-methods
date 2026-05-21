// SPDX-License-Identifier: CECILL-2.1
//
// Internal deterministic WVC-PLS variable-selection primitive.

#pragma once

#include <cstdint>
#include <vector>

#include "pls4all/p4a.h"

#include "core/common/context.hpp"

namespace n4m::core {

struct WvcSelectionResult {
    std::int32_t n_features{0};
    std::int32_t n_targets{0};
    std::int32_t n_components{0};
    std::int32_t top_k{0};
    std::int32_t normalize{0};

    std::vector<double> weights_w;
    std::vector<double> loadings_p;
    std::vector<double> y_loadings_q;
    std::vector<double> wvc_scores;
    std::vector<double> final_scores;
    std::vector<std::int64_t> selected_indices;
};

struct WvcThresholdSelectionResult {
    std::int32_t n_features{0};
    std::int32_t n_targets{0};
    std::int32_t n_components{0};
    std::int32_t min_selected{0};
    std::int32_t normalize{0};
    double score_threshold{0.0};
    double threshold_factor{0.0};
    double mean_score{0.0};
    double effective_threshold{0.0};

    std::vector<double> final_scores;
    std::vector<std::int64_t> ranked_indices;
    std::vector<std::int64_t> selected_indices;
};

[[nodiscard]] n4m_status_t select_by_wvc(Context& ctx,
                                         const n4m_matrix_view_t& X,
                                         const n4m_matrix_view_t& Y,
                                         std::int32_t n_components,
                                         std::int32_t top_k,
                                         bool normalize,
                                         WvcSelectionResult& out);

[[nodiscard]] n4m_status_t select_by_wvc_threshold(Context& ctx,
                                                   const n4m_matrix_view_t& X,
                                                   const n4m_matrix_view_t& Y,
                                                   std::int32_t n_components,
                                                   bool normalize,
                                                   double score_threshold,
                                                   double threshold_factor,
                                                   std::int32_t min_selected,
                                                   WvcThresholdSelectionResult& out);

}  // namespace n4m::core
