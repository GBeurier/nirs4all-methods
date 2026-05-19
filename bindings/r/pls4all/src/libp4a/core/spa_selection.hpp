// SPDX-License-Identifier: CECILL-2.1
//
// Internal SPA-PLS variable-selection primitive.

#pragma once

#include <cstdint>
#include <vector>

#include "pls4all/p4a.h"

#include "core/config.hpp"
#include "core/context.hpp"

namespace pls4all::core {

struct SpaSelectionResult {
    std::int32_t n_features{0};
    std::int32_t n_targets{0};
    std::int32_t n_components{0};
    std::int32_t top_k{0};

    std::vector<double> coefficient_scores;
    std::vector<double> selection_scores;
    std::vector<std::int64_t> selected_indices;
};

[[nodiscard]] p4a_status_t select_by_spa(
    Context& ctx,
    const Config& cfg,
    const p4a_matrix_view_t& X,
    const p4a_matrix_view_t& Y,
    std::int32_t top_k,
    SpaSelectionResult& out);

}  // namespace pls4all::core
