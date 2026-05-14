// SPDX-License-Identifier: CeCILL-2.1
//
// Internal deterministic IPW-PLS variable-selection primitive.

#pragma once

#include <cstdint>
#include <vector>

#include "pls4all/p4a.h"

#include "core/config.hpp"
#include "core/context.hpp"
#include "core/validation.hpp"

namespace pls4all::core {

struct IpwSelectionResult {
    std::int32_t n_features{0};
    std::int32_t n_targets{0};
    std::int32_t n_components{0};
    std::int32_t n_iterations{0};
    std::int32_t top_k{0};
    std::int32_t best_iteration{-1};
    double damping{0.0};
    double weight_floor{0.0};
    double best_rmse{0.0};

    // Row-major n_iterations x n_features.
    std::vector<double> score_path;
    std::vector<double> weight_path;
    std::vector<double> rmse_by_iteration;

    // Row-major n_iterations x top_k, with each selected subset sorted ascending.
    std::vector<std::int64_t> selected_by_iteration;
    std::vector<std::int64_t> ranking_indices;
    std::vector<std::int64_t> selected_indices;
};

[[nodiscard]] p4a_status_t select_by_ipw(Context& ctx,
                                         const Config& cfg,
                                         const p4a_matrix_view_t& X,
                                         const p4a_matrix_view_t& Y,
                                         const ValidationPlan& plan,
                                         std::int32_t n_iterations,
                                         std::int32_t top_k,
                                         double damping,
                                         double weight_floor,
                                         IpwSelectionResult& out);

}  // namespace pls4all::core
