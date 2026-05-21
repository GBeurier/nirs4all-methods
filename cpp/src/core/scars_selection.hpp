// SPDX-License-Identifier: CECILL-2.1
//
// Internal deterministic SCARS-PLS variable-selection primitive.

#pragma once

#include <cstdint>
#include <vector>

#include "pls4all/p4a.h"

#include "core/config.hpp"
#include "core/context.hpp"
#include "core/validation.hpp"

namespace n4m::core {

struct ScarsSelectionResult {
    std::int32_t n_features{0};
    std::int32_t n_targets{0};
    std::int32_t n_components{0};
    std::int32_t n_iterations{0};
    std::int32_t min_features{0};
    std::int32_t sample_count{0};
    std::int32_t best_iteration{-1};
    double sample_fraction{0.0};
    std::uint64_t seed{0};
    double best_rmse{0.0};

    // Row-major n_iterations x n_features, zeros for inactive variables.
    std::vector<double> coefficient_scores;
    std::vector<double> stability_scores;
    std::vector<double> rmse_by_iteration;
    std::vector<std::int64_t> retained_counts;
    std::vector<std::int64_t> selected_indices;
};

[[nodiscard]] n4m_status_t select_by_scars(Context& ctx,
                                           const Config& cfg,
                                           const n4m_matrix_view_t& X,
                                           const n4m_matrix_view_t& Y,
                                           const ValidationPlan& plan,
                                           std::int32_t n_iterations,
                                           std::int32_t min_features,
                                           double sample_fraction,
                                           std::uint64_t seed,
                                           ScarsSelectionResult& out);

}  // namespace n4m::core
