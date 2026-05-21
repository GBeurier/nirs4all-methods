// SPDX-License-Identifier: CECILL-2.1
//
// Internal deterministic Random Frog PLS variable-selection primitive.

#pragma once

#include <cstdint>
#include <vector>

#include "pls4all/p4a.h"

#include "core/config.hpp"
#include "core/context.hpp"
#include "core/validation.hpp"

namespace n4m::core {

struct RandomFrogSelectionResult {
    std::int32_t n_features{0};
    std::int32_t n_targets{0};
    std::int32_t n_components{0};
    std::int32_t n_iterations{0};
    std::int32_t initial_size{0};
    std::int32_t min_size{0};
    std::int32_t max_size{0};
    std::int32_t top_k{0};
    std::uint64_t seed{0};
    double best_rmse{0.0};

    std::vector<double> global_scores;
    std::vector<double> inclusion_frequencies;
    std::vector<double> rmse_by_iteration;
    std::vector<std::int64_t> subset_sizes;
    std::vector<std::int64_t> selected_indices;
    std::vector<std::int64_t> best_indices;
};

[[nodiscard]] n4m_status_t select_by_random_frog(
    Context& ctx,
    const Config& cfg,
    const n4m_matrix_view_t& X,
    const n4m_matrix_view_t& Y,
    const ValidationPlan& plan,
    std::int32_t n_iterations,
    std::int32_t initial_size,
    std::int32_t min_size,
    std::int32_t max_size,
    std::int32_t top_k,
    std::uint64_t seed,
    RandomFrogSelectionResult& out);

}  // namespace n4m::core
