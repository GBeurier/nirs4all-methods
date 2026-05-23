// SPDX-License-Identifier: CECILL-2.1
//
// Internal deterministic GA-PLS variable-selection primitive.

#pragma once

#include <cstdint>
#include <vector>

#include "n4m/n4m.h"

#include "core/config.hpp"
#include "core/common/context.hpp"
#include "core/validation.hpp"

namespace n4m::core {

struct GaSelectionResult {
    std::int32_t n_features{0};
    std::int32_t n_targets{0};
    std::int32_t n_components{0};
    std::int32_t n_generations{0};
    std::int32_t population_size{0};
    std::int32_t min_features{0};
    std::int32_t max_features{0};
    double mutation_rate{0.0};
    std::uint64_t seed{0};
    double best_rmse{0.0};

    std::vector<double> global_scores;
    std::vector<double> inclusion_frequencies;
    std::vector<double> best_rmse_by_generation;
    std::vector<double> mean_rmse_by_generation;
    std::vector<std::int64_t> best_subset_sizes;
    std::vector<std::int64_t> selected_indices;
};

[[nodiscard]] n4m_status_t select_by_ga(Context& ctx,
                                        const Config& cfg,
                                        const n4m_matrix_view_t& X,
                                        const n4m_matrix_view_t& Y,
                                        const ValidationPlan& plan,
                                        std::int32_t n_generations,
                                        std::int32_t population_size,
                                        std::int32_t min_features,
                                        std::int32_t max_features,
                                        double mutation_rate,
                                        std::uint64_t seed,
                                        GaSelectionResult& out);

}  // namespace n4m::core
