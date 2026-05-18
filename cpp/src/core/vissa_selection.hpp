// SPDX-License-Identifier: CECILL-2.1
//
// VISSA-PLS (§49) — Variable Iterative Space Shrinkage Approach.
// Weighted Binary Matrix Sampling (WBMS) iteratively shrinks the
// variable inclusion probabilities by averaging the masks of the top-K
// best-scoring submodels per iteration.
//
// Reference: Deng B., Yun Y., Liang Y. (2014). "A novel variable
// selection approach that iteratively optimizes variable space using
// weighted binary matrix sampling." Anal. Chim. Acta 838:27-40.
// (Paper-only — no widely-installable external port.)

#pragma once

#include <cstdint>
#include <vector>

#include "pls4all/p4a.h"

#include "core/config.hpp"
#include "core/context.hpp"
#include "core/validation.hpp"

namespace pls4all::core {

struct VissaSelectionResult {
    std::int32_t n_features{0};
    std::int32_t n_targets{0};
    std::int32_t n_components{0};
    std::int32_t n_iterations{0};
    std::int32_t n_submodels{0};
    double ratio_kept{0.0};
    double threshold{0.0};
    double floor_probability{0.0};
    std::uint64_t seed{0};
    double best_rmse{0.0};

    std::vector<double> final_probabilities;     // length p
    std::vector<double> inclusion_frequencies;   // alias (length p)
    std::vector<double> best_rmse_by_iteration;  // length n_iterations
    std::vector<double> mean_rmse_by_iteration;  // length n_iterations
    std::vector<double> top_k_per_iteration;     // n_iterations × p row-major
    std::vector<std::int64_t> selected_indices;  // {j : w_j > threshold}
};

[[nodiscard]] p4a_status_t select_by_vissa(
    Context& ctx,
    const Config& cfg,
    const p4a_matrix_view_t& X,
    const p4a_matrix_view_t& Y,
    const ValidationPlan& plan,
    std::int32_t n_iterations,
    std::int32_t n_submodels,
    double ratio_kept,
    double threshold,
    double floor_probability,
    std::uint64_t seed,
    VissaSelectionResult& out);

}  // namespace pls4all::core
