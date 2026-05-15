// SPDX-License-Identifier: CeCILL-2.1
//
// Internal AOM global-selection kernels.

#pragma once

#include <cstdint>
#include <vector>

#include "pls4all/p4a.h"

#include "core/config.hpp"
#include "core/context.hpp"
#include "core/operator_bank.hpp"
#include "core/validation.hpp"

namespace pls4all::core {

struct AomGlobalSelectionResult {
    std::int32_t n_operators{0};
    std::int32_t max_components{0};
    std::int32_t selected_operator_index{0};
    std::int32_t selected_n_components{0};
    double best_score{0.0};

    std::vector<std::int64_t> operator_kinds;
    std::vector<double> operator_scores;  // n_operators
    std::vector<double> rmse_curves;      // row-major n_operators x max_components
    std::vector<double> predictions;      // row-major n_samples x n_targets
};

struct AomPerComponentSelectionResult {
    std::int32_t n_operators{0};
    std::int32_t max_components{0};
    std::int32_t selected_n_components{0};
    double best_score{0.0};

    std::vector<std::int64_t> operator_kinds;
    std::vector<std::int64_t> selected_operator_indices;  // selected_n_components
    std::vector<double> component_scores;                 // row-major max_components x n_operators
    std::vector<double> prefix_scores;                    // max_components
    std::vector<double> predictions;                      // row-major n_samples x n_targets
};

[[nodiscard]] p4a_status_t select_aom_global(
    Context& ctx,
    const Config& cfg,
    const OperatorBank& bank,
    const p4a_matrix_view_t& X,
    const p4a_matrix_view_t& Y,
    const ValidationPlan& plan,
    std::int32_t max_components,
    AomGlobalSelectionResult& out);

[[nodiscard]] p4a_status_t select_aom_per_component(
    Context& ctx,
    const Config& cfg,
    const OperatorBank& bank,
    const p4a_matrix_view_t& X,
    const p4a_matrix_view_t& Y,
    const ValidationPlan& plan,
    std::int32_t max_components,
    AomPerComponentSelectionResult& out);

}  // namespace pls4all::core
