// SPDX-License-Identifier: CECILL-2.1
//
// Internal AOM global-selection kernels.

#pragma once

#include <cstdint>
#include <vector>

#include "n4m/n4m.h"

#include "core/config.hpp"
#include "core/context.hpp"
#include "core/operator_bank.hpp"
#include "core/validation.hpp"

namespace n4m::core {

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

[[nodiscard]] n4m_status_t select_aom_global(
    Context& ctx,
    const Config& cfg,
    const OperatorBank& bank,
    const n4m_matrix_view_t& X,
    const n4m_matrix_view_t& Y,
    const ValidationPlan& plan,
    std::int32_t max_components,
    AomGlobalSelectionResult& out);

[[nodiscard]] n4m_status_t select_aom_per_component(
    Context& ctx,
    const Config& cfg,
    const OperatorBank& bank,
    const n4m_matrix_view_t& X,
    const n4m_matrix_view_t& Y,
    const ValidationPlan& plan,
    std::int32_t max_components,
    AomPerComponentSelectionResult& out);

// Public-ABI-side wrappers. These hold the row/col shape of `predictions`
// alongside the internal selection result and expose typed buffer slices so
// the C ABI does not need to leak `std::vector` to consumers.
struct PublicAomGlobalResult {
    AomGlobalSelectionResult inner;
    std::int64_t predictions_rows{0};
    std::int64_t predictions_cols{0};
    std::vector<n4m_operator_kind_t> operator_kinds_typed;
};

struct PublicAomPerComponentResult {
    AomPerComponentSelectionResult inner;
    std::int64_t predictions_rows{0};
    std::int64_t predictions_cols{0};
    std::vector<n4m_operator_kind_t> operator_kinds_typed;
    std::vector<std::int32_t> selected_operator_indices_i32;
};

}  // namespace n4m::core

struct n4m_aom_global_result_s
    : public ::n4m::core::PublicAomGlobalResult {};
struct n4m_aom_per_component_result_s
    : public ::n4m::core::PublicAomPerComponentResult {};
