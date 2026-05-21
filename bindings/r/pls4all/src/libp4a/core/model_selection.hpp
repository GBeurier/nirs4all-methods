// SPDX-License-Identifier: CECILL-2.1
//
// Internal component-count model selection helpers.

#pragma once

#include <cstdint>
#include <vector>

#include "n4m/n4m.h"

#include "core/config.hpp"
#include "core/context.hpp"
#include "core/metrics.hpp"
#include "core/validation.hpp"

namespace n4m::core {

struct ComponentCvResult {
    std::int32_t max_components{0};
    std::int32_t best_n_components{0};
    std::vector<RegressionMetrics> metrics_by_component;

    // Row-major max_components x 9: rmse, mae, bias, r2, q2, slope,
    // intercept, rpd, rpiq.
    std::vector<double> metrics_matrix;

    // Row-major max_components x n_folds matrix of per-fold RMSE values.
    // Populated by cross_validate_component_prefixes and consumed by the
    // one-SE selection rule below.
    std::vector<double> fold_rmse_matrix;
    std::int32_t        n_folds{0};

    // Smallest component count whose mean fold RMSE is within one standard
    // error of the best mean fold RMSE. Equal to best_n_components when no
    // smaller candidate qualifies. Populated by select_one_se_components.
    std::int32_t one_se_n_components{0};

    // Standard error of the fold RMSEs at best_n_components.
    double one_se_standard_error{0.0};
    double one_se_threshold{0.0};
};

[[nodiscard]] n4m_status_t cross_validate_component_prefixes(
    Context& ctx,
    const Config& cfg,
    const n4m_matrix_view_t& X,
    const n4m_matrix_view_t& Y,
    const ValidationPlan& plan,
    std::int32_t max_components,
    ComponentCvResult& out);

// Apply the one-SE rule on top of a populated ComponentCvResult. Mutates
// the result in place: sets one_se_n_components, one_se_standard_error,
// one_se_threshold. Requires `result.fold_rmse_matrix` to be populated by
// `cross_validate_component_prefixes`.
[[nodiscard]] n4m_status_t select_one_se_components(
    Context& ctx,
    ComponentCvResult& result);

}  // namespace n4m::core
