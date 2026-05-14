// SPDX-License-Identifier: CeCILL-2.1
//
// Internal component-count model selection helpers.

#pragma once

#include <cstdint>
#include <vector>

#include "pls4all/p4a.h"

#include "core/config.hpp"
#include "core/context.hpp"
#include "core/metrics.hpp"
#include "core/validation.hpp"

namespace pls4all::core {

struct ComponentCvResult {
    std::int32_t max_components{0};
    std::int32_t best_n_components{0};
    std::vector<RegressionMetrics> metrics_by_component;

    // Row-major max_components x 9: rmse, mae, bias, r2, q2, slope,
    // intercept, rpd, rpiq.
    std::vector<double> metrics_matrix;
};

[[nodiscard]] p4a_status_t cross_validate_component_prefixes(
    Context& ctx,
    const Config& cfg,
    const p4a_matrix_view_t& X,
    const p4a_matrix_view_t& Y,
    const ValidationPlan& plan,
    std::int32_t max_components,
    ComponentCvResult& out);

}  // namespace pls4all::core
