// SPDX-License-Identifier: CECILL-2.1
//
// Internal interval / moving-window variable-selection scans.

#pragma once

#include <cstdint>
#include <vector>

#include "pls4all/p4a.h"

#include "core/config.hpp"
#include "core/context.hpp"
#include "core/metrics.hpp"
#include "core/validation.hpp"

namespace n4m::core {

struct IntervalSelectionResult {
    std::int32_t interval_width{0};
    std::int32_t step{0};
    std::int32_t best_interval_index{-1};
    std::int64_t best_start{0};
    std::int64_t best_length{0};

    // Row-major n_intervals x 2: start, length.
    std::vector<std::int64_t> intervals;
    std::vector<double> rmse;
    std::vector<RegressionMetrics> metrics_by_interval;

    // Row-major n_intervals x 9: rmse, mae, bias, r2, q2, slope,
    // intercept, rpd, rpiq.
    std::vector<double> metrics_matrix;
};

[[nodiscard]] n4m_status_t cross_validate_intervals(
    Context& ctx,
    const Config& cfg,
    const n4m_matrix_view_t& X,
    const n4m_matrix_view_t& Y,
    const ValidationPlan& plan,
    std::int32_t interval_width,
    std::int32_t step,
    IntervalSelectionResult& out);

}  // namespace n4m::core
