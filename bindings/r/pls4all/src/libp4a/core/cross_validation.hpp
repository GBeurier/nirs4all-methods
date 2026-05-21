// SPDX-License-Identifier: CECILL-2.1
//
// Internal cross-validation orchestration over deterministic validation plans.

#pragma once

#include <cstdint>
#include <vector>

#include "n4m/n4m.h"

#include "core/config.hpp"
#include "core/context.hpp"
#include "core/metrics.hpp"
#include "core/validation.hpp"

namespace n4m::core {

struct CrossValidationResult {
    std::int64_t n_samples{0};
    std::int64_t n_targets{0};
    std::int64_t n_folds{0};

    // Row-major n_samples x n_targets, stored in original sample order.
    std::vector<double> predictions;

    // CSR-style flattening of the fold test indices, preserving plan order.
    std::vector<std::int64_t> test_offsets;
    std::vector<std::int64_t> test_indices;

    RegressionMetrics metrics;
};

[[nodiscard]] n4m_status_t cross_validate_regression(
    Context& ctx,
    const Config& cfg,
    const n4m_matrix_view_t& X,
    const n4m_matrix_view_t& Y,
    const ValidationPlan& plan,
    CrossValidationResult& out);

}  // namespace n4m::core
