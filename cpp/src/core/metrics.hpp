// SPDX-License-Identifier: CECILL-2.1
//
// Internal validation and regression metric kernels.

#pragma once

#include <cstdint>

#include "pls4all/p4a.h"

#include "core/context.hpp"

namespace n4m::core {

struct RegressionMetrics {
    std::int64_t rows{0};
    std::int64_t cols{0};
    std::int64_t count{0};

    double rmse{0.0};
    double mae{0.0};
    double bias{0.0};       // mean(predicted - observed)
    double r2{0.0};
    double q2{0.0};
    double slope{0.0};      // observed = intercept + slope * predicted
    double intercept{0.0};
    double rpd{0.0};
    double rpiq{0.0};
};

[[nodiscard]] n4m_status_t compute_regression_metrics(
    Context& ctx,
    const n4m_matrix_view_t& observed,
    const n4m_matrix_view_t& predicted,
    RegressionMetrics& out);

}  // namespace n4m::core
