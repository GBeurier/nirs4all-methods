// SPDX-License-Identifier: CECILL-2.1
//
// Internal validation and regression metric kernels.

#pragma once

#include <cstdint>

#include "pls4all/p4a.h"

#include "core/context.hpp"

namespace pls4all::core {

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

[[nodiscard]] p4a_status_t compute_regression_metrics(
    Context& ctx,
    const p4a_matrix_view_t& observed,
    const p4a_matrix_view_t& predicted,
    RegressionMetrics& out);

}  // namespace pls4all::core
