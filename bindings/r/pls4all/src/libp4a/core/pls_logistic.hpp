// SPDX-License-Identifier: CECILL-2.1
//
// Internal PLS-logistic classifier kernel.

#pragma once

#include <cstdint>
#include <vector>

#include "n4m/n4m.h"

#include "core/config.hpp"
#include "core/context.hpp"

namespace n4m::core {

struct PlsLogisticResult {
    std::int64_t n_samples{0};
    std::int32_t n_classes{0};
    std::int32_t n_components{0};

    std::vector<std::int32_t> predictions;
    std::vector<double> decision_scores; // row-major n_samples x n_classes
    std::vector<double> probabilities;   // row-major n_samples x n_classes
    std::vector<double> intercepts;      // n_classes - 1 baseline-logit intercepts
    std::vector<double> coefficients;    // row-major (n_classes - 1) x n_components
};

[[nodiscard]] n4m_status_t fit_predict_pls_logistic(
    Context& ctx,
    const Config& cfg,
    const n4m_matrix_view_t& X,
    const n4m_matrix_view_t& labels,
    std::int32_t n_classes,
    PlsLogisticResult& out);

}  // namespace n4m::core
