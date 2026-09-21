// SPDX-License-Identifier: CECILL-2.1
#pragma once
#include <vector>

#include "core/common/context.hpp"
#include "core/operator_bank.hpp"
namespace n4m::core {
struct AomCalibrationResult {
    std::vector<double> scores, coefficients, state;
    std::int32_t branch{0}, chain{0}, parameter{0};
};
n4m_status_t fit_aom_calibration(Context&,
                                 const n4m_matrix_view_t&,
                                 const n4m_matrix_view_t&,
                                 const OperatorBank&,
                                 const std::int32_t*,
                                 std::int32_t,
                                 const std::int32_t*,
                                 std::int32_t,
                                 const std::int32_t*,
                                 std::int32_t,
                                 std::int32_t,
                                 bool,
                                 std::int32_t,
                                 const double*,
                                 std::int32_t,
                                 AomCalibrationResult&);
n4m_status_t predict_aom_calibration(
    const n4m_matrix_view_t&, int, const double*, const double*, n4m_matrix_view_t&);
}  // namespace n4m::core
