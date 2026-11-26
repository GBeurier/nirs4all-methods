// SPDX-License-Identifier: CECILL-2.1 OR AGPL-3.0-or-later
#include "core/preprocessing/resampling/axis_unit_conversion.hpp"

#include <algorithm>
#include <cmath>
#include <numeric>

namespace n4m::core {

namespace {

// Target value of x and |d source / d target| at x. Reciprocal pairs use
// their exact constant (1 cm-1 = 1e7 nm-1 = 1e4 um-1) in one division.
void convert(double x, std::int32_t source, std::int32_t target, double& value, double& jacobian) {
    if (source == target) {
        value = x;
        jacobian = 1.0;
        return;
    }
    if (source == kAxisWavenumberCm || target == kAxisWavenumberCm) {
        const double constant =
            (source == kAxisNm || target == kAxisNm) ? 1e7 : 1e4;  // nm or um <-> cm-1
        value = constant / x;
        // source = constant / target, so |d source / d target| = source^2 / constant.
        jacobian = x * x / constant;
        return;
    }
    if (source == kAxisNm) {  // nm -> um
        value = x / 1000.0;
        jacobian = 1000.0;
    } else {  // um -> nm
        value = x * 1000.0;
        jacobian = 1.0 / 1000.0;
    }
}

}  // namespace

n4m_status_t plan_axis_unit_conversion(const double* axis,
                                       std::int64_t n,
                                       std::int32_t source,
                                       std::int32_t target,
                                       bool spectral_density,
                                       AxisUnitPlan& plan) {
    if (axis == nullptr || n <= 0)
        return N4M_ERR_INVALID_ARGUMENT;
    const auto size = static_cast<std::size_t>(n);
    std::vector<double> converted(size), jacobian(size);
    for (std::size_t i = 0; i < size; ++i) {
        if (!std::isfinite(axis[i]) || axis[i] <= 0.0)
            return N4M_ERR_INVALID_ARGUMENT;
        convert(axis[i], source, target, converted[i], jacobian[i]);
        if (!std::isfinite(converted[i]) || converted[i] <= 0.0)
            return N4M_ERR_INVALID_ARGUMENT;
    }
    std::vector<std::int64_t> order(size);
    std::iota(order.begin(), order.end(), std::int64_t{0});
    std::stable_sort(order.begin(), order.end(), [&](std::int64_t a, std::int64_t b) {
        return converted[static_cast<std::size_t>(a)] < converted[static_cast<std::size_t>(b)];
    });
    plan.order = order;
    plan.axis.resize(size);
    plan.factor.resize(size);
    for (std::size_t j = 0; j < size; ++j) {
        const auto i = static_cast<std::size_t>(order[j]);
        plan.axis[j] = converted[i];
        plan.factor[j] = spectral_density ? jacobian[i] : 1.0;
        if (j > 0 && !(plan.axis[j] > plan.axis[j - 1]))
            return N4M_ERR_INVALID_ARGUMENT;
    }
    return N4M_OK;
}

n4m_status_t apply_axis_unit_conversion(const AxisUnitPlan& plan,
                                        const n4m_matrix_view_t& X,
                                        n4m_matrix_view_t& out) noexcept {
    const auto* values = static_cast<const double*>(X.data);
    auto* result = static_cast<double*>(out.data);
    for (std::int64_t row = 0; row < X.rows; ++row) {
        for (std::size_t j = 0; j < plan.order.size(); ++j) {
            result[row * out.row_stride + static_cast<std::int64_t>(j) * out.col_stride] =
                values[row * X.row_stride + plan.order[j] * X.col_stride] * plan.factor[j];
        }
    }
    return N4M_OK;
}

}  // namespace n4m::core
