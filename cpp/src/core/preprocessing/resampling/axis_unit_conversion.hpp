// SPDX-License-Identifier: CECILL-2.1 OR AGPL-3.0-or-later
//
// Spectral axis unit conversion between vacuum wavelengths (nm, um) and
// wavenumbers (cm-1), reached through the generic estimator roles only.
#pragma once

#include <cstdint>
#include <vector>

#include "n4m/n4m.h"

namespace n4m::core {

// Order of the `source_unit` / `target_unit` enum choices.
enum AxisUnit : std::int32_t { kAxisNm = 0, kAxisUm = 1, kAxisWavenumberCm = 2 };

// How a source axis maps onto the target unit: output column j holds input
// column order[j] times factor[j], and `axis` is strictly ascending.
struct AxisUnitPlan {
    std::vector<std::int64_t> order;
    std::vector<double> axis;
    std::vector<double> factor;
};

// Plans the conversion of `n` axis values. Values must be finite, positive and
// distinct. With `spectral_density`, factor is the Jacobian |d source / d
// target| that turns a density per source unit into a density per target
// unit; otherwise it is 1 (absorbance, reflectance, transmittance, counts).
n4m_status_t plan_axis_unit_conversion(const double* axis,
                                       std::int64_t n,
                                       std::int32_t source,
                                       std::int32_t target,
                                       bool spectral_density,
                                       AxisUnitPlan& plan);

// out[:, j] = X[:, plan.order[j]] * plan.factor[j].
n4m_status_t apply_axis_unit_conversion(const AxisUnitPlan& plan,
                                        const n4m_matrix_view_t& X,
                                        n4m_matrix_view_t& out) noexcept;

}  // namespace n4m::core
