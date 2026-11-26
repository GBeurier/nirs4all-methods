// SPDX-License-Identifier: CECILL-2.1 OR AGPL-3.0-or-later
//
// Reflectance from raw references and the inverse Kubelka-Munk function,
// reached through the generic estimator roles only.
#pragma once

#include <cstdint>

#include "n4m/n4m.h"

namespace n4m::core {

// out = scale * (X - dark) / (white - dark), column by column. `dark` may be
// null (no dark spectrum). Points whose denominator is not a finite positive
// number are NaN: a saturated or dead detector pixel has no reflectance.
n4m_status_t reflectance_from_references(const n4m_matrix_view_t& X,
                                         const double* white,
                                         const double* dark,
                                         double scale,
                                         n4m_matrix_view_t& out) noexcept;

// Diffuse reflectance R of a Kubelka-Munk value F = (1 - R)^2 / (2 R), the
// root of R^2 - 2 (1 + F) R + 1 = 0 in (0, 1], written
// R = 1 / (1 + F + sqrt(F (F + 2))) to avoid cancellation for large F.
// out = scale * R; F < 0 has no reflectance and gives NaN.
n4m_status_t
from_kubelka_munk(const n4m_matrix_view_t& X, double scale, n4m_matrix_view_t& out) noexcept;

}  // namespace n4m::core
