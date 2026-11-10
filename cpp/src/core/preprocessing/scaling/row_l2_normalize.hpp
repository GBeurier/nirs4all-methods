// SPDX-License-Identifier: CECILL-2.1 OR AGPL-3.0-or-later
#pragma once

#include "n4m/n4m.h"

namespace n4m::core {

// Internal role kernel. The generic estimator boundary validates views and shapes.
n4m_status_t row_l2_validate_input(const n4m_matrix_view_t& X) noexcept;
// Inputs must be finite (checked once by the role adapter before calling).
n4m_status_t row_l2_normalize(const n4m_matrix_view_t& X, n4m_matrix_view_t& out) noexcept;

}  // namespace n4m::core
