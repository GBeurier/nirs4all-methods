// SPDX-License-Identifier: CECILL-2.1 OR AGPL-3.0-or-later
#pragma once
#include <vector>
#include "n4m/fanova.h"

namespace n4m::fanova {
struct Result {
    std::vector<double> main_effects;
    std::vector<double> pair_effects;
    n4m_fanova_summary_v1_t summary;
};
n4m_status_t fit(const n4m_matrix_view_t& X, const n4m_matrix_view_t& y,
                 const n4m_matrix_view_t& bounds, const uint32_t* types,
                 const n4m_matrix_view_t* X_validation, const n4m_matrix_view_t* y_validation,
                 const n4m_fanova_options_v1_t& options, Result& result);
}  // namespace n4m::fanova
