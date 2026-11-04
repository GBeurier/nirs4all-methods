// SPDX-License-Identifier: CECILL-2.1 OR AGPL-3.0-or-later
#pragma once
#include <array>
#include <vector>

#include "n4m/ranking.h"
namespace n4m::ranking {
struct Result {
    std::vector<double> candidates;
    std::array<double, 9> summary;
};
n4m_status_t compute(const n4m_matrix_view_t& scores,
                     const uint8_t* present,
                     const n4m_ranking_options_v1_t& options,
                     Result& out);
n4m_status_t relative_gain(const double* baseline,
                           const double* candidate,
                           const uint8_t* present,
                           const double* variances,
                           int64_t n,
                           std::array<double, 8>& out);
}  // namespace n4m::ranking
