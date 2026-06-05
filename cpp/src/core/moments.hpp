// SPDX-License-Identifier: CECILL-2.1
//
// Raw sufficient statistics for moment-based linear screens.
//
// The raw buffers are additive over rows, so CV folds can be handled by
// subtracting held-out moments from all-sample moments. Centered moments are
// then recomputed from the raw sums for the remaining train rows.

#pragma once

#include <cstdint>
#include <vector>

#include "n4m/n4m.h"

#include "core/common/context.hpp"

namespace n4m::core {

struct MomentStats {
    std::int64_t n_samples{0};
    std::int64_t n_features{0};
    std::int64_t n_targets{0};

    std::vector<double> x_sum;  // 1 x p
    std::vector<double> y_sum;  // 1 x q
    std::vector<double> xtx;    // p x p, raw X'X
    std::vector<double> xty;    // p x q, raw X'Y
    std::vector<double> yty;    // q x q, raw Y'Y

    std::vector<double> x_mean;  // 1 x p
    std::vector<double> y_mean;  // 1 x q
    std::vector<double> cxx;     // p x p, centered X'X
    std::vector<double> cxy;     // p x q, centered X'Y
    std::vector<double> cyy;     // q x q, centered Y'Y
};

[[nodiscard]] n4m_status_t compute_moments(
    Context& ctx,
    const n4m_matrix_view_t& X,
    const n4m_matrix_view_t& Y,
    MomentStats& out);

[[nodiscard]] n4m_status_t compute_moments_subset(
    Context& ctx,
    const n4m_matrix_view_t& X,
    const n4m_matrix_view_t& Y,
    const std::int64_t* row_indices,
    std::int64_t n_indices,
    MomentStats& out);

[[nodiscard]] n4m_status_t subtract_moments(
    Context& ctx,
    const MomentStats& lhs,
    const MomentStats& rhs,
    MomentStats& out);

[[nodiscard]] n4m_status_t recompute_centered_moments(
    Context& ctx,
    MomentStats& stats);

}  // namespace n4m::core
