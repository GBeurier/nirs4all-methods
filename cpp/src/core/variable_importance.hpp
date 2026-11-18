// SPDX-License-Identifier: CECILL-2.1 OR AGPL-3.0-or-later
//
// Internal variable-importance kernels for fitted latent-variable models.

#pragma once

#include <cstddef>
#include <vector>

#include "n4m/n4m.h"

#include "core/common/context.hpp"
#include "core/model.hpp"

namespace n4m::core {

[[nodiscard]] n4m_status_t compute_vip_scores(
    Context& ctx,
    const Model& model,
    std::vector<double>& out);

[[nodiscard]] n4m_status_t compute_selectivity_ratio(
    Context& ctx,
    const Model& model,
    const n4m_matrix_view_t& X,
    std::vector<double>& out);

// VIP of a latent model from the scores T (n x n_components, row-major) of
// any n rows (the stored training scores, or ((X - x_mean) / x_scale) R of
// rows a host supplies). Same formula and conventions as
// compute_vip_scores.
[[nodiscard]] n4m_status_t compute_vip_from_scores(
    Context& ctx,
    const Model& model,
    const std::vector<double>& scores,
    std::size_t n,
    std::vector<double>& out);

// Selectivity ratio SR_j = ||T P_j'||^2 / ||z_j - T P_j'||^2 of model-space
// rows Z (n x p, row-major) against a rank-a reconstruction T P'
// (T: n x a, P: p x a, row-major); the zero-residual conventions of
// compute_selectivity_ratio. Z must be finite.
[[nodiscard]] n4m_status_t selectivity_ratio_from_reconstruction(
    Context& ctx,
    const std::vector<double>& z,
    std::size_t n,
    std::size_t p,
    const std::vector<double>& scores,
    const std::vector<double>& loadings,
    std::size_t a,
    std::vector<double>& out);

}  // namespace n4m::core
