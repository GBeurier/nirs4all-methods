// SPDX-License-Identifier: CECILL-2.1
//
// Internal variable-importance kernels for fitted latent-variable models.

#pragma once

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

}  // namespace n4m::core
