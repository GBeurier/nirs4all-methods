// SPDX-License-Identifier: CeCILL-2.1
//
// Internal variable-importance kernels for fitted latent-variable models.

#pragma once

#include <vector>

#include "pls4all/p4a.h"

#include "core/context.hpp"
#include "core/model.hpp"

namespace pls4all::core {

[[nodiscard]] p4a_status_t compute_vip_scores(
    Context& ctx,
    const Model& model,
    std::vector<double>& out);

[[nodiscard]] p4a_status_t compute_selectivity_ratio(
    Context& ctx,
    const Model& model,
    const p4a_matrix_view_t& X,
    std::vector<double>& out);

}  // namespace pls4all::core
