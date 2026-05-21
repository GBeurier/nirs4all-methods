// SPDX-License-Identifier: CECILL-2.1
//
// Internal per-component regression coefficient kernels.

#pragma once

#include <vector>

#include "pls4all/p4a.h"

#include "core/common/context.hpp"
#include "core/model.hpp"

namespace n4m::core {

// Output layout is row-major [n_components, n_features, n_targets], flattened
// as component-major blocks of n_features x n_targets coefficients. Component
// block k contains the coefficients for the first k+1 latent components.
[[nodiscard]] n4m_status_t compute_regression_coefficients_by_component(
    Context& ctx,
    const Model& model,
    std::vector<double>& out);

}  // namespace n4m::core
