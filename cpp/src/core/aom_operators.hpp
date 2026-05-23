// SPDX-License-Identifier: CECILL-2.1
//
// Strict-linear AOM operator kernels matching nirs4all/bench/AOM_v0/aompls.

#pragma once

#include <vector>

#include "n4m/n4m.h"

#include "core/common/context.hpp"
#include "core/operator_entry.hpp"

namespace n4m::core {

[[nodiscard]] n4m_status_t transform_aom_strict_operator(
    Context& ctx,
    const OperatorEntry& entry,
    const n4m_matrix_view_t& X,
    std::vector<double>& out);

}  // namespace n4m::core
