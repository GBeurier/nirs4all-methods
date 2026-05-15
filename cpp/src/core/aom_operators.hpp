// SPDX-License-Identifier: CeCILL-2.1
//
// Strict-linear AOM operator kernels matching nirs4all/bench/AOM_v0/aompls.

#pragma once

#include <vector>

#include "pls4all/p4a.h"

#include "core/context.hpp"
#include "core/operator_entry.hpp"

namespace pls4all::core {

[[nodiscard]] p4a_status_t transform_aom_strict_operator(
    Context& ctx,
    const OperatorEntry& entry,
    const p4a_matrix_view_t& X,
    std::vector<double>& out);

}  // namespace pls4all::core
