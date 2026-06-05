// SPDX-License-Identifier: CECILL-2.1
//
// Strict-linear AOM operator kernels matching nirs4all/bench/AOM_v0/aompls.

#pragma once

#include <vector>

#include "n4m/n4m.h"

#include "core/common/context.hpp"
#include "core/operator_entry.hpp"

namespace n4m::core {

struct BandedLinearOperator {
    std::int64_t n_features{0};
    std::vector<std::int32_t> offsets;      // n_features + 1, per output column
    std::vector<std::int32_t> src_indices;  // source feature indices
    std::vector<double> coeffs;             // A[src, output_col]
};

[[nodiscard]] n4m_status_t transform_aom_strict_operator(
    Context& ctx,
    const OperatorEntry& entry,
    const n4m_matrix_view_t& X,
    std::vector<double>& out);

[[nodiscard]] n4m_status_t build_aom_banded_operator(
    Context& ctx,
    const OperatorEntry& entry,
    std::int64_t n_features,
    BandedLinearOperator& out);

}  // namespace n4m::core
