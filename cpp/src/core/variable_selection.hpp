// SPDX-License-Identifier: CECILL-2.1
//
// Internal variable-selection rankers over fitted latent-variable models.

#pragma once

#include <cstdint>
#include <vector>

#include "pls4all/p4a.h"

#include "core/context.hpp"
#include "core/model.hpp"

namespace pls4all::core {

enum class VariableSelectionMethod : std::int32_t {
    Vip = 0,
    CoefficientMagnitude = 1,
    SelectivityRatio = 2,
};

struct VariableSelectionResult {
    std::int32_t n_features{0};
    std::int32_t top_k{0};
    std::vector<double> scores;
    std::vector<std::int64_t> selected_indices;
};

[[nodiscard]] p4a_status_t select_variables(
    Context& ctx,
    const Model& model,
    const p4a_matrix_view_t& X,
    VariableSelectionMethod method,
    std::int32_t top_k,
    VariableSelectionResult& out);

}  // namespace pls4all::core
