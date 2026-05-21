// SPDX-License-Identifier: CECILL-2.1
//
// Internal deterministic IRIV (Iteratively Retains Informative Variables)
// variable-selection primitive. Reference:
//   Yun Y.-H., Wang W.-T., Tan M.-L., Liang Y.-Z., Li H.-D., Cao D.-S.,
//   Lu H.-M., Xu Q.-S. (2014). "A strategy that iteratively retains
//   informative variables for selecting optimal variable subset in
//   multivariate calibration." Analytica Chimica Acta 807, 36–43.
//   Implemented in libPLS 1.95 `iriv.m`.

#pragma once

#include <cstdint>
#include <vector>

#include "pls4all/p4a.h"

#include "core/config.hpp"
#include "core/context.hpp"
#include "core/validation.hpp"

namespace n4m::core {

struct IrivSelectionResult {
    std::int32_t n_features{0};
    std::int32_t n_targets{0};
    std::int32_t n_components{0};
    std::int32_t n_rounds{0};
    std::int32_t binary_matrix_rows{0};
    std::uint64_t seed{0};

    // remaining_per_round[k] = number of variables surviving round k.
    // remaining_per_round[0] = n_features (input).
    std::vector<std::int64_t> remaining_per_round;
    std::vector<std::int64_t> removed_per_round;
    std::vector<std::int64_t> selected_indices;
};

[[nodiscard]] n4m_status_t select_by_iriv(
    Context& ctx,
    const Config& cfg,
    const n4m_matrix_view_t& X,
    const n4m_matrix_view_t& Y,
    const ValidationPlan& plan,
    std::int32_t max_rounds,
    std::uint64_t seed,
    IrivSelectionResult& out);

}  // namespace n4m::core
