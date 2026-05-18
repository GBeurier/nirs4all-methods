// SPDX-License-Identifier: CECILL-2.1
//
// Internal AOM preprocessing-bank transform primitive.

#pragma once

#include <cstdint>
#include <vector>

#include "pls4all/p4a.h"

#include "core/context.hpp"
#include "core/gating_strategy.hpp"
#include "core/operator_bank.hpp"

namespace pls4all::core {

struct AomPreprocessingResult {
    std::int64_t n_samples{0};
    std::int64_t n_features{0};
    std::int32_t n_operators{0};
    p4a_gating_mode_t mode{P4A_GATING_SOFT};

    std::vector<double> weights;
    std::vector<std::int64_t> operator_kinds;
    std::vector<double> operator_outputs;  // operator-major, each block row-major n_samples x n_features
    std::vector<double> transformed;       // row-major n_samples x n_features
};

[[nodiscard]] p4a_status_t apply_aom_preprocessing(Context& ctx,
                                                   const OperatorBank& bank,
                                                   const GatingStrategy& gate,
                                                   const p4a_matrix_view_t& X,
                                                   const p4a_matrix_view_t* Y,
                                                   AomPreprocessingResult& out);

}  // namespace pls4all::core
