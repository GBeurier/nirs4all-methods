// SPDX-License-Identifier: CECILL-2.1
//
// Internal AOM preprocessing-bank transform primitive.

#pragma once

#include <cstdint>
#include <vector>

#include "n4m/n4m.h"

#include "core/context.hpp"
#include "core/gating_strategy.hpp"
#include "core/operator_bank.hpp"

namespace n4m::core {

struct AomPreprocessingResult {
    std::int64_t n_samples{0};
    std::int64_t n_features{0};
    std::int32_t n_operators{0};
    n4m_gating_mode_t mode{N4M_GATING_SOFT};

    std::vector<double> weights;
    std::vector<std::int64_t> operator_kinds;
    std::vector<double> operator_outputs;  // operator-major, each block row-major n_samples x n_features
    std::vector<double> transformed;       // row-major n_samples x n_features
};

[[nodiscard]] n4m_status_t apply_aom_preprocessing(Context& ctx,
                                                   const OperatorBank& bank,
                                                   const GatingStrategy& gate,
                                                   const n4m_matrix_view_t& X,
                                                   const n4m_matrix_view_t* Y,
                                                   AomPreprocessingResult& out);

}  // namespace n4m::core
