// SPDX-License-Identifier: CECILL-2.1
//
// Internal PLS-LDA classifier kernel.

#pragma once

#include <cstdint>
#include <vector>

#include "pls4all/p4a.h"

#include "core/config.hpp"
#include "core/context.hpp"

namespace pls4all::core {

struct PlsLdaResult {
    std::int64_t n_samples{0};
    std::int32_t n_classes{0};
    std::int32_t n_components{0};

    std::vector<std::int32_t> predictions;
    std::vector<double> decision_scores; // row-major n_samples x n_classes
};

[[nodiscard]] p4a_status_t fit_predict_pls_lda(
    Context& ctx,
    const Config& cfg,
    const p4a_matrix_view_t& X,
    const p4a_matrix_view_t& labels,
    std::int32_t n_classes,
    PlsLdaResult& out);

}  // namespace pls4all::core
