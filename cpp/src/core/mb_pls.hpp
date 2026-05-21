// SPDX-License-Identifier: CECILL-2.1
//
// Internal block-weighted MB-PLS kernel.

#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>

#include "pls4all/p4a.h"

#include "core/config.hpp"
#include "core/common/context.hpp"

namespace n4m::core {

struct MbPlsResult {
    std::int64_t n_samples{0};
    std::int32_t n_features{0};
    std::int32_t n_targets{0};
    std::int32_t n_components{0};
    std::int32_t n_blocks{0};

    std::vector<double> predictions;   // row-major n_samples x n_targets
    std::vector<double> coefficients;  // row-major n_features x n_targets, original X scale
    std::vector<double> intercept;     // n_targets
    std::vector<double> x_mean;        // n_features, original X scale
    std::vector<double> x_scale;       // n_features, original X scale
    std::vector<double> block_weights; // n_blocks
};

[[nodiscard]] n4m_status_t fit_predict_mb_pls(
    Context& ctx,
    const Config& cfg,
    const n4m_matrix_view_t& X,
    const n4m_matrix_view_t& Y,
    const std::int64_t* block_sizes,
    std::size_t n_blocks,
    MbPlsResult& out);

}  // namespace n4m::core
