// SPDX-License-Identifier: CECILL-2.1
//
// Recursive (moving-window) PLS. Ships the simplest viable variant: fit
// SIMPLS on the last `window_size` samples and predict the next one. The
// fully-incremental NIPALS update is deferred to a future phase.

#pragma once

#include <cstdint>
#include <vector>

#include "pls4all/p4a.h"

#include "core/config.hpp"
#include "core/context.hpp"

namespace n4m::core {

struct RecursivePlsResult {
    std::int64_t n_samples{0};
    std::int32_t n_features{0};
    std::int32_t n_targets{0};
    std::int32_t n_components{0};
    std::int32_t window_size{0};

    // Predictions for the FIRST `window_size` samples are 0 (warmup); for
    // every subsequent sample i the prediction is from the SIMPLS model
    // fitted on samples [i - window_size, i).
    std::vector<double> predictions;        // n_samples x n_targets, row-major
    std::vector<std::int32_t> in_window;    // 1 if a prediction was produced
};

[[nodiscard]] n4m_status_t fit_predict_recursive_pls(
    Context& ctx,
    const Config& cfg,
    const n4m_matrix_view_t& X,
    const n4m_matrix_view_t& Y,
    std::int32_t window_size,
    RecursivePlsResult& out);

}  // namespace n4m::core
