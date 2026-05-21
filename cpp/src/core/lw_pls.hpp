// SPDX-License-Identifier: CECILL-2.1
//
// Internal local-window PLS kernel.

#pragma once

#include <cstdint>
#include <vector>

#include "pls4all/p4a.h"

#include "core/config.hpp"
#include "core/context.hpp"

namespace n4m::core {

struct LwPlsResult {
    std::int64_t n_samples{0};
    std::int32_t n_features{0};
    std::int32_t n_targets{0};
    std::int32_t n_components{0};
    std::int32_t n_neighbors{0};

    std::vector<double> predictions;          // row-major n_samples x n_targets
    std::vector<std::int64_t> neighbor_indices; // row-major n_samples x n_neighbors
};

[[nodiscard]] n4m_status_t fit_predict_lw_pls(
    Context& ctx,
    const Config& cfg,
    const n4m_matrix_view_t& X,
    const n4m_matrix_view_t& Y,
    std::int32_t n_neighbors,
    LwPlsResult& out);

}  // namespace n4m::core
