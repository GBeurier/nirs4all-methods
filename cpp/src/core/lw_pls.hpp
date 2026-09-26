// SPDX-License-Identifier: CECILL-2.1
//
// Internal local-window PLS kernel.

#pragma once

#include <cstdint>
#include <vector>

#include "n4m/n4m.h"

#include "core/config.hpp"
#include "core/common/context.hpp"

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

// Predictions for new rows X (n × n_targets, row-major) from the training
// set (x_train n_train × p, y_train n_train × n_targets, row-major): each row
// gets its own local model, exactly as fit_predict_lw_pls predicts its
// training rows. cfg supplies n_components and the mode (solver SIMPLS =
// k-NN cutoff, otherwise Gaussian-weighted).
[[nodiscard]] n4m_status_t predict_lw_pls(
    Context& ctx,
    const Config& cfg,
    const std::vector<double>& x_train,
    const std::vector<double>& y_train,
    std::int64_t n_train,
    std::int32_t n_neighbors,
    const n4m_matrix_view_t& X,
    std::vector<double>& predictions);

}  // namespace n4m::core
