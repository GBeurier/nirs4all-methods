// SPDX-License-Identifier: CECILL-2.1
//
// 3-way tensor PLS (Bro 1996, N-PLS). X is a flat row-major buffer of
// shape (n × J × K), Y is (n × q). The kernel finds a sequence of unit
// rank-1 components (w_J ⊗ w_K) that maximize covariance between t and Y.

#pragma once

#include <cstdint>
#include <vector>

#include "pls4all/p4a.h"

#include "core/config.hpp"
#include "core/context.hpp"

namespace n4m::core {

struct NPlsResult {
    std::int64_t n_samples{0};
    std::int32_t mode_j{0};   // number of "second-mode" features
    std::int32_t mode_k{0};   // number of "third-mode" features
    std::int32_t n_components{0};
    std::int32_t n_targets{0};

    std::vector<double> w_j_per_component;     // J × n_components
    std::vector<double> w_k_per_component;     // K × n_components
    std::vector<double> scores_t;              // n × n_components
    std::vector<double> y_loadings_q;          // q × n_components
    std::vector<double> regression_b;          // n_components
    std::vector<double> coefficients;          // (J*K) × q row-major
    std::vector<double> x_mean;                // J*K
    std::vector<double> y_mean;                // q
};

// Fit N-PLS via Bro's algorithm. `x_data` is the row-major (n × J × K)
// tensor flattened as n × (J*K). `n_components` is the number of latent
// components.
[[nodiscard]] n4m_status_t fit_n_pls(
    Context& ctx,
    const Config& cfg,
    const n4m_matrix_view_t& X_flat,
    std::int32_t mode_j,
    std::int32_t mode_k,
    const n4m_matrix_view_t& Y,
    NPlsResult& out);

// Apply a fitted N-PLS model to new tensor data.
[[nodiscard]] n4m_status_t predict_n_pls(
    Context& ctx,
    const NPlsResult& model,
    const n4m_matrix_view_t& X_flat,
    std::vector<double>& out_predictions);

}  // namespace n4m::core
