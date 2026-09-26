// SPDX-License-Identifier: CECILL-2.1
//
// Multi-block PLS extensions: O2PLS, SO-PLS, OnPLS, ROSA. Internal
// kernels — public ABI wrappers ship with the binding tranche.

#pragma once

#include <cstdint>
#include <vector>

#include "n4m/n4m.h"

#include "core/config.hpp"
#include "core/common/context.hpp"

namespace n4m::core {

// O2PLS: bi-directional OPLS with predictive + X-orthogonal + Y-orthogonal
// components.
struct O2PlsResult {
    std::int64_t n_samples{0};
    std::int32_t n_features_x{0};
    std::int32_t n_features_y{0};
    std::int32_t n_predictive{0};
    std::int32_t n_x_orthogonal{0};
    std::int32_t n_y_orthogonal{0};

    std::vector<double> x_mean;
    std::vector<double> y_mean;
    std::vector<double> w_predictive;      // p × n_predictive
    std::vector<double> c_predictive;      // q × n_predictive
    std::vector<double> w_x_orthogonal;    // p × n_x_orthogonal
    std::vector<double> c_y_orthogonal;    // q × n_y_orthogonal
    std::vector<double> b_predictive;      // n_predictive scalar coefficients
    std::vector<double> coefficients;      // p × q
};

[[nodiscard]] n4m_status_t fit_o2pls(
    Context& ctx,
    const Config& cfg,
    const n4m_matrix_view_t& X,
    const n4m_matrix_view_t& Y,
    std::int32_t n_predictive,
    std::int32_t n_x_orthogonal,
    std::int32_t n_y_orthogonal,
    O2PlsResult& out);

// SO-PLS: sequential and orthogonalized PLS for B X-blocks predicting one
// Y. Each block is orthogonalized against the scores of the previous blocks,
// then regressed on the Y residual. The whole chain is linear in the input
// row: new rows are predicted as y_mean + (x - x_mean) coefficients, where x
// concatenates the blocks in input order.
struct SoPlsResult {
    std::int32_t n_blocks{0};
    std::vector<std::int32_t> n_components_per_block;
    std::vector<double> y_mean;
    std::vector<double> x_mean;        // sum(p_b)
    std::vector<double> coefficients;  // sum(p_b) × n_targets, row-major
    std::vector<double> predictions;   // n_samples × n_targets, row-major
};

[[nodiscard]] n4m_status_t fit_so_pls(
    Context& ctx,
    const Config& cfg,
    const std::vector<n4m_matrix_view_t>& X_blocks,
    const n4m_matrix_view_t& Y,
    const std::vector<std::int32_t>& n_components_per_block,
    SoPlsResult& out);

// OnPLS: generalized orthogonal projection for multi-block PLS
// (Löfstedt & Trygg, 2011). For each block, separates joint (globally
// shared) and unique (block-orthogonal) variation. `joint_*` arrays hold
// the predictive nPLS components computed after the unique components
// have been filtered out. `unique_*` arrays hold the per-block orthogonal
// projections. `block_reconstruction_per_block[b]` (n × p_b) is the
// in-sample reconstruction of block b from the joint components of the
// other connected blocks — the natural OnPLS prediction (sign- and
// rotation-invariant under the algorithm's gauge freedom).
struct OnPlsResult {
    std::int32_t n_blocks{0};
    std::int32_t n_joint{0};
    std::vector<std::int32_t> n_unique_per_block;
    std::vector<std::vector<double>> joint_loadings_per_block;        // p_b × n_joint
    std::vector<std::vector<double>> unique_loadings_per_block;       // p_b × n_unique_b
    std::vector<std::vector<double>> joint_scores_per_block;          // n × n_joint
    std::vector<std::vector<double>> block_reconstruction_per_block;  // n × p_b
    // What on_pls_joint_scores needs besides the loadings: the block means,
    // the weights of the unique components actually extracted (one row of
    // p_b values each, at most n_unique_b rows) and the joint weights.
    std::vector<std::vector<double>> x_mean_per_block;                // p_b
    std::vector<std::vector<double>> unique_weights_per_block;        // n_extracted_b × p_b
    std::vector<std::vector<double>> joint_weights_per_block;         // p_b × n_joint
};

[[nodiscard]] n4m_status_t fit_on_pls(
    Context& ctx,
    const Config& cfg,
    const std::vector<n4m_matrix_view_t>& X_blocks,
    std::int32_t n_joint,
    const std::vector<std::int32_t>& n_unique_per_block,
    OnPlsResult& out);

// Joint scores of (new) rows, `OnPLS.predict`'s score step: each block is
// centered with the training means, its unique components are filtered out,
// then it is projected on the joint weights with sequential deflation.
// `scores[b]` is n × n_joint, row-major.
[[nodiscard]] n4m_status_t on_pls_joint_scores(
    Context& ctx,
    const OnPlsResult& model,
    const std::vector<n4m_matrix_view_t>& X_blocks,
    std::vector<std::vector<double>>& scores);

// ROSA: Response-Oriented Sequential Alternation. At each component, pick
// the block whose latent direction yields the highest correlation with the
// current Y residual. Like SO-PLS, new rows are predicted as
// y_mean + (x - x_mean) coefficients over the concatenated blocks.
struct RosaResult {
    std::int32_t n_components{0};
    std::vector<std::int32_t> selected_block_per_component;
    std::vector<double> y_mean;
    std::vector<double> x_mean;        // sum(p_b)
    std::vector<double> coefficients;  // sum(p_b) × n_targets, row-major
    std::vector<double> predictions;
};

[[nodiscard]] n4m_status_t fit_rosa(
    Context& ctx,
    const Config& cfg,
    const std::vector<n4m_matrix_view_t>& X_blocks,
    const n4m_matrix_view_t& Y,
    std::int32_t n_components,
    RosaResult& out);

}  // namespace n4m::core
