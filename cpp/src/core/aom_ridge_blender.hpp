// SPDX-License-Identifier: CECILL-2.1
//
// Native fold-safe AOM Ridge simplex blender over strict-linear chain banks.

#pragma once

#include <cstdint>
#include <vector>

#include "n4m/n4m.h"

#include "core/common/context.hpp"
#include "core/config.hpp"

namespace n4m::core {

struct AomRidgeBlenderResult {
    std::int64_t n_samples{0};
    std::int64_t n_features{0};
    std::int64_t n_targets{0};
    std::int32_t profile{0};
    std::int32_t cv{0};
    std::int64_t n_chains{0};
    std::int64_t n_candidates{0};
    double regularizer{0.0};

    std::int64_t selected_candidate_id{-1};
    std::int64_t selected_chain_id{-1};
    double selected_param{0.0};
    double selected_cv_rmse{0.0};
    double blend_oof_rmse{0.0};

    std::vector<double> candidate_scores;          // n_candidates x 5:
                                                   // candidate_id, chain_id,
                                                   // lambda, cv_rmse, weight
    std::vector<double> weights;                   // 1 x n_candidates
    std::vector<double> oof_predictions;           // n x q weighted blend
    std::vector<double> predictions;               // n x q weighted final blend
    std::vector<double> input_coefficients;         // p x q weighted input-space
                                                   // linear coefficients
    std::vector<double> intercept;                 // 1 x q weighted intercept
    std::vector<double> oof_candidate_predictions; // (n*q) x n_candidates
    std::vector<double> candidate_predictions;     // (n*q) x n_candidates
    std::vector<std::int32_t> fold_ids;            // n
};

[[nodiscard]] n4m_status_t fit_aom_ridge_blender(
    Context& ctx,
    const Config& cfg,
    const n4m_matrix_view_t& X,
    const n4m_matrix_view_t& Y,
    std::int32_t profile,
    std::int32_t cv,
    const std::int32_t* fold_ids,
    std::int64_t n_fold_ids,
    const double* ridge_lambdas,
    std::int64_t n_ridge_lambdas,
    double regularizer,
    AomRidgeBlenderResult& out);

}  // namespace n4m::core
