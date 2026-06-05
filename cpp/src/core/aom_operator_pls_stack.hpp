// SPDX-License-Identifier: CECILL-2.1
//
// Native strict-linear AOM operator PLS score stack with Ridge head.

#pragma once

#include <cstdint>
#include <vector>

#include "n4m/n4m.h"

#include "core/common/context.hpp"
#include "core/config.hpp"

namespace n4m::core {

struct AomOperatorPlsStackResult {
    std::int64_t n_samples{0};
    std::int64_t n_features{0};
    std::int64_t n_targets{0};
    std::int32_t profile{0};
    std::int32_t cv{0};
    std::int64_t n_operators{0};
    std::int64_t n_specs{0};
    std::int64_t n_operator_features{0};

    std::int64_t selected_spec_id{-1};
    std::int32_t selected_components{0};
    double selected_alpha{0.0};
    double selected_oof_rmse{0.0};
    double selected_train_rmse{0.0};
    double selected_criterion{0.0};
    double std_penalty{0.0};
    double gap_penalty{0.0};

    // n_specs x 7:
    // spec_id, n_components, alpha, mean_oof_rmse, std_oof_rmse,
    // mean_train_rmse, criterion
    std::vector<double> candidate_scores;
    std::vector<double> fold_scores;      // n_specs x cv
    std::vector<double> oof_predictions;  // n x 1 for selected spec
    std::vector<double> predictions;      // n x 1 final refit
    std::vector<double> stack_features;   // n x n_operator_features final scores
    std::vector<double> coefficients;     // n_operator_features x 1 Ridge head
    std::vector<double> intercept;        // 1 x 1
    std::vector<double> input_coefficients; // n_features x 1 folded input-space
                                            // final linear coefficients
    std::vector<double> input_intercept;     // 1 x 1 folded input-space intercept
    std::vector<std::int32_t> fold_ids;   // n
    std::vector<std::int32_t> operator_feature_offsets; // n_operators + 1
};

[[nodiscard]] n4m_status_t fit_aom_operator_pls_stack(
    Context& ctx,
    const Config& cfg,
    const n4m_matrix_view_t& X,
    const n4m_matrix_view_t& Y,
    std::int32_t profile,
    std::int32_t cv,
    const std::int32_t* fold_ids,
    std::int64_t n_fold_ids,
    const std::int32_t* components,
    std::int64_t n_components,
    const double* alphas,
    std::int64_t n_alphas,
    double std_penalty,
    double gap_penalty,
    AomOperatorPlsStackResult& out);

}  // namespace n4m::core
