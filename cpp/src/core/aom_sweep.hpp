// SPDX-License-Identifier: CECILL-2.1
//
// Native configurable AOM preprocessing sweep. This applies the strict-linear
// AOM chain banks, then delegates head screening to run_moment_sweep.

#pragma once

#include <cstdint>
#include <vector>

#include "n4m/n4m.h"

#include "core/common/context.hpp"
#include "core/config.hpp"

namespace n4m::core {

struct AomSweepResult {
    std::int64_t n_samples{0};
    std::int64_t n_features{0};
    std::int64_t n_targets{0};
    std::int32_t profile{0};
    std::int32_t cv{0};
    std::int64_t n_chains{0};
    std::int64_t n_candidates{0};
    std::int64_t n_operator_moment_candidates{0};
    std::int64_t n_ridge_operator_moment_candidates{0};
    std::int64_t n_pls_operator_moment_candidates{0};
    std::int64_t n_banded_operator_moment_candidates{0};
    std::int64_t n_structured_operator_moment_candidates{0};
    std::int64_t n_dense_operator_moment_candidates{0};
    std::int64_t n_materialized_candidates{0};
    std::int64_t n_ridge_materialized_candidates{0};
    std::int64_t n_pls_materialized_candidates{0};
    std::int64_t n_moment_prefix_cache_hits{0};
    std::int64_t n_moment_prefix_cache_misses{0};
    std::int64_t n_ridge_moment_cv_fits{0};
    std::int64_t n_ridge_dual_materialized_cv_fits{0};
    std::int64_t n_ridge_dual_cross_cv_fits{0};
    std::int64_t n_ridge_moment_score_batch_calls{0};
    std::int64_t n_ridge_moment_score_batch_jobs{0};
    std::int64_t n_ridge_moment_final_fits{0};
    std::int64_t n_ridge_dual_materialized_final_fits{0};
    std::int64_t n_pls_moment_cv_fits{0};
    std::int64_t n_pls_moment_host_cv_fits{0};
    std::int64_t n_pls_moment_cuda_device_cv_fits{0};
    std::int64_t n_pls_moment_cuda_parallel_fold_batches{0};
    std::int64_t n_pls_moment_cuda_parallel_fold_jobs{0};
    std::int64_t n_pls_moment_score_batch_calls{0};
    std::int64_t n_pls_moment_score_batch_jobs{0};
    std::int64_t n_pls_materialized_cv_fits{0};
    std::int64_t n_pls_gcv_proxy_candidates{0};
    std::int64_t n_pls_gcv_proxy_fits{0};
    std::int64_t n_pls_gcv_proxy_batch_calls{0};
    std::int64_t n_pls_gcv_proxy_batch_jobs{0};
    std::int64_t n_pls_moment_final_fits{0};
    std::int64_t n_pls_moment_host_final_fits{0};
    std::int64_t n_pls_moment_cuda_device_final_fits{0};
    std::int64_t n_pls_materialized_final_fits{0};
    std::int32_t aom_pls_score_mode{N4M_AOM_PLS_SCORE_CV};
    std::int32_t score_only{0};

    std::int64_t selected_candidate_id{-1};
    std::int64_t selected_chain_id{-1};
    std::int64_t selected_sweep_candidate_id{-1};
    std::int32_t selected_head_id{-1};
    double selected_param{0.0};
    double selected_cv_rmse{0.0};

    std::vector<double> candidate_scores;  // n_candidates x 5:
                                           // candidate_id, chain_id,
                                           // head_id, param, cv_rmse
    std::vector<std::int32_t> candidate_routes; // n_candidates:
                                                // 0 materialized,
                                                // 1 dense moment,
                                                // 2 banded moment,
                                                // 3 structured moment
    std::vector<double> predictions;       // selected final refit, n x q
    std::vector<double> oof_predictions;   // selected OOF, n x q
    std::vector<double> coefficients;      // transformed p x q
    std::vector<double> input_coefficients; // original-input p x q
    std::vector<double> intercept;         // 1 x q
    std::vector<double> x_mean;            // 1 x p
    std::vector<double> x_scale;           // 1 x p
    std::vector<double> y_mean;            // 1 x q
    std::vector<std::int32_t> fold_ids;    // n
    std::vector<std::int32_t> chain_offsets;  // n_chains + 1
    std::vector<std::int32_t> op_kinds;       // n_ops
    std::vector<std::int32_t> param_offsets;  // n_ops + 1
    std::vector<double> chain_params;         // 1 x n_params
};

[[nodiscard]] n4m_status_t run_aom_sweep(
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
    const std::int32_t* pls_components,
    std::int64_t n_pls_components,
    std::int32_t heads_mask,
    AomSweepResult& out);

[[nodiscard]] n4m_status_t run_aom_chain_sweep(
    Context& ctx,
    const Config& cfg,
    const n4m_matrix_view_t& X,
    const n4m_matrix_view_t& Y,
    std::int32_t cv,
    const std::int32_t* fold_ids,
    std::int64_t n_fold_ids,
    const std::int32_t* chain_offsets,
    std::int64_t n_chain_offsets,
    const std::int32_t* op_kinds,
    std::int64_t n_op_kinds,
    const std::int32_t* param_offsets,
    std::int64_t n_param_offsets,
    const double* params,
    std::int64_t n_params,
    const double* ridge_lambdas,
    std::int64_t n_ridge_lambdas,
    const std::int32_t* pls_components,
    std::int64_t n_pls_components,
    std::int32_t heads_mask,
    AomSweepResult& out);

[[nodiscard]] n4m_status_t run_aom_chain_fixed_fit(
    Context& ctx,
    const Config& cfg,
    const n4m_matrix_view_t& X,
    const n4m_matrix_view_t& Y,
    const std::int32_t* chain_offsets,
    std::int64_t n_chain_offsets,
    const std::int32_t* op_kinds,
    std::int64_t n_op_kinds,
    const std::int32_t* param_offsets,
    std::int64_t n_param_offsets,
    const double* params,
    std::int64_t n_params,
    std::int32_t head_id,
    double param,
    AomSweepResult& out);

}  // namespace n4m::core
