// SPDX-License-Identifier: CECILL-2.1
//
// Native moment-sweep entry points. v1 ships exact Ridge CV over row-additive
// moments/dual Ridge designs plus exact single-target PLS1 grids from moment
// cross-products when the requested solver/deflation is compatible. Other PLS
// regimes still use the materialized fold-local fallback. A fused batched
// IKPLS path can replace the remaining fallback and reduce many-job launch
// overhead without changing this ABI.

#pragma once

#include <cstdint>
#include <vector>

#include "n4m/n4m.h"

#include "core/common/context.hpp"
#include "core/config.hpp"
#include "core/moments.hpp"

namespace n4m::core {

enum SweepHeadBits : std::int32_t {
    kSweepHeadRidge = 1,
    kSweepHeadPls = 2,
};

struct SweepResult {
    std::int64_t n_samples{0};
    std::int64_t n_features{0};
    std::int64_t n_targets{0};
    std::int32_t cv{0};
    std::int64_t n_candidates{0};

    std::int64_t selected_candidate_id{-1};
    std::int32_t selected_head_id{-1};  // 0 = Ridge, 1 = PLS
    double selected_param{0.0};
    double selected_cv_rmse{0.0};
    std::int32_t score_only{0};
    std::int64_t n_ridge_moment_candidates{0};
    std::int64_t n_ridge_dual_materialized_candidates{0};
    std::int64_t n_ridge_moment_cv_fits{0};
    std::int64_t n_ridge_dual_materialized_cv_fits{0};
    std::int64_t n_ridge_dual_cross_cv_fits{0};
    std::int64_t n_ridge_moment_score_batch_calls{0};
    std::int64_t n_ridge_moment_score_batch_jobs{0};
    std::int64_t n_ridge_moment_final_fits{0};
    std::int64_t n_ridge_dual_materialized_final_fits{0};
    std::int64_t n_pls_moment_candidates{0};
    std::int64_t n_pls_gcv_proxy_candidates{0};
    std::int64_t n_pls_moment_cv_fits{0};
    std::int64_t n_pls_moment_host_cv_fits{0};
    std::int64_t n_pls_moment_cuda_device_cv_fits{0};
    std::int64_t n_pls_moment_cuda_parallel_fold_batches{0};
    std::int64_t n_pls_moment_cuda_parallel_fold_jobs{0};
    std::int64_t n_pls_moment_score_batch_calls{0};
    std::int64_t n_pls_moment_score_batch_jobs{0};
    std::int64_t n_pls_materialized_cv_fits{0};
    std::int64_t n_pls_gcv_proxy_fits{0};
    std::int64_t n_pls_gcv_proxy_batch_calls{0};
    std::int64_t n_pls_gcv_proxy_batch_jobs{0};
    std::int64_t n_pls_moment_final_fits{0};
    std::int64_t n_pls_moment_host_final_fits{0};
    std::int64_t n_pls_moment_cuda_device_final_fits{0};
    std::int64_t n_pls_materialized_final_fits{0};

    std::vector<double> candidate_scores;  // n_candidates x 4:
                                           // candidate_id, head_id, param, cv_rmse
    std::vector<double> predictions;       // final refit in-sample, n x q
    std::vector<double> oof_predictions;   // selected candidate OOF, n x q
    std::vector<double> coefficients;      // p x q
    std::vector<double> intercept;         // 1 x q
    std::vector<double> x_mean;            // 1 x p
    std::vector<double> x_scale;           // 1 x p
    std::vector<double> y_mean;            // 1 x q
    std::vector<std::int32_t> fold_ids;    // n
};

[[nodiscard]] n4m_status_t run_moment_sweep(
    Context& ctx,
    const Config& cfg,
    const n4m_matrix_view_t& X,
    const n4m_matrix_view_t& Y,
    std::int32_t cv,
    const std::int32_t* fold_ids,
    std::int64_t n_fold_ids,
    const double* ridge_lambdas,
    std::int64_t n_ridge_lambdas,
    const std::int32_t* pls_components,
    std::int64_t n_pls_components,
    std::int32_t heads_mask,
    SweepResult& out);

// Fit one already-selected Ridge/PLS candidate on all rows without running CV.
// This is intentionally not a ranking entry point: selected_cv_rmse and
// candidate_scores[..., cv_rmse] are NaN unless a higher-level wrapper injects
// a previously verified exact-CV score.
[[nodiscard]] n4m_status_t run_moment_final_fit(
    Context& ctx,
    const Config& cfg,
    const n4m_matrix_view_t& X,
    const n4m_matrix_view_t& Y,
    std::int32_t head_id,
    double param,
    SweepResult& out);

// Internal acceleration hook used by strict-linear AOM chain sweeps. The
// caller provides all-sample and held-out moments already expressed in the
// feature space being screened. This scores Ridge candidates from sufficient
// statistics only; prediction buffers are intentionally left empty.
[[nodiscard]] n4m_status_t score_ridge_moment_sweep(
    Context& ctx,
    const Config& cfg,
    const MomentStats& all_stats,
    const std::vector<MomentStats>& heldout_stats,
    const double* ridge_lambdas,
    std::int64_t n_ridge_lambdas,
    SweepResult& out);

// Score-only batch hook for strict-linear AOM Ridge screens. All entries share
// the same lambda grid and fold count; each output row corresponds to the
// same-index all/held-out moment set. This preserves exact held-out-moment CV
// while letting callers submit many transformed chains through one dispatch.
[[nodiscard]] n4m_status_t score_ridge_moment_sweeps_score_only(
    Context& ctx,
    const Config& cfg,
    const std::vector<MomentStats>& all_stats_by_chain,
    const std::vector<std::vector<MomentStats>>& heldout_stats_by_chain,
    const double* ridge_lambdas,
    std::int64_t n_ridge_lambdas,
    std::vector<SweepResult>& out);

// Internal acceleration hook for strict-linear AOM chain sweeps. This scores
// single-target PLS1 component grids from train/held-out sufficient statistics
// only. Prediction buffers are intentionally left empty; callers materialize
// the selected chain once if they need OOF/final predictions.
[[nodiscard]] n4m_status_t score_pls1_moment_sweep(
    Context& ctx,
    const Config& cfg,
    const MomentStats& all_stats,
    const std::vector<MomentStats>& heldout_stats,
    const std::int32_t* pls_components,
    std::int64_t n_pls_components,
    SweepResult& out);

// Score-only batch hook for strict-linear AOM PLS screens. All entries share
// the same component grid and fold count; each output row corresponds to the
// same-index all/held-out moment set. This keeps the exact CV scoring rule but
// lets callers submit many transformed chains through one internal dispatch.
[[nodiscard]] n4m_status_t score_pls1_moment_sweeps_score_only(
    Context& ctx,
    const Config& cfg,
    const std::vector<MomentStats>& all_stats_by_chain,
    const std::vector<std::vector<MomentStats>>& heldout_stats_by_chain,
    const std::int32_t* pls_components,
    std::int64_t n_pls_components,
    std::vector<SweepResult>& out);

// Fast strict-moment PLS1 first-pass screen. This scores component prefixes
// from all-sample moments with a nominal GCV RMSE proxy. It is intentionally
// not exact CV and is only used when callers explicitly request the AOM
// score-only proxy mode.
[[nodiscard]] n4m_status_t score_pls1_gcv_moment_screen(
    Context& ctx,
    const Config& cfg,
    const MomentStats& all_stats,
    const std::int32_t* pls_components,
    std::int64_t n_pls_components,
    SweepResult& out);

// Score-only batch hook for explicit AOM PLS GCV proxy screens. Each output
// row corresponds to the same-index all-sample transformed moment set. This is
// not exact CV; it is the many-chain form of score_pls1_gcv_moment_screen.
[[nodiscard]] n4m_status_t score_pls1_gcv_moment_screens_score_only(
    Context& ctx,
    const Config& cfg,
    const std::vector<MomentStats>& all_stats_by_chain,
    const std::int32_t* pls_components,
    std::int64_t n_pls_components,
    std::vector<SweepResult>& out);

}  // namespace n4m::core
