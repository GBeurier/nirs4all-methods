/* SPDX-License-Identifier: CECILL-2.1 */
/* cpp/include/n4m/ensemble.h — ensemble role header (ABI 2.0). */
#ifndef N4M_ENSEMBLE_H
#define N4M_ENSEMBLE_H
#include "n4m/n4m.h"   /* shared infra: status, matrix view, context, N4M_API */

#ifdef __cplusplus
extern "C" {
#endif

/* Native AOM Ridge simplex blender over strict-linear chain banks.
 *
 * This evaluates every (chain, lambda) Ridge candidate with fold-local OOF
 * predictions, solves a non-negative simplex blend over the OOF prediction
 * columns, then refits every candidate on all rows and returns the weighted
 * final in-sample predictions. Lambdas must be finite and strictly positive.
 *
 * `profile`: 0=compact, 1=wide. `regularizer` is the non-negative shrinkage
 * toward uniform simplex weights. `fold_ids` follows n4m_model_selection_sweep_run semantics.
 *
 * Result double matrices:
 *   "candidate_scores" (n_candidates x 5), columns:
 *      candidate_id, chain_id, lambda, cv_rmse, simplex_weight
 *   "weights" (1 x n_candidates)
 *   "oof_predictions" (n_samples x n_targets) weighted OOF blend
 *   "predictions" (n_samples x n_targets) weighted final-refit blend
 *   "input_coefficients" (n_features x n_targets) weighted final-refit
 *      coefficients folded into the original input feature space
 *   "intercept" (1 x n_targets) weighted final-refit intercept
 *   "oof_candidate_predictions" ((n_samples*n_targets) x n_candidates)
 *   "candidate_predictions" ((n_samples*n_targets) x n_candidates)
 * Int vectors:
 *   "fold_ids" (n_samples)
 * Scalars:
 *   "selected_candidate_id", "selected_chain_id", "selected_param",
 *   "selected_cv_rmse", "blend_oof_rmse", "regularizer", "n_candidates",
 *   "n_chains", "profile", "cv", "n_samples", "n_features", "n_targets"
 */
N4M_API n4m_status_t n4m_ensemble_aom_ridge_blender_fit(
    n4m_context_t* ctx,
    const n4m_config_t* cfg,
    const n4m_matrix_view_t* X,
    const n4m_matrix_view_t* Y,
    int32_t profile,
    int32_t cv,
    const int32_t* fold_ids,
    int64_t n_fold_ids,
    const double* ridge_lambdas,
    int64_t n_ridge_lambdas,
    double regularizer,
    n4m_method_result_t** out_result);

/* Native strict-linear AOM operator PLS score stack with Ridge head.
 *
 * The method builds the compact/wide strict-linear AOM operator bank, fits a
 * fold-local PLS1 score projector for each operator, concatenates the scores,
 * selects (n_components, alpha) by CV criterion, then refits the selected
 * stack on all rows. This ABI surface is intentionally single-target (`Y`
 * must have one column), matching the PLS1 reference estimator.
 *
 * `profile`: 0=compact, 1=wide. If `fold_ids` is null, contiguous balanced
 * folds are generated from `cv`; otherwise `fold_ids` must have n_samples
 * entries. `components` and `alphas` must be non-empty. `std_penalty` and
 * `gap_penalty` are non-negative additions to the selection criterion:
 * mean_oof_rmse + std_penalty * std_oof_rmse
 *               + gap_penalty * max(0, mean_oof_rmse - mean_train_rmse).
 *
 * Result double matrices:
 *   "candidate_scores" (n_specs x 7), columns:
 *      spec_id, n_components, alpha, mean_oof_rmse, std_oof_rmse,
 *      mean_train_rmse, criterion
 *   "fold_scores" (n_specs x cv)
 *   "oof_predictions" (n_samples x 1) selected-spec OOF predictions
 *   "predictions" (n_samples x 1) final refit predictions
 *   "stack_features" (n_samples x n_operator_features) final score stack
 *   "coefficients" (n_operator_features x 1) final Ridge head
 *   "intercept" (1 x 1) final Ridge head intercept on stack_features
 *   "input_coefficients" (n_features x 1) selected stack folded into the
 *      original input feature space
 *   "input_intercept" (1 x 1) folded input-space intercept
 * Int vectors:
 *   "fold_ids" (n_samples)
 *   "operator_feature_offsets" (n_operators + 1)
 * Scalars:
 *   "selected_spec_id", "selected_components", "selected_alpha",
 *   "selected_oof_rmse", "selected_train_rmse", "selected_criterion",
 *   "std_penalty", "gap_penalty", "n_operator_features", "n_specs",
 *   "n_operators", "profile", "cv", "n_samples", "n_features",
 *   "n_targets"
 */
N4M_API n4m_status_t n4m_ensemble_aom_operator_pls_stack_fit(
    n4m_context_t* ctx,
    const n4m_config_t* cfg,
    const n4m_matrix_view_t* X,
    const n4m_matrix_view_t* Y,
    int32_t profile,
    int32_t cv,
    const int32_t* fold_ids,
    int64_t n_fold_ids,
    const int32_t* components,
    int64_t n_components,
    const double* alphas,
    int64_t n_alphas,
    double std_penalty,
    double gap_penalty,
    n4m_method_result_t** out_result);

/* Bagging PLS (§20). Bootstrap aggregation of `n_estimators` PLS
 * regressors with the configured seed. Returns the average regression
 * coefficient matrix:
 *   "coefficients"   (n_features x n_targets)
 *   "predictions"    (n_samples x n_targets)
 *   "x_mean", "y_mean"
 *   scalar "rmse", scalar "n_estimators"
 */
N4M_API n4m_status_t n4m_ensemble_bagging_pls_fit(
    n4m_context_t* ctx,
    const n4m_config_t* cfg,
    const n4m_matrix_view_t* X,
    const n4m_matrix_view_t* Y,
    int32_t n_estimators,
    uint64_t seed,
    n4m_method_result_t** out_result);

/* Boosting PLS (§20). Gradient-boosting style stage-wise refit of
 * `n_estimators` PLS regressors with a per-stage `learning_rate`.
 * Output shape identical to bagging_pls_fit.
 */
N4M_API n4m_status_t n4m_ensemble_boosting_pls_fit(
    n4m_context_t* ctx,
    const n4m_config_t* cfg,
    const n4m_matrix_view_t* X,
    const n4m_matrix_view_t* Y,
    int32_t n_estimators,
    double learning_rate,
    n4m_method_result_t** out_result);

/* Random-subspace PLS (§20). Each of `n_estimators` PLS regressors is
 * fit on a random subset of `features_per_subspace` columns. The
 * result averages predictions over the missing columns by zero-padding
 * coefficients; output shape identical to bagging_pls_fit plus a
 * scalar `features_per_subspace`.
 */
N4M_API n4m_status_t n4m_ensemble_random_subspace_pls_fit(
    n4m_context_t* ctx,
    const n4m_config_t* cfg,
    const n4m_matrix_view_t* X,
    const n4m_matrix_view_t* Y,
    int32_t n_estimators,
    int32_t features_per_subspace,
    uint64_t seed,
    n4m_method_result_t** out_result);

#ifdef __cplusplus
}  /* extern "C" */
#endif
#endif /* N4M_ENSEMBLE_H */
