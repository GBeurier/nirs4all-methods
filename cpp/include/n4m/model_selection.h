/* SPDX-License-Identifier: CECILL-2.1 */
/* cpp/include/n4m/model_selection.h — model_selection role header (ABI 2.0). */
#ifndef N4M_MODEL_SELECTION_H
#define N4M_MODEL_SELECTION_H
#include "n4m/n4m.h"   /* shared infra: status, matrix view, context, N4M_API */

#ifdef __cplusplus
extern "C" {
#endif

/* ---------- Splitter enums (passed as int32_t through the create API) -- */
/* Bin-edge strategy for KBinsStratified + BinnedStratifiedGroupKFold. */
typedef enum n4m_split_kbins_strategy_t {
    N4M_SPLIT_KBINS_UNIFORM  = 0,  /* equal-width bins between y.min/y.max  */
    N4M_SPLIT_KBINS_QUANTILE = 1   /* equal-frequency bins (quantile edges) */
} n4m_split_kbins_strategy_t;

/* Y-metric mode for SPXYFold / SPXYGFold. Selects how Y participates in
 * the alternating max-min distance computation:
 *   NONE       : ignore Y entirely (pure Kennard-Stone on X)
 *   EUCLIDEAN  : standard SPXY (X and Y both Euclidean, summed)
 *   HAMMING    : SPXY with Hamming on Y (classification / multilabel) */
typedef enum n4m_split_y_metric_t {
    N4M_SPLIT_Y_METRIC_NONE      = 0,
    N4M_SPLIT_Y_METRIC_EUCLIDEAN = 1,
    N4M_SPLIT_Y_METRIC_HAMMING   = 2
} n4m_split_y_metric_t;

/* Group-aggregation mode for SPXYGFold (per-group representative). */
typedef enum n4m_split_aggregation_t {
    N4M_SPLIT_AGGREGATION_MEAN   = 0,
    N4M_SPLIT_AGGREGATION_MEDIAN = 1
} n4m_split_aggregation_t;

/* ---------- KennardStone ----------------------------------------------- */
typedef struct n4m_split_kennard_stone_handle_t n4m_split_kennard_stone_handle_t;
N4M_API n4m_status_t n4m_model_selection_kennard_stone_create(
    n4m_split_kennard_stone_handle_t** out, double test_size);
N4M_API void n4m_model_selection_kennard_stone_destroy(n4m_split_kennard_stone_handle_t* h);
N4M_API n4m_status_t n4m_model_selection_kennard_stone_split(
    const n4m_split_kennard_stone_handle_t* h,
    n4m_matrix_view_t X, n4m_split_result_t* out);

/* ---------- SPXY (single train/test, joint X-Y) ----------------------- */
typedef struct n4m_split_spxy_handle_t n4m_split_spxy_handle_t;
N4M_API n4m_status_t n4m_model_selection_spxy_create(n4m_split_spxy_handle_t** out,
                                            double test_size);
N4M_API void n4m_model_selection_spxy_destroy(n4m_split_spxy_handle_t* h);
N4M_API n4m_status_t n4m_model_selection_spxy_split(const n4m_split_spxy_handle_t* h,
                                           n4m_matrix_view_t X,
                                           n4m_matrix_view_t Y,
                                           n4m_split_result_t* out);

/* ---------- SPXYFold (k-fold) ---------------------------------------- */
typedef struct n4m_split_spxy_fold_handle_t n4m_split_spxy_fold_handle_t;
N4M_API n4m_status_t n4m_model_selection_spxy_fold_create(n4m_split_spxy_fold_handle_t** out,
                                                 int32_t n_splits,
                                                 int32_t y_metric);
N4M_API void n4m_model_selection_spxy_fold_destroy(n4m_split_spxy_fold_handle_t* h);
N4M_API n4m_status_t n4m_model_selection_spxy_fold_n_splits(
    const n4m_split_spxy_fold_handle_t* h, int32_t* out);
N4M_API n4m_status_t n4m_model_selection_spxy_fold_split_fold(
    const n4m_split_spxy_fold_handle_t* h,
    n4m_matrix_view_t X, n4m_matrix_view_t Y,
    int32_t fold_idx, n4m_split_result_t* out);

/* ---------- SPXYGFold (group-aware k-fold) --------------------------- */
typedef struct n4m_split_spxy_g_fold_handle_t n4m_split_spxy_g_fold_handle_t;
N4M_API n4m_status_t n4m_model_selection_spxy_g_fold_create(n4m_split_spxy_g_fold_handle_t** out,
                                                    int32_t n_splits,
                                                    int32_t y_metric,
                                                    int32_t aggregation);
N4M_API void n4m_model_selection_spxy_g_fold_destroy(n4m_split_spxy_g_fold_handle_t* h);
N4M_API n4m_status_t n4m_model_selection_spxy_g_fold_n_splits(
    const n4m_split_spxy_g_fold_handle_t* h, int32_t* out);
N4M_API n4m_status_t n4m_model_selection_spxy_g_fold_split_fold(
    const n4m_split_spxy_g_fold_handle_t* h,
    n4m_matrix_view_t X, n4m_matrix_view_t Y,
    const int64_t* groups, int64_t groups_len,
    int32_t fold_idx, n4m_split_result_t* out);

/* ---------- KMeans (k-means++) ---------------------------------------- */
typedef struct n4m_split_kmeans_handle_t n4m_split_kmeans_handle_t;
N4M_API n4m_status_t n4m_model_selection_kmeans_create(n4m_split_kmeans_handle_t** out,
                                               double test_size, uint64_t seed,
                                               int32_t max_iter);
N4M_API void n4m_model_selection_kmeans_destroy(n4m_split_kmeans_handle_t* h);
N4M_API n4m_status_t n4m_model_selection_kmeans_split(const n4m_split_kmeans_handle_t* h,
                                              n4m_matrix_view_t X,
                                              n4m_split_result_t* out);

/* ---------- KBinsStratified ------------------------------------------ */
typedef struct n4m_split_kbins_stratified_handle_t n4m_split_kbins_stratified_handle_t;
N4M_API n4m_status_t n4m_model_selection_kbins_stratified_create(
    n4m_split_kbins_stratified_handle_t** out,
    double test_size, uint64_t seed,
    int32_t n_bins, int32_t strategy);
N4M_API void n4m_model_selection_kbins_stratified_destroy(n4m_split_kbins_stratified_handle_t* h);
N4M_API n4m_status_t n4m_model_selection_kbins_stratified_split(
    const n4m_split_kbins_stratified_handle_t* h,
    n4m_matrix_view_t Y, n4m_split_result_t* out);

/* ---------- BinnedStratifiedGroupKFold ------------------------------- */
typedef struct n4m_split_binned_strat_group_kfold_handle_t
    n4m_split_binned_strat_group_kfold_handle_t;
N4M_API n4m_status_t n4m_model_selection_binned_strat_group_kfold_create(
    n4m_split_binned_strat_group_kfold_handle_t** out,
    int32_t n_splits, int32_t n_bins, int32_t strategy,
    int32_t shuffle, uint64_t seed);
N4M_API void n4m_model_selection_binned_strat_group_kfold_destroy(
    n4m_split_binned_strat_group_kfold_handle_t* h);
N4M_API n4m_status_t n4m_model_selection_binned_strat_group_kfold_n_splits(
    const n4m_split_binned_strat_group_kfold_handle_t* h, int32_t* out);
N4M_API n4m_status_t n4m_model_selection_binned_strat_group_kfold_split_fold(
    const n4m_split_binned_strat_group_kfold_handle_t* h,
    n4m_matrix_view_t Y,
    const int64_t* groups, int64_t groups_len,
    int32_t fold_idx, n4m_split_result_t* out);

/* ---------- SystematicCircular --------------------------------------- */
typedef struct n4m_split_systematic_circular_handle_t n4m_split_systematic_circular_handle_t;
N4M_API n4m_status_t n4m_model_selection_systematic_circular_create(
    n4m_split_systematic_circular_handle_t** out, double test_size, uint64_t seed);
N4M_API void n4m_model_selection_systematic_circular_destroy(
    n4m_split_systematic_circular_handle_t* h);
N4M_API n4m_status_t n4m_model_selection_systematic_circular_split(
    const n4m_split_systematic_circular_handle_t* h,
    n4m_matrix_view_t Y, n4m_split_result_t* out);

/* ---------- SPlit (data twinning) ------------------------------------ */
typedef struct n4m_split_split_splitter_handle_t n4m_split_split_splitter_handle_t;
N4M_API n4m_status_t n4m_model_selection_data_twinning_create(
    n4m_split_split_splitter_handle_t** out, double test_size, uint64_t seed);
N4M_API void n4m_model_selection_data_twinning_destroy(n4m_split_split_splitter_handle_t* h);
N4M_API n4m_status_t n4m_model_selection_data_twinning_split(
    const n4m_split_split_splitter_handle_t* h,
    n4m_matrix_view_t X, n4m_split_result_t* out);

N4M_API n4m_status_t n4m_model_selection_aom_pls_select(
    n4m_context_t* ctx,
    const n4m_config_t* cfg,
    const n4m_operator_bank_t* bank,
    const n4m_matrix_view_t* X,
    const n4m_matrix_view_t* Y,
    const n4m_validation_plan_t* plan,
    int32_t max_components,
    n4m_aom_global_result_t** out_result);

N4M_API void n4m_model_selection_aom_pls_result_destroy(n4m_aom_global_result_t* result);

N4M_API n4m_status_t n4m_model_selection_aom_pls_result_get_n_operators(
    const n4m_aom_global_result_t* result, int32_t* out);
N4M_API n4m_status_t n4m_model_selection_aom_pls_result_get_max_components(
    const n4m_aom_global_result_t* result, int32_t* out);
N4M_API n4m_status_t n4m_model_selection_aom_pls_result_get_selected_operator_index(
    const n4m_aom_global_result_t* result, int32_t* out);
N4M_API n4m_status_t n4m_model_selection_aom_pls_result_get_selected_n_components(
    const n4m_aom_global_result_t* result, int32_t* out);
N4M_API n4m_status_t n4m_model_selection_aom_pls_result_get_best_score(
    const n4m_aom_global_result_t* result, double* out);

N4M_API n4m_status_t n4m_model_selection_aom_pls_result_get_operator_kinds(
    const n4m_aom_global_result_t* result,
    const n4m_operator_kind_t** out_data, int32_t* out_size);
N4M_API n4m_status_t n4m_model_selection_aom_pls_result_get_operator_scores(
    const n4m_aom_global_result_t* result,
    const double** out_data, int32_t* out_size);
N4M_API n4m_status_t n4m_model_selection_aom_pls_result_get_rmse_curves(
    const n4m_aom_global_result_t* result,
    const double** out_data, int32_t* out_rows, int32_t* out_cols);
N4M_API n4m_status_t n4m_model_selection_aom_pls_result_get_predictions(
    const n4m_aom_global_result_t* result,
    const double** out_data, int64_t* out_rows, int64_t* out_cols);
N4M_API n4m_status_t n4m_model_selection_aom_pls_result_get_coefficients(
    const n4m_aom_global_result_t* result,
    const double** out_data, int64_t* out_rows, int64_t* out_cols);
N4M_API n4m_status_t n4m_model_selection_aom_pls_result_get_input_coefficients(
    const n4m_aom_global_result_t* result,
    const double** out_data, int64_t* out_rows, int64_t* out_cols);
N4M_API n4m_status_t n4m_model_selection_aom_pls_result_get_intercept(
    const n4m_aom_global_result_t* result,
    const double** out_data, int64_t* out_rows, int64_t* out_cols);

N4M_API n4m_status_t n4m_model_selection_pop_pls_select(
    n4m_context_t* ctx,
    const n4m_config_t* cfg,
    const n4m_operator_bank_t* bank,
    const n4m_matrix_view_t* X,
    const n4m_matrix_view_t* Y,
    const n4m_validation_plan_t* plan,
    int32_t max_components,
    n4m_aom_per_component_result_t** out_result);

N4M_API void n4m_model_selection_pop_pls_result_destroy(
    n4m_aom_per_component_result_t* result);

N4M_API n4m_status_t n4m_model_selection_pop_pls_result_get_n_operators(
    const n4m_aom_per_component_result_t* result, int32_t* out);
N4M_API n4m_status_t n4m_model_selection_pop_pls_result_get_max_components(
    const n4m_aom_per_component_result_t* result, int32_t* out);
N4M_API n4m_status_t n4m_model_selection_pop_pls_result_get_selected_n_components(
    const n4m_aom_per_component_result_t* result, int32_t* out);
N4M_API n4m_status_t n4m_model_selection_pop_pls_result_get_best_score(
    const n4m_aom_per_component_result_t* result, double* out);

N4M_API n4m_status_t n4m_model_selection_pop_pls_result_get_operator_kinds(
    const n4m_aom_per_component_result_t* result,
    const n4m_operator_kind_t** out_data, int32_t* out_size);
N4M_API n4m_status_t n4m_model_selection_pop_pls_result_get_selected_operator_indices(
    const n4m_aom_per_component_result_t* result,
    const int32_t** out_data, int32_t* out_size);
N4M_API n4m_status_t n4m_model_selection_pop_pls_result_get_component_scores(
    const n4m_aom_per_component_result_t* result,
    const double** out_data, int32_t* out_rows, int32_t* out_cols);
N4M_API n4m_status_t n4m_model_selection_pop_pls_result_get_prefix_scores(
    const n4m_aom_per_component_result_t* result,
    const double** out_data, int32_t* out_size);
N4M_API n4m_status_t n4m_model_selection_pop_pls_result_get_predictions(
    const n4m_aom_per_component_result_t* result,
    const double** out_data, int64_t* out_rows, int64_t* out_cols);
N4M_API n4m_status_t n4m_model_selection_pop_pls_result_get_coefficients(
    const n4m_aom_per_component_result_t* result,
    const double** out_data, int64_t* out_rows, int64_t* out_cols);
N4M_API n4m_status_t n4m_model_selection_pop_pls_result_get_input_coefficients(
    const n4m_aom_per_component_result_t* result,
    const double** out_data, int64_t* out_rows, int64_t* out_cols);
N4M_API n4m_status_t n4m_model_selection_pop_pls_result_get_intercept(
    const n4m_aom_per_component_result_t* result,
    const double** out_data, int64_t* out_rows, int64_t* out_cols);

/* Native sweep. ABI v1 supports exact Ridge CV over row-additive moments or
 * precomputed dual Ridge folds. Compatible single-target NIPALS/regression PLS1
 * component grids are scored from train/held-out moments; CUDA builds use a
 * device-resident cuBLAS component loop for very wide p>=1024 PLS1 moment
 * screens and a scalar host loop below that measured crossover. Other PLS
 * regimes use materialized fold-local prefix scoring. Batched/fused IKPLS
 * acceleration across many preprocessing variants is not part of ABI v1.
 *
 * `cv` must be in [2, n_samples] when `fold_ids` is NULL+0. In that case
 * contiguous balanced folds are generated. When `fold_ids` is provided,
 * `n_fold_ids` must equal n_samples; `cv <= 0` means infer max(fold_ids)+1.
 *
 * heads_mask bits: 1 = Ridge, 2 = PLS. At least one bit is required.
 * `ridge_lambdas` can be NULL+0 to use cfg.ridge_lambda, or a non-empty array
 * of finite lambdas >= 0. `pls_components` can be NULL+0 to use
 * cfg.n_components, or a non-empty array of positive component counts.
 * When `n4m_config_set_aom_score_only(cfg, 1)` is enabled, the sweep keeps
 * candidate scores and selected ids but returns model/prediction matrices as
 * empty 0 x 0 outputs.
 *
 * Result double matrices:
 *   "candidate_scores" (n_candidates x 4), row-major columns:
 *      candidate_id, head_id (0=Ridge, 1=PLS), param (lambda/components),
 *      cv_rmse
 *   "oof_predictions" (n_samples x n_targets) for the selected candidate
 *   "predictions"     (n_samples x n_targets) final refit in-sample
 *   "coefficients"    (n_features x n_targets), "intercept" (1 x n_targets)
 *   "x_mean" (1 x n_features), "x_scale" (1 x n_features),
 *   "y_mean" (1 x n_targets)
 * Int vectors:
 *   "fold_ids" (n_samples)
 * Scalars:
 *   "selected_candidate_id", "selected_head_id", "selected_param",
 *   "selected_cv_rmse", "n_candidates", "n_pls_moment_candidates",
 *   "n_pls_moment_cv_fits", "n_pls_moment_host_cv_fits",
 *   "n_pls_moment_cuda_device_cv_fits", "n_pls_materialized_cv_fits",
 *   "n_pls_moment_final_fits", "n_pls_moment_host_final_fits",
 *   "n_pls_moment_cuda_device_final_fits",
 *   "n_pls_materialized_final_fits",
 *   "score_only", "cv", "n_samples", "n_features", "n_targets"
 */
N4M_API n4m_status_t n4m_model_selection_sweep_run(
    n4m_context_t* ctx,
    const n4m_config_t* cfg,
    const n4m_matrix_view_t* X,
    const n4m_matrix_view_t* Y,
    int32_t cv,
    const int32_t* fold_ids,
    int64_t n_fold_ids,
    const double* ridge_lambdas,
    int64_t n_ridge_lambdas,
    const int32_t* pls_components,
    int64_t n_pls_components,
    int32_t heads_mask,
    n4m_method_result_t** out_result);

/* Native AOM preprocessing sweep. Applies the strict-linear AOM chain bank
 * selected by `profile` (0=compact, 1=wide), then runs the same Ridge/PLS
 * candidate screen as n4m_model_selection_sweep_run on every transformed matrix.
 *
 * Result double matrices mirror n4m_model_selection_sweep_run except:
 *   "candidate_scores" has shape (n_candidates x 5), row-major columns:
 *      candidate_id, chain_id, head_id (0=Ridge, 1=PLS),
 *      param (lambda/components), cv_rmse
 *   "chain_params" has shape (1 x n_chain_params) and stores the flat
 *      parameter payload for the exported chain descriptor.
 * Result int vectors additionally include:
 *   "candidate_routes" (n_candidates), route codes
 *      0=materialized, 1=dense operator moment, 2=banded operator moment,
 *      3=structured operator moment,
 *   "chain_offsets" (n_chains + 1), "op_kinds" (n_ops), and
 *   "param_offsets" (n_ops + 1). Together with "chain_params" they reproduce
 *   the exact strict-linear preprocessing bank used for candidate chain_id
 *   values.
 * Scalars additionally include:
 *   "selected_chain_id", "selected_sweep_candidate_id", "n_chains",
 *   "profile", route counters:
 *   "n_operator_moment_candidates",
 *   "n_ridge_operator_moment_candidates",
 *   "n_pls_operator_moment_candidates",
 *   "n_banded_operator_moment_candidates",
 *   "n_structured_operator_moment_candidates",
 *   "n_dense_operator_moment_candidates",
 *   "n_materialized_candidates",
 *   "n_ridge_materialized_candidates",
 *   "n_pls_materialized_candidates",
 *   "n_moment_prefix_cache_hits", "n_moment_prefix_cache_misses",
 *   "n_pls_moment_cv_fits", "n_pls_moment_host_cv_fits",
 *   "n_pls_moment_cuda_device_cv_fits", "n_pls_materialized_cv_fits",
 *   "n_pls_moment_final_fits", "n_pls_moment_host_final_fits",
 *   "n_pls_moment_cuda_device_final_fits",
 *   "n_pls_materialized_final_fits"
 */
N4M_API n4m_status_t n4m_model_selection_aom_sweep_run(
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
    const int32_t* pls_components,
    int64_t n_pls_components,
    int32_t heads_mask,
    n4m_method_result_t** out_result);

/* Native AOM preprocessing sweep over caller-provided strict-linear chains.
 *
 * Chain descriptor:
 *   chain_offsets: length n_chains + 1, monotonic, first=0, last=n_ops.
 *   op_kinds: length n_ops, values from n4m_operator_kind_t.
 *   param_offsets: length n_ops + 1, monotonic, first=0, last=n_params.
 *   params: flat double parameter payload.
 *
 * Supported strict-linear operators are identity, detrend polynomial,
 * Savitzky-Golay smooth/derivative, Norris-Williams, finite difference,
 * Whittaker, FCK and Gaussian. Empty chains are rejected; use an explicit
 * identity operator for the raw chain.
 *
 * Result double matrices/scalars match n4m_model_selection_aom_sweep_run. The scalar
 * "profile" is -1 to indicate a caller-provided chain descriptor.
 */
N4M_API n4m_status_t n4m_model_selection_aom_chain_sweep_run(
    n4m_context_t* ctx,
    const n4m_config_t* cfg,
    const n4m_matrix_view_t* X,
    const n4m_matrix_view_t* Y,
    int32_t cv,
    const int32_t* fold_ids,
    int64_t n_fold_ids,
    const int32_t* chain_offsets,
    int64_t n_chain_offsets,
    const int32_t* op_kinds,
    int64_t n_op_kinds,
    const int32_t* param_offsets,
    int64_t n_param_offsets,
    const double* params,
    int64_t n_params,
    const double* ridge_lambdas,
    int64_t n_ridge_lambdas,
    const int32_t* pls_components,
    int64_t n_pls_components,
    int32_t heads_mask,
    n4m_method_result_t** out_result);

/* Fit one already-selected caller-provided AOM chain/head/param on all rows
 * without running CV. This is a model-building endpoint, not a ranking
 * endpoint: the returned "selected_cv_rmse" and candidate score RMSE are NaN
 * unless a higher-level wrapper replaces them with an externally verified
 * score. `head_id` is 0=Ridge or 1=PLS; `param` is Ridge lambda or PLS
 * component count. Result matrices/scalars match n4m_model_selection_aom_chain_sweep_run, but
 * OOF predictions and fold ids are empty and "cv" is 0.
 */
N4M_API n4m_status_t n4m_model_selection_aom_chain_fixed_fit_run(
    n4m_context_t* ctx,
    const n4m_config_t* cfg,
    const n4m_matrix_view_t* X,
    const n4m_matrix_view_t* Y,
    const int32_t* chain_offsets,
    int64_t n_chain_offsets,
    const int32_t* op_kinds,
    int64_t n_op_kinds,
    const int32_t* param_offsets,
    int64_t n_param_offsets,
    const double* params,
    int64_t n_params,
    int32_t head_id,
    double param,
    n4m_method_result_t** out_result);

/* AOM robust-HPO preprocessing screen (strict-linear native subset).
 *
 * Profiles:
 *   profile = 0: compact bank (raw, detrend, SavGol variants, NW, diff,
 *                a few strict-linear compositions)
 *   profile = 1: wide bank (compact + wider SavGol/NW/Whittaker variants)
 *
 * heads_mask bits:
 *   1 = Ridge heads, 2 = PLS heads, 3 = both.
 *
 * The result contains:
 *   "predictions"                (n_samples x 1)
 *   "coefficients_transformed"   (n_transformed_features x 1)
 *   "intercept"                  (1 x 1)
 *   "candidate_scores"           (n_candidates x 4), row-major columns:
 *                                 chain_id, head_id (0=ridge, 1=pls),
 *                                 param, mean CV RMSE
 *   scalar "selected_chain_id", "selected_head_id", "selected_param",
 *          "selected_cv_rmse", "n_chains", "n_candidates", "profile", "cv"
 *
 * Native v1 is intentionally limited to strict-linear, shape-preserving
 * AOM operators so the screen can transform each chain once without
 * train-fold-fitted preprocessing leakage. The Python sklearn wrapper remains
 * the wider fold-local surface for stateful SNV/MSC/EMSC-style chains.
 */
N4M_API n4m_status_t n4m_model_selection_robust_hpo_fit(
    n4m_context_t* ctx,
    const n4m_config_t* cfg,
    const n4m_matrix_view_t* X,
    const n4m_matrix_view_t* Y,
    int32_t profile,
    int32_t cv,
    int32_t heads_mask,
    n4m_method_result_t** out_result);

/* AOM preprocessing fit/transform (Phase 6a). Applies the strict-linear
 * AOM operator bank through the gating strategy and returns both the
 * per-operator outputs and the gated/mixed transformed matrix. Supported
 * operator kinds are the strict AOM family (identity, detrend, Savitzky-Golay,
 * Norris-Williams, finite difference, Whittaker, FCK and Gaussian); non-strict
 * sample/reference-dependent operators are rejected. Y is optional and may be
 * NULL for this strict-linear bank. Result keys:
 *   "transformed"        (n × n_features) — final gated transform
 *   "operator_outputs"   (n_operators × (n × n_features), operator-major,
 *                         stored as a (n_operators × (n*n_features)) matrix)
 *   "weights"            (1 × n_operators) — gating weights at fit time
 *   int64 "operator_kinds" (n_operators) — N4M_OP_* enum values
 *   scalar "n_operators", scalar "mode" (gating mode integer)
 */
N4M_API n4m_status_t n4m_model_selection_aom_preprocessing_fit(
    n4m_context_t* ctx,
    const n4m_operator_bank_t* bank,
    const n4m_gating_strategy_t* gate,
    const n4m_matrix_view_t* X,
    const n4m_matrix_view_t* Y,
    n4m_method_result_t** out_result);

#ifdef __cplusplus
}  /* extern "C" */
#endif
#endif /* N4M_MODEL_SELECTION_H */
