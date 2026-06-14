/* SPDX-License-Identifier: CECILL-2.1 */
/* cpp/include/n4m/feature_selection.h — feature_selection role header (ABI 2.0). */
#ifndef N4M_FEATURE_SELECTION_H
#define N4M_FEATURE_SELECTION_H
#include "n4m/n4m.h"   /* shared infra: status, matrix view, context, N4M_API */

#ifdef __cplusplus
extern "C" {
#endif

typedef struct n4m_filter_variance_handle_t n4m_filter_variance_handle_t;
N4M_API n4m_status_t n4m_feature_selection_variance_create(
    n4m_filter_variance_handle_t** out, double threshold, int32_t top_k);
N4M_API void n4m_feature_selection_variance_destroy(n4m_filter_variance_handle_t* handle);
N4M_API n4m_status_t n4m_feature_selection_variance_fit(
    n4m_filter_variance_handle_t* handle, n4m_matrix_view_t X);
N4M_API n4m_status_t n4m_feature_selection_variance_transform(
    const n4m_filter_variance_handle_t* handle,
    n4m_matrix_view_t X, n4m_matrix_view_t out);
N4M_API n4m_status_t n4m_feature_selection_variance_output_cols(
    const n4m_filter_variance_handle_t* handle, int64_t* out_cols);
N4M_API n4m_status_t n4m_feature_selection_variance_is_fitted(
    const n4m_filter_variance_handle_t* handle, int* out_fitted);

typedef struct n4m_filter_correlation_handle_t n4m_filter_correlation_handle_t;
N4M_API n4m_status_t n4m_feature_selection_correlation_create(
    n4m_filter_correlation_handle_t** out, double threshold, int32_t top_k);
N4M_API void n4m_feature_selection_correlation_destroy(
    n4m_filter_correlation_handle_t* handle);
N4M_API n4m_status_t n4m_feature_selection_correlation_fit(
    n4m_filter_correlation_handle_t* handle,
    n4m_matrix_view_t X, const double* y, int64_t n_y);
N4M_API n4m_status_t n4m_feature_selection_correlation_transform(
    const n4m_filter_correlation_handle_t* handle,
    n4m_matrix_view_t X, n4m_matrix_view_t out);
N4M_API n4m_status_t n4m_feature_selection_correlation_output_cols(
    const n4m_filter_correlation_handle_t* handle, int64_t* out_cols);
N4M_API n4m_status_t n4m_feature_selection_correlation_is_fitted(
    const n4m_filter_correlation_handle_t* handle, int* out_fitted);

typedef struct n4m_interval_generator_handle_t n4m_interval_generator_handle_t;
N4M_API n4m_status_t n4m_feature_selection_interval_generator_create(
    n4m_interval_generator_handle_t** out, int32_t interval_size, int32_t step);
N4M_API void n4m_feature_selection_interval_generator_destroy(
    n4m_interval_generator_handle_t* handle);
N4M_API n4m_status_t n4m_feature_selection_interval_generator_fit(
    n4m_interval_generator_handle_t* handle, n4m_matrix_view_t X);
N4M_API n4m_status_t n4m_feature_selection_interval_generator_transform(
    const n4m_interval_generator_handle_t* handle,
    n4m_matrix_view_t X, n4m_matrix_view_t out);
N4M_API n4m_status_t n4m_feature_selection_interval_generator_output_cols(
    const n4m_interval_generator_handle_t* handle, int64_t* out_cols);
N4M_API n4m_status_t n4m_feature_selection_interval_generator_is_fitted(
    const n4m_interval_generator_handle_t* handle, int* out_fitted);

/* VIP / coefficient-magnitude / selectivity-ratio rankers (Phase 5a). Method
 * enum: 0=VIP, 1=coefficient magnitude, 2=selectivity ratio. */
N4M_API n4m_status_t n4m_feature_selection_variable_select_rank(
    n4m_context_t* ctx,
    const n4m_model_t* model,
    const n4m_matrix_view_t* X,
    int32_t method,
    int32_t top_k,
    n4m_method_result_t** out_result);

/* Interval / iPLS / mwPLS scan (Phase 5b). */
N4M_API n4m_status_t n4m_feature_selection_interval_select(
    n4m_context_t* ctx,
    const n4m_config_t* cfg,
    const n4m_matrix_view_t* X,
    const n4m_matrix_view_t* Y,
    const n4m_validation_plan_t* plan,
    int32_t interval_width,
    int32_t step,
    n4m_method_result_t** out_result);

/* MCUVE / coefficient-stability selector (Phase 5c). */
N4M_API n4m_status_t n4m_feature_selection_stability_select(
    n4m_context_t* ctx,
    const n4m_config_t* cfg,
    const n4m_matrix_view_t* X,
    const n4m_matrix_view_t* Y,
    const n4m_validation_plan_t* plan,
    int32_t top_k,
    n4m_method_result_t** out_result);

/* UVE — artificial-noise-thresholded selector (Phase 5d). */
N4M_API n4m_status_t n4m_feature_selection_uve_select(
    n4m_context_t* ctx,
    const n4m_config_t* cfg,
    const n4m_matrix_view_t* X,
    const n4m_matrix_view_t* Y,
    const n4m_validation_plan_t* plan,
    int32_t noise_features,
    uint64_t noise_seed,
    n4m_method_result_t** out_result);

/* SPA — Successive Projections Algorithm (Phase 5e). */
N4M_API n4m_status_t n4m_feature_selection_spa_select(
    n4m_context_t* ctx,
    const n4m_config_t* cfg,
    const n4m_matrix_view_t* X,
    const n4m_matrix_view_t* Y,
    int32_t top_k,
    n4m_method_result_t** out_result);

/* CARS — Competitive Adaptive Reweighted Sampling (Phase 5f). */
N4M_API n4m_status_t n4m_feature_selection_cars_select(
    n4m_context_t* ctx,
    const n4m_config_t* cfg,
    const n4m_matrix_view_t* X,
    const n4m_matrix_view_t* Y,
    const n4m_validation_plan_t* plan,
    int32_t n_iterations,
    int32_t min_features,
    n4m_method_result_t** out_result);

/* Random Frog (Phase 5g). */
N4M_API n4m_status_t n4m_feature_selection_random_frog_select(
    n4m_context_t* ctx,
    const n4m_config_t* cfg,
    const n4m_matrix_view_t* X,
    const n4m_matrix_view_t* Y,
    const n4m_validation_plan_t* plan,
    int32_t n_iterations,
    int32_t initial_size,
    int32_t min_size,
    int32_t max_size,
    int32_t top_k,
    uint64_t seed,
    n4m_method_result_t** out_result);

/* SCARS — Stability + CARS (Phase 5h). */
N4M_API n4m_status_t n4m_feature_selection_scars_select(
    n4m_context_t* ctx,
    const n4m_config_t* cfg,
    const n4m_matrix_view_t* X,
    const n4m_matrix_view_t* Y,
    const n4m_validation_plan_t* plan,
    int32_t n_iterations,
    int32_t min_features,
    double sample_fraction,
    uint64_t seed,
    n4m_method_result_t** out_result);

/* GA-PLS (Phase 5i). */
N4M_API n4m_status_t n4m_feature_selection_ga_select(
    n4m_context_t* ctx,
    const n4m_config_t* cfg,
    const n4m_matrix_view_t* X,
    const n4m_matrix_view_t* Y,
    const n4m_validation_plan_t* plan,
    int32_t n_generations,
    int32_t population_size,
    int32_t min_features,
    int32_t max_features,
    double mutation_rate,
    uint64_t seed,
    n4m_method_result_t** out_result);

/* PSO-PLS (§48): Binary Particle Swarm Optimization variable selection
 * (Kennedy & Eberhart 1997). Each particle is a binary mask over the
 * p features; fitness is the CV-RMSE of a PLS regression on the
 * selected subset. Position update uses sigmoid stochastic threshold
 * on the continuous velocity.
 *
 * Defaults from Clerc & Kennedy (2002) convergence analysis:
 *   w = 0.729 (inertia), c1 = c2 = 1.494, v_max = 4.0.
 *
 * Result keys:
 *   "inclusion_frequencies"     (1 × n_features)
 *   "best_rmse_by_iteration"    (1 × n_iterations)
 *   "mean_rmse_by_iteration"    (1 × n_iterations)
 *   int64 "selected_indices"
 *   scalars: n_features, n_targets, n_components, n_swarm, n_iterations,
 *            w, c1, c2, v_max, seed, best_rmse
 */
N4M_API n4m_status_t n4m_feature_selection_pso_select(
    n4m_context_t* ctx,
    const n4m_config_t* cfg,
    const n4m_matrix_view_t* X,
    const n4m_matrix_view_t* Y,
    const n4m_validation_plan_t* plan,
    int32_t n_swarm,
    int32_t n_iterations,
    double w,
    double c1,
    double c2,
    double v_max,
    uint64_t seed,
    n4m_method_result_t** out_result);

/* VISSA-PLS (§49): Variable Iterative Space Shrinkage Approach.
 * Reference: Deng B., Yun Y., Liang Y. (2014) Anal. Chim. Acta 838:27-40.
 * Paper-only — no widely installable port.
 *
 * Weighted Binary Matrix Sampling iteratively shrinks the per-feature
 * inclusion probabilities by averaging the masks of the top-K best
 * submodels per iteration. floor_probability clamps w_j to
 * [floor, 1-floor] each iteration to preserve exploration.
 *
 * Defaults: n_iterations=20, n_submodels=100, ratio_kept=0.1,
 * threshold=0.5, floor_probability=0.01.
 *
 * Result keys:
 *   "final_probabilities"   (1 × p) — final w_j vector
 *   "inclusion_frequencies" (1 × p) — alias for final_probabilities
 *   "best_rmse_by_iteration"  (1 × n_iterations)
 *   "mean_rmse_by_iteration"  (1 × n_iterations)
 *   "top_k_per_iteration"   (n_iterations × p)
 *   int64 "selected_indices" — {j : w_j > threshold}
 *   scalars: n_features, n_targets, n_components, n_iterations,
 *            n_submodels, ratio_kept, threshold, floor_probability,
 *            seed, best_rmse
 */
N4M_API n4m_status_t n4m_feature_selection_vissa_select(
    n4m_context_t* ctx,
    const n4m_config_t* cfg,
    const n4m_matrix_view_t* X,
    const n4m_matrix_view_t* Y,
    const n4m_validation_plan_t* plan,
    int32_t n_iterations,
    int32_t n_submodels,
    double ratio_kept,
    double threshold,
    double floor_probability,
    uint64_t seed,
    n4m_method_result_t** out_result);

/* Shaving (Phase 5j). */
N4M_API n4m_status_t n4m_feature_selection_shaving_select(
    n4m_context_t* ctx,
    const n4m_config_t* cfg,
    const n4m_matrix_view_t* X,
    const n4m_matrix_view_t* Y,
    const n4m_validation_plan_t* plan,
    int32_t n_steps,
    int32_t min_features,
    double shave_fraction,
    n4m_method_result_t** out_result);

/* BVE-PLS (Phase 5k). */
N4M_API n4m_status_t n4m_feature_selection_bve_select(
    n4m_context_t* ctx,
    const n4m_config_t* cfg,
    const n4m_matrix_view_t* X,
    const n4m_matrix_view_t* Y,
    const n4m_validation_plan_t* plan,
    int32_t n_steps,
    int32_t min_features,
    n4m_method_result_t** out_result);

/* T2-PLS (Phase 5l). `alpha_thresholds` is an array of `n_alphas` α values
 * in (0, 1). */
N4M_API n4m_status_t n4m_feature_selection_t2_select(
    n4m_context_t* ctx,
    const n4m_config_t* cfg,
    const n4m_matrix_view_t* X,
    const n4m_matrix_view_t* Y,
    const n4m_validation_plan_t* plan,
    const double* alpha_thresholds,
    int64_t n_alphas,
    int32_t min_selected,
    n4m_method_result_t** out_result);

/* WVC-PLS (Phase 5m). */
N4M_API n4m_status_t n4m_feature_selection_wvc_select(
    n4m_context_t* ctx,
    const n4m_matrix_view_t* X,
    const n4m_matrix_view_t* Y,
    int32_t n_components,
    int32_t top_k,
    int32_t normalize,
    n4m_method_result_t** out_result);

/* WVC-threshold (Phase 5r). */
N4M_API n4m_status_t n4m_feature_selection_wvc_threshold_select(
    n4m_context_t* ctx,
    const n4m_matrix_view_t* X,
    const n4m_matrix_view_t* Y,
    int32_t n_components,
    int32_t normalize,
    double score_threshold,
    double threshold_factor,
    int32_t min_selected,
    n4m_method_result_t** out_result);

/* EMCUVE (Phase 5n). */
N4M_API n4m_status_t n4m_feature_selection_emcuve_select(
    n4m_context_t* ctx,
    const n4m_config_t* cfg,
    const n4m_matrix_view_t* X,
    const n4m_matrix_view_t* Y,
    const n4m_validation_plan_t* plan,
    int32_t noise_features,
    uint64_t noise_seed,
    int32_t n_ensembles,
    double vote_threshold,
    n4m_method_result_t** out_result);

/* Randomization-test selector (Phase 5o). */
N4M_API n4m_status_t n4m_feature_selection_randomization_select(
    n4m_context_t* ctx,
    const n4m_config_t* cfg,
    const n4m_matrix_view_t* X,
    const n4m_matrix_view_t* Y,
    int32_t n_permutations,
    uint64_t randomization_seed,
    double alpha,
    n4m_method_result_t** out_result);

/* biPLS — backward iPLS (Phase 5p). */
N4M_API n4m_status_t n4m_feature_selection_bipls_select(
    n4m_context_t* ctx,
    const n4m_config_t* cfg,
    const n4m_matrix_view_t* X,
    const n4m_matrix_view_t* Y,
    const n4m_validation_plan_t* plan,
    int32_t interval_width,
    int32_t min_intervals,
    n4m_method_result_t** out_result);

/* siPLS — synergistic interval PLS (Phase 5q). */
N4M_API n4m_status_t n4m_feature_selection_sipls_select(
    n4m_context_t* ctx,
    const n4m_config_t* cfg,
    const n4m_matrix_view_t* X,
    const n4m_matrix_view_t* Y,
    const n4m_validation_plan_t* plan,
    int32_t interval_width,
    int32_t combination_size,
    n4m_method_result_t** out_result);

/* REP-PLS (Phase 5s). */
N4M_API n4m_status_t n4m_feature_selection_rep_select(
    n4m_context_t* ctx,
    const n4m_config_t* cfg,
    const n4m_matrix_view_t* X,
    const n4m_matrix_view_t* Y,
    const n4m_validation_plan_t* plan,
    int32_t n_steps,
    int32_t min_features,
    int32_t remove_count,
    n4m_method_result_t** out_result);

/* IPW-PLS (Phase 5t). */
N4M_API n4m_status_t n4m_feature_selection_ipw_select(
    n4m_context_t* ctx,
    const n4m_config_t* cfg,
    const n4m_matrix_view_t* X,
    const n4m_matrix_view_t* Y,
    const n4m_validation_plan_t* plan,
    int32_t n_iterations,
    int32_t top_k,
    double damping,
    double weight_floor,
    n4m_method_result_t** out_result);

/* ST-PLS — score-threshold (Phase 5u). `thresholds` array of `n_thresholds`
 * positive doubles. */
N4M_API n4m_status_t n4m_feature_selection_st_select(
    n4m_context_t* ctx,
    const n4m_config_t* cfg,
    const n4m_matrix_view_t* X,
    const n4m_matrix_view_t* Y,
    const n4m_validation_plan_t* plan,
    const double* thresholds,
    int64_t n_thresholds,
    int32_t min_selected,
    n4m_method_result_t** out_result);

/* IRIV — Iteratively Retains Informative Variables (Phase 51). Yun 2014.
 * Iterative backward variable selection driven by a Mann-Whitney U test
 * on permuted-replacement CV-RMSE values.
 *   cfg.n_components — capped per round to surviving feature count.
 *   max_rounds       — hard cap on IRIV iterations (the algorithm stops
 *                      earlier if no variables are flagged uninformative).
 *   seed             — splitmix64 seed driving the binary-mask generator.
 * The result contains:
 *   "remaining_per_round" (1 x (n_rounds+1)) — features alive after each round
 *   "removed_per_round"   (1 x n_rounds)
 *   int64 "selected_indices"
 *   scalars: n_features, n_targets, n_components, n_rounds,
 *            binary_matrix_rows, seed
 */
N4M_API n4m_status_t n4m_feature_selection_iriv_select(
    n4m_context_t* ctx,
    const n4m_config_t* cfg,
    const n4m_matrix_view_t* X,
    const n4m_matrix_view_t* Y,
    const n4m_validation_plan_t* plan,
    int32_t max_rounds,
    uint64_t seed,
    n4m_method_result_t** out_result);

/* IRF — Interval Random Frog (Phase 52). Yun 2013. Random Frog applied to
 * fixed-width sliding spectral intervals (window of `window_size` features
 * each). Final selection is the union of features under the top-K
 * inclusion-frequency intervals.
 *   window_size       — interval width (1 <= w <= n_features)
 *   initial_intervals — Q in the libPLS paper (initial subset size)
 *   top_k             — number of best intervals to union into the
 *                       returned feature set
 *   seed              — splitmix64 seed
 * Result contains:
 *   "probability"        (1 x n_intervals)
 *   "rmse_by_iteration"  (1 x n_iterations)
 *   "subset_sizes"       (1 x n_iterations)
 *   int64 "top_k_intervals"
 *   int64 "selected_indices"  (union of top-K intervals)
 *   scalars: n_features, n_targets, n_components, n_iterations, window_size,
 *            initial_intervals, n_intervals, top_k, seed, best_rmse
 */
N4M_API n4m_status_t n4m_feature_selection_irf_select(
    n4m_context_t* ctx,
    const n4m_config_t* cfg,
    const n4m_matrix_view_t* X,
    const n4m_matrix_view_t* Y,
    const n4m_validation_plan_t* plan,
    int32_t n_iterations,
    int32_t window_size,
    int32_t initial_intervals,
    int32_t top_k,
    uint64_t seed,
    n4m_method_result_t** out_result);

/* VIP_SPA — VIP-then-SPA hybrid variable selection (Phase 53).
 * Composition of Favilla 2013 VIP scoring (mask features by VIP > threshold)
 * and Araujo 2001 Successive Projections Algorithm (greedy collinearity-
 * minimising pick over the surviving subset). Matches auswahl.VIP_SPA's
 * LSX-UniWue algorithm; the SPA start is taken as argmax(VIP) within the
 * surviving subset (deterministic) rather than auswahl's seed enumeration,
 * and successive projections are computed on raw masked X to match
 * auswahl._spa._fit_spa (which L2-normalizes only the current direction).
 *   cfg.n_components — components used to fit PLS for VIP scoring
 *   vip_threshold    — drop any feature with VIP <= threshold (auswahl default 0.3)
 *   top_k            — upper bound on selected features; truncated to the
 *                      surviving-candidate count when that is smaller.
 * Result contains:
 *   "vip_scores"        (1 x n_features)
 *   "vip_mask"          (1 x n_features) as 0/1 doubles
 *   "selection_scores"  (1 x n_selected)
 *   int64 "selected_indices" (length n_selected)
 *   scalars: n_features, n_targets, n_components, top_k (requested),
 *            n_selected (actual count), n_candidates, vip_threshold
 */
N4M_API n4m_status_t n4m_feature_selection_vip_spa_select(
    n4m_context_t* ctx,
    const n4m_config_t* cfg,
    const n4m_matrix_view_t* X,
    const n4m_matrix_view_t* Y,
    double vip_threshold,
    int32_t top_k,
    n4m_method_result_t** out_result);

#ifdef __cplusplus
}  /* extern "C" */
#endif
#endif /* N4M_FEATURE_SELECTION_H */
