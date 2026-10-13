/* SPDX-License-Identifier: CECILL-2.1 */
/* cpp/include/n4m/metrics.h — metrics role header (ABI 2.0). */
#ifndef N4M_METRICS_H
#define N4M_METRICS_H
#include "n4m/n4m.h"   /* shared infra: status, matrix view, context, N4M_API */

#ifdef __cplusplus
extern "C" {
#endif

/* Preflight diagnostics. All outputs are caller-owned. X and matrix outputs
 * accept positive element strides and F64 only. PCA requires n>=2, p>=1,
 * 1<=k<=min(n,p); scores is n x k, loadings k x p, eigenvalues has k
 * elements, and full_moments has two elements: sum(lambda) and
 * sum(lambda^2), over the FULL centered covariance spectrum (ddof=1).
 * The moments remain exact when PCA is truncated or approximated. */
N4M_API n4m_status_t n4m_preflight_pca_compute(
    n4m_context_t* ctx, const n4m_matrix_view_t* X, int32_t k,
    const n4m_matrix_view_t* scores, const n4m_matrix_view_t* loadings,
    double* eigenvalues, double* full_moments);

/* Pooled within-group repeatability for each feature. group_ids has X.rows
 * elements. valid_mask is optional row-major X.rows x X.cols bytes; nonzero
 * means valid. All unmasked X values must be finite. Output arrays have
 * X.cols elements. pooled_sd is sqrt(sum within-group squared deviations /
 * sum(r_i-1)) over groups with r_i>=2; NaN when that denominator is zero.
 * ICC(1,1) is computed only when at least two complete groups share the
 * same r>=2. An unbalanced feature keeps its pooled_sd but has NaN ICC.
 * band_status: 0=ICC computed, 1=insufficient groups/repeats,
 * 2=unbalanced, 3=zero total variance. Negative ICC estimates are retained.
 * n_complete_groups counts groups having any valid observations. */
N4M_API n4m_status_t n4m_preflight_repeatability_compute(
    n4m_context_t* ctx, const n4m_matrix_view_t* X,
    const int64_t* group_ids, const uint8_t* valid_mask,
    double* pooled_sd, double* icc, int64_t* n_complete_groups,
    int32_t* band_status);

/* Eight finite-value summary outputs: count, min, max, mean, sample SD,
 * q25, median, q75. Quantiles use linear type-7 interpolation. The optional
 * mask has n bytes; zero excludes a value. Unmasked nonfinite values fail.
 * For no values, all fields except count=0 are NaN; SD is NaN for n<2. */
N4M_API n4m_status_t n4m_preflight_summary_compute(
    n4m_context_t* ctx, const double* values, const uint8_t* valid_mask,
    int64_t n, double* out_eight);

/* Euclidean k-nearest reference rows for each query row, sorted by ascending
 * distance then reference index. When BOTH group arrays are non-NULL, rows
 * with the same group ID are excluded. Outputs are row-major n_query x k,
 * using index=-1 and distance=NaN when too few eligible neighbors remain.
 * All input values must be finite. k>0; callers own every buffer. */
N4M_API n4m_status_t n4m_preflight_nearest_neighbors(
    n4m_context_t* ctx, const n4m_matrix_view_t* query,
    const n4m_matrix_view_t* reference, const int64_t* query_groups,
    const int64_t* reference_groups, int32_t k, int64_t* indices,
    double* distances);

/* Local target contrast over a row-major n x k neighbor-index table. -1
 * marks an unavailable slot. Every real neighbor must differ from its row
 * and, when group_ids is non-NULL, from that row's group. All targets must
 * be finite. categorical=1 requires exactly representable integer category
 * codes (|code| <= 2^53-1); categorical=0 accepts continuous targets.
 * out_four[1] is the fraction of rows with >=1 eligible neighbor. Each such
 * row contributes equally, after averaging its available neighbors.
 * Continuous contrast out_four[0] is 1 minus mean row squared Y difference
 * divided by (2*sample variance of all n targets).
 * Categorical contrast is (sum_i expected_i - observed_i)/sum_i expected_i,
 * where expected_i is the disagreement fraction among all admissible rows
 * for focal row i (distinct rows, and distinct groups if supplied), and
 * observed_i is the disagreement fraction among that row's neighbors.
 * out_four[2:4] are the observed and expected SAME-class fractions, each
 * averaged equally over eligible focal rows. Continuous sets these to NaN.
 * A zero denominator or no eligible rows yields NaN contrast, with N4M_OK
 * and coverage still reported. */
N4M_API n4m_status_t n4m_preflight_local_contrast(
    n4m_context_t* ctx, const double* targets, const int64_t* neighbor_indices,
    const int64_t* group_ids, int64_t n, int32_t k, int32_t categorical,
    double* out_four);

/* Association between one categorical factor and one target over n
 * exchangeable analysis units (one row per unit), with an optional seeded
 * Monte Carlo permutation test of the target labels over all units (a single
 * unrestricted block). factor_codes are arbitrary int64 level codes. targets
 * must be finite; categorical=1 requires exactly representable integer class
 * codes (|code| <= 2^53-1). n >= 2, 0 <= n_permutations <= 1000000, and
 * (n + levels*classes) * (n_permutations + 1) <= 250 million operations.
 * Permutation draws use PCG64 seeded like numpy.random.default_rng(seed); the
 * b-th draw equals the b-th rng.permutation(n), target i -> targets[perm[i]].
 * out_eight (caller-owned, NaN when undefined):
 *   [0] effect size: eta^2 = SS_between/SS_total (continuous) or Cramer's V
 *       (categorical); NaN with fewer than two levels or target values.
 *   [1] statistic: one-way ANOVA F (NaN when n == levels or SS_within == 0)
 *       or Pearson chi-square without continuity correction.
 *   [2] df1: levels-1, or (levels-1)*(classes-1).
 *   [3] df2: n-levels (continuous), NaN (categorical).
 *   [4] number of factor levels; [5] number of distinct target values.
 *   [6] log10 of min(n!/prod(level_count!), n!/prod(target_multiplicity!)),
 *       an upper bound on distinct permutation outcomes of the statistic.
 *   [7] Monte Carlo p-value (1+b)/(1+B); NaN when B == 0 or [0] is NaN.
 * The permutation statistic is eta^2 or Cramer's V^2; b counts permuted
 * values >= observed - sqrt(DBL_EPSILON). out_two = {b, executed B}; no
 * permutation runs when the effect size is undefined. */
N4M_API n4m_status_t n4m_preflight_factor_association(
    n4m_context_t* ctx, const int64_t* factor_codes, const double* targets,
    int64_t n, int32_t categorical, int32_t n_permutations, uint64_t seed,
    double* out_eight, int64_t* out_two);

/* Holm step-down adjusted p-values for one family of m >= 1 hypotheses.
 * Inputs must lie in [0, 1]; adjusted[i] = min(1, max_{j<=rank(i)}
 * (m-j+1) p_(j)) with a stable ascending order, as R p.adjust("holm"). */
N4M_API n4m_status_t n4m_preflight_holm_adjust(
    n4m_context_t* ctx, const double* p_values, int64_t m, double* adjusted);

/* ---------- 20.2 NIRS regression metrics ------------------------------ */
N4M_API n4m_status_t n4m_metrics_regression_metrics_rmse (const double* y_true,
                                       const double* y_pred,
                                       int64_t n, double* out);
N4M_API n4m_status_t n4m_metrics_regression_metrics_mae  (const double* y_true,
                                       const double* y_pred,
                                       int64_t n, double* out);
N4M_API n4m_status_t n4m_metrics_regression_metrics_bias (const double* y_true,
                                       const double* y_pred,
                                       int64_t n, double* out);
N4M_API n4m_status_t n4m_metrics_regression_metrics_sep  (const double* y_true,
                                       const double* y_pred,
                                       int64_t n, double* out);
N4M_API n4m_status_t n4m_metrics_regression_metrics_rpd  (const double* y_true,
                                       const double* y_pred,
                                       int64_t n, double* out);
N4M_API n4m_status_t n4m_metrics_regression_metrics_rpiq (const double* y_true,
                                       const double* y_pred,
                                       int64_t n, double* out);
N4M_API n4m_status_t n4m_metrics_regression_metrics_r2   (const double* y_true,
                                       const double* y_pred,
                                       int64_t n, double* out);
N4M_API n4m_status_t n4m_metrics_regression_metrics_nrmse(const double* y_true,
                                       const double* y_pred,
                                       int64_t n, double* out);

/* Approximate-PRESS component selection (§29). For each component count
 * k in [1, max_components], fits SIMPLS, then approximates PRESS via
 * leverage-inflated in-sample residuals. The result contains:
 *   "press_per_component" (1 x max_components)
 *   "rmse_per_component"  (1 x max_components)
 *   int "selected_n_components" — argmin of press_per_component
 *   scalar "selected_n_components_d" — same as a double for convenience
 */
N4M_API n4m_status_t n4m_metrics_approximate_press_compute(
    n4m_context_t* ctx,
    const n4m_config_t* cfg,
    const n4m_matrix_view_t* X,
    const n4m_matrix_view_t* Y,
    int32_t max_components,
    n4m_method_result_t** out_result);

/* PLS process monitoring (§19). Computes Hotelling T² and Q-residual
 * thresholds from `X_reference` (phase-1) at confidence level
 * (1 - alpha), then evaluates `X_monitor` (phase-2) and flags rows
 * exceeding the thresholds. Result contains:
 *   "t2"            (1 x n_monitor)
 *   "q"             (1 x n_monitor)
 *   "t2_reference"  (1 x n_reference)
 *   "q_reference"   (1 x n_reference)
 *   int "t2_alarms"  (length n_monitor)
 *   int "q_alarms"   (length n_monitor)
 *   int "any_alarms" (length n_monitor)
 *   scalar "t2_threshold", scalar "q_threshold", scalar "alpha"
 */
N4M_API n4m_status_t n4m_metrics_pls_monitoring_run(
    n4m_context_t* ctx,
    const n4m_model_t* model,
    const n4m_matrix_view_t* X_reference,
    const n4m_matrix_view_t* X_monitor,
    double alpha,
    n4m_method_result_t** out_result);

/* One-SE rule for PLS component selection (§10). Given a (max_components
 * x n_folds) row-major matrix of fold RMSE values, returns the smallest
 * k whose mean fold RMSE is within one standard error of the best mean
 * fold RMSE. Result contains:
 *   "mean_rmse_per_component" (1 x max_components)
 *   int "best_n_components"       (length 1)
 *   int "one_se_n_components"     (length 1)
 *   scalar "one_se_standard_error", scalar "one_se_threshold"
 */
N4M_API n4m_status_t n4m_metrics_one_se_rule_compute(
    n4m_context_t* ctx,
    const double* fold_rmse_matrix,
    int32_t max_components,
    int32_t n_folds,
    n4m_method_result_t** out_result);

/* PLS diagnostics (§9). Computes Hotelling T², Q residuals (SPE) and
 * DModX from a fitted model and a design matrix X. When `X_reference`
 * is NULL, score variances and sigma_train fall back to the model's
 * stored training scores — this requires `cfg.store_scores = 1` at fit
 * time. When `X_reference` is non-NULL, its rows define the reference
 * score distribution.
 *
 * The result contains:
 *   "t2"     (1 x n_samples) — Hotelling T² statistic per row
 *   "q"      (1 x n_samples) — squared reconstruction residual per row
 *   "dmodx"  (1 x n_samples) — distance-to-model X per row
 *   scalar "n_components", scalar "n_features"
 */
N4M_API n4m_status_t n4m_metrics_pls_diagnostics_compute(
    n4m_context_t* ctx,
    const n4m_model_t* model,
    const n4m_matrix_view_t* X,
    const n4m_matrix_view_t* X_reference,
    n4m_method_result_t** out_result);

#ifdef __cplusplus
}  /* extern "C" */
#endif
#endif /* N4M_METRICS_H */
