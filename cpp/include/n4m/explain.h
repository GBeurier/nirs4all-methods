/* SPDX-License-Identifier: CECILL-2.1 OR AGPL-3.0-or-later */
#ifndef N4M_EXPLAIN_H
#define N4M_EXPLAIN_H
#include "n4m/n4m.h"
#ifdef __cplusplus
extern "C" {
#endif

/* ============================================================================
 * Model explanation (attribution) kernels.
 * ============================================================================
 *
 * Conventions shared by every function in this header:
 *
 * - Every input matrix is an F64 strided n4m_matrix_view_t. X holds the rows
 *   to explain (n_rows x n_features) in the SAME feature space the explained
 *   function consumes. Hosts decide and record whether that space is the raw
 *   spectral input of a complete fitted pipeline or the transformed features
 *   of a terminal estimator; this library never relabels axes.
 * - Feature groups: feature_group[j] in [0, n_groups) assigns feature j to one
 *   group. Groups form a PARTITION: every feature belongs to exactly one group
 *   and every group is nonempty. A NULL feature_group means one group per
 *   feature (n_groups == n_features).
 * - Attributions are caller-owned row-major [row][group][output] arrays of
 *   n_rows * n_groups * n_outputs doubles. expected / predictions are
 *   [row][output] arrays of n_rows * n_outputs doubles.
 * - Output columns are whatever the host predicts (regression targets in
 *   original units, one class probability, a decision score). The library
 *   never chooses or relabels an output space.
 * - Invalid input fails with a status and a context message without writing
 *   any output buffer.
 *
 * Additivity: for the SHAP-family estimators (exact, permutation, kernel,
 * linear and tree), sum_g attributions[r][g][o] + expected[r][o] equals
 * predictions[r][o] (exactly for exact/linear/tree; enforced by construction
 * for permutation and kernel). LIME and occlusion are NOT additive.
 */

/* ---------------------------------------------------------------------------
 * 1. Ask/tell session: model-agnostic explainers.
 * ---------------------------------------------------------------------------
 * The session never calls a model. It hands the host batches of perturbed
 * rows to predict (next_batch) and receives the predictions back (submit),
 * so any backend works: Python estimators, WASM pipelines, ONNX Runtime Web,
 * asynchronous JS workers. Every requested row lives in the explained feature
 * space. The session plan is deterministic for a given (inputs, options,
 * seed); next_batch/submit must strictly alternate. Once next_batch writes 0
 * rows, every evaluation was submitted and result_v1 is available. */

typedef enum n4m_explain_method_t {
    /* Exact Shapley values of the group game, enumerating all 2^G coalitions.
     * Limited to n_groups <= 16. Validation oracle and small-group explainer. */
    N4M_EXPLAIN_EXACT_SHAP = 0,
    /* Antithetic permutation sampling of the group game (forward and reversed
     * permutation per draw, as in shap.PermutationExplainer). Unbiased; additive
     * by construction (each chain telescopes from baseline to x). */
    N4M_EXPLAIN_PERMUTATION_SHAP = 1,
    /* Kernel SHAP: Shapley-kernel weighted least squares over sampled
     * coalitions with the efficiency constraint enforced exactly. */
    N4M_EXPLAIN_KERNEL_SHAP = 2,
    /* LIME over binary group masks: exponential kernel on the cosine distance
     * between the mask and the all-present mask, ridge-weighted linear fit.
     * expected holds the local model intercept. Not additive. */
    N4M_EXPLAIN_LIME = 3,
    /* Occlusion / ablation: attribution[g] = f(x) - f(x with group g replaced
     * by the baseline). expected holds f(baseline). Not additive. */
    N4M_EXPLAIN_OCCLUSION = 4,
    /* ABI 2.24. Local gradient (first order, not a Shapley method): per
     * explained row x and feature j, the central finite difference
     *   g_j = (f(x + h_j e_j) - f(x - h_j e_j)) / ((x_j + h_j) - (x_j - h_j))
     * (2 n_features evaluations per row, plus f(x) per row and f(r) once),
     * and the first-order attribution phi_j = g_j (x_j - r_j) with the
     * reference r = column mean of the background rows (the BACKGROUND and
     * MEAN baselines both select this mean reference; INTERPOLATE is
     * refused). Grouped attributions are sums over the group's features.
     * expected = f(r), predictions = f(x). Not additive in general
     * (sum_j phi_j + f(r) = f(x) exactly when f is affine, where the result
     * equals linear SHAP with the mean reference). Step h_j = step * s_j,
     * s_j the population standard deviation of background column j, or
     * |r_j| when that is 0 (1 when r_j is 0 too); step defaults to 1e-3
     * (n4m_explain_session_set_gradient_step_v1). The gradients are read
     * with n4m_explain_session_gradients_v1. */
    N4M_EXPLAIN_GRADIENT = 5
} n4m_explain_method_t;

/* How an "absent" group is filled. */
typedef enum n4m_explain_baseline_t {
    /* Interventional: average over every background row (absent features take
     * the background row values). expected = mean_b f(background_b). */
    N4M_EXPLAIN_BASELINE_BACKGROUND = 0,
    /* Single reference row: column mean of the background rows. */
    N4M_EXPLAIN_BASELINE_MEAN = 1,
    /* Spectral bridge: every maximal run of absent features (adjacent absent
     * groups merge into one run) is replaced, in the explained row, by the
     * straight line joining the explained row's values at the nearest PRESENT
     * features on each side (feature index order). A run touching an edge is
     * filled flat with its single present neighbour; when every feature is
     * absent the row is replaced by the line between its first and last
     * values. Keeps spectra continuous for highly correlated neighbouring
     * wavelengths. Requires every group to be a contiguous index range; the
     * background may be empty. expected is therefore per row. */
    N4M_EXPLAIN_BASELINE_INTERPOLATE = 2
} n4m_explain_baseline_t;

typedef struct n4m_explain_options_v1_t {
    uint64_t struct_size;
    int32_t method;          /* n4m_explain_method_t */
    int32_t baseline;        /* n4m_explain_baseline_t */
    int64_t n_outputs;       /* prediction columns per row, >= 1 */
    int64_t n_permutations;  /* PERMUTATION: antithetic pairs per row (>= 1, default 10) */
    int64_t n_coalitions;    /* KERNEL/LIME: sampled coalitions per row (0 = auto:
                                min(2^G - 2, 2*G + 2048) for KERNEL, 1000 for LIME) */
    double lime_kernel_width; /* LIME: exponential kernel width (<= 0 = auto: 0.75*sqrt(G)) */
    double lime_ridge;        /* LIME: ridge penalty on coefficients (>= 0, default 1.0) */
    int64_t max_batch_rows;   /* upper bound of rows per requested batch (default 4096) */
    uint64_t seed;            /* sampling seed (PERMUTATION/KERNEL/LIME) */
} n4m_explain_options_v1_t;

typedef struct n4m_explain_session_s n4m_explain_session_t;

N4M_API n4m_status_t n4m_explain_options_init_v1(n4m_explain_options_v1_t* options);

/* Copies X, background and feature_group; the caller may free them after
 * return. background must have n_features columns and >= 1 row for the
 * BACKGROUND and MEAN baselines; it may have 0 rows for INTERPOLATE. */
N4M_API n4m_status_t n4m_explain_session_create_v1(n4m_context_t* ctx,
                                                   const n4m_matrix_view_t* X,
                                                   const n4m_matrix_view_t* background,
                                                   const int64_t* feature_group,
                                                   int64_t n_groups,
                                                   const n4m_explain_options_v1_t* options,
                                                   n4m_explain_session_t** out_session);

N4M_API void n4m_explain_session_destroy(n4m_explain_session_t* session);

/* n_rows, n_features, n_groups, n_outputs and the total number of rows the
 * session will request (exact plan size) for progress reporting. Any output
 * pointer may be NULL. */
N4M_API n4m_status_t n4m_explain_session_info_v1(const n4m_explain_session_t* session,
                                                 int64_t* n_rows,
                                                 int64_t* n_features,
                                                 int64_t* n_groups,
                                                 int64_t* n_outputs,
                                                 int64_t* total_evaluations);

/* Writes up to min(capacity_rows, max_batch_rows) perturbed rows, row-major
 * (rows x n_features), into caller-owned out_rows and their count into
 * out_rows_written. 0 rows means the plan is complete. Calling next_batch
 * again before submitting the previous batch fails with INVALID_ARGUMENT. */
N4M_API n4m_status_t n4m_explain_session_next_batch_v1(n4m_context_t* ctx,
                                                       n4m_explain_session_t* session,
                                                       double* out_rows,
                                                       int64_t capacity_rows,
                                                       int64_t* out_rows_written);

/* predictions: row-major (rows x n_outputs), rows equal to the last batch.
 * Nonfinite predictions fail with NUMERICAL_FAILURE and poison the session. */
N4M_API n4m_status_t n4m_explain_session_submit_v1(n4m_context_t* ctx,
                                                   n4m_explain_session_t* session,
                                                   const double* predictions,
                                                   int64_t rows);

/* Available once the plan is complete. std_error (nullable) receives the
 * Monte-Carlo standard error of each attribution for PERMUTATION (NaN when a
 * single antithetic pair was drawn) and NaN for every other method. */
N4M_API n4m_status_t n4m_explain_session_result_v1(n4m_context_t* ctx,
                                                   const n4m_explain_session_t* session,
                                                   double* attributions,
                                                   int64_t attributions_size,
                                                   double* expected,
                                                   int64_t expected_size,
                                                   double* predictions,
                                                   int64_t predictions_size,
                                                   double* std_error,
                                                   int64_t std_error_size);

/* ABI 2.24. GRADIENT sessions only, before the first next_batch: relative
 * step of the central differences (finite and > 0; default 1e-3). */
N4M_API n4m_status_t n4m_explain_session_set_gradient_step_v1(n4m_context_t* ctx,
                                                              n4m_explain_session_t* session,
                                                              double relative_step);

/* ABI 2.24. GRADIENT sessions only, once the plan is complete. Every output
 * is nullable: gradients [row][feature][output] (n_rows * n_features *
 * n_outputs), mean_gradient [feature][output] (mean over the explained
 * rows, n_features * n_outputs) and steps (the absolute step h_j of each
 * feature, n_features). */
N4M_API n4m_status_t n4m_explain_session_gradients_v1(n4m_context_t* ctx,
                                                      const n4m_explain_session_t* session,
                                                      double* gradients,
                                                      int64_t gradients_size,
                                                      double* mean_gradient,
                                                      int64_t mean_gradient_size,
                                                      double* steps,
                                                      int64_t steps_size);

/* ABI 2.24. The rows a session sends to the model for one coalition: for
 * every row of X, the features of the n_absent groups listed in
 * absent_groups (distinct, in [0, n_groups)) are filled by the baseline
 * with the session's own filling code, the other features keep X. Inputs
 * are validated as by n4m_explain_session_create_v1 (background required
 * for BACKGROUND and MEAN, contiguous groups for INTERPOLATE). Output,
 * row-major n_features columns: BACKGROUND writes n_background rows per
 * row of X (row i, background row b at i * n_background + b); MEAN and
 * INTERPOLATE one row per row of X. *out_n_rows receives the row count;
 * out_rows may be NULL (with capacity 0) to query it. */
N4M_API n4m_status_t n4m_explain_perturb_rows_v1(n4m_context_t* ctx,
                                                 const n4m_matrix_view_t* X,
                                                 const n4m_matrix_view_t* background,
                                                 const int64_t* feature_group,
                                                 int64_t n_groups,
                                                 int32_t baseline,
                                                 const int64_t* absent_groups,
                                                 int64_t n_absent,
                                                 double* out_rows,
                                                 int64_t out_rows_capacity,
                                                 int64_t* out_n_rows);

/* ---------------------------------------------------------------------------
 * 2. Model-specific exact explainers.
 * --------------------------------------------------------------------------- */

/* Exact interventional SHAP of an affine model f(z) = z . B + c evaluated in
 * the feature space of B (coefficients: n_features x n_outputs, F64 view).
 * reference holds the background column means (n_features). For group g:
 *   attribution[g][o] = sum_{j in g} B[j][o] * (x[j] - reference[j])
 *   expected[o]       = reference . B[:,o] + intercept[o]   (same for every row)
 * intercept may be NULL (zero). Group sums are exact Shapley values of the
 * group game because the model is additive. predictions (nullable) receives
 * f(x). expected is written per row ([row][output]). */
N4M_API n4m_status_t n4m_explain_linear_v1(n4m_context_t* ctx,
                                           const n4m_matrix_view_t* X,
                                           const n4m_matrix_view_t* coefficients,
                                           const double* intercept,
                                           const double* reference,
                                           const int64_t* feature_group,
                                           int64_t n_groups,
                                           double* attributions,
                                           int64_t attributions_size,
                                           double* expected,
                                           int64_t expected_size,
                                           double* predictions,
                                           int64_t predictions_size);

/* Path-dependent TreeSHAP (Lundberg et al. 2020, Algorithm 2) over a generic
 * flat tree ensemble. Nodes of all trees are concatenated; tree t owns nodes
 * [tree_offsets[t], tree_offsets[t+1]) and child indices are LOCAL to the
 * tree (-1 for none). For node k:
 *   feature[k]    split feature (-1 for a leaf),
 *   threshold[k]  split threshold,
 *   left[k]/right[k] local child indices (-1 for a leaf),
 *   default_left[k] (nullable array) 1 when a NaN feature goes left
 *                 (NULL: NaN input is rejected),
 *   cover[k]      training weight reaching the node (> 0 for reachable nodes),
 *   value[k*n_outputs + o] leaf value (internal node values ignored).
 * decision_type 0: x <= threshold goes left (scikit-learn, LightGBM);
 *               1: x <  threshold goes left (XGBoost).
 * The ensemble output is base_offset[o] + tree_scale * sum_t leaf_t[o]
 * (tree_scale = 1/n_trees for averaging forests, 1 for boosting;
 * base_offset nullable = 0). Attributions are per feature and, when
 * feature_group is non-NULL, summed within each group: grouped tree values are
 * SUMS of per-feature Shapley values, not Shapley values of the group game.
 * expected[row][o] = base_offset[o] + tree_scale * sum_t cover-weighted mean
 * leaf value of tree t. predictions (nullable) receives the ensemble output. */
N4M_API n4m_status_t n4m_explain_tree_path_dependent_v1(n4m_context_t* ctx,
                                                        const n4m_matrix_view_t* X,
                                                        int64_t n_trees,
                                                        const int64_t* tree_offsets,
                                                        const int64_t* feature,
                                                        const double* threshold,
                                                        const int64_t* left,
                                                        const int64_t* right,
                                                        const uint8_t* default_left,
                                                        const double* cover,
                                                        const double* value,
                                                        int64_t n_outputs,
                                                        int32_t decision_type,
                                                        const double* base_offset,
                                                        double tree_scale,
                                                        const int64_t* feature_group,
                                                        int64_t n_groups,
                                                        double* attributions,
                                                        int64_t attributions_size,
                                                        double* expected,
                                                        int64_t expected_size,
                                                        double* predictions,
                                                        int64_t predictions_size);

/* ---------------------------------------------------------------------------
 * 3. Grouping and summary helpers.
 * --------------------------------------------------------------------------- */

/* Contiguous fixed-width partition: feature j -> group j / width. The last
 * group may be shorter. out_n_groups receives ceil(n_features / width). */
N4M_API n4m_status_t n4m_explain_groups_fixed_width_v1(int64_t n_features,
                                                       int64_t width,
                                                       int64_t* feature_group,
                                                       int64_t feature_group_size,
                                                       int64_t* out_n_groups);

/* Partition from n_ranges half-open feature-index ranges [start, end)
 * (ranges: 2*n_ranges int64, e.g. chemically informed bands already mapped to
 * feature indices by the host). Ranges must be nonempty, inside
 * [0, n_features) and pairwise disjoint (any order). Range r becomes group r;
 * each maximal run of uncovered features becomes one additional group,
 * numbered after the ranges in feature order. range_of_group (nullable,
 * capacity n_ranges + n_features) receives, per group, the source range index
 * or -1 for uncovered runs. */
N4M_API n4m_status_t n4m_explain_groups_from_ranges_v1(int64_t n_features,
                                                       const int64_t* ranges,
                                                       int64_t n_ranges,
                                                       int64_t* feature_group,
                                                       int64_t feature_group_size,
                                                       int64_t* range_of_group,
                                                       int64_t range_of_group_size,
                                                       int64_t* out_n_groups);

/* Per-group summaries over rows for one output: mean |attribution| and mean
 * signed attribution (each n_groups doubles). */
N4M_API n4m_status_t n4m_explain_summarize_v1(const double* attributions,
                                              int64_t n_rows,
                                              int64_t n_groups,
                                              int64_t n_outputs,
                                              int64_t output_index,
                                              double* mean_abs,
                                              double* mean_signed,
                                              int64_t summary_size);

/* ABI 2.24. Monte-Carlo standard error of the row-mean signed attribution
 * of one output, per group: sqrt(sum_r std_error[r][g][o]^2) / n_rows (rows
 * are independent draws). NaN propagates (a single antithetic pair). */
N4M_API n4m_status_t n4m_explain_summarize_std_error_v1(const double* std_error,
                                                        int64_t n_rows,
                                                        int64_t n_groups,
                                                        int64_t n_outputs,
                                                        int64_t output_index,
                                                        double* mean_signed_std_error,
                                                        int64_t summary_size);

/* ---------------------------------------------------------------------------
 * 4. Importance profiles (ABI 2.24): aggregation and comparison.
 * --------------------------------------------------------------------------- */

typedef enum n4m_profile_aggregation_t {
    N4M_PROFILE_SUM = 0,      /* sum of the group's values */
    N4M_PROFILE_MEAN = 1,     /* mean of the group's values */
    N4M_PROFILE_SUM_ABS = 2,  /* sum of |value| */
    N4M_PROFILE_MEAN_ABS = 3, /* mean of |value| */
    N4M_PROFILE_MAX_ABS = 4   /* max of |value| */
} n4m_profile_aggregation_t;

/* Aggregates a per-feature profile (n_features values, not NaN; +-inf is
 * kept, e.g. an infinite selectivity ratio) over a feature partition
 * (feature_group as in section 1; NULL = one group per feature) into
 * n_groups values (out_size >= n_groups). */
N4M_API n4m_status_t n4m_explain_aggregate_profile_v1(n4m_context_t* ctx,
                                                      const double* values,
                                                      int64_t n_features,
                                                      const int64_t* feature_group,
                                                      int64_t n_groups,
                                                      int32_t rule,
                                                      double* out,
                                                      int64_t out_size);

/* Pairwise agreement of m profiles over the same G entries (profiles: F64
 * view m x G, m >= 1, G >= 2, no NaN; +-inf allowed). spearman (m x m,
 * row-major, nullable): Spearman correlation of average ranks, computed by
 * the Spearman correlation primitive of metrics.h (equals
 * scipy.stats.spearmanr; NaN when a
 * profile is constant). top_k_jaccard (m x m, nullable): |A_i n A_k| /
 * |A_i u A_k| with A_i the K = min(top_k, G) entries of largest |value| of
 * profile i (lowest index on ties); top_k >= 1. */
N4M_API n4m_status_t n4m_explain_compare_profiles_v1(n4m_context_t* ctx,
                                                     const n4m_matrix_view_t* profiles,
                                                     int64_t top_k,
                                                     double* spearman,
                                                     int64_t spearman_size,
                                                     double* top_k_jaccard,
                                                     int64_t jaccard_size);

#ifdef __cplusplus
}
#endif
#endif
