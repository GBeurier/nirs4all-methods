/* SPDX-License-Identifier: CECILL-2.1 */
/* cpp/include/n4m/metrics.h — metrics role header (ABI 2.0). */
#ifndef N4M_METRICS_H
#define N4M_METRICS_H
#include "n4m/n4m.h"   /* shared infra: status, matrix view, context, N4M_API */

#ifdef __cplusplus
extern "C" {
#endif

/* Supported numerical domain of the Preflight kernels (lead rules 13/15):
 * every nonzero |x| in [2^-N4M_DOMAIN_MAX_MAGNITUDE_LOG2,
 * 2^N4M_DOMAIN_MAX_MAGNITUDE_LOG2] and, wherever a statistic depends on
 * differences between input values, per row max/min nonzero |x| <=
 * 2^N4M_DOMAIN_MAX_SPREAD_LOG2. Inputs outside are refused before computing
 * with N4M_ERR_UNSUPPORTED ("input outside the supported numerical domain").
 * Exempt: summaries (magnitude bound only) and integer labels/codes. */
#define N4M_DOMAIN_MAX_MAGNITUDE_LOG2 256
#define N4M_DOMAIN_MAX_SPREAD_LOG2 32

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

/* HDBSCAN* on the rows of X with the Euclidean metric, ported from
 * scikit-learn 1.9.1 sklearn.cluster.HDBSCAN (alpha=1,
 * cluster_selection_epsilon=0, max_cluster_size=None) and, for GLOSH outlier
 * scores and cluster persistence, scikit-learn-contrib/hdbscan 0.8.44.
 * min_samples counts the row itself (scikit-learn convention; the contrib
 * hdbscan value is min_samples-1). Requires X.rows>=2, X.cols>=1, finite
 * values, min_cluster_size>=2, 1<=min_samples<=X.rows, selection_method
 * 0=EOM or 1=leaf, allow_single_cluster 0 or 1. Workloads with
 * rows*rows*cols above 2.5e9 return N4M_ERR_UNSUPPORTED.
 * Numerical domain (rules 13/15, N4M_DOMAIN_* above): finite X with every
 * nonzero |x| in [2^-256, 2^256] and max/min nonzero |x| <= 2^32; outside
 * it the call returns N4M_ERR_UNSUPPORTED ("input outside the supported numerical
 * domain ...") before computing (non-finite X stays N4M_ERR_INVALID_ARGUMENT).
 * Inside it every nonzero coordinate difference lies in [2^-308, 2^257], so
 * squared distances, densities (lambda = 1/distance <= 2^308) and stability
 * sums are normal doubles and the scikit-learn operation order is kept:
 * results equal the reference, and scaling X by 2^k scales lambdas and
 * stabilities by exactly 2^-k. Memory: 8*rows*cols + 512*rows bytes at most
 * 256 MiB. Documented non-finite values: lambda +inf at distance 0,
 * stability +inf for a node reaching distance 0 (inf - inf counts as 0
 * excess, where scikit-learn would give NaN), GLOSH NaN where an infinite-
 * density subtree makes it undefined.
 * Tie rule: MST edges of equal mutual-reachability weight are processed in
 * Prim insertion order (stable sort); Prim keeps the lowest row index on
 * equal weights. EOM replaces a node by its children only when their summed
 * stability is strictly larger.
 * Per-row outputs (X.rows elements): labels (-1 = noise, clusters numbered
 * 0..n_clusters-1 by increasing condensed-tree node index); membership =
 * scikit-learn probabilities_ (a relative membership strength, NOT a
 * probability; 0 for noise); outlier_scores = GLOSH in [0,1], NaN only when
 * an infinite-density subtree makes it undefined.
 * Cluster and tree outputs are caller-owned arrays of X.rows capacity (the
 * condensed tree never exceeds X.rows-1 nodes). cluster_persistence[c] is the
 * contrib cluster_persistence_ (stability / (members * max lambda), 1 when
 * the max lambda is 0 or infinite). The condensed cluster tree has
 * *n_tree_nodes nodes, node 0 is the root: tree_parent (-1 for the root),
 * tree_size (points at birth), tree_label (selected label or -1),
 * tree_lambda_birth (1/distance, root 0), tree_lambda_death (largest lambda
 * at which a point or child leaves the node) and tree_stability (excess of
 * mass before selection). Lambda is +inf at distance 0. */
N4M_API n4m_status_t n4m_cluster_hdbscan(
    n4m_context_t* ctx, const n4m_matrix_view_t* X, int64_t min_cluster_size,
    int64_t min_samples, int32_t selection_method, int32_t allow_single_cluster,
    int64_t* labels, double* membership, double* outlier_scores, int64_t* n_clusters,
    double* cluster_persistence, int64_t* n_tree_nodes, int64_t* tree_parent,
    int64_t* tree_size, int64_t* tree_label, double* tree_lambda_birth,
    double* tree_lambda_death, double* tree_stability);

/* Adjusted Rand index (Hubert-Arabie) between two label vectors of n
 * elements, computed as scikit-learn adjusted_rand_score from the pair
 * confusion counts. Every label value, including -1, is an ordinary class:
 * callers filter noise first. *ari is NaN when n<2 or when both partitions
 * are the same trivial partition (one class or all singletons), where
 * scikit-learn returns 1.0 by convention. Pair counts and the numerator and
 * denominator are exact integers (128-bit products, ABI 2.20), so values near
 * 0 do not come from cancelled rounded products; only the final division
 * rounds. Invariant to the label values. Integer inputs: the rule-13
 * numerical domain does not apply; memory 32*n bytes at most 256 MiB
 * (n <= 2^23), else N4M_ERR_UNSUPPORTED. */
N4M_API n4m_status_t n4m_preflight_adjusted_rand_index(
    n4m_context_t* ctx, const int64_t* labels_a, const int64_t* labels_b, int64_t n,
    double* ari);

/* Cluster overlap between two partitions of the same n elements (ABI 2.20).
 * labels_a[i] in [0, n_clusters_a) or negative (unassigned, e.g. HDBSCAN
 * noise), likewise labels_b with n_clusters_b; unassigned elements belong to
 * no cluster. Caller-owned row-major outputs: intersections[i*n_clusters_b+j]
 * = |A_i ∩ B_j| and jaccard[i*n_clusters_b+j] = |A_i ∩ B_j| / |A_i ∪ B_j|
 * (NaN when both clusters are empty); sizes_a[i] = |A_i|, sizes_b[j] = |B_j|.
 * n >= 0, n_clusters_a and n_clusters_b >= 0, and
 * n_clusters_a * n_clusters_b <= 100 million cells. Counts are exact integers
 * and each Jaccard value is one correctly rounded division; n > 2^53 returns
 * N4M_ERR_UNSUPPORTED. Integer inputs: the rule-13 numerical domain does not
 * apply, and no working memory is allocated. */
N4M_API n4m_status_t n4m_cluster_jaccard_matrix(
    n4m_context_t* ctx, const int64_t* labels_a, const int64_t* labels_b, int64_t n,
    int64_t n_clusters_a, int64_t n_clusters_b, double* jaccard, int64_t* intersections,
    int64_t* sizes_a, int64_t* sizes_b);

/* Rectangular linear sum assignment (ABI 2.20), a port of SciPy 1.18.1
 * scipy.optimize.linear_sum_assignment (Crouse's shortest augmenting path,
 * BSD-3-Clause). Minimizes (maximize=0) or maximizes (maximize=1) the summed
 * cost of a one-to-one assignment of min(rows, cols) row/column pairs. The
 * cost view is any strided f64 matrix; +inf forbids a pair, NaN and -inf
 * (after negation when maximizing) return N4M_ERR_INVALID_ARGUMENT, and an
 * assignment without forbidden pairs that does not exist returns
 * N4M_ERR_NUMERICAL_FAILURE. Ties follow SciPy exactly (columns scanned from
 * last to first, an unassigned column wins equal reduced costs; a constant
 * matrix gives the identity). Caller-owned row_ind/col_ind receive
 * min(rows, cols) pairs in increasing row order; *n_assigned their count.
 * Numerical domain (rules 13/15, N4M_DOMAIN_* above): finite costs with every nonzero |cost| in
 * [2^-256, 2^256] and max/min nonzero |cost| <= 2^32 (+inf forbidden pairs
 * excluded); outside it N4M_ERR_UNSUPPORTED is returned before computing.
 * Inside it every cost is an integer multiple of 2^E (E = last mantissa bit
 * of the smallest nonzero |cost|) below 2^87 units, and duals, reduced and
 * path costs are kept as exact signed 192-bit integers in those units: the
 * result is an exact optimum of the given costs, including where SciPy's
 * double arithmetic is suboptimal; ties follow SciPy's order on exact values.
 * Work: rows*cols*min(rows, cols) <= 1e8 exact operations; memory:
 * 16*rows*cols + 128*(rows+cols) bytes at most 256 MiB, checked before any
 * allocation. */
N4M_API n4m_status_t n4m_linear_sum_assignment(
    n4m_context_t* ctx, const n4m_matrix_view_t* cost, int32_t maximize, int64_t* row_ind,
    int64_t* col_ind, int64_t* n_assigned);

/* n_permutations successive uniform permutations of [0, n) (ABI 2.20), equal
 * to rng = numpy.random.default_rng(seed); [rng.permutation(n) for _ in
 * range(n_permutations)]: one Methods PCG64 stream (NumPy SeedSequence
 * expansion) and NumPy's Fisher-Yates over random_interval. out is caller-owned
 * row-major n_permutations x n. n >= 0, n_permutations >= 0 and
 * n * n_permutations <= 250 million. */
N4M_API n4m_status_t n4m_random_permutations(
    n4m_context_t* ctx, uint64_t seed, int64_t n, int64_t n_permutations, int64_t* out);

/* n draws of a d-variate normal distribution (ABI 2.20), equal to
 * numpy.random.default_rng(seed).multivariate_normal(mean, cov, size=n,
 * method="cholesky"): standard normals from the Methods PCG64 Ziggurat stream
 * (NumPy-identical) fill z row by row, L is the lower Cholesky factor of cov
 * (only the lower triangle of cov is read, as LAPACK potrf "L"), and
 * out[i*d+j] = sum_k z[i*d+k] L[j][k] + mean[j]. cov is a d x d strided f64
 * view (d >= 1), mean has d finite values, out is caller-owned row-major
 * n x d (n >= 0). A covariance that is not numerically positive definite
 * returns N4M_ERR_NUMERICAL_FAILURE; regularization is the caller's choice.
 * Limits (N4M_ERR_UNSUPPORTED, checked before any allocation, also for n = 0):
 * 8*(d*d + n*d) bytes at most 256 MiB, d^3/6 and n*d*d at most 2.5e9.
 * Numerical domain (rules 13/15, N4M_DOMAIN_* above): mean and the lower
 * triangle of cov each finite with every nonzero |value| in [2^-256, 2^256]
 * and max/min nonzero |value| <= 2^32; outside it N4M_ERR_UNSUPPORTED before computing. Inside it the
 * factor (computed on cov / 4^m with 2^m near the largest standard deviation,
 * an exact rescaling) and the draws stay normal, so scaling cov by 4^k scales
 * the draws by exactly 2^k. */
N4M_API n4m_status_t n4m_random_multivariate_normal(
    n4m_context_t* ctx, uint64_t seed, const double* mean, const n4m_matrix_view_t* cov,
    int64_t n, double* out);

/* Smoothing-residual noise proxy along the feature axis. Features are assumed
 * equally spaced. Each row is smoothed by the Methods Savitzky-Golay filter
 * (deriv 0, odd window_length >= 3, 0 <= polyorder <= window_length-2).
 * A residual r_ij = x_ij - smooth_ij exists only for interior features
 * (h <= j < cols-h, h = window_length/2) whose whole window is valid in row
 * i: edges are never padded. g = sqrt(sum_k (delta_k - w_k)^2) is the
 * residual gain for unit white noise, w the smoothing weights. Outputs:
 * band_noise[j] = sqrt(mean_i r_ij^2) / g, NaN when no row contributes;
 * band_counts[j] = contributing rows; out_two[0] the same root mean square
 * pooled over every evaluable residual cell (NaN if none) divided by g;
 * out_two[1] = g. valid_mask is optional row-major rows x cols bytes, nonzero
 * = valid; unmasked X must be finite and cols >= window_length. The value is
 * a noise proxy under a locally polynomial signal, never an SNR. */
N4M_API n4m_status_t n4m_preflight_noise_proxy_compute(
    n4m_context_t* ctx, const n4m_matrix_view_t* X, const uint8_t* valid_mask,
    int32_t window_length, int32_t polyorder, double* band_noise,
    int64_t* band_counts, double* out_two);

/* Row pairs whose root-mean-square feature difference
 * sqrt(sum_j (q_ij - r_hj)^2 / cols) is <= tolerance (finite, > 0).
 * reference == NULL compares query rows with each other (pairs i < h, using
 * query_groups for both rows; reference_groups and reference_flags ignored).
 * Otherwise every query row is compared with every reference row, and both
 * group arrays are NULL or both non-NULL. Pairs whose rows share a group ID
 * are excluded from candidates. Up to max_pairs pairs are written in
 * ascending (query, reference) order to pair_indices (row-major max_pairs x
 * 2) and pair_distances; both may be NULL only when max_pairs == 0.
 * query_flags (query rows) and, for two matrices, reference_flags (reference
 * rows) are set to 1 for rows in at least one pair, else 0. out_counts
 * receives [total pairs within tolerance, candidate pairs compared]. All
 * values must be finite. */
N4M_API n4m_status_t n4m_preflight_pairs_within_tolerance(
    n4m_context_t* ctx, const n4m_matrix_view_t* query,
    const n4m_matrix_view_t* reference, const int64_t* query_groups,
    const int64_t* reference_groups, double tolerance, int64_t max_pairs,
    int64_t* pair_indices, double* pair_distances, uint8_t* query_flags,
    uint8_t* reference_flags, int64_t* out_counts);

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
