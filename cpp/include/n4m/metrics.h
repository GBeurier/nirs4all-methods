/* SPDX-License-Identifier: CECILL-2.1 */
/* cpp/include/n4m/metrics.h — metrics role header (ABI 2.0). */
#ifndef N4M_METRICS_H
#define N4M_METRICS_H
#include "n4m/n4m.h"   /* shared infra: status, matrix view, context, N4M_API */

#ifdef __cplusplus
extern "C" {
#endif

/* Supported numerical domain D of the Preflight and statistics kernels
 * (lead rules 13 and 15): finite values, every nonzero |x| in
 * [2^-N4M_DOMAIN_MAX_MAGNITUDE_LOG2, 2^N4M_DOMAIN_MAX_MAGNITUDE_LOG2] and,
 * wherever a statistic depends on differences between input values, max
 * nonzero |x| / min nonzero |x| <= 2^N4M_DOMAIN_MAX_SPREAD_LOG2 inside the
 * array (or the slice, e.g. one feature) whose values are differenced. With a
 * spread below 2^53, subtracting two values of one input never erases their
 * difference. Exact zeros are always allowed. Kernels refuse other inputs
 * before computing with N4M_ERR_UNSUPPORTED and the message "input outside
 * the supported numerical domain (...)". Exempt from the spread limit:
 * summaries and nearest neighbours (magnitude bounds only), Holm/BH p-values
 * (nonzero p >= 2^-1022) and integer labels/codes (not checked). PCA declares
 * its own magnitude bounds (see n4m_preflight_pca_compute_resolved). */
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
 * For no values, all fields except count=0 are NaN; SD is NaN for n<2.
 * Supported domain: every nonzero unmasked |x| in [2^-256, 2^256], with no
 * spread bound; otherwise N4M_ERR_UNSUPPORTED ("input outside the supported
 * numerical domain") before computing. Inside it the mean
 * is a double-double from an unevaluated Neumaier pair (error <= (n u)^2
 * mean|x| + u |m|), the SD a compensated sum of squared deviations from it,
 * and each quartile a + w (b - a) with b - a kept exactly (within 2 ulp). */
N4M_API n4m_status_t n4m_preflight_summary_compute(
    n4m_context_t* ctx, const double* values, const uint8_t* valid_mask,
    int64_t n, double* out_eight);

/* Euclidean k-nearest reference rows for each query row, sorted by ascending
 * distance then reference index. When BOTH group arrays are non-NULL, rows
 * with the same group ID are excluded. Outputs are row-major n_query x k,
 * using index=-1 and distance=NaN when too few eligible neighbors remain.
 * All input values must be finite. k>0; callers own every buffer.
 * Supported domain: every nonzero |x| in [2^-256, 2^256], with no spread
 * bound; otherwise N4M_ERR_UNSUPPORTED ("input outside the supported
 * numerical domain") before computing. Inside it distances use a compensated sum (relative error < 4u); two candidates
 * within 2^-45 (relative) of each other are ordered exactly from
 * ||q - a||^2 - ||q - b||^2 in an exact accumulator, ties by index. */
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
 * and coverage still reported. Continuous targets must lie in the supported
 * domain D (every nonzero |y| in [2^-256, 2^256], max/min nonzero |y| <=
 * 2^32), else N4M_ERR_UNSUPPORTED; category codes are labels and are not
 * bounded by D. Inside D the variance uses a double-double mean and
 * compensated sums; out_four[0] = 1 - ratio carries an absolute error of a
 * few u times the ratio. */
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
 * (deriv 0, odd 3 <= window_length <= 1001, 0 <= polyorder <=
 * min(10, window_length-2): the domain where the weights were verified to
 * reproduce polynomials; other values give N4M_ERR_INVALID_ARGUMENT). The
 * weights must satisfy sum_k w_k (k/h)^m = [m == 0] within 1e-9 for every
 * m <= polyorder, else N4M_ERR_NUMERICAL_FAILURE.
 * A residual r_ij = x_ij - smooth_ij exists only for interior features
 * (h <= j < cols-h, h = window_length/2) whose whole window is valid in row
 * i: edges are never padded. g = sqrt(sum_k (delta_k - w_k)^2) is the
 * residual gain for unit white noise, w the smoothing weights. Outputs:
 * band_noise[j] = sqrt(mean_i r_ij^2) / g, NaN when no row contributes;
 * band_counts[j] = contributing rows; out_two[0] the same root mean square
 * pooled over every evaluable residual cell (NaN if none) divided by g;
 * out_two[1] = g. valid_mask is optional row-major rows x cols bytes, nonzero
 * = valid; unmasked X must be finite and cols >= window_length. The value is
 * a noise proxy under a locally polynomial signal, never an SNR.
 * Supported numerical domain D: every nonzero valid |x| in [2^-256, 2^256]
 * and, per row, max/min nonzero valid |x| <= 2^32; otherwise
 * N4M_ERR_UNSUPPORTED ("input outside the supported numerical domain")
 * before computing. Inside D the residual of each evaluable window is
 * computed from that window alone as sum_{k>0} w_k (2 x_j - x_{j+k} -
 * x_{j-k}) with double-double Savitzky-Golay weights (Gram polynomials,
 * cross-checked against the SG engine) and exact second differences, rounded
 * once. Values outside every evaluable window never change the results, and
 * exact offsets cancel exactly. Precision (honest, conservative): each
 * residual has absolute error <= 2 W eps sum_k |c_k| max_k |w_k| against the
 * exact SG residual (W window length, eps = 2^-52, coefficient errors and the
 * final rounding included); squares, sums, the square root and the gain add
 * a relative error of a few eps. These are rounding-level discrepancies. */
N4M_API n4m_status_t n4m_preflight_noise_proxy_compute(
    n4m_context_t* ctx, const n4m_matrix_view_t* X, const uint8_t* valid_mask,
    int32_t window_length, int32_t polyorder, double* band_noise,
    int64_t* band_counts, double* out_two);

/* SNV (centre and scale by the population SD, ddof 0) of each row of X, with
 * the scheme of n4m_transform_snv_* (constant rows map to exact zeros;
 * power-of-two rescaling, compensated sums and a double-double mean, so no
 * finite row overflows or underflows and large offsets do not leak), then
 * optionally aggregated. group_ids == NULL: n_groups == X->rows and out
 * receives the SNV rows. Otherwise group_ids[i] in [0, n_groups) and every
 * group is non-empty: out row g is the equal-weight mean of the SNV rows of
 * group g (compensated per band). out is n_groups x cols, caller-owned.
 * error_bounds[g] (n_groups doubles) bounds the absolute rounding error of
 * every entry of out row g: for one SNV row,
 * B = (3uA + n^2 u^2 mean|y| + 2u^2 |m|) / sd_y + 8 u max|z| with u = 2^-53, y the row
 * rescaled by 2^-ilogb(max|x|), m, sd_y and A = mean|y - m| its mean, SD and
 * mean absolute deviation, n = cols, z the SNV row (0 for a constant row);
 * for a group of r > 1 rows, max member B + (4u + 2 r u^2) max member |z|.
 * X must be finite with cols >= 2. Supported domain D: every nonzero |x| in
 * [2^-256, 2^256] and, per row, max/min nonzero |x| <= 2^32;
 * otherwise N4M_ERR_UNSUPPORTED before computing. Added in ABI 2.20. */
N4M_API n4m_status_t n4m_preflight_snv_compute(
    n4m_context_t* ctx, const n4m_matrix_view_t* X, const int64_t* group_ids,
    int64_t n_groups, const n4m_matrix_view_t* out, double* error_bounds);

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
 * values must be finite. Supported domain D: every nonzero |x| in
 * [2^-256, 2^256] and, per row, max/min nonzero |x| <= 2^32; otherwise
 * N4M_ERR_UNSUPPORTED before computing. Inside D the decisions are exact. A fast estimate on power-of-two rescaled differences
 * (exact halves when some q - r overflows) decides when it is farther than
 * 2^-45 (relative) from the tolerance; otherwise the sign of
 * sum_j (q_j - r_j)^2 - cols tol^2 is computed exactly in a fixed-point
 * accumulator of exact products. An accepted pair publishes
 * min(estimated RMS, tolerance). */
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
 * permutation runs when the effect size is undefined. Targets are centered
 * after an exact power-of-two scaling and a shift to the first target, with
 * a residual-mean correction, so eta^2 and F are translation invariant.
 * categorical=1 with levels * classes > 10 million cells returns
 * N4M_ERR_UNSUPPORTED before any table allocation or statistic.
 * Supported numerical domain D: continuous targets in D (nonzero |y| in
 * [2^-256, 2^256], max/min nonzero |y| <= 2^N4M_DOMAIN_MAX_SPREAD_LOG2,
 * exact zeros allowed; categorical class codes are labels and are not
 * checked). Inputs outside D return N4M_ERR_UNSUPPORTED before any
 * computation. Inside D the results are scale invariant (exact power-of-two
 * rescaling) and translation invariant up to the rounding of the shifted
 * inputs, and F cannot overflow (F <= n * 2^168).
 * Accuracy, stated conservatively: eta^2 and V have an absolute error of at
 * most about c * n * DBL_EPSILON (c a small constant); eta^2 = 0 and V = 0
 * are returned exactly for equal level means on dyadic data and at exact
 * independence. V^2 = chi^2 / (n (min(levels, classes) - 1)), with chi^2
 * summed from exact integer residuals O*n - r*c (nonnegative terms;
 * n <= 3e9 for categorical targets). SSW is summed from per-level residuals
 * of the original targets (each level scaled and centered on its own), so it
 * has a relative error of about c * n * DBL_EPSILON. F = eta^2 SST/(levels-1)
 * / (SSW/(n-levels)) therefore has a relative error of about c * n *
 * DBL_EPSILON when eta^2 is not small, and otherwise an absolute error of
 * about c * n * DBL_EPSILON * (SST/SSW) * (n-levels)/(levels-1). F is NaN
 * only when every level is constant or n == levels; an F that is not a
 * normal double while eta^2 > 0 returns N4M_ERR_NUMERICAL_FAILURE. */
N4M_API n4m_status_t n4m_preflight_factor_association(
    n4m_context_t* ctx, const int64_t* factor_codes, const double* targets,
    int64_t n, int32_t categorical, int32_t n_permutations, uint64_t seed,
    double* out_eight, int64_t* out_two);

/* Holm step-down adjusted p-values for one family of m >= 1 hypotheses.
 * Inputs must lie in [0, 1]; adjusted[i] = min(1, max_{j<=rank(i)}
 * (m-j+1) p_(j)) with a stable ascending order, as R p.adjust("holm").
 * No spread limit (no subtraction is performed): nonzero p-values must be
 * normal, p >= 2^-1022 (DBL_MIN); subnormal p-values return
 * N4M_ERR_UNSUPPORTED ("input outside the supported numerical domain
 * (nonzero p-values >= 2^-1022)"). */
N4M_API n4m_status_t n4m_preflight_holm_adjust(
    n4m_context_t* ctx, const double* p_values, int64_t m, double* adjusted);

/* Pearson (method 0) or Spearman (method 1) correlation between paired
 * finite vectors x and y over n exchangeable analysis units, with an
 * optional seeded two-sided Monte Carlo permutation test of y over all
 * units (a single unrestricted block). Spearman correlates average ranks
 * (ties share the mean of their ranks, as scipy.stats.rankdata "average"),
 * so it equals scipy.stats.spearmanr; Pearson equals scipy.stats.pearsonr.
 * n >= 2, 0 <= n_permutations <= 1000000 and n * (n_permutations + 1) <=
 * 250 million operations. Permutation draws use PCG64 seeded like
 * numpy.random.default_rng(seed); the b-th draw equals the b-th
 * rng.permutation(n), y_i -> y[perm[i]]. -0.0 and +0.0 are one value.
 * out_five (caller-owned, NaN when undefined):
 *   [0] correlation coefficient in [-1, 1]; NaN when x or y is constant.
 *   [1] number of distinct x values; [2] number of distinct y values.
 *   [3] log10 of min(n!/prod(x_multiplicity!), n!/prod(y_multiplicity!)),
 *       an upper bound on distinct permutation outcomes of the statistic.
 *   [4] two-sided Monte Carlo p-value (1+b)/(1+B); NaN when B == 0 or [0]
 *       is NaN.
 * b counts permuted |r| >= |observed r| - sqrt(DBL_EPSILON). out_two =
 * {b, executed B}; no permutation runs when the coefficient is undefined.
 * Both vectors must lie in the supported domain D (each checked
 * separately; outside D: N4M_ERR_UNSUPPORTED) and use the same scale- and
 * translation-stable centering as n4m_preflight_factor_association. The
 * absolute error of r is at most about c * n * DBL_EPSILON; exactly
 * uncorrelated dyadic data give r = 0. */
N4M_API n4m_status_t n4m_stats_correlation_test(
    n4m_context_t* ctx, const double* x, const double* y, int64_t n,
    int32_t method, int32_t n_permutations, uint64_t seed,
    double* out_five, int64_t* out_two);

/* Exact permutation distributions of the statistics above over n
 * exchangeable units. The side with fewer distinct arrangements
 * (n!/prod(multiplicity!)) is enumerated (std::next_permutation over its
 * multiset of codes) with the other side fixed; every distinct arrangement
 * has equal probability under uniform permutation. The statistic is eta^2 or
 * Cramer's V^2 (factor association, same input contract as
 * n4m_preflight_factor_association) or |r| (correlation, method 0 Pearson,
 * 1 Spearman). 1 <= max_arrangements <= 10000000.
 * Inputs follow the domain D of the Monte Carlo kernels (outside D:
 * N4M_ERR_UNSUPPORTED before any computation). A larger arrangement count,
 * or 2 * count * (n + levels * classes) > 250 million operations, returns
 * N4M_ERR_UNSUPPORTED before any statistic is evaluated (and, as above, a
 * categorical table above 10 million cells is refused before allocation).
 * out_four (caller-owned): [0] number N of distinct arrangements;
 * [1] exact p-value #{T >= T_obs - sqrt(DBL_EPSILON)}/N; [2] minimal
 * attainable exact p-value #{T >= T_max - sqrt(DBL_EPSILON)}/N; [3] T_max.
 * [1]-[3] are NaN when the statistic is undefined. */
N4M_API n4m_status_t n4m_stats_factor_association_exact(
    n4m_context_t* ctx, const int64_t* factor_codes, const double* targets,
    int64_t n, int32_t categorical, int64_t max_arrangements, double* out_four);
N4M_API n4m_status_t n4m_stats_correlation_exact(
    n4m_context_t* ctx, const double* x, const double* y, int64_t n,
    int32_t method, int64_t max_arrangements, double* out_four);

/* Benjamini-Hochberg step-up adjusted p-values for one family of m >= 1
 * hypotheses. Inputs must lie in [0, 1]; with p_(1) <= ... <= p_(m),
 * adjusted p_(i) = min(1, min_{j>=i} m p_(j) / j), as R p.adjust("BH")
 * (same domain as Holm: no spread limit, subnormal p-values refused;
 * positive p-values stay positive)
 * and statsmodels multipletests(method="fdr_bh"). Controls the false
 * discovery rate under independence or positive regression dependence. */
N4M_API n4m_status_t n4m_stats_bh_adjust(
    n4m_context_t* ctx, const double* p_values, int64_t m, double* adjusted);
/* PERMANOVA (Anderson 2001) on Euclidean distances between the rows of X
 * (n x k coordinates, e.g. PCA scores), for one categorical term given
 * optional strata. term_codes and strata_codes are arbitrary int64 level
 * codes of n elements; strata_codes == NULL is one stratum (the term alone).
 * Sums of squares follow the additive model strata + term: ss_strata is the
 * between-strata SS and ss_term = SS(strata + term) - SS(strata), the vegan
 * adonis2(by = "margin") SS of the term when the strata are the other factor.
 * The identifiable term df is levels - c, with c the connected components of
 * the bipartite level-stratum occupancy graph (0 when the term is aliased
 * with the strata); df_residual = n - strata - df_term.
 * Monte Carlo test (n_permutations > 0 and a defined observed pseudo-F):
 * PCG64 seeded like numpy.random.default_rng(seed). For each draw, every
 * stratum in ascending code order (members in ascending row order idx) takes
 * q = rng.permutation(n_s), and row idx[j] receives the data of row idx[q[j]]
 * (rows move only inside their stratum). As vegan anova.cca, b counts
 * pseudo-F_perm >= F_obs - sqrt(DBL_EPSILON) (+inf when the permuted residual
 * SS is zero); p = (1+b)/(1+B).
 * Numerical domain D (shared by n4m_stats_permanova_freedman_lane and
 * n4m_stats_permdisp): X finite, every nonzero |x| in [2^-256, 2^256] and
 * max/min nonzero |x| <= 2^32 over the whole matrix (exact zeros allowed),
 * the same D as the association kernels (N4M_DOMAIN_MAX_SPREAD_LOG2 = 32).
 * Outside D the call returns N4M_ERR_UNSUPPORTED, before any computation,
 * with the context message "input outside the supported numerical domain
 * (...)". Inside D each column is centred (exact power-of-two prescale, shift
 * to its first row, compensated mean) and all columns are expressed in one
 * power-of-two unit chosen from the CENTRED variation, so constant columns
 * and common offsets do not change any output and the rescaling is exact: no
 * varying column is erased and no squared term underflows. Centring rounds
 * each value by about n DBL_EPSILON times the largest |x| of its column,
 * which the spread bound keeps below about n 2^-20 times the smallest nonzero
 * |x|. Every sum of squares is a sum of squared terms (residuals are
 * projected explicitly, never obtained as a difference of totals)
 * accumulated in long double, and F and all ratios are formed in the scaled
 * unit; for a well-conditioned design the relative error of a sum of
 * squares is about c n DBL_EPSILON and its absolute error about c n
 * DBL_EPSILON times the raw sum of squares. A pseudo-F is NaN (undefined, no
 * test) only when its residual SS is exactly zero. Otherwise it is published
 * with a resolution ratio, the error estimate of the residual SS divided by
 * it: estimate = gamma^2 n sum_j (max_i y_ij - min_i y_ij)^2, the range of
 * each centred column y in the common scaled unit (same unit as the residual
 * SS, so the ratio is unit-free), gamma = 4 (rank + k + 2) n DBL_EPSILON
 * (rank = design rank, groups for PERMDISP), a conservative
 * first-order estimate, not a proof. A ratio >= 1 sets the "numerically
 * unresolved" flag: F and its test are still computed but the residual SS,
 * hence F, is not resolved above rounding; below 1 the relative error of the
 * residual SS is about 2 sqrt(ratio). Published sums
 * of squares (and PERMDISP distances) carry an exponent: the true value is
 * out * 2^(*ss_exponent); it is 0, with true values written, when every
 * positive value is a normal double (always the case for ss_total inside
 * D), otherwise the values are written in the scaled unit so that a positive
 * quantity is never published as 0.
 * N4M_ERR_NUMERICAL_FAILURE is returned if a positive value is not
 * representable even in the scaled unit.
 * Sizes, products and work budgets are checked before any allocation that
 * depends on the number of levels.
 * Requires n >= 2, k >= 1, finite X, at most 2048 term levels, 0 <=
 * n_permutations <= 1000000 and (2n(k+1) + 2 levels^2 k)(B+1) + levels n <=
 * 250 million operations (N4M_ERR_UNSUPPORTED otherwise).
 * out_fifteen (caller-owned, NaN when undefined):
 *   [0] ss_total  [1] ss_strata  [2] ss_term  [3] ss_residual
 *   [4] R^2 = ss_term/ss_total  [5] partial R^2 = ss_term/(ss_term+ss_residual)
 *   [6] pseudo-F = (ss_term/df_term)/(ss_residual/df_residual)
 *   [7] df_term  [8] df_residual  [9] term levels  [10] strata
 *   [11] log10 prod_s (n_s!/prod_a n_as!), the number of distinct
 *        within-strata level assignments (an upper bound on distinct outcomes)
 *   [12] Monte Carlo p-value
 *   [13] resolution ratio of the residual SS (+inf when it is 0)
 *   [14] 1 when the pseudo-F is numerically unresolved (ratio >= 1), else 0
 *   ([13], [14] NaN when df_term or df_residual is 0).
 * out_two = {b, executed B}; *ss_exponent: true SS = out[0..3] * 2^exponent. */
N4M_API n4m_status_t n4m_stats_permanova(
    n4m_context_t* ctx, const n4m_matrix_view_t* X, const int64_t* term_codes,
    const int64_t* strata_codes, int32_t n_permutations, uint64_t seed,
    double* out_fifteen, int64_t* out_two, int32_t* ss_exponent);

/* Partial PERMANOVA of one categorical term given the additive main effects
 * of n_conditions categorical factors, tested by Freedman-Lane permutation of
 * the residuals of the reduced model (Freedman & Lane 1983; Anderson 2001,
 * Can. J. Fish. Aquat. Sci. 58:626). Euclidean distances between the rows of
 * X (n x k). condition_codes is row-major n x n_conditions (NULL when 0); the
 * reduced design is the intercept plus the indicators of every condition
 * factor (no interaction), the full design adds the term indicators, and
 * ranks are numerical (Gram-Schmidt, relative tolerance 1e-9).
 * df_term = rank(full) - rank(reduced), 0 when the term is aliased with the
 * conditions; df_residual = n - rank(full). pseudo-F =
 * (SS_term/df_term)/(SS_residual/df_residual), SS_term = SS(full) -
 * SS(reduced). The reduced-model residuals are projected off the reduced
 * design twice; SS_term and SS_residual come from one projection of them on
 * the full design, so the partial R^2 is SS_term/(SS_term + SS_residual).
 * Monte Carlo test (n_permutations > 0 and a defined pseudo-F):
 * PCG64 as numpy.random.default_rng(seed); the b-th draw perm =
 * rng.permutation(n) gives data fitted_reduced[i] + residual[perm[i]], refitted
 * on both designs; b counts pseudo-F_perm >= F_obs - sqrt(DBL_EPSILON) (+inf
 * for an exactly zero residual SS), p = (1+b)/(1+B). Residual SS of the
 * observation and of every draw are explicit projections off the full
 * design. Numerical domain as n4m_stats_permanova. Limits, checked before
 * the design is generated: at most 4096 design columns c, n c <= 32 Mi
 * values (256 MiB) and 2 n c^2 + 4 n k (c+1)(B+1) <= 250 million
 * operations.
 * out_fourteen (NaN when undefined): [0] ss_total [1] ss_reduced (model SS of
 * the reduced design) [2] ss_term [3] ss_residual [4] ss_term/ss_total
 * [5] partial R^2 ss_term/(ss_term+ss_residual) [6] pseudo-F [7] df_term
 * [8] df_residual [9] rank of the reduced design (intercept included)
 * [10] log10(n!), the distinct residual permutations bound [11] p-value
 * [12] resolution ratio of the residual SS [13] numerically unresolved flag
 * (as n4m_stats_permanova [13], [14]).
 * out_two = {b, executed B}; *ss_exponent: true SS = out[0..3] * 2^exponent. */
N4M_API n4m_status_t n4m_stats_permanova_freedman_lane(
    n4m_context_t* ctx, const n4m_matrix_view_t* X, const int64_t* term_codes,
    const int64_t* condition_codes, int32_t n_conditions, int32_t n_permutations,
    uint64_t seed, double* out_fourteen, int64_t* out_two, int32_t* ss_exponent);

/* PERMDISP (Anderson 2006), as vegan betadisper(type = "centroid") with
 * permutest: Euclidean distances of the rows of X to their group centroid,
 * one-way ANOVA F on those distances, and a Monte Carlo test permuting the
 * ANOVA residuals of the distances over all rows (PCG64 as
 * numpy.random.default_rng(seed); row i receives residual perm[i] of the
 * b-th rng.permutation(n)), refits the one-way model and counts
 * F_perm >= F_obs - sqrt(DBL_EPSILON), as vegan permutest.betadisper (the
 * comparison is on F; a tolerance on a monotone transform of F is not
 * equivalent); the between and within SS of each draw are explicit sums of
 * squares around its own group means. Numerical domain as
 * n4m_stats_permanova (distances share one published exponent). Dispersion
 * differences do not validate exchangeability or a PERMANOVA. Same input
 * limits as n4m_stats_permanova; work n(2k + 3(B + 1)) <= 250 million
 * operations.
 * distances has n elements; group_mean_distances has capacity n and
 * receives one mean per group in ascending code order.
 * out_eight: [0] F (NaN with one group, n == groups or a zero within SS)
 * [1] df1 = groups-1 [2] df2 = n-groups [3] groups
 * [4] log10(n!/prod n_g!) [5] Monte Carlo p-value [6] resolution ratio of
 * the within SS of the distances [7] numerically unresolved flag (as
 * n4m_stats_permanova [13], [14]; NaN with one group or n == groups);
 * out_two = {b, executed B};
 * *distance_exponent: true distance = out * 2^exponent (both arrays). */
N4M_API n4m_status_t n4m_stats_permdisp(
    n4m_context_t* ctx, const n4m_matrix_view_t* X, const int64_t* group_codes,
    int32_t n_permutations, uint64_t seed, double* distances,
    double* group_mean_distances, double* out_eight, int64_t* out_two,
    int32_t* distance_exponent);
/* One-way random-effects variance components x_ij = mu + u_i + e_ij per
 * feature (column of X), by REML, for balanced or unbalanced groups.
 * group_ids has X.rows elements; valid_mask is optional row-major
 * X.rows x X.cols bytes, nonzero = valid; unmasked X values must be finite.
 * Every group with >= 1 valid value enters, singletons included.
 * Supported numerical domain (checked first, over every feature): each
 * nonzero |x| in [2^-256, 2^256] and max/min nonzero |x| <= 2^32
 * (N4M_DOMAIN_MAX_SPREAD_LOG2) over the nonzero magnitudes and over the
 * pairwise differences of each feature ((max - min) / smallest nonzero gap
 * between sorted values); outside it the call returns
 * N4M_ERR_UNSUPPORTED ("input outside the supported numerical domain").
 * Inside it, each feature is rescaled by exact powers of two chosen
 * from its CENTRED variation (translation and scale invariant: shifting by
 * 1e300 or scaling by 2^k changes the variances by 2^2k only); within-group
 * residuals carry their own exponent, so a within variation many orders
 * below the between variation is kept. The restricted likelihood is
 * profiled in gamma = s2b/s2w and evaluated with its derivative on gamma = 0
 * and log10(gamma) in [-8, log10(1000 max(1, F))] with step 1/8 (F the
 * one-way ANOVA ratio). Every grid cell where the derivative changes sign
 * from - to + is bisected to a relative width of 4 DBL_EPSILON, gamma = 0
 * is a candidate when the derivative there is >= 0, and the candidate with
 * the largest restricted likelihood is kept (several local maxima occur for
 * unbalanced groups). Grid limitation: a local maximum whose whole basin
 * (a - to + sign change of the derivative and back) lies inside one grid
 * cell, a factor 10^(1/8) = 1.334 in gamma, is not seen; balanced designs
 * are unimodal and never affected; with K distinct group sizes, f' has the
 * sign of a polynomial of degree <= 4K-3 (derivation in
 * core/variance_components.hpp), so there are at most 4K-3 stationary
 * points, but their spacing is not bounded and two of them can fall in one
 * cell. When F > 2^64 the REML limit s2w = SSW/(N-g),
 * s2b = sample variance of the group means is used (relative error O(1/F)).
 * out is row-major X.cols x 8: [0] sigma2_between, [1] sigma2_within,
 * [2] ICC(1) = s2b/(s2b+s2w), [3] restricted log-likelihood
 * -(1/2)[(N-1) log 2pi + log|V| + log(1'V^-1 1) + r'V^-1 r] (statsmodels
 * MixedLM convention), [4] groups, [5] observations, [6] local-maximum
 * candidates compared, [7] bisection iterations.
 * band_status: 0 interior optimum, 1 insufficient (fewer than two groups
 * or no within-group degree of freedom), 2 boundary sigma2_between = 0
 * (ICC 0), 3 zero total variance (undefined), 4 boundary sigma2_within = 0
 * with distinct group means (ICC 1, sigma2_between = sample variance of
 * the group means, log-likelihood NaN), 5 not converged (derivative still
 * negative at the search bound, or a bisection failed). Undefined outputs
 * are NaN; counts are always written. Inside the domain every variance is
 * representable; a value that is not returns N4M_ERR_NUMERICAL_FAILURE. */
N4M_API n4m_status_t n4m_stats_variance_components_reml(
    n4m_context_t* ctx, const n4m_matrix_view_t* X, const int64_t* group_ids,
    const uint8_t* valid_mask, double* out, int32_t* band_status);

/* Confidence intervals at level 0 < confidence < 1 for the same model,
 * per feature, as row-major X.cols x 8 (NaN when undefined):
 * [0,1] within-group SD, exact chi-square interval on the pooled
 *       within SS with N-g df: sqrt(SSW/chi2_{1-a/2}), sqrt(SSW/chi2_{a/2});
 * [2,3] balanced ANOVA ICC(1,1), exact F interval (Searle 1971):
 *       F=MSB/MSW, FL=F/F_{1-a/2}(g-1,N-g), FU=F*F_{1-a/2}(N-g,g-1),
 *       bounds (F*-1)/(F*+r-1), not truncated (both 1 when MSW = 0 or
 *       F > 2^200); only when every group has the same r >= 2 valid values;
 *       it assumes independent groups, which the caller must establish;
 * [4,5] within-group pooled SD and [6,7] REML ICC(1): percentile (type-7)
 *       intervals of a cluster bootstrap over n_bootstrap replicates.
 * cluster_ids (optional, X.rows elements) names the resampling unit; each
 * group must lie in one cluster; NULL resamples groups. Units are ordered
 * by ascending ID; replicate b draws numpy.random.default_rng(seed)
 * .integers(0, U, size=(n_bootstrap, U))[b] and takes every group of each
 * drawn unit with that multiplicity. The same draws serve every feature.
 * A replicate is valid for a feature when its REML status is 0, 2 or 4;
 * n_valid_replicates (X.cols) counts them, and the percentiles use valid
 * replicates only. The percentile interval is not coverage-calibrated
 * (about 91 % for 95 % nominal with 20 balanced pairs of repeats).
 * n_bootstrap = 0 skips the bootstrap. confidence <= 1 - 2^-63 (half tails
 * >= 2^-64, the quantile domain), otherwise N4M_ERR_INVALID_ARGUMENT. The
 * numerical domain is that of n4m_stats_variance_components_reml. More than
 * 2.5e9 bootstrap operations, n_bootstrap x X.cols x (groups + 1024),
 * return N4M_ERR_UNSUPPORTED. */
N4M_API n4m_status_t n4m_stats_variance_components_intervals(
    n4m_context_t* ctx, const n4m_matrix_view_t* X, const int64_t* group_ids,
    const int64_t* cluster_ids, const uint8_t* valid_mask, double confidence,
    int32_t n_bootstrap, uint64_t seed, double* out, int64_t* n_valid_replicates);

/* Supported numerical domain of n4m_stats_energy_distance and
 * n4m_stats_pca_monitor (lead rule 15): finite values, every nonzero |x| in
 * [2^-256, 2^256] and max/min nonzero |x| <= 2^32 over the input arrays
 * (N4M_DOMAIN_MAX_SPREAD_LOG2 = 32), so subtracting two input values never
 * erases their difference. Outside it the call returns N4M_ERR_UNSUPPORTED
 * with the message "input outside the supported numerical domain (...)"
 * before any computation. Precision statements below are conservative
 * estimates, not certified bounds. */

/* Energy distance (Szekely-Rizzo) between the groups of the rows of X
 * (n x p, Euclidean), with an optional K-sample permutation test.
 * group_codes has n elements in [0, n_groups); every group is nonempty;
 * 2 <= n_groups <= n. Distances are norms of local pairwise differences of
 * the input rows (no common centre is subtracted), squares summed in
 * double-double. M[a][b] is the mean distance over all ordered pairs i in a,
 * j in b (V-statistic: zero self-pairs included), accumulated in
 * double-double. Outputs (row-major n_groups x n_groups, caller-owned,
 * symmetric by construction): mean_distances = M; energy[a][b] = E =
 * 2M[a][b] - M[a][a] - M[b][b] (unit of X; sqrt(E) equals
 * scipy.stats.energy_distance for one column), combined in double-double;
 * energy_bounds[a][b] = (8u + 2^-80)(2M_ab + M_aa + M_bb), u = DBL_EPSILON/2,
 * a conservative estimate of the rounding of E carried over from the
 * distances (each distance has relative error below about 3u). Since E >= 0,
 * a computed value in [-bound, 0) is published as 0; below -bound the call
 * returns N4M_ERR_NUMERICAL_FAILURE. out_five: [0] statistic T = sum_{a<b}
 * n_a n_b / (n_a + n_b) E_ab (R energy::eqdist.etest), [1] its rounding
 * estimate bound(T), [2] log10(n!/prod n_a!), [3] Monte Carlo p-value
 * (1+b)/(1+B) under the declared tie rule, [4] conservative p-value
 * (1+b')/(1+B); NaN when B == 0. Draws use PCG64 seeded like
 * numpy.random.default_rng(seed): the b-th draw is rng.permutation(n) and
 * row i receives group_codes[perm[i]]. Declared tie rule: b counts draws with
 * T_perm - T >= -sqrt(DBL_EPSILON) |T| (difference in double-double). A draw
 * whose difference lies within 2 bound(T) of that threshold is numerically
 * undecidable: out_three = {b, B, undecidable}, and b' counts draws with
 * T_perm - T >= -sqrt(DBL_EPSILON) |T| - 2 bound(T) (every undecidable draw
 * taken as a tie). Requires finite X in the domain, 0 <= n_permutations <=
 * 1000000, n(n-1)/2 <= 20 million pairs, n p <= 50 million cells,
 * n_groups^2 <= 10 million and n(n-1)/2 (p + B + 1) <= 2.5 billion
 * operations, checked before allocation (N4M_ERR_UNSUPPORTED). */
N4M_API n4m_status_t n4m_stats_energy_distance(
    n4m_context_t* ctx, const n4m_matrix_view_t* X, const int64_t* group_codes,
    int32_t n_groups, int32_t n_permutations, uint64_t seed, double* energy,
    double* energy_bounds, double* mean_distances, double* out_five, int64_t* out_three);

/* PCA monitoring of rows against a reference PCA. Each fit centres its rows
 * on their own double-double column means and is decomposed with the Methods
 * PCA helper (as n4m_preflight_pca_compute) with max_components components,
 * on every training fold (reference row i belongs to fold i mod n_folds) and
 * on all reference rows. Eigenvalues carry an absolute resolution
 * r = 16 m eps lambda_1 (m = max(rows, p), eps = DBL_EPSILON, rule 14); the
 * effective k (*effective_components) is the smallest number of components
 * above the resolution among these fits, capped at max_components. For a
 * centred row: scores t (compensated dot products), T^2 = sum_a t_a^2 /
 * lambda_a (NaN when k == 0) and Q = ||x_c - sum_a t_a v_a||^2 from the
 * explicit residual. reference_t2/q (n_r) are out-of-fold; monitored_t2/q
 * (n_m) use the fit on all reference rows. Limits are type-7 (1 - alpha)
 * quantiles of the out-of-fold values. Tolerance (a conservative estimate,
 * not a proven bound): tau = 16 m eps cond, cond = lambda_1 / lambda_k of the
 * full fit (1 when k == 0); T^2 is compared with tolerance tau max(T^2,
 * limit), Q with tau S, S the larger of the row's and the reference rows'
 * squared centred norms. out_counts (6): monitored rows above the T^2 limit,
 * the Q limit, either limit by more than the tolerance, then rows strictly
 * within the tolerance band below that (a row at or below limit - tolerance
 * is not above it), for T^2, Q, either (numerically undetermined,
 * never a decision; T^2 ignored when k == 0). eigenvalues (max_components,
 * ddof 1): the first k, then NaN. out_thirteen: [0] T^2 limit [1] Q limit
 * [2] reference total variance trace(cov), ddof 1 [3] monitored total
 * variance about the monitored rows' own centre, ddof 1, NaN for one row
 * [4] ||c_m - c_r|| between double-double centroids [5] its norm inside the
 * reference subspace [6] its residual norm [7] T^2 of the monitored centroid
 * [8] retained fraction sum(eigenvalues) / [2] (NaN for zero variance)
 * [9] tau [10] cond [11] Q tolerance scale (largest S) [12] eigenvalue
 * resolution r of the full fit. Requires inputs in the domain, n_r >= 3,
 * n_m >= 1, equal columns, 2 <= n_folds <= n_r, 1 <= max_components <= p,
 * max_components <= n_r - ceil(n_r / n_folds) - 1, 0 < alpha < 1,
 * (n_r + n_m) p <= 50 million cells and (n_folds + 1) min(n_r,p)^2
 * max(n_r,p) + 4 (n_r + n_m) max_components p <= 10 billion operations,
 * checked before allocation (N4M_ERR_UNSUPPORTED). */
N4M_API n4m_status_t n4m_stats_pca_monitor(
    n4m_context_t* ctx, const n4m_matrix_view_t* reference,
    const n4m_matrix_view_t* monitored, int32_t max_components, int32_t n_folds,
    double alpha, double* eigenvalues, double* reference_t2, double* reference_q,
    double* monitored_t2, double* monitored_q, int32_t* effective_components,
    int64_t* out_counts, double* out_thirteen);

/* Declared numerical domain D of the n4m_stats_group_means and
 * n4m_stats_regression_scores kernels: finite values and every nonzero |x|
 * in [2^-256, 2^256]; regression scores, which depend on differences, also
 * require a spread max|x| / min nonzero |x| <= 2^32 inside each input array
 * (rule 15; group means are a summary and are exempt). Outside D they return
 * N4M_ERR_UNSUPPORTED with the context message "input outside the supported
 * numerical domain (...)" before computing; nonfinite values are
 * N4M_ERR_INVALID_ARGUMENT.
 *
 * Equal-weight per-group column means of the rows of X (n x k, any positive
 * strides). group_codes has n entries in [0, n_groups); every group needs at
 * least one row (n_groups <= n). Each mean is a double-double (TwoSum) sum
 * divided once by the group size: within D no partial sum overflows; the
 * relative error is of order n eps for same-sign values and the absolute
 * error of order n eps max|x| otherwise. out_means is n_groups x k row-major; out_counts receives the rows
 * per group. Generic aggregation of homogeneous technical repeats. */
N4M_API n4m_status_t n4m_stats_group_means(
    n4m_context_t* ctx, const n4m_matrix_view_t* X, const int64_t* group_codes,
    int64_t n_groups, double* out_means, int64_t* out_counts);

/* RMSE, MAE and R^2 of paired vectors (n >= 1), with the domain D above for
 * y_true and y_pred. Residuals and deviations from a
 * double-double mean are explicit and summed in double-double, so no large
 * nearly-equal accumulations are subtracted. RMSE and MAE have relative
 * errors of order n eps; R^2 an absolute error of order n eps (1 + SSE/SST). out_three = {RMSE, MAE, R^2};
 * R^2 = 1 - SSE/SST is NaN ("undefined") when y_true is constant. */
N4M_API n4m_status_t n4m_stats_regression_scores(
    n4m_context_t* ctx, const double* y_true, const double* y_pred, int64_t n,
    double* out_three);

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
