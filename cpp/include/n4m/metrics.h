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

/* HDBSCAN* on the rows of X with the Euclidean metric, ported from
 * scikit-learn 1.9.1 sklearn.cluster.HDBSCAN (alpha=1,
 * cluster_selection_epsilon=0, max_cluster_size=None) and, for GLOSH outlier
 * scores and cluster persistence, scikit-learn-contrib/hdbscan 0.8.44.
 * min_samples counts the row itself (scikit-learn convention; the contrib
 * hdbscan value is min_samples-1). Requires X.rows>=2, X.cols>=1, finite
 * values, min_cluster_size>=2, 1<=min_samples<=X.rows, selection_method
 * 0=EOM or 1=leaf, allow_single_cluster 0 or 1. Workloads with
 * rows*rows*cols above 2.5e9 return N4M_ERR_UNSUPPORTED.
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
 * scikit-learn returns 1.0 by convention. */
N4M_API n4m_status_t n4m_preflight_adjusted_rand_index(
    n4m_context_t* ctx, const int64_t* labels_a, const int64_t* labels_b, int64_t n,
    double* ari);

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
 * permutation runs when the effect size is undefined. Targets are centered
 * after an exact power-of-two scaling and a shift to the first target, with
 * a residual-mean correction, so eta^2 and F are translation invariant.
 * categorical=1 with levels * classes > 10 million cells returns
 * N4M_ERR_UNSUPPORTED before any table allocation or statistic.
 * Supported numerical domain D: finite targets whose nonzero |y| lie in
 * [2^-256, 2^256] with max/min nonzero |y| <= 2^64 (exact zeros allowed;
 * categorical class codes are labels and are not domain-checked). Inputs
 * outside D return N4M_ERR_UNSUPPORTED with the message "input outside the
 * supported numerical domain (...)" before any computation. Inside D the
 * centered copy is rescaled by exact powers of two chosen from max|y| and
 * from the centered variation, so eta^2, F, V and the permutation counts are
 * scale invariant and translation invariant up to the rounding of the
 * shifted inputs; F cannot overflow inside D (F <= n * 2^232).
 * Accuracy: eta^2 is a ratio of sums of O(1) scaled centered terms with no
 * subtraction of accumulated quantities; its absolute error is of order
 * n * DBL_EPSILON (equal level means give exactly 0 on dyadic data).
 * V^2 = chi^2 / (n (min(levels, classes) - 1)) with chi^2 accumulated from
 * exact integer residuals d = O*n - r*c as sum d^2/(n r c): nonnegative
 * terms, relative error O(cells * DBL_EPSILON), exactly 0 at independence
 * (n <= 3e9 for categorical targets). SSW is computed from the original
 * targets, each level scaled by its own power of two and centered on its own
 * values, and the level sums are combined by exponents, so a within-level
 * spread is never erased by another level's magnitude; SSW has relative
 * error O(n * DBL_EPSILON). F = eta^2 SST/(levels-1) / (SSW/(n-levels)) is
 * therefore accurate relative to SSB when SSB is not small, and to an
 * absolute error of order n * DBL_EPSILON * SST/SSW * (n-levels)/(levels-1)
 * near eta^2 = 0. F is NaN only when every level is constant (SSW = 0
 * exactly) or n == levels; an F that overflows, or falls below DBL_MIN while
 * eta^2 > 0, returns N4M_ERR_NUMERICAL_FAILURE. */
N4M_API n4m_status_t n4m_preflight_factor_association(
    n4m_context_t* ctx, const int64_t* factor_codes, const double* targets,
    int64_t n, int32_t categorical, int32_t n_permutations, uint64_t seed,
    double* out_eight, int64_t* out_two);

/* Holm step-down adjusted p-values for one family of m >= 1 hypotheses.
 * Inputs must lie in [0, 1]; adjusted[i] = min(1, max_{j<=rank(i)}
 * (m-j+1) p_(j)) with a stable ascending order, as R p.adjust("holm").
 * Nonzero p-values must lie in [2^-256, 1] with max/min nonzero p <= 2^64
 * (domain D; zeros allowed), otherwise N4M_ERR_UNSUPPORTED. */
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
 * Both vectors must lie in the same supported domain D as
 * n4m_preflight_factor_association (each vector checked separately; outside
 * D: N4M_ERR_UNSUPPORTED) and use its translation- and scale-stable
 * centering (r is a ratio of sums
 * of O(1) scaled centered terms, so its absolute error is of order
 * n * DBL_EPSILON; the denominator is >= 1/4 after rescaling and never
 * underflows; exactly uncorrelated dyadic data give r = 0). */
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
 * (domain D as for Holm, otherwise N4M_ERR_UNSUPPORTED; positive p-values
 * stay positive)
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
 * max/min nonzero |x| <= 2^64 over the whole matrix (exact zeros allowed),
 * the same D as the association kernels. Outside D the call returns N4M_ERR_UNSUPPORTED, before any
 * computation, with the context message "input outside the supported
 * numerical domain (...)". Inside D each column is centred (exact
 * power-of-two prescale, shift to its first row, compensated mean) and all
 * columns are expressed in one power-of-two unit chosen from the CENTRED
 * variation, so constant columns and common offsets do not change any output,
 * and the rescaling is exact: no varying column is erased and no squared term
 * underflows. Every sum of squares is a sum of squared terms (residuals are
 * projected explicitly, never obtained as a difference of totals)
 * accumulated in long double, and F and all ratios are formed in the scaled
 * unit. A pseudo-F (and its test) is NaN when the residual SS is at or below
 * its rounding resolution gamma^2 times the projected SS, gamma = 16
 * (rank + 2) sqrt(n k) DBL_EPSILON (a backward-error estimate). Published sums
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
 * out_thirteen (caller-owned, NaN when undefined):
 *   [0] ss_total  [1] ss_strata  [2] ss_term  [3] ss_residual
 *   [4] R^2 = ss_term/ss_total  [5] partial R^2 = ss_term/(ss_term+ss_residual)
 *   [6] pseudo-F = (ss_term/df_term)/(ss_residual/df_residual)
 *   [7] df_term  [8] df_residual  [9] term levels  [10] strata
 *   [11] log10 prod_s (n_s!/prod_a n_as!), the number of distinct
 *        within-strata level assignments (an upper bound on distinct outcomes)
 *   [12] Monte Carlo p-value.
 * out_two = {b, executed B}; *ss_exponent: true SS = out[0..3] * 2^exponent. */
N4M_API n4m_status_t n4m_stats_permanova(
    n4m_context_t* ctx, const n4m_matrix_view_t* X, const int64_t* term_codes,
    const int64_t* strata_codes, int32_t n_permutations, uint64_t seed,
    double* out_thirteen, int64_t* out_two, int32_t* ss_exponent);

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
 * SS(reduced). Monte Carlo test (n_permutations > 0 and a defined pseudo-F):
 * PCG64 as numpy.random.default_rng(seed); the b-th draw perm =
 * rng.permutation(n) gives data fitted_reduced[i] + residual[perm[i]], refitted
 * on both designs; b counts pseudo-F_perm >= F_obs - sqrt(DBL_EPSILON) (+inf
 * for an exactly zero residual SS), p = (1+b)/(1+B). Residual SS of the
 * observation and of every draw are explicit projections off the full
 * design. Numerical domain as n4m_stats_permanova. Limits, checked before
 * the design is generated: at most 4096 design columns c, n c <= 32 Mi
 * values (256 MiB) and 2 n c^2 + 4 n k (c+1)(B+1) <= 250 million
 * operations.
 * out_twelve (NaN when undefined): [0] ss_total [1] ss_reduced (model SS of
 * the reduced design) [2] ss_term [3] ss_residual [4] ss_term/ss_total
 * [5] partial R^2 ss_term/(ss_term+ss_residual) [6] pseudo-F [7] df_term
 * [8] df_residual [9] rank of the reduced design (intercept included)
 * [10] log10(n!), the distinct residual permutations bound [11] p-value.
 * out_two = {b, executed B}; *ss_exponent: true SS = out[0..3] * 2^exponent. */
N4M_API n4m_status_t n4m_stats_permanova_freedman_lane(
    n4m_context_t* ctx, const n4m_matrix_view_t* X, const int64_t* term_codes,
    const int64_t* condition_codes, int32_t n_conditions, int32_t n_permutations,
    uint64_t seed, double* out_twelve, int64_t* out_two, int32_t* ss_exponent);

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
 * out_six: [0] F (NaN with one group, n == groups or an unresolved within SS)
 * [1] df1 = groups-1 [2] df2 = n-groups [3] groups
 * [4] log10(n!/prod n_g!) [5] Monte Carlo p-value; out_two = {b, executed B};
 * *distance_exponent: true distance = out * 2^exponent (both arrays). */
N4M_API n4m_status_t n4m_stats_permdisp(
    n4m_context_t* ctx, const n4m_matrix_view_t* X, const int64_t* group_codes,
    int32_t n_permutations, uint64_t seed, double* distances,
    double* group_mean_distances, double* out_six, int64_t* out_two,
    int32_t* distance_exponent);

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
