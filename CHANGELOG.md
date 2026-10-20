# Changelog

All notable changes to `nirs4all-methods` (`n4m`) will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.3.8] - 2026-10-20

ABI 2.20.0 unchanged (no new or changed symbol). Patch release for the CI
failures of 1.3.7 on MSVC, Apple clang and the clang sanitizer presets.

### Fixed

- Precision where `long double` is `double` (MSVC, Apple arm64): three
  kernels relied on the x87 extended format, found by running ctest and the
  Python suite on an x86-64 build with `-mlong-double-64`, which both pass.
  - `n4m_stats_permanova_freedman_lane`: the reduced-design basis and the
    reduced-model residuals are computed in double-double. A basis rounded
    to double left about `DBL_EPSILON` times a large between-condition
    offset in every residual: an offset stratum `c = 2^31` (inside the
    domain D) gave `SS_term` off by 6e-8 relative on MSVC and Apple clang.
  - `n4m_stats_permanova`: stratum means in double-double. A rounded mean
    shifted every within-stratum row by about `DBL_EPSILON c` (relative
    error 5e-11 on x86, 7e-9 where `long double` is `double`, for a 3/5
    stratum split at `c = 2^31`); both kernels now match the rational
    oracle to 1e-12 with 80-bit and 64-bit `long double`.
  - `utilities.ragged_summary`: Welford updates in double on each channel
    prescaled by an exact power of two; `[1e308, -1e308]` overflowed the
    deltas and was refused where `long double` is `double`.
  - Headers and method pages no longer imply extended precision for these
    kernels.
- clang `-Wdouble-promotion -Werror` (linux-clang16, macOS universal2,
  sanitizer presets, Rust sanitized jobs): explicit `long double`
  promotions in `cpp/tests/test_models_pls.cpp`. MinGW GCC 16
  `-Wnull-dereference` in `cpp/tests/test_pls_null_components.cpp`. A
  clang 20 sweep with the warning set finds no `-Wdouble-promotion` left in
  the sources, the tests or the WebAssembly build; the tests compiled
  without the warning set (shadowing, `NAN` and integer promotions) were
  cleaned as well.

## [1.3.7] - 2026-10-20

ABI 2.20.0 (minor: 20 new symbols and two new `n4m_model_array_t` tags,
no signature change of the 2.19 surface). Model-state serialization: PLS
regression, PLS-DA and PCR models are now written as N4MM format 3
(component resolution section, see Compatibility); formats 1 and 2 are
unchanged for the other models and still read.

### Behaviour changes of symbols published in 1.3.6

- `n4m_preflight_factor_association`: targets are centred translation
  stably (exact power-of-two scaling, shifted origin, residual-mean
  correction). Two identical groups at `1e16 + {0, 2, 4}` gave eta^2 = 0.27
  and F = 1.5; they now give 0. Permutation draws and counts are unchanged.
- `n4m_cluster_hdbscan`: inputs whose squared distances overflow (for
  example `[0, 1e155, 2e155, 3e155]`) could write past the caller's buffers;
  they are now outside the numerical domain and refused. A zero-distance
  `inf - inf` excess counts as 0 instead of NaN, MST edges and tree sizes are
  validated before anything is written, and each call checks a 256 MiB
  working-memory budget before allocating.
- SNV (`n4m_transform_snv_*`, the registry `StandardNormalVariate` and the
  pipelines using it): a constant finite row now maps to exact zeros instead
  of the standardized rounding residue of its mean (eleven `0.1` gave eleven
  `1.0`); a non-finite result for a finite row returns
  `N4M_ERR_NUMERICAL_FAILURE`. Agreement with NumPy stays within the 1e-12
  parity tolerance but is no longer bit-for-bit.
- The ABI 2.18/2.19 Preflight kernels (`n4m_preflight_*`) and
  `n4m_cluster_hdbscan` refuse inputs outside their declared numerical
  domain with `N4M_ERR_UNSUPPORTED` (see Changed). `n4m_preflight_pca_compute`
  publishes components below its resolution bound as 0 (use
  `n4m_preflight_pca_compute_resolved` for the bound and the flags), and
  `n4m_preflight_noise_proxy_compute` accepts only the verified
  Savitzky-Golay domain.
- Internal SVD / symmetric-eigen helpers no longer fail or return wrong
  factors on degenerate inputs (see Fixed); on non-degenerate inputs the
  results are bit-identical to 1.3.6.
- `n4m_preflight_pca_compute` no longer uses a randomized sketch on large
  truncated requests (a first variance 7 % low with `N4M_OK`): it
  eigendecomposes the smaller-side Gram matrix for every n, p and k.
  `n4m_outlier_detection_hotelling_t2` (and the `utilities.hotelling_t2`
  procedure) uses only resolved directions and the UCL of the rank actually
  used, so collinear columns no longer inflate T².
- `n4m_outlier_detection_hotelling_t2`: full-rank results are no longer
  bit-identical to 1.3.6. The corrected two-pass centring, T² evaluated as
  `(n-1) * sum_j U_ij^2` and the UCL solved from the beta tail change them
  at the rounding level of the centring (no fixed bound claimed; see
  Fixed); constant columns centre to exact zeros (rank 0, T² 0, UCL 0).
- `n4m_outlier_detection_q_residuals`: shares the Hotelling centring, uses
  the signed Jackson-Mudholkar h0 (h0 < 0 previously gave a far too low
  UCL), returns Q = UCL = 0 exactly when no residual direction is resolved,
  and returns `N4M_ERR_NUMERICAL_FAILURE` for an unrepresentable result or a
  non-positive Jackson-Mudholkar base instead of UCL 0 with `N4M_OK`.
- `n4m_preflight_pca_compute` / `_resolved`: loadings of wide inputs
  (n <= p) are orthonormal rows; the request is checked against guaranteed
  work (2.4e9) and memory (1 GiB) bounds before allocating, so some inputs
  1.3.6 accepted beyond the Preflight profile (e.g. n = 3, p = 26e6, k = 1)
  are now refused with `N4M_ERR_UNSUPPORTED`. The conservation covers only
  shapes that were already admissible: every shape of the Preflight profile
  (at most 10^7 cells, k <= 10) that Methods 1.3.6 (Gram-only budget
  s^2 l <= 2.5e8) or the intermediate estimate accepted is still accepted
  (enumerated in `cpp/tests/test_preflight.cpp`). The profile alone is not
  sufficient: every request must also satisfy the work and memory bounds, so
  for example a 1000 x 1000 input with k = 10 is refused by the work bound.
- PLS regression (all eight solvers) and PCR without autoscaling are
  scale-equivariant: data at 2^-30 that returned `N4M_ERR_NUMERICAL_FAILURE`
  now fit. Previously successful fits are bit-identical except NIPALS
  (coefficients within a few 1e-12 relative, latent arrays up to ~4e-7
  relative when Y was much larger than X) and the PCR Y scores it zeroed
  (see Fixed).

### Added

- Public ABI 2.20.0 `n4m_stats_group_means` and `n4m_stats_regression_scores`.
  Group means are equal-weight per-group column means of a strided matrix,
  computed as double-double (TwoSum) sums divided once by the group size.
  Regression scores are RMSE, MAE and R² computed from explicit residuals
  and from deviations around a double-double mean, with every sum in
  double-double, written as four doubles {RMSE, MAE, R² significand, R²
  exponent}: R² below the double range (SSE/SST ≥ 2^999, possible inside
  the domain) is published scaled, never as -Inf. R² is NaN (undefined)
  for a constant target.
  Both kernels declare a numerical domain: finite values and every nonzero
  |x| in [2^-256, 2^256]; regression scores, which depend on differences,
  also need a spread of at most 2^32 per input array (rule 15, shared
  `N4M_DOMAIN_MAX_SPREAD_LOG2`). Precision: relative error of order n eps
  for the means of same-sign data, RMSE and MAE; absolute error of order
  n eps (1 + SSE/SST) for R².
  Outside this domain they refuse with `N4M_ERR_UNSUPPORTED` ("input
  outside the supported numerical domain"); nonfinite values return
  `N4M_ERR_INVALID_ARGUMENT`.
  nirs4all-preflight uses them to aggregate technical repeats and to score
  its validation probe.
  New JS/WASM wrappers: `statsGroupMeans` and `statsRegressionScores`.
  Catalog entries: `diagnostics.group_means` and
  `diagnostics.regression_scores`.
  `n4m_group_means_tests` covers exact dyadic cases, scale sweeps from
  2^-200 to 2^200, a 2^52 offset, a one-ulp case, and both sides of the
  domain boundaries; the JS test mirrors it.
- Public ABI 2.20.0 generic statistics: `n4m_stats_correlation_test`
  (Pearson or Spearman correlation with a seeded two-sided permutation
  p-value, NumPy `default_rng` permutation parity) and `n4m_stats_bh_adjust`
  (Benjamini-Hochberg adjusted p-values). JS/WASM wrappers
  `statsCorrelationTest` and `statsBhAdjust`. SciPy 1.15.3 `pearsonr` /
  `spearmanr`, statsmodels 0.15.0 `multipletests(method="fdr_bh")` and NumPy
  2.2.6 permutation counts are pinned in `cpp/tests/test_preflight.cpp`.
- `n4m_stats_factor_association_exact` and `n4m_stats_correlation_exact`:
  exhaustive enumeration of distinct arrangements, returning the exact
  p-value and the minimal attainable exact p-value (JS
  `statsFactorAssociationExact`, `statsCorrelationExact`). Brute-force SciPy
  references are pinned in the native and JS tests.
- Public ABI 2.20.0 multivariate tests in `n4m/metrics.h`:
  `n4m_stats_permanova` (PERMANOVA of one categorical term on Euclidean
  distances, optionally given strata, with the vegan
  `adonis2(by = "margin")` sums of squares for one other factor, the exact
  identifiable term df from the connected components of the level-stratum
  graph, so an aliased term reports df 0, and a seeded within-strata Monte
  Carlo test) and `n4m_stats_permdisp` (vegan `betadisper(type =
  "centroid")` + `permutest` residual permutation). Both reuse the PCG64
  permutation helper (`numpy.random.default_rng` parity) and are wrapped in
  JS/WASM as `statsPermanova` and `statsPermdisp`. As in vegan, both count
  permuted `F >= F_obs - sqrt(DBL_EPSILON)` on the F scale. Inputs are scaled
  by an exact power of two and centred per column; sums of squares are mapped
  back with `scalbn`, so subnormal and near-`DBL_MAX` results stay
  representable.
- `n4m_stats_permanova_freedman_lane` (ABI 2.20.0): partial PERMANOVA of a
  term given the additive main effects of condition factors, tested by
  Freedman-Lane permutation of the reduced-model residuals (refit of both
  designs, `F_perm >= F_obs - sqrt(DBL_EPSILON)`, numerical ranks so an
  aliased term or a saturated design gives no test). JS/WASM
  `statsPermanovaFreedmanLane`. It complements the within-strata test, whose
  null also excludes interactions with the conditions. The reduced-model
  residuals are projected off the reduced design twice, and the partial R^2
  is SS_term / (SS_term + SS_residual) of one projection (24- and 32-row
  dyadic references: R^2 = 0.5, F = 21 and 29).
- Reference gate `parity/fixtures/permanova_v1.json`
  (`parity/generators/permanova_v1.py`, `n4m_permanova_tests`): NumPy 2.2.6
  least squares on full one-hot designs, SciPy 1.15.3 `f_oneway`,
  scikit-bio 0.7.4 `permanova`/`permdisp` and NumPy permutation counts, on
  free, conditional, partially aliased, fully aliased, singleton-stratum,
  dispersion-only and PERMDISP F-boundary cases (vegan `permutest.betadisper`
  transcribed as the counting oracle), plus hand-derived closed forms; the
  former 1e-162 / 1e150 / 1e155-offset scale cases are now refused. Freedman-Lane goldens come from an
  independent lstsq refit oracle (one, two and zero condition factors,
  aliased and saturated designs); vegan's own margin and partial-permutest
  schemes, which are not Freedman-Lane, are transcribed for information only.
  The reference packages are test-only.
- Public ABI 2.20.0 one-way random-effects variance components:
  `n4m_stats_variance_components_reml` (REML for balanced and unbalanced
  groups; every local maximum of the restricted likelihood the grid resolves
  is refined and compared with the boundary s2b = 0; boundary and
  convergence status per feature; the grid limitation is documented with a
  degree bound, 4K-3 stationary points for K distinct group sizes) and
  `n4m_stats_variance_components_intervals` (exact chi-square interval of the
  within-group SD, exact F interval of the balanced ICC(1,1), seeded
  cluster-bootstrap percentile intervals of the pooled SD and REML ICC with
  NumPy-parity PCG64 draws). Supported numerical domain (rules 13/15):
  nonzero |x| in [2^-256, 2^256], spread <= 2^32 over values and differences;
  outside it both calls refuse with `N4M_ERR_UNSUPPORTED`. The percentile
  intervals are not coverage-calibrated: 0.86-0.94 measured at 95 % nominal
  (`parity/generators/variance_components_coverage.py`); BCa was rejected
  (0.74 at ICC 0.1 with 20 groups). Chi-square and F quantiles
  (`cpp/src/core/common/distributions.cpp`) take tail probabilities and use
  cancellation-free prefactors. Native tests compare with dense REML score
  equations (NumPy 2.5.4 / SciPy 1.18.1), statsmodels 0.15.0 MixedLM,
  balanced closed forms, SciPy and mpmath 1.4.1 quantiles, a multimodal
  witness, 2^k / offset sweeps and both sides of every domain boundary. The
  centring helpers `bounding_exponent` and `center_in_place` move to
  `cpp/src/core/common/centring.hpp`, shared with the Preflight kernels.
  JS/WASM wrappers `statsVarianceComponentsReml` and
  `statsVarianceComponentsIntervals`.
- Public ABI 2.20.0 domain-shift statistics: `n4m_stats_energy_distance`
  (energy distances between groups of rows, symmetric, with conservative
  rounding estimates, the K-sample statistic of R `energy::eqdist.etest` and
  a seeded Monte Carlo label permutation test with `numpy.random.default_rng`
  parity; declared tie rule `T_perm >= T - sqrt(eps)|T|`, numerically
  undecidable draws counted separately with a conservative p-value) and
  `n4m_stats_pca_monitor` (reference PCA with an effective k of components
  above the eigenvalue resolution, T² and Q of monitored rows, out-of-fold
  empirical limits, counts beyond the limits by more than a conservative
  tolerance and counts within it, centroid shift, variance traces). JS/WASM
  `statsEnergyDistance` and `statsPcaMonitor` check the budgets before any
  WASM allocation. Supported domain (lead rule 15, shared constant
  `N4M_DOMAIN_MAX_SPREAD_LOG2 = 32`): finite values, nonzero |x| in
  [2^-256, 2^256], spread <= 2^32; other inputs return `N4M_ERR_UNSUPPORTED`.
  Reference gate `parity/fixtures/domain_shift_v1.json` (NumPy 2.2.6,
  SciPy 1.15.3, exact rational oracles).
- Public ABI 2.20.0 cluster matching and null-model kernels in
  `n4m/metrics.h`: `n4m_cluster_jaccard_matrix` (intersection counts, cluster
  sizes and Jaccard indices between two partitions, negative labels
  unassigned), `n4m_linear_sum_assignment` (a BSD-3 port of SciPy 1.18.1
  `scipy.optimize.linear_sum_assignment` with SciPy's tie order, vendored in
  `cpp/src/core/optimization/_vendored`) and `n4m_random_multivariate_normal`
  (NumPy `default_rng(seed).multivariate_normal(..., method="cholesky")` on
  the existing PCG64 Ziggurat stream). A reference gate
  (`parity/fixtures/cluster_matching_null_v1.json`,
  `n4m_cluster_matching_null_tests`) pins SciPy assignments, scikit-learn
  contingency tables, NumPy draws and stable-tie scikit-learn HDBSCAN
  partitions of simulated Gaussian data. JS/WASM wrappers:
  `clusterJaccardMatrix`, `linearSumAssignment`, `randomMultivariateNormal`.
- `n4m_random_permutations` (ABI 2.20.0): successive
  `default_rng(seed).permutation(n)` draws from one Methods PCG64 stream, for
  seeded subsampling plans; JS/WASM `randomPermutations`.
- `n4m_preflight_snv_compute` (ABI 2.20.0) and the JS wrapper
  `preflightSnv`: SNV with an optional compensated equal-weight aggregation
  by group. Each output row carries a first-order bound on the rounding error
  of its entries, so callers can tell rounding residue from structure.
- Property and oracle tests for SNV, the noise proxy and tolerance pairs on
  the supported domain: scales 2^-256..2^248 and their out-of-domain
  neighbours, exact offsets near 1e16 and 2^200, constant columns up to the
  spread limit, mixed exponents within a row, exact rational and 400-digit
  references. Each result is correct within its stated error, or the input is
  refused with `N4M_ERR_UNSUPPORTED` outside the domain.
- ABI 2.20.0: `n4m_outlier_detection_hotelling_t2_resolved` publishes the
  rank used by Hotelling T²; the `utilities.hotelling_t2` procedure reports
  it as `n_components_used`.
- ABI 2.20.0: `n4m_preflight_pca_compute_resolved` publishes the PCA
  resolution bound and a per-component `unresolved` flag (rule-14 accuracy
  contract). The JS `preflightPca` wrapper returns `resolution` and
  `unresolved`. Tested against mpmath references (`B_n(t)` cancellation
  families, a Kronecker product, permutations) on CPU and WASM.
- Component resolution of PLS regression and PCR models (ABI 2.20).
  `n4m_model_get_array(..., N4M_MODEL_COMPONENT_RESOLUTION)` returns (1, 1)
  the absolute resolution below which a component's X-Y cross-covariance
  cannot be told apart from rounding, in centred (scaled) X times Y units:
  (n + a) eps ||X||_F ||Y||_F on ||X_k'Y_k||_F for every PLS regression
  solver (and PLS-DA), (n + p) eps sigma_max(X) ||Y||_F on ||Y't_k|| for PCR.
  `N4M_MODEL_UNRESOLVED_COMPONENTS` returns (1, n_components) flags, 1.0 for
  a component whose cross-covariance is not above it. One flag covers two
  cases that double precision cannot tell apart: a component that is
  mathematically undefined (exact cross-covariance zero, e.g. beyond the rank
  of X, or u = Y q / q'q with q = 0) and one that is merely not resolved
  numerically (nonzero but below the bound). Neither is refused any more
  inside the supported domain (see Changed for the PLS convention). PCR
  keeps a component whose X direction is resolved, publishes its Y loading
  (absolute accuracy only) and a zero Y-score column, and now flags it: with
  a = (-3,-1,1,3), b = (1,-3,3,-1), X = a, Y = b + 2^-50 a (one component,
  centred, unscaled) the exact Y scores are 2^50 Y, but the cross-covariance
  20 * 2^-50 is below the resolution and the zero column used to be
  published without any indication. Other models return
  `N4M_ERR_UNSUPPORTED` for both tags. The values are kept by export/import
  in N4MM format 3 (see Changed) and exposed by the JS/WASM
  `NativeModel.getArray` / `componentResolution()` and `ModelArray` enum, and
  by Python `pls4all.Model.component_resolution()`, `ModelArrayKind` and the
  `component_resolution_` / `unresolved_components_` attributes of the
  sklearn PLS regression and PCR wrappers.

### Changed

- One supported numerical domain D for the Preflight and statistics kernels
  (lead rules 13 and 15): finite values, every nonzero |x| in
  [2^-256, 2^256] and, wherever a statistic depends on differences of input
  values, max/min nonzero |x| <= 2^32 inside the array or slice that is
  differenced. Public macros `N4M_DOMAIN_MAX_MAGNITUDE_LOG2` (256) and
  `N4M_DOMAIN_MAX_SPREAD_LOG2` (32) in `n4m/metrics.h`; one internal helper
  (`cpp/src/core/common/numerical_domain.hpp`) and one message, "input
  outside the supported numerical domain (nonzero |x| in [2^-256, 2^256],
  max/min nonzero |x| <= 2^32)", or its magnitude-only variant for the
  exempt kernels. Exempt from the spread limit: summaries, nearest neighbours
  and group means (magnitude bounds only), Holm/BH p-values (nonzero p >=
  2^-1022) and integer labels/codes. The PCA declares a tighter magnitude
  bound, [2^-128, 2^128]. Compensated sums of the kernels use one
  double-double helper (`cpp/src/core/common/double_double.hpp`).
- Numerical domain of `n4m_stats_permanova`, `n4m_stats_permanova_freedman_lane`
  and `n4m_stats_permdisp`: the shared supported domain D (finite X, every
  nonzero |x| in [2^-256, 2^256], max/min nonzero |x| <= 2^32 over the whole
  matrix, `N4M_DOMAIN_MAX_SPREAD_LOG2 = 32`, zeros allowed) is checked in one
  O(n k) pre-pass; outside D the kernels return `N4M_ERR_UNSUPPORTED` before any
  computation with the message "input outside the supported numerical domain
  (...)". Inside D columns are centred before scaling and the common
  power-of-two unit comes from the centred variation, so constant columns and
  offsets change nothing, and the rescaling is exact (no varying column is
  erased, no squared term underflows); every sum of squares (including each
  permutation's residual SS) is a sum of squared terms of explicit residuals,
  never a difference of totals; centring rounds each value by about n
  DBL_EPSILON times its column's range; a pseudo-F is NaN only for an
  exactly zero residual SS, and is otherwise published with a resolution
  ratio (a conservative first-order error estimate of the residual SS over
  it) and a "numerically unresolved" flag when that ratio is >= 1; sums of
  squares and
  distances carry a binary exponent output (`ss_exponent`,
  `distance_exponent`): 0 with true values when all positive values are
  normal doubles, otherwise scaled values with the unit's exponent, so a
  positive quantity is never published as 0; design sizes and budgets are
  checked before any level-dependent allocation. Native and WASM property
  tests cover both sides of D: magnitudes 2^k (k in [-256, 251] inside;
  -1069, -257, 252, 1000 refused), offsets (1e16 and 2^200 inside; 2^300 and
  1e300 refused), constant columns (2^28 inside; 2^30 and 1e308 refused),
  mixed exponents (2^13 next to -2^-13 inside, 2^28 next to -2^-28
  refused), the value-range edges 2^+-256 (inside) and 2^+-257 (refused),
  exact dyadic cases at the spread boundary (columns 2^16 B and 2^-17
  (A + r), 16/24/32 rows: SS = n 2^-34, F = n - 3) whose 2^33- and
  2^64-spread variants are refused, an offset stratum (2^31 next to 1..4:
  SS_A = 2, SS_res = 3, F = 10/3; 2^60 refused), and resolution cases (F =
  14 2^62 resolved; one-ulp residuals flagged unresolved; a zero residual SS
  gives NaN).
- Supported numerical domain D for the association, correlation and exact
  kernels: nonzero |x| in [2^-256, 2^256], max/min nonzero
  |x| <= 2^32 per input array (public macro `N4M_DOMAIN_MAX_SPREAD_LOG2`),
  zeros allowed. Inputs outside D are refused
  with `N4M_ERR_UNSUPPORTED` ("input outside the supported numerical domain
  (...)") by a pre-pass; native and JS/WASM tests cover both sides of every
  boundary, and the magnitude sweeps run inside D.
- Holm and BH are exempt from the spread limit: they accept any
  p-values in [0, 1] whose nonzero values are normal (>= 2^-1022), e.g.
  p = 1 next to 1e-25; only subnormal p-values are refused.
- Supported numerical domain of the Preflight kernels (`n4m_preflight_snv_compute`,
  `n4m_preflight_noise_proxy_compute`, `n4m_preflight_pairs_within_tolerance`):
  every nonzero |x| in [2^-256, 2^256] and, per row, max/min nonzero
  |x| <= 2^32 (`N4M_DOMAIN_MAX_SPREAD_LOG2`, lead rule 15). Inputs outside are refused before computing with
  `N4M_ERR_UNSUPPORTED` ("input outside the supported numerical domain").
  Inside, results are correct to the documented precision. The machinery that
  only served out-of-domain inputs (halved differences, 1/8 prescaling,
  subnormal clamps) is removed. The same domain now applies to
  `n4m_preflight_local_contrast` (continuous targets).
  `n4m_preflight_summary_compute` and `n4m_preflight_nearest_neighbors` keep
  only the magnitude bound: compensated moments and quartiles, compensated
  distances and exact ordering near ties need no spread bound, so
  rounding-level PCA score components are accepted.

- Noise proxy precision contract made honest. Each residual has an absolute
  error of at most `2 W eps sum|c_k| max|w_k|` against the exact
  Savitzky-Golay residual, which includes the coefficient errors and the
  final binary64 rounding. The former `2^-95 sum|w_k||c_k|` claim did not
  hold: a zero exact weight gave a 2e-33 proxy, and the residual is rounded
  to binary64. Tests compare with exact rational SG residuals (windows 3,
  13 and 31) and check that no window/order pair of the validated SG domain
  is refused inside D.
- Rules 13/15 numerical domain for the clustering kernels with real inputs
  (`n4m_cluster_hdbscan`, `n4m_linear_sum_assignment`,
  `n4m_random_multivariate_normal`): finite values, every nonzero `|x|` in
  `[2^-256, 2^256]` and max/min nonzero `|x|` `<= 2^32` per input array,
  through the public macros `N4M_DOMAIN_MAX_MAGNITUDE_LOG2` (256) and
  `N4M_DOMAIN_MAX_SPREAD_LOG2` (32) in `n4m/metrics.h`.
  Outside it the call returns `N4M_ERR_UNSUPPORTED` ("input outside the
  supported numerical domain ...") before computing; inside it results are
  correct (exact optimum for the assignment, reference operation order for
  HDBSCAN, exact `2^k` scaling). Each kernel also checks a cumulative
  working-memory budget of 256 MiB before allocating, because the WASM32
  build (2 GiB maximum, C++ exceptions not caught) aborts on a failed
  allocation. ARI and Jaccard take integer labels: no numerical domain, but
  memory and size limits (`n <= 2^23` for ARI, `n <= 2^53` for Jaccard).
- Unresolved PLS components are published as flagged null components
  instead of being refused. For every PLS regression solver (NIPALS, SIMPLS,
  orthogonal scores, kernel, wide kernel, SVD, power, randomized SVD) and
  PLS-DA, with one convention across solvers: a component whose X-Y
  cross-covariance is not above the published resolution (not resolved
  numerically, or mathematically undefined) has exactly zero W, P, Q, R and
  (with store_scores=1) T and U columns, does not contribute to the
  coefficients, and is flagged in `N4M_MODEL_UNRESOLVED_COMPONENTS`. Its
  deflation leaves the residuals unchanged, so every later component is null
  and flagged too (the solver stops there); the coefficients are those of
  the model truncated to the resolved components, bit for bit, and all zero
  with predictions equal to the Y mean when none is resolved. Previously the
  fit returned `N4M_ERR_NUMERICAL_FAILURE` ("X-Y covariance is below its
  numerical resolution"). With X = a, Y = b + 2^-50 a (a, b as above), NIPALS
  and every other PLS solver now return one null component, coefficient 0,
  predictions = the Y mean; with X = [a, a] and two components, the second
  component is null and the coefficients are those of the one-component
  model. PCR keeps its convention for a component whose X direction is
  resolved (direction, Y loading and coefficients kept, Y-score column zero,
  flagged); a component whose X singular value is not resolved (beyond the
  rank of X), which PCR also refused, is now a flagged null component, like
  every later one. Requests that are invalid (e.g. more components than
  min(n - 1, p)) and inputs outside the supported numerical domain are still
  refused. `n4m_model_coefficients_by_component` returns the truncated
  coefficients for prefixes that include null components, and the
  FastAOM-PLS CV path still scores only the resolved prefixes (the others
  remain infinite, as when the fit was refused). A before/after bitwise sweep
  (3690 fits, see Fixed) found no change of array or prediction on any fit
  that succeeded before; the 298 fits that were refused (35 per PLS solver:
  20 with more components than the rank of X, 6 with Y exactly in the span
  of X, 9 with X columns scaled over up to 2^60 whose remaining covariance
  falls below the absolute resolution; 18 PCR, all beyond the rank of X)
  now return flagged null components after the resolved prefix, whose
  coefficients, latent arrays and predictions are bit-identical to a fit
  with the resolved number of components.
- N4MM fitted-model format 3 (`N4M_SERIALIZATION_FORMAT_VERSION_V3`, now the
  current version) is written for every model that records a component
  resolution, i.e. every PLS regression (PLS-DA) and PCR fit: the eleven
  format-1 vectors, then the resolution, the per-component flags and an
  explicit pipeline flag before the optional SNV/Savitzky-Golay block.
  Models without a resolution (canonical PLS, PLS-SVD, OPLS, imported linear
  predictors) keep formats 1 and 2 byte for byte, and formats 1 and 2 remain
  readable (without a recorded resolution). Readers before ABI 2.20 reject
  format 3 with `N4M_ERR_VERSION_INCOMPATIBLE`. The Rust binding accepts it.

- PCR with exactly equal (or, within rounding, indistinguishable) singular
  values now documents its convention (`n4m.h`, `n4m_model_array_t`): only
  the singular subspace is defined, and PCR returns the deterministic basis
  of its one-sided Jacobi SVD (fixed rotation order, stable descending sort,
  largest-magnitude entry positive). With a = (-3,-1,1,3), b = (1,-3,3,-1),
  X = [a, b] (X'X = 20 I), Y = [a + d b, d a + b], d = 2^-12, one component
  has W = (1, 0) and coefficients [[1, d], [0, 0]] (column order decides;
  [[d, 1], [0, 0]] for X = [b, a]); two components do not depend on the
  convention. Behaviour unchanged; C++ and JS/WASM tests pin it.

### Compatibility

- N4MM version 3: consumers that validate the N4MM byte version of a stored
  model must accept 3 when moving to Methods >= 1.3.7, because every PLS
  regression, PLS-DA and PCR model is now exported as format 3. This
  includes the nirs4all-core V2/V3 archive contracts, whose `format_version`
  field declares the N4MM payload version and is compared with the version
  read from the bytes at replay: a new PLS/PCR model declared as 3 is
  rejected by validators that only allow 1 or 2, and declaring 1 or 2 fails
  the comparison with the bytes. Formats 1 and 2 are unchanged for the
  other models.
- PLS fits that used to fail with `N4M_ERR_NUMERICAL_FAILURE` on an
  unresolved component now succeed with flagged null components (see
  Changed). Callers that relied on that refusal (e.g. to find the usable
  rank by retrying with fewer components) should read
  `N4M_MODEL_UNRESOLVED_COMPONENTS` instead.

### Fixed

- `n4m_preflight_factor_association` centered targets on a naive mean after
  dividing by max|y|, so a large common offset produced a spurious
  between-group effect (two identical groups at `1e16` gave eta^2 = 0.27).
  Centering is now translation stable (power-of-two scaling, shifted origin,
  residual-mean correction), with translation-invariance tests.
- The categorical association kernels computed `levels x classes` in
  `size_t` without overflow check and allocated the table before the budget
  check; 65536 x 65536 wrapped to 0 in WASM32 and the exact API then wrote
  out of bounds. The product is now checked and capped at 10 million cells
  before any allocation, and both the Monte Carlo and exact APIs refuse
  oversized work before evaluating any statistic.
- Magnitude robustness of the association kernels: the centered copy is
  also rescaled from the centered variation, SSW is computed from per-level
  residuals at their own power-of-two scale (a 2^-510 within-level spread
  next to a unit gap gives F = (2/3) 2^1021 instead of NaN), and an
  unrepresentable F returns `N4M_ERR_NUMERICAL_FAILURE`. Native and WASM
  sweeps over 2^k (k from -1060 to 1000) and translations by 1e16 and 1e300
  check invariance of eta^2, F, r, rho, counts, exact distributions, Holm
  and BH.
- Within-level sums of squares are computed from the original targets with a
  per-level scale (no global scale or centre erases them); a lost or
  unrepresentable F is an explicit `N4M_ERR_NUMERICAL_FAILURE`. Cramer's V
  is computed from exact integer chi-square residuals (exactly 0 at
  independence). Mixed-exponent and near-zero-effect tests use exact dyadic
  and rational references; the precision contract in `metrics.h` is revised
  to what these computations guarantee.
- Summary: the `long double` running mean, whose precision differed between
  x86 (64-bit significand) and WASM (113-bit), is replaced by a double-double
  mean (double-double sum, one exact division remainder). It has no refinement pass, which
  would add `u mean|x - m|` on cancelling values such as
  `[-2^200, 2^200, 2^150, -2^140, 7·2^180]`. The SD is a compensated sum of
  squared deviations, and the quartiles keep `b - a` exactly.
- Nearest neighbours: compensated distances. Candidates within 2^-45 of each
  other are ordered exactly (exact accumulator), so rounding can no longer
  swap distinct distances or break equal ones out of index order.
- Local contrast: double-double variance and compensated sums, no
  `long double`.
- SNV: the same mean (no refinement pass); bound
  `(3uA + n^2 u^2 mean|y| + 2u^2|m|)/sd_y + 8u max|z|`.
- SNV numerical scheme: each finite row is rescaled by a power of two. The
  mean is a compensated double-double, so offsets no longer leak into the
  deviations, and the deviations are rescaled again before squaring. `[a, 0, -a]` at the
  smallest subnormal now gives `[sqrt(1.5), 0, -sqrt(1.5)]` instead of
  returning the row unchanged. Rows with 100,000 offset bands no longer carry
  a mean error of about 1e-8 SD. Agreement with numpy remains within the
  1e-12 parity tolerance, but is no longer bit-for-bit.
- Tolerance-pair distances are computed on rescaled differences independently
  of the tolerance: `[2^-200, 0]` against `[0, 0]` lists 2^-200/sqrt(2) at any
  tolerance instead of 0. Pairs within 2^-45 (relative) of the
  tolerance are decided exactly from `sum (q - r)^2` against `cols tol^2`,
  using a fixed-point accumulator of exact products. `[2^-255, 0]` against
  `[0, 0]` at 2^-256, `[1, 1]` against `[-2^-54, -2^-54]` at 1, and
  `[1, 1, 2^-26]` against 0 at `sqrt(2/3)` are far.
- Noise proxy: each window's residual is computed from the window alone.
  Its second differences are kept exactly until a double-double weighted sum
  with double-double Savitzky-Golay weights. A huge value outside the
  evaluable windows, masked or not, no longer erases the noise, and
  `[-2^60, 2^60, 1, 0, 0]` gives its analytic `sqrt(4/5)`. The gain no longer
  uses `long double`, which differed between x86 and WASM.

- SNV (`n4m_transform_snv_*`, the registry `StandardNormalVariate` and the
  pipelines using it): a constant finite row now maps to exact zeros when
  centred instead of standardizing the rounding residue of its mean (eleven
  `0.1` gave eleven `1.0`). `[1e200, 0, -1e200]` no longer gives a zero row.
  A non-finite result for a finite row returns `N4M_ERR_NUMERICAL_FAILURE`.
  Rows with NaN/Inf propagate them as before.
- `n4m_preflight_noise_proxy_compute` accepts only the verified
  Savitzky-Golay domain (odd window 3..1001, polyorder <= min(10, window-2))
  and checks the polynomial reproduction moments of the weights; window 41 /
  polyorder 38 previously returned a residual of 0.83 on a constant row.
- `n4m_preflight_pairs_within_tolerance` no longer drops pairs whose raw
  difference overflows: such a pair is measured on exact halves
  (`q=[1e308,0]`, `r=[-1e308,0]`, tolerance `1.5e308` is now a pair).
- JS: `preflightNoiseProxy`, `preflightPca`, `preflightNearestNeighbors`,
  `preflightLocalContrast` and `preflightFactorAssociation` range-check every
  `int32_t` argument before the Emscripten call (4294967301 was wrapped to a
  window of 5); the noise proxy also checks the Savitzky-Golay domain and
  `windowLength <= cols`.
- `n4m_cluster_hdbscan` (since ABI 2.19.0, Methods 1.3.6) could write past
  the caller's buffers: with finite inputs whose squared distances overflow
  (for example `[0, 1e155, 2e155, 3e155]`), Prim produced self-edges and the
  condensed tree had more nodes than rows. Such inputs are now outside the
  numerical domain and refused; inside it squared distances, densities and
  stability sums stay normal doubles. MST edges and tree sizes are validated
  and nothing is written when an invariant fails. `inf - inf` excess at zero
  distance counts as 0 instead of NaN. Sentinel, domain-boundary and
  multi-magnitude regression tests (scikit-learn on exact distances) run
  natively and in WASM.
- `n4m_preflight_adjusted_rand_index` uses exact integer pair counts and
  128-bit numerator and denominator (no cancellation near 0).
- `n4m_linear_sum_assignment` keeps duals, reduced and path costs as exact
  signed 192-bit integers in units of the last mantissa bit of the smallest
  nonzero cost: the result is an exact optimum, also where SciPy's double
  arithmetic is suboptimal (signed costs with spread exponents). Costs are
  never centred. Work limit `rows*cols*min(rows,cols) <= 1e8` exact
  operations; memory `16*rows*cols + 128*(rows+cols)` bytes at most 256 MiB
  (a `1 x 32,000,000` matrix is refused).
- `n4m_random_multivariate_normal` checks storage, Cholesky cost and draws
  before any allocation, including `n = 0`.
- Property tests (`n4m_cluster_robustness_tests`, JS/WASM): domain
  boundaries on both sides, `2^k` sweeps, exact translations, constant
  columns, mixed-exponent assignments against an exact enumeration with an
  independent oracle, memory refusals.
- Internal SVD / symmetric-eigen helpers (`cpp/src/core/common/svd.c`) no
  longer return `N4M_ERR_CONVERGENCE_FAILED`, or silently wrong factors, on
  degenerate or extreme inputs. This affects PCA (`n4m_pca_fit`,
  `n4m_preflight_pca_compute`), PCR, Ridge, MB-PLS, Hotelling T2 / Q
  residuals, FlexiblePCA/SVD and wavelet PCA/SVD.
  - One-sided Jacobi runs on the input scaled exactly into [0.5, 1) and
    sweeps to convergence, with no early stop on norm reduction. Singular
    values are accurate to an absolute O(max(m,n) eps s_max) bound; smaller
    ones are unresolved (weak directions of graded or decoupled inputs are
    usually also resolved relatively, as a bonus). A column cancelled below
    2^-500 of the scaled input stops rotating and is published as 0, inside
    the unresolved range: this makes exactly rank-deficient inputs terminate
    (e.g. +-1 spectra where one row is the opposite of another). Inputs whose
    dynamic range exceeds 2^480 are refused.
  - Tridiagonal QL follows the LAPACK dsteqr organisation: unreduced blocks
    split by the local test, exact per-block scaling of extreme blocks
    (subnormal spectra are resolved) and a 30n total iteration budget.
  - Randomized SVD: the sketch basis is re-orthogonalised and dependent
    sketch columns are dropped.
  - Every helper refuses non-finite input and dynamic ranges above 2^480,
    and reports a significant value it cannot represent (overflow, or below
    2^-1042), with `N4M_ERR_NUMERICAL_FAILURE` instead of a silent zero.
  - On non-degenerate inputs the Jacobi and QL results are bit-identical to
    1.3.6, and all 145 registry parity predictions are bit-identical.
  - Covered by NumPy 2.2.6 / LAPACK goldens
    (`parity/scripts/svd_degenerate_reference.py`) and analytic oracles: the
    review counter-examples, permutations, mixed exponents and both sides of
    the resolution domain.
- `n4m_preflight_pca_compute` declares the supported numerical domain D
  (nonzero |x| in [2^-128, 2^128], spread <= 2^32) and refuses inputs outside
  it with `N4M_ERR_UNSUPPORTED`. Inside D it centres with compensated sums,
  ignores constant columns, and is exact under power-of-two scaling.
  Variances carry an absolute error of order `max(n,p) * eps * lambda_max`
  (no relative accuracy is claimed); components not above the operational
  threshold `16 * max(n,p) * eps * lambda_max` are unresolved and published
  as 0. The header separates that threshold from the total error bound,
  which adds a worst-case Gram accumulation term `L * u * trace(C)` (L = p
  for the row-side Gram) that the threshold does not include.
- The Jacobi convergence threshold no longer underflows for weak column pairs
  (`sqrt(alpha) * sqrt(beta)` when `alpha * beta` is subnormal). Inputs such as
  `B_6(2^-50) (x) [[1,3/4],[1/2,1]]` previously exhausted the 60 sweeps
  inside D.
- PLS regression (`models.pls.pls_regression`, every solver: NIPALS, SIMPLS,
  orthogonal scores, kernel, wide kernel, SVD, power, randomized SVD) and PCR
  (`models.pls.pcr`) without autoscaling are now scale-equivariant. Their
  scikit-learn-compatible absolute `DBL_EPSILON` thresholds refused finite
  data with `N4M_ERR_NUMERICAL_FAILURE` from X 2^-30 or Y 2^-30/2^-60 onwards
  (NIPALS also lost equivariance at Y 2^30, and PCR returned zero Y scores at
  X 2^30). One shared step rescales the centred X and Y by exact powers of
  two to a largest magnitude in [1, 2) and maps every latent array back to
  the caller's units, so fitting X 2^k, Y 2^m scales coefficients by exactly
  2^(m-k) and every weight, loading, rotation and score by its exact power
  of two. On every compared case where they previously succeeded, SIMPLS,
  orthogonal scores, power and randomized SVD are bit-identical; PCR is
  bit-identical except for the stored Y scores it previously zeroed; SVD is
  bit-identical except for multi-target fits whose X-Y covariance had no
  entry of magnitude 1 or more, where its Jacobi stop test now applies its
  relative tolerance instead of an absolute floor (closer to the exact
  singular vectors); kernel and wide kernel move by rounding only (latent
  arrays <= 5e-14 relative). NIPALS no longer adds scikit-learn's absolute
  eps to its weight norm and Y-score denominators: its coefficients move by
  at most a few 1e-12 relative, and its latent arrays now satisfy their
  definitions (unit-norm W, U = Y Q / Q'Q) where that eps distorted them by
  up to ~4e-7 relative when Y was much larger than X. C++ sweeps over
  k, m in {0, +-30, +-60, +-200} (one and two targets) check exact
  equivariance of every fitted array and agreement with scikit-learn 1.7.2
  `PLSRegression(scale=False)` / PCA + linear regression and NumPy analytic
  references (exact-singular-vector deflation, de Jong SIMPLS); a JS/WASM
  sweep checks the same on predictions.
- The same solvers no longer apply absolute `DBL_EPSILON` thresholds to
  derived quantities (X-Y covariance, weight, loading and score norms),
  which the unit scaling alone did not make safe because it bounds X and Y
  but not their covariance. With a = (-1,-1,1,1), b = (-1,1,-1,1), centred,
  unscaled: for X = 2^15 a, Y = 2^25 b + a, NIPALS returned Y scores ~20 %
  too small, and for Y = 2^30 b + a it refused the fit ("Y weights
  vanished"), SIMPLS and PCR returned all-zero Y scores and kernel PLS
  refused; PCR refused X = [2^30 a, b], Y = a + b (singular values 2^31 and
  2) at its second component; every PLS solver refused X = [a, 2^-32 b],
  Y = b. A PLS component is now refused (`N4M_ERR_NUMERICAL_FAILURE`, "X-Y
  covariance is below its numerical resolution") only when
  ||X_k'Y_k||_F <= (n + a) eps ||X_0||_F ||Y_0||_F on the unit-scaled centred
  working matrices, and a PCR component only when its singular value is
  <= (n + p) eps sigma_max (the SVD resolution); above these bounds the
  remaining guards act on unit vectors or exact zeros only (NIPALS keeps
  scikit-learn's 10 eps residual mask only where it leaves the covariance
  resolved; see below). NIPALS, kernel
  and orthogonal scores start the power iteration from a Y column whose
  covariance with X is resolved when their usual start column is not
  (previously a refusal). Kernel PLS forms its raw X scores as X (X'u)
  instead of through the n x n kernel X X', which rounded away X columns
  below ~2^-26 of the largest. PCR leaves the Y-score column of a component
  zero only when its Y loading is below the rounding of Y't, about
  (n + p) eps ||Y||_F sigma_max, and flags that component unresolved (see
  Added). C++ and JS/WASM tests check these cases against analytic W, P, Q,
  R, T, U, coefficients and predictions, on the fitted and on the
  exported/imported model, and the rank-one X = [a, a] second component
  (now a flagged null component, see Changed).
- NIPALS and kernel / wide-kernel PLS decided that the Y residual was
  constant from absolute thresholds before checking the cross-covariance:
  NIPALS zeroed every Y column whose residual stayed below 10 eps
  (scikit-learn's mask) and both refused when no residual entry exceeded
  eps, although many such columns can carry a resolved covariance together.
  With a, b as above, X = [a, b] and Y = a followed by 64 identical columns
  2^-29 a + 2^-51 b (two components, centred, unscaled), the residual
  columns after the first component are 2^-51 b, below the mask, while
  ||X_1'Y_1||_F ~ 1.8e-14 exceeds the resolution ~ 9.4e-15; NIPALS refused
  ("Y residual became constant at component 1"). With the unit vectors
  (-1,-1,1,1), (-1,1,-1,1) and 100 columns 2^-29 a + 2^-52 b, no residual
  entry exceeds eps and NIPALS and kernel PLS both refused. Both solvers now
  decide whether a component is resolved on the intact residuals, apply the
  mask only when the remaining columns keep the covariance resolved, and
  start from the covariance-preferred column when no column exceeds eps:
  every solver now returns the exact coefficients
  [[1, 2^-29 ...], [0, 2^-51 ...]] and predictions = Y (NIPALS also its
  analytic W, P, Q, T, U). The other solvers had no such pre-check. A
  before/after bitwise sweep (3690 fits: random and structured X, Y with
  1-5 targets, scales 2^-60..2^60, autoscaling on and off, every solver and
  PCR) found no change of status, array or prediction.
- The preflight PCA no longer uses the randomized sketch on large truncated
  requests, whose error the published resolution did not cover (a first
  variance 7 % low with `N4M_OK`). It now eigendecomposes the smaller-side
  Gram matrix with the converged QL solver for every n, p and k. The 2.19
  entry point validates `k` before allocating and keeps allocations inside
  its exception barrier.
- Hotelling T² uses only resolved directions and computes its UCL with the
  rank actually used. Before, an exactly collinear column gave a
  rounding-level direction that T² amplified (2.998 instead of 4113/2428).
- Hotelling T² centres with a corrected two-pass mean. The plain rounded
  mean left the same offset on every observation, a spurious direction the
  rank selection kept: ten observations of a constant column 0.1 with k = 1
  gave rank 1, T² = 0.9 and UCL 5.63 instead of rank 0, T² 0 and UCL 0.
  Constant columns now centre to exact zeros. Full-rank results change at
  the rounding level of the centring; no fixed bound on that change is
  claimed (a review measured up to 1.59e-9 relative on a 101 x 3 matrix
  translated by 10^6, where the new values are the closer ones to an
  extended-precision reference: error 3.5e-13 against 2.5e-9 before).
- Hotelling T² is evaluated as `(n-1) * sum_j U_ij^2`, without forming
  sigma_j^2: inputs of magnitude about 2^512 and above gave NaN with
  `N4M_OK`; T² is now exactly invariant under power-of-two scaling. The F
  quantile is the root of the upper tail on a bracket grown by factors of
  two. The former fixed bracket [1e-10, 1e12] silently capped larger
  quantiles (n = 3, r = 1, alpha = 1e-14: UCL 1.47e12 instead of 1.33e14),
  and a UCL that is not a finite normal double is now
  `N4M_ERR_NUMERICAL_FAILURE`.
- Q residuals (`n4m_outlier_detection_q_residuals`) share the Hotelling
  centring and compute Q and the Jackson-Mudholkar UCL at one power-of-two
  scale, rescaled at the end. The residual moments used to overflow or
  underflow: `2^200 * [[2,1],[2,-1],[-2,1],[-2,-1]]`, k = 1, gave the
  correct Q_i = 2^400 but UCL 0 with `N4M_OK` instead of 1.29001e121 (same
  at 2^-200), turning every observation into an exceedance. Q and the UCL
  now scale exactly by 4^s with X scaled by 2^s. When no residual direction
  is resolved (n_components = cols, constant X, numerical rank <=
  n_components) Q and the UCL are exactly 0, the only case with UCL 0 (ten
  rows of 0.1 gave Q = 1.9e-65 next to UCL 0). The upper normal quantile
  takes alpha directly (alpha = 1e-20 gave UCL 0). Unrepresentable results
  and a non-positive Jackson-Mudholkar base (alpha > 0.5 only, formerly UCL
  0) return `N4M_ERR_NUMERICAL_FAILURE`. Tested against analytic
  single-eigenvalue limits and an mpmath oracle on CPU and WASM.
- Q residuals: the Jackson-Mudholkar UCL uses the signed h0. For h0 < 0,
  `(Q / theta_1)^h0` decreases with Q, so the upper quantile of Q is the
  lower quantile of the approximating normal (Nomikos & MacGregor 1995,
  appendix B); `|h0| * A` kept the upper one. Columns 1..14 of the Sylvester
  H_16 times `[2, 1, 1/4 x 12]`, k = 1, alpha = 0.05 (h0 = -0.0676) gave UCL
  0.365 instead of 5.229, flagging all 16 observations with `N4M_OK`. The
  limit is continuous through h0 = 0; h0 > 0 results are unchanged. A
  non-positive base, possible only for h0 > 0 with alpha > 0.5 or h0 < 0 with
  alpha < 1 - Phi(sqrt 2) ~ 0.0786, means the approximation has no
  alpha-quantile: `N4M_ERR_NUMERICAL_FAILURE` (documented, with the
  conditions, in `n4m/outlier_detection.h`).
- Hotelling T²: the UCL is solved directly from its beta tail, inverting the
  small tail (lower for alpha > 0.5) in logarithms. A small quantile went
  through a beta argument that rounds to 1: `X = [-1, 2^-28, 1]`, alpha =
  1 - 2^-30, gave UCL 2.96e-16 instead of 2.312964635e-18 and missed the
  central T² (6.17e-18). The bracket saturates at DBL_MAX: alpha = 1e-308,
  `X = [1, 2, 3]`, returns 1.333e308 instead of `N4M_ERR_NUMERICAL_FAILURE`.
  Log-gamma (Lanczos, 1e-15) and log-beta are more accurate and the
  continued fraction fails explicitly instead of truncating; measured
  relative error below 1e-12 against mpmath up to n = 10^6 on the sampled
  grid; not uniform in the rank (7.2e-12 at r = 127, alpha = 1e-100).
  Tested against
  the closed forms of F(1,2) and F(2,4) at both extremes, on CPU and WASM.
  No other outlier or Preflight kernel inverts a distribution by bisection
  (the x-outlier and MCD chi-square cut-offs are fixed Wilson-Hilferty
  values; Preflight p-values are permutation counts).
- Preflight PCA loadings of a wide input (n <= p): the resolved rows
  `Xc^T u / sigma` are re-orthonormalised (modified Gram-Schmidt applied
  twice; they drifted up to 5e-5 from orthonormality near the resolution),
  and the unresolved rows complete them by Householder QR, so every row is a
  unit vector orthogonal to the others. The former canonical-vector
  Gram-Schmidt kept a candidate only above norm 0.5 and otherwise published
  the last one unnormalised (`X = I_8`, k = 8: last loading of norm 0.354),
  and retried every canonical vector for each row (Θ(n^4): `X = 0`,
  320 x 320, k = 320 took about 10 s in WASM).
- The preflight PCA checks two guaranteed bounds before allocating, listed
  pass by pass in `cpp/src/core/diagnostics/preflight_pca_budget.hpp`. Work
  counts every long-double multiply-add (sum of squares, Gram, projections)
  and every double pass at 1/16, with the QL iteration at the solver's
  ceiling (30 s sweeps), against 2.4e9. Memory counts every buffer live at
  the peak against 1 GiB. These are bounds, not cost estimates (a square
  s x s input typically costs about 1.1 s^3 against a bound of 8 s^3). The
  former estimate omitted the sum of squares and the column exponents and
  took two QL sweeps per eigenvalue: n = 3, p = 26e6, k = 1 was accepted
  although it needs 1.144 GB, and is now refused. Every shape of the
  Preflight profile (at most 10^7 cells, k <= 10) that Methods 1.3.6 or the
  former estimate accepted is still accepted (enumerated in the tests); the
  profile alone is not sufficient, since every request must also satisfy the
  work and memory bounds (1000 x 1000 with k = 10, or 700 x 700 with
  k = 700, is refused).
  The limit is a working bound on the computation, not a browser performance
  qualification: review measurements of the WebAssembly kernel alone (`-O3`,
  Node 22 / V8, Intel i9-12900K) took 6.27 s for 658 x 658, k = 1, and 12.48 s
  for 600 x 600, k = 600 (outside the Preflight profile k <= 10). A browser
  profile needs its own execution budget (p50/p95 timings, inter-module
  copies, cumulative memory) before qualification.
- PLS-QDA (`n4m_estimators_pls_qda_fit` and the `pls_qda` classifier role)
  now takes its latent space from the qualified PLS-DA SIMPLS solver
  (centred, unscaled X, unit-norm scores) and fits the class Gaussians on
  the resolved components only. Its private SIMPLS used absolute 1e-12
  thresholds: with x = (-4,-3,-2,-1,1,2,3,5), X = [x, x], classes
  (0,0,1,0,1,1,0,1), two components were refused ("a PLS-QDA class
  covariance is not positive definite": the second component of the rank-one
  X was rounding noise), and after multiplying X by 2^-50 one component was
  refused too (no component computed). The second component is now a
  flagged null component of the solver: zero rotation column, zero
  class-mean and covariance rows / columns, left out of the discriminant
  (log class priors only when no component is resolved). The decision,
  -0.5 ((x_i - m_c)^2 / v_c + log(v_c / S)) + log(1/2) (m = (-1.25, 1.5),
  v = (28.75, 25) / 3, S = 68.875), is now the same for one and two
  components and for X 2^e, e in {0, -50, 50, -200, 250} (C++ and JS/WASM
  tests, with the class means, covariances and rotations). Sweep against
  the previous kernel (60 random fits): one refusal now fits, decisions
  move by <= 1.6e-9 relative (3.6e-13 on the role fixture, within its
  replay tolerance). The shared SIMPLS helper of the other extra PLS
  estimators (sparse PLS-DA legacy, group sparse, CPPLS, weighted, ridge,
  continuum, DS, MIR-PLS, missing-aware NIPALS, approximate PRESS,
  bagging, boosting, random subspace) no longer uses scale-dependent
  absolute thresholds either: a component is computed while the deflated
  covariance is above (n + a) eps ||X||_F ||Y||_F, the other guards test
  exact zeros (the Gauss-Jordan pivots of P'W, which is dimensionless, keep
  their 1e-12). Same sweep: random subspace on X 2^-31 used to stop early
  (predictions 4 % off those of the unscaled fit, now equal); bagging,
  boosting, MIR-PLS and missing-aware NIPALS move by <= 1.3e-7 relative on
  5 or fewer of 60 fits, the others are bit-identical.
- PLS diagnostics (`n4m_metrics_pls_diagnostics_compute`) no longer
  invent a reference scale for DModX. When the reference residual vanished
  (sum Q_ref = 0) sigma_ref was replaced by 1: with a = (-3,-1,1,3),
  b = (1,-3,3,-1), a model fitted on X_ref = [a, 0], Y = a (one component,
  centred, unscaled) and X = [a, b] evaluated against X_ref, DModX was
  (1, 3, 3, 1), and (9.3e-10, 2.8e-9, ...) after multiplying both X by 2^-30,
  i.e. the unnormalized residual in the caller's units; orthogonal scores,
  whose reconstruction leaves a rounding-level residual, returned ~2.3e15.
  The reference residual is now degenerate when its norm sqrt(sum Q_ref) is
  not above (n_ref + n_features + k) eps ||X_ref||_F in model space: "dmodx"
  is NaN on every row and the new scalar `dmodx_reference_degenerate` is 1
  (0 otherwise); Q and T2 are unchanged. Every other DModX is bit-identical
  (the `pls_diagnostics` role fixture gains only the new scalar). No other
  substituted scale or absolute floor remains in the diagnostics and
  monitoring kernels (the monitoring thresholds are empirical quantiles of
  T2 and Q). C++ and JS/WASM tests, every PLS solver, X_ref = 2^e [a, 0],
  e in {0, -30, 200}: NaN and flag 1, Q = 2^2e b^2, T2 = 3 a^2 / 20; with
  X_ref = 2^e [a, 2^-20 b]: DModX = 2^20 |b| / sqrt 10, flag 0.
- VIP and the selectivity ratio (`n4m_feature_selection_variable_select_rank`
  methods 0 and 2, also used by VIP-SPA) no longer depend on the units of X
  and Y. With a = (-3,-1,1,3), b = (1,-3,3,-1), centred, unscaled, one
  component: VIP skipped every weight column whose sum of squares was below
  eps, so SIMPLS (whose stored weights scale as 1 / ||X||) on X = 2^30 [a, b],
  Y = a returned VIP (0, 0) with status OK instead of (sqrt 2, 0) (same at
  2^254), and it refused NIPALS on X = [a, b], Y = 2^-30 a ("Y variance
  explained by components vanished") because sum_a ||t_a||^2 ||q_a||^2 was
  below eps. The selectivity ratio divided by max(residual, eps): with
  X = s [a + b, a - b], Y = a (explained and residual sums both 20 s^2, ratio
  1) it returned 0.078125 at s = 2^-30 and ~6.7e-138 at s = 2^-256. VIP now
  forms SSY_a / sum SSY and w_ja / ||w_a|| from scaled norms with no
  threshold and leaves out only components that are flagged unresolved or
  have an exactly zero weight, score or Y-loading column (refusal only when
  none remains: VIP undefined). The selectivity ratio divides the two sums
  accumulated on the column scaled by the exact power of two of its largest
  value; an exactly zero residual gives +inf (0 for a column that is zero in
  model space), as documented in `feature_selection.h`. A before/after sweep
  (300 random fits x NIPALS, SIMPLS, kernel, SVD, PCR, VIP and SR, 3000
  rankings): 90 VIP refusals now succeed; 71 VIP results that were all zero
  and 15 that dropped components with small SIMPLS/kernel weights are now
  correct; 545 SR results that hit the floor change (residual below eps, e.g.
  scaled data or as many components as the rank of X); every other VIP moves
  by <= 6e-16 relative and every other SR by <= 5e-14 (residual just below
  eps) or not at all. C++ and JS/WASM tests: VIP (sqrt 2, 0) for
  X = 2^e [a, b], Y = 2^f a, e in {0, 30, 254, -200}, f in {0, -30, -200, 200};
  (sqrt 1.6, sqrt 0.4) for Y = 2^f [2 a, b] with two components; (1, 1) for
  X = [a, a] with a null second component; SR (1, 1) for s = 2^e,
  e in {0, -30, -256, 30, 250} (every PLS solver); SR (+inf, 0) for NIPALS on
  X = [a, 0].
- T2-PLS variable selection (`n4m_feature_selection_t2_select`) works on
  models with null components. With a = (-3,-1,1,3), X = [a, 2a, 3a, 4a],
  Y = a and two components, the zero second weight column made the weight
  covariance singular and the selection returned `N4M_ERR_INVALID_ARGUMENT`.
  The weight T2 now uses the m components with a nonzero weight column and
  the limit (p - 1)^2 / p Beta_{1-alpha}(m / 2, (p - m - 1) / 2), so scores,
  limits, masks and CV RMSE equal those of the model truncated to its
  resolved components (here T2 = (1.35, 0.15, 0.15, 1.35), limit
  2.25 (1 - alpha)^2); with m = 0 every T2 and limit is 0. The result gains
  the scalar `n_latent_directions` (JS/WASM tests cover the diagnostics,
  PLS-LDA and T2-selection cases above). Its singularity test was an absolute
  1e-14 on the pivots of the weight covariance, whose scale follows the
  solver (W ~ 1 / ||X|| for SIMPLS): SIMPLS on unscaled X of magnitude
  ~2^18 or more was refused; the test is now relative to the largest
  diagonal entry. A before/after sweep (150 random selections x NIPALS,
  SIMPLS, alpha 0.001-0.75) is bit-identical on the 249 that ran before; the
  7 refused ones (SIMPLS, X ~ 2^18-2^20) now run.
- The T2-PLS selection limit inverted the Beta CDF against 1 - alpha by
  bisection on [0, 1] with lgamma and a complement: 1 - alpha rounding to 1
  made the quantile ~1 (alpha = 1e-20, Beta(1, 50): 0.6019 expected), and the
  small lower tail was lost for alpha close to 1. As for the Hotelling T2
  outlier limit, the small tail (upper for alpha <= 1/2, lower otherwise) is
  now inverted in logarithms on the odds x / (1 - x), with a Lanczos ln B
  free of large cancellations and a continued fraction that reports
  non-convergence (`N4M_ERR_NUMERICAL_FAILURE`), by bisection on the normal
  doubles with midpoint lo + (hi - lo) / 2. Internal tests check
  Beta(1, b) and Beta(a, 1) closed forms to 1e-12 relative for alpha from
  1e-300 to 1 - 2^-40 and shapes 0.5 to 5000.

- PLS-LDA (`n4m_estimators_pls_lda_fit` and the `pls_lda` classifier role)
  fits its discriminant on the resolved PLS-DA components. With
  a = (-3,-1,1,3), X = [a, a], classes (0, 0, 1, 1) and two components, the
  null second component's zero score column made the pooled within-class
  covariance singular (diag(4, 0) with NIPALS) and the fit was refused
  ("failed to invert PLS-LDA pooled covariance"). Identically zero score
  columns now get zero discriminant weights at their positions, so
  decisions and predictions are bit-identical to the head fitted on the
  resolved components (here log(1/2) - 1 -+ a_i); with no resolved
  component the decision is the log class prior (prediction: most frequent
  class, lowest id on ties). PLS-logistic (ridge-regularised Newton)
  already gave the truncated fit with a zero coefficient; it is now tested.

- PLS diagnostics (`n4m_metrics_pls_diagnostics_compute`) and monitoring
  (`n4m_metrics_pls_monitoring_run`) accept models with null components.
  Hotelling T2 required a positive reference variance on every component,
  so with a = (-3,-1,1,3), b = (1,-3,3,-1), X = [a, a, b], Y = a, two
  components (centred, unscaled) it refused the whole result ("Hotelling T2
  reference variance is zero at component 1"), and DModX used
  n_features - n_components degrees of freedom (a factor sqrt 2 too large
  without reference). T2 now sums over the model's latent directions, the
  components with a nonzero rotation column: a PLS component flagged
  unresolved is a null component without X direction, while a PCR component
  flagged only for its Y side keeps its resolved X direction. DModX uses
  n_features - k and n_reference - k - 1 degrees of freedom for k latent
  directions; with k = 0 (no resolved component) T2 is 0 on every row. A
  model with null components now has exactly the diagnostics of the model
  truncated to its resolved components. The result gains the scalar
  `n_latent_directions` (k).
- PLS diagnostics and monitoring now map the rows into the model's X space
  as `n4m_model_transform` does (fitted SNV/Savitzky-Golay pipeline, then
  x_mean and x_scale). They used to project the centred but unscaled rows
  onto rotations fitted on autoscaled X, so with `scale_x = 1` (the default)
  T2 and DModX were wrong (X = [a, 2 b], Y = a, one component: T2 = a^2
  instead of 3 a^2 / 20) and Q mixed raw and scaled units. Unscaled models
  without pipeline are unchanged up to rounding. The `pls_diagnostics` and
  `pls_monitoring` outputs of the cross-language estimator-role fixture
  (`parity/fixtures/estimator_roles_n4me.json` and its R copy), computed with
  the default autoscaling, are regenerated; every other entry is unchanged.

- Power-type PLS iterations no longer fail, or stop on a wrong direction,
  when the two leading singular values of X_k'Y_k are close. NIPALS,
  canonical NIPALS, orthogonal scores, kernel, wide kernel, power, randomized
  SVD and SIMPLS (also used by PLS-DA and di-PLS) contract their error by
  about (sigma_2 / sigma_1)^2 per step. With a = (-3,-1,1,3), b = (1,-3,3,-1),
  X = [a, b], Y = [a + d b, d a + b], d = 2^-12 (centred, unscaled, one
  component), X'Y = 20 [[1, d], [d, 1]] has the singular values
  20.0048828125 and 19.9951171875: SIMPLS, and every iterative solver at
  tol 1e-12, returned `N4M_ERR_CONVERGENCE_FAILED`, and at the default tol
  1e-6 NIPALS, kernel, wide kernel, power and randomized SVD returned
  status OK with the weights (1, 0) instead of (1, 1) / sqrt 2 (coefficients
  off by 1/2 instead of (1 + d) / 2 everywhere). With several targets, each
  of these solvers now keeps a converged iterate only when it is proven
  within 16 times its nominal step tolerance (sqrt(tol) for NIPALS,
  canonical NIPALS, kernel, power and randomized SVD, tol for orthogonal
  scores, min(max(tol, 1e-14), 1e-12) for SIMPLS) of the dominant singular
  vector of the component's cross-covariance C, by an a posteriori
  Davis-Kahan bound: sin(angle) <= ||M x - l x|| / (lambda_1 - lambda_2) for
  M = C C' (or C'C), the Rayleigh quotient l <= lambda_1 and
  lambda_2 <= min(trace M - lambda_1, sqrt(||M||_F^2 - lambda_1^2)), every
  term widened by a bound on its rounding error (O(p q) work, plus
  O(m^2 max(p, q)) for ||M||_F, m = min(p, q), only when the trace bound
  does not suffice). Otherwise (no convergence, close or unproven leading
  singular values) the dominant pair is computed from the Householder
  tridiagonalisation and implicit QL of the smaller Gram matrix of C
  (O(m^3), no spectral-gap dependence) and used when the iteration did not
  converge or stopped farther than the same distance from it. The step size
  alone is never taken as proof: in the d cases the bound fails and the
  reference pair is used. One-target fits converge in one step and are
  unchanged. A bitwise before/after sweep of the 3690 fits of the sweep
  below changes two cases only: a random 3-target fit with
  sigma_2 / sigma_1 = 0.98, whose first-component weights were 0.018-0.023
  away from the dominant singular vector for NIPALS, kernel, wide kernel and
  power (2.4e-15 now, against a 50-digit SVD; SIMPLS failed to converge and
  now fits; coefficients and predictions move by <= 8e-16 relative since
  that fit uses all components), and the exactly degenerate X = [a, b],
  Y = [a, b] (sigma_1 = sigma_2), where randomized SVD now returns the first
  coordinate direction of the reference pair instead of its random one
  (coefficients unchanged). Cost (median of 5 fits, one component,
  n = 256, p = 128, before the iteration control -> now): 1 target
  unchanged (0.3 ms), 8 targets 0.6-2.7 ms unchanged, 128 targets
  5.5-13 ms -> 12-20 ms (5 components: 21-76 ms -> 50-108 ms). The
  certificate costs one matrix-vector product pair; the reference pair,
  when needed, O(m^3). C++ and JS/WASM tests check the analytic arrays of
  the d = 2^-12, 2^-20, 2^-30 cases for every PLS solver and PLS-DA, at
  tol 1e-6 and 1e-12, the canonical weights, and the coefficients and
  predictions of X = [a, b, c] with c = (1, -1, -1, 1) and 64 targets
  (Y = [a + d b, d a + b, s_j c], X'Y with singular values 20 (1 +- d) and
  ~2.9; coefficients (1 + d) / 2 in the leading 2 x 2 block, 0 elsewhere).

## [1.3.6] - 2026-10-13

ABI 2.19.0. No model-state serialization format changes.

### Added

- Public ABI 2.19.0 generic clustering: `n4m_cluster_hdbscan`, a
  Euclidean HDBSCAN* ported from scikit-learn 1.9.1 (BSD-3) with GLOSH outlier
  scores and cluster persistence from scikit-learn-contrib/hdbscan 0.8.44,
  returning labels, membership strengths, outlier scores, persistence and the
  condensed cluster tree into caller buffers; and
  `n4m_preflight_adjusted_rand_index`. Equal-weight MST edges are stably
  sorted, a documented tie rule. A frozen reference gate
  (`parity/fixtures/hdbscan_v1.json`, `n4m_hdbscan_tests`) covers separated
  blobs, varying densities, pure noise, a continuum, duplicates/ties and
  `min_samples` different from `min_cluster_size`, with EOM, leaf and
  `allow_single_cluster`. Both calls are exported through WASM and thin
  TypeScript wrappers (`clusterHdbscan`, `preflightAdjustedRandIndex`).
- Public ABI 2.19.0 Preflight diagnostics: a Savitzky-Golay smoothing-residual
  noise proxy (interior features only, masked windows skipped, normalized by
  the filter's white-noise residual gain; never an SNR) and row pairs whose
  root-mean-square feature difference is within a caller-declared tolerance,
  with group exclusion, exact totals, per-row flags and a bounded pair list.
  Both reuse existing Methods code (the Savitzky-Golay engine; matrix-view
  validation) and are exported through WASM with thin TypeScript wrappers.
  Binding ABI constants and README examples move to 2.19.0.
- Public ABI 2.19.0 Preflight association kernels:
  `n4m_preflight_factor_association` (eta^2 and one-way F, or Cramer's V and
  Pearson chi-square, over exchangeable units, with a seeded Monte Carlo
  permutation p-value `(1+b)/(1+B)` and a log10 bound on distinct permutation
  outcomes) and `n4m_preflight_holm_adjust`. Permutations reuse the PCG64
  engine through a new NumPy-parity `random_interval`/`permutation` helper, so
  `numpy.random.default_rng(seed).permutation(n)` reproduces every draw. Thin
  TypeScript wrappers `preflightFactorAssociation` and `preflightHolmAdjust`.
- Native and WASM cases compare hand-computed effects and NumPy 2.2.6 / SciPy
  1.15.3 permutation counts recorded as test-only goldens.

The project and npm package version is prepared as 1.3.5. No model-state
serialization format changes.

## [1.3.5] - 2026-10-12

ABI 2.18.0, published to npm only. No model-state serialization format changes.

### Added

- Public ABI 2.18.0 diagnostics for Preflight: centered PCA scores, loadings,
  retained eigenvalues and exact full covariance moments; masked pooled
  repeatability with balanced ICC(1,1); finite summaries with quartiles; and
  group-excluding Euclidean neighbors and local target contrast. The same five
  calls are exported through
  WASM and thin TypeScript wrappers.
- Native scientific cases for truncated PCA moments, negative and unbalanced
  ICC, strided views, quantiles, neighbor exclusions, and finite extreme-value
  summaries and distances. Scaled variance and distance calculations retain
  representable f64 results and reject unrepresentable results explicitly.

## [1.3.3] - 2026-10-06

ABI 2.17.0 and the independently published Rust crate n4m 0.4.0 are unchanged.
This patch publishes the Python mixed-metadata replay correction; existing
1.3.2 artifacts and tags are preserved.

### Fixed

- Accept object and fixed-width text storage for raw mixed metadata without
  casting categories to a saved NumPy text width. Preserve the captured native
  source descriptor and reject non-text categorical cells during fit and replay.

## [1.3.2] - 2026-10-04

ABI 2.17.0 and Rust crate n4m 0.4.0 remain unchanged. Existing 1.3.1
artifacts and tags are preserved.

### Fixed

- Import the standard Self annotation through its backport on Python 3.10,
  with an explicit conditional dependency in generated wheel metadata.
- Make fixed-fixture index and floating-point conversions explicit in the
  multimodal C++ tests under the complete strict GCC 12 build.
- Bound R optimizer tuple component formatting with snprintf.

## [1.3.1] - 2026-10-04

Native ABI 2.17.0 and the independently published Rust crate n4m 0.4.0 are unchanged.
This patch supersedes the 1.3.0 source and npm release without replacing its assets.

### Fixed

- Give private multimodal helpers internal linkage for strict GCC builds.
- Classify all 13 multimodal classifier lifecycle symbols as ABI infrastructure.
- Generate the StandardScale scientific documentation from its maintained native
  contract, including train-only population statistics and portable state.

## [1.3.0] - 2026-10-04

Rust crate `n4m` is released separately as 0.4.0. The native ABI is 2.17.0;
regression state formats and existing compatible replay profiles are preserved.

### Added

- Native multimodal encoding and PLS-logistic classification, with explicit
  class vocabularies, probability outputs and validated N4MC fitted states.
- Compile-time build capabilities for admission of the documented sequential
  numerical profile and independent pipeline lifecycle support.
- Matching complete/slim Python, R and JS/WASM facades for the qualified
  multimodal fit, state export/import and prediction profiles.

### Fixed

- Overflow-safe classifier working-set admission before native fitting.
- Generated Python classifier declarations and binding ABI constants now track
  the actual 2.17 native surface.

## [1.2.1] - 2026-09-27

Rust crate `n4m` stays 0.3.0: it links the libn4m that carries these fixes.
ABI unchanged (2.14.0): input validation and facade contract fixes from the
2026-09-27 re-audit (findings R03, R05, R06, R07, R08, R10, R12, R13).

### Fixed

- The estimator C ABI read a float32 view as doubles, past its declared
  extent: `n4m_estimator_fit`, `n4m_procedure_run` and every estimator
  operation now validate each input matrix (X, y, target domain, new rows,
  mask target: layout and float64 dtype) before dispatch and refuse another
  dtype with `N4M_ERR_DTYPE_MISMATCH` (R03).
- A role pipeline predicting zero rows returned `NULL_POINTER` when a
  transformer preceded the final step; zero rows now give an empty output of
  the right width whatever the steps (R13).
- Shared label contract in the Python, R and JS facades (pipelines and the
  estimator classifiers that keep label names): integer labels must fit int64
  (a uint64 2^63 was wrapped to a negative id in Python; a Python list mixing
  small ints with 2^63 no longer turns into floats); missing, non-finite and
  boolean labels are refused at fit (Python fitted NaN as a class); strings
  and non-integer numbers become a sorted label table in every facade (JS and
  R now accept non-integer numbers, as Python did). An imported label table
  (`class_names`) must be non-empty, unique, strings or finite numbers, and
  cover every native class id (`n4m_role_pipeline_classes`); an empty, short,
  duplicated or non-matching table used to import and then fail or merge
  predictions (R05, R08, R12). A table is all strings or all numbers, and its
  numbers must be exact as float64: an integral number beyond ±2^53 (in a table, or
  beside fractions at fit) is refused instead of being rounded into another
  label. R predictions keep numeric labels as numbers (string labels stay a
  factor), and probability column names keep close fractions distinct.
- Column names containing NUL are refused at fit, import and predict in
  Python and JS; they were truncated at the NUL, so a renamed column passed
  the identity check (R10).
- R `n4m_role_pipeline` fit converts its auxiliary inputs (weights, groups,
  fold ids, feature groups, blocks, axis, target domain) as strictly as the
  estimators: a matrix is no longer flattened into another length (R06).
- The legacy R dispatcher (`n4m_method`, also in the `pls4all` R subset)
  refuses fractional, NA, non-finite and out-of-range class ids, groups,
  block sizes and event indicators instead of truncating them; `mb_pls`
  block sizes likewise (R07). R integer inputs and Python integer inputs
  (groups, fold ids, feature groups, blocks) must fit int64 exactly.
- `parity/fixtures/role_pipeline_negative.json` gains the facade-level
  `label_cases` and `name_cases`, replayed by the Python, R and JS suites
  (Rust replays the name cases; it takes `i64` class ids and holds no table).

## [1.2.0] - 2026-09-27

Rust crate `n4m` 0.3.0 ships the same API (`n4m::roles::RolePipeline`, `Estimator::contains_training_rows`).

### Added

- ABI 2.14: native role pipeline (`n4m_role_pipeline_*` in
  `n4m/estimator.h`). One owner for the portable trained recipe of estimator
  steps (sample filters, transformers / selectors, one regressor or
  classifier): recipe validation, fit-input routing (multi-target Y reaches
  supervised transformers, filters subset every row input), stored feature
  names with a width/name/order check, per-step N4ME export with an explicit
  training-row opt-in, and an import that refuses states contradicting the
  recipe (method, resolved parameters, role, widths). Facades: Python
  `n4m.roles.RolePipeline`, R `n4m_role_pipeline()`, JS `RolePipeline`, Rust
  `n4m::roles::RolePipeline`; a shared negative and cross-language fixture
  (`parity/fixtures/role_pipeline_negative.json`) is replayed by the four
  suites.
- ABI 2.14: `n4m_estimator_contains_training_rows`, and a `recorded` flag on
  `n4m_param_info_v1_t` and in the manifest JSON for the parameters a fitted
  state records. Every binding exports a state that embeds training rows only
  with an explicit opt-in (Python `allow_training_rows=True`, R
  `allow_training_rows = TRUE`, JS `{ allowTrainingRows: true }`, Rust
  `to_n4me(&ctx, true)`); JS no longer passes it implicitly.
- Shared negative fixture `parity/fixtures/estimator_roles_negative.json`
  (`scripts/generate_estimator_negative_fixture.py`), replayed by the Python,
  R, JS/WASM and Rust suites: mismatched target and per-row input lengths, a
  transposed target, an N4ME payload with a contradicting parameter, an export
  of training rows without opt-in, a failed classifier refit.

### Changed

- Seeds (`seed`, `noise_seed`, `randomization_seed`) are optional parameters:
  the manifest publishes no default (None / NULL / undefined in the facades)
  and an unset seed runs as 0, so default results are unchanged; an N4ME state
  records the seed it ran with.
- A failed `n4m_estimator_fit` keeps the previous fitted state instead of
  leaving the estimator unfitted.

### Fixed

- R recycled targets, weights, groups and labels to the rows of X
  (`y = 1:20` for 40 rows was accepted): every per-row input must now have
  exactly one entry per row, a matrix target one row per row of X, and scalar
  parameters exactly one whole value (audit F01).
- Python reshaped a target by its element count (`(2, 20)` for 40 rows was
  accepted): `y` must be `(n,)` or `(n, q)`, and per-row / per-column inputs
  1-D of the matching length; JS checks the same lengths before the native
  call, and the core refuses them for every binding (audit F02).
- N4ME import refused corrupted bytes but accepted parameters contradicting
  the restored state (a PLS header claiming one component for a
  two-component state): each adapter family now checks the parameters its
  state records (audit F04).
- `n4m_estimator_transform` / `_predict` / `_decision_function` /
  `_predict_proba` wrote through an output view of the wrong width (past its
  end for a narrower one); the view must now be F64 with exactly X rows and
  the operation's width, else `N4M_ERR_SHAPE_MISMATCH` before any write.
- A failed Python classifier refit replaced the class names before the fit
  failed, breaking the previous model; label names, output shape and feature
  names are now published with the new state only on success (audit F08).

- The R packages generate `src/Makevars.win` from `configure.win` with
  `-ffp-contract=off`, like the Unix `configure`: Windows arm64 (clang) fused
  multiply-adds and AsLS / IAsLS drifted from the other platforms.

## [1.1.1] - 2026-09-27

### Fixed

- The R packages `n4m` and `pls4all` are versioned 1.1.1: the 1.1.0 bump kept
  the R-universe development suffix (1.1.0.9006). `scripts/bump_version.sh`
  now drops that suffix on a bump.
- The shared estimator-role fixture uses a full-rank transfer target; with a
  rank-2 target `utilities.transfer_metrics` compared null-space directions,
  which made the macOS arm64 R check fail. No library code changed.

## [1.1.0] - 2026-09-27

### Added

- ABI 2.13: generic typed role interfaces for the catalog methods
  (`n4m/estimator.h`). A native manifest (`n4m_method_*`,
  `n4m_method_manifest_json`), named typed parameters with core-owned
  defaults (`n4m_params_*`), one estimator life cycle (`n4m_estimator_*`:
  transformer, selector, regressor, classifier and sample-filter roles) with a
  portable fitted state (N4ME), and one call for procedures
  (`n4m_procedure_run`: splitters, augmenters, diagnostics, utilities).
  Python `n4m.roles` (scikit-learn classes), R `n4m_<method>()` constructors
  with S3 role classes, JS/WASM classes and the Rust `n4m::roles` module are
  generated from or driven by the manifest; a shared fixture replays every
  fitted state and procedure identically in the four bindings. See
  `docs/abi/estimator_roles_design.md` and
  `docs/parity/estimator_roles_coverage.md`. 208 of the 213 catalog entries
  are covered (148 estimators, 60 procedures); the other five are
  compositions (AOM screen/refit, staged campaign, linear Ridge stack, moment
  stack, composite filter) that belong to DAG-ML.
- Native kernels for methods that were Python-only: the six AOM superblock
  regressors (Ridge, MKL, active, PLS, Ridge-PLS, chain Ridge-PLS), PLS-GLM
  (plsRglm algorithm; Gaussian, Poisson, binomial) and PLS-Cox (deviance
  residuals, Breslow ties; Y = time, event). All match their references to
  1e-12.
- Rust crate `n4m` 0.2.0 with the `n4m::roles` module.

### Changed

- The draft ABI 2.12 `n4m_sample_filter_*` / `n4m_feature_filter_*` surface,
  never released, is replaced by the sample-filter and selector roles.
- Kernel fixes found while adding roles: PLS-QDA now applies the fitted
  covariances; sparse PLS-DA honours `sparsity_lambda`; SO-PLS / ROSA return
  affine predictors on raw blocks; `n4m_estimators_pls_glm_fit` and
  `n4m_estimators_pls_cox_fit` now fit the real PLS-GLM / PLS-Cox models
  (they ran plain SIMPLS); missing-aware NIPALS imputes in-sample; bagging / random-subspace
  PLS draw portably (fits change once); alignment refits relearn their
  reference; the cubic resampler no longer shares scratch space; the AOM
  calibration kernel reports infeasible candidates by status (WASM builds
  have no exception catching). Details in `docs/abi/changes_log.md`.
- `augmentation.splines.spline_smoothing` runs a C translation of the FITPACK
  routines on every platform, equal bitwise to the former Fortran build and
  to scipy's `UnivariateSpline`; Windows, WASM and CRAN R builds no longer
  fall back to the Reinsch approximation, and gfortran is no longer needed.
- GCC and Clang builds (CMake and the R packages on Unix) compile with
  `-ffp-contract=off`: arm64 builds no longer fuse multiply-adds, so every
  platform returns the same bits as x86-64 and WASM.

- Expose native domain-invariant PLS as `n4m.domain_adaptation.invariant.di_pls`
  and the scikit-learn-compatible `DIPLS` estimator. Both require an explicit
  unlabeled target-domain matrix at fit time; the fitted affine predictor
  retains no target cohort. Tighten the existing `pls4all.sklearn`
  DI-PLS wrapper's target-feature validation and failed-refit behavior.

## [1.0.21] - 2026-09-21

### Fixed

- Aligned the R package ABI regression test with the released ABI 2.6 surface,
  restoring the Linux, macOS ARM, and Windows CRAN-check matrix.
- Integrated the ABI 2.6 AOM calibration and affine-stack surfaces into the
  generated scientific documentation coverage gate (212/212 methods).

## [1.0.20] - 2026-09-21

### Added

- ABI 2.6.0: native branch-aware AOM calibration fit/predict and affine stack
  compression (`n4m_model_selection_aom_calibration_fit`,
  `n4m_model_selection_aom_calibration_predict`,
  `n4m_ensemble_linear_stack_compress`). Existing ABI symbols remain available.
- Python AOM/FastAOM PLS and Ridge calibration facades, and a nested-OOF linear
  Ridge stack with an independent one-vector-plus-intercept export.
  See `docs/methods/aom_calibration.md` for versioned numerical contracts.
- Stack training shares a smaller-Gram Ridge eigenpath across penalties on wide inputs, preserves pooled-CV selection, and falls back to the native direct Ridge solver on numerical convergence failure.
- Optional detected LAPACKE acceleration for the new AOM Ridge eigenpath
  (`N4M_AOM_WITH_LAPACKE`), retaining the portable backend when unavailable.
- Optional Gaussian descriptor normalization flag; historical descriptors
  remain normalized. The strict10 bank explicitly requests unnormalized kernels.

### Documentation

- Distinguish native L2-augmented Ridge-PLS from Ridge on PLS scores. Historical
  entry points retain their algorithms rather than silently changing semantics.

## [1.0.19] - 2026-09-06

### Fixed

- Publish the ctypes-only `nirs4all-methods` binding as portable
  `py3-none-<platform>` wheels and validate the same native payload on CPython
  3.10 through 3.14, including the `nirs4all==1.0.1` installation path.

## [1.0.18] - 2026-09-05

### Fixed

- Preserve the published AOM forced-moment behavior on wide spectra while
  retaining the 1.0.17 FFI ownership, package-import, and loader-indirection
  corrections.
- Expose the matching configurable AOM chain sweep through the JavaScript/WASM
  binding so the corrected native and browser surfaces ship together.

## [1.0.17] - 2026-09-05

### Fixed

- Preserve the runtime library indirection used by Python packages instead of
  baking a build-machine path into generated loaders.
- Keep public FFI declarations synchronized with the authoritative C ABI and
  validate generated package imports and serialized-pipeline inspection.
- Tighten release-surface metadata checks so the full and slim Python packages
  remain independently installable from the same native engine source.

## [1.0.15] - 2026-09-03

### Added
- **Bounded N4MM pipeline payload.** PLS regression can now serialize and
  replay the exact `SNV -> Savitzky-Golay smooth -> PLS` product profile as
  N4MM format 2. The semantic profile fixes row-wise SNV with `ddof=0` and SG
  edge handling to `mode=interp`; models without this pipeline remain
  byte-compatible N4MM format 1.
- **Typed N4MM pipeline inspection (ABI 2.5.0).** Added a sized, versioned C
  descriptor and matching Rust/Python/JavaScript surfaces for the validated
  operator order, canonical Savitzky-Golay parameters, raw/model widths and
  stable plan fingerprint. N4MM v1 reports no pipeline; malformed v2 remains
  fail-closed with zero output.
- **Authoritative model inspection (ABI 2.4.0).** Added an allocation-free
  descriptor derived from complete validated N4MM bytes, including algorithm,
  dimensions and native capability bits. Consumers no longer need to trust
  filenames or host metadata to decide whether a model can predict.

## [1.0.13] - 2026-08-25

### Added
- **Public Python migration export.** The slim, publicly distributed
  `pls4all` binding now exposes `export_linear_predictor_n4mm(...)` for a
  caller that has already verified a finite affine predictor equation.  It
  builds the ABI-2.3 imported-linear-predictor N4MM without loading a foreign
  pickle/joblib payload or retraining, and documents the PREDICT-only limit.

## [1.0.12] - 2026-08-25

### Fixed
- **R binding ABI contract.** Updated the locked ABI namespace assertion to
  2.3, matching the additive ABI used by the released N4MM affine-import
  surface.

## [1.0.11] - 2026-08-25

### Added
- **Verified affine predictor import (ABI 2.3.0).** Added
  `n4m_model_import_linear_predictor` and matching Python/Rust binding support
  to construct an N4MM from an already-attested finite affine equation.  The
  imported representation supports native `PREDICT` and N4MM round-tripping,
  but explicitly refuses latent `transform`: it does not claim to reconstruct
  a source PLS decomposition and never deserializes joblib/pickle or retrains.
- **Owning HPO lifecycle trace (ABI 2.2.0).** Added terminal
  `N4M_TRIAL_CANCELLED`, structured and versioned trial errors, immutable
  terminal/intermediate replay semantics, optimizer-global event sequences,
  and a rich owning `n4m_optimizer_get_trials` snapshot containing exact ids,
  typed ordered parameters, activation/category metadata, intermediates, and
  errors. Text persisted into the trace is now validated as UTF-8; failed pruner
  evaluation rolls back its staged intermediate, and failed asks do not consume
  queued warm-start payloads. Python exposes decoded `TrialRecord` snapshots through
  `Optimizer.get_trials(since_id=...)`.
- **Portable optimizer checkpoint/resume (N4MOPT v1).** Activated the existing
  `n4m_optimizer_save/load` ABI reservations with a versioned little-endian,
  checksummed, 64 MiB-bounded format covering the ordered space/options, complete
  lifecycle and enqueue state, RNG, and all nine samplers/five pruners. Python
  exposes `Optimizer.save() -> bytes` and transactional `Optimizer.load(bytes)`;
  random and TPE continuation tapes plus the full native sampler matrix freeze
  bit-exact sequential resume behavior.
- **Exhaustive HPO sampler-pruner conformance.** Added a versioned,
  machine-readable 9 × 5 compatibility matrix and a fail-closed Track-Q gate.
  Every native/Python cell now terminalizes a non-trivial intermediate tape,
  crosses its adaptive or population boundary, replays exactly, and resumes
  exactly from a terminal-prefix checkpoint. Stable transverse refusals are
  checked by C ABI status code. The contract explicitly excludes wall-clock
  durations, independent-algorithm claims, and unimplemented cross-binding
  45/45 parity.
- **Native HPO optimizer (`optimization` role, ABI 2.1.0) — Phase 1 / F0.** A
  handle-based ask/tell hyperparameter optimizer with a typed search space
  (`n4m_search_space_*`), the optimizer handle (`n4m_optimizer_*`: ask/tell,
  batch ask, warm-start enqueue, intermediate reporting, trial streaming), trial
  accessors (`n4m_trial_*`), and a pure-native single-level driver
  `n4m_finetune_estimator` (closed registry for PLS regression/canonical/SVD,
  OPLS, sparse PLS and PCR over one native regression-validation plan). The
  Python binding exposes the same selection-only route through owning
  `ValidationPlan` / `FinetuneResult` objects, including active best parameters,
  the full trial trace and explicit partial-timeout state. All sampler/pruner
  algorithms sit behind reserved enum values; F0 ships `random` + `none`, with
  reserved kinds returning `N4M_ERR_NOT_IMPLEMENTED`. Search space supports
  int/float/log/categorical (typed values)/ordinal/sorted-tuple params plus
  generic constraints (mutex/requires/exclude + conditional activation). See
  `docs/FINETUNING_ROADMAP.md`, `docs/FINETUNING_F0_PR.md`,
  `docs/methods/optimization.md`.

## [1.0.5] - 2026-07-06

### Fixed
- Align the slim `pls4all` Linux wheel builder with the full
  `nirs4all-methods` wheel policy by using `manylinux_2_28` images. This keeps
  the runtime `numpy>=1.21` dependency unconstrained while avoiding NumPy 2.5
  source builds on manylinux2014 during cibuildwheel smoke installs.

## [1.0.4] - 2026-07-06

### Fixed
- Constrain the `pls4all` cibuildwheel smoke-test NumPy dependency to a
  wheel-backed range on manylinux2014, avoiding a NumPy 2.5 source build with
  GCC 10.2 during Linux wheel validation.

### Changed
- Synchronized patch-release metadata across C/CMake, Python, R,
  MATLAB/Octave, JavaScript/WASM, citation, documentation, and parity fixtures.

## [1.0.3] - 2026-07-06

### Changed
- Synchronized patch-release metadata across C/CMake, Python, R,
  MATLAB/Octave, JavaScript/WASM, citation, documentation, and parity fixtures.

## [1.0.0] — 2026-06-14

**Breaking release.** The public surface is reorganized under an explicit
`n4m.<role>` namespace and the C ABI is cleanly broken to **2.0** — there are
**no runtime aliases**. Downstream consumers must migrate; see
[`docs/MIGRATION_ABI2.md`](docs/MIGRATION_ABI2.md). ABI semver is bumped
independently to 2.0.0; project version 0.99.0 → 1.0.0.

### Changed — BREAKING
- **C ABI 2.0 (clean break).** All 566 exported method symbols are renamed to
  `n4m_<role>_<leaf><verb>` — e.g. `n4m_ridge_fit` → `n4m_estimators_ridge_fit`,
  `n4m_pp_snv_create` → `n4m_transform_snv_create`, `n4m_cars_select` →
  `n4m_feature_selection_cars_select`, `n4m_pls_fit_simple` →
  `n4m_estimators_pls_fit`. The terse 1.x symbols are **removed** (no aliases).
  SONAME → `libn4m.so.2`; linker version node `N4M_1` → `N4M_2`.
- **Header layout.** `n4m.h` is now an umbrella over 12 role headers (`transform`,
  `estimators`, `feature_selection`, `augmentation`, `model_selection`,
  `domain_adaptation`, `ensemble`, `compose`, `outlier_detection`, `metrics`,
  `decomposition`, `lowlevel`) + per-role sub-headers. `pls.h` and the legacy
  category stub headers are removed.
- **Python.** The flat `n4m.python` module is replaced by `n4m.<role>` subpackages
  with public class names (`from n4m.estimators.regression.regularized import Ridge`,
  `from n4m.transform.scatter import SNV`, `from n4m.feature_selection.wrapper import CARS`).
  No top-level re-exports, no public `Native*`.
- **R / MATLAB·Octave / JS-WASM** bindings call the ABI-2 symbols; R adds
  role-named wrappers (`n4m_regression_ridge()`, `n4m_transform_snv()`, …).

### Added
- **Catalog is the namespace source of truth.** Every method carries
  `namespace` / `leaf` / `fq_name` / `c_surface` / `legacy_ids` (209 methods; the
  `selection.wvc` double surface is split into `wvc` + `wvc_threshold`).
- `docs/MIGRATION_ABI2.md` — old→new symbol / import / header migration guide.
- A doc-lint that fails if any legacy ABI-1 symbol or prefix leaks into the
  generated method pages.

### Notes
- Parity fixtures were **not** regenerated — this release renames surfaces only;
  numerical behaviour is unchanged.

## [0.99.0] — 2026-06-10

The public C ABI grew additively from **1.10.0 → 1.22.0** since 0.98.0; every
bump is backward-compatible (new symbols only — nothing removed or changed).
Full per-symbol record: [`docs/abi/changes_log.md`](docs/abi/changes_log.md).

### Added — public C ABI (all additive MINOR bumps)
- **Direct closed-form Ridge regression** — `n4m_ridge_fit` (primal QR + dual
  forms), parity with numpy/sklearn at ~1e-15 (ABI 1.11.0).
- **Native AOM / moment screening family** — robust-HPO screen (1.12.0),
  row-additive moment substrate (1.13.0), native Ridge/PLS sweep (1.14.0),
  configurable native AOM preprocessing sweep (1.15.0), user-defined AOM chain
  sweep `n4m_aom_chain_sweep_run` (1.16.0), AOM Ridge OOF simplex blender
  `n4m_aom_ridge_blender_fit` (1.17.0), AOM operator PLS score stack
  `n4m_aom_operator_pls_stack_fit` (1.18.0), plus AOM chain fixed final fit
  `n4m_aom_chain_fixed_fit_run`, a score-only screen output mode, and the
  `N4M_OP_GAUSSIAN` strict operator kind (1.18.x).
- **PLS cross-validation reference surface** — `n4m_pls_cross_validate`, exact
  PLS-only CV over one matrix; catalogued as ABI infrastructure and reserved for
  a future fused/batched IKPLS executor (1.22.0).
- **CUDA PLS scheduling config helpers** — parallel-fold scheduling (1.19.0),
  device-feature threshold (1.20.0) and many-design batching (1.21.0), each a
  paired get/set with additive route counters; defaults preserve prior behaviour
  and candidate scores stay fold-exact.

### Added — bindings
- **R**: `coef.n4m_fit` returns the (p × q) regression-coefficient matrix (via
  the existing `n4m_model_get_array` C ABI), in both the `n4m` and slim
  `pls4all` packages.
- **R**: the `n4m` and `pls4all` packages are now installable from **R-universe**
  — a `.prepare` pre-build hook vendors the libn4m core into a self-contained
  source tarball, so R-universe (and `R CMD build` from a checkout) build with no
  external libn4m. See `docs/dev/release_process.md` (CRAN + R-universe registration).

### Changed
- Renamed the R packages' internal `.Call` registration prefix `r_p4a_` →
  `r_n4m_` (internal symbols only — no public ABI change; the `pls4all` package
  name and slim subset are unchanged).

### Performance — CUDA (opt-in `cuda-on` build; no ABI change)
- GPU full-execution pass (B1–B3): the Ridge GCV solve now routes to cuSOLVER
  SPD Cholesky (`spd_solve`), with device-resident dual-Ridge and a size-gated
  device moment/gram build — **cumulative 4.22× on BERRY/COLZA/LUCAS**, RMSEP
  preserved/bit-identical, bit-equivalent host fallback.

### Fixed
- LW-PLS parity fixture/doctest realigned to the 2026-05-23 Gaussian-weighted
  local-PLS engine (the reference generator now matches the C++ engine to
  6.7e-16; the previously-skipped doctest is active again).

### Benchmarks
- Full-57 staged AOM/moment campaign vs AOM-PLS / AOM-Ridge / TabPFN oracles
  (51 datasets): competitive but not dominant (paired median ratios
  1.04 / 1.10 / 1.11). Recall characterized as largely irreducible selection
  variance (top-5 recall 0.784 — a selection, not candidate-budget, gap).

Next track: language bindings (Rust, Go, Ruby, Lua, Nim, …) catch-up to the
current n4m ABI 1.22 symbols.

## [0.98.0] — 2026-05-26

First tagged release under the unified `n4m` / nirs4all-methods name — the
`p4a`/`pls4all` → `n4m` rename and donor merge are complete. Primarily a
build-system, CI and packaging-consistency hardening pass on the 0.97.x lineage.

### Fixed
- C++ core compiles cleanly under the strict `-Werror` CI warning set across
  gcc-12/13, clang-16, MSVC and MinGW (dead-code / `static` / missing-declaration
  cleanups; ~330 `int64`↔`size_t` sign-conversion fixes at the C ABI boundary).
- Windows C ABI export: `n4m_export.h` honours `N4M_BUILDING_C_DLL` and the
  static target defines `N4M_STATIC` — fixes MSVC C2491 / MinGW dllimport errors.
- Version-sync release gate repaired; every active manifest synced to 0.98.0.
- Python wheel loader locates the Windows `n4m.dll` (MSVC `OUTPUT_NAME "n4m"`).
- k-means splitter parity tolerates cross-architecture floating-point variation.
- abi-check allows the ELF version node `N4M_1`; parity-reference lockfile
  regenerated.

### Known limitations
- The R CRAN package (`bindings/r/n4m`) vendoring is a Phase-F deliverable; its
  release workflow is manual-only (`workflow_dispatch`) until that work lands.

## [phase-54-bindings-tier2] — 2026-05-16

Two-tier binding sprint: every binding can now ship a *tier 1* raw FFI
shim (one-to-one with the C ABI) AND a *tier 2* idiomatic wrapper
honouring the host language's conventional ML estimator contract.

**Honest coverage status** (matches the matrix in README §"Tier 2
idiomatic-wrapper coverage"):

| Binding | Tier 1 coverage | Tier 2 coverage |
|---------|---|---|
| Python (`pls4all.sklearn`) | 64 / 68 reachable | **42 / 68** sklearn classes (62 %) |
| R / MATLAB / Julia / JS    | tier-1 SIMPLS only | **1 class each** (PoC validating design) |
| Go / Rust / .NET / JNI / Ruby / Lua / Nim | tier-1 parity | 0 (tier 2 not started) |

Python tier 2 is the only mostly-complete layer. The 4 other tier-2
PoCs are deliberate single-class slices that prove the pattern (formula
+ S3 / classdef / mutable struct / async TS class) end-to-end with
bit-exact tier1-vs-tier2 agreement (max abs diff ≤ 2e-15). Replicating
the Python sklearn breadth to R / MATLAB / Julia / JS is tracked as a
follow-up phase, as is wrapping the remaining 26 Python methods (17
in-sample-only methods that need an ABI coefficient export, 3 AOM
entries, 6 diagnostics).

### Python sklearn — what shipped

- 8 Model-based estimators (`PLSRegression`, `OPLSRegression`, `PCR`,
  `SparsePLSRegression`, `PLSCanonical`, `PLSSVD`, `PLSDAClassifier`,
  `OPLSDAClassifier`) backed by `Algorithm` enum + `Model.fit`, with
  bit-exact raw N4MM fitted-model payload round-trip pickling via the new
  `Model.to_bytes()` / `Model.from_bytes()` ABI shim.
- 6 MethodResult regressors (`SparseSimplsRegression`, `CPPLSRegression`,
  `ECRegression`, `DIPLSRegression`, `MIRPLSRegression`,
  `MBPLSRegression`) with predict-on-new-X via the C predict math
  `(X - x_mean) @ coef.T + y_mean`.
- 28 SelectorMixin variable-selection wrappers (VIP, CARS, SPA,
  RandomFrog, IRIV, IRF, VIPSPA, GA, PSO, VISSA, Shaving, BVE, REP,
  IPW, ST, Stability, UVE, EMCUVE, Randomization, Interval, biPLS,
  siPLS, T2, WVC, WVCThreshold, SCARS, CoefficientSelector,
  SelectivityRatioSelector).
- 209 sklearn tests pass: bit-exact wrapper-vs-tier1 parity, pickle /
  deepcopy bit-exact, Pipeline + GridSearchCV smoke, edge-case
  regressions for codex catches (NaN-in-X, wrong-feature-count,
  failed-refit state hygiene, multi-target score average, etc.).
- Codex MUST FIX (2 rounds) applied: predict math `(X - x_mean) @ coef`
  instead of dividing by `x_scale`; MBPLS predict directly via the
  C-side intercept; `_validate_X_y_no_mutate` for failed-fit state
  hygiene; `_ManagedConfig` context manager for deterministic Config
  lifecycle; JS `Model.fit` honouring n_components; JS multi-target
  score now per-target then averaged.

### Tier 1 prereqs landed this phase

- `bindings/python/src/pls4all/_ffi.py`: added ctypes bindings for
  `p4a_model_export_size / export_to_buffer / import_from_buffer /
  p4a_serialization_inspect`.
- `bindings/python/src/pls4all/_model.py`: new `Model.to_bytes()`
  method and `Model.from_bytes(ctx, payload)` classmethod (bit-exact
  round-trip on a 3128-byte payload for a 50×20 PLS-5 model).
- ABI unchanged (no new symbols).

## [parity-audit-may-2026] — 2026-05-16

End-to-end parity-gate audit. The full historical closeout report,
`docs/parity_audit_2026_05/03_closeout.md`, is preserved in the
[private development archive](docs/dev/documentation.md).

| Metric | Before | After |
|--------|------:|------:|
| Methods in the parity gate | 33 | **68** |
| Methods with real `ok` rows | 11 (33 %) | **64 (94 %)** |
| `paper_only` methods | 22 | **4** |

### Phase 53 — VIP_SPA + DI-PLS unlock + IRIV unlock

- **VIP_SPA** new pls4all method (Favilla 2013 VIP + Araujo 2001 SPA
  composition, mirrors `auswahl.VIP_SPA`). SPA greedy projections on
  raw masked X; start = `argmax(VIP)` within mask.
  - C++ core: `cpp/src/core/vip_spa_selection.{hpp,cpp}` (new)
  - C ABI: `p4a_vip_spa_select`
  - Python: `pls4all.vip_spa_select(...)`
  - Parity vs `auswahl.VIP_SPA`: `rmse_rel = 1.08` (mask metric).
- **DI-PLS** moved from `paper_only` to `ok`: `diPLSlib 2.5.0`
  (B-Analytics; Nikzad-Langerodi 2018 authors) wired as canonical
  reference. `rmse_rel = 5e-3` under `tol = 2e-2`.
- **IRIV** unlocked: Octave Forge `statistics 1.7.7` (last release
  before the `datatypes >= 1.2.0` cascade) installs on Octave 10.3.0
  and provides `ranksum`. MethodSpec auto-wires through
  `_octave_has_statistics()`.
- ChemometricsTools.jl audited on Julia 1.6.7 LTS — no PLS variant that
  pls4all doesn't already have; out-of-scope tools (MCR-ALS, BTEM,
  SIMPLISMA, CORAL, COW, …) reserved for a future dedicated
  chemometrics library layered on top of pls4all.

### Codex review fixes

- VIP_SPA SPA pass now runs on raw masked X (not autoscaled) to faithfully
  match `auswahl._spa._fit_spa`.
- `VipSpaSelectionResult` exposes `n_selected` (actual count) alongside
  `top_k` (requested upper bound).
- `_VipSpaAuswahlReference` asserts `vip_threshold == 0.3` (auswahl
  hard-codes that constant; silent drift would falsely pass).
- DI-PLS sklearn-1.8 `force_all_finite` → `ensure_all_finite` shim is
  now signature-gated.

### Release

- ABI version: `1.15.0` → `1.16.0` (one new additive symbol:
  `p4a_vip_spa_select`).
- Tag: `phase-53-vip-spa`.

## [phases-50-52-libpls] — 2026-05-16

Three new methods ported from libPLS 1.95 (MATLAB toolbox):

- **ECR** (Liu 2013 Elastic Component Regression) — bridges PCR (α=0)
  to PLS (α=1). C++ kernel `core/ecr.{hpp,cpp}`. Parity vs libPLS
  `ecr.m`: `rmse_rel ≈ 1e-7` (tight).
- **IRIV** (Yun 2014 Iteratively Retains Informative Variables) —
  iterative variable selection via Mann-Whitney U test on
  permuted-replacement CV-RMSE. C++ kernel `core/iriv_selection.{hpp,cpp}`.
- **IRF** (Yun 2013 Interval Random Frog) — Random Frog applied to
  fixed-width sliding intervals. C++ kernel `core/irf_selection.{hpp,cpp}`.

ABI bumped 1.14 → 1.15 (three new additive symbols).

## [0.97.0-gpr-pso-vissa] — 2026-05-16

Three new methods previously deferred:

### Phase 47 — GPR-on-PLS (Gaussian Process Regression on PLS scores)

Two-stage regression: SIMPLS rotation to k latent components, then a
Gaussian Process with RBF kernel + diagonal noise variance solved via
Cholesky. The GP head (`fit_gp_on_scores`) is exposed as a standalone
internal primitive so a future GPR-on-AOMPLS reuses it unchanged.
Single-target only in Phase 47.

- C++ core: `cpp/src/core/gpr_pls.{hpp,cpp}` (new)
- C ABI: `p4a_gpr_pls_fit` (additive — ABI minor 1.13 → 1.14)
- Python: `pls4all.gpr_pls_fit(...)`
- Parity gate: sklearn `GaussianProcessRegressor(RBF + WhiteKernel,
  optimizer=None)` on shared T_train. `rmse_rel = 2.3e-10` under
  the `1e-8` tolerance. The adapter factors out the PLS rotation-
  convention mismatch by re-running pls4all to get the same T.
- `noise_level` is the diagonal noise **variance** (σ², not σ),
  matching sklearn `WhiteKernel(noise_level=...)`. Documented in
  both the C header and the Python docstring.

### Phase 48 — PSO-PLS (Binary Particle Swarm Optimisation)

Variable selection via Binary PSO (Kennedy & Eberhart 1997).
Continuous velocity update + sigmoid stochastic binarisation, with a
repair step that flips random zero bits to ensure
`popcount(mask) ≥ n_components`. Defaults follow Clerc-Kennedy 2002
constriction: `w=0.729, c1=c2=1.494, v_max=4.0`.

- C++ core: `cpp/src/core/pso_selection.{hpp,cpp}` (new)
- C ABI: `p4a_pso_select`
- Python: `pls4all.pso_select(...)`
- Parity: paper-only (Kennedy & Eberhart 1997). Deterministic via
  splitmix64 seeded by the `seed` parameter.

### Phase 49 — VISSA-PLS (Variable Iterative Space Shrinkage Approach)

Weighted Binary Matrix Sampling iterative shrinkage (Deng et al.
2014). Initial w_j=0.5; per iteration sample n_submodels via
Bernoulli(w_j), keep top-K by CV-RMSE, average their masks to update
w_j, clamp to `[floor, 1-floor]`. Final selection: w_j > threshold.

- C++ core: `cpp/src/core/vissa_selection.{hpp,cpp}` (new)
- C ABI: `p4a_vissa_select`
- Python: `pls4all.vissa_select(...)`
- Parity: paper-only (Deng et al. 2014).

### Codex review findings

- PSO `inclusion_frequencies` undercounted repair-added bits;
  fixed by moving the increment into the post-repair sync loop.
- GPR `noise_level` parameter convention (σ² vs σ) was ambiguous
  in the API doc; clarified in both the C header and Python
  docstring.

### Release

- Library version: `0.96.0` → `0.97.0`.
- ABI version: `1.13.0` → `1.14.0` (three new additive symbols:
  `p4a_gpr_pls_fit`, `p4a_pso_select`, `p4a_vissa_select`).
- Tags: `phase-47-gpr-on-pls`, `phase-48-pso-pls`,
  `phase-49-vissa-pls`.

## [0.96.0-cuda-backend] — 2026-05-16

Optional cuBLAS backend. Default OFF — the OFF build is bit-
identical to pre-Phase-45 (no CUDA headers, no CUDA symbol
references). When `PLS4ALL_WITH_CUDA=ON` AND a compatible GPU is
present at runtime, `linalg::gemv/gemm/ger` route through
`cublasDgemv_v2 / cublasDgemm_v2 / cublasDger_v2`. Public ABI
unchanged at 1.13.0.

### CUDA backend

- `cpp/src/core/cuda_dispatch.hpp` + `.cpp` (new): Meyers-singleton
  cuBLAS handle, `DevicePtr<T>` RAII for cudaMalloc/cudaFree, the
  three dispatch entry points using the row-major → column-major
  swap trick (`op_cm = op_row`, no flip; B passed first, A second,
  with dimensions `m_cm=N, n_cm=M, k_cm=K`).
- `cpp/src/core/linalg.hpp`: `Trans_t` uses plain int when CUDA is
  on; each dispatch function tries CUDA first, then BLAS, then
  scalar.
- `cpp/src/c_api/c_api_version.cpp`: `p4a_backend_is_available
  (P4A_BACKEND_CUDA)` is the first backend whose result is a
  **runtime probe** (compile-time + GPU detection at first call,
  memoised in the singleton).
- `cpp/src/core/context.cpp`: `set_backend(CUDA)` accepts when
  compile-time AND runtime checks both pass; clear error otherwise.
- `cpp/cli/p4a_cli.cpp` + `cpp/cli/CMakeLists.txt`: selfcheck
  accepts `P4A_OK` or `P4A_ERR_BACKEND_UNAVAILABLE` under CUDA-ON.
- `cpp/tests/abi/test_cuda_backend.cpp` (new): fixture parity +
  200×20×2 smoke; skips gracefully if no GPU at runtime.
- `cpp/tests/abi/test_error_messages.cpp`: switched the "must be
  unavailable" probe from CUDA to `P4A_BACKEND_OPENCL` (truly
  never-implemented).
- Top-level `CMakeLists.txt` + `cpp/src/CMakeLists.txt` +
  `cpp/tests/CMakeLists.txt`: `find_package(CUDAToolkit REQUIRED)`
  hoisted, `CUDA::cudart` + `CUDA::cublas` linked into
  `pls4all_core`, `pls4all_c`, `pls4all_c_static`, and
  `pls4all_tests` via the existing helper pattern.

Ship gate (live on RTX 4090 + RTX 5090, CUDA 12.8):

- OFF build: 265/265 tests, all 10 cross-binding parity gates green.
- CUDA-ON build: 265/265 tests (3 CUDA tests run live on GPU 0),
  all 10 cross-binding parity gates green against fixture
  regenerated through the cuBLAS path.
- ABI symbol count unchanged at 160.

Codex review: all math correct, RAII covers leak paths, the
P4A_BACKEND_OPENCL substitution is valid. One latent footgun
documented in the header: cuda_dispatch assumes **contiguous**
row-major buffers (`lda == cols`); all current call sites honour
this. Phase 45b will extend to strided inputs.

Phase 45b backlog: pinned host memory, async streams, strided
buffers, persistent device-side model state.

## [0.95.0-openmp-backend] — 2026-05-16

Optional OpenMP backend. Default OFF — the OFF build is bit-
identical to pre-Phase-44 (no `<omp.h>` include, empty
`P4A_PARALLEL_FOR_STATIC` macro). When `PLS4ALL_WITH_OPENMP=ON`
the rank-1 deflations, dot-product loading kernels, and
`fill_prediction`/`fill_transform` outer loops in `model.cpp`
are annotated with `#pragma omp parallel for schedule(static)`.
Public ABI unchanged at 1.13.0; symbol count stays at 160.

### OpenMP backend

- `cpp/src/core/parallel.hpp` (new): `P4A_PARALLEL_FOR_STATIC`
  macro with GCC/Clang `_Pragma` and MSVC `__pragma` branches.
- `cpp/src/core/model.cpp`: 21 pragma sites covering every solver's
  deflation and loading kernels plus the two fill helpers. Every
  site has thread-disjoint writes and zero `ctx` access inside the
  parallel region (Codex review confirmed).
- `cpp/src/c_api/c_api_version.cpp`: `p4a_backend_is_available`
  reports OpenMP availability based on `P4A_USE_OPENMP`.
- `cpp/src/CMakeLists.txt` + top-level `CMakeLists.txt`:
  `find_package(OpenMP REQUIRED COMPONENTS CXX)` hoisted to the
  top so `OpenMP::OpenMP_CXX` is visible in `cpp/src/` and
  `cpp/tests/`. CMake `WARNING` when BLAS+OMP both on (OpenBLAS
  USE_OPENMP coexistence) or OMP+TSAN (libgomp not TSAN-clean).
- `cpp/tests/abi/test_openmp_backend.cpp` (new): fixture parity
  + 200x20x2 smoke; passes whether OMP is compiled in or not.

All 4 build configurations (OFF/OFF, BLAS-only, OMP-only,
BLAS+OMP) pass 262/262 tests + 10/10 cross-binding parity gates.

Phase 44b backlog: outer-loop parallelism (CV folds, EMCUVE
ensembles, AOM operator sweeps), `Context::num_threads` plumbing
into `omp_set_num_threads`, benchmark CLI knob.

## [0.94.0-blas-backend] — 2026-05-16

Optional CBLAS backend. Default is **OFF** — the OFF build is
bit-identical to pre-Phase-43 with the same scalar loops and no new
link dependency. When `PLS4ALL_WITH_BLAS=ON` libp4a routes the hot
linear-algebra kernels (matrix-vector products in SIMPLS / NIPALS;
cross-covariance `X^T Y` in SIMPLS / SVD / kernel-algo / power /
randomized-SVD / canonical pair / OPLS / DI-PLS) through a system
CBLAS implementation. Public C ABI is unchanged at 1.13.0; symbol
count stays at 160; all 10 cross-binding parity gates pass with
maximum drift `5.2e-16` rmse_rel.

### BLAS backend

- `cpp/src/core/linalg.hpp` — internal `pls4all::linalg` namespace
  with `gemv`, `matvec`, `gemm` and `ger` entry points. Reference
  scalar implementations under `!P4A_USE_BLAS`, CBLAS dispatch
  under `P4A_USE_BLAS`. The macOS `<Accelerate/Accelerate.h>`
  header is auto-selected when `<cblas.h>` is unavailable.
- `cpp/src/core/model.cpp` — `matrix_vector_product` body forwards
  to `linalg::matvec` (covers 14 callers). The two NIPALS
  `X^T y_score / y_score_ss` inner-loop sites become `linalg::gemv`
  with the alpha factor folded in. The 9 cross-covariance triple
  loops become `linalg::gemm(Trans_Yes, Trans_No, ...)`.
- `cpp/src/c_api/c_api_version.cpp` — `p4a_backend_is_available`
  returns 1 for `P4A_BACKEND_BLAS` when compiled in.
- `cpp/src/CMakeLists.txt` + `cpp/tests/CMakeLists.txt` —
  `find_package(BLAS)` + `check_include_file_cxx(cblas.h)` gate
  the option; both hard-fail with actionable diagnostics if the
  toolchain is incomplete. `${BLAS_LIBRARIES}` linked into
  `pls4all_c`, `pls4all_c_static` and `pls4all_tests`.
- `cpp/tests/abi/test_blas_backend.cpp` — fixture parity + 200x20x2
  smoke test; passes whether BLAS is compiled in or not.
- Phase 43b backlog: rank-1 deflations via `linalg::ger`,
  `compute_rotations_and_coefficients` GEMMs, BLAS-aware bench CLI.

## [0.93.0-nim-binding] — 2026-05-16

Zero-nimble Nim binding via the language's built-in `{.importc,
dynlib.}` pragmas. Parity-gated against the shared cross-binding
fixture; result is bit-exact (rmse_rel = 0.0 on all four output
arrays).

### Nim binding

- `bindings/nim/pls4all.nim` — public API (`pls4all.version`,
  `pls4all.abiVersion`, `pls4all.plsFit`). Uses `openArray[float64]`
  for input slices (contiguous, length-aware), and `seq[float64]`
  for outputs.
- `bindings/nim/test/test_parity.nim` — parity gate using the
  stdlib `std/json` module (no extra package).
- `bindings/nim/README.md`, `bindings/nim/.gitignore`.
- Compile-time library override via `-d:libp4aName=...`.

No ABI change (uses `p4a_pls_fit_simple` from 1.13).

## [0.92.0-lua-binding] — 2026-05-16

Zero-rock Lua / LuaJIT binding via LuaJIT's built-in FFI module.
Parity-gated against the shared cross-binding fixture; result is
bit-exact (rmse_rel = 0.0 on all four output arrays, matching
JNI / Rust / .NET / Ruby).

### Lua binding

- `bindings/lua/pls4all.lua` — public API (`pls4all.version`,
  `pls4all.abi_version`, `pls4all.pls_fit`). Uses LuaJIT's
  `ffi.new("double[?]", N, src_table)` to populate the C array
  directly from a Lua sequence — the most direct FFI hand-off in
  the binding set.
- `bindings/lua/test/test_parity.lua` — parity gate (regex-based
  fixture parser; no JSON-rock required).
- `bindings/lua/README.md`.

Requires LuaJIT (the `ffi` module is LuaJIT-specific). Plain Lua
support is on the backlog. No ABI change (uses `p4a_pls_fit_simple`
from 1.13).

## [0.91.0-ruby-binding] — 2026-05-16

Zero-gem Ruby binding via the stdlib `Fiddle` library. Parity-gated
against the shared cross-binding fixture; result is bit-exact
(rmse_rel = 0.0 on all four output arrays, matching JNI / Rust /
.NET).

### Ruby binding

- `bindings/ruby/lib/pls4all.rb` — public API
  (`Pls4all.version`, `Pls4all.abi_version`, `Pls4all.pls_fit`).
  Uses `Array#pack("E*")` / `String#unpack("E*")` to marshal Float
  arrays to/from IEEE 754 little-endian double buffers — the exact
  layout libp4a's row-major C ABI expects.
- `bindings/ruby/test/test_parity.rb` — parity gate.
- `bindings/ruby/README.md`.

No gem install required: Fiddle has been in Ruby core since 1.9.
No ABI change (uses `p4a_pls_fit_simple` from 1.13).

## [0.90.0-dotnet-binding] — 2026-05-16

.NET 9 class library wrapping libp4a via P/Invoke
(`[DllImport("p4a", ...)]`). Parity-gated against the shared
cross-binding fixture; result is bit-exact (rmse_rel = 0.0 on all
four output arrays, identical to JNI and Rust).

### .NET binding

- `bindings/dotnet/Pls4all/Pls4all.cs` — public API. Top-level
  `Pls4all` static class with `Version()`, `AbiVersion()` and
  `PlsFit(ReadOnlySpan<double> x, ReadOnlySpan<double> y, int n,
  int p, int q, int nComponents)`. `Span<double>` inputs are pinned
  via `fixed` for zero-copy hand-off into libp4a.
- `bindings/dotnet/Pls4all/Pls4all.csproj` — net9.0 class library,
  NuGet metadata in place.
- `bindings/dotnet/TestParity/Program.cs` — top-level statement
  console app for the parity gate.
- `bindings/dotnet/README.md`, `bindings/dotnet/.gitignore`.

No ABI change (uses `p4a_pls_fit_simple` from 1.13).

## [0.89.0-rust-binding] — 2026-05-16

Rust crate `pls4all`. Zero-dep, hand-rolled `extern "C"` block
against `p4a_pls_fit_simple`. Parity-gated against the shared
cross-binding fixture; result is bit-exact (rmse_rel = 0.0 on all
four output arrays, identical to JNI's exact result).

### Rust binding

- `bindings/rust/pls4all/` — rlib crate. `Cargo.toml` declares no
  dependencies; `FitError` implements `Display` and
  `std::error::Error` by hand.
- `src/lib.rs` — `pls4all::version()`, `pls4all::abi_version()`,
  `pls4all::pls_fit(&[f64], &[f64], n, p, q, k) -> Result<FitResult,
  FitError>`. Uses `#![deny(unsafe_op_in_unsafe_fn)]` so every
  unsafe block carries a SAFETY comment.
- `build.rs` — auto-detects libp4a's location by walking up from
  the manifest dir; overridable via `PLS4ALL_LIB_DIR` and
  `PLS4ALL_RUNTIME_RPATH`.
- `examples/test_parity.rs` — parity gate (run via
  `cargo run --example test_parity --release`).

No ABI change (uses `p4a_pls_fit_simple` from 1.13).

## [0.88.0-go-binding] — 2026-05-16

Go binding via cgo. Same `p4a_pls_fit_simple` C-ABI helper, gated
against the shared cross-binding parity fixture at machine epsilon.

### Go binding

- `bindings/go/go.mod` — module
  `github.com/GBeurier/pls4all/bindings/go`.
- `bindings/go/pls4all/pls4all.go` — public Go API. `Version()`,
  `AbiVersion()` and `Fit(x, y []float64, n, p, q, k int)
  (*FitResult, error)`. Go `[]float64` is row-major like the C ABI,
  so the slice pointer is handed straight through cgo — no copy,
  no transpose.
- `bindings/go/cmd/test_parity/main.go` — parity gate against the
  shared `bindings/js/test/parity_fixture.json`. RMSE-rel ~ 2e-16
  on all four output arrays.

No ABI change (uses `p4a_pls_fit_simple` from 1.13).

### Phase 36 deferred

Android NDK packaging is on hold pending NDK availability. The
Phase 35 JNI source is unchanged; only the build script needs to
target the NDK toolchain.

## [0.87.0-jni-binding] — 2026-05-16

Desktop JNI binding for the JVM. Same `p4a_pls_fit_simple` C-ABI
helper, gated against the shared cross-binding parity fixture.

### JNI binding

- `bindings/jni/java/io/github/pls4all/Pls4all.java` — public Java
  API with `version()`, `abiVersion()` and
  `plsFit(double[] x, double[] y, int n, int p, int q, int k)`
  returning a `FitResult` record.
- `bindings/jni/c/p4a_jni.c` — JNI implementation. Java arrays are
  row-major like the C ABI, so `GetPrimitiveArrayCritical` pins them
  without copying.
- `bindings/jni/test/TestParity.java` — parity gate against the
  shared `bindings/js/test/parity_fixture.json`. RMSE-rel = 0.0
  bit-exact across all four output arrays.
- `bindings/jni/build.sh` — one-shot build (javac → JNI header →
  `libp4a_jni.so` → TestParity.class). Auto-detects the conda-forge
  `<env>/lib/jvm` JDK layout.

No ABI change (uses `p4a_pls_fit_simple` from 1.13).

## [0.86.0-octave-matlab-binding] — 2026-05-16

MATLAB / GNU Octave binding via MEX shims, parity-gated against the
shared cross-binding fixture at machine-epsilon precision.

### Octave / MATLAB binding

- `bindings/matlab/+pls4all/pls_fit.m` — `pls4all.pls_fit(X, Y, k)`
  fits a SIMPLS regression and returns the coefficient matrix, the
  centring vectors and the in-sample predictions.
- `bindings/matlab/+pls4all/version.m` — `pls4all.version()` returns
  `"0.86.0+abi.1.13.0"`.
- `bindings/matlab/mex/p4a_pls_fit_mex.c` — column-major
  (MATLAB / Octave) ↔ row-major (libp4a) adapter that calls
  `p4a_pls_fit_simple`.
- `bindings/matlab/mex/p4a_version_mex.c` — thin wrapper around
  `p4a_get_version_string`.
- `bindings/matlab/build_mex.m` — single build script that runs on
  both Octave (`mkoctfile --mex`) and MATLAB (`mex`).
- `bindings/matlab/test/test_parity.m` — parity gate against
  `bindings/js/test/parity_fixture.json` (generated by the native
  Python binding). RMSE-rel at numerical floor: `coef = 0`,
  `x_mean = 2.7e-17`, `y_mean = 0`, `preds = 4.7e-17`.

No ABI change (uses the `p4a_pls_fit_simple` helper added in 1.13).

## [0.85.0-julia-binding] — 2026-05-16

Julia binding (`Pls4all.jl`) + new public ABI helper that bypasses
the matrix-view-pointer interop ambiguity that bit both Emscripten
and Julia's ccall:

- `p4a_pls_fit_simple(x, y, n, p, q, n_components, coefs_out,
  x_mean_out, y_mean_out, preds_out)` — raw-pointer SIMPLS PLS
  regression. Default-on centring and scaling matching the public
  Config defaults. Returns the model triple plus optional in-sample
  predictions.

ABI minor 1.12 → 1.13.

### Julia binding

- `bindings/julia/Pls4all.jl/` — Julia 1.10+ package wrapping
  the public C ABI via `ccall`. Set `PLS4ALL_LIB` env var to point
  at `libp4a.so` and `using Pls4all` does the rest.
- `Pls4all.version()`, `Pls4all.abi_version()`,
  `Pls4all.pls_fit(X, Y; n_components=3)`.
- `bindings/julia/Pls4all.jl/test/test_parity.jl` — parity gate
  against `bindings/js/test/parity_fixture.json` (generated by the
  native Python binding). RMSE-rel at numerical floor:
  - coefficients: 1.16e-16
  - x_mean: 1.35e-16
  - y_mean: 2.20e-16
  - predictions: 1.74e-16

### R binding

- `bindings/r/pls4all/src/r_gateway.c` now sets `scale_x = scale_y =
  1` to align with the Python binding's defaults. Without this the R
  binding diverged by ~7.6e-6 from the Python fixture; with the
  alignment the prediction-space match is exact (rmse_rel = 0).
- `bindings/r/test_parity.R` added — parity gate against the shared
  fixture.
- `Makevars` now accepts `PLS4ALL_GENERATED_DIR` for the CMake-
  generated `pls4all/p4a_export.h` path.

### WASM binding

- TypeScript `fitPls` / `predictPls` (and the underlying smoke
  test) now call the public `p4a_pls_fit_simple` symbol instead of
  the binding-internal `p4a_wasm_pls_fit_legacy` (renamed for
  historical reference). The Emscripten parity gate result is
  unchanged at numerical floor.

### Cross-binding parity matrix (this snapshot)

| Binding | rmse_rel vs native Python fixture |
|---------|-----------------------------------|
| R (`pls4all`) | 0.0 (exact) |
| Julia (`Pls4all.jl`) | 1.7e-16 |
| WASM (`@pls4all/wasm`) | 2.1e-16 |

### Verified

- 256 native C++ tests pass.
- ABI symbol diff: 160 symbols (+1, clean).
- `ldd` audit: only libc/libstdc++/libgcc/libm/loader.
- Cross-binding parity gates: all PASS at numerical floor.

### Changed

- Project version `0.85.0+abi.1.13.0`. C ABI minor 12 → 13 (additive
  — 1 new symbol, `p4a_pls_fit_simple`).

## [0.84.0-wasm-parity] — 2026-05-16

WebAssembly binding now passes a parity gate vs the native Python
pls4all binding at numerical floor (1e-16 RMSE-rel).

### Root cause of the WASM fit regression

Emscripten 5.0.7 mis-compiles direct calls to `p4a_model_fit` (and
related entry points) when the matrix-view pointer arguments come
from a JS-allocated buffer. The same C++ kernel called from a C
translation unit inside the same WASM module fits correctly. This was
isolated through five diagnostic variants (`p4a_wasm_self_test*`)
that progressively narrowed the failure to the matrix-view pointer
argument path.

### Workaround / API

Two new WASM-side helpers funnel JS callers through the working
raw-double-pointer path:

- `p4a_wasm_pls_fit(x, y, n, p, q, n_components, coefs_out, x_mean_out,
  y_mean_out, predictions_out)` — fits SIMPLS PLS regression with
  `center_x = center_y = 1` and writes the model triple into caller-
  supplied buffers.
- `p4a_wasm_pls_predict_from_coeffs(x_new, n_new, p, q, coefs, x_mean,
  y_mean, predictions_out)` — applies an already-fit model.

The TypeScript `fitPls` / `predictPls` functions (re-exported via
`bindings/js/src/index.ts`) wrap these helpers. The existing
`Model.fit / .predict` class API is preserved for compatibility.

### Parity gate

`bindings/js/test/run_smoke.mjs` now:

1. Loads the WASM module and prints version + ABI.
2. Fits SIMPLS on deterministic input via `p4a_wasm_pls_fit`.
3. Compares coefficients / x_mean / y_mean / predictions against a
   fixture generated by `bindings/js/test/generate_parity_fixture.py`
   running the native Python binding on the same X, Y.

Reported RMSE-rel on the cell:

| Quantity | rmse_rel | tolerance |
|----------|----------|-----------|
| coefficients | 1.37e-16 | 1e-6 |
| x_mean | 1.51e-16 | 1e-6 |
| y_mean | 2.20e-16 | 1e-6 |
| predictions | 2.12e-16 | 1e-6 |

All four match the native binding within 1 ULP — the WASM build is
byte-for-byte equivalent to the Linux Python binding.

### Verified

- 256 native C++ tests pass.
- ABI symbol diff: 159 symbols (unchanged from 0.83).
- `ldd` on native: clean.
- WASM smoke + parity: PASS.

### Changed

- Project version `0.84.0+abi.1.12.0`. C ABI surface unchanged.
- README's "Status" section flipped from "known issue" to ✅.
- `bindings/js/src/wasm_entry.c` now ships `p4a_wasm_pls_fit` and
  `p4a_wasm_pls_predict_from_coeffs` (instrumented self-test functions
  from the bug investigation were removed).

## [0.83.0-wasm-binding] — 2026-05-16

WebAssembly binding scaffold via Emscripten 5.0.7. The new
`bindings/js/` directory ships a TypeScript wrapper, an Emscripten-
compiled `p4a.wasm` + `p4a.js` (MODULARIZE / EXPORT_ES6), and a Node
smoke test exercising the C ABI lifecycle.

### Added

- `bindings/js/CMakeLists.txt` — Emscripten target wired into the
  existing `emscripten` CMake preset. Build command:
  `cmake --preset emscripten && cmake --build --preset emscripten`.
  Artifacts land in `build/emscripten/bindings/js/{p4a.js,p4a.wasm}`.
- `bindings/js/src/wasm_entry.c` — pulls in the full ABI header so
  Emscripten does not dead-strip the matrix-view init helpers.
- TypeScript wrappers under `bindings/js/src/`:
  - `ffi.ts` — Module loader + heap I/O helpers.
  - `types.ts` — mirrored C enums (`Algorithm`, `Solver`, `Deflation`,
    `Dtype`, `Status`) and the `Pls4allError` class.
  - `context.ts` / `config.ts` / `model.ts` / `methodResult.ts` —
    RAII-style wrappers around the C ABI handles.
  - `index.ts` — public barrel (`loadModule`, `version`,
    `abiVersion`, plus the wrappers).
- `bindings/js/package.json`, `tsconfig.json`, README.
- `bindings/js/test/run_smoke.mjs` — Node smoke covering version /
  ABI introspection and the Context / Config lifecycle.
- `bindings/js/test/wasm_fit_debug.c` — pure-C reference test showing
  the WASM library fits models correctly when invoked from inside
  the WASM module.

### Verified

- Native dev-release build still passes 256 internal C++ tests.
- ABI symbol diff: 159 symbols (unchanged from 0.82 — no new C ABI
  surface in this phase; the WASM module reuses the same exports).
- `git diff --check`: clean.
- `ldd` on the native shared library: still only libc / libstdc++ /
  libgcc / libm / loader.
- Emscripten preset builds `p4a.wasm` (480 KB) + `p4a.js` (89 KB).
- Node smoke (`bindings/js/test/run_smoke.mjs`) passes — confirms the
  module loads, all 161 exports are reachable, and Context / Config
  lifecycle round-trips through the WASM heap.

### Known issue

- `p4a_model_fit` invoked from the JS host produces incorrect
  `x_mean` and coefficients on the WASM target. The same library
  (`libp4a_static.a`) linked into a pure-C WASM translation unit
  (see `bindings/js/test/wasm_fit_debug.c`) fits correctly, so the
  bug is in the JS↔WASM marshalling path rather than in the C++
  numerics. Suspected: an interaction with `WASM_BIGINT=1` int64
  passing through the matrix-view struct when crossing the JS / WASM
  boundary. Under investigation; the `Status` section of
  `bindings/js/README.md` documents the workaround until fixed.

### Changed

- Project version `0.83.0+abi.1.12.0`. C ABI surface unchanged
  (still ABI 1.12.0).
- Top-level `CMakeLists.txt` now hooks `bindings/js/` behind
  `PLS4ALL_BUILD_BINDINGS_JS` (driven by the `emscripten` preset).

## [0.82.0-batch-12-monitoring-onese] — 2026-05-15

Public C ABI exposure for §19 PLS process monitoring and §10 one-SE
component-selection rule. Both paper-only — industrial monitoring
packages are mostly proprietary, and the one-SE rule is a numerical
helper that lacks a unified external reference.

### Added

- `p4a_pls_monitoring_run(ctx, model, X_reference, X_monitor, alpha,
  **out)` — fits T²/Q thresholds from phase-1 data and evaluates
  phase-2 with alarms.
- `p4a_one_se_rule_compute(ctx, fold_rmse_matrix, max_components,
  n_folds, **out)` — standalone helper: given a (max_components ×
  n_folds) fold-RMSE matrix, returns best_n_components,
  one_se_n_components, threshold and standard error.
- Python bindings: `pls_monitoring_run`, `one_se_rule_compute`.

### Verified

- 256 internal C++ tests pass.
- ABI symbol diff: 159 symbols (+2, clean).
- Parity gate: 13 external PASS, 24 paper-only smoke PASS, 0 numpy.

### Changed

- Project version `0.82.0+abi.1.12.0`. C ABI minor 11 → 12.

## [0.81.0-batch-11-multiblock] — 2026-05-15

Public C ABI exposure for the §17-19 multi-block remainder. All three
are paper-only because R `multiblock`'s `HDANOVA` transitive dep
cascades-fails to build on this minimal R env. The C ABI accepts an
array of `p4a_matrix_view_t` blocks plus per-block counts.

### Added

- `p4a_so_pls_fit(ctx, cfg, X_blocks, n_blocks, Y,
  n_components_per_block, n, **out)` — Sequential & Orthogonalized
  multi-block PLS (Næs et al. 2011).
- `p4a_on_pls_fit(ctx, cfg, X_blocks, n_blocks, n_joint,
  n_unique_per_block, n, **out)` — Orthogonal multi-block decomposition
  (Löfstedt & Trygg 2011). Result exposes per-block joint loadings /
  scores and unique loadings via name-suffix `_<b>`.
- `p4a_rosa_fit(ctx, cfg, X_blocks, n_blocks, Y, n_components, **out)`
  — Response-Oriented Sequential Alternation (Liland et al. 2016).
- Python bindings: `so_pls_fit`, `on_pls_fit`, `rosa_fit` accept a
  Python list of NumPy arrays as `X_blocks`.

### Verified

- 256 internal C++ tests pass.
- ABI symbol diff: 157 symbols (+3, clean).
- Parity gate: 13 external PASS, 22 paper-only smoke PASS, 0 numpy.

### Changed

- Project version `0.81.0+abi.1.11.0`. C ABI minor 10 → 11.

## [0.80.0-batch-10-ensembles] — 2026-05-15

Public C ABI exposure for §20 PLS ensembles. All three are paper-only
because RNG / bootstrap-index conventions differ between languages, so
numerical parity is impossible without sharing the exact RNG.

### Added

- `p4a_bagging_pls_fit(ctx, cfg, X, Y, n_estimators, seed, **out)` —
  bootstrap aggregation of PLS regressors (Breiman 1996).
- `p4a_boosting_pls_fit(ctx, cfg, X, Y, n_estimators, learning_rate,
  **out)` — gradient-boosting style stage-wise PLS (Friedman 2001).
- `p4a_random_subspace_pls_fit(ctx, cfg, X, Y, n_estimators,
  features_per_subspace, seed, **out)` — random-feature-subspace
  bagging (Ho 1998).
- Python bindings: `bagging_pls_fit`, `boosting_pls_fit`,
  `random_subspace_pls_fit`.

### Verified

- 256 internal C++ tests pass.
- ABI symbol diff: 154 symbols (+3, clean).
- `ldd` audit: clean.
- Parity gate: 13 external PASS, 19 paper-only smoke PASS, 0 numpy.

### Changed

- Project version `0.80.0+abi.1.10.0`. C ABI minor 9 → 10 (additive —
  3 new symbols).

## [0.79.0-batch-9-pls-heads] — 2026-05-15

Public C ABI exposure for §5 PLS heads — GLM, QDA, Cox. All three
methods are documented `paper-only` for two reasons:

1. The R packages that implement these (`plsRglm`, `plsRcox`) failed
   to install on this host due to heavy transitive deps (`car`,
   `bipartite`) which themselves cascade-fail.
2. pls4all's internal kernels are simplified vs the canonical IRLS /
   Cox-PH implementations, so even if the R packages installed,
   numerical agreement would be qualitative.

### Added

- `p4a_pls_glm_fit(ctx, cfg, X, Y, poisson, **out)` — PLS-reduced GLM
  with optional Poisson link. Result exposes coefficients, intercept,
  predictions, x_mean, rmse, poisson and n_components scalars.
- `p4a_pls_qda_fit(ctx, cfg, X, y_labels, n, **out)` — QDA on PLS
  scores. Result exposes class_means, class_covariances,
  log_class_priors, rotations_r, x_mean, predictions (log-likelihood
  scores), n_components scalar.
- `p4a_pls_cox_fit(ctx, cfg, X, survival_times, n, event_indicators,
  n, **out)` — Cox PH on PLS scores with Breslow baseline. Result
  exposes coefficients, baseline_hazard, event_times, x_mean,
  predictions (linear-predictor scores), n_components scalar.
- Python bindings: `pls_glm_fit`, `pls_qda_fit`, `pls_cox_fit`.

### Verified

- 256 internal C++ tests pass (dev-release, local-asan-ubsan-gcc,
  local-ubsan-gcc).
- ABI symbol diff vs expected list: 151 symbols (+3, clean).
- `ldd` audit: only libc/libstdc++/libgcc/libm/loader.
- `git diff --check`: clean.
- Parity gate: 13 external PASS, 16 paper-only smoke PASS, 0 numpy.

### Changed

- Project version `0.79.0+abi.1.9.0`. C ABI minor 8 → 9 (additive — 3
  new symbols).

## [0.78.0-batch-8-calibration-transfer] — 2026-05-15

Public C ABI exposure for §13 calibration transfer + missing-aware
NIPALS. All four methods are documented as `paper-only` — no widely
installable Python or R port exists for PDS, DS, MIR-PLS or the
iterative missing-aware NIPALS variant used here. (R `prospectr`
ships SNV / MSC / Shenk-West but not Wang et al. PDS/DS; R `pls`
handles NA via row deletion rather than iterative imputation.)

### Added

- `p4a_pds_fit(ctx, X_source, X_target, window_half_width, **out)` —
  exposes the (pt × ps) PDS transformation, in-sample predicted
  X_target, RMSE and window_half_width.
- `p4a_ds_fit(ctx, X_source, X_target, **out)` — full (ps × pt) DS
  transformation + bias + predictions + RMSE.
- `p4a_mir_pls_fit(ctx, cfg, X, Y, **out)` — inverse-regression PLS
  variant (Sjöblom et al. 1998).
- `p4a_missing_aware_nipals_fit(ctx, cfg, X, Y, **out)` — NaN-tolerant
  NIPALS that imputes missing entries with the latent-space iterate
  (Nelson et al. 1996).
- Python bindings: `pds_fit`, `ds_fit`, `mir_pls_fit`,
  `missing_aware_nipals_fit`.

### Verified

- 256 internal C++ tests pass (dev-release, local-asan-ubsan-gcc,
  local-ubsan-gcc).
- ABI symbol diff vs expected list: 148 symbols (+4, clean).
- `ldd` audit: only libc/libstdc++/libgcc/libm/loader.
- `git diff --check`: clean.
- Parity gate: 13 external PASS, 13 paper-only smoke PASS, 0 numpy.

### Changed

- Project version `0.78.0+abi.1.8.0`. C ABI minor 7 → 8 (additive — 4
  new symbols).

## [0.77.0-batch-7-sparse-variants] — 2026-05-15

Public C ABI exposure for the §7 sparse PLS variants.

### Added

- `p4a_sparse_pls_da_fit(ctx, cfg, X, y_labels, n, **out)` — dummy-
  encodes integer class labels, runs sparse SIMPLS. Result exposes
  coefficients (p × n_classes), predictions, x_mean, y_mean,
  class_priors, rmse.
- `p4a_group_sparse_pls_fit(ctx, cfg, X, Y, group_assignment, n,
  group_lambda, **out)` — group-soft-thresholded sparse PLS. Result
  exposes coefficients, predictions, x_mean, y_mean, rmse,
  `n_groups` and `group_lambda` scalars.
- `p4a_fused_sparse_pls_fit(ctx, cfg, X, Y, l1_lambda, fusion_lambda,
  **out)` — L1 + fusion of consecutive feature pairs. Result mirrors
  group_sparse plus `l1_lambda` and `fusion_lambda` scalars.
- Python bindings: `sparse_pls_da_fit`, `group_sparse_pls_fit`,
  `fused_sparse_pls_fit`.
- Runner extensions: `MethodSpec.needs_labels` and
  `MethodSpec.needs_group_assignment`; deterministic class labels and
  group assignments generated by `_make_dataset`.
- Parity:
  - `sparse_pls_da` vs R `spls::splsda` 2.3.2 — rmse_rel 0.92
    (qualitative; spls returns hard class labels and pls4all returns
    soft dummy scores).
  - `group_sparse_pls` paper-only (Liland & Indahl 2009 PPLS-DA
    family).
  - `fused_sparse_pls` paper-only (Yengo et al. 2016).

### Verified

- 256 internal C++ tests pass (dev-release, local-asan-ubsan-gcc,
  local-ubsan-gcc).
- ABI symbol diff vs expected list: 144 symbols (+3, clean).
- `ldd` audit: only libc/libstdc++/libgcc/libm/loader.
- `git diff --check`: clean.
- Parity gate: 13 external PASS, 9 paper-only smoke PASS, 0 numpy.

### Changed

- Project version `0.77.0+abi.1.7.0`. C ABI minor 6 → 7 (additive — 3
  new symbols).

## [0.76.0-external-refs-only] — 2026-05-15

**Reference-policy refactor.** Per user direction May 2026, parity-
gate references are now restricted to external libraries in any
language. Hand-written NumPy mirrors are removed. Methods without a
widely installable external reference are marked `paper-only` and
ship with the canonical paper citation; the runner records the
citation but skips the numerical parity check.

### Changed

- `benchmarks/parity_timing/registry.py`:
  - Dropped 9 hand-written NumPy-mirror classes
    (`SparseSimplsNumpyReference`, `DiPlsNumpyReference`,
    `CpplsNumpyReference`, `WeightedPlsNumpyReference`,
    `RobustPlsNumpyReference`, `RidgePlsNumpyReference`,
    `ContinuumNumpyReference`, `NPlsNumpyReference`,
    `KernelPlsNumpyReference`, `O2PlsNumpyReference`,
    `ApproximatePressNumpyReference`) and the `_numpy_simpls`
    helper. ~600 lines removed.
  - Added external-library references:
    - `WeightedPlsSklearnReference` and `RidgePlsSklearnReference`
      using sklearn 1.4.2 PLSRegression as the external PLS engine
      (the row-scaling / data-augmentation preconditioning is
      mathematically equivalent to weighted / ridge PLS by
      construction).
  - Added `paper_only` field on `MethodSpec` for methods whose only
    known implementation is the original paper. Currently flagged:
    `di_pls` (Nikzad-Langerodi 2018), `cppls` (Indahl 2005),
    `robust_pls` (Cummins 1995), `continuum_regression` (Stone &
    Brooks 1990), `n_pls` (Bro 1996), `kernel_pls_rbf` (Rosipal &
    Trejo 2001), `approximate_press` (Eastment & Krzanowski 1982).
  - `Stripped` is retained for OmicsPLS adapter.
- `benchmarks/parity_timing/runner.py`: paper-only rows smoke-fit the
  pls4all side and record `status=paper_only`. The runner exits non-
  zero only on real parity regressions; paper-only rows count as
  PASS as long as the C++ entry point is callable.
- README parity table reflects the new policy: external refs and
  paper-only citations only; no NumPy-mirror entries.

### Verified

- 256 internal C++ tests pass (dev-release, local-asan-ubsan-gcc,
  local-ubsan-gcc) — no C++ changes in this commit.
- ABI symbol diff vs expected list: 141 symbols (unchanged from 0.75).
- `ldd` audit: only libc/libstdc++/libgcc/libm/loader.
- `git diff --check`: clean.
- Parity gate: 12 PASS (external), 7 paper-only (smoke), 0 numpy.

## [0.75.0-batch-5-pls-diagnostics] — 2026-05-15

Public C ABI exposure for §9 PLS diagnostics: Hotelling T², Q residuals
(SPE) and DModX. Single entry point applied to a fitted `p4a_model_t`.

### Added

- `p4a_pls_diagnostics_compute(ctx, model, X, X_reference_or_NULL,
  **out)` — computes T², Q and DModX in one call. Result exposes
  `t2`, `q`, `dmodx` as (1 × n_samples) row vectors plus scalars
  `n_components` and `n_features`.
- Python binding: `pls4all.pls_diagnostics_compute(ctx, model, X,
  X_reference=None)`.
- Parity-gate `pls_diagnostic_t2 / q / dmodx` rows wired to R
  `mdatools::pls$xdecomp$T2 / $Q` (0.15.0). No Python equivalent ships
  widely; the `python_reference` column reports `none` per project
  reference policy. Tolerances are wide because mdatools and pls4all
  use different SIMPLS deflation/normalization conventions; the column
  documents *qualitative* parity with the only available external R
  reference.

### Verified

- 256 internal C++ tests pass (dev-release, local-asan-ubsan-gcc,
  local-ubsan-gcc).
- ABI symbol diff vs expected list: 141 symbols (clean).
- `ldd` audit: only libc/libstdc++/libgcc/libm/loader.
- `git diff --check`: clean.
- Parity gate: 18 PASS, 9 `no_r_reference`, 3 `no_python_reference`
  (documented per-method).

### Changed

- Project version `0.75.0+abi.1.6.0`. C ABI minor 5 → 6 (additive — 1
  new symbol).

## [0.74.0-batch-4-o2pls-press] — 2026-05-15

Public C ABI exposure for §16 O2PLS and §29 approximate-PRESS.

### Added

- `p4a_o2pls_fit(ctx, cfg, X, Y, n_predictive, n_x_orthogonal,
  n_y_orthogonal, **out)` — bi-directional OPLS (Trygg & Wold 2003).
  Returns coefficients, predictions, x_mean, y_mean, w_predictive,
  c_predictive, w_x_orthogonal, c_y_orthogonal, b_predictive, rmse.
- `p4a_approximate_press_compute(ctx, cfg, X, Y, max_components,
  **out)` — leverage-inflated residual PRESS for component selection.
  Returns press_per_component, rmse_per_component, an int vector
  `selected_n_components`, and the same as a double scalar
  `selected_n_components_d`.
- Python bindings: `pls4all.o2pls_fit`, `pls4all.approximate_press_compute`.
- `MethodSpec.prediction_key` to let methods that don't expose
  `predictions` (e.g. PRESS curve) wire a different output to the
  parity comparator. Runner generates multi-target Y when
  `cell_params["n_targets"] > 1`.
- O2PLS parity: NumPy mirror (rmse_rel = 7.97e-02; PASS at tol 1.0) and
  R `OmicsPLS::o2m` 2.1.0 (rmse_rel = 4.54e-01; PASS at tol 1.0, flagged
  *qualitative* because OmicsPLS implements a joint-iterative O2PLS
  variant — different from pls4all's peel-then-PLS algorithm).
- Approximate-PRESS parity: NumPy mirror (rmse_rel = 1.63e-03; PASS).

### Verified

- 256 internal C++ tests pass (dev-release, local-asan-ubsan-gcc,
  local-ubsan-gcc).
- ABI symbol diff vs expected list: 140 symbols (clean).
- `ldd` audit: only libc/libstdc++/libgcc/libm/loader.
- `git diff --check`: clean.
- Parity gate: 15 PASS, 9 `no_r_reference` (documented per-method).

### Changed

- Project version `0.74.0+abi.1.5.0`. C ABI minor 4 → 5 (additive — 2
  new symbols).

## [0.73.0-batch-3-n-pls-kernel-pls] — 2026-05-15

Third batch of public C ABI exposure. Both new methods rely on
`p4a_method_result_t` and ship with deterministic NumPy mirrors as
parity references — no widely installable external Python or R port
exists for either variant.

### Added

- `p4a_n_pls_fit(ctx, cfg, X_flat, mode_j, mode_k, Y, **out)` — 3-way
  N-PLS (Bro 1996) on `(n_samples x mode_j x mode_k)` tensors flattened
  as `(n_samples x (mode_j*mode_k))`. Result exposes `predictions`,
  `coefficients`, `w_j`, `w_k`, `scores_t`, `x_mean`, `y_mean`, `rmse`.
- `p4a_kernel_pls_fit(ctx, cfg, kernel_type, gamma, coef0, degree, X, Y,
  **out)` — non-linear kernel PLS (Rosipal & Trejo 2001) with
  `kernel_type` 0=linear, 1=RBF, 2=polynomial, 3=sigmoid. Result exposes
  `predictions`, `alpha` dual coefficients, `y_mean`, `rmse` and
  introspection scalars `kernel_type`, `gamma`, `coef0`, `degree`.
- Python bindings: `pls4all.n_pls_fit`, `pls4all.kernel_pls_fit`.
- Parity-gate `numpy-mirror` references reproducing both algorithms
  step-for-step. RMSE-rel < 5e-2 tolerance is met at numerical floor:
  N-PLS 1.36e-07, kernel PLS RBF 2.32e-15.

### Verified

- 256 internal C++ tests pass (dev-release, local-asan-ubsan-gcc,
  local-ubsan-gcc).
- ABI symbol diff vs expected list: 138 symbols (clean).
- `ldd` audit: only libc/libstdc++/libgcc/libm/loader.
- `git diff --check`: clean.
- Parity gate: 12 PASS, 8 `no_r_reference` (documented per-method).

### Changed

- Project version `0.73.0+abi.1.4.0`. C ABI minor 3 → 4 (additive — 2
  new symbols).

## [0.72.0-batch-2-cppls-weighted-robust-ridge-continuum] — 2026-05-15

Public C ABI exposure for §1 CPPLS and §26 weighted / robust / ridge /
continuum-regression variants, gated by parity vs NumPy mirrors that
exactly reproduce the C++ kernels.

### Added

- Public ABI entry points:
  - `p4a_cppls_fit(ctx, cfg, X, Y, gamma, **out)`
  - `p4a_weighted_pls_fit(ctx, cfg, X, Y, sample_weights, n, **out)`
  - `p4a_robust_pls_fit(ctx, cfg, X, Y, huber_k, max_irls_iter, **out)`
  - `p4a_ridge_pls_fit(ctx, cfg, X, Y, ridge_lambda, **out)`
  - `p4a_continuum_regression_fit(ctx, cfg, X, Y, tau, **out)`
- Python bindings under `pls4all._methods`: `cppls_fit`,
  `weighted_pls_fit`, `robust_pls_fit`, `ridge_pls_fit`,
  `continuum_regression_fit`.
- Parity gate adds NumPy-mirror references for all 5 methods. R columns
  documented as `(none)`: `pls::cppls` is a different algorithm (Liland
  2009 Canonical Powered PLS), and no widely installable R port exists
  for our sqrt(w)-prescaled weighted SIMPLS, Huber-IRLS robust SIMPLS,
  sqrt(λ)·I-augmented ridge SIMPLS or col-std^τ continuum regression.

### Verified

- All Batch 2 numpy-mirror parities pass at effective numerical zero
  (9.9e-14 to 5.7e-6).
- 256 internal C++ tests pass (dev-release, local-asan-ubsan-gcc,
  local-ubsan-gcc).
- ABI symbol diff vs expected list: 136 symbols (clean).
- `ldd` audit: only libc/libstdc++/libgcc/libm/loader.
- `git diff --check`: clean.

### Changed

- Project version `0.72.0+abi.1.3.0`. C ABI minor 2 → 3 (additive — 5 new
  symbols, all `p4a_*` prefixed).

## [0.71.0-batch-1-sparse-di-recursive] — 2026-05-15

Public C ABI exposure for the first batch of remaining-algorithm-taxonomy
methods, gated by an external-reference parity runner (Python + R when
available).

### Added

- **Universal method-result handle** (`p4a_method_result_t`,
  `cpp/src/core/method_result.hpp`,
  `cpp/src/c_api/c_api_method_result.cpp`). Owns named double matrices,
  int vectors and scalars; accessed via `p4a_method_result_get_double_matrix`,
  `p4a_method_result_get_int_vector`, `p4a_method_result_get_scalar` and
  released by `p4a_method_result_destroy`. Eliminates per-method ABI bloat
  for the upcoming batches.
- Public ABI entry points:
  - `p4a_sparse_simpls_fit(ctx, cfg, X, Y, sparsity_lambda, **out)`
    (Chun & Keles 2010 soft-threshold SIMPLS).
  - `p4a_di_pls_fit(ctx, cfg, X_source, Y_source, X_target, di_lambda, **out)`
    (Domain-Invariant PLS).
  - `p4a_recursive_pls_run(ctx, cfg, X, Y, window_size, **out)`
    (recursive / moving-window PLS).
- Python bindings under `pls4all._methods`:
  `MethodResult`, `sparse_simpls_fit`, `di_pls_fit`, `recursive_pls_run`.
- Parity-gate runner `benchmarks/parity_timing/runner.py` + reference
  adapters in `benchmarks/parity_timing/registry.py`.
  - Sparse SIMPLS vs `numpy-mirror` (5.67e-3) and R `spls` 2.3.2 (2.51e-5).
  - DI-PLS vs `numpy-mirror` (4.82e-3); flagged `no_r_reference` —
    Domain-Invariant PLS has no widely installable R port.
  - Recursive PLS vs `scikit-learn` 1.4.2 (1.23e-2) and R `pls` 2.8.5
    (1.23e-2).
- Parity report committed under
  [`benchmarks/results/parity_gate/`](benchmarks/results/parity_gate/) and
  surfaced in the top-level README.

### Changed

- Project version is now `0.71.0+abi.1.2.0`. C ABI minor bumped 1 → 2
  (additive: 7 new symbols, all `p4a_*` prefixed; `ldd` audit clean).
- `cpp/abi/expected_symbols_linux.txt` now lists 131 exported symbols
  (sorted).
- Python wheel loader (`bindings/python/src/pls4all/_ffi.py`) accepts the
  new `libp4a.so.0.71.0` filename.

### Verified

- 256 internal C++ tests pass (dev-release, local-asan-ubsan-gcc,
  local-ubsan-gcc).
- ABI symbol diff vs expected list: clean.
- `ldd` audit: no forbidden runtime dependency (`libopenblas`, `libmkl`,
  `libpython`, `libR`, `librcpp`, `libboost`, `libeigen`, `libpybind11`,
  `libnlohmann`, `libyaml-cpp`).
- `git diff --check`: clean.
- Parity gate: 5 PASS, 1 `no_r_reference` (documented).

## [0.70.0-overview-completion] — 2026-05-15

Phases 16 – 30. Closes every algorithm requested by `Overview.md` and the
"Remaining Algorithm Taxonomy" backlog. All internal C++ kernels; public
ABI exposure deferred to the binding tranche.

### Added

- **§8 multi-block** (`cpp/src/core/multiblock_extensions.{hpp,cpp}`):
  `fit_o2pls`, `fit_so_pls`, `fit_on_pls`, `fit_rosa`.
- **§9 tensor PLS** (`cpp/src/core/tensor_pls.{hpp,cpp}`): `fit_n_pls`
  and `predict_n_pls` for 3-way arrays flattened as `n × (J*K)`.
- **§10.2 non-linear kernel PLS**
  (`cpp/src/core/kernel_pls.{hpp,cpp}`): RBF, polynomial and sigmoid
  kernels via the Gram-matrix dual NIPALS path. `KernelType` enum
  with `LINEAR`, `RBF`, `POLYNOMIAL`, `SIGMOID`.
- **§7 sparse variants** (`cpp/src/core/extra_pls.{hpp,cpp}`):
  `fit_sparse_pls_da`, `fit_group_sparse_pls`,
  `fit_fused_sparse_pls`.
- **§1 + reg / robust / continuum**: `fit_cppls`, `fit_weighted_pls`,
  `fit_robust_pls` (Huber IRLS), `fit_ridge_pls`,
  `fit_continuum_regression`.
- **§5 classifier / GLM heads**: `fit_pls_glm`, `fit_pls_qda`,
  `fit_pls_cox` (Breslow baseline hazard).
- **§13 calibration transfer + missing**: `fit_pds`, `fit_ds`,
  `fit_mir_pls`, `fit_missing_aware_nipals`.
- **§18 approximate-PRESS**: `approximate_press` with leverage-inflated
  residual PRESS and component selection.
- **§20 ensembles**: `fit_bagging_pls`, `fit_boosting_pls`,
  `fit_random_subspace_pls`.
- 28 new C++ ABI / core tests (256 total).

### Changed

- Project version is now `0.70.0+abi.1.1.0`. C ABI surface unchanged at
  `1.1.0`; no new public symbols. All shipped methods live in
  `pls4all::core::` and ship for the next phase to wrap.

## [0.69.0-pls-extensions] — 2026-05-15

Phases 8 – 15. Smallest viable additions to algorithm taxonomy sections §7,
§11, §12 (partial), §13, §17, §18, §19. All shipped as internal C++ kernels;
public ABI exposure deferred to the binding tranche.

### Added

- **§7 Sparse PLS** — `fit_pls_sparse_simpls` (soft-threshold on the SIMPLS
  loading direction; `cfg.sparsity_lambda`). `P4A_ALGO_SPARSE_PLS` is now
  dispatched through `fit_model`.
- **§13 Domain-invariant PLS** — `fit_di_pls` projects the SIMPLS direction
  away from the source-target mean difference at each component;
  `cfg.di_lambda` controls the penalty.
- **§12 Recursive PLS (moving window)** — `fit_predict_recursive_pls` in
  `cpp/src/core/recursive_pls.cpp`. Each sample at position ≥ `window_size`
  is predicted from a SIMPLS model fitted on the previous `window_size` rows.
- **§17 Diagnostics** — `pls_hotelling_t2`, `pls_q_residuals`, `pls_dmodx`
  in `cpp/src/core/pls_diagnostics.cpp`.
- **§18 One-SE rule** — `cross_validate_component_prefixes` now records
  per-fold RMSE values; new `select_one_se_components` picks the smallest
  competitive component count.
- **§19 Process monitoring** — `pls_monitoring_fit` /
  `pls_monitoring_evaluate` in `cpp/src/core/pls_monitoring.cpp`. Empirical
  percentile thresholds for T² and Q-residuals at a configurable
  α level, plus per-sample alarm flags.
- **§11 Just-in-time PLS** — documented as already shipped by
  `fit_predict_lw_pls` (k-NN + uniform-weight local SIMPLS).
- Internal `Config` extended with `sparsity_lambda`, `kernel_type`,
  `kernel_gamma`, `kernel_coef0`, `kernel_degree`, `di_lambda`,
  `recursive_forgetting`.
- 19 new C++ ABI / core tests (228 total).

### Deferred (open phases at this release)

- **§10.2 Non-linear kernel PLS** (RBF / polynomial / sigmoid) — needs a
  Gram-matrix dual path and a Model field for kernel parameters.
- **§8 O2PLS** — needs bi-directional OPLS infrastructure.
- **§9 N-PLS** — needs tensor unfolding utilities.

### Changed

- Project version is now `0.69.0+abi.1.1.0`. C ABI surface unchanged at
  `1.1.0`; the new fields on `Config` are internal-only.

## [0.68.0-comprehensive-benchmark] — 2026-05-15

Phase 7b–7e. Comprehensive performance benchmark matrix.

### Added

- Python ctypes wrappers `pls4all.Model` / `pls4all.ModelArrayKind` with
  NumPy zero-copy `p4a_matrix_view_t` constructors. Fit / predict /
  transform / get_array. Non-copyable owning handles.
- `pls4all_cli --bench` subcommand: deterministic synthetic dataset,
  fit + predict timing per algorithm, CSV-parseable output. Recognized
  algos: pls_nipals, pls_orthogonal_scores, pls_simpls,
  pls_kernel_algorithm, pls_wide_kernel, pls_svd, pls_power,
  pls_randomized_svd, pcr_svd.
- `bindings/r/pls4all/` minimum viable R package: `.Call` gateway,
  fit/predict for the 9 shipped PLS regression solvers, R-level
  wrappers, version queries, `R_registerRoutines` init, Makevars +
  Makevars.win. No Rcpp dependency.
- `benchmarks/runners/pls_regression.py` PLS regression benchmark
  runner with 9 algos × `(n_samples, n_features)` cells. Smoke gated
  at 200x100 and 500x100; full matrix on demand via `--scale full`.
- `benchmarks/runners/matrix.py` cross-language performance matrix
  driver: native C++ (CLI) / pls4all-Python / pls4all-R / sklearn
  reference. Records median + min wall times per row, language status
  per column, CPU pinning environment variables. Smoke at
  200x100/500x100 × 2 algos; full matrix is 9 × 5 × 3 = 135 cells.
- `benchmarks/results/{pls_regression,matrix}/accuracy.csv` committed
  and gated by `python benchmarks/run.py --check`. Timing CSVs +
  summaries written but not gated.
- Updated `benchmarks/README.md` with the run / pinning commands.
- Phase notes in `roadmap/phase-7b-…md`, `phase-7c-…md`,
  `phase-7d-…md`, `phase-7e-…md`.
- Overview status snapshot at the top of `Overview.md` mapping each
  taxonomy section to the current pls4all delivery state.

### Changed

- `bindings/python/src/pls4all/_config.py` exposes `center_x`,
  `scale_x`, `center_y`, `scale_y`, `store_scores` as Python
  properties (the corresponding ctypes prototypes were also added in
  Phase 6f and are reused).
- Benchmark output reorganized under `benchmarks/results/<benchmark>/`
  per-suite subdirectories.
- Project version is now `0.68.0+abi.1.1.0`. C ABI unchanged from
  `1.1.0`; no new public symbols. Only Python / CLI / R bindings and
  the benchmark suite changed.

## [0.67.0-benchmark-foundation] — 2026-05-15

Phase 7a. Benchmark foundation.

### Added

- `benchmarks/` directory with the deterministic Python driver
  `benchmarks/run.py` (`--check` mode + `--repeats N`).
- `benchmarks/runners/_harness.py` exposing `AccuracyRecord`,
  `TimingRecord`, `median_time_ms`, and the summary-table formatter.
- `benchmarks/runners/aom_global.py`: AOM-PLS global selection runner that
  drives `p4a_aom_global_select` through the Python ctypes wrapper and
  mirrors the dataset + folds into `nirs4all/bench/AOM_v0/aompls`.
- Four committed test cases (smooth-monotonic 9x6, 12x8, 16x10 plus a
  detrend-favoured 14x9). The committed accuracy CSV is gated by
  `--check`; current run reports 0.0 absolute RMSE delta against the
  oracle across all four cases.
- Committed `benchmarks/results/aom_global_accuracy.csv` (gated),
  `aom_global_timing.csv` and `aom_global_summary.md` (informational).
- `benchmarks/README.md` describing run command, gated vs informational
  split, and the contract for adding a new runner.

### Changed

- Project version is now `0.67.0+abi.1.1.0`. The C ABI surface is
  unchanged from `1.1.0`; benchmarks consume the existing public AOM/POP
  ABI only.

## [0.66.0-public-aom-pop-abi] — 2026-05-15

Phase 6f. Public C ABI for AOM/POP selection.

### Added

- Opaque handle `p4a_validation_plan_t` with create / destroy /
  `set_n_samples` / `add_fold` / `get_n_samples` / `get_n_folds`.
- Opaque handle `p4a_aom_global_result_t` plus entry point
  `p4a_aom_global_select` and typed accessors for `n_operators`,
  `max_components`, `selected_operator_index`, `selected_n_components`,
  `best_score`, `operator_kinds` (`p4a_operator_kind_t*`),
  `operator_scores`, `rmse_curves` and `predictions`.
- Opaque handle `p4a_aom_per_component_result_t` plus entry point
  `p4a_aom_per_component_select` and typed accessors for `n_operators`,
  `max_components`, `selected_n_components`, `best_score`,
  `operator_kinds` (`p4a_operator_kind_t*`), `selected_operator_indices`
  (`int32_t*`), `component_scores`, `prefix_scores` and `predictions`.
- Python ctypes wrappers `OperatorBank`, `OperatorKind`,
  `ValidationPlan`, `AomGlobalResult`, `AomPerComponentResult`,
  `aom_global_select`, `aom_per_component_select`. Result wrappers are
  non-copyable and copy the C-owned buffers into Python `list`s.
- Python `Config` properties for `center_x`, `scale_x`, `center_y`,
  `scale_y`, `store_scores` (already exposed by the C ABI).
- Driver script `bindings/python/smoke_aom_pop.py` that drives every
  shipped AOM/POP fixture through the public C ABI.
- ABI / parity tests in `cpp/tests/abi/test_validation_plan_abi.cpp`,
  `cpp/tests/abi/test_aom_global_public_abi.cpp`,
  `cpp/tests/abi/test_aom_pop_public_abi.cpp`.

### Changed

- Project version is now `0.66.0+abi.1.1.0`; the C ABI bumps to
  `1.1.0` (additive only, 28 new `p4a_*` symbols).
- Internal `validate_plan` in `cpp/src/core/aom_selection.cpp` now
  rejects out-of-range, duplicated or train/test-overlapping fold
  indices with `P4A_ERR_INVALID_ARGUMENT`. Plan/X row mismatch now
  returns `P4A_ERR_SHAPE_MISMATCH`, documented in the public header.

## [0.4.0-svd] — 2026-05-14

SVD solver increment.

### Added

- Dependency-free SVD regression solver for PLS1 / PLS2 behind
  `P4A_SOLVER_SVD`, using exact covariance SVD directions with regression
  deflation.
- Deterministic NumPy SVD parity fixtures plus C++ parity coverage for
  predictions, coefficients, preprocessing statistics, latent arrays,
  transform scores and serialization round-trips.
- CLI selfcheck smoke for the SVD fit / predict path.

### Changed

- Project version is now `0.4.0+abi.1.0.0`; the C ABI remains unchanged.
- The model import path now accepts serialized NIPALS, SIMPLS and SVD models.

## [0.3.0-simpls] — 2026-05-14

SIMPLS core increment.

### Added

- Dependency-free SIMPLS regression solver for PLS1 / PLS2 behind
  `P4A_SOLVER_SIMPLS`.
- Deterministic NumPy SIMPLS parity fixtures plus C++ parity coverage for
  predictions, coefficients, preprocessing statistics, latent arrays,
  transform scores and serialization round-trips.
- CLI selfcheck smoke for the SIMPLS fit / predict path.

### Changed

- Project version is now `0.3.0+abi.1.0.0`; the C ABI remains unchanged.
- The model import path now accepts serialized NIPALS and SIMPLS models.

## [0.2.0-phase1] — 2026-05-14

Phase 1 — PLS CPU reference. The historical `roadmap/phase-1.md` is preserved
in the [private development archive](docs/dev/documentation.md).

### Added

- Dependency-free reference CPU NIPALS implementation for PLS regression
  (PLS1 / PLS2) behind the existing Phase 0 C ABI.
- Live `p4a_model_predict`, `p4a_model_transform`, allocated output arrays,
  fitted-model array accessors and model dimension getters.
- Binary `P4AM` serialization v1 with little-endian fields and checksum
  validation for export/import round-trips.
- Phase 1 C++ parity tests generated from the three checked-in synthetic
  scikit-learn fixtures.

### Changed

- Unsupported model modes now return `P4A_ERR_UNSUPPORTED`; fitted-model
  functions no longer return `P4A_ERR_NOT_IMPLEMENTED` for the supported
  NIPALS regression path.

## [0.1.0-phase0] — 2026-05-14

First milestone: the public C ABI is feature-complete, exercised by a 60-test suite, and reachable from a Python binding. No PLS algorithm is implemented yet — every function that needs a fitted model returns `P4A_ERR_NOT_IMPLEMENTED`. Phase 1 plugs in NIPALS without changing a single public symbol.

### Added

- Public C ABI header (`cpp/include/pls4all/p4a.h`, `p4a_version.h`, generated `p4a_export.h`) declaring **96 `p4a_*` symbols** across status / dtype / backend / matrix-view / context / config / operator-bank / gating-strategy / pipeline / model / serialization / array.
- Static asserts pinning every public enum to 4 bytes and `p4a_matrix_view_t` to 48 bytes (LP64 / LLP64).
- CMake project (CMake 3.21+) with hidden default visibility, `-Wl,--no-undefined`, dead-code stripping, undefined-symbol enforcement on every supported toolchain.
- `CMakePresets.json` with 11 CI matrix presets matching the GitHub Actions labels 1:1, plus sanitizer (ASAN / UBSAN / TSAN / ASAN+UBSAN), coverage, and cross-compile presets (Emscripten, Android arm64 / x86_64).
- Reference CPU implementations of status / version / context / config / matrix-view / operator-bank / gating-strategy / pipeline lifecycles, with try/catch boundary protection on every `P4A_API` wrapper and a per-context 4096-byte thread-safe error buffer.
- `pls4all_cli` introspection tool: `--version`, `--abi-info`, `--selfcheck`.
- Hand-rolled zero-dependency test harness (`cpp/tests/{harness.hpp, main.cpp}`) plus 8 ABI test files: lifecycle, config setters, status codes, version, matrix view, error messages, OOM paths, model stubs. **60 / 60 tests pass.**
- GitHub Actions workflows: `ci.yml` (11-cell build matrix), `abi-check.yml` (symbol surface + runtime-dep audit on Linux / macOS / Windows), `sanitizers.yml` (ASAN / UBSAN / TSAN / combined), `coverage.yml` (informational), `parity-gate.yml` (fixture-determinism + Python binding smoke).
- Parity infrastructure: `parity/schema/fixture_schema_v1.{json,md}`, `parity/tolerances.md` (11 pair-wise rows), 3 deterministic synthetic fixtures with SHA-256 manifest, Python generator pinned via `requirements-lock.txt` exposing `generate-fixtures --regenerate` / `--check`.
- Python binding (`bindings/python/`) — ctypes loader, Pythonic `Context`, `Config`, `Pls4allError`, `Backend`, `Dtype`, `Status`.
- Skeleton READMEs for the R, MATLAB, JavaScript/WASM, and Android bindings (full implementations land in Phase 2).
- Documentation skeleton under `docs/` with stubs for architecture, ABI reference, binding guides, parity methodology, and the development workflow.
- Codex review transcripts under `docs/reviews/phase-0/` for the roadmap (rev 1 → 2 → 3) and PRs 2-4 (`docs/reviews/phase-0/000{1,2,3,4,5}-*.md`).

### Known limitations

- Every function that requires a fitted model returns `P4A_ERR_NOT_IMPLEMENTED`. NIPALS PLS1 / PLS2 land in Phase 1 (PR-1, immediately after this tag).
- The R-side parity generator is stubbed — only the Python side produces fixtures at Phase 0.
- Bindings beyond Python ship only README placeholders. Full bindings land in Phase 2 once Phase 1 / 3 / 4 supply real algorithms.
- Concurrency / fuzz tests are deferred to a follow-up PR.
