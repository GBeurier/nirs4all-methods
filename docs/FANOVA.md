# Native forest functional ANOVA

`n4m_fanova_fit_v1` fits a fresh bootstrap regression forest to a finite design
and scalar response. It reports descriptive surrogate variance fractions under
an explicitly supplied independent uniform product measure. Each continuous
feature has a finite increasing interval. Each categorical feature has K exact
integer codes, 0 through K-1, all assigned probability 1/K, including categories
absent from the observed sample. Bounds and category cardinality are part of the
scientific question, rather than inferred empirical weights.

For each tree t, let its response surface be f_t and its variance under that
measure be V_t. The main variance for feature j is
`Var(E[f_t | x_j])`. The pair variance for j,k is
`Var(E[f_t | x_j,x_k]) - V_j - V_k`. The adapter enumerates each tree's split
intervals and integrates the piecewise constant conditional means at interval
midpoints, weighted by interval length. It enumerates every category for
categorical axes. These are exact tree integrals, with floating-point roundoff;
they are not Monte Carlo estimates.

Each informative tree's component variance is divided by that tree's own total
variance before averaging. Constant trees are excluded and counted. The reported
sample SD measures variation between these normalized tree fractions; it is not
a confidence interval, standard error, or observation sampling uncertainty.
With one informative tree its SD is unavailable. A remainder reports the mean
fraction left after the requested main effects or main and pair effects; higher
interactions are not silently assigned to the reported pairs. This decomposition
differs from decomposing the ensemble mean surface, whose variance also contains
cross-tree covariances.

Constant responses and forests without informative trees return explicit
statistical refusal states and NaN effect buffers. Numerical violations of the
variance decomposition also withhold all effects. Technical failures, including
exhausting the exact marginal query budget, leave all caller outputs unchanged.
The caller must inspect fit state as well as the technical return status.

Training RMSE describes fit to the supplied observations. Optional validation
data yield RMSE and R-squared; validation R-squared is unavailable for constant
validation responses. No holdout, independence, generalization or causal claim
is inferred. For conditional parameter spaces or correlated inputs, an
independent product measure can include unsupported combinations and extrapolate
far from the observed design. The host must assess that limitation and surrogate
adequacy before interpreting attribution.

Resource limits are 8192 observations, 32 features, 128 trees, depth 12, and up to
10 million marginal queries, with lower initialized defaults. Continuous
coordinates are mapped to unit intervals using the declared domains, and y is
centered and range scaled internally. These affine transformations preserve the
variance fractions. The summary rescales tree variance and prediction errors to
the caller's response units. Extreme cases outside representable output ranges
are statistically refused.

The backend is optional and disabled by default. Its unavailable stub retains
the versioned public C symbols and returns `N4M_ERR_BACKEND_UNAVAILABLE`, without
silently substituting another estimator. Source and license provenance are in
`cpp/backends/rfr/README.md`. The formulas follow Hutter, Hoos and Leyton-Brown
(2014), [An Efficient Approach for Assessing Hyperparameter Importance](https://proceedings.mlr.press/v32/hutter14.html),
ICML, pages 754–762. No code from the separately licensed Python `fanova`
package is used.

Targeted qualification uses `n4m_fanova_tests` against the actual shared C ABI
and `n4m_fanova_kernel_tests` for rapid execution of those same production
sources. Cases cover additive continuous functions, a categorical XOR interaction,
categorical uniform-domain variance, constant response refusal, held-out metrics,
strided views, seed reproducibility, repeated fits, invalid input atomicity and
query-budget refusal.

Seed repeatability is defined within the same compiled artifact. C++ standard
library random-distribution implementations may differ between toolchains;
native/WASM qualification uses mathematical decomposition oracles and numerical
tolerances, rather than promising identical continuous-tree thresholds.
