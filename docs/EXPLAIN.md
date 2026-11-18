# Model explanations (ABI 2.23, importances ABI 2.24)

`n4m/explain.h` provides attribution kernels for any fitted model: Shapley-value
estimators of a feature-group game, LIME, occlusion, exact linear SHAP and
path-dependent TreeSHAP, a local finite-difference gradient explainer (ABI
2.24) and model-intrinsic importance profiles (ABI 2.24, see
[Model-intrinsic importances](#model-intrinsic-importances-abi-224)). The numerics live once in the C++17 core and reach
every host (Python `n4m.explain`, JavaScript/WebAssembly `explainModel`, Rust
FFI) through the same C ABI. Results are exchanged as the
[`nirs4all.explanation.v1`](explain/explanation_v1.schema.json) envelope built
by the host.

Status vocabulary: **implemented** (tested in this release), **experimental**
(implemented, limited validation), **planned** (not implemented).

| Explainer | Entry point | Status | Exactness | Additive |
| --- | --- | --- | --- | --- |
| Exact group SHAP (G <= 16) | session `N4M_EXPLAIN_EXACT_SHAP` | implemented | exact | yes |
| Permutation SHAP (antithetic) | session `N4M_EXPLAIN_PERMUTATION_SHAP` | implemented | sampled, unbiased, with standard errors | yes, by construction |
| Kernel SHAP | session `N4M_EXPLAIN_KERNEL_SHAP` | experimental | exact when every coalition fits the budget, else sampled | yes, enforced |
| LIME over group masks | session `N4M_EXPLAIN_LIME` | experimental | local surrogate | no |
| Occlusion / ablation | session `N4M_EXPLAIN_OCCLUSION` | implemented | perturbation | no |
| Linear SHAP | `n4m_explain_linear_v1` | implemented | exact (interventional) | yes |
| Path-dependent TreeSHAP | `n4m_explain_tree_path_dependent_v1` | implemented | exact | yes |
| Local gradient (ABI 2.24) | session `N4M_EXPLAIN_GRADIENT` | implemented | perturbation (central finite differences) | no (exact for affine models) |
| Integrated Gradients | — | planned | — | — |
| Conditional / on-manifold baselines | — | planned | — | — |

## Concepts

* **Explained space.** `X` holds the rows to explain in the feature space the
  explained function consumes. Explaining a complete fitted pipeline in its raw
  spectral input (`attribution_space = raw_input`) and explaining a terminal
  estimator after preprocessing (`transformed_features`) are different
  questions; the host decides and records which one it asked. The library never
  relabels axes, units or outputs.
* **Outputs.** Each output column is whatever the host predicts: a regression
  target in original units, one class probability or a decision score.
* **Groups.** `feature_group[j]` assigns feature `j` to one of `n_groups`
  groups that partition the features (`NULL` = one group per feature). For the
  session explainers a group is a *player* of the coalition game: the group is
  present or absent as a whole, so group values are true group Shapley values,
  not sums of per-wavelength values. Linear group sums coincide with group
  Shapley values (the model is additive). TreeSHAP group values are sums of
  per-feature values (`group_aggregation = sum_of_feature_values`).
* **Value function.** `v(S)` is the model output when the groups in `S` take
  their values from the explained row and the other groups take baseline
  values. `v(full) = f(x)`; `v(empty)` is the expected value.
* **Additivity.** For the SHAP family,
  `sum_g attributions[r][g][o] + expected[r][o] == predictions[r][o]` up to
  rounding. LIME and occlusion are not additive.

### Baselines

| Baseline | Absent group takes | `v(S)` | When to use |
| --- | --- | --- | --- |
| `BACKGROUND` | the values of each background row | mean over the B background rows | interventional SHAP; cost grows linearly with B |
| `MEAN` | the column mean of the background | one evaluation | cheap reference; a single mean spectrum |
| `INTERPOLATE` | a straight-line bridge in the explained row | one evaluation | contiguous spectral groups; keeps perturbed spectra continuous |

`INTERPOLATE`: every maximal run of absent features (adjacent absent groups
merge) is replaced by the straight line, in feature-index order, joining the
explained row's values at the nearest present features on each side. A run
touching an edge is filled flat with its single present neighbour; when every
feature is absent the row is replaced by the line from its first to its last
value. It requires every group to be a contiguous index range and needs no
background (the expected value is therefore per row).

**NIR spectra.** Neighbouring wavelengths are strongly correlated. Per-wavelength
Shapley values with marginal baselines splice background values into the
explained spectrum, producing off-manifold spectra, and split credit almost
arbitrarily among correlated wavelengths. Prefer grouped explanations
(`n4m_explain_groups_fixed_width_v1` for windows of 10–50 wavelengths, or
`n4m_explain_groups_from_ranges_v1` for chemically informed bands already mapped
to feature indices by the host) together with the `INTERPOLATE` baseline, which
removes a band's absorption feature while keeping the spectrum continuous. Use
`BACKGROUND` with a small representative background (20–100 rows) when an
interventional expectation over the population is required.

## Ask/tell protocol

The session never calls a model. It emits batches of perturbed rows; the host
predicts them with any backend (Python estimator, WASM pipeline, ONNX Runtime,
asynchronous worker) and submits the predictions. `next_batch` and `submit`
strictly alternate.

```c
n4m_explain_options_v1_t opt;
n4m_explain_options_init_v1(&opt);            /* permutation SHAP, 10 pairs */
opt.method = N4M_EXPLAIN_PERMUTATION_SHAP;
opt.baseline = N4M_EXPLAIN_BASELINE_INTERPOLATE;
opt.n_outputs = 1;
opt.seed = 42;

n4m_explain_session_t* s = NULL;
n4m_explain_session_create_v1(ctx, &X, NULL, feature_group, n_groups, &opt, &s);
int64_t total = 0;                              /* exact plan size, for progress */
n4m_explain_session_info_v1(s, NULL, NULL, NULL, NULL, &total);

double* rows = malloc(cap * n_features * sizeof(double));
double* pred = malloc(cap * opt.n_outputs * sizeof(double));
for (;;) {
    int64_t n = 0;
    if (n4m_explain_session_next_batch_v1(ctx, s, rows, cap, &n) != N4M_OK) break;
    if (n == 0) break;                          /* plan complete */
    host_predict(rows, n, n_features, pred);    /* n x n_outputs, row-major */
    if (n4m_explain_session_submit_v1(ctx, s, pred, n) != N4M_OK) break;
}
n4m_explain_session_result_v1(ctx, s, attributions, n_rows * n_groups * n_outputs,
                              expected, n_rows * n_outputs, predictions, n_rows * n_outputs,
                              std_error, n_rows * n_groups * n_outputs);
n4m_explain_session_destroy(s);
```

Protocol rules:

* The plan is deterministic for (inputs, options, seed) and independent of the
  batch sizes. Rows are generated lazily; memory is bounded by the rows spanned
  by one batch (per-row coalition sums, permutations or sampled masks), never by
  the full plan.
* Every request includes `f(x)` itself and the empty coalition, so `expected`
  and `predictions` come from the model. For `BACKGROUND`/`MEAN` the empty
  coalition (the background rows, or the mean row) is evaluated once at the
  start of the plan and shared by all explained rows.
* A second `next_batch` before `submit`, a `submit` without a pending batch, or
  `result` before completion returns `N4M_ERR_INVALID_ARGUMENT`; a wrong
  submitted row count returns `N4M_ERR_SHAPE_MISMATCH`. These leave the session
  usable. A nonfinite prediction returns `N4M_ERR_NUMERICAL_FAILURE` and poisons
  the session: every later call fails and the session must be destroyed.
* `X` and the background must be finite; NaN-tolerant models are explained by
  the host after its own imputation.

## Evaluation cost

Let `R` explained rows, `G` groups and `B` evaluations per non-full coalition
(`B` = number of background rows for `BACKGROUND`, 1 for `MEAN` and
`INTERPOLATE`). The exact number of model evaluations reported by
`n4m_explain_session_info_v1` is

```
total = E0 + R * (1 + B * C)
```

where `E0 = B` for `BACKGROUND`/`MEAN` (shared empty coalition) and 0 otherwise,
and `C` is the number of non-full coalitions per row:

| Method | `C` (BACKGROUND, MEAN) | `C` (INTERPOLATE) |
| --- | --- | --- |
| Exact SHAP | `2^G - 2` | `2^G - 1` |
| Permutation SHAP | `2 * P * (G - 1)` | `2 * P * (G - 1) + 1` |
| Kernel SHAP | `K` | `K + 1` |
| LIME | `N - 1` | `N - 1` (no empty coalition) |
| Occlusion | `G` | `G + 1` |
| Gradient | `2F` (`F` features; `B` = 1) | refused |

`P = n_permutations` (antithetic pairs), `N = n_coalitions` for LIME (default
1000, the all-present mask included), and for Kernel SHAP `K = 2^G - 2` when the
budget `n_coalitions` (default `min(2^G - 2, 2G + 2048)`) covers every proper
coalition, else `K = 2 * ceil(n_coalitions / 2)` (complement pairs). For
example, permutation SHAP with `P = 10`, `G = 50` groups and the `INTERPOLATE`
baseline costs `1 + 981 = 982` evaluations per row; with a 50-row background it
costs `50 + R * (1 + 50 * 980)`.

## Explainers

### Exact group SHAP — implemented

Enumerates the `2^G` coalitions (`G <= 16`, larger values return
`N4M_ERR_UNSUPPORTED`) and applies the Shapley weights
`|S|! (G - |S| - 1)! / G!`. It is the validation oracle of the other estimators
and the method of choice for a handful of chemically defined bands.

### Permutation SHAP — implemented

Per row, `n_permutations` uniform group permutations are drawn (PCG64 stream
seeded by `seed`). Each draw evaluates the forward chain (groups added in
permutation order from the empty coalition) and its antithetic reverse (groups
removed in the same order from the full coalition), as in
`shap.PermutationExplainer`. The draw estimate is the mean of both chains'
marginal contributions; each chain telescopes from `v(empty)` to `v(full)`, so
the result is additive by construction. The attribution is the mean of the draw
estimates and `std_error` their standard error (`sd / sqrt(P)`, NaN when
`P = 1`). The estimator is unbiased; its error decreases as `1 / sqrt(P)`. With
`G <= 2` one pair is already exact.

### Kernel SHAP — experimental

Solves the Shapley-kernel weighted least squares
`v(S) - v(empty) = sum_{g in S} phi_g` with the efficiency constraint
`sum_g phi_g = v(full) - v(empty)` enforced exactly by eliminating the last
group (as `shap.KernelExplainer`). When the budget covers every proper
coalition, all `2^G - 2` coalitions are enumerated with weights
`(G - 1) / (C(G, |S|) |S| (G - |S|))` and the result equals exact SHAP.
Otherwise coalition sizes are sampled proportionally to their total kernel
weight `(G - 1) / (k (G - k))`, members uniformly, and every sample is paired
with its complement (equal weights). Sampled Kernel SHAP is consistent, not
unbiased, and reports no standard error. If the normal equations are
numerically rank deficient (fewer independent coalitions than groups), a ridge
of `1e-8` times their mean diagonal is added once; a still-singular system
returns `N4M_ERR_NUMERICAL_FAILURE`.

### LIME — experimental

Binary group masks: the all-present mask plus `n_coalitions - 1` samples whose
number of absent groups is uniform in `{1, ..., G}` and whose absent groups are
uniform (as `lime`'s text sampler). Each mask `z` is weighted by
`pi(z) = sqrt(exp(-d(z)^2 / w^2))` where
`d(z) = 1 - sqrt(|z| / G)` is the cosine distance between `z` and the all-ones
mask and `w = lime_kernel_width` (default `0.75 * sqrt(G)`, `lime`'s default
form). A weighted ridge regression of `v(z)` on `z` with weighted centering
(penalty `lime_ridge` on the coefficients, not on the intercept, as
`sklearn.linear_model.Ridge` with sample weights) gives the attributions
(coefficients) and `expected` (intercept). `predictions` holds `f(x)`. LIME is a
local surrogate: it is not additive and depends on the kernel width.

### Occlusion — implemented

One coalition per group, "everything present except `g`":
`attribution[g] = f(x) - v(all except g)` and `expected = v(empty)`. Cheap and
easy to read for band ablation; not additive and blind to interactions.

### Linear SHAP — implemented

For an affine model `f(z) = z . B + c` in the space of `B`
(`n_features x n_outputs`) and the background column means `reference`:
`attribution[g][o] = sum_{j in g} B[j][o] (x[j] - reference[j])` and
`expected[o] = reference . B[:, o] + c[o]`. This equals exact interventional
SHAP with the `BACKGROUND` baseline. It applies to PLS, PCR, Ridge and any
linear pipeline whose coefficients are expressed in the explained space.

### Path-dependent TreeSHAP — implemented

Lundberg et al. 2020, Algorithm 2, written after the reference
`shap/cext/tree_shap.h` with an explicit stack (deep trees cannot exhaust small
WebAssembly stacks). The flat ensemble format concatenates the nodes of all
trees; tree `t` owns nodes `[tree_offsets[t], tree_offsets[t+1])` with local
child indices (`-1/-1` for leaves). Covers must be finite and positive on every
reachable node; thresholds and leaf values finite; nodes must form a tree.
Zero fractions are `cover[child] / (cover[left] + cover[right])` and the expected
value is the same recursive cover-weighted mean, so additivity is exact for any
tree, even when a stored parent cover differs from the sum of its children.
`decision_type` 0 sends `x <= threshold` left (scikit-learn, LightGBM), 1 sends
`x < threshold` left (XGBoost). NaN inputs follow `default_left` when given and
are rejected otherwise. Leaf values may have several outputs. The ensemble
output is `base_offset[o] + tree_scale * sum_t leaf_t[o]` (`tree_scale` =
`1 / n_trees` for averaging forests, the learning rate or 1 for boosting
exports). Path-dependent TreeSHAP conditions on the training distribution
encoded by the covers; it is not interventional SHAP against a background.
Hosts own the model exporters.

### Local gradient — implemented (ABI 2.24)

`N4M_EXPLAIN_GRADIENT` evaluates, per explained row `x` and feature `j`, the
central difference `g_j = (f(x + h_j e_j) - f(x - h_j e_j)) / ((x_j + h_j) -
(x_j - h_j))` (the denominator is the step actually represented, so rounding of
the perturbed coordinate is exact in the quotient), and the first-order
attribution `phi_j = g_j (x_j - r_j)` with `r` the column mean of the
background (both `BACKGROUND` and `MEAN` select this mean reference;
`INTERPOLATE` is refused). `expected = f(r)` (one evaluation), `predictions =
f(x)`; grouped values are sums over the group's features
(`sum_of_feature_values`). The cost is `R (2F + 1) + 1` evaluations.

Step: `h_j = step * s_j`, `s_j` the population standard deviation of background
column `j` (or `|r_j|` when it is zero, 1 when `r_j` is zero too); `step`
defaults to `1e-3` and is set with `n4m_explain_session_set_gradient_step_v1`
before the first batch. For smooth models the truncation error is `O(h^2)`; for
piecewise-constant models (trees) a central difference measures a local slope
over `[x - h, x + h]` and is generally not meaningful.

The method is not additive in general; for an exactly affine model `f(z) = z B
+ c`, `g = B` and the attributions equal linear SHAP with the mean reference
(tested to 1e-8). `n4m_explain_session_gradients_v1` returns the per-row
gradients `[row][feature][output]`, their mean over rows (the "mean local
coefficient" profile) and the steps.

## Model-intrinsic importances (ABI 2.24)

Exact properties of fitted estimators, through the generic estimator surface
(`n4m/estimator.h`), for any method whose fitted state exposes them (no
per-method lists):

* `n4m_estimator_affine_coefficients_v1`: `predict(X) = X B + b` of any
  estimator declaring `N4M_CAP_AFFINE` (PLS family, PCR, OPLS, Ridge, the
  affine AOM and multiblock heads), also after N4ME import. Typed refusal
  (`N4M_ERR_UNSUPPORTED`) otherwise.
* `n4m_estimator_latent_structure_v1`: `W`, `R = W (P'W)^-1`, `P`, `Q`, x mean
  and scale of latent-variable models fitted through `n4m_model_fit` (PLS
  regression, PCR, OPLS, recursive PLS). Affine predictors that keep only
  coefficients refuse.
* `n4m_estimator_importance_v1` on rows supplied in the estimator's input space
  (normally the preprocessed training partition; the host records which):
  * VIP: `SSY_a = ||t_a||^2 ||q_a||^2` with `T = ((X - x_mean) / x_scale) R`
    of the supplied rows; the formula of
    `n4m_feature_selection_variable_select_rank` (shared kernel). Mean of
    `VIP_j^2` is 1. All outputs together.
  * Selectivity ratio: target projection (Rajalahti et al. 2009) on one output:
    `Z = (X - row mean) / x_scale`, `w = B_model[:, o] / ||B_model[:, o]||`
    (`B_model = x_scale * B`), `t = Z w`, `p = Z't / t't`,
    `SR_j = ||t p_j||^2 / ||z_j - t p_j||^2` (shared ratio kernel; `+inf` for a
    column reconstructed exactly). Any affine estimator. This is not the
    all-components ratio of `n4m_feature_selection_variable_select_rank`.
  * Standardized coefficients: `B[j][o] * sd_j`, sample standard deviation
    (`ddof = 1`) of the supplied rows.
* `n4m_role_pipeline_input_affine_coefficients_v1`: exact raw-input map of a
  pipeline whose transformers / selectors declare `N4M_CAP_AFFINE_TRANSFORM`
  and whose terminal step is affine: `B_raw = A' B`, `b_raw = c' B + b`, with
  `z = A x + c` the composed transform chain read from the fitted transforms
  (`c = T(0)`, `A e_j = T(e_j) - T(0)`, batched). The first non-affine step is
  named in the refusal. The method-level traits `N4M_CAP_AFFINE_TRANSFORM`
  and `N4M_CAP_COLUMN_ALIGNED` are reported by `n4m_method_info_v1`, the
  manifest (`affine_transform`, `column_aligned`) and, per step,
  `n4m_role_pipeline_step_info_v1` (`affine`, `column_aligned`). Every
  declaration of `N4M_CAP_AFFINE_TRANSFORM` is checked exactly
  (`T(a x + (1 - a) z) = a T(x) + (1 - a) T(z)`) by `cpp/tests/test_importance.cpp`.

The raw-input map is as exact as the pipeline itself. Columns that are zero up
to rounding no longer inflate it: every autoscaling path maps a standard
deviation below `10 * DBL_EPSILON` to 1 (scikit-learn `_handle_zeros_in_scale`
rule), and the mirror-mode Savitzky-Golay odd derivative writes its exactly
zero edge values as 0.

Hosts wrap global profiles in the
[`nirs4all.importance_profile.v1`](explain/importance_profile_v1.schema.json)
envelope.

## Helpers

* `n4m_explain_groups_fixed_width_v1`: feature `j` goes to group `j / width`;
  the last group may be shorter.
* `n4m_explain_groups_from_ranges_v1`: disjoint half-open ranges (any order)
  become groups `0..n_ranges-1`; each maximal uncovered run becomes one extra
  group, numbered after the ranges in feature order; `range_of_group` records
  the source range or `-1`. Overlapping, empty or out-of-range ranges are
  refused.
* `n4m_explain_summarize_v1`: per-group mean absolute (magnitude) and mean
  signed attribution over rows for one output.
* `n4m_explain_summarize_std_error_v1` (ABI 2.24): Monte-Carlo standard error
  of the row-mean signed attribution, `sqrt(sum_r se_rg^2) / R`.
* `n4m_explain_perturb_rows_v1` (ABI 2.24): the rows a session sends to the
  model for one coalition (absent groups filled by the baseline with the
  session's filling code), for baseline previews.
* `n4m_explain_aggregate_profile_v1` (ABI 2.24): per-feature profile to groups
  (`sum`, `mean`, `sum_abs`, `mean_abs`, `max_abs`).
* `n4m_explain_compare_profiles_v1` (ABI 2.24): `m x m` Spearman correlation of
  average ranks (through `n4m_stats_correlation_test`, equal to
  `scipy.stats.spearmanr`; NaN for a constant profile) and top-K Jaccard overlap
  by `|value|` (lowest index on ties).

## Limitations

* No conditional or on-manifold baselines: absent groups take background, mean
  or interpolated values independently of the present groups. Interpolation
  mitigates, but does not remove, off-manifold perturbations.
* No Integrated Gradients or analytic gradients (neural models are not
  available in every host); the local gradient explainer uses finite
  differences of the host predictions.
* TreeSHAP is path-dependent only; grouped tree values are sums of per-feature
  values, not group-game values.
* Kernel SHAP and LIME are experimental; validation covers exactness on full
  enumeration, efficiency and linear recovery, not parity with the `shap` and
  `lime` packages on sampled budgets.
* All inputs are F64; explanations of probability outputs explain the
  probability the host returns, never a relabelled score.
