# `model_agnostic` — n4m.explain.model_agnostic

_Namespace_: **`n4m.explain`** · _Fully-qualified_: `n4m.explain.model_agnostic` · _Catalog id_: `diagnostics.explain.model_agnostic`

## API surface

**C ABI (ABI 2):** [`n4m_explain_options_init_v1`](https://github.com/GBeurier/nirs4all-methods/blob/main/cpp/include/n4m/explain.h#L121) · [`n4m_explain_session_create_v1`](https://github.com/GBeurier/nirs4all-methods/blob/main/cpp/include/n4m/explain.h#L126) · [`n4m_explain_session_destroy`](https://github.com/GBeurier/nirs4all-methods/blob/main/cpp/include/n4m/explain.h#L134) · [`n4m_explain_session_info_v1`](https://github.com/GBeurier/nirs4all-methods/blob/main/cpp/include/n4m/explain.h#L139) · [`n4m_explain_session_next_batch_v1`](https://github.com/GBeurier/nirs4all-methods/blob/main/cpp/include/n4m/explain.h#L150) · [`n4m_explain_session_submit_v1`](https://github.com/GBeurier/nirs4all-methods/blob/main/cpp/include/n4m/explain.h#L158) · [`n4m_explain_session_result_v1`](https://github.com/GBeurier/nirs4all-methods/blob/main/cpp/include/n4m/explain.h#L166) · [`n4m_explain_groups_fixed_width_v1`](https://github.com/GBeurier/nirs4all-methods/blob/main/cpp/include/n4m/explain.h#L295) · [`n4m_explain_groups_from_ranges_v1`](https://github.com/GBeurier/nirs4all-methods/blob/main/cpp/include/n4m/explain.h#L309) · [`n4m_explain_summarize_v1`](https://github.com/GBeurier/nirs4all-methods/blob/main/cpp/include/n4m/explain.h#L320) · [`n4m_explain_session_set_gradient_step_v1`](https://github.com/GBeurier/nirs4all-methods/blob/main/cpp/include/n4m/explain.h#L81) · [`n4m_explain_session_gradients_v1`](https://github.com/GBeurier/nirs4all-methods/blob/main/cpp/include/n4m/explain.h#L82) · [`n4m_explain_perturb_rows_v1`](https://github.com/GBeurier/nirs4all-methods/blob/main/cpp/include/n4m/explain.h#L207) · [`n4m_explain_summarize_std_error_v1`](https://github.com/GBeurier/nirs4all-methods/blob/main/cpp/include/n4m/explain.h#L332) · [`n4m_explain_aggregate_profile_v1`](https://github.com/GBeurier/nirs4all-methods/blob/main/cpp/include/n4m/explain.h#L356) · [`n4m_explain_compare_profiles_v1`](https://github.com/GBeurier/nirs4all-methods/blob/main/cpp/include/n4m/explain.h#L373). Use the linked public header for the exact signature, configuration, and result handles.

**Python (verified public re-export):** `from n4m.explain import explain`

**Signature:** [`explain(predict_batch: Callable[[np.ndarray], ArrayLike], X: ArrayLike, *, background: ArrayLike | None = None, feature_group: ArrayLike | None = None, n_groups: int | None = None, method: str = 'permutation_shap', baseline: str = 'background', n_outputs: int | None = None, n_permutations: int = 10, n_coalitions: int = 0, lime_kernel_width: float = 0.0, lime_ridge: float = 1.0, max_batch_rows: int = 4096, seed: int = 0, progress: Callable[[int, int], None] | None = None)`](https://github.com/GBeurier/nirs4all-methods/blob/main/bindings/python/src/n4m/explain.py#L271)

**R:** no current source-verified entry point was found for this catalog method.

**MATLAB / Octave:** no current source-verified entry point was found for this catalog method.

### Parameters

| Name | Type | Default |
|---|---|---|
| `predict_batch` | `Callable[[np.ndarray], ArrayLike]` | `required` |
| `X` | `ArrayLike` | `required` |
| `background` | `ArrayLike \| None` | `None` |
| `feature_group` | `ArrayLike \| None` | `None` |
| `n_groups` | `int \| None` | `None` |
| `method` | `str` | `'permutation_shap'` |
| `baseline` | `str` | `'background'` |
| `n_outputs` | `int \| None` | `None` |
| `n_permutations` | `int` | `10` |
| `n_coalitions` | `int` | `0` |
| `lime_kernel_width` | `float` | `0.0` |
| `lime_ridge` | `float` | `1.0` |
| `max_batch_rows` | `int` | `4096` |
| `seed` | `int` | `0` |
| `progress` | `Callable[[int, int], None] \| None` | `None` |

## Explanations

### Bibliographic source

Shapley (1953), A Value for n-Person Games, in Contributions to the Theory of Games II, Annals of Mathematics Studies 28, 307-317, https://doi.org/10.1515/9781400881970-018. Permutation sampling: Strumbelj and Kononenko (2014), Explaining prediction models and individual predictions with feature contributions, Knowledge and Information Systems 41(3), 647-665, https://doi.org/10.1007/s10115-013-0679-x; antithetic (forward/reverse) orders: Mitchell, Cooper, Frank and Holmes (2022), Sampling Permutations for Shapley Value Estimation, Journal of Machine Learning Research 23, https://www.jmlr.org/papers/v23/21-0439.html. Kernel SHAP: Lundberg and Lee (2017), A Unified Approach to Interpreting Model Predictions, NeurIPS 30, https://arxiv.org/abs/1705.07874. LIME: Ribeiro, Singh and Guestrin (2016), "Why Should I Trust You?": Explaining the Predictions of Any Classifier, KDD, 1135-1144, https://doi.org/10.1145/2939672.2939778. Occlusion: Zeiler and Fergus (2014), Visualizing and Understanding Convolutional Networks, ECCV, LNCS 8689, 818-833, https://doi.org/10.1007/978-3-319-10590-1_53. Interventional value function: Janzing, Minorics and Bloebaum (2020), Feature relevance quantification in explainable AI: A causal problem, AISTATS, PMLR 108, 2907-2916, https://proceedings.mlr.press/v108/janzing20a.html. Local gradient explanations: Baehrens, Schroeter, Harmeling, Kawanabe, Hansen and Mueller (2010), How to Explain Individual Classification Decisions, Journal of Machine Learning Research 11, 1803-1831, https://www.jmlr.org/papers/v11/baehrens10a.html.

### Mathematical principle

Feature groups partitioning the explained features are the players of a coalition game. The value v(S) is the host model output when the groups in S keep the explained row's values and absent groups take baseline values: each background row averaged over the background (interventional), the background column mean, or a straight-line bridge across each maximal run of absent contiguous groups. Exact SHAP enumerates all $2^G$ coalitions ($G\le 16$) with Shapley weights $|S|!(G-|S|-1)!/G!$. Permutation SHAP draws uniform group orders and averages the marginal contributions of each forward chain and its antithetic reverse chain; each chain telescopes from v(empty) to v(full), so the estimate is additive by construction and its standard error is $\mathrm{sd}/\sqrt{P}$ over P draws. Kernel SHAP solves the Shapley-kernel weighted least squares with the efficiency constraint eliminated exactly; it enumerates every proper coalition when the budget allows (then equal to exact SHAP) and otherwise samples coalition sizes proportionally to their kernel mass with complement pairing. LIME fits a weighted ridge surrogate of v(z) on binary group masks with an exponential kernel of the cosine distance to the all-present mask. Occlusion reports $f(x)-v(\text{all except } g)$. The local gradient method (ABI 2.24) is not a coalition game: it estimates each partial derivative by a central difference $g_j=(f(x+h_je_j)-f(x-h_je_j))/(2h_j)$ with a step scaled by the background spread, and reports $g_j(x_j-r_j)$ against the background mean r, summed within groups.

### Appropriate uses

Explaining any fitted regressor or classifier output (prediction in original units, one class probability or a decision score) whose predictions the host can compute in batches with any backend, including Python, WebAssembly, ONNX or asynchronous workers. For NIR spectra, grouped explanations over contiguous wavelength windows or chemically informed bands with the interpolation baseline, exact SHAP for a handful of bands, and permutation SHAP with standard errors for larger group counts.

### Limits and validation

Status: exact SHAP, permutation SHAP and occlusion are implemented; Kernel SHAP and LIME are experimental, validated for exactness under full enumeration, efficiency and linear recovery but not for parity with the shap or lime packages on sampled budgets. Sampled Kernel SHAP is consistent, not unbiased, and reports no standard error; LIME and occlusion are not additive and occlusion ignores interactions. Absent groups take baseline values independently of present groups, so correlated spectral features can be evaluated off the data manifold; interpolation mitigates but does not remove this, and no conditional baseline is provided. Attributions describe the model's response under the declared value function, not causal effects in the measured system. Cost grows as 2^G for exact SHAP and linearly with the background size; inputs and predictions must be finite. Local gradients are first-order, not additive except for affine models (where they equal linear SHAP with the mean reference), and meaningless for piecewise-constant models such as trees.

### Implementation

A C++17 ask/tell session in the ABI 2.23 n4m/explain.h surface never calls a model: it emits deterministic batches of perturbed rows (PCG64 stream seeded by the caller) and receives host predictions, with memory bounded by one batch and the exact evaluation count reported up front. Protocol misuse is rejected without losing the session; nonfinite predictions poison it. Fixed-width and range grouping helpers and per-group summaries are native. Python n4m.explain and JavaScript explainModel only marshal the ABI. Tests check exact, permutation and full-enumeration Kernel SHAP against brute-force Shapley values and pin parity with shap 0.51 ExactExplainer and PermutationExplainer.

### Sources and provenance

https://github.com/GBeurier/nirs4all-methods/blob/main/cpp/include/n4m/explain.h; https://github.com/GBeurier/nirs4all-methods/blob/main/cpp/src/core/explain/explain_session.cpp; https://github.com/GBeurier/nirs4all-methods/blob/main/cpp/src/core/explain/grouping.cpp; https://github.com/GBeurier/nirs4all-methods/blob/main/cpp/src/core/explain/explain_common.cpp; https://github.com/GBeurier/nirs4all-methods/blob/main/cpp/src/c_api/c_api_explain.cpp; https://github.com/GBeurier/nirs4all-methods/blob/main/cpp/tests/test_explain.cpp; https://github.com/GBeurier/nirs4all-methods/blob/main/bindings/python/tests/test_explain.py; https://github.com/GBeurier/nirs4all-methods/blob/main/bindings/js/test/run_explain.mjs; https://github.com/GBeurier/nirs4all-methods/blob/main/docs/EXPLAIN.md; https://github.com/GBeurier/nirs4all-methods/blob/main/docs/explain/explanation_v1.schema.json

## Catalog note

Ask/tell session (never calls a model): exact group SHAP (G <= 16), antithetic permutation SHAP, Kernel SHAP with exact efficiency, LIME over group masks, occlusion and (ABI 2.24) local central-difference gradients against the background mean, with background, mean or spectral interpolate baselines. Groups are coalition players. Grouping, summary and standard-error helpers, coalition-row previews, and (ABI 2.24) importance-profile aggregation and comparison (Spearman, top-K Jaccard).


_See also_: [methods index](index.md).