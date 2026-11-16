# `linear_shap` — n4m.explain.linear_shap

_Namespace_: **`n4m.explain`** · _Fully-qualified_: `n4m.explain.linear_shap` · _Catalog id_: `diagnostics.explain.linear_shap`

## API surface

**C ABI (ABI 2):** [`n4m_explain_linear_v1`](https://github.com/GBeurier/nirs4all-methods/blob/main/cpp/include/n4m/explain.h#L173). Use the linked public header for the exact signature, configuration, and result handles.

**Python (verified public re-export):** `from n4m.explain import linear_shap`

**Signature:** [`linear_shap(X: ArrayLike, coefficients: ArrayLike, *, reference: ArrayLike, intercept: ArrayLike | None = None, feature_group: ArrayLike | None = None, n_groups: int | None = None)`](https://github.com/GBeurier/nirs4all-methods/blob/main/bindings/python/src/n4m/explain.py#L492)

**R:** no current source-verified entry point was found for this catalog method.

**MATLAB / Octave:** no current source-verified entry point was found for this catalog method.

### Parameters

| Name | Type | Default |
|---|---|---|
| `X` | `ArrayLike` | `required` |
| `coefficients` | `ArrayLike` | `required` |
| `reference` | `ArrayLike` | `required` |
| `intercept` | `ArrayLike \| None` | `None` |
| `feature_group` | `ArrayLike \| None` | `None` |
| `n_groups` | `int \| None` | `None` |

## Explanations

### Bibliographic source

Lundberg and Lee (2017), A Unified Approach to Interpreting Model Predictions, Advances in Neural Information Processing Systems 30 (Linear SHAP under feature independence), https://arxiv.org/abs/1705.07874. The interventional value function follows Janzing, Minorics and Bloebaum (2020), AISTATS, PMLR 108, 2907-2916, https://proceedings.mlr.press/v108/janzing20a.html.

### Mathematical principle

For an affine model $f(z)=z\cdot B+c$ with coefficients B (features by outputs) expressed in the explained space and a reference vector r of background column means, the attribution of group g to output o is $\sum_{j\in g}B_{jo}(x_j-r_j)$ and the expected value is $r\cdot B_{:,o}+c_o$. Because the model is additive, this equals exact interventional SHAP against the background, and group sums are exact group Shapley values; attributions plus the expected value reproduce the prediction.

### Appropriate uses

Exact, evaluation-free explanations of PLS, PCR, Ridge and any linear pipeline whose coefficients and intercept are expressed in the space being explained, per wavelength or over spectral groups, and as an oracle for sampled model-agnostic estimators.

### Limits and validation

Valid only when the supplied coefficients represent the explained function in the explained space; coefficients of a terminal estimator after preprocessing explain the transformed features, not raw spectra. Interventional attributions ignore feature correlation, so strongly collinear wavelengths can receive large opposing values. The reference is caller supplied; inputs, coefficients, intercept and reference must be finite.

### Implementation

A dependency-free C++17 kernel behind n4m_explain_linear_v1 accepts strided F64 views, an optional feature-to-group map and caller-owned output buffers, and returns attributions, expected values and optional predictions. Python n4m.explain.linear_shap and JavaScript linearShap marshal the ABI. Tests check agreement with permutation SHAP and the group game, and pin parity with shap 0.51 LinearExplainer for Ridge and PLSRegression.

### Sources and provenance

https://github.com/GBeurier/nirs4all-methods/blob/main/cpp/include/n4m/explain.h; https://github.com/GBeurier/nirs4all-methods/blob/main/cpp/src/core/explain/linear_shap.cpp; https://github.com/GBeurier/nirs4all-methods/blob/main/cpp/src/core/explain/explain_common.cpp; https://github.com/GBeurier/nirs4all-methods/blob/main/cpp/src/c_api/c_api_explain.cpp; https://github.com/GBeurier/nirs4all-methods/blob/main/cpp/tests/test_explain.cpp; https://github.com/GBeurier/nirs4all-methods/blob/main/bindings/python/tests/test_explain.py; https://github.com/GBeurier/nirs4all-methods/blob/main/docs/EXPLAIN.md

## Catalog note

Exact interventional SHAP of an affine model against background column means; group sums are exact group Shapley values.


_See also_: [methods index](index.md).