# `fanova` — n4m.component_analysis.fanova

_Namespace_: **`n4m.component_analysis`** · _Fully-qualified_: `n4m.component_analysis.fanova` · _Catalog id_: `model_selection.fanova`

## API surface

**C ABI (ABI 2):** [`n4m_fanova_options_init_v1`](https://github.com/GBeurier/nirs4all-methods/blob/main/cpp/include/n4m/fanova.h#L44) · [`n4m_fanova_backend_available_v1`](https://github.com/GBeurier/nirs4all-methods/blob/main/cpp/include/n4m/fanova.h#L45) · [`n4m_fanova_fit_v1`](https://github.com/GBeurier/nirs4all-methods/blob/main/cpp/include/n4m/fanova.h#L73). Use the linked public header for the exact signature, configuration, and result handles.

**Python (verified public re-export):** `from n4m.component_analysis import fanova`

**Signature:** [`fanova(X: ArrayLike, y: ArrayLike, feature_types: ArrayLike, bounds: ArrayLike, *, n_trees: int = 32, max_depth: int = 10, min_samples_leaf: int = 2, max_features: int = 0, seed: int = 42, max_order: int = 2, max_marginal_evaluations: int = 1000000, X_valid: ArrayLike | None = None, y_valid: ArrayLike | None = None)`](https://github.com/GBeurier/nirs4all-methods/blob/main/bindings/python/src/n4m/component_analysis.py#L212)

**R:** no current source-verified entry point was found for this catalog method.

**MATLAB / Octave:** no current source-verified entry point was found for this catalog method.

### Parameters

| Name | Type | Default |
|---|---|---|
| `X` | `ArrayLike` | `required` |
| `y` | `ArrayLike` | `required` |
| `feature_types` | `ArrayLike` | `required` |
| `bounds` | `ArrayLike` | `required` |
| `n_trees` | `int` | `32` |
| `max_depth` | `int` | `10` |
| `min_samples_leaf` | `int` | `2` |
| `max_features` | `int` | `0` |
| `seed` | `int` | `42` |
| `max_order` | `int` | `2` |
| `max_marginal_evaluations` | `int` | `1000000` |
| `X_valid` | `ArrayLike \| None` | `None` |
| `y_valid` | `ArrayLike \| None` | `None` |

## Explanations

### Bibliographic source

Hutter, Hoos and Leyton-Brown (2014), An Efficient Approach for Assessing Hyperparameter Importance, ICML, PMLR 32(1), 754-762, https://proceedings.mlr.press/v32/hutter14.html.

### Mathematical principle

A fresh bootstrap regression forest provides a surrogate surface. Under the declared independent uniform product measure, each tree's main variance is Var(E[f|x_j]); a pair variance is Var(E[f|x_j,x_k]) minus both main variances. Exact integration enumerates split intervals weighted by length and categorical codes weighted by 1/K. Component fractions normalize each informative tree by its own total variance before averaging. The remainder contains all unreported interaction orders.

### Appropriate uses

Descriptive surrogate attribution of observed pipeline, component or hyperparameter responses when the host can define valid finite continuous domains, category codes and a scientifically meaningful independent product measure.

### Limits and validation

Correlated or conditional parameters can make the product measure include unsupported combinations. Fractions describe the fitted surrogate and are not causal effects. Tree sample SD is variability across normalized trees, not an SE or confidence interval. Constant trees are excluded and counted; constant responses and noninformative forests withhold attribution. Training RMSE is descriptive; optional validation RMSE and R-squared do not confer a generalization guarantee. The host owns matching and dependence policy.

### Implementation

An optional compile-time C++17 backend vendors only the reviewed BSD-3-Clause header closure of automl/random_forest_run at 1ba17650e734edd3ea536efc91d32815c4874e72, including cereal BSD-3-Clause and RapidJSON MIT headers and notices. No Python fanova code is used. The native adapter independently implements the decomposition, preserves strides and output atomicity, bounds tree and query resources, and retains backend-unavailable C ABI stubs by default. Tests include additive, XOR, category, constant, repeated-fit, holdout, stride, seed, invalid-input and budget cases.

### Sources and provenance

https://github.com/GBeurier/nirs4all-methods/blob/main/cpp/src/core/component_analysis/fanova.cpp; https://github.com/GBeurier/nirs4all-methods/blob/main/cpp/src/c_api/c_api_fanova.cpp; https://github.com/GBeurier/nirs4all-methods/blob/main/cpp/include/n4m/fanova.h; https://github.com/GBeurier/nirs4all-methods/blob/main/cpp/tests/test_fanova.cpp; https://github.com/GBeurier/nirs4all-methods/blob/main/cpp/backends/rfr/README.md; https://github.com/GBeurier/nirs4all-methods/blob/main/docs/FANOVA.md

## Catalog note

Exact per-tree marginal integration under explicit independent uniform product domains. Native forest surrogate; first-order and pairwise variance fractions averaged after per-tree normalization; tree dispersion is not CI. Optional static backend, no Python fanova code.


_See also_: [methods index](index.md).