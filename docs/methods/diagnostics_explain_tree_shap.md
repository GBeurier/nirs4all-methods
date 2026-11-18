# `tree_shap` — n4m.explain.tree_shap

_Namespace_: **`n4m.explain`** · _Fully-qualified_: `n4m.explain.tree_shap` · _Catalog id_: `diagnostics.explain.tree_shap`

## API surface

**C ABI (ABI 2):** [`n4m_explain_tree_path_dependent_v1`](https://github.com/GBeurier/nirs4all-methods/blob/main/cpp/include/n4m/explain.h#L265). Use the linked public header for the exact signature, configuration, and result handles.

**Python (verified public re-export):** `from n4m.explain import tree_shap`

**Signature:** [`tree_shap(X: ArrayLike, ensemble: TreeEnsemble, *, feature_group: ArrayLike | None = None, n_groups: int | None = None)`](https://github.com/GBeurier/nirs4all-methods/blob/main/bindings/python/src/n4m/explain.py#L568)

**R:** no current source-verified entry point was found for this catalog method.

**MATLAB / Octave:** no current source-verified entry point was found for this catalog method.

### Parameters

| Name | Type | Default |
|---|---|---|
| `X` | `ArrayLike` | `required` |
| `ensemble` | `TreeEnsemble` | `required` |
| `feature_group` | `ArrayLike \| None` | `None` |
| `n_groups` | `int \| None` | `None` |

## Explanations

### Bibliographic source

Lundberg, Erion, Chen, DeGrave, Prutkin, Nair, Katz, Himmelfarb, Bansal and Lee (2020), From local explanations to global understanding with explainable AI for trees, Nature Machine Intelligence 2, 56-67, Algorithm 2, https://doi.org/10.1038/s42256-019-0138-9.

### Mathematical principle

Path-dependent TreeSHAP computes exact Shapley values of the conditional expectation estimated by each tree, where an absent feature follows both children weighted by their training covers. Algorithm 2 tracks, along each root-to-leaf path, the proportion of coalitions of each size that reach the node (extend and unwind of the unique-feature path), giving polynomial cost in depth instead of exponential cost in features. Zero fractions are cover[child]/(cover[left]+cover[right]) and the expected value is the same recursive cover-weighted mean, so additivity is exact for every tree. The ensemble output is base_offset plus tree_scale times the sum of leaf values; grouped values are sums of per-feature values.

### Appropriate uses

Exact explanations of exported decision trees, random forests (tree_scale 1/n_trees) and gradient-boosted ensembles (learning rate and base offset), with multi-output leaves, scikit-learn/LightGBM or XGBoost split conventions and default-left handling of missing values.

### Limits and validation

Path-dependent values condition on the training distribution encoded by the covers; they are not interventional SHAP against a background and differ from it when features are dependent. Grouped values are sums of per-feature values, not group-game values. Covers must be finite and positive on reachable nodes, thresholds and leaf values finite and nodes must form a tree; NaN inputs require default_left. Model exporters are host responsibilities.

### Implementation

A C++17 kernel behind n4m_explain_tree_path_dependent_v1, written after the reference shap tree_shap.h with an explicit stack so deep trees cannot exhaust small WebAssembly stacks, consumes a flat ensemble of concatenated nodes with per-tree offsets and local child indices. Python n4m.explain.tree_shap and JavaScript treeShap marshal the ABI. Tests compare a brute-force conditional-expectation oracle and pin parity with shap 0.51 TreeExplainer (tree_path_dependent) for scikit-learn trees, forests and gradient boosting.

### Sources and provenance

https://github.com/GBeurier/nirs4all-methods/blob/main/cpp/include/n4m/explain.h; https://github.com/GBeurier/nirs4all-methods/blob/main/cpp/src/core/explain/tree_shap.cpp; https://github.com/GBeurier/nirs4all-methods/blob/main/cpp/src/core/explain/explain_common.cpp; https://github.com/GBeurier/nirs4all-methods/blob/main/cpp/src/c_api/c_api_explain.cpp; https://github.com/GBeurier/nirs4all-methods/blob/main/cpp/tests/test_explain.cpp; https://github.com/GBeurier/nirs4all-methods/blob/main/bindings/python/tests/test_explain.py; https://github.com/GBeurier/nirs4all-methods/blob/main/docs/EXPLAIN.md

## Catalog note

Path-dependent TreeSHAP (Lundberg et al. 2020, Algorithm 2) over a generic flat multi-output tree ensemble with cover-consistent zero fractions; grouped values are sums of per-feature values.


_See also_: [methods index](index.md).