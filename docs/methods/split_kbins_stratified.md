# `split_kbins_stratified` — Continuous-target binned stratified split

_Group_: **Splitters** · _C ABI_: `n4m_model_selection_kbins_stratified_*`

## Description

Stratified split using K equal-width or quantile bins of ``y``.

## Parameters

| Name | Type | Default |
|------|------|---------|
| `test_size` | `float` | `0.25` |
| `seed` | `int` | `0` |
| `n_bins` | `int` | `5` |
| `strategy` | `str \| int` | `'uniform'` |

## API and bindings

**C ABI (ABI 2):** [`n4m_model_selection_kbins_stratified_create`](https://github.com/GBeurier/nirs4all-methods/blob/main/cpp/include/n4m/model_selection.h#L94) · [`n4m_model_selection_kbins_stratified_destroy`](https://github.com/GBeurier/nirs4all-methods/blob/main/cpp/include/n4m/model_selection.h#L98) · [`n4m_model_selection_kbins_stratified_split`](https://github.com/GBeurier/nirs4all-methods/blob/main/cpp/include/n4m/model_selection.h#L99). Use the linked public header for the exact signature, configuration, and result handles.

**Python (verified public re-export):**

```python
from n4m.model_selection.splitters import KBinsStratified
```

Source signature: [`KBinsStratified(test_size: float = 0.25, seed: int = 0, n_bins: int = 5, strategy: str | int = 'uniform')`](https://github.com/GBeurier/nirs4all-methods/blob/main/bindings/python/src/n4m/_impl/splitters.py#L302).

**R:** no current source-verified entry point was found for this catalog method.

**MATLAB / Octave:** no current source-verified entry point was found for this catalog method.

## Explanations

### Bibliographic source

No single paper defines this composition of discretization and stratified shuffle split. The parity target is the scikit-learn `KBinsDiscretizer` plus `StratifiedShuffleSplit` behavior documented at https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.StratifiedShuffleSplit.html.

### Mathematical principle

Discretize y into `n_bins` by equal-width (`uniform`) or linearly interpolated equal-frequency (`quantile`) edges. Allocate class-wise train/test counts with sklearn's approximate-mode rule, shuffle members and final outputs with legacy MT19937 seeded by `seed`, and choose $n_{test}=\lceil n\,test\_size\rceil$.

### Appropriate uses

Preserving an approximate continuous-target distribution in one random train/test split.

### Limits and validation

Sparse or repeated y values can collapse bins or leave too few members for stratification. Output index order is shuffled, not sorted, and only y drives the split. The current Python binding requires y with shape `(n_samples, 1)`; its generic 1-D promotion produces shape `(1, n_samples)` and the ABI rejects it.

### Implementation

Python role API `n4m.model_selection.splitters.KBinsStratified`; binding class `KBinsStratifiedSplitter`; ABI 2 family `n4m_model_selection_kbins_stratified_{create,split,destroy}`.

The ABI-2 implementation is the `n4m_model_selection_kbins_stratified_*` lifecycle in libn4m.

### Sources and provenance

https://github.com/GBeurier/nirs4all-methods/blob/main/cpp/src/core/splitters/kbins_stratified.h; https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.StratifiedShuffleSplit.html


---

_See also_: [methods index](index.md) · [interactive dashboard](../landing/dashboard.md)