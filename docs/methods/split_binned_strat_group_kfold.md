# `split_binned_strat_group_kfold` — Binned, group-preserving round-robin folds

_Group_: **Splitters** · _C ABI_: `n4m_model_selection_binned_strat_group_kfold_*`

## Description

Stratified group k-fold splitter after binning continuous ``y``.

## Parameters

| Name | Type | Default |
|------|------|---------|
| `n_splits` | `int` | `5` |
| `n_bins` | `int` | `5` |
| `strategy` | `str \| int` | `'uniform'` |
| `shuffle` | `bool` | `True` |
| `seed` | `int` | `0` |

## API and bindings

**C ABI (ABI 2):** [`n4m_model_selection_binned_strat_group_kfold_create`](https://github.com/GBeurier/nirs4all-methods/blob/main/cpp/include/n4m/model_selection.h#L106) · [`n4m_model_selection_binned_strat_group_kfold_destroy`](https://github.com/GBeurier/nirs4all-methods/blob/main/cpp/include/n4m/model_selection.h#L110) · [`n4m_model_selection_binned_strat_group_kfold_n_splits`](https://github.com/GBeurier/nirs4all-methods/blob/main/cpp/include/n4m/model_selection.h#L112) · [`n4m_model_selection_binned_strat_group_kfold_split_fold`](https://github.com/GBeurier/nirs4all-methods/blob/main/cpp/include/n4m/model_selection.h#L114). Use the linked public header for the exact signature, configuration, and result handles.

**Python (verified public re-export):**

```python
from n4m.model_selection.splitters import BinnedStratifiedGroupKFold
```

Source signature: [`BinnedStratifiedGroupKFold(n_splits: int = 5, n_bins: int = 5, strategy: str | int = 'uniform', shuffle: bool = True, seed: int = 0)`](https://github.com/GBeurier/nirs4all-methods/blob/main/bindings/python/src/n4m/_impl/splitters.py#L344).

**R:** no current source-verified entry point was found for this catalog method.

**MATLAB / Octave:** no current source-verified entry point was found for this catalog method.

## Explanations

### Bibliographic source

No canonical paper defines this simplified grouped heuristic. It is inspired by binned stratification and `StratifiedGroupKFold`, but the native source explicitly documents a different assignment algorithm.

### Mathematical principle

Bin continuous y by uniform or quantile edges. Label each group using the bin of its first encountered sample; within each bin, optionally PCG64-shuffle groups and assign them round-robin to `n_splits`. Expand group fold labels to samples, guaranteeing group integrity.

### Appropriate uses

Approximate target stratification when all rows from a group must stay in one fold.

### Limits and validation

This is not sklearn's constraint-aware `StratifiedGroupKFold`: a group's first sample alone determines its bin. Heterogeneous or unequal-size groups can produce poor class balance and sample-count imbalance. The current Python binding also requires y as `(n_samples, 1)`, not a 1-D vector.

### Implementation

Python role API `n4m.model_selection.splitters.BinnedStratifiedGroupKFold`; binding class `BinnedStratifiedGroupKFoldSplitter`; ABI 2 family `n4m_model_selection_binned_strat_group_kfold_{create,n_splits,split_fold,destroy}`.

The ABI-2 implementation is the `n4m_model_selection_binned_strat_group_kfold_*` lifecycle in libn4m.

### Sources and provenance

https://github.com/GBeurier/nirs4all-methods/blob/main/cpp/src/core/splitters/binned_strat_group_kfold.h


---

_See also_: [methods index](index.md) · [interactive dashboard](../landing/dashboard.md)