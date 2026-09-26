# `split_spxy_g_fold` — Group-preserving SPXY K-fold assignment

_Group_: **Splitters** · _C ABI_: `n4m_model_selection_spxy_g_fold_*`

## Description

Group-aware SPXY k-fold splitter.

## Parameters

| Name | Type | Default |
|------|------|---------|
| `n_splits` | `int` | `5` |
| `y_metric` | `str \| int` | `'mahalanobis'` |
| `aggregation` | `str \| int` | `'mean'` |

## API and bindings

**C ABI (ABI 2):** [`n4m_model_selection_spxy_g_fold_create`](https://github.com/GBeurier/nirs4all-methods/blob/main/cpp/include/n4m/model_selection.h#L109) · [`n4m_model_selection_spxy_g_fold_destroy`](https://github.com/GBeurier/nirs4all-methods/blob/main/cpp/include/n4m/model_selection.h#L113) · [`n4m_model_selection_spxy_g_fold_n_splits`](https://github.com/GBeurier/nirs4all-methods/blob/main/cpp/include/n4m/model_selection.h#L114) · [`n4m_model_selection_spxy_g_fold_split_fold`](https://github.com/GBeurier/nirs4all-methods/blob/main/cpp/include/n4m/model_selection.h#L116). Use the linked public header for the exact signature, configuration, and result handles.

**Python (verified public re-export):**

```python
from n4m.model_selection.splitters import SPXYGroupFold
```

Source signature: [`SPXYGroupFold(n_splits: int = 5, y_metric: str | int = 'mahalanobis', aggregation: str | int = 'mean')`](https://github.com/GBeurier/nirs4all-methods/blob/main/bindings/python/src/n4m/_impl/splitters.py#L202).

**R:** no current source-verified entry point was found for this catalog method.

**MATLAB / Octave:** no current source-verified entry point was found for this catalog method.

## Explanations

### Bibliographic source

This is an internal grouped extension of SPXY (Galvão et al., 2005, DOI 10.1016/j.talanta.2005.03.025, https://doi.org/10.1016/j.talanta.2005.03.025); no canonical paper defines its representative aggregation.

### Mathematical principle

Aggregate every integer-labeled group columnwise by mean or median in X and Y. Run the same alternating per-fold maximin SPXY assignment on group representatives, then expand each group assignment back to all member rows.

### Appropriate uses

Preventing replicates, batches, subjects, or lots from crossing train/test folds while spreading representatives.

### Limits and validation

Only int64 group labels cross the C ABI. A mean or median representative can hide within-group heterogeneity; fold sample counts may be unbalanced. As in SPXYFold, `'mahalanobis'` currently aliases Euclidean Y distance.

### Implementation

Python role API `n4m.model_selection.splitters.SPXYGroupFold`; binding class `SPXYGroupFoldSplitter`; ABI 2 family `n4m_model_selection_spxy_g_fold_{create,n_splits,split_fold,destroy}`.

The ABI-2 implementation is the `n4m_model_selection_spxy_g_fold_*` lifecycle in libn4m.

### Sources and provenance

https://github.com/GBeurier/nirs4all-methods/blob/main/cpp/src/core/splitters/spxy_g_fold.h; https://github.com/GBeurier/nirs4all-methods/blob/main/bindings/python/src/n4m/_impl/splitters.py#L202; https://doi.org/10.1016/j.talanta.2005.03.025


---

_See also_: [methods index](index.md) · [interactive dashboard](../landing/dashboard.md)