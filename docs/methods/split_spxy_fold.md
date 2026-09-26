# `split_spxy_fold` — Alternating maximin SPXY K-fold assignment

_Group_: **Splitters** · _C ABI_: `n4m_model_selection_spxy_fold_*`

## Description

SPXY k-fold splitter over paired ``X`` and ``y`` matrices.

## Parameters

| Name | Type | Default |
|------|------|---------|
| `n_splits` | `int` | `5` |
| `y_metric` | `str \| int` | `'mahalanobis'` |

## API and bindings

**C ABI (ABI 2):** [`n4m_model_selection_spxy_fold_create`](https://github.com/GBeurier/nirs4all-methods/blob/main/cpp/include/n4m/model_selection.h#L96) · [`n4m_model_selection_spxy_fold_destroy`](https://github.com/GBeurier/nirs4all-methods/blob/main/cpp/include/n4m/model_selection.h#L99) · [`n4m_model_selection_spxy_fold_n_splits`](https://github.com/GBeurier/nirs4all-methods/blob/main/cpp/include/n4m/model_selection.h#L100) · [`n4m_model_selection_spxy_fold_split_fold`](https://github.com/GBeurier/nirs4all-methods/blob/main/cpp/include/n4m/model_selection.h#L102). Use the linked public header for the exact signature, configuration, and result handles.

**Python (verified public re-export):**

```python
from n4m.model_selection.splitters import SPXYFold
```

Source signature: [`SPXYFold(n_splits: int = 5, y_metric: str | int = 'mahalanobis')`](https://github.com/GBeurier/nirs4all-methods/blob/main/bindings/python/src/n4m/_impl/splitters.py#L146).

**R:** no current source-verified entry point was found for this catalog method.

**MATLAB / Octave:** no current source-verified entry point was found for this catalog method.

## Explanations

### Bibliographic source

This is an internal K-fold extension of Galvão et al.'s SPXY criterion (2005), DOI 10.1016/j.talanta.2005.03.025 (https://doi.org/10.1016/j.talanta.2005.03.025); the alternating fold assignment itself is not defined by that paper.

### Mathematical principle

Build normalized X-plus-Y pairwise distances (or X alone). The engine seeds each fold with one of the `n_splits` samples having the largest mean distance, then cycles through folds. On each turn it assigns to that fold the remaining sample whose minimum distance to members already in that same fold is largest, up to `ceil(n/n_splits)`. Each requested fold returns that assignment as test and its complement as train.

### Appropriate uses

Deterministic folds intended to spread X/Y diversity across validation partitions.

### Limits and validation

The public string aliases `y_metric='euclidean'` and `'mahalanobis'` both map to the same Euclidean-Y code; the latter name does not invoke a Mahalanobis kernel. `x_only` is pure Kennard–Stone. Using y in fold design is supervised resampling and must remain inside the training data.

### Implementation

Python role API `n4m.model_selection.splitters.SPXYFold`; binding class `SPXYFoldSplitter`; ABI 2 family `n4m_model_selection_spxy_fold_{create,n_splits,split_fold,destroy}`.

The ABI-2 implementation is the `n4m_model_selection_spxy_fold_*` lifecycle in libn4m.

### Sources and provenance

https://github.com/GBeurier/nirs4all-methods/blob/main/bindings/python/src/n4m/_impl/splitters.py#L146; https://github.com/GBeurier/nirs4all-methods/blob/main/cpp/src/core/splitters/spxy_fold.h; https://doi.org/10.1016/j.talanta.2005.03.025


---

_See also_: [methods index](index.md) · [interactive dashboard](../landing/dashboard.md)