# `split_binned_strat_group_kfold` — Binned Stratified Group K Fold Splitter

_Group_: **Splitters** · _Binding_: `n4m.sklearn.BinnedStratifiedGroupKFoldSplitter` · _C ABI_: `n4m_split_binned_strat_group_kfold_*`

## Description

Stratified group k-fold splitter after binning continuous ``y``.

### Parameters

| Name | Type | Default |
|------|------|---------|
| `n_splits` | `int` | `5` |
| `n_bins` | `int` | `5` |
| `strategy` | `str | int` | `'uniform'` |
| `shuffle` | `bool` | `True` |
| `seed` | `int` | `0` |

## Explanations

### Bibliographic source

_Standard spectroscopic operator — see the nirs4all preprocessing / augmentation handbook and the cited literature within the binding docstring._

### Mathematical principle

Stratified group k-fold splitter after binning continuous ``y``.

### Implementation

C ABI `n4m_split_binned_strat_group_kfold_*` in libn4m (create / apply / destroy lifecycle), wrapped by `n4m.sklearn.BinnedStratifiedGroupKFoldSplitter`. The same numerical kernel backs every language binding.

### Usage

```python
from n4m.sklearn import BinnedStratifiedGroupKFoldSplitter
op = BinnedStratifiedGroupKFoldSplitter()
X_transformed = op.fit_transform(X)
```

---

_See also_: [methods index](index.md) · [interactive dashboard](../landing/dashboard.md)