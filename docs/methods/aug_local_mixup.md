# `aug_local_mixup` — Local Mixup Augmenter

_Group_: **Augmentation** · _Binding_: `n4m.sklearn.LocalMixupAugmenter` · _C ABI_: `n4m_aug_local_mixup_*`

## Description

Neighbor-constrained mixup augmentation.

### Parameters

| Name | Type | Default |
|------|------|---------|
| `alpha` | `float` | `0.2` |
| `k_neighbors` | `int` | `5` |
| `rng` | `Optional[PCG64]` | `None` |
| `seed` | `int` | `0` |

## Explanations

### Bibliographic source

_Standard spectroscopic operator — see the nirs4all preprocessing / augmentation handbook and the cited literature within the binding docstring._

### Mathematical principle

Neighbor-constrained mixup augmentation.

### Implementation

C ABI `n4m_aug_local_mixup_*` in libn4m (create / apply / destroy lifecycle), wrapped by `n4m.sklearn.LocalMixupAugmenter`. The same numerical kernel backs every language binding.

### Usage

```python
from n4m.sklearn import LocalMixupAugmenter
op = LocalMixupAugmenter()
X_transformed = op.fit_transform(X)
```

---

_See also_: [methods index](index.md) · [interactive dashboard](../landing/dashboard.md)