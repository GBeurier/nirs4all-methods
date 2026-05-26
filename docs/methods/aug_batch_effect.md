# `aug_batch_effect` — Batch Effect Augmenter

_Group_: **Augmentation** · _Binding_: `n4m.sklearn.BatchEffectAugmenter` · _C ABI_: `n4m_aug_batch_effect_*`

## Description

Random offset, slope and gain batch effects.

### Parameters

| Name | Type | Default |
|------|------|---------|
| `offset_std` | `float` | `0.0` |
| `slope_std` | `float` | `0.0` |
| `gain_std` | `float` | `0.0` |
| `variation_scope` | `int` | `0` |
| `wavelengths` | `—` | `None` |
| `rng` | `Optional[PCG64]` | `None` |
| `seed` | `int` | `0` |

## Explanations

### Bibliographic source

_Standard spectroscopic operator — see the nirs4all preprocessing / augmentation handbook and the cited literature within the binding docstring._

### Mathematical principle

Random offset, slope and gain batch effects.

### Implementation

C ABI `n4m_aug_batch_effect_*` in libn4m (create / apply / destroy lifecycle), wrapped by `n4m.sklearn.BatchEffectAugmenter`. The same numerical kernel backs every language binding.

### Usage

```python
from n4m.sklearn import BatchEffectAugmenter
op = BatchEffectAugmenter()
X_transformed = op.fit_transform(X)
```

---

_See also_: [methods index](index.md) · [interactive dashboard](../landing/dashboard.md)