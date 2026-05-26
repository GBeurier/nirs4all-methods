# `aug_dead_band` — Dead Band Augmenter

_Group_: **Augmentation** · _Binding_: `n4m.sklearn.DeadBandAugmenter` · _C ABI_: `n4m_aug_dead_band_*`

## Description

Simulate dead spectral detector bands.

### Parameters

| Name | Type | Default |
|------|------|---------|
| `n_bands` | `int` | `1` |
| `width_low` | `int` | `5` |
| `width_high` | `int` | `10` |
| `noise_std` | `float` | `0.05` |
| `probability` | `float` | `0.0` |
| `variation_scope` | `int` | `0` |
| `rng` | `Optional[PCG64]` | `None` |
| `seed` | `int` | `0` |

## Explanations

### Bibliographic source

_Standard spectroscopic operator — see the nirs4all preprocessing / augmentation handbook and the cited literature within the binding docstring._

### Mathematical principle

Simulate dead spectral detector bands.

### Implementation

C ABI `n4m_aug_dead_band_*` in libn4m (create / apply / destroy lifecycle), wrapped by `n4m.sklearn.DeadBandAugmenter`. The same numerical kernel backs every language binding.

### Usage

```python
from n4m.sklearn import DeadBandAugmenter
op = DeadBandAugmenter()
X_transformed = op.fit_transform(X)
```

---

_See also_: [methods index](index.md) · [interactive dashboard](../landing/dashboard.md)