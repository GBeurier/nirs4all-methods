# `aug_moisture` — Moisture Augmenter

_Group_: **Augmentation** · _Binding_: `n4m.sklearn.MoistureAugmenter` · _C ABI_: `n4m_aug_moisture_*`

## Description

Water activity and moisture-content spectral perturbation.

### Parameters

| Name | Type | Default |
|------|------|---------|
| `water_activity_delta` | `float` | `0.0` |
| `use_aw_range` | `bool` | `False` |
| `aw_low` | `float` | `0.0` |
| `aw_high` | `float` | `1.0` |
| `reference_water_activity` | `float` | `0.5` |
| `free_water_fraction` | `float` | `0.3` |
| `bound_water_shift` | `float` | `25.0` |
| `moisture_content` | `float` | `0.1` |
| `enable_shift` | `bool` | `True` |
| `enable_intensity` | `bool` | `True` |
| `wavelengths` | `—` | `None` |
| `rng` | `Optional[PCG64]` | `None` |
| `seed` | `int` | `0` |

## Explanations

### Bibliographic source

_Standard spectroscopic operator — see the nirs4all preprocessing / augmentation handbook and the cited literature within the binding docstring._

### Mathematical principle

Water activity and moisture-content spectral perturbation.

### Implementation

C ABI `n4m_aug_moisture_*` in libn4m (create / apply / destroy lifecycle), wrapped by `n4m.sklearn.MoistureAugmenter`. The same numerical kernel backs every language binding.

### Usage

```python
from n4m.sklearn import MoistureAugmenter
op = MoistureAugmenter()
X_transformed = op.fit_transform(X)
```

---

_See also_: [methods index](index.md) · [interactive dashboard](../landing/dashboard.md)