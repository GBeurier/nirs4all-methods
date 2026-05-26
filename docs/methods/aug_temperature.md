# `aug_temperature` — Temperature Augmenter

_Group_: **Augmentation** · _Binding_: `n4m.sklearn.TemperatureAugmenter` · _C ABI_: `n4m_aug_temperature_*`

## Description

Temperature-induced shift, intensity and broadening perturbations.

### Parameters

| Name | Type | Default |
|------|------|---------|
| `temperature_delta` | `float` | `0.0` |
| `use_temp_range` | `bool` | `False` |
| `temp_low` | `float` | `-5.0` |
| `temp_high` | `float` | `5.0` |
| `enable_shift` | `bool` | `True` |
| `enable_intensity` | `bool` | `True` |
| `enable_broadening` | `bool` | `True` |
| `region_specific` | `bool` | `True` |
| `wavelengths` | `—` | `None` |
| `rng` | `Optional[PCG64]` | `None` |
| `seed` | `int` | `0` |

## Explanations

### Bibliographic source

_Standard spectroscopic operator — see the nirs4all preprocessing / augmentation handbook and the cited literature within the binding docstring._

### Mathematical principle

Temperature-induced shift, intensity and broadening perturbations.

### Implementation

C ABI `n4m_aug_temperature_*` in libn4m (create / apply / destroy lifecycle), wrapped by `n4m.sklearn.TemperatureAugmenter`. The same numerical kernel backs every language binding.

### Usage

```python
from n4m.sklearn import TemperatureAugmenter
op = TemperatureAugmenter()
X_transformed = op.fit_transform(X)
```

---

_See also_: [methods index](index.md) · [interactive dashboard](../landing/dashboard.md)