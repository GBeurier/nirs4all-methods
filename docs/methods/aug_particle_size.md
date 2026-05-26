# `aug_particle_size` — Particle Size Augmenter

_Group_: **Augmentation** · _Binding_: `n4m.sklearn.ParticleSizeAugmenter` · _C ABI_: `n4m_aug_particle_size_*`

## Description

Particle-size and path-length scattering simulation.

### Parameters

| Name | Type | Default |
|------|------|---------|
| `mean_size_um` | `float` | `50.0` |
| `size_variation_um` | `float` | `15.0` |
| `use_size_range` | `bool` | `False` |
| `size_range_low_um` | `float` | `5.0` |
| `size_range_high_um` | `float` | `500.0` |
| `reference_size_um` | `float` | `50.0` |
| `wavelength_exponent` | `float` | `1.5` |
| `size_effect_strength` | `float` | `0.1` |
| `include_path_length` | `bool` | `True` |
| `path_length_sensitivity` | `float` | `0.5` |
| `wavelengths` | `—` | `None` |
| `rng` | `Optional[PCG64]` | `None` |
| `seed` | `int` | `0` |

## Explanations

### Bibliographic source

_Standard spectroscopic operator — see the nirs4all preprocessing / augmentation handbook and the cited literature within the binding docstring._

### Mathematical principle

Particle-size and path-length scattering simulation.

### Implementation

C ABI `n4m_aug_particle_size_*` in libn4m (create / apply / destroy lifecycle), wrapped by `n4m.sklearn.ParticleSizeAugmenter`. The same numerical kernel backs every language binding.

### Usage

```python
from n4m.sklearn import ParticleSizeAugmenter
op = ParticleSizeAugmenter()
X_transformed = op.fit_transform(X)
```

---

_See also_: [methods index](index.md) · [interactive dashboard](../landing/dashboard.md)