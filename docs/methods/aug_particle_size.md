# `aug_particle_size` — Particle-size and path-length scatter heuristic

_Group_: **Augmentation** · _C ABI_: `n4m_augmentation_particle_size_*`

## Description

Particle-size and path-length scattering simulation.

## Parameters

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

## API and bindings

**C ABI (ABI 2):** [`n4m_augmentation_particle_size_apply`](https://github.com/GBeurier/nirs4all-methods/blob/main/cpp/include/n4m/augmentation/scattering.h#L39) · [`n4m_augmentation_particle_size_create`](https://github.com/GBeurier/nirs4all-methods/blob/main/cpp/include/n4m/augmentation/scattering.h#L30) · [`n4m_augmentation_particle_size_destroy`](https://github.com/GBeurier/nirs4all-methods/blob/main/cpp/include/n4m/augmentation/scattering.h#L42). Use the linked public header for the exact signature, configuration, and result handles.

**Python (verified public re-export):**

```python
from n4m.augmentation.scattering import ParticleSizeAugmenter
```

Source signature: [`ParticleSizeAugmenter(mean_size_um: float = 50.0, size_variation_um: float = 15.0, use_size_range: bool = False, size_range_low_um: float = 5.0, size_range_high_um: float = 500.0, reference_size_um: float = 50.0, wavelength_exponent: float = 1.5, size_effect_strength: float = 0.1, include_path_length: bool = True, path_length_sensitivity: float = 0.5, wavelengths = None, rng: Optional[PCG64] = None, seed: int = 0)`](https://github.com/GBeurier/nirs4all-methods/blob/main/bindings/python/src/n4m/_impl/augmentation.py#L760).

**R:** no current source-verified entry point was found for this catalog method.

**MATLAB / Octave:** no current source-verified entry point was found for this catalog method.

## Explanations

### Bibliographic source

No single paper defines this simulator. It is physically motivated by diffuse scattering but uses an explicit empirical power law documented in the source, not a Mie or Kubelka–Munk solver.

### Mathematical principle

Draw a particle size $d_i$ from the configured uniform range or a normal law clipped to 5–500 µm. With $r_i=d_i/d_0$ and $w(\lambda)=\operatorname{clip}(\lambda/1500,0.1,10)^{-e}$, add the zero-mean baseline $s(r_i^{-1/2}-1)w(\lambda)$. Optionally multiply by $\operatorname{clip}[1+p\log r_i,0.7,1.5]$ and add smoothed Gaussian scatter noise.

### Appropriate uses

Qualitative robustness studies for powders with particle-size and path-length variation.

### Limits and validation

The coefficients and power law are heuristic; wavelengths are required and assumed compatible with the 1500-unit scale. It should not be used for quantitative optical simulation.

### Implementation

Python role API `n4m.augmentation.scattering.ParticleSizeAugmenter`; ABI 2 family `n4m_augmentation_particle_size_{create,apply,destroy}`.

The ABI-2 implementation is the `n4m_augmentation_particle_size_*` lifecycle in libn4m.

### Sources and provenance

https://github.com/GBeurier/nirs4all-methods/blob/main/cpp/src/core/augmentation/scattering/particle_size.h


---

_See also_: [methods index](index.md) · [interactive dashboard](../landing/dashboard.md)