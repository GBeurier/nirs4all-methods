# `aug_multiplicative_noise` — Per-spectrum multiplicative gain noise

_Group_: **Augmentation** · _C ABI_: `n4m_augmentation_multiplicative_noise_*`

## Description

Apply per-element multiplicative Gaussian noise.

## Parameters

| Name | Type | Default |
|------|------|---------|
| `sigma_gain` | `float` | `0.01` |
| `rng` | `Optional[PCG64]` | `None` |
| `seed` | `int` | `0` |

## API and bindings

**C ABI (ABI 2):** [`n4m_augmentation_multiplicative_noise_apply`](https://github.com/GBeurier/nirs4all-methods/blob/main/cpp/include/n4m/augmentation/noise.h#L31) · [`n4m_augmentation_multiplicative_noise_create`](https://github.com/GBeurier/nirs4all-methods/blob/main/cpp/include/n4m/augmentation/noise.h#L25) · [`n4m_augmentation_multiplicative_noise_destroy`](https://github.com/GBeurier/nirs4all-methods/blob/main/cpp/include/n4m/augmentation/noise.h#L29). Use the linked public header for the exact signature, configuration, and result handles.

**Python (verified public re-export):**

```python
from n4m.augmentation.noise import MultiplicativeNoise
```

Source signature: [`MultiplicativeNoise(sigma_gain: float = 0.01, rng: Optional[PCG64] = None, seed: int = 0)`](https://github.com/GBeurier/nirs4all-methods/blob/main/bindings/python/src/n4m/_impl/augmentation.py#L395).

**R:** no current source-verified entry point was found for this catalog method.

**MATLAB / Octave:** no current source-verified entry point was found for this catalog method.

## Explanations

### Bibliographic source

No publication uniquely defines this implementation. It belongs to the family of spectral offset/slope/gain augmentation discussed by Bjerrum et al. (2017), arXiv:1710.01927 (https://arxiv.org/abs/1710.01927).

### Mathematical principle

One draw is made per row: $g_i=1+\sigma_g Z_i$, $Z_i\sim\mathcal N(0,1)$, then $X'_{ij}=g_iX_{ij}$ for every wavelength. `sigma_gain` controls relative gain variation, so all channels in a spectrum move together.

### Appropriate uses

Simulating sample-wise optical gain, concentration scale, or path-length changes.

### Limits and validation

The public operator implements per-sample gain only: it does not draw an independent gain at every wavelength and does not constrain $g_i$ to be positive. Large `sigma_gain` can invert a spectrum.

### Implementation

Python role API `n4m.augmentation.noise.MultiplicativeNoise`; ABI 2 family `n4m_augmentation_multiplicative_noise_{create,apply,destroy}`.

The ABI-2 implementation is the `n4m_augmentation_multiplicative_noise_*` lifecycle in libn4m.

### Sources and provenance

https://github.com/GBeurier/nirs4all-methods/blob/main/cpp/src/core/augmentation/noise/multiplicative_noise.h; https://arxiv.org/abs/1710.01927


---

_See also_: [methods index](index.md) · [interactive dashboard](../landing/dashboard.md)