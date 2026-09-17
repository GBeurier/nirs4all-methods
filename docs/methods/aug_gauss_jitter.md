# `aug_gauss_jitter` — Random Gaussian smoothing jitter

_Group_: **Augmentation** · _C ABI_: `n4m_augmentation_gauss_jitter_*`

## Description

Gaussian smoothing jitter.

## Parameters

| Name | Type | Default |
|------|------|---------|
| `sigma_lo` | `float` | `0.5` |
| `sigma_hi` | `float` | `1.5` |
| `kernel_width` | `int` | `9` |
| `rng` | `Optional[PCG64]` | `None` |
| `seed` | `int` | `0` |

## API and bindings

**C ABI (ABI 2):** [`n4m_augmentation_gauss_jitter_apply`](https://github.com/GBeurier/nirs4all-methods/blob/main/cpp/include/n4m/augmentation/spectral.h#L57) · [`n4m_augmentation_gauss_jitter_create`](https://github.com/GBeurier/nirs4all-methods/blob/main/cpp/include/n4m/augmentation/spectral.h#L52) · [`n4m_augmentation_gauss_jitter_destroy`](https://github.com/GBeurier/nirs4all-methods/blob/main/cpp/include/n4m/augmentation/spectral.h#L60). Use the linked public header for the exact signature, configuration, and result handles.

**Python (verified public re-export):**

```python
from n4m.augmentation.spectral import GaussianJitter
```

Source signature: [`GaussianJitter(sigma_lo: float = 0.5, sigma_hi: float = 1.5, kernel_width: int = 9, rng: Optional[PCG64] = None, seed: int = 0)`](https://github.com/GBeurier/nirs4all-methods/blob/main/bindings/python/src/n4m/_impl/augmentation.py#L635).

**R:** no current source-verified entry point was found for this catalog method.

**MATLAB / Octave:** no current source-verified entry point was found for this catalog method.

## Explanations

### Bibliographic source

No unique paper defines randomizing the smoothing width. The operator is a stochastic Gaussian convolution whose source defines boundary behavior.

### Mathematical principle

For each spectrum draw $s_i\sim U(\text{sigma_lo},\text{sigma_hi})$, build the normalized odd-width kernel $k(t)\propto\exp[-t^2/(2s_i^2)]$, and convolve along wavelength with reflect padding. `kernel_width` truncates the Gaussian.

### Appropriate uses

Robustness to small variations in spectral resolution or smoothing strength.

### Limits and validation

This is smoothing, despite the name “jitter”; it adds no random residual noise. Finite kernel width and reflected boundaries affect edge bands.

### Implementation

Python role API `n4m.augmentation.spectral.GaussianJitter`; ABI 2 family `n4m_augmentation_gauss_jitter_{create,apply,destroy}`.

The ABI-2 implementation is the `n4m_augmentation_gauss_jitter_*` lifecycle in libn4m.

### Sources and provenance

https://github.com/GBeurier/nirs4all-methods/blob/main/cpp/src/core/augmentation/spectral/gauss_jitter.h


---

_See also_: [methods index](index.md) · [interactive dashboard](../landing/dashboard.md)