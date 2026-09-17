# `aug_hetero_noise` — Signal-dependent heteroscedastic noise

_Group_: **Augmentation** · _C ABI_: `n4m_augmentation_hetero_noise_*`

## Description

Noise whose standard deviation depends on signal magnitude.

## Parameters

| Name | Type | Default |
|------|------|---------|
| `noise_base` | `float` | `0.001` |
| `noise_signal_dep` | `float` | `0.01` |
| `rng` | `Optional[PCG64]` | `None` |
| `seed` | `int` | `0` |

## API and bindings

**C ABI (ABI 2):** [`n4m_augmentation_hetero_noise_apply`](https://github.com/GBeurier/nirs4all-methods/blob/main/cpp/include/n4m/augmentation/noise.h#L50) · [`n4m_augmentation_hetero_noise_create`](https://github.com/GBeurier/nirs4all-methods/blob/main/cpp/include/n4m/augmentation/noise.h#L45) · [`n4m_augmentation_hetero_noise_destroy`](https://github.com/GBeurier/nirs4all-methods/blob/main/cpp/include/n4m/augmentation/noise.h#L49). Use the linked public header for the exact signature, configuration, and result handles.

**Python (verified public re-export):**

```python
from n4m.augmentation.noise import HeteroscedasticNoiseAugmenter
```

Source signature: [`HeteroscedasticNoiseAugmenter(noise_base: float = 0.001, noise_signal_dep: float = 0.01, rng: Optional[PCG64] = None, seed: int = 0)`](https://github.com/GBeurier/nirs4all-methods/blob/main/bindings/python/src/n4m/_impl/augmentation.py#L432).

**R:** no current source-verified entry point was found for this catalog method.

**MATLAB / Octave:** no current source-verified entry point was found for this catalog method.

## Explanations

### Bibliographic source

No single publication defines this affine noise law. It is an internal measurement-noise heuristic documented by the native implementation.

### Mathematical principle

At cell $(i,j)$ the noise scale is $s_{ij}=b+c|X_{ij}|$, where $b$ is `noise_base` and $c$ is `noise_signal_dep`; the output is $X'_{ij}=X_{ij}+s_{ij}Z_{ij}$ with $Z_{ij}\sim\mathcal N(0,1)$.

### Appropriate uses

Simulating instruments whose absolute noise increases with signal magnitude.

### Limits and validation

The affine variance law is phenomenological, independent across channels, and can yield a negative scale if parameters are chosen outside their intended non-negative range. It is not a calibrated photon-counting model.

### Implementation

Python role API `n4m.augmentation.noise.HeteroscedasticNoiseAugmenter`; ABI 2 family `n4m_augmentation_hetero_noise_{create,apply,destroy}`.

The ABI-2 implementation is the `n4m_augmentation_hetero_noise_*` lifecycle in libn4m.

### Sources and provenance

https://github.com/GBeurier/nirs4all-methods/blob/main/cpp/src/core/augmentation/noise/hetero_noise.h


---

_See also_: [methods index](index.md) · [interactive dashboard](../landing/dashboard.md)