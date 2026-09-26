# `aug_stray_light` — Stray-light distortion in transmittance space

_Group_: **Augmentation** · _C ABI_: `n4m_augmentation_stray_light_*`

## Description

Stray-light edge artifact.

## Parameters

| Name | Type | Default |
|------|------|---------|
| `stray_light_fraction` | `float` | `0.001` |
| `edge_enhancement` | `float` | `2.0` |
| `edge_width` | `float` | `0.1` |
| `include_peak_truncation` | `bool` | `True` |
| `wavelengths` | `—` | `None` |
| `rng` | `Optional[PCG64]` | `None` |
| `seed` | `int` | `0` |

## API and bindings

**C ABI (ABI 2):** [`n4m_augmentation_stray_light_apply`](https://github.com/GBeurier/nirs4all-methods/blob/main/cpp/include/n4m/augmentation/instrument.h#L73) · [`n4m_augmentation_stray_light_create`](https://github.com/GBeurier/nirs4all-methods/blob/main/cpp/include/n4m/augmentation/instrument.h#L66) · [`n4m_augmentation_stray_light_destroy`](https://github.com/GBeurier/nirs4all-methods/blob/main/cpp/include/n4m/augmentation/instrument.h#L78). Use the linked public header for the exact signature, configuration, and result handles.

**Python (verified public re-export):**

```python
from n4m.augmentation.instrument import StrayLightAugmenter
```

Source signature: [`StrayLightAugmenter(stray_light_fraction: float = 0.001, edge_enhancement: float = 2.0, edge_width: float = 0.1, include_peak_truncation: bool = True, wavelengths = None, rng: Optional[PCG64] = None, seed: int = 0)`](https://github.com/GBeurier/nirs4all-methods/blob/main/bindings/python/src/n4m/_impl/augmentation.py#L1012).

**R:** no current source-verified entry point was found for this catalog method.

**MATLAB / Octave:** no current source-verified entry point was found for this catalog method.

## Explanations

### Bibliographic source

No canonical publication defines this edge-enhanced profile. The physical mixing equation is standard, while its wavelength profile and optional truncation are source-defined.

### Mathematical principle

Convert absorbance $A$ to transmittance $T=10^{-A}$, form an edge-enhanced stray fraction $s(\lambda)$, compute $T_{obs}=(T+s)/(1+s)$, and return $A'=-\log_{10}T_{obs}$. `edge_width` and `edge_enhancement` shape $s$; optional peak truncation models loss of high absorbance.

### Appropriate uses

Testing nonlinear absorbance compression caused by out-of-band or internal stray light.

### Limits and validation

Input is assumed to be base-10 absorbance. Applying the formula directly to reflectance, transmittance, derivatives, or standardized data is not physically meaningful.

### Implementation

Python role API `n4m.augmentation.instrument.StrayLightAugmenter`; ABI 2 family `n4m_augmentation_stray_light_{create,apply,destroy}`.

The ABI-2 implementation is the `n4m_augmentation_stray_light_*` lifecycle in libn4m.

### Sources and provenance

https://github.com/GBeurier/nirs4all-methods/blob/main/cpp/src/core/augmentation/edge_artifacts/stray_light.h


---

_See also_: [methods index](index.md) · [interactive dashboard](../landing/dashboard.md)