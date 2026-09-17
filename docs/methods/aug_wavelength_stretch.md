# `aug_wavelength_stretch` — Wavelength-axis stretch about its center

_Group_: **Augmentation** · _C ABI_: `n4m_augmentation_wavelength_stretch_*`

## Description

Random wavelength-axis stretching.

## Parameters

| Name | Type | Default |
|------|------|---------|
| `stretch_lo` | `float` | `0.99` |
| `stretch_hi` | `float` | `1.01` |
| `wavelengths` | `—` | `None` |
| `rng` | `Optional[PCG64]` | `None` |
| `seed` | `int` | `0` |

## API and bindings

**C ABI (ABI 2):** [`n4m_augmentation_wavelength_stretch_apply`](https://github.com/GBeurier/nirs4all-methods/blob/main/cpp/include/n4m/augmentation/wavelength.h#L30) · [`n4m_augmentation_wavelength_stretch_create`](https://github.com/GBeurier/nirs4all-methods/blob/main/cpp/include/n4m/augmentation/wavelength.h#L25) · [`n4m_augmentation_wavelength_stretch_destroy`](https://github.com/GBeurier/nirs4all-methods/blob/main/cpp/include/n4m/augmentation/wavelength.h#L33). Use the linked public header for the exact signature, configuration, and result handles.

**Python (verified public re-export):**

```python
from n4m.augmentation.wavelength import WavelengthStretch
```

Source signature: [`WavelengthStretch(stretch_lo: float = 0.99, stretch_hi: float = 1.01, wavelengths = None, rng: Optional[PCG64] = None, seed: int = 0)`](https://github.com/GBeurier/nirs4all-methods/blob/main/bindings/python/src/n4m/_impl/augmentation.py#L530).

**R:** no current source-verified entry point was found for this catalog method.

**MATLAB / Octave:** no current source-verified entry point was found for this catalog method.

## Explanations

### Bibliographic source

No canonical publication specifies this augmenter; it is a wavelength-scale calibration heuristic whose exact interpolation is defined by the source.

### Mathematical principle

With axis center $\bar\lambda$, draw $f_i\sim U(\text{stretch_lo},\text{stretch_hi})$ and evaluate $X'_i(\lambda_j)=X_i(\bar\lambda+(\lambda_j-\bar\lambda)/f_i)$ using linear interpolation. Factors above one broaden the coordinate scale.

### Appropriate uses

Simulating small wavelength-scale expansion or compression across instruments.

### Limits and validation

The deformation is globally affine and cannot represent local calibration error. Endpoint clamping and the implicit unit grid apply as for wavelength shift.

### Implementation

Python role API `n4m.augmentation.wavelength.WavelengthStretch`; ABI 2 family `n4m_augmentation_wavelength_stretch_{create,apply,destroy}`.

The ABI-2 implementation is the `n4m_augmentation_wavelength_stretch_*` lifecycle in libn4m.

### Sources and provenance

https://github.com/GBeurier/nirs4all-methods/blob/main/cpp/src/core/augmentation/wavelength/wavelength_stretch.h


---

_See also_: [methods index](index.md) · [interactive dashboard](../landing/dashboard.md)