# `aug_wavelength_shift` — Rigid wavelength-axis shift

_Group_: **Augmentation** · _C ABI_: `n4m_augmentation_wavelength_shift_*`

## Description

Random spectral shift with linear interpolation.

## Parameters

| Name | Type | Default |
|------|------|---------|
| `shift_lo` | `float` | `-1.0` |
| `shift_hi` | `float` | `1.0` |
| `wavelengths` | `—` | `None` |
| `rng` | `Optional[PCG64]` | `None` |
| `seed` | `int` | `0` |

## API and bindings

**C ABI (ABI 2):** [`n4m_augmentation_wavelength_shift_apply`](https://github.com/GBeurier/nirs4all-methods/blob/main/cpp/include/n4m/augmentation/wavelength.h#L20) · [`n4m_augmentation_wavelength_shift_create`](https://github.com/GBeurier/nirs4all-methods/blob/main/cpp/include/n4m/augmentation/wavelength.h#L15) · [`n4m_augmentation_wavelength_shift_destroy`](https://github.com/GBeurier/nirs4all-methods/blob/main/cpp/include/n4m/augmentation/wavelength.h#L23). Use the linked public header for the exact signature, configuration, and result handles.

**Python (verified public re-export):**

```python
from n4m.augmentation.wavelength import WavelengthShift
```

Source signature: [`WavelengthShift(shift_lo: float = -1.0, shift_hi: float = 1.0, wavelengths = None, rng: Optional[PCG64] = None, seed: int = 0)`](https://github.com/GBeurier/nirs4all-methods/blob/main/bindings/python/src/n4m/_impl/augmentation.py#L513).

**R:** no current source-verified entry point was found for this catalog method.

**MATLAB / Octave:** no current source-verified entry point was found for this catalog method.

## Explanations

### Bibliographic source

No unique paper defines this interpolation augmenter. It is an internal model of wavelength-registration uncertainty.

### Mathematical principle

For each spectrum draw $s_i\sim U(\text{shift_lo},\text{shift_hi})$ and evaluate $X'_i(\lambda_j)=X_i(\lambda_j-s_i)$ by linear interpolation. Outside the measured interval, endpoint values are held as in `numpy.interp`.

### Appropriate uses

Robustness to small calibration offsets in the wavelength axis.

### Limits and validation

Interpolation smooths sharp features and clamps beyond the endpoints. If no wavelength array is supplied, shift units are channel indices rather than nm.

### Implementation

Python role API `n4m.augmentation.wavelength.WavelengthShift`; ABI 2 family `n4m_augmentation_wavelength_shift_{create,apply,destroy}`.

The ABI-2 implementation is the `n4m_augmentation_wavelength_shift_*` lifecycle in libn4m.

### Sources and provenance

https://github.com/GBeurier/nirs4all-methods/blob/main/cpp/src/core/augmentation/wavelength/wavelength_shift.h


---

_See also_: [methods index](index.md) · [interactive dashboard](../landing/dashboard.md)