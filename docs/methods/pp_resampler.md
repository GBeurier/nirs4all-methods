# `pp_resampler` — Wavelength-grid resampler

_Group_: **Preprocessing** · _C ABI_: `n4m_transform_resampler_*`

## Description

Interpolate spectra from a fitted source wavelength grid to a target grid.

## Parameters

| Name | Type | Default |
|------|------|---------|
| `target_wavelengths` | `Sequence[float] \| None` | `None` |
| `method` | `int` | `0` |
| `crop_min` | `float` | `0.0` |
| `crop_max` | `float` | `0.0` |
| `use_crop` | `bool` | `False` |
| `fill_value` | `float` | `0.0` |
| `bounds_error` | `bool` | `False` |
| `extrapolate` | `bool` | `False` |

## API and bindings

**C ABI (ABI 2):** [`n4m_transform_resampler_create`](https://github.com/GBeurier/nirs4all-methods/blob/main/cpp/include/n4m/transform/resampling.h#L13) · [`n4m_transform_resampler_destroy`](https://github.com/GBeurier/nirs4all-methods/blob/main/cpp/include/n4m/transform/resampling.h#L21) · [`n4m_transform_resampler_fit`](https://github.com/GBeurier/nirs4all-methods/blob/main/cpp/include/n4m/transform/resampling.h#L22) · [`n4m_transform_resampler_is_fitted`](https://github.com/GBeurier/nirs4all-methods/blob/main/cpp/include/n4m/transform/resampling.h#L25) · [`n4m_transform_resampler_output_cols`](https://github.com/GBeurier/nirs4all-methods/blob/main/cpp/include/n4m/transform/resampling.h#L27) · [`n4m_transform_resampler_transform`](https://github.com/GBeurier/nirs4all-methods/blob/main/cpp/include/n4m/transform/resampling.h#L29). Use the linked public header for the exact signature, configuration, and result handles.

**Python (verified public re-export):**

```python
from n4m.transform.resampling import Resampler
```

Source signature: [`Resampler(target_wavelengths: Sequence[float] | None = None, method: int = 0, crop_min: float = 0.0, crop_max: float = 0.0, use_crop: bool = False, fill_value: float = 0.0, bounds_error: bool = False, extrapolate: bool = False, *, tgt_min: float | None = None, tgt_step: float | None = None, tgt_n: int | None = None)`](https://github.com/GBeurier/nirs4all-methods/blob/main/bindings/python/src/n4m/_impl/resampling.py#L74).

**R:** no current source-verified entry point was found for this catalog method.

**MATLAB / Octave:** no current source-verified entry point was found for this catalog method.

## Explanations

### Bibliographic source

No single canonical paper: the supported linear, nearest, and not-a-knot cubic interpolants follow SciPy interpolation semantics documented at https://docs.scipy.org/doc/scipy/reference/generated/scipy.interpolate.CubicSpline.html.

### Mathematical principle

Fit stores a strictly increasing source wavelength grid, optional crop indices, and target bracketing. Transform evaluates each row on configured target wavelengths by linear, nearest, or not-a-knot cubic interpolation, with explicit fill/extrapolation rules.

### Appropriate uses

Harmonizing spectra acquired on different physical wavelength grids before transfer or modeling.

### Limits and validation

Both grids must be ordered and represent the same physical coordinate. Extrapolation is poorly constrained; cubic interpolation can overshoot and resampling introduces correlated errors.

### Implementation

`n4m.transform.resampling.Resampler` wraps `n4m_transform_resampler_*`; cached brackets and all three interpolation paths are in `resampling/resampler.c`.

The ABI-2 implementation is the `n4m_transform_resampler_*` lifecycle in libn4m.

### Sources and provenance

https://docs.scipy.org/doc/scipy/reference/generated/scipy.interpolate.CubicSpline.html; https://github.com/GBeurier/nirs4all-methods/blob/main/cpp/src/core/preprocessing/resampling/resampler.c


---

_See also_: [methods index](index.md) · [interactive dashboard](../landing/dashboard.md)