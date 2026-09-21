# `pp_resample` — Fixed-length normalized-axis resampling

_Group_: **Preprocessing** · _C ABI_: `n4m_transform_resample_transformer_*`

## Description

Resize spectra to a fixed column count.

## Parameters

| Name | Type | Default |
|------|------|---------|
| `num_samples` | `int` | `—` |

## API and bindings

**C ABI (ABI 2):** [`n4m_transform_resample_transformer_create`](https://github.com/GBeurier/nirs4all-methods/blob/main/cpp/include/n4m/transform/resampling.h#L46) · [`n4m_transform_resample_transformer_destroy`](https://github.com/GBeurier/nirs4all-methods/blob/main/cpp/include/n4m/transform/resampling.h#L48) · [`n4m_transform_resample_transformer_output_cols`](https://github.com/GBeurier/nirs4all-methods/blob/main/cpp/include/n4m/transform/resampling.h#L49) · [`n4m_transform_resample_transformer_transform`](https://github.com/GBeurier/nirs4all-methods/blob/main/cpp/include/n4m/transform/resampling.h#L51). Use the linked public header for the exact signature, configuration, and result handles.

**Python (verified public re-export):**

```python
from n4m.transform.resampling import ResampleTransformer
```

Source signature: [`ResampleTransformer(num_samples: int)`](https://github.com/GBeurier/nirs4all-methods/blob/main/bindings/python/src/n4m/_impl/resampling.py#L46).

**R:** no current source-verified entry point was found for this catalog method.

**MATLAB / Octave:** no current source-verified entry point was found for this catalog method.

## Explanations

### Bibliographic source

No unique canonical paper: this is piecewise-linear interpolation on normalized sample positions, matching SciPy `interp1d(kind='linear')`: https://docs.scipy.org/doc/scipy/reference/generated/scipy.interpolate.interp1d.html.

### Mathematical principle

For an input of width $p$ and requested width $q$, source and destination coordinates are `linspace(0,1,p)` and `linspace(0,1,q)`. Each row is linearly interpolated; equal width is an exact copy and `-1` is identity.

### Appropriate uses

Making variable-width or differently sampled spectra a common feature length when physical wavelength coordinates are unavailable.

### Limits and validation

Normalized positions assume matching endpoints and uniform semantic coverage. The operator ignores actual wavelengths and linear interpolation can smooth narrow bands.

### Implementation

`n4m.transform.resampling.ResampleTransformer` uses `n4m_transform_resample_transformer_*`; the linspace and interpolation arithmetic are in `resampling/resample_transformer.c`.

The ABI-2 implementation is the `n4m_transform_resample_transformer_*` lifecycle in libn4m.

### Sources and provenance

https://docs.scipy.org/doc/scipy/reference/generated/scipy.interpolate.interp1d.html; https://github.com/GBeurier/nirs4all-methods/blob/main/cpp/src/core/preprocessing/resampling/resample_transformer.c


---

_See also_: [methods index](index.md) · [interactive dashboard](../landing/dashboard.md)