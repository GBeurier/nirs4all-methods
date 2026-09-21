# `pp_gaussian` — Gaussian smoothing and derivative filtering

_Group_: **Preprocessing** · _C ABI_: `n4m_transform_gaussian_*`

## Description

SciPy-compatible 1-D Gaussian filter along the wavelength axis.

## Parameters

| Name | Type | Default |
|------|------|---------|
| `sigma` | `float` | `1.0` |
| `order` | `int` | `0` |
| `mode` | `str` | `'reflect'` |
| `cval` | `float` | `0.0` |
| `truncate` | `float` | `4.0` |

## API and bindings

**C ABI (ABI 2):** [`n4m_transform_gaussian_create`](https://github.com/GBeurier/nirs4all-methods/blob/main/cpp/include/n4m/transform/smoothing.h#L156) · [`n4m_transform_gaussian_destroy`](https://github.com/GBeurier/nirs4all-methods/blob/main/cpp/include/n4m/transform/smoothing.h#L161) · [`n4m_transform_gaussian_transform`](https://github.com/GBeurier/nirs4all-methods/blob/main/cpp/include/n4m/transform/smoothing.h#L162). Use the linked public header for the exact signature, configuration, and result handles.

**Python (verified public re-export):**

```python
from n4m.transform.smoothing import Gaussian
```

Source signature: [`Gaussian(sigma: float = 1.0, order: int = 0, mode: str = 'reflect', cval: float = 0.0, truncate: float = 4.0)`](https://github.com/GBeurier/nirs4all-methods/blob/main/bindings/python/src/n4m/_impl/preprocessing.py#L353).

**R:** no current source-verified entry point was found for this catalog method.

**MATLAB / Octave:** no current source-verified entry point was found for this catalog method.

## Explanations

### Bibliographic source

No single canonical spectroscopic paper: this is discrete Gaussian convolution. The public reference behaviour is SciPy `gaussian_filter1d`: https://docs.scipy.org/doc/scipy/reference/generated/scipy.ndimage.gaussian_filter1d.html.

### Mathematical principle

A truncated discrete Gaussian of standard deviation `sigma` is convolved along each spectrum; positive `order` uses the corresponding derivative-of-Gaussian kernel. Boundary samples follow the selected extension mode and `cval`.

### Appropriate uses

Low-pass denoising, or smoothed derivative estimation, with a continuously tunable spectral scale.

### Limits and validation

Smoothing broadens narrow bands and derivative orders amplify noise. `sigma` is in channel units, so unequal wavelength grids require resampling; boundary mode affects edge bands.

### Implementation

`n4m.transform.smoothing.Gaussian` calls `n4m_transform_gaussian_*`; kernel generation and boundary handling are in `preprocessing/smoothing/gaussian.c`.

The ABI-2 implementation is the `n4m_transform_gaussian_*` lifecycle in libn4m.

### Sources and provenance

https://docs.scipy.org/doc/scipy/reference/generated/scipy.ndimage.gaussian_filter1d.html; https://github.com/GBeurier/nirs4all-methods/blob/main/cpp/src/core/preprocessing/smoothing/gaussian.c


---

_See also_: [methods index](index.md) · [interactive dashboard](../landing/dashboard.md)