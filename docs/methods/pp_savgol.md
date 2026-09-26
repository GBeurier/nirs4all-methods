# `pp_savgol` — Savitzky–Golay smoothing and differentiation

_Group_: **Preprocessing** · _C ABI_: `n4m_transform_savitzky_golay_*`

## Description

scipy.signal.savgol_filter parity.

## Parameters

| Name | Type | Default |
|------|------|---------|
| `window_length` | `int` | `5` |
| `polyorder` | `int` | `2` |
| `deriv` | `int` | `0` |
| `delta` | `float` | `1.0` |
| `mode` | `str` | `'mirror'` |
| `cval` | `float` | `0.0` |

## API and bindings

**C ABI (ABI 2):** [`n4m_transform_savitzky_golay_create`](https://github.com/GBeurier/nirs4all-methods/blob/main/cpp/include/n4m/transform/smoothing.h#L74) · [`n4m_transform_savitzky_golay_destroy`](https://github.com/GBeurier/nirs4all-methods/blob/main/cpp/include/n4m/transform/smoothing.h#L81) · [`n4m_transform_savitzky_golay_transform`](https://github.com/GBeurier/nirs4all-methods/blob/main/cpp/include/n4m/transform/smoothing.h#L82). Use the linked public header for the exact signature, configuration, and result handles.

**Python (verified public re-export):**

```python
from n4m.transform.smoothing import SavitzkyGolay
```

Source signature: [`SavitzkyGolay(window_length: int = 5, polyorder: int = 2, deriv: int = 0, delta: float = 1.0, mode: str = 'mirror', cval: float = 0.0)`](https://github.com/GBeurier/nirs4all-methods/blob/main/bindings/python/src/n4m/_impl/preprocessing.py#L385).

**R (source-verified):** [`savgol_transform(X, window_length, polyorder = 3L, deriv = 0L, delta = 1.0, mode = "mirror", cval = 0.0)`](https://github.com/GBeurier/nirs4all-methods/blob/main/bindings/r/n4m/R/preprocessing.R).

The source signature has additional required inputs, so no example call is fabricated.

**MATLAB / Octave (source-verified):** [`savgol_transform(X, varargin)`](https://github.com/GBeurier/nirs4all-methods/blob/main/bindings/matlab/+n4m/savgol_transform.m).

```matlab
addpath('bindings/matlab')
result = n4m.savgol_transform(X);
```

## Explanations

### Bibliographic source

Savitzky & Golay (1964), *Smoothing and Differentiation of Data by Simplified Least Squares Procedures*, Analytical Chemistry 36, 1627–1639, https://doi.org/10.1021/ac60214a047.

### Mathematical principle

In every odd window, a polynomial of degree `polyorder` is fit by least squares. The value or derivative of order `deriv` at the center is a precomputed convolution of the window, scaled by `delta`; `mode` defines edge extension.

### Appropriate uses

Denoising while preserving polynomial peak shape, or estimating smoothed first and second derivatives.

### Limits and validation

Window, degree, derivative, and grid spacing must be coherent. Derivatives amplify noise, large windows merge narrow bands, and edge modes produce different boundary values.

### Implementation

`n4m.transform.smoothing.SavitzkyGolay` wraps `n4m_transform_savitzky_golay_*`; coefficient generation and modes are in `preprocessing/derivatives/savitzky_golay.c`.

The ABI-2 implementation is the `n4m_transform_savitzky_golay_*` lifecycle in libn4m.

### Sources and provenance

https://doi.org/10.1021/ac60214a047; https://github.com/GBeurier/nirs4all-methods/blob/main/cpp/src/core/preprocessing/derivatives/savitzky_golay.c


---

_See also_: [methods index](index.md) · [interactive dashboard](../landing/dashboard.md)