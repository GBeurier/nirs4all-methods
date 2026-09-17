# `pp_detrend` — Polynomial spectral detrending

_Group_: **Baseline correction** · _C ABI_: `n4m_transform_detrend_*`

## Description

Polynomial baseline subtraction.

## Parameters

| Name | Type | Default |
|------|------|---------|
| `polyorder` | `int` | `1` |

## API and bindings

**C ABI (ABI 2):** [`n4m_transform_detrend_create`](https://github.com/GBeurier/nirs4all-methods/blob/main/cpp/include/n4m/transform/baseline.h#L14) · [`n4m_transform_detrend_destroy`](https://github.com/GBeurier/nirs4all-methods/blob/main/cpp/include/n4m/transform/baseline.h#L16) · [`n4m_transform_detrend_transform`](https://github.com/GBeurier/nirs4all-methods/blob/main/cpp/include/n4m/transform/baseline.h#L17). Use the linked public header for the exact signature, configuration, and result handles.

**Python (verified public re-export):**

```python
from n4m.transform.baseline import Detrend
```

Source signature: [`Detrend(polyorder: int = 1)`](https://github.com/GBeurier/nirs4all-methods/blob/main/bindings/python/src/n4m/_impl/baseline.py#L12).

**R:** no current source-verified entry point was found for this catalog method.

**MATLAB / Octave:** no current source-verified entry point was found for this catalog method.

## Explanations

### Bibliographic source

Barnes, Dhanoa & Lister (1989), *Standard Normal Variate Transformation and De-trending of Near-Infrared Diffuse Reflectance Spectra*, Applied Spectroscopy 43, 772–777, https://doi.org/10.1366/0003702894202201.

### Mathematical principle

For each row, least squares fits a polynomial of degree `polyorder` against channel position and subtracts the fitted curve. Degree one removes offset and slope; higher degrees remove progressively curved backgrounds.

### Appropriate uses

Removing smooth per-spectrum baseline drift before calibration, often after or alongside scatter normalization.

### Limits and validation

A high polynomial degree can remove broad chemical bands or oscillate near edges; channel positions are treated as equally spaced unless the data were resampled.

### Implementation

`n4m.transform.baseline.Detrend` uses `n4m_transform_detrend_*`; the row-wise normal equations are implemented in `preprocessing/baselines/detrend.c`.

The ABI-2 implementation is the `n4m_transform_detrend_*` lifecycle in libn4m.

### Sources and provenance

https://doi.org/10.1366/0003702894202201; https://github.com/GBeurier/nirs4all-methods/blob/main/cpp/src/core/preprocessing/baselines/detrend.c


---

_See also_: [methods index](index.md) · [interactive dashboard](../landing/dashboard.md)