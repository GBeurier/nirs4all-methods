# `pp_piecewise_snv` — Piecewise standard normal variate

_Group_: **Signal transforms** · _C ABI_: `n4m_transform_piecewise_snv_*`

## Description

Apply SNV independently inside fixed wavelength intervals.

## Parameters

| Name | Type | Default |
|------|------|---------|
| `window_size` | `int` | `32` |
| `ddof` | `int` | `0` |
| `eps` | `float` | `1e-12` |

## API and bindings

**C ABI (ABI 2):** [`n4m_transform_piecewise_snv_create`](https://github.com/GBeurier/nirs4all-methods/blob/main/cpp/include/n4m/transform/scatter.h#L184) · [`n4m_transform_piecewise_snv_destroy`](https://github.com/GBeurier/nirs4all-methods/blob/main/cpp/include/n4m/transform/scatter.h#L187) · [`n4m_transform_piecewise_snv_fit`](https://github.com/GBeurier/nirs4all-methods/blob/main/cpp/include/n4m/transform/scatter.h#L188) · [`n4m_transform_piecewise_snv_is_fitted`](https://github.com/GBeurier/nirs4all-methods/blob/main/cpp/include/n4m/transform/scatter.h#L193) · [`n4m_transform_piecewise_snv_transform`](https://github.com/GBeurier/nirs4all-methods/blob/main/cpp/include/n4m/transform/scatter.h#L190). Use the linked public header for the exact signature, configuration, and result handles.

**Python (verified public re-export):**

```python
from n4m.transform.scatter import PiecewiseSNV
```

Source signature: [`PiecewiseSNV(window_size: int = 32, ddof: int = 0, eps: float = 1e-12)`](https://github.com/GBeurier/nirs4all-methods/blob/main/bindings/python/src/n4m/_impl/advanced.py#L277).

**R:** no current source-verified entry point was found for this catalog method.

**MATLAB / Octave:** no current source-verified entry point was found for this catalog method.

## Explanations

### Bibliographic source

No single paper canonically defines this interval variant. It applies the SNV normalization of Barnes, Dhanoa & Lister (1989), https://doi.org/10.1366/0003702894202201, independently by interval.

### Mathematical principle

The axis is partitioned into non-overlapping windows of `window_size`. Within each window, each row is centered by its local mean and divided by its local standard deviation using `ddof`, with variance floored by `eps`.

### Appropriate uses

Removing region-dependent scatter while preserving separation between spectral regions better than a single global SNV statistic.

### Limits and validation

Window boundaries can cause jumps, and low-variance or narrow intervals amplify noise. Local normalization may erase broad analyte differences between regions.

### Implementation

`n4m.transform.scatter.PiecewiseSNV` calls `n4m_transform_piecewise_snv_*`; the exact interval and variance rules are in `cpp/src/c_api/c_api_advanced.cpp`.

The ABI-2 implementation is the `n4m_transform_piecewise_snv_*` lifecycle in libn4m.

### Sources and provenance

https://doi.org/10.1366/0003702894202201; https://github.com/GBeurier/nirs4all-methods/blob/main/cpp/src/c_api/c_api_advanced.cpp


---

_See also_: [methods index](index.md) · [interactive dashboard](../landing/dashboard.md)