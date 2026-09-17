# `pp_dtw_align` — Dynamic time warping spectral alignment

_Group_: **Signal transforms** · _C ABI_: `n4m_transform_dtw_align_*`

## Description

Dynamic-time-warping alignment to a fixed-length reference.

## Parameters

| Name | Type | Default |
|------|------|---------|
| `reference` | `—` | `None` |

## API and bindings

**C ABI (ABI 2):** [`n4m_transform_dtw_align_create`](https://github.com/GBeurier/nirs4all-methods/blob/main/cpp/include/n4m/transform/alignment.h#L38) · [`n4m_transform_dtw_align_destroy`](https://github.com/GBeurier/nirs4all-methods/blob/main/cpp/include/n4m/transform/alignment.h#L41) · [`n4m_transform_dtw_align_fit`](https://github.com/GBeurier/nirs4all-methods/blob/main/cpp/include/n4m/transform/alignment.h#L42) · [`n4m_transform_dtw_align_is_fitted`](https://github.com/GBeurier/nirs4all-methods/blob/main/cpp/include/n4m/transform/alignment.h#L47) · [`n4m_transform_dtw_align_transform`](https://github.com/GBeurier/nirs4all-methods/blob/main/cpp/include/n4m/transform/alignment.h#L44). Use the linked public header for the exact signature, configuration, and result handles.

**Python (verified public re-export):**

```python
from n4m.transform.alignment import DynamicTimeWarpingAlignment
```

Source signature: [`DynamicTimeWarpingAlignment(reference = None)`](https://github.com/GBeurier/nirs4all-methods/blob/main/bindings/python/src/n4m/_impl/advanced.py#L427).

**R:** no current source-verified entry point was found for this catalog method.

**MATLAB / Octave:** no current source-verified entry point was found for this catalog method.

## Explanations

### Bibliographic source

Sakoe & Chiba (1978), *Dynamic programming algorithm optimization for spoken word recognition*, IEEE TASSP 26, 43–49, https://doi.org/10.1109/TASSP.1978.1163055.

### Mathematical principle

A dynamic program finds a minimum-cost monotone path through pairwise sample/reference distances. The path maps local stretches and compressions to the reference grid; multiple matched input points are aggregated to keep a fixed output length.

### Appropriate uses

Aligning spectra with nonlinear local wavelength shifts that cannot be represented by one global offset.

### Limits and validation

Unconstrained DTW can align unrelated bands and is quadratic in spectrum length. Amplitude differences affect the path, and the result depends on the reference.

### Implementation

`n4m.transform.alignment.DynamicTimeWarpingAlignment` wraps `n4m_transform_dtw_align_*`; the native path computation is in `cpp/src/c_api/c_api_advanced.cpp`.

The ABI-2 implementation is the `n4m_transform_dtw_align_*` lifecycle in libn4m.

### Sources and provenance

https://doi.org/10.1109/TASSP.1978.1163055; https://github.com/GBeurier/nirs4all-methods/blob/main/cpp/src/c_api/c_api_advanced.cpp


---

_See also_: [methods index](index.md) · [interactive dashboard](../landing/dashboard.md)