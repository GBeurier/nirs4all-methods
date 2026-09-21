# `pp_piecewise_msc` — Piecewise multiplicative scatter correction

_Group_: **Signal transforms** · _C ABI_: `n4m_transform_piecewise_msc_*`

## Description

Apply MSC independently inside fixed wavelength intervals.

## Parameters

| Name | Type | Default |
|------|------|---------|
| `window_size` | `int` | `32` |
| `reference` | `—` | `None` |
| `eps` | `float` | `1e-12` |

## API and bindings

**C ABI (ABI 2):** [`n4m_transform_piecewise_msc_create`](https://github.com/GBeurier/nirs4all-methods/blob/main/cpp/include/n4m/transform/scatter.h#L182) · [`n4m_transform_piecewise_msc_destroy`](https://github.com/GBeurier/nirs4all-methods/blob/main/cpp/include/n4m/transform/scatter.h#L185) · [`n4m_transform_piecewise_msc_fit`](https://github.com/GBeurier/nirs4all-methods/blob/main/cpp/include/n4m/transform/scatter.h#L187) · [`n4m_transform_piecewise_msc_is_fitted`](https://github.com/GBeurier/nirs4all-methods/blob/main/cpp/include/n4m/transform/scatter.h#L192) · [`n4m_transform_piecewise_msc_transform`](https://github.com/GBeurier/nirs4all-methods/blob/main/cpp/include/n4m/transform/scatter.h#L189). Use the linked public header for the exact signature, configuration, and result handles.

**Python (verified public re-export):**

```python
from n4m.transform.scatter import PiecewiseMSC
```

Source signature: [`PiecewiseMSC()`](https://github.com/GBeurier/nirs4all-methods/blob/main/bindings/python/src/n4m/_impl/advanced.py#L346).

**R:** no current source-verified entry point was found for this catalog method.

**MATLAB / Octave:** no current source-verified entry point was found for this catalog method.

## Explanations

### Bibliographic source

No single paper canonically defines this fixed-window variant. It applies the MSC model of Geladi, MacDougall & Martens (1985), https://doi.org/10.1366/0003702854248656, independently by interval.

### Mathematical principle

The wavelength axis is partitioned into non-overlapping windows. Within each window and row, local intercept and slope are estimated against the corresponding reference segment, and every value in that window is corrected by $(x-a)/b$.

### Appropriate uses

Scatter correction when gain and offset differ between broad spectral regions.

### Limits and validation

Independent windows can create discontinuities and short or flat segments make slopes unstable. The reference is training-dependent and the method can remove regional chemical amplitude differences.

### Implementation

`n4m.transform.scatter.PiecewiseMSC` wraps `n4m_transform_piecewise_msc_*`; reference fitting and segment OLS are in `cpp/src/c_api/c_api_advanced.cpp`.

The ABI-2 implementation is the `n4m_transform_piecewise_msc_*` lifecycle in libn4m.

### Sources and provenance

https://doi.org/10.1366/0003702854248656; https://github.com/GBeurier/nirs4all-methods/blob/main/cpp/src/c_api/c_api_advanced.cpp


---

_See also_: [methods index](index.md) · [interactive dashboard](../landing/dashboard.md)