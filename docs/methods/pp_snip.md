# `pp_snip` — Statistics-sensitive nonlinear iterative peak clipping (SNIP)

_Group_: **Baseline correction** · _C ABI_: `n4m_transform_snip_*`

## Description

Statistics-sensitive nonlinear iterative peak-clipping baseline.

## Parameters

| Name | Type | Default |
|------|------|---------|
| `max_half_window` | `int` | `20` |

## API and bindings

**C ABI (ABI 2):** [`n4m_transform_snip_create`](https://github.com/GBeurier/nirs4all-methods/blob/main/cpp/include/n4m/transform/baseline.h#L97) · [`n4m_transform_snip_destroy`](https://github.com/GBeurier/nirs4all-methods/blob/main/cpp/include/n4m/transform/baseline.h#L99) · [`n4m_transform_snip_transform`](https://github.com/GBeurier/nirs4all-methods/blob/main/cpp/include/n4m/transform/baseline.h#L100). Use the linked public header for the exact signature, configuration, and result handles.

**Python (verified public re-export):**

```python
from n4m.transform.baseline import SNIP
```

Source signature: [`SNIP(max_half_window: int = 20)`](https://github.com/GBeurier/nirs4all-methods/blob/main/bindings/python/src/n4m/_impl/baseline.py#L158).

**R:** no current source-verified entry point was found for this catalog method.

**MATLAB / Octave:** no current source-verified entry point was found for this catalog method.

## Explanations

### Bibliographic source

Ryan et al. (1988), *SNIP, a statistics-sensitive background treatment for the quantitative analysis of PIXE spectra in geoscience applications*, Nuclear Instruments and Methods B 34, 396–402, https://doi.org/10.1016/0168-583X(88)90063-8.

### Mathematical principle

After a stabilizing transform, iterative half-windows compare each center with a symmetric neighborhood estimate and clip peaks downward; reversing the transform yields a slowly varying baseline that is subtracted.

### Appropriate uses

Background removal for spectra dominated by positive peaks with a characteristic maximum peak width.

### Limits and validation

`max_half_window` sets which structures are treated as peaks; broad bands can be removed and boundaries receive fewer symmetric comparisons. Published SNIP variants differ in transform and window order.

### Implementation

`n4m.transform.baseline.SNIP` uses `n4m_transform_snip_*`; the exact transform and clipping schedule are in `preprocessing/baselines/snip.c`.

The ABI-2 implementation is the `n4m_transform_snip_*` lifecycle in libn4m.

### Sources and provenance

https://doi.org/10.1016/0168-583X(88)90063-8; https://github.com/GBeurier/nirs4all-methods/blob/main/cpp/src/core/preprocessing/baselines/snip.c


---

_See also_: [methods index](index.md) · [interactive dashboard](../landing/dashboard.md)