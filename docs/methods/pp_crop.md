# `pp_crop` — Wavelength-column cropping

_Group_: **Preprocessing** · _C ABI_: `n4m_transform_crop_*`

## Description

Slice wavelength columns in the half-open interval ``[start, end)``.

## Parameters

| Name | Type | Default |
|------|------|---------|
| `start` | `int` | `—` |
| `end` | `int` | `—` |

## API and bindings

**C ABI (ABI 2):** [`n4m_transform_crop_create`](https://github.com/GBeurier/nirs4all-methods/blob/main/cpp/include/n4m/transform/resampling.h#L35) · [`n4m_transform_crop_destroy`](https://github.com/GBeurier/nirs4all-methods/blob/main/cpp/include/n4m/transform/resampling.h#L37) · [`n4m_transform_crop_output_cols`](https://github.com/GBeurier/nirs4all-methods/blob/main/cpp/include/n4m/transform/resampling.h#L38) · [`n4m_transform_crop_transform`](https://github.com/GBeurier/nirs4all-methods/blob/main/cpp/include/n4m/transform/resampling.h#L40). Use the linked public header for the exact signature, configuration, and result handles.

**Python (verified public re-export):**

```python
from n4m.transform.resampling import CropTransformer
```

Source signature: [`CropTransformer(start: int, end: int)`](https://github.com/GBeurier/nirs4all-methods/blob/main/bindings/python/src/n4m/_impl/resampling.py#L17).

**R:** no current source-verified entry point was found for this catalog method.

**MATLAB / Octave:** no current source-verified entry point was found for this catalog method.

## Explanations

### Bibliographic source

No canonical paper: cropping is an indexing operation, defined here by Python's half-open slice convention `[start, end)`.

### Mathematical principle

For every row the transform copies columns $j$ satisfying $\mathrm{start}\le j<\mathrm{end}$. It changes feature count without interpolation or wavelength-aware lookup.

### Appropriate uses

Restricting a model to a known spectral region or removing detector edge bands.

### Limits and validation

Indices are positions, not physical wavelengths; grids must already be consistent. Cropping can discard predictive bands and changes downstream feature alignment.

### Implementation

`n4m.transform.resampling.CropTransformer` wraps `n4m_transform_crop_*`; the copy and bounds checks are in `preprocessing/resampling/crop.c`.

The ABI-2 implementation is the `n4m_transform_crop_*` lifecycle in libn4m.

### Sources and provenance

https://github.com/GBeurier/nirs4all-methods/blob/main/cpp/src/core/preprocessing/resampling/crop.c


---

_See also_: [methods index](index.md) · [interactive dashboard](../landing/dashboard.md)