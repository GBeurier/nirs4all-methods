# `pp_simple_scale` — Column-wise min–max scaling to [0, 1]

_Group_: **Preprocessing** · _C ABI_: `n4m_transform_simple_scale_*`

## Description

Column-wise min-max scaling to ``[0, 1]``.

## Parameters

_No constructor parameters._

## API and bindings

**C ABI (ABI 2):** [`n4m_transform_simple_scale_create`](https://github.com/GBeurier/nirs4all-methods/blob/main/cpp/include/n4m/transform/scaling.h#L26) · [`n4m_transform_simple_scale_destroy`](https://github.com/GBeurier/nirs4all-methods/blob/main/cpp/include/n4m/transform/scaling.h#L28) · [`n4m_transform_simple_scale_transform`](https://github.com/GBeurier/nirs4all-methods/blob/main/cpp/include/n4m/transform/scaling.h#L30). Use the linked public header for the exact signature, configuration, and result handles.

**Python (verified public re-export):**

```python
from n4m.transform.scaling import SimpleScale
```

Source signature: [`SimpleScale()`](https://github.com/GBeurier/nirs4all-methods/blob/main/bindings/python/src/n4m/_impl/preprocessing.py#L153).

**R:** no current source-verified entry point was found for this catalog method.

**MATLAB / Octave:** no current source-verified entry point was found for this catalog method.

## Explanations

### Bibliographic source

No canonical paper: this is the affine min–max transform $(x_j-\min x_j)/(\max x_j-\min x_j)$.

### Mathematical principle

For each column, the operation computes its minimum and maximum over the supplied rows and maps the extrema to zero and one, respectively.

### Appropriate uses

Putting wavelength variables on a common bounded numeric scale before distance- or regularization-sensitive procedures.

### Limits and validation

Extrema are outlier-sensitive and, in this stateless operation, are recomputed on each input matrix rather than retained from training. Constant columns require guarded handling and absolute spectral scale is lost.

### Implementation

`n4m.transform.scaling.SimpleScale` uses `n4m_transform_simple_scale_*`; column extrema and affine scaling are in `preprocessing/scaling/simple_scale.c`.

The ABI-2 implementation is the `n4m_transform_simple_scale_*` lifecycle in libn4m.

### Sources and provenance

https://github.com/GBeurier/nirs4all-methods/blob/main/cpp/src/core/preprocessing/scaling/simple_scale.c


---

_See also_: [methods index](index.md) · [interactive dashboard](../landing/dashboard.md)