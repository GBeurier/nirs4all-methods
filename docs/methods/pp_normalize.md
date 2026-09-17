# `pp_normalize` — Column-wise normalization

_Group_: **Preprocessing** · _C ABI_: `n4m_transform_normalize_*`

## Description

Column-wise normalisation.

## Parameters

| Name | Type | Default |
|------|------|---------|
| `feature_min` | `float` | `-1.0` |
| `feature_max` | `float` | `1.0` |

## API and bindings

**C ABI (ABI 2):** [`n4m_transform_normalize_create`](https://github.com/GBeurier/nirs4all-methods/blob/main/cpp/include/n4m/transform/scaling.h#L15) · [`n4m_transform_normalize_destroy`](https://github.com/GBeurier/nirs4all-methods/blob/main/cpp/include/n4m/transform/scaling.h#L18) · [`n4m_transform_normalize_transform`](https://github.com/GBeurier/nirs4all-methods/blob/main/cpp/include/n4m/transform/scaling.h#L19). Use the linked public header for the exact signature, configuration, and result handles.

**Python (verified public re-export):**

```python
from n4m.transform.scaling import Normalize
```

Source signature: [`Normalize(feature_min: float = -1.0, feature_max: float = 1.0)`](https://github.com/GBeurier/nirs4all-methods/blob/main/bindings/python/src/n4m/_impl/preprocessing.py#L121).

**R:** no current source-verified entry point was found for this catalog method.

**MATLAB / Octave:** no current source-verified entry point was found for this catalog method.

## Explanations

### Bibliographic source

No canonical paper: the default is column L2 normalization; a non-default feature range selects column-wise min–max scaling. The source defines this mode switch.

### Mathematical principle

With the default range `(-1, 1)`, every column is divided by $\sqrt{\sum_i X_{ij}^2}$. If either range endpoint is changed, each column is instead mapped affinely from its observed minimum and maximum to the requested interval.

### Appropriate uses

Equalizing feature magnitudes before algorithms sensitive to Euclidean scale, or mapping every wavelength to a prescribed numeric range.

### Limits and validation

The name hides two different operations. Statistics are computed from the matrix passed to the operation rather than stored as a fitted training state, and zero-norm or constant columns can produce non-finite values.

### Implementation

`n4m.transform.scaling.Normalize` wraps `n4m_transform_normalize_*`; the exact default-mode branch is in `preprocessing/scaling/normalize.c`.

The ABI-2 implementation is the `n4m_transform_normalize_*` lifecycle in libn4m.

### Sources and provenance

https://github.com/GBeurier/nirs4all-methods/blob/main/cpp/src/core/preprocessing/scaling/normalize.c


---

_See also_: [methods index](index.md) · [interactive dashboard](../landing/dashboard.md)