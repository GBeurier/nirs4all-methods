# `pp_area` — Area normalization

_Group_: **Preprocessing** · _C ABI_: `n4m_transform_area_normalization_*`

## Description

Per-row area normalisation.

## Parameters

| Name | Type | Default |
|------|------|---------|
| `method` | `str` | `'sum'` |

## API and bindings

**C ABI (ABI 2):** [`n4m_transform_area_normalization_create`](https://github.com/GBeurier/nirs4all-methods/blob/main/cpp/include/n4m/transform/scatter.h#L53) · [`n4m_transform_area_normalization_destroy`](https://github.com/GBeurier/nirs4all-methods/blob/main/cpp/include/n4m/transform/scatter.h#L55) · [`n4m_transform_area_normalization_transform`](https://github.com/GBeurier/nirs4all-methods/blob/main/cpp/include/n4m/transform/scatter.h#L56). Use the linked public header for the exact signature, configuration, and result handles.

**Python (verified public re-export):**

```python
from n4m.transform.scatter import AreaNormalization
```

Source signature: [`AreaNormalization(method: str = 'sum')`](https://github.com/GBeurier/nirs4all-methods/blob/main/bindings/python/src/n4m/_impl/preprocessing.py#L104).

**R (source-verified):** [`area_normalization_transform(X, method = "sum")`](https://github.com/GBeurier/nirs4all-methods/blob/main/bindings/r/n4m/R/preprocessing.R).

```r
library(n4m)
result <- area_normalization_transform(X)
```

**MATLAB / Octave:** no current source-verified entry point was found for this catalog method.

## Explanations

### Bibliographic source

No single canonical paper: total-area normalization is a direct normalization rule. The exact sum, absolute-sum, and trapezoidal definitions are specified by the implementation source.

### Mathematical principle

Each spectrum $x_i$ is divided by a scalar area $a_i$: the signed sum, the sum of absolute values, or the unit-spacing trapezoidal integral. The implementation uses $a_i=1$ when $|a_i|<10^{-10}$ to avoid division by a near-zero area.

### Appropriate uses

Comparing spectra on a common integrated-intensity scale when total signal varies but relative band shape is meaningful.

### Limits and validation

Signed sums can cancel and all modes couple every wavelength to one scalar. Normalization removes absolute concentration or path-length information.

### Implementation

`n4m.transform.scatter.AreaNormalization` uses `n4m_transform_area_normalization_*`; `area_normalization.c` implements all three row-wise denominators.

The ABI-2 implementation is the `n4m_transform_area_normalization_*` lifecycle in libn4m.

### Sources and provenance

https://github.com/GBeurier/nirs4all-methods/blob/main/cpp/src/core/preprocessing/scatter/area_normalization.c


---

_See also_: [methods index](index.md) · [interactive dashboard](../landing/dashboard.md)