# `pp_range_disc` — Fixed-edge range discretization

_Group_: **Preprocessing** · _C ABI_: `n4m_transform_range_discretizer_*`

## Description

Integer binning against monotonic numeric edges.

## Parameters

| Name | Type | Default |
|------|------|---------|
| `edges` | `Sequence[float] \| None` | `None` |
| `edges_csv` | `str \| None` | `None` |

## API and bindings

**C ABI (ABI 2):** [`n4m_transform_range_discretizer_create`](https://github.com/GBeurier/nirs4all-methods/blob/main/cpp/include/n4m/transform/resampling.h#L71) · [`n4m_transform_range_discretizer_destroy`](https://github.com/GBeurier/nirs4all-methods/blob/main/cpp/include/n4m/transform/resampling.h#L74) · [`n4m_transform_range_discretizer_transform`](https://github.com/GBeurier/nirs4all-methods/blob/main/cpp/include/n4m/transform/resampling.h#L75). Use the linked public header for the exact signature, configuration, and result handles.

**Python (verified public re-export):**

```python
from n4m.transform.resampling import RangeDiscretizer
```

Source signature: [`RangeDiscretizer(edges: Sequence[float] | None = None, edges_csv: str | None = None)`](https://github.com/GBeurier/nirs4all-methods/blob/main/bindings/python/src/n4m/_impl/resampling.py#L218).

**R:** no current source-verified entry point was found for this catalog method.

**MATLAB / Octave:** no current source-verified entry point was found for this catalog method.

## Explanations

### Bibliographic source

No canonical paper: this is scalar quantization against user-supplied ordered edges, equivalent in concept to `numpy.digitize`: https://numpy.org/doc/stable/reference/generated/numpy.digitize.html.

### Mathematical principle

Every value is replaced by the integer index of the interval delimited by the stored monotonic numeric edges. Unlike k-bin discretization, no distribution statistics are learned from $X$.

### Appropriate uses

Applying domain-defined concentration or intensity bands consistently across datasets.

### Limits and validation

Discretization loses continuous information and results depend on edge inclusion semantics. The same edges are applied to all columns and may be unsuitable for wavelengths with different scales.

### Implementation

`n4m.transform.resampling.RangeDiscretizer` wraps `n4m_transform_range_discretizer_*`; edge validation and bin lookup are in `resampling/range_discretizer.c`.

The ABI-2 implementation is the `n4m_transform_range_discretizer_*` lifecycle in libn4m.

### Sources and provenance

https://numpy.org/doc/stable/reference/generated/numpy.digitize.html; https://github.com/GBeurier/nirs4all-methods/blob/main/cpp/src/core/preprocessing/resampling/range_discretizer.c


---

_See also_: [methods index](index.md) · [interactive dashboard](../landing/dashboard.md)