# `filter_composite` — Boolean composition of sample keep masks

_Group_: **Sample / feature filters** · _C ABI_: `n4m_outlier_detection_composite_*`

## Description

Boolean composition of leverage and quality filters.

## Parameters

| Name | Type | Default |
|------|------|---------|
| `mode` | `str \| int` | `'any'` |
| `filters` | `—` | `()` |

## API and bindings

**C ABI (ABI 2):** [`n4m_outlier_detection_composite_add_leverage`](https://github.com/GBeurier/nirs4all-methods/blob/main/cpp/include/n4m/outlier_detection.h#L107) · [`n4m_outlier_detection_composite_add_quality`](https://github.com/GBeurier/nirs4all-methods/blob/main/cpp/include/n4m/outlier_detection.h#L110) · [`n4m_outlier_detection_composite_apply`](https://github.com/GBeurier/nirs4all-methods/blob/main/cpp/include/n4m/outlier_detection.h#L113) · [`n4m_outlier_detection_composite_create`](https://github.com/GBeurier/nirs4all-methods/blob/main/cpp/include/n4m/outlier_detection.h#L103) · [`n4m_outlier_detection_composite_destroy`](https://github.com/GBeurier/nirs4all-methods/blob/main/cpp/include/n4m/outlier_detection.h#L105). Use the linked public header for the exact signature, configuration, and result handles.

**Python (verified public re-export):**

```python
from n4m.outlier_detection import CompositeFilter
```

Source signature: [`CompositeFilter(mode: str | int = 'any', filters = ())`](https://github.com/GBeurier/nirs4all-methods/blob/main/bindings/python/src/n4m/_impl/filters.py#L388).

**R:** no current source-verified entry point was found for this catalog method.

**MATLAB / Octave:** no current source-verified entry point was found for this catalog method.

## Explanations

### Bibliographic source

No paper canonically defines this composition wrapper. Boolean aggregation of precomputed masks is an implementation utility, not a new outlier-detection algorithm.

### Mathematical principle

Each child filter computes a binary keep mask. Mode `all` keeps row i iff every child keeps it ($m_i=\bigwedge_km_{ki}$); mode `any` keeps it iff at least one child does ($m_i=\bigvee_km_{ki}$). An empty child list keeps every row.

### Appropriate uses

Combining independent leverage and spectral-quality policies into one reproducible decision.

### Limits and validation

The current Python wrapper accepts only `HighLeverageFilter` and `SpectralQualityFilter` children. “any” is permissive for keep masks, whereas “all” is restrictive; this is easy to invert mentally.

### Implementation

Python role API `n4m.outlier_detection.CompositeFilter`; ABI 2 family `n4m_outlier_detection_composite_{create,add_leverage,add_quality,apply,destroy}`.

The ABI-2 implementation is the `n4m_outlier_detection_composite_*` lifecycle in libn4m.

### Sources and provenance

https://github.com/GBeurier/nirs4all-methods/blob/main/cpp/src/core/filters/composite.h


---

_See also_: [methods index](index.md) · [interactive dashboard](../landing/dashboard.md)