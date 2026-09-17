# `interval_generator` — Fixed and overlapping interval expansion

_Group_: **Signal transforms** · _C ABI_: `n4m_feature_selection_interval_generator_*`

## Description

Generate fixed or overlapping wavelength intervals.

## Parameters

| Name | Type | Default |
|------|------|---------|
| `interval_size` | `int` | `32` |
| `step` | `int \| None` | `None` |

## API and bindings

**C ABI (ABI 2):** [`n4m_feature_selection_interval_generator_create`](https://github.com/GBeurier/nirs4all-methods/blob/main/cpp/include/n4m/feature_selection.h#L42) · [`n4m_feature_selection_interval_generator_destroy`](https://github.com/GBeurier/nirs4all-methods/blob/main/cpp/include/n4m/feature_selection.h#L44) · [`n4m_feature_selection_interval_generator_fit`](https://github.com/GBeurier/nirs4all-methods/blob/main/cpp/include/n4m/feature_selection.h#L46) · [`n4m_feature_selection_interval_generator_is_fitted`](https://github.com/GBeurier/nirs4all-methods/blob/main/cpp/include/n4m/feature_selection.h#L53) · [`n4m_feature_selection_interval_generator_output_cols`](https://github.com/GBeurier/nirs4all-methods/blob/main/cpp/include/n4m/feature_selection.h#L51) · [`n4m_feature_selection_interval_generator_transform`](https://github.com/GBeurier/nirs4all-methods/blob/main/cpp/include/n4m/feature_selection.h#L48) · [`n4m_feature_selection_interval_select`](https://github.com/GBeurier/nirs4all-methods/blob/main/cpp/include/n4m/feature_selection.h#L67). Use the linked public header for the exact signature, configuration, and result handles.

**Python (verified public re-export):**

```python
from n4m.feature_selection.interval import IntervalGenerator
```

Source signature: [`IntervalGenerator(interval_size: int = 32, step: int | None = None)`](https://github.com/GBeurier/nirs4all-methods/blob/main/bindings/python/src/n4m/_impl/advanced.py#L522).

**R (source-verified):** [`interval_select(X, Y, n_components, interval_width = 10L, step = 1L)`](https://github.com/GBeurier/nirs4all-methods/blob/main/bindings/r/n4m/R/methods_extra.R).

The source signature has additional required inputs, so no example call is fabricated.

**MATLAB / Octave (source-verified):** [`interval_select(X, Y, n_components, interval_width, step)`](https://github.com/GBeurier/nirs4all-methods/blob/main/bindings/matlab/+n4m/interval_select.m).

The source signature has additional required inputs, so no example call is fabricated.

## Explanations

### Bibliographic source

No canonical paper: this is a deterministic interval-construction and column-copy operation; `interval_fit` is its normative definition.

### Mathematical principle

Fit creates half-open bands of width `interval_size` starting every `step` columns (`step=interval_size` by default), truncating the last band at the feature count. Transform concatenates every band, so overlapping columns are deliberately repeated.

### Appropriate uses

Building interval blocks for downstream interval selection or block-wise models, with optional overlap.

### Limits and validation

It does not score or select intervals. Overlap increases output dimension and repeats features, while position indices assume an already aligned common wavelength grid.

### Implementation

`n4m.feature_selection.interval.IntervalGenerator` and `interval_generator` use `n4m_feature_selection_interval_generator_*`; construction/copying are in `cpp/src/c_api/c_api_advanced.cpp`.

The ABI-2 implementation is the `n4m_feature_selection_interval_generator_*` lifecycle in libn4m.

### Sources and provenance

https://github.com/GBeurier/nirs4all-methods/blob/main/cpp/src/c_api/c_api_advanced.cpp#L1225-L1303


---

_See also_: [methods index](index.md) · [interactive dashboard](../landing/dashboard.md)