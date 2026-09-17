# `pp_kbins_disc` — Per-feature integer k-bin discretization

_Group_: **Preprocessing** · _C ABI_: `n4m_transform_kbins_discretizer_*`

## Description

Per-column integer binning using uniform or quantile edges.

## Parameters

| Name | Type | Default |
|------|------|---------|
| `n_bins` | `int` | `5` |
| `strategy` | `str \| int` | `0` |

## API and bindings

**C ABI (ABI 2):** [`n4m_transform_kbins_discretizer_create`](https://github.com/GBeurier/nirs4all-methods/blob/main/cpp/include/n4m/transform/resampling.h#L57) · [`n4m_transform_kbins_discretizer_destroy`](https://github.com/GBeurier/nirs4all-methods/blob/main/cpp/include/n4m/transform/resampling.h#L60) · [`n4m_transform_kbins_discretizer_fit`](https://github.com/GBeurier/nirs4all-methods/blob/main/cpp/include/n4m/transform/resampling.h#L61) · [`n4m_transform_kbins_discretizer_is_fitted`](https://github.com/GBeurier/nirs4all-methods/blob/main/cpp/include/n4m/transform/resampling.h#L63) · [`n4m_transform_kbins_discretizer_transform`](https://github.com/GBeurier/nirs4all-methods/blob/main/cpp/include/n4m/transform/resampling.h#L65). Use the linked public header for the exact signature, configuration, and result handles.

**Python (verified public re-export):**

```python
from n4m.transform.resampling import IntegerKBinsDiscretizer
```

Source signature: [`IntegerKBinsDiscretizer(n_bins: int = 5, strategy: str | int = 0)`](https://github.com/GBeurier/nirs4all-methods/blob/main/bindings/python/src/n4m/_impl/resampling.py#L182).

**R:** no current source-verified entry point was found for this catalog method.

**MATLAB / Octave:** no current source-verified entry point was found for this catalog method.

## Explanations

### Bibliographic source

No single canonical paper: this is uniform or empirical-quantile scalar quantization. Scikit-learn documents the corresponding estimator at https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.KBinsDiscretizer.html.

### Mathematical principle

Fit computes `n_bins + 1` edges independently for every column, either equally spaced between its extrema or at empirical quantiles. Transform replaces each value by the integer bin index selected by those stored edges.

### Appropriate uses

Robust coarse encoding for threshold models or exploratory analyses where continuous amplitude resolution is unnecessary.

### Limits and validation

Quantization discards within-bin information; quantile edges can collapse with ties and all edges are training-data dependent. It is rarely appropriate before models that exploit smooth spectral geometry.

### Implementation

`n4m.transform.resampling.IntegerKBinsDiscretizer` calls `n4m_transform_kbins_discretizer_*`; fitting and integer encoding are in `resampling/kbins_discretizer.c`.

The ABI-2 implementation is the `n4m_transform_kbins_discretizer_*` lifecycle in libn4m.

### Sources and provenance

https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.KBinsDiscretizer.html; https://github.com/GBeurier/nirs4all-methods/blob/main/cpp/src/core/preprocessing/resampling/kbins_discretizer.c


---

_See also_: [methods index](index.md) · [interactive dashboard](../landing/dashboard.md)