# `pp_weighted_snv` — Weighted standard normal variate

_Group_: **Signal transforms** · _C ABI_: `n4m_transform_weighted_snv_*`

## Description

Weighted standard normal variate normalization.

## Parameters

| Name | Type | Default |
|------|------|---------|
| `weights` | `—` | `None` |
| `ddof` | `int` | `0` |
| `eps` | `float` | `1e-12` |

## API and bindings

**C ABI (ABI 2):** [`n4m_transform_weighted_snv_create`](https://github.com/GBeurier/nirs4all-methods/blob/main/cpp/include/n4m/transform/scatter.h#L160) · [`n4m_transform_weighted_snv_destroy`](https://github.com/GBeurier/nirs4all-methods/blob/main/cpp/include/n4m/transform/scatter.h#L163) · [`n4m_transform_weighted_snv_fit`](https://github.com/GBeurier/nirs4all-methods/blob/main/cpp/include/n4m/transform/scatter.h#L164) · [`n4m_transform_weighted_snv_is_fitted`](https://github.com/GBeurier/nirs4all-methods/blob/main/cpp/include/n4m/transform/scatter.h#L169) · [`n4m_transform_weighted_snv_transform`](https://github.com/GBeurier/nirs4all-methods/blob/main/cpp/include/n4m/transform/scatter.h#L166). Use the linked public header for the exact signature, configuration, and result handles.

**Python (verified public re-export):**

```python
from n4m.transform.scatter import WeightedSNV
```

Source signature: [`WeightedSNV(weights = None, ddof: int = 0, eps: float = 1e-12)`](https://github.com/GBeurier/nirs4all-methods/blob/main/bindings/python/src/n4m/_impl/advanced.py#L215).

**R:** no current source-verified entry point was found for this catalog method.

**MATLAB / Octave:** no current source-verified entry point was found for this catalog method.

## Explanations

### Bibliographic source

No single canonical paper defines this exact weighted estimator. It generalizes the SNV transform of Barnes, Dhanoa & Lister (1989), https://doi.org/10.1366/0003702894202201.

### Mathematical principle

Fit validates and normalizes supplied nonnegative wavelength weights, or uses uniform weights. For every row it computes weighted mean and variance, applies the configured `ddof` correction, floors variance by `eps`, and standardizes all wavelengths with that weighted location and scale.

### Appropriate uses

Reducing scatter while making trusted or diagnostically useful wavelength regions dominate the normalization statistics.

### Limits and validation

Weights require scientific justification and can bias the whole row statistic toward a small band. Negative/zero-total weights are invalid, and normalization can remove absolute amplitude information.

### Implementation

`n4m.transform.scatter.WeightedSNV` uses `n4m_transform_weighted_snv_*`; normalized weights and weighted moments are in `cpp/src/c_api/c_api_advanced.cpp`.

The ABI-2 implementation is the `n4m_transform_weighted_snv_*` lifecycle in libn4m.

### Sources and provenance

https://doi.org/10.1366/0003702894202201; https://github.com/GBeurier/nirs4all-methods/blob/main/cpp/src/c_api/c_api_advanced.cpp#L526-L635


---

_See also_: [methods index](index.md) · [interactive dashboard](../landing/dashboard.md)