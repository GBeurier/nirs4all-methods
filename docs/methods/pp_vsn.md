# `pp_vsn` — Variable-sorting normalization (VSN-style weighted SNV)

_Group_: **Signal transforms** · _C ABI_: `n4m_transform_vsn_*`

## Description

VSN-style data-derived weighted SNV.

## Parameters

| Name | Type | Default |
|------|------|---------|
| `eps` | `float` | `1e-12` |

## API and bindings

**C ABI (ABI 2):** [`n4m_transform_vsn_create`](https://github.com/GBeurier/nirs4all-methods/blob/main/cpp/include/n4m/transform/scatter.h#L157) · [`n4m_transform_vsn_destroy`](https://github.com/GBeurier/nirs4all-methods/blob/main/cpp/include/n4m/transform/scatter.h#L158) · [`n4m_transform_vsn_fit`](https://github.com/GBeurier/nirs4all-methods/blob/main/cpp/include/n4m/transform/scatter.h#L159) · [`n4m_transform_vsn_is_fitted`](https://github.com/GBeurier/nirs4all-methods/blob/main/cpp/include/n4m/transform/scatter.h#L164) · [`n4m_transform_vsn_transform`](https://github.com/GBeurier/nirs4all-methods/blob/main/cpp/include/n4m/transform/scatter.h#L161). Use the linked public header for the exact signature, configuration, and result handles.

**Python (verified public re-export):**

```python
from n4m.transform.scatter import VariableSortingNormalization
```

Source signature: [`VariableSortingNormalization(eps: float = 1e-12)`](https://github.com/GBeurier/nirs4all-methods/blob/main/bindings/python/src/n4m/_impl/advanced.py#L255).

**R:** no current source-verified entry point was found for this catalog method.

**MATLAB / Octave:** no current source-verified entry point was found for this catalog method.

## Explanations

### Bibliographic source

Rabatel, Marini, Walczak & Roger (2020), *VSN: Variable sorting for normalization*, Journal of Chemometrics 34, e3164, https://doi.org/10.1002/cem.3164. The shipped correlation-weighted rule is a compact VSN-style implementation, not a claim of full paper parity.

### Mathematical principle

Fit computes each wavelength's absolute correlation across samples with the vector of per-spectrum means, floors correlations by `eps`, and normalizes them to weights. Transform uses those weights for each row's mean and variance before standardization.

### Appropriate uses

Emphasizing wavelengths associated with global row-level scatter while performing SNV normalization.

### Limits and validation

The learned weights can capture chemical or batch structure instead of scatter and must be fit within folds. This implementation should not be assumed equivalent to other algorithms also called VSN.

### Implementation

`n4m.transform.scatter.VariableSortingNormalization` wraps `n4m_transform_vsn_*`; weight learning and weighted standardization are in `cpp/src/c_api/c_api_advanced.cpp`.

The ABI-2 implementation is the `n4m_transform_vsn_*` lifecycle in libn4m.

### Sources and provenance

https://doi.org/10.1002/cem.3164; https://github.com/GBeurier/nirs4all-methods/blob/main/cpp/src/c_api/c_api_advanced.cpp#L560-L635


---

_See also_: [methods index](index.md) · [interactive dashboard](../landing/dashboard.md)