# `pp_direct_standardization` — Direct standardization (DS)

_Group_: **Signal transforms** · _C ABI_: `n4m_domain_adaptation_direct_standardization_*`

## Description

Direct standardization transfer map between paired instruments.

## Parameters

| Name | Type | Default |
|------|------|---------|
| `fit_intercept` | `bool` | `True` |
| `ridge` | `float` | `0.0` |

## API and bindings

**C ABI (ABI 2):** [`n4m_domain_adaptation_direct_standardization_create`](https://github.com/GBeurier/nirs4all-methods/blob/main/cpp/include/n4m/domain_adaptation.h#L64) · [`n4m_domain_adaptation_direct_standardization_destroy`](https://github.com/GBeurier/nirs4all-methods/blob/main/cpp/include/n4m/domain_adaptation.h#L66) · [`n4m_domain_adaptation_direct_standardization_fit`](https://github.com/GBeurier/nirs4all-methods/blob/main/cpp/include/n4m/domain_adaptation.h#L68) · [`n4m_domain_adaptation_direct_standardization_is_fitted`](https://github.com/GBeurier/nirs4all-methods/blob/main/cpp/include/n4m/domain_adaptation.h#L74) · [`n4m_domain_adaptation_direct_standardization_transform`](https://github.com/GBeurier/nirs4all-methods/blob/main/cpp/include/n4m/domain_adaptation.h#L71). Use the linked public header for the exact signature, configuration, and result handles.

**Python (verified public re-export):**

```python
from n4m.domain_adaptation.standardization import direct_standardization
```

Source signature: [`direct_standardization(X_source, X_target, X = None, fit_intercept: bool = True, ridge: float = 0.0)`](https://github.com/GBeurier/nirs4all-methods/blob/main/bindings/python/src/n4m/_impl/native.py#L9651).

**R:** no current source-verified entry point was found for this catalog method.

**MATLAB / Octave:** no current source-verified entry point was found for this catalog method.

## Explanations

### Bibliographic source

Wang et al. (1991), *Improvement of multivariate calibration through instrument standardization*, Analytical Chemistry 63, 2750–2756, https://doi.org/10.1021/ac00023a016.

### Mathematical principle

Using paired source and target spectra, DS fits a global affine map $X_s B+\mathbf{1}b\approx X_t$ by least squares, optionally with ridge regularization. New source spectra are multiplied by the learned transfer matrix.

### Appropriate uses

Transferring a calibration between instruments or measurement conditions when representative paired standards are available.

### Limits and validation

It requires paired samples and assumes one global linear relation. With many wavelengths the map can overfit unless regularized; extrapolation beyond transfer standards is unsafe.

### Implementation

`n4m.domain_adaptation.standardization.DirectStandardization` wraps `n4m_domain_adaptation_direct_standardization_*`; fitting is in `cpp/src/c_api/c_api_advanced.cpp`.

The ABI-2 implementation is the `n4m_domain_adaptation_direct_standardization_*` lifecycle in libn4m.

### Sources and provenance

https://doi.org/10.1021/ac00023a016; https://github.com/GBeurier/nirs4all-methods/blob/main/cpp/src/c_api/c_api_advanced.cpp


---

_See also_: [methods index](index.md) · [interactive dashboard](../landing/dashboard.md)