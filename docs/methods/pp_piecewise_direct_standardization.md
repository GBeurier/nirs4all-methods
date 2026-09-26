# `pp_piecewise_direct_standardization` — Piecewise direct standardization (PDS)

_Group_: **Signal transforms** · _C ABI_: `n4m_domain_adaptation_piecewise_direct_standardization_*`

## Description

PDS: local regressions mapping source windows to target wavelengths.

## Parameters

| Name | Type | Default |
|------|------|---------|
| `window_size` | `int` | `5` |
| `fit_intercept` | `bool` | `True` |
| `ridge` | `float` | `0.0` |

## API and bindings

**C ABI (ABI 2):** [`n4m_domain_adaptation_piecewise_direct_standardization_create`](https://github.com/GBeurier/nirs4all-methods/blob/main/cpp/include/n4m/domain_adaptation.h#L95) · [`n4m_domain_adaptation_piecewise_direct_standardization_destroy`](https://github.com/GBeurier/nirs4all-methods/blob/main/cpp/include/n4m/domain_adaptation.h#L98) · [`n4m_domain_adaptation_piecewise_direct_standardization_fit`](https://github.com/GBeurier/nirs4all-methods/blob/main/cpp/include/n4m/domain_adaptation.h#L100) · [`n4m_domain_adaptation_piecewise_direct_standardization_is_fitted`](https://github.com/GBeurier/nirs4all-methods/blob/main/cpp/include/n4m/domain_adaptation.h#L106) · [`n4m_domain_adaptation_piecewise_direct_standardization_transform`](https://github.com/GBeurier/nirs4all-methods/blob/main/cpp/include/n4m/domain_adaptation.h#L103). Use the linked public header for the exact signature, configuration, and result handles.

**Python (verified public re-export):**

```python
from n4m.domain_adaptation.standardization import piecewise_direct_standardization
```

Source signature: [`piecewise_direct_standardization(X_source, X_target, X = None, window_size: int = 5, fit_intercept: bool = True, ridge: float = 0.0)`](https://github.com/GBeurier/nirs4all-methods/blob/main/bindings/python/src/n4m/_impl/native.py#L9764).

**R:** no current source-verified entry point was found for this catalog method.

**MATLAB / Octave:** no current source-verified entry point was found for this catalog method.

## Explanations

### Bibliographic source

Bouveresse & Massart (1996), *Improvement of the piecewise direct standardization procedure for the transfer of NIR spectra for multivariate calibration*, Chemometrics and Intelligent Laboratory Systems 32, 201–213, https://doi.org/10.1016/0169-7439(95)00074-7.

### Mathematical principle

For target wavelength $j$, PDS fits a local affine regression from a source window around $j$ to the paired target value at $j$. The fitted local coefficients form a banded transfer map; optional ridge regularization stabilizes each window solve.

### Appropriate uses

Instrument transfer when wavelength-local response differences make one dense global DS map unnecessarily flexible.

### Limits and validation

Requires aligned paired standards. Window width trades locality against conditioning; edge windows differ in size and local linear maps cannot correct nonlinear detector effects.

### Implementation

`n4m.domain_adaptation.standardization.PiecewiseDirectStandardization` uses `n4m_domain_adaptation_piecewise_direct_standardization_*`; window regressions are implemented in `cpp/src/c_api/c_api_advanced.cpp`.

The ABI-2 implementation is the `n4m_domain_adaptation_piecewise_direct_standardization_*` lifecycle in libn4m.

### Sources and provenance

https://doi.org/10.1016/0169-7439(95)00074-7; https://github.com/GBeurier/nirs4all-methods/blob/main/cpp/src/c_api/c_api_advanced.cpp


---

_See also_: [methods index](index.md) · [interactive dashboard](../landing/dashboard.md)