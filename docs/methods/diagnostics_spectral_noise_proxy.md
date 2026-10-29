# `spectral_noise_proxy` — n4m.diagnostics.spectral_noise_proxy

_Namespace_: **`n4m.diagnostics`** · _Fully-qualified_: `n4m.diagnostics.spectral_noise_proxy` · _Catalog id_: `diagnostics.spectral_noise_proxy`

## API surface

**C ABI (ABI 2):** [`n4m_procedure_run`](https://github.com/GBeurier/nirs4all-methods/blob/main/cpp/include/n4m/estimator.h#L8). Use the linked public header for the exact signature, configuration, and result handles.

**Python (verified public re-export):** `from n4m.roles import SpectralNoiseProxy`

**Signature:** [`SpectralNoiseProxy(*, valid_mask = ())`](https://github.com/GBeurier/nirs4all-methods/blob/main/bindings/python/src/n4m/roles/_generated.py#L2166)

**R (source-verified):** [`n4m_spectral_noise_proxy(valid_mask = c())`](https://github.com/GBeurier/nirs4all-methods/blob/main/bindings/r/n4m/R/estimator_roles_generated.R).

```r
library(n4m)
result <- n4m_spectral_noise_proxy()
```

**MATLAB / Octave:** no current source-verified entry point was found for this catalog method.

### Parameters

| Name | Type | Default |
|---|---|---|
| `valid_mask` | `—` | `()` |

## Explanations

### Bibliographic source

A declared noise proxy under an approximately smooth signal, using normal-scaled MAD. SciPy documents the median absolute deviation definition used by the independent oracle: https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.median_abs_deviation.html.

### Mathematical principle

For each row, median absolute deviation of index-space second differences is multiplied by 1.4826/sqrt(6). The sqrt(6) removes variance inflation for independent equal-variance white noise under differencing twice. Native summaries expose mean absolute amplitude, amplitude/noise and 20log10 of that positive finite ratio.

### Appropriate uses

Describing high-frequency variation on an explicitly declared index-grid protocol and comparing that same protocol on identical populations and masks.

### Limits and validation

Smooth signal structure, correlated/heteroscedastic noise and nonuniform physical sampling can invalidate an instrumental interpretation. This is not instrument SNR and is distinct from the existing SG residual/noise-gain estimator. Zero estimated noise does not create an infinite or arbitrarily large finite ratio.

### Implementation

One mask-aware native procedure with per-row sparse availability and native global median/noise ratio. Independent NumPy reductions validate the normalization and dB; no MAD or percentile is recomputed by a binding.

### Sources and provenance

https://github.com/GBeurier/nirs4all-methods/blob/main/cpp/src/core/estimator/intrinsic_diagnostics.cpp; https://github.com/GBeurier/nirs4all-methods/blob/main/bindings/python/tests/test_intrinsic_diagnostics.py; https://github.com/GBeurier/nirs4all-methods/blob/main/docs/intrinsic_diagnostics_contract.v1.json

## Catalog note

Versioned descriptive protocol with native aggregates and explicit per-row availability; see docs/intrinsic_diagnostics_contract.v1.json.


_See also_: [methods index](index.md).