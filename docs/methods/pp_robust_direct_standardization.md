# `pp_robust_direct_standardization` — Trimmed robust direct standardization

_Group_: **Signal transforms** · _C ABI_: `n4m_domain_adaptation_robust_direct_standardization_*`

## Description

Direct standardization with iterative residual trimming.

## Parameters

| Name | Type | Default |
|------|------|---------|
| `fit_intercept` | `bool` | `True` |
| `ridge` | `float` | `0.0` |
| `trim_quantile` | `float` | `0.9` |
| `max_iter` | `int` | `3` |

## API and bindings

**C ABI (ABI 2):** [`n4m_domain_adaptation_robust_direct_standardization_create`](https://github.com/GBeurier/nirs4all-methods/blob/main/cpp/include/n4m/domain_adaptation.h#L79) · [`n4m_domain_adaptation_robust_direct_standardization_destroy`](https://github.com/GBeurier/nirs4all-methods/blob/main/cpp/include/n4m/domain_adaptation.h#L82) · [`n4m_domain_adaptation_robust_direct_standardization_fit`](https://github.com/GBeurier/nirs4all-methods/blob/main/cpp/include/n4m/domain_adaptation.h#L84) · [`n4m_domain_adaptation_robust_direct_standardization_is_fitted`](https://github.com/GBeurier/nirs4all-methods/blob/main/cpp/include/n4m/domain_adaptation.h#L90) · [`n4m_domain_adaptation_robust_direct_standardization_transform`](https://github.com/GBeurier/nirs4all-methods/blob/main/cpp/include/n4m/domain_adaptation.h#L87). Use the linked public header for the exact signature, configuration, and result handles.

**Python (verified public re-export):**

```python
from n4m.domain_adaptation.standardization import robust_direct_standardization
```

Source signature: [`robust_direct_standardization(X_source, X_target, X = None, fit_intercept: bool = True, ridge: float = 0.0, trim_quantile: float = 0.9, max_iter: int = 3)`](https://github.com/GBeurier/nirs4all-methods/blob/main/bindings/python/src/n4m/_impl/native.py#L9779).

**R:** no current source-verified entry point was found for this catalog method.

**MATLAB / Octave:** no current source-verified entry point was found for this catalog method.

## Explanations

### Bibliographic source

No single canonical paper defines this implementation. It robustifies direct standardization from Wang et al. (1991), https://doi.org/10.1021/ac00023a016, by iterative residual trimming.

### Mathematical principle

A global affine DS map is fit on paired spectra. After each fit, row residual norms are computed and only rows at or below `trim_quantile` are retained for the next fit, up to `max_iter`; optional ridge regularization remains active.

### Appropriate uses

Instrument transfer with a small proportion of mismatched or corrupted paired standards.

### Limits and validation

Hard trimming can discard legitimate domain extremes and the method is not a formal high-breakdown estimator. It still assumes one global linear map and enough retained pairs to identify it.

### Implementation

`n4m.domain_adaptation.standardization.RobustDirectStandardization` calls `n4m_domain_adaptation_robust_direct_standardization_*`; trimming is implemented in `cpp/src/c_api/c_api_advanced.cpp`.

The ABI-2 implementation is the `n4m_domain_adaptation_robust_direct_standardization_*` lifecycle in libn4m.

### Sources and provenance

https://doi.org/10.1021/ac00023a016; https://github.com/GBeurier/nirs4all-methods/blob/main/cpp/src/c_api/c_api_advanced.cpp


---

_See also_: [methods index](index.md) · [interactive dashboard](../landing/dashboard.md)