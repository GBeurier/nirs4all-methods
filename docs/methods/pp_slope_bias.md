# `pp_slope_bias` — Slope-and-bias prediction correction

_Group_: **Signal transforms** · _C ABI_: `n4m_domain_adaptation_slope_bias_*`

## Description

Linear slope/bias correction for transferred predictions.

## Parameters

_No constructor parameters._

## API and bindings

**C ABI (ABI 2):** [`n4m_domain_adaptation_slope_bias_create`](https://github.com/GBeurier/nirs4all-methods/blob/main/cpp/include/n4m/domain_adaptation.h#L111) · [`n4m_domain_adaptation_slope_bias_destroy`](https://github.com/GBeurier/nirs4all-methods/blob/main/cpp/include/n4m/domain_adaptation.h#L113) · [`n4m_domain_adaptation_slope_bias_fit`](https://github.com/GBeurier/nirs4all-methods/blob/main/cpp/include/n4m/domain_adaptation.h#L114) · [`n4m_domain_adaptation_slope_bias_is_fitted`](https://github.com/GBeurier/nirs4all-methods/blob/main/cpp/include/n4m/domain_adaptation.h#L120) · [`n4m_domain_adaptation_slope_bias_transform`](https://github.com/GBeurier/nirs4all-methods/blob/main/cpp/include/n4m/domain_adaptation.h#L117). Use the linked public header for the exact signature, configuration, and result handles.

**Python (verified public re-export):**

```python
from n4m.domain_adaptation.standardization import SlopeBiasCorrection
```

Source signature: [`SlopeBiasCorrection()`](https://github.com/GBeurier/nirs4all-methods/blob/main/bindings/python/src/n4m/_impl/advanced.py#L160).

**R:** no current source-verified entry point was found for this catalog method.

**MATLAB / Octave:** no current source-verified entry point was found for this catalog method.

## Explanations

### Bibliographic source

No single canonical paper defines this two-parameter post-calibration operation. It is ordinary least-squares bias/slope correction, with source code as the exact specification.

### Mathematical principle

Given paired source predictions $x$ and target values $y$, fit estimates $y=ax+b$ by closed-form OLS. Transform returns $ax+b$ for new source predictions; it acts on a one-dimensional prediction vector, not on spectra.

### Appropriate uses

Correcting a stable linear bias and gain error after moving a calibration between instruments or populations.

### Limits and validation

A constant source prediction makes the slope unidentified. The correction cannot fix nonlinear, heteroscedastic, or wavelength-specific transfer errors and needs paired target values.

### Implementation

`n4m.domain_adaptation.standardization.SlopeBiasCorrection` wraps `n4m_domain_adaptation_slope_bias_*`; the closed-form fit is in `cpp/src/c_api/c_api_advanced.cpp`.

The ABI-2 implementation is the `n4m_domain_adaptation_slope_bias_*` lifecycle in libn4m.

### Sources and provenance

https://github.com/GBeurier/nirs4all-methods/blob/main/cpp/src/c_api/c_api_advanced.cpp#L499-L523


---

_See also_: [methods index](index.md) · [interactive dashboard](../landing/dashboard.md)