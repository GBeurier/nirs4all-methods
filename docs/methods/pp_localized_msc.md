# `pp_localized_msc` — Localized moving-window multiplicative scatter correction

_Group_: **Signal transforms** · _C ABI_: `n4m_transform_localized_msc_*`

## Description

Feature-wise MSC using a moving local wavelength window.

## Parameters

| Name | Type | Default |
|------|------|---------|
| `window_size` | `int` | `32` |
| `reference` | `—` | `None` |
| `eps` | `float` | `1e-12` |

## API and bindings

**C ABI (ABI 2):** [`n4m_transform_localized_msc_create`](https://github.com/GBeurier/nirs4all-methods/blob/main/cpp/include/n4m/transform/scatter.h#L194) · [`n4m_transform_localized_msc_destroy`](https://github.com/GBeurier/nirs4all-methods/blob/main/cpp/include/n4m/transform/scatter.h#L197) · [`n4m_transform_localized_msc_fit`](https://github.com/GBeurier/nirs4all-methods/blob/main/cpp/include/n4m/transform/scatter.h#L199) · [`n4m_transform_localized_msc_is_fitted`](https://github.com/GBeurier/nirs4all-methods/blob/main/cpp/include/n4m/transform/scatter.h#L204) · [`n4m_transform_localized_msc_transform`](https://github.com/GBeurier/nirs4all-methods/blob/main/cpp/include/n4m/transform/scatter.h#L201). Use the linked public header for the exact signature, configuration, and result handles.

**Python (verified public re-export):**

```python
from n4m.transform.scatter import LocalizedMSC
```

Source signature: [`LocalizedMSC()`](https://github.com/GBeurier/nirs4all-methods/blob/main/bindings/python/src/n4m/_impl/advanced.py#L352).

**R:** no current source-verified entry point was found for this catalog method.

**MATLAB / Octave:** no current source-verified entry point was found for this catalog method.

## Explanations

### Bibliographic source

No single paper canonically defines this implementation. It is a moving-window extension of Geladi, MacDougall & Martens (1985), https://doi.org/10.1366/0003702854248656.

### Mathematical principle

At each wavelength, a local window is regressed on the corresponding reference window to obtain an intercept and slope; the center value is corrected by those local coefficients. The fitted reference is either supplied or learned from the training mean.

### Appropriate uses

Correcting scatter whose offset or gain changes gradually across the wavelength axis.

### Limits and validation

Local regressions become unstable in flat windows or with small windows; `eps` only guards degeneracy. Moving coefficients can distort broad bands and require a stable reference.

### Implementation

`n4m.transform.scatter.LocalizedMSC` wraps `n4m_transform_localized_msc_*`; the local regressions are in `cpp/src/c_api/c_api_advanced.cpp`.

The ABI-2 implementation is the `n4m_transform_localized_msc_*` lifecycle in libn4m.

### Sources and provenance

https://doi.org/10.1366/0003702854248656; https://github.com/GBeurier/nirs4all-methods/blob/main/cpp/src/c_api/c_api_advanced.cpp


---

_See also_: [methods index](index.md) · [interactive dashboard](../landing/dashboard.md)