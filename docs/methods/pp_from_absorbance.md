# `pp_from_absorbance` — Absorbance-to-reflectance/transmittance conversion

_Group_: **Preprocessing** · _C ABI_: `n4m_transform_from_absorbance_*`

## Description

R = 10**(-A), optionally returned as percent.

## Parameters

| Name | Type | Default |
|------|------|---------|
| `is_percent` | `bool` | `False` |

## API and bindings

**C ABI (ABI 2):** [`n4m_transform_from_absorbance_create`](https://github.com/GBeurier/nirs4all-methods/blob/main/cpp/include/n4m/transform/signal_conversion.h#L24) · [`n4m_transform_from_absorbance_destroy`](https://github.com/GBeurier/nirs4all-methods/blob/main/cpp/include/n4m/transform/signal_conversion.h#L26) · [`n4m_transform_from_absorbance_transform`](https://github.com/GBeurier/nirs4all-methods/blob/main/cpp/include/n4m/transform/signal_conversion.h#L28). Use the linked public header for the exact signature, configuration, and result handles.

**Python (verified public re-export):**

```python
from n4m.transform.signal_conversion import FromAbsorbance
```

Source signature: [`FromAbsorbance(is_percent: bool = False)`](https://github.com/GBeurier/nirs4all-methods/blob/main/bindings/python/src/n4m/_impl/preprocessing.py#L577).

**R:** no current source-verified entry point was found for this catalog method.

**MATLAB / Octave:** no current source-verified entry point was found for this catalog method.

## Explanations

### Bibliographic source

Beer (1852), *Bestimmung der Absorption des rothen Lichts in farbigen Flüssigkeiten*, Annalen der Physik 162, 78–88, https://doi.org/10.1002/andp.18521620505.

### Mathematical principle

The inverse base-10 absorbance relation is applied element-wise: $R=10^{-A}$ (or $T=10^{-A}$). With `is_percent`, the result is additionally multiplied by 100.

### Appropriate uses

Returning log-transformed spectra to fractional or percent reflectance/transmittance units.

### Limits and validation

The formula cannot distinguish reflectance from transmittance and assumes base-10 absorbance. Large negative or positive values can overflow or underflow.

### Implementation

`n4m.transform.signal_conversion.FromAbsorbance` wraps `n4m_transform_from_absorbance_*`; the power transform is in `signal_conversion/from_absorbance.c`.

The ABI-2 implementation is the `n4m_transform_from_absorbance_*` lifecycle in libn4m.

### Sources and provenance

https://doi.org/10.1002/andp.18521620505; https://github.com/GBeurier/nirs4all-methods/blob/main/cpp/src/core/preprocessing/signal_conversion/from_absorbance.c


---

_See also_: [methods index](index.md) · [interactive dashboard](../landing/dashboard.md)