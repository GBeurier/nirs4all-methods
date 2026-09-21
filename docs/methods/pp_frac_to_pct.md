# `pp_frac_to_pct` — Fraction-to-percent signal conversion

_Group_: **Preprocessing** · _C ABI_: `n4m_transform_fraction_to_percent_*`

## Description

Convert fractional reflectance/transmittance to percent.

## Parameters

_No constructor parameters._

## API and bindings

**C ABI (ABI 2):** [`n4m_transform_fraction_to_percent_create`](https://github.com/GBeurier/nirs4all-methods/blob/main/cpp/include/n4m/transform/signal_conversion.h#L44) · [`n4m_transform_fraction_to_percent_destroy`](https://github.com/GBeurier/nirs4all-methods/blob/main/cpp/include/n4m/transform/signal_conversion.h#L46) · [`n4m_transform_fraction_to_percent_transform`](https://github.com/GBeurier/nirs4all-methods/blob/main/cpp/include/n4m/transform/signal_conversion.h#L48). Use the linked public header for the exact signature, configuration, and result handles.

**Python (verified public re-export):**

```python
from n4m.transform.signal_conversion import FractionToPercent
```

Source signature: [`FractionToPercent()`](https://github.com/GBeurier/nirs4all-methods/blob/main/bindings/python/src/n4m/_impl/preprocessing.py#L502).

**R:** no current source-verified entry point was found for this catalog method.

**MATLAB / Octave:** no current source-verified entry point was found for this catalog method.

## Explanations

### Bibliographic source

No canonical paper: percent is exactly the conventional unit conversion $x_{\%}=100x$.

### Mathematical principle

Every matrix element is multiplied by 100; no statistics are learned and spectral shape is unchanged.

### Appropriate uses

Converting fractional reflectance or transmittance to percent units for interfaces or models that explicitly require that scale.

### Limits and validation

This operation does not infer the signal type and will silently produce wrong units if the input is already a percent or absorbance value.

### Implementation

`n4m.transform.signal_conversion.FractionToPercent` uses `n4m_transform_fraction_to_percent_*`; the element-wise kernel is in `signal_conversion/fraction_to_percent.c`.

The ABI-2 implementation is the `n4m_transform_fraction_to_percent_*` lifecycle in libn4m.

### Sources and provenance

https://github.com/GBeurier/nirs4all-methods/blob/main/cpp/src/core/preprocessing/signal_conversion/fraction_to_percent.c


---

_See also_: [methods index](index.md) · [interactive dashboard](../landing/dashboard.md)