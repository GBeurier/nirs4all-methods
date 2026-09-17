# `pp_pct_to_frac` — Percent-to-fraction signal conversion

_Group_: **Preprocessing** · _C ABI_: `n4m_transform_percent_to_fraction_*`

## Description

Convert percent reflectance/transmittance to fraction.

## Parameters

_No constructor parameters._

## API and bindings

**C ABI (ABI 2):** [`n4m_transform_percent_to_fraction_create`](https://github.com/GBeurier/nirs4all-methods/blob/main/cpp/include/n4m/transform/signal_conversion.h#L34) · [`n4m_transform_percent_to_fraction_destroy`](https://github.com/GBeurier/nirs4all-methods/blob/main/cpp/include/n4m/transform/signal_conversion.h#L36) · [`n4m_transform_percent_to_fraction_transform`](https://github.com/GBeurier/nirs4all-methods/blob/main/cpp/include/n4m/transform/signal_conversion.h#L38). Use the linked public header for the exact signature, configuration, and result handles.

**Python (verified public re-export):**

```python
from n4m.transform.signal_conversion import PercentToFraction
```

Source signature: [`PercentToFraction()`](https://github.com/GBeurier/nirs4all-methods/blob/main/bindings/python/src/n4m/_impl/preprocessing.py#L485).

**R:** no current source-verified entry point was found for this catalog method.

**MATLAB / Octave:** no current source-verified entry point was found for this catalog method.

## Explanations

### Bibliographic source

No canonical paper: this is exactly the unit conversion $x=x_{\%}/100$.

### Mathematical principle

Every element is divided by 100 without estimating statistics or changing spectral shape.

### Appropriate uses

Converting percent reflectance or transmittance to the fractional scale required by absorbance and Kubelka–Munk transforms.

### Limits and validation

The operator does not detect current units; applying it to fractional or absorbance data silently creates an invalid scale.

### Implementation

`n4m.transform.signal_conversion.PercentToFraction` calls `n4m_transform_percent_to_fraction_*`; the kernel is `signal_conversion/percent_to_fraction.c`.

The ABI-2 implementation is the `n4m_transform_percent_to_fraction_*` lifecycle in libn4m.

### Sources and provenance

https://github.com/GBeurier/nirs4all-methods/blob/main/cpp/src/core/preprocessing/signal_conversion/percent_to_fraction.c


---

_See also_: [methods index](index.md) · [interactive dashboard](../landing/dashboard.md)