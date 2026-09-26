# `pp_to_absorbance` — Reflectance/transmittance-to-absorbance conversion

_Group_: **Preprocessing** · _C ABI_: `n4m_transform_to_absorbance_*`

## Description

A = -log10(max(R, epsilon)). Optional %-scaling.

## Parameters

| Name | Type | Default |
|------|------|---------|
| `is_percent` | `bool` | `False` |
| `epsilon` | `float` | `1e-10` |
| `clip_negative` | `bool` | `True` |

## API and bindings

**C ABI (ABI 2):** [`n4m_transform_to_absorbance_create`](https://github.com/GBeurier/nirs4all-methods/blob/main/cpp/include/n4m/transform/signal_conversion.h#L13) · [`n4m_transform_to_absorbance_destroy`](https://github.com/GBeurier/nirs4all-methods/blob/main/cpp/include/n4m/transform/signal_conversion.h#L16) · [`n4m_transform_to_absorbance_transform`](https://github.com/GBeurier/nirs4all-methods/blob/main/cpp/include/n4m/transform/signal_conversion.h#L18). Use the linked public header for the exact signature, configuration, and result handles.

**Python (verified public re-export):**

```python
from n4m.transform.signal_conversion import ToAbsorbance
```

Source signature: [`ToAbsorbance(is_percent: bool = False, epsilon: float = 1e-10, clip_negative: bool = True)`](https://github.com/GBeurier/nirs4all-methods/blob/main/bindings/python/src/n4m/_impl/preprocessing.py#L547).

**R:** no current source-verified entry point was found for this catalog method.

**MATLAB / Octave:** no current source-verified entry point was found for this catalog method.

## Explanations

### Bibliographic source

Beer (1852), *Bestimmung der Absorption des rothen Lichts in farbigen Flüssigkeiten*, Annalen der Physik 162, 78–88, https://doi.org/10.1002/andp.18521620505.

### Mathematical principle

Percent inputs are first divided by 100, then the operator computes $A=-\log_{10}(\max(R,\epsilon))$. With `clip_negative`, negative signals are guarded before the logarithm according to the implementation contract.

### Appropriate uses

Converting reflectance or transmittance into a log scale closer to additive optical density before calibration.

### Limits and validation

The transform assumes correct input units and positive signals. Clipping hides invalid measurements, and reflectance absorbance is an empirical pseudo-absorbance rather than a guaranteed Beer–Lambert quantity.

### Implementation

`n4m.transform.signal_conversion.ToAbsorbance` uses `n4m_transform_to_absorbance_*`; unit handling and guards are in `signal_conversion/to_absorbance.c`.

The ABI-2 implementation is the `n4m_transform_to_absorbance_*` lifecycle in libn4m.

### Sources and provenance

https://doi.org/10.1002/andp.18521620505; https://github.com/GBeurier/nirs4all-methods/blob/main/cpp/src/core/preprocessing/signal_conversion/to_absorbance.c


---

_See also_: [methods index](index.md) · [interactive dashboard](../landing/dashboard.md)