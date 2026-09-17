# `aug_temperature` — Region-specific temperature perturbation heuristic

_Group_: **Augmentation** · _C ABI_: `n4m_augmentation_temperature_*`

## Description

Temperature-induced shift, intensity and broadening perturbations.

## Parameters

| Name | Type | Default |
|------|------|---------|
| `temperature_delta` | `float` | `0.0` |
| `use_temp_range` | `bool` | `False` |
| `temp_low` | `float` | `-5.0` |
| `temp_high` | `float` | `5.0` |
| `enable_shift` | `bool` | `True` |
| `enable_intensity` | `bool` | `True` |
| `enable_broadening` | `bool` | `True` |
| `region_specific` | `bool` | `True` |
| `wavelengths` | `—` | `None` |
| `rng` | `Optional[PCG64]` | `None` |
| `seed` | `int` | `0` |

## API and bindings

**C ABI (ABI 2):** [`n4m_augmentation_temperature_apply`](https://github.com/GBeurier/nirs4all-methods/blob/main/cpp/include/n4m/augmentation/instrument.h#L21) · [`n4m_augmentation_temperature_create`](https://github.com/GBeurier/nirs4all-methods/blob/main/cpp/include/n4m/augmentation/instrument.h#L13) · [`n4m_augmentation_temperature_destroy`](https://github.com/GBeurier/nirs4all-methods/blob/main/cpp/include/n4m/augmentation/instrument.h#L24). Use the linked public header for the exact signature, configuration, and result handles.

**Python (verified public re-export):**

```python
from n4m.augmentation.instrument import TemperatureAugmenter
```

Source signature: [`TemperatureAugmenter(temperature_delta: float = 0.0, use_temp_range: bool = False, temp_low: float = -5.0, temp_high: float = 5.0, enable_shift: bool = True, enable_intensity: bool = True, enable_broadening: bool = True, region_specific: bool = True, wavelengths = None, rng: Optional[PCG64] = None, seed: int = 0)`](https://github.com/GBeurier/nirs4all-methods/blob/main/bindings/python/src/n4m/_impl/augmentation.py#L911).

**R:** no current source-verified entry point was found for this catalog method.

**MATLAB / Octave:** no current source-verified entry point was found for this catalog method.

## Explanations

### Bibliographic source

No publication canonically defines the six coefficient presets used here. This is an internal phenomenological model; source coefficients and regions are normative.

### Mathematical principle

Use fixed `temperature_delta` or draw one from `temp_low`–`temp_high`. In six built-in O–H/C–H/N–H/water regions, sigmoid windows blend coefficient-scaled wavelength shift, intensity change, and Gaussian broadening. With `region_specific=False`, averages of those coefficients act over the full axis.

### Appropriate uses

Qualitative robustness experiments for temperature-sensitive NIR band positions and shapes.

### Limits and validation

The coefficients are presets, not fitted thermodynamic parameters. Physical wavelengths are required; data outside the encoded regions receive little or averaged effect. Changes below 0.01 degree-equivalent are ignored.

### Implementation

Python role API `n4m.augmentation.instrument.TemperatureAugmenter`; ABI 2 family `n4m_augmentation_temperature_{create,apply,destroy}`.

The ABI-2 implementation is the `n4m_augmentation_temperature_*` lifecycle in libn4m.

### Sources and provenance

https://github.com/GBeurier/nirs4all-methods/blob/main/cpp/src/core/augmentation/environmental/temperature.h


---

_See also_: [methods index](index.md) · [interactive dashboard](../landing/dashboard.md)