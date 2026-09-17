# `aug_moisture` — Water-activity and moisture-band perturbation

_Group_: **Augmentation** · _C ABI_: `n4m_augmentation_moisture_*`

## Description

Water activity and moisture-content spectral perturbation.

## Parameters

| Name | Type | Default |
|------|------|---------|
| `water_activity_delta` | `float` | `0.0` |
| `use_aw_range` | `bool` | `False` |
| `aw_low` | `float` | `0.0` |
| `aw_high` | `float` | `1.0` |
| `reference_water_activity` | `float` | `0.5` |
| `free_water_fraction` | `float` | `0.3` |
| `bound_water_shift` | `float` | `25.0` |
| `moisture_content` | `float` | `0.1` |
| `enable_shift` | `bool` | `True` |
| `enable_intensity` | `bool` | `True` |
| `wavelengths` | `—` | `None` |
| `rng` | `Optional[PCG64]` | `None` |
| `seed` | `int` | `0` |

## API and bindings

**C ABI (ABI 2):** [`n4m_augmentation_moisture_apply`](https://github.com/GBeurier/nirs4all-methods/blob/main/cpp/include/n4m/augmentation/instrument.h#L40) · [`n4m_augmentation_moisture_create`](https://github.com/GBeurier/nirs4all-methods/blob/main/cpp/include/n4m/augmentation/instrument.h#L29) · [`n4m_augmentation_moisture_destroy`](https://github.com/GBeurier/nirs4all-methods/blob/main/cpp/include/n4m/augmentation/instrument.h#L43). Use the linked public header for the exact signature, configuration, and result handles.

**Python (verified public re-export):**

```python
from n4m.augmentation.instrument import MoistureAugmenter
```

Source signature: [`MoistureAugmenter(water_activity_delta: float = 0.0, use_aw_range: bool = False, aw_low: float = 0.0, aw_high: float = 1.0, reference_water_activity: float = 0.5, free_water_fraction: float = 0.3, bound_water_shift: float = 25.0, moisture_content: float = 0.1, enable_shift: bool = True, enable_intensity: bool = True, wavelengths = None, rng: Optional[PCG64] = None, seed: int = 0)`](https://github.com/GBeurier/nirs4all-methods/blob/main/bindings/python/src/n4m/_impl/augmentation.py#L947).

**R:** no current source-verified entry point was found for this catalog method.

**MATLAB / Octave:** no current source-verified entry point was found for this catalog method.

## Explanations

### Bibliographic source

No canonical paper specifies this combined sigmoid/free-bound-water model. It is an internal heuristic with source-defined 1435 and 1930 wavelength bands.

### Mathematical principle

Set water activity $a_w=\operatorname{clip}(a_{ref}+\Delta a,0,1)$, with fixed or uniformly drawn $\Delta a$. A sigmoid allocates free versus bound water. Bound fraction shifts windows around 1435 and 1930; optional intensity adds Gaussian band profiles scaled by `moisture_content/0.10 - 1` and the row mean magnitude.

### Appropriate uses

Sensitivity studies for water-band displacement and amplitude changes in moist materials.

### Limits and validation

Band centers assume compatible wavelength units and coverage. The model is qualitative, not a calibration from water activity or moisture percentage to absorbance.

### Implementation

Python role API `n4m.augmentation.instrument.MoistureAugmenter`; ABI 2 family `n4m_augmentation_moisture_{create,apply,destroy}`.

The ABI-2 implementation is the `n4m_augmentation_moisture_*` lifecycle in libn4m.

### Sources and provenance

https://github.com/GBeurier/nirs4all-methods/blob/main/cpp/src/core/augmentation/environmental/moisture.h


---

_See also_: [methods index](index.md) · [interactive dashboard](../landing/dashboard.md)