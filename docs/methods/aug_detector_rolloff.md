# `aug_detector_rolloff` — Detector sensitivity roll-off artifact

_Group_: **Augmentation** · _C ABI_: `n4m_augmentation_detector_rolloff_*`

## Description

Detector edge roll-off artifact.

## Parameters

| Name | Type | Default |
|------|------|---------|
| `detector_model` | `int` | `4` |
| `effect_strength` | `float` | `1.0` |
| `noise_amplification` | `float` | `0.02` |
| `include_baseline_distortion` | `bool` | `True` |
| `wavelengths` | `—` | `None` |
| `rng` | `Optional[PCG64]` | `None` |
| `seed` | `int` | `0` |

## API and bindings

**C ABI (ABI 2):** [`n4m_augmentation_detector_rolloff_apply`](https://github.com/GBeurier/nirs4all-methods/blob/main/cpp/include/n4m/augmentation/instrument.h#L56) · [`n4m_augmentation_detector_rolloff_create`](https://github.com/GBeurier/nirs4all-methods/blob/main/cpp/include/n4m/augmentation/instrument.h#L49) · [`n4m_augmentation_detector_rolloff_destroy`](https://github.com/GBeurier/nirs4all-methods/blob/main/cpp/include/n4m/augmentation/instrument.h#L61). Use the linked public header for the exact signature, configuration, and result handles.

**Python (verified public re-export):**

```python
from n4m.augmentation.instrument import DetectorRollOffAugmenter
```

Source signature: [`DetectorRollOffAugmenter(detector_model: int = 4, effect_strength: float = 1.0, noise_amplification: float = 0.02, include_baseline_distortion: bool = True, wavelengths = None, rng: Optional[PCG64] = None, seed: int = 0)`](https://github.com/GBeurier/nirs4all-methods/blob/main/bindings/python/src/n4m/_impl/augmentation.py#L988).

**R:** no current source-verified entry point was found for this catalog method.

**MATLAB / Octave:** no current source-verified entry point was found for this catalog method.

## Explanations

### Bibliographic source

No single paper defines the five detector presets in this implementation. They are literature-inspired internal curves; the enum and native source are the auditable specification.

### Mathematical principle

The selected InGaAs, extended InGaAs, PbS, silicon CCD, or generic NIR preset defines an optimal range, roll-off rate, and minimum sensitivity. Falling sensitivity toward the edges amplifies random noise and, when enabled, adds a small baseline distortion; `effect_strength` and `noise_amplification` scale these terms.

### Appropriate uses

Robustness testing near detector range limits or across detector technologies.

### Limits and validation

Presets are not manufacturer response calibrations and require a meaningful wavelength axis. They should not be used to identify a detector or predict its signal-to-noise ratio.

### Implementation

Python role API `n4m.augmentation.instrument.DetectorRollOffAugmenter`; ABI 2 family `n4m_augmentation_detector_rolloff_{create,apply,destroy}`.

The ABI-2 implementation is the `n4m_augmentation_detector_rolloff_*` lifecycle in libn4m.

### Sources and provenance

https://github.com/GBeurier/nirs4all-methods/blob/main/cpp/src/core/augmentation/edge_artifacts/detector_rolloff.h


---

_See also_: [methods index](index.md) · [interactive dashboard](../landing/dashboard.md)