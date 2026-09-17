# `aug_truncated_peak` — Off-range Gaussian peak tails

_Group_: **Augmentation** · _C ABI_: `n4m_augmentation_truncated_peak_*`

## Description

Truncated peaks near spectral edges.

## Parameters

| Name | Type | Default |
|------|------|---------|
| `peak_probability` | `float` | `0.5` |
| `amplitude_min` | `float` | `0.01` |
| `amplitude_max` | `float` | `0.1` |
| `width_min` | `float` | `50.0` |
| `width_max` | `float` | `200.0` |
| `left_edge` | `bool` | `True` |
| `right_edge` | `bool` | `True` |
| `wavelengths` | `—` | `None` |
| `rng` | `Optional[PCG64]` | `None` |
| `seed` | `int` | `0` |

## API and bindings

**C ABI (ABI 2):** [`n4m_augmentation_truncated_peak_apply`](https://github.com/GBeurier/nirs4all-methods/blob/main/cpp/include/n4m/augmentation/instrument.h#L108) · [`n4m_augmentation_truncated_peak_create`](https://github.com/GBeurier/nirs4all-methods/blob/main/cpp/include/n4m/augmentation/instrument.h#L100) · [`n4m_augmentation_truncated_peak_destroy`](https://github.com/GBeurier/nirs4all-methods/blob/main/cpp/include/n4m/augmentation/instrument.h#L113). Use the linked public header for the exact signature, configuration, and result handles.

**Python (verified public re-export):**

```python
from n4m.augmentation.instrument import TruncatedPeakAugmenter
```

Source signature: [`TruncatedPeakAugmenter(peak_probability: float = 0.5, amplitude_min: float = 0.01, amplitude_max: float = 0.1, width_min: float = 50.0, width_max: float = 200.0, left_edge: bool = True, right_edge: bool = True, wavelengths = None, rng: Optional[PCG64] = None, seed: int = 0)`](https://github.com/GBeurier/nirs4all-methods/blob/main/bindings/python/src/n4m/_impl/augmentation.py#L1059).

**R:** no current source-verified entry point was found for this catalog method.

**MATLAB / Octave:** no current source-verified entry point was found for this catalog method.

## Explanations

### Bibliographic source

No canonical paper defines this artifact generator. It is an internal model of bands whose centers lie just outside the measured interval.

### Mathematical principle

With `peak_probability` per sample, choose an enabled left or right edge, draw amplitude and width from their configured ranges, place a Gaussian center beyond that edge, and add the in-range tail to the spectrum.

### Appropriate uses

Testing sensitivity to partially observed absorption bands at cropped spectral edges.

### Limits and validation

Only Gaussian positive tails from enabled edges are represented. Width and amplitude units must match the wavelength axis and data scale; overlapping real bands are not inferred.

### Implementation

Python role API `n4m.augmentation.instrument.TruncatedPeakAugmenter`; ABI 2 family `n4m_augmentation_truncated_peak_{create,apply,destroy}`.

The ABI-2 implementation is the `n4m_augmentation_truncated_peak_*` lifecycle in libn4m.

### Sources and provenance

https://github.com/GBeurier/nirs4all-methods/blob/main/cpp/src/core/augmentation/edge_artifacts/truncated_peak.h


---

_See also_: [methods index](index.md) · [interactive dashboard](../landing/dashboard.md)