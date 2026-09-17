# `aug_spline_smooth` — Deterministic natural-cubic spline smoothing

_Group_: **Augmentation** · _C ABI_: `n4m_augmentation_spline_smoothing_*`

## Description

Spline smoothing augmenter.

## Parameters

| Name | Type | Default |
|------|------|---------|
| `rng` | `Optional[PCG64]` | `None` |
| `seed` | `int` | `0` |

## API and bindings

**C ABI (ABI 2):** [`n4m_augmentation_spline_smoothing_apply`](https://github.com/GBeurier/nirs4all-methods/blob/main/cpp/include/n4m/augmentation/splines.h#L17) · [`n4m_augmentation_spline_smoothing_create`](https://github.com/GBeurier/nirs4all-methods/blob/main/cpp/include/n4m/augmentation/splines.h#L14) · [`n4m_augmentation_spline_smoothing_destroy`](https://github.com/GBeurier/nirs4all-methods/blob/main/cpp/include/n4m/augmentation/splines.h#L20). Use the linked public header for the exact signature, configuration, and result handles.

**Python (verified public re-export):**

```python
from n4m.augmentation.splines import SplineSmoothingAugmenter
```

Source signature: [`SplineSmoothingAugmenter(rng: Optional[PCG64] = None, seed: int = 0)`](https://github.com/GBeurier/nirs4all-methods/blob/main/bindings/python/src/n4m/_impl/augmentation.py#L1118).

**R:** no current source-verified entry point was found for this catalog method.

**MATLAB / Octave:** no current source-verified entry point was found for this catalog method.

## Explanations

### Bibliographic source

Reinsch (1967), *Smoothing by spline functions*, Numerische Mathematik 10, 177–183, DOI 10.1007/BF02162161 (https://doi.org/10.1007/BF02162161), provides the classical smoothing-spline basis; the exact fixed smoothing choice here is implementation-specific.

### Mathematical principle

Fit a natural cubic smoothing spline independently to each row on the channel grid and evaluate it on that same grid. The shipped parity target corresponds to a fixed smoothing budget $s=1/p$ for $p$ channels; the RNG and `seed` are accepted by the common wrapper but unused.

### Appropriate uses

Creating a smoothed view of spectra to reduce high-frequency variation during training.

### Limits and validation

This operator is deterministic and therefore does not enlarge a dataset with multiple draws. A fixed $1/p$ smoothing choice may be inappropriate across data scales.

### Implementation

Python role API `n4m.augmentation.splines.SplineSmoothingAugmenter`; ABI 2 family `n4m_augmentation_spline_smoothing_{create,apply,destroy}`.

The ABI-2 implementation is the `n4m_augmentation_spline_smoothing_*` lifecycle in libn4m.

### Sources and provenance

https://github.com/GBeurier/nirs4all-methods/blob/main/cpp/src/core/augmentation/splines/spline_smoothing.h; https://doi.org/10.1007/BF02162161


---

_See also_: [methods index](index.md) · [interactive dashboard](../landing/dashboard.md)