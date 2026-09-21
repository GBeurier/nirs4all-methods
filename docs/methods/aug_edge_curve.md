# `aug_edge_curve` — Smooth detector-edge curvature

_Group_: **Augmentation** · _C ABI_: `n4m_augmentation_edge_curvature_*`

## Description

Curved edge response artifact.

## Parameters

| Name | Type | Default |
|------|------|---------|
| `curvature_strength` | `float` | `0.02` |
| `curvature_type` | `int` | `0` |
| `asymmetry` | `float` | `0.0` |
| `edge_focus` | `float` | `0.7` |
| `wavelengths` | `—` | `None` |
| `rng` | `Optional[PCG64]` | `None` |
| `seed` | `int` | `0` |

## API and bindings

**C ABI (ABI 2):** [`n4m_augmentation_edge_curvature_apply`](https://github.com/GBeurier/nirs4all-methods/blob/main/cpp/include/n4m/augmentation/instrument.h#L90) · [`n4m_augmentation_edge_curvature_create`](https://github.com/GBeurier/nirs4all-methods/blob/main/cpp/include/n4m/augmentation/instrument.h#L83) · [`n4m_augmentation_edge_curvature_destroy`](https://github.com/GBeurier/nirs4all-methods/blob/main/cpp/include/n4m/augmentation/instrument.h#L95). Use the linked public header for the exact signature, configuration, and result handles.

**Python (verified public re-export):**

```python
from n4m.augmentation.instrument import EdgeCurvatureAugmenter
```

Source signature: [`EdgeCurvatureAugmenter(curvature_strength: float = 0.02, curvature_type: int = 0, asymmetry: float = 0.0, edge_focus: float = 0.7, wavelengths = None, rng: Optional[PCG64] = None, seed: int = 0)`](https://github.com/GBeurier/nirs4all-methods/blob/main/bindings/python/src/n4m/_impl/augmentation.py#L1036).

**R:** no current source-verified entry point was found for this catalog method.

**MATLAB / Octave:** no current source-verified entry point was found for this catalog method.

## Explanations

### Bibliographic source

No canonical paper defines the smile/frown/asymmetric templates. They are internal edge-response heuristics with formulas fixed by the native source.

### Mathematical principle

A normalized wavelength coordinate drives a smooth edge-focused curve. `curvature_type` selects random, smile, frown, or asymmetric shape; `curvature_strength` sets amplitude, `asymmetry` differentiates left and right, and `edge_focus` concentrates the effect toward edges.

### Appropriate uses

Robustness to smooth baseline curvature near the boundaries of an instrument's range.

### Limits and validation

Template curvature is phenomenological and depends on the supplied wavelength span. It does not estimate optical smile or detector geometry from metadata.

### Implementation

Python role API `n4m.augmentation.instrument.EdgeCurvatureAugmenter`; ABI 2 family `n4m_augmentation_edge_curvature_{create,apply,destroy}`.

The ABI-2 implementation is the `n4m_augmentation_edge_curvature_*` lifecycle in libn4m.

### Sources and provenance

https://github.com/GBeurier/nirs4all-methods/blob/main/cpp/src/core/augmentation/edge_artifacts/edge_curvature.h


---

_See also_: [methods index](index.md) · [interactive dashboard](../landing/dashboard.md)