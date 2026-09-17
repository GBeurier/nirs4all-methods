# `aug_edge_artifacts` — Configured pipeline of four edge artifacts

_Group_: **Augmentation** · _C ABI_: `n4m_augmentation_edge_artifacts_*`

## Description

Combined edge artifact augmenter.

## Parameters

| Name | Type | Default |
|------|------|---------|
| `enabled_flags` | `int` | `15` |
| `overall_strength` | `float` | `1.0` |
| `detector_model` | `int` | `4` |
| `wavelengths` | `—` | `None` |
| `rng` | `Optional[PCG64]` | `None` |
| `seed` | `int` | `0` |

## API and bindings

**C ABI (ABI 2):** [`n4m_augmentation_edge_artifacts_apply`](https://github.com/GBeurier/nirs4all-methods/blob/main/cpp/include/n4m/augmentation/instrument.h#L137) · [`n4m_augmentation_edge_artifacts_create`](https://github.com/GBeurier/nirs4all-methods/blob/main/cpp/include/n4m/augmentation/instrument.h#L131) · [`n4m_augmentation_edge_artifacts_destroy`](https://github.com/GBeurier/nirs4all-methods/blob/main/cpp/include/n4m/augmentation/instrument.h#L142). Use the linked public header for the exact signature, configuration, and result handles.

**Python (verified public re-export):**

```python
from n4m.augmentation.instrument import EdgeArtifactsAugmenter
```

Source signature: [`EdgeArtifactsAugmenter(enabled_flags: int = 15, overall_strength: float = 1.0, detector_model: int = 4, wavelengths = None, rng: Optional[PCG64] = None, seed: int = 0)`](https://github.com/GBeurier/nirs4all-methods/blob/main/bindings/python/src/n4m/_impl/augmentation.py#L1092).

**R:** no current source-verified entry point was found for this catalog method.

**MATLAB / Octave:** no current source-verified entry point was found for this catalog method.

## Explanations

### Bibliographic source

No paper defines this composite. It is an implementation-defined pipeline combining truncated peaks, curvature, stray light, and detector roll-off.

### Mathematical principle

`enabled_flags` selects four suboperators. Each receives a deterministic child PCG64 stream and source-defined defaults scaled by `overall_strength`. Enabled stages run in this exact order: truncated peaks, edge curvature, stray light, detector roll-off, with every stage consuming the previous stage's output.

### Appropriate uses

Combined stress tests of several boundary artifacts with one reproducible operator.

### Limits and validation

Composition is noncommutative and `overall_strength` scales suboperators differently; it is not a universal scalar severity. Suboperator defaults are fixed rather than individually exposed.

### Implementation

Python role API `n4m.augmentation.instrument.EdgeArtifactsAugmenter`; ABI 2 family `n4m_augmentation_edge_artifacts_{create,apply,destroy}`.

The ABI-2 implementation is the `n4m_augmentation_edge_artifacts_*` lifecycle in libn4m.

### Sources and provenance

https://github.com/GBeurier/nirs4all-methods/blob/main/cpp/src/c_api/c_api_augmenters_edge_splines_random.cpp#L426


---

_See also_: [methods index](index.md) · [interactive dashboard](../landing/dashboard.md)