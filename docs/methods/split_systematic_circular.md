# `split_systematic_circular` — Systematic circular sampling over sorted targets

_Group_: **Splitters** · _C ABI_: `n4m_model_selection_systematic_circular_*`

## Description

Systematic circular split over sorted or ordered targets.

## Parameters

| Name | Type | Default |
|------|------|---------|
| `test_size` | `float` | `0.25` |
| `seed` | `int` | `0` |

## API and bindings

**C ABI (ABI 2):** [`n4m_model_selection_systematic_circular_create`](https://github.com/GBeurier/nirs4all-methods/blob/main/cpp/include/n4m/model_selection.h#L122) · [`n4m_model_selection_systematic_circular_destroy`](https://github.com/GBeurier/nirs4all-methods/blob/main/cpp/include/n4m/model_selection.h#L124) · [`n4m_model_selection_systematic_circular_split`](https://github.com/GBeurier/nirs4all-methods/blob/main/cpp/include/n4m/model_selection.h#L126). Use the linked public header for the exact signature, configuration, and result handles.

**Python (verified public re-export):**

```python
from n4m.model_selection.splitters import SystematicCircular
```

Source signature: [`SystematicCircular(test_size: float = 0.25, seed: int = 0)`](https://github.com/GBeurier/nirs4all-methods/blob/main/bindings/python/src/n4m/_impl/splitters.py#L414).

**R:** no current source-verified entry point was found for this catalog method.

**MATLAB / Octave:** no current source-verified entry point was found for this catalog method.

## Explanations

### Bibliographic source

No canonical paper defines this exact rotate-and-systematically-sample implementation. It is an internal systematic sampling heuristic whose seeded offset and rounding rules are source-defined.

### Mathematical principle

Sort sample indices by y, draw a seeded circular offset, rotate that order, and choose `n_train` positions at approximately equal spacing $step=n/n_{train}$ using rounded $step\,i$. The remaining rotated positions form the test set; the engine sorts both final index arrays for stable output.

### Appropriate uses

Spreading calibration samples across the ordered response range without bin boundaries.

### Limits and validation

It uses y and therefore is inappropriate for a blind final test. Periodic ordering and rounding can create structure; ties are resolved by sorting behavior, and multicolumn y is rejected. The current Python binding requires y as `(n_samples, 1)` because its generic 1-D promotion creates the wrong orientation.

### Implementation

Python role API `n4m.model_selection.splitters.SystematicCircular`; binding class `SystematicCircularSplitter`; ABI 2 family `n4m_model_selection_systematic_circular_{create,split,destroy}`.

The ABI-2 implementation is the `n4m_model_selection_systematic_circular_*` lifecycle in libn4m.

### Sources and provenance

https://github.com/GBeurier/nirs4all-methods/blob/main/cpp/src/core/splitters/systematic_circular.h


---

_See also_: [methods index](index.md) · [interactive dashboard](../landing/dashboard.md)