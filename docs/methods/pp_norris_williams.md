# `pp_norris_williams` — Norris–Williams segment-gap derivative

_Group_: **Preprocessing** · _C ABI_: `n4m_transform_norris_williams_*`

## Description

Segment smoothing followed by gap finite differences.

## Parameters

| Name | Type | Default |
|------|------|---------|
| `segment` | `int` | `5` |
| `gap` | `int` | `5` |
| `derivative_order` | `int` | `1` |
| `delta` | `float` | `1.0` |

## API and bindings

**C ABI (ABI 2):** [`n4m_transform_norris_williams_create`](https://github.com/GBeurier/nirs4all-methods/blob/main/cpp/include/n4m/transform/smoothing.h#L125) · [`n4m_transform_norris_williams_destroy`](https://github.com/GBeurier/nirs4all-methods/blob/main/cpp/include/n4m/transform/smoothing.h#L128) · [`n4m_transform_norris_williams_transform`](https://github.com/GBeurier/nirs4all-methods/blob/main/cpp/include/n4m/transform/smoothing.h#L130). Use the linked public header for the exact signature, configuration, and result handles.

**Python (verified public re-export):**

```python
from n4m.transform.smoothing import NorrisWilliams
```

Source signature: [`NorrisWilliams(segment: int = 5, gap: int = 5, derivative_order: int = 1, delta: float = 1.0)`](https://github.com/GBeurier/nirs4all-methods/blob/main/bindings/python/src/n4m/_impl/preprocessing.py#L320).

**R:** no current source-verified entry point was found for this catalog method.

**MATLAB / Octave:** no current source-verified entry point was found for this catalog method.

## Explanations

### Bibliographic source

Norris & Williams (1984), *Optimization of mathematical treatments of raw near-infrared signal in the measurement of protein in hard red spring wheat*, Cereal Chemistry 61, 158–165 (bibliographic record: https://www.cerealsgrains.org/publications/cc/backissues/1984/Documents/61_158.pdf).

### Mathematical principle

Each row is first averaged in non-overlapping or sliding segments of length `segment`; first or second finite differences are then taken between segment means separated by `gap`, scaled by `delta`. Smoothing precedes differentiation.

### Appropriate uses

Traditional NIR pretreatment for attenuating noise while removing offsets or linear background and sharpening broad bands.

### Limits and validation

Segment and gap are channel counts and assume a uniform grid. The output loses edge positions, large gaps blur narrow bands, and differentiation still amplifies residual noise.

### Implementation

`n4m.transform.smoothing.NorrisWilliams` uses `n4m_transform_norris_williams_*`; segment averaging and gap derivatives are in `preprocessing/derivatives/norris_williams.c`.

The ABI-2 implementation is the `n4m_transform_norris_williams_*` lifecycle in libn4m.

### Sources and provenance

https://github.com/GBeurier/nirs4all-methods/blob/main/cpp/src/core/preprocessing/derivatives/norris_williams.c; https://www.cerealsgrains.org/publications/cc/backissues/1984/Documents/61_158.pdf


---

_See also_: [methods index](index.md) · [interactive dashboard](../landing/dashboard.md)