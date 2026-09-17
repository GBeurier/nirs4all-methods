# `pp_rolling_ball` — Rolling-ball morphological baseline correction

_Group_: **Baseline correction** · _C ABI_: `n4m_transform_rolling_ball_*`

## Description

Rolling-ball morphological baseline correction.

## Parameters

| Name | Type | Default |
|------|------|---------|
| `half_window` | `int` | `20` |
| `smooth_half_window` | `int` | `0` |

## API and bindings

**C ABI (ABI 2):** [`n4m_transform_rolling_ball_create`](https://github.com/GBeurier/nirs4all-methods/blob/main/cpp/include/n4m/transform/baseline.h#L110) · [`n4m_transform_rolling_ball_destroy`](https://github.com/GBeurier/nirs4all-methods/blob/main/cpp/include/n4m/transform/baseline.h#L113) · [`n4m_transform_rolling_ball_transform`](https://github.com/GBeurier/nirs4all-methods/blob/main/cpp/include/n4m/transform/baseline.h#L115). Use the linked public header for the exact signature, configuration, and result handles.

**Python (verified public re-export):**

```python
from n4m.transform.baseline import RollingBall
```

Source signature: [`RollingBall(half_window: int = 20, smooth_half_window: int = 0)`](https://github.com/GBeurier/nirs4all-methods/blob/main/bindings/python/src/n4m/_impl/baseline.py#L178).

**R:** no current source-verified entry point was found for this catalog method.

**MATLAB / Octave:** no current source-verified entry point was found for this catalog method.

## Explanations

### Bibliographic source

Kneen & Annegarn (1996), *Algorithm for fitting XRF, SEM and PIXE X-ray spectra backgrounds*, Nuclear Instruments and Methods in Physics Research B 109, 209–213, https://doi.org/10.1016/0168-583X(95)00908-6.

### Mathematical principle

A lower morphological envelope is estimated with a structuring radius `half_window` (erosion followed by dilation/rolling-ball analogue), optionally smoothed, and subtracted from the spectrum.

### Appropriate uses

Removing slowly varying positive backgrounds without fitting a global polynomial.

### Limits and validation

Features wider than the structuring element may enter the baseline, while too large a window underfits drift. Morphological operations can create edge artifacts and are not differentiable.

### Implementation

`n4m.transform.baseline.RollingBall` uses `n4m_transform_rolling_ball_*`; the 1-D morphological envelope is in `preprocessing/baselines/rolling_ball.c`.

The ABI-2 implementation is the `n4m_transform_rolling_ball_*` lifecycle in libn4m.

### Sources and provenance

https://doi.org/10.1016/0168-583X(95)00908-6; https://github.com/GBeurier/nirs4all-methods/blob/main/cpp/src/core/preprocessing/baselines/rolling_ball.c


---

_See also_: [methods index](index.md) · [interactive dashboard](../landing/dashboard.md)