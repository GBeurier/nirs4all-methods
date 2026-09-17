# `aug_spline_x_perturb` — Spline-based x-axis perturbation

_Group_: **Augmentation** · _C ABI_: `n4m_augmentation_spline_x_perturbations_*`

## Description

Spline x-axis perturbation augmenter.

## Parameters

| Name | Type | Default |
|------|------|---------|
| `spline_degree` | `int` | `3` |
| `perturbation_density` | `float` | `0.05` |
| `perturbation_range_min` | `float` | `-0.1` |
| `perturbation_range_max` | `float` | `0.1` |
| `rng` | `Optional[PCG64]` | `None` |
| `seed` | `int` | `0` |

## API and bindings

**C ABI (ABI 2):** [`n4m_augmentation_spline_x_perturbations_apply`](https://github.com/GBeurier/nirs4all-methods/blob/main/cpp/include/n4m/augmentation/splines.h#L33) · [`n4m_augmentation_spline_x_perturbations_create`](https://github.com/GBeurier/nirs4all-methods/blob/main/cpp/include/n4m/augmentation/splines.h#L26) · [`n4m_augmentation_spline_x_perturbations_destroy`](https://github.com/GBeurier/nirs4all-methods/blob/main/cpp/include/n4m/augmentation/splines.h#L36). Use the linked public header for the exact signature, configuration, and result handles.

**Python (verified public re-export):**

```python
from n4m.augmentation.splines import SplineXPerturbationAugmenter
```

Source signature: [`SplineXPerturbationAugmenter(spline_degree: int = 3, perturbation_density: float = 0.05, perturbation_range_min: float = -0.1, perturbation_range_max: float = 0.1, rng: Optional[PCG64] = None, seed: int = 0)`](https://github.com/GBeurier/nirs4all-methods/blob/main/bindings/python/src/n4m/_impl/augmentation.py#L1127).

**R:** no current source-verified entry point was found for this catalog method.

**MATLAB / Octave:** no current source-verified entry point was found for this catalog method.

## Explanations

### Bibliographic source

No canonical paper defines this random x-grid perturbation. The native implementation is a parity-oriented analogue of SciPy spline resampling.

### Mathematical principle

Fit a not-a-knot cubic B-spline to each row. Set the number of equally spaced perturbation anchors to $\max[2,\operatorname{round}((p+4)d)]$, where d is `perturbation_density`; draw an offset at each anchor from `perturbation_range_min` to `perturbation_range_max`, linearly interpolate those offsets onto the spline knot vector, perturb the knots, and evaluate the altered spline on the original channel grid.

### Appropriate uses

Robustness to sparse wavelength-registration errors with smooth interpolation between points.

### Limits and validation

Only `spline_degree=3` is accepted by the native constructor. Perturbed knots can become poorly ordered at excessive ranges; cubic evaluation can overshoot, and channel coordinates rather than physical wavelengths drive this API.

### Implementation

Python role API `n4m.augmentation.splines.SplineXPerturbationAugmenter`; ABI 2 family `n4m_augmentation_spline_x_perturbations_{create,apply,destroy}`.

The ABI-2 implementation is the `n4m_augmentation_spline_x_perturbations_*` lifecycle in libn4m.

### Sources and provenance

https://github.com/GBeurier/nirs4all-methods/blob/main/cpp/src/core/augmentation/splines/spline_x_perturbations.h


---

_See also_: [methods index](index.md) · [interactive dashboard](../landing/dashboard.md)