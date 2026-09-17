# `aug_poly_drift` — Random polynomial baseline drift

_Group_: **Augmentation** · _C ABI_: `n4m_augmentation_poly_drift_*`

## Description

Add random polynomial baseline drift.

## Parameters

| Name | Type | Default |
|------|------|---------|
| `degree` | `int` | `2` |
| `coeff_min` | `—` | `None` |
| `coeff_max` | `—` | `None` |
| `rng` | `Optional[PCG64]` | `None` |
| `seed` | `int` | `0` |

## API and bindings

**C ABI (ABI 2):** [`n4m_augmentation_poly_drift_apply`](https://github.com/GBeurier/nirs4all-methods/blob/main/cpp/include/n4m/augmentation/drift.h#L31) · [`n4m_augmentation_poly_drift_create`](https://github.com/GBeurier/nirs4all-methods/blob/main/cpp/include/n4m/augmentation/drift.h#L25) · [`n4m_augmentation_poly_drift_destroy`](https://github.com/GBeurier/nirs4all-methods/blob/main/cpp/include/n4m/augmentation/drift.h#L30). Use the linked public header for the exact signature, configuration, and result handles.

**Python (verified public re-export):**

```python
from n4m.augmentation.drift import PolynomialBaselineDrift
```

Source signature: [`PolynomialBaselineDrift(degree: int = 2, coeff_min = None, coeff_max = None, rng: Optional[PCG64] = None, seed: int = 0)`](https://github.com/GBeurier/nirs4all-methods/blob/main/bindings/python/src/n4m/_impl/augmentation.py#L472).

**R:** no current source-verified entry point was found for this catalog method.

**MATLAB / Octave:** no current source-verified entry point was found for this catalog method.

## Explanations

### Bibliographic source

No canonical publication specifies these random coefficient ranges. The method is a controlled extension of offset/slope spectral augmentation.

### Mathematical principle

Channels are mapped to $t_j\in[-1,1]$. For every order $k=0,\ldots,d$, a coefficient $c_{ik}$ is sampled uniformly between `coeff_min[k]` and `coeff_max[k]`, and $X'_{ij}=X_{ij}+\sum_{k=0}^{d}c_{ik}t_j^k$. `degree` sets curvature order; the coefficient arrays must contain `degree + 1` bounds.

### Appropriate uses

Generating smooth, low-frequency baseline shapes beyond an affine tilt.

### Limits and validation

Polynomial drift is a numerical nuisance model rather than a physical scatter model. High degrees can oscillate and extrapolation meaning changes with the number or ordering of channels.

### Implementation

Python role API `n4m.augmentation.drift.PolynomialBaselineDrift`; ABI 2 family `n4m_augmentation_poly_drift_{create,apply,destroy}`.

The ABI-2 implementation is the `n4m_augmentation_poly_drift_*` lifecycle in libn4m.

### Sources and provenance

https://github.com/GBeurier/nirs4all-methods/blob/main/cpp/src/core/augmentation/drift/poly_drift.h


---

_See also_: [methods index](index.md) · [interactive dashboard](../landing/dashboard.md)