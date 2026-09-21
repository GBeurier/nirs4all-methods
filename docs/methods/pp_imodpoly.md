# `pp_imodpoly` — Improved modified-polynomial baseline correction (IModPoly)

_Group_: **Baseline correction** · _C ABI_: `n4m_transform_imodpoly_*`

## Description

Improved modified polynomial baseline correction.

## Parameters

| Name | Type | Default |
|------|------|---------|
| `polyorder` | `int` | `2` |
| `max_iter` | `int` | `250` |
| `tol` | `float` | `0.001` |

## API and bindings

**C ABI (ABI 2):** [`n4m_transform_imodpoly_create`](https://github.com/GBeurier/nirs4all-methods/blob/main/cpp/include/n4m/transform/baseline.h#L82) · [`n4m_transform_imodpoly_destroy`](https://github.com/GBeurier/nirs4all-methods/blob/main/cpp/include/n4m/transform/baseline.h#L85) · [`n4m_transform_imodpoly_transform`](https://github.com/GBeurier/nirs4all-methods/blob/main/cpp/include/n4m/transform/baseline.h#L86). Use the linked public header for the exact signature, configuration, and result handles.

**Python (verified public re-export):**

```python
from n4m.transform.baseline import IModPoly
```

Source signature: [`IModPoly(polyorder: int = 2, max_iter: int = 250, tol: float = 0.001)`](https://github.com/GBeurier/nirs4all-methods/blob/main/bindings/python/src/n4m/_impl/baseline.py#L133).

**R:** no current source-verified entry point was found for this catalog method.

**MATLAB / Octave:** no current source-verified entry point was found for this catalog method.

## Explanations

### Bibliographic source

Gan, Ruan & Mo (2006), *Baseline correction by improved iterative polynomial fitting with automatic threshold*, Chemometrics and Intelligent Laboratory Systems 82, 59–65, https://doi.org/10.1016/j.chemolab.2005.08.009.

### Mathematical principle

A polynomial is repeatedly fit to a working spectrum; points identified as peaks relative to the current baseline and residual spread are replaced/downweighted before refitting. IModPoly uses a residual-noise criterion to reduce the systematic underfit of basic ModPoly.

### Appropriate uses

Automatic subtraction of smooth fluorescence backgrounds under positive Raman-like bands.

### Limits and validation

Polynomial order controls bias and edge behaviour; broad or negative peaks violate the clipping model. Iterative variants differ, so parity claims apply to this kernel.

### Implementation

`n4m.transform.baseline.IModPoly` wraps `n4m_transform_imodpoly_*`; the exact clipping and convergence rules are in `preprocessing/baselines/imodpoly.c`.

The ABI-2 implementation is the `n4m_transform_imodpoly_*` lifecycle in libn4m.

### Sources and provenance

https://doi.org/10.1016/j.chemolab.2005.08.009; https://github.com/GBeurier/nirs4all-methods/blob/main/cpp/src/core/preprocessing/baselines/imodpoly.c


---

_See also_: [methods index](index.md) · [interactive dashboard](../landing/dashboard.md)