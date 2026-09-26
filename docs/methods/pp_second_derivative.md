# `pp_second_derivative` — Shape-preserving second numerical derivative

_Group_: **Preprocessing** · _C ABI_: `n4m_transform_second_derivative_*`

## Description

Two passes of ``np.gradient`` (shape-preserving).

## Parameters

| Name | Type | Default |
|------|------|---------|
| `delta` | `float` | `1.0` |
| `edge_order` | `int` | `2` |

## API and bindings

**C ABI (ABI 2):** [`n4m_transform_second_derivative_create`](https://github.com/GBeurier/nirs4all-methods/blob/main/cpp/include/n4m/transform/smoothing.h#L109) · [`n4m_transform_second_derivative_destroy`](https://github.com/GBeurier/nirs4all-methods/blob/main/cpp/include/n4m/transform/smoothing.h#L111) · [`n4m_transform_second_derivative_transform`](https://github.com/GBeurier/nirs4all-methods/blob/main/cpp/include/n4m/transform/smoothing.h#L113). Use the linked public header for the exact signature, configuration, and result handles.

**Python (verified public re-export):**

```python
from n4m.transform.smoothing import SecondDerivative
```

Source signature: [`SecondDerivative(delta: float = 1.0, edge_order: int = 2)`](https://github.com/GBeurier/nirs4all-methods/blob/main/bindings/python/src/n4m/_impl/preprocessing.py#L524).

**R:** no current source-verified entry point was found for this catalog method.

**MATLAB / Octave:** no current source-verified entry point was found for this catalog method.

## Explanations

### Bibliographic source

No unique paper defines two `numpy.gradient` passes; the spectroscopic derivative rationale follows Norris & Williams (1984), while the numerical reference is https://numpy.org/doc/stable/reference/generated/numpy.gradient.html.

### Mathematical principle

The implementation applies the shape-preserving first-gradient stencil twice along each row, using `delta` and the requested boundary `edge_order` at both passes. It approximates $d^2x/d\lambda^2$ without dropping edge columns.

### Appropriate uses

Suppressing constant and linear backgrounds and resolving overlapping absorption bands.

### Limits and validation

Second differentiation strongly amplifies high-frequency noise and assumes a uniform grid. Boundary estimates compound one-sided errors and smoothing is usually needed.

### Implementation

`n4m.transform.smoothing.SecondDerivative` calls `n4m_transform_second_derivative_*`; both gradient passes are in `preprocessing/derivatives/second_derivative.c`.

The ABI-2 implementation is the `n4m_transform_second_derivative_*` lifecycle in libn4m.

### Sources and provenance

https://numpy.org/doc/stable/reference/generated/numpy.gradient.html; https://github.com/GBeurier/nirs4all-methods/blob/main/cpp/src/core/preprocessing/derivatives/second_derivative.c


---

_See also_: [methods index](index.md) · [interactive dashboard](../landing/dashboard.md)