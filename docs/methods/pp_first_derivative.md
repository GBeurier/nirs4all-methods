# `pp_first_derivative` — Shape-preserving first numerical derivative

_Group_: **Preprocessing** · _C ABI_: `n4m_transform_first_derivative_*`

## Description

``np.gradient(X, delta, axis=1, edge_order=...)`` (shape-preserving).

## Parameters

| Name | Type | Default |
|------|------|---------|
| `delta` | `float` | `1.0` |
| `edge_order` | `int` | `2` |

## API and bindings

**C ABI (ABI 2):** [`n4m_transform_first_derivative_create`](https://github.com/GBeurier/nirs4all-methods/blob/main/cpp/include/n4m/transform/smoothing.h#L94) · [`n4m_transform_first_derivative_destroy`](https://github.com/GBeurier/nirs4all-methods/blob/main/cpp/include/n4m/transform/smoothing.h#L96) · [`n4m_transform_first_derivative_transform`](https://github.com/GBeurier/nirs4all-methods/blob/main/cpp/include/n4m/transform/smoothing.h#L98). Use the linked public header for the exact signature, configuration, and result handles.

**Python (verified public re-export):**

```python
from n4m.transform.smoothing import FirstDerivative
```

Source signature: [`FirstDerivative(delta: float = 1.0, edge_order: int = 2)`](https://github.com/GBeurier/nirs4all-methods/blob/main/bindings/python/src/n4m/_impl/preprocessing.py#L393).

**R:** no current source-verified entry point was found for this catalog method.

**MATLAB / Octave:** no current source-verified entry point was found for this catalog method.

## Explanations

### Bibliographic source

No unique paper defines `numpy.gradient`; the spectroscopic use of derivatives is discussed by Norris & Williams (1984), *Optimization of mathematical treatments of raw near-infrared signal*.

### Mathematical principle

The operator reproduces a first `numpy.gradient` pass along each row: centered differences inside and one-sided formulas of edge order one or two at boundaries, all divided by `delta`. Output shape equals input shape.

### Appropriate uses

Suppressing constant offsets and sharpening overlapping bands when retaining the original number of channels is useful.

### Limits and validation

It amplifies noise and assumes uniform channel spacing. Boundary samples use different stencils, and `edge_order=2` needs enough wavelengths.

### Implementation

`n4m.transform.smoothing.FirstDerivative` wraps `n4m_transform_first_derivative_*`; stencils are in `preprocessing/derivatives/first_derivative.c`.

The ABI-2 implementation is the `n4m_transform_first_derivative_*` lifecycle in libn4m.

### Sources and provenance

https://github.com/GBeurier/nirs4all-methods/blob/main/cpp/src/core/preprocessing/derivatives/first_derivative.c; https://numpy.org/doc/stable/reference/generated/numpy.gradient.html


---

_See also_: [methods index](index.md) · [interactive dashboard](../landing/dashboard.md)