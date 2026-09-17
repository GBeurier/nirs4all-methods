# `pp_derivate` — Order-$d$ finite-difference derivative

_Group_: **Preprocessing** · _C ABI_: `n4m_transform_derivative_*`

## Description

Finite-difference derivative along the wavelength axis.

## Parameters

| Name | Type | Default |
|------|------|---------|
| `order` | `int` | `1` |
| `delta` | `float` | `1.0` |

## API and bindings

**C ABI (ABI 2):** [`n4m_transform_derivative_create`](https://github.com/GBeurier/nirs4all-methods/blob/main/cpp/include/n4m/transform/smoothing.h#L35) · [`n4m_transform_derivative_destroy`](https://github.com/GBeurier/nirs4all-methods/blob/main/cpp/include/n4m/transform/smoothing.h#L37) · [`n4m_transform_derivative_fit`](https://github.com/GBeurier/nirs4all-methods/blob/main/cpp/include/n4m/transform/smoothing.h#L38) · [`n4m_transform_derivative_output_cols`](https://github.com/GBeurier/nirs4all-methods/blob/main/cpp/include/n4m/transform/smoothing.h#L30) · [`n4m_transform_derivative_transform`](https://github.com/GBeurier/nirs4all-methods/blob/main/cpp/include/n4m/transform/smoothing.h#L40). Use the linked public header for the exact signature, configuration, and result handles.

**Python (verified public re-export):**

```python
from n4m.transform.smoothing import Derivate
```

Source signature: [`Derivate(order: int = 1, delta: float = 1.0)`](https://github.com/GBeurier/nirs4all-methods/blob/main/bindings/python/src/n4m/_impl/preprocessing.py#L245).

**R:** no current source-verified entry point was found for this catalog method.

**MATLAB / Octave:** no current source-verified entry point was found for this catalog method.

## Explanations

### Bibliographic source

No unique canonical paper: this is repeated forward finite differencing. For the spectroscopic rationale see Norris & Williams (1984), *Optimization of mathematical treatments of raw near-infrared signal*.

### Mathematical principle

The operator applies $d$ successive first differences along wavelength and divides by `delta` at each pass, reducing the output width by `order`. A first difference removes constant offsets; a second also suppresses linear trends.

### Appropriate uses

Emphasizing slopes or narrow bands when a reduced-width difference representation is acceptable.

### Limits and validation

Differencing amplifies high-frequency noise, shortens the feature axis, and assumes uniform spacing represented by `delta`. It is distinct from the shape-preserving gradient operators.

### Implementation

`n4m.transform.smoothing.Derivate` calls `n4m_transform_derivative_*`; repeated difference kernels are in `preprocessing/derivatives/derivate.c`.

The ABI-2 implementation is the `n4m_transform_derivative_*` lifecycle in libn4m.

### Sources and provenance

https://github.com/GBeurier/nirs4all-methods/blob/main/cpp/src/core/preprocessing/derivatives/derivate.c


---

_See also_: [methods index](index.md) · [interactive dashboard](../landing/dashboard.md)