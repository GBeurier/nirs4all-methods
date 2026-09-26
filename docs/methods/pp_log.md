# `pp_log` — Element-wise logarithmic transform

_Group_: **Preprocessing** · _C ABI_: `n4m_transform_log_transform_*`

## Description

Element-wise logarithm with optional fit-time auto-offset.

## Parameters

| Name | Type | Default |
|------|------|---------|
| `base` | `float` | `0.0` |
| `offset` | `float` | `0.0` |
| `auto_offset` | `bool` | `True` |
| `min_value` | `float` | `1e-08` |

## API and bindings

**C ABI (ABI 2):** [`n4m_transform_log_transform_create`](https://github.com/GBeurier/nirs4all-methods/blob/main/cpp/include/n4m/transform/scaling.h#L57) · [`n4m_transform_log_transform_destroy`](https://github.com/GBeurier/nirs4all-methods/blob/main/cpp/include/n4m/transform/scaling.h#L60) · [`n4m_transform_log_transform_fit`](https://github.com/GBeurier/nirs4all-methods/blob/main/cpp/include/n4m/transform/scaling.h#L47) · [`n4m_transform_log_transform_is_fitted`](https://github.com/GBeurier/nirs4all-methods/blob/main/cpp/include/n4m/transform/scaling.h#L63) · [`n4m_transform_log_transform_transform`](https://github.com/GBeurier/nirs4all-methods/blob/main/cpp/include/n4m/transform/scaling.h#L65). Use the linked public header for the exact signature, configuration, and result handles.

**Python (verified public re-export):**

```python
from n4m.transform.scaling import LogTransform
```

Source signature: [`LogTransform(base: float = 0.0, offset: float = 0.0, auto_offset: bool = True, min_value: float = 1e-08)`](https://github.com/GBeurier/nirs4all-methods/blob/main/bindings/python/src/n4m/_impl/preprocessing.py#L170).

**R:** no current source-verified entry point was found for this catalog method.

**MATLAB / Octave:** no current source-verified entry point was found for this catalog method.

## Explanations

### Bibliographic source

No canonical paper: this is the mathematical logarithm with an implementation-defined offset policy.

### Mathematical principle

Transform evaluates $\log_b(x+c)$, using the natural logarithm when `base=0`. With `auto_offset`, fit chooses an offset that moves the training minimum to at least `min_value`; the same offset is reused on later data.

### Appropriate uses

Compressing right-skewed positive intensities or expressing multiplicative changes on an additive scale.

### Limits and validation

Nonpositive shifted values are invalid. A data-derived offset changes interpretation and must be fit without leakage; logs are not interchangeable with physical absorbance unless the correct sign and base are used.

### Implementation

`n4m.transform.scaling.LogTransform` uses `n4m_transform_log_transform_*`; offset fitting and base conversion are in `preprocessing/scaling/log_transform.c`.

The ABI-2 implementation is the `n4m_transform_log_transform_*` lifecycle in libn4m.

### Sources and provenance

https://github.com/GBeurier/nirs4all-methods/blob/main/cpp/src/core/preprocessing/scaling/log_transform.c


---

_See also_: [methods index](index.md) · [interactive dashboard](../landing/dashboard.md)