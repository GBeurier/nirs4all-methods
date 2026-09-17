# `aug_random_x_op` — Independent random elementwise arithmetic

_Group_: **Augmentation** · _C ABI_: `n4m_augmentation_random_x_op_*`

## Description

Random element-wise multiply/add/subtract operation.

## Parameters

| Name | Type | Default |
|------|------|---------|
| `op_kind` | `str \| int` | `'multiply'` |
| `operator_range_min` | `float` | `0.97` |
| `operator_range_max` | `float` | `1.03` |
| `rng` | `Optional[PCG64]` | `None` |
| `seed` | `int` | `0` |

## API and bindings

**C ABI (ABI 2):** [`n4m_augmentation_random_x_op_apply`](https://github.com/GBeurier/nirs4all-methods/blob/main/cpp/include/n4m/augmentation/mixup.h#L54) · [`n4m_augmentation_random_x_op_create`](https://github.com/GBeurier/nirs4all-methods/blob/main/cpp/include/n4m/augmentation/mixup.h#L48) · [`n4m_augmentation_random_x_op_destroy`](https://github.com/GBeurier/nirs4all-methods/blob/main/cpp/include/n4m/augmentation/mixup.h#L57). Use the linked public header for the exact signature, configuration, and result handles.

**Python (verified public re-export):**

```python
from n4m.augmentation.mixup import RandomXOperation
```

Source signature: [`RandomXOperation(op_kind: str | int = 'multiply', operator_range_min: float = 0.97, operator_range_max: float = 1.03, rng: Optional[PCG64] = None, seed: int = 0)`](https://github.com/GBeurier/nirs4all-methods/blob/main/bindings/python/src/n4m/_impl/augmentation.py#L1208).

**R:** no current source-verified entry point was found for this catalog method.

**MATLAB / Octave:** no current source-verified entry point was found for this catalog method.

## Explanations

### Bibliographic source

No canonical paper defines this operator. It is a deliberately simple internal perturbation baseline; the native source is the algorithmic specification.

### Mathematical principle

Draw one operand $u_{ij}\sim U(r_{min},r_{max})$ per cell and apply the selected `op_kind`: $X'_{ij}=X_{ij}u_{ij}$, $X_{ij}+u_{ij}$, or $X_{ij}-u_{ij}$. Results are clipped to the finite float32 range even though computation uses doubles.

### Appropriate uses

A generic sensitivity baseline for small unstructured amplitude perturbations.

### Limits and validation

Independent per-channel operands usually lack spectroscopic smoothness. Float32 clipping is a parity behavior, not an instrument saturation model.

### Implementation

Python role API `n4m.augmentation.mixup.RandomXOperation`; ABI 2 family `n4m_augmentation_random_x_op_{create,apply,destroy}`.

The ABI-2 implementation is the `n4m_augmentation_random_x_op_*` lifecycle in libn4m.

### Sources and provenance

https://github.com/GBeurier/nirs4all-methods/blob/main/cpp/src/core/augmentation/random/random_x_op.h


---

_See also_: [methods index](index.md) · [interactive dashboard](../landing/dashboard.md)