# `aug_random_x_op` — Random X Operation

_Group_: **Augmentation** · _Binding_: `n4m.sklearn.RandomXOperation` · _C ABI_: `n4m_aug_random_x_op_*`

## Description

Random element-wise multiply/add/subtract operation.

### Parameters

| Name | Type | Default |
|------|------|---------|
| `op_kind` | `str | int` | `'multiply'` |
| `operator_range_min` | `float` | `0.97` |
| `operator_range_max` | `float` | `1.03` |
| `rng` | `Optional[PCG64]` | `None` |
| `seed` | `int` | `0` |

## Explanations

### Bibliographic source

_Standard spectroscopic operator — see the nirs4all preprocessing / augmentation handbook and the cited literature within the binding docstring._

### Mathematical principle

Random element-wise multiply/add/subtract operation.

### Implementation

C ABI `n4m_aug_random_x_op_*` in libn4m (create / apply / destroy lifecycle), wrapped by `n4m.sklearn.RandomXOperation`. The same numerical kernel backs every language binding.

### Usage

```python
from n4m.sklearn import RandomXOperation
op = RandomXOperation()
X_transformed = op.fit_transform(X)
```

---

_See also_: [methods index](index.md) · [interactive dashboard](../landing/dashboard.md)