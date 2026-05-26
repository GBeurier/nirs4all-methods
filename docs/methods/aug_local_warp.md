# `aug_local_warp` — Local Warp Augmenter

_Group_: **Augmentation** · _Binding_: `n4m.sklearn.LocalWarpAugmenter` · _C ABI_: `n4m_aug_local_warp_*`

## Description

Random local wavelength warping.

### Parameters

| Name | Type | Default |
|------|------|---------|
| `n_control_points` | `int` | `5` |
| `max_shift` | `float` | `1.0` |
| `wavelengths` | `—` | `None` |
| `rng` | `Optional[PCG64]` | `None` |
| `seed` | `int` | `0` |

## Explanations

### Bibliographic source

_Standard spectroscopic operator — see the nirs4all preprocessing / augmentation handbook and the cited literature within the binding docstring._

### Mathematical principle

Random local wavelength warping.

### Implementation

C ABI `n4m_aug_local_warp_*` in libn4m (create / apply / destroy lifecycle), wrapped by `n4m.sklearn.LocalWarpAugmenter`. The same numerical kernel backs every language binding.

### Usage

```python
from n4m.sklearn import LocalWarpAugmenter
op = LocalWarpAugmenter()
X_transformed = op.fit_transform(X)
```

---

_See also_: [methods index](index.md) · [interactive dashboard](../landing/dashboard.md)