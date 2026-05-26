# `aug_magnitude_warp` — Magnitude Warp

_Group_: **Augmentation** · _Binding_: `n4m.sklearn.MagnitudeWarp` · _C ABI_: `n4m_aug_magnitude_warp_*`

## Description

Smooth multiplicative magnitude warp.

### Parameters

| Name | Type | Default |
|------|------|---------|
| `n_control_points` | `int` | `5` |
| `gain_lo` | `float` | `0.9` |
| `gain_hi` | `float` | `1.1` |
| `wavelengths` | `—` | `None` |
| `rng` | `Optional[PCG64]` | `None` |
| `seed` | `int` | `0` |

## Explanations

### Bibliographic source

_Standard spectroscopic operator — see the nirs4all preprocessing / augmentation handbook and the cited literature within the binding docstring._

### Mathematical principle

Smooth multiplicative magnitude warp.

### Implementation

C ABI `n4m_aug_magnitude_warp_*` in libn4m (create / apply / destroy lifecycle), wrapped by `n4m.sklearn.MagnitudeWarp`. The same numerical kernel backs every language binding.

### Usage

```python
from n4m.sklearn import MagnitudeWarp
op = MagnitudeWarp()
X_transformed = op.fit_transform(X)
```

---

_See also_: [methods index](index.md) · [interactive dashboard](../landing/dashboard.md)