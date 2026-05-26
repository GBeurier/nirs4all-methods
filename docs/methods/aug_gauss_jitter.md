# `aug_gauss_jitter` — Gaussian Jitter

_Group_: **Augmentation** · _Binding_: `n4m.sklearn.GaussianJitter` · _C ABI_: `n4m_aug_gauss_jitter_*`

## Description

Gaussian smoothing jitter.

### Parameters

| Name | Type | Default |
|------|------|---------|
| `sigma_lo` | `float` | `0.5` |
| `sigma_hi` | `float` | `1.5` |
| `kernel_width` | `int` | `9` |
| `rng` | `Optional[PCG64]` | `None` |
| `seed` | `int` | `0` |

## Explanations

### Bibliographic source

_Standard spectroscopic operator — see the nirs4all preprocessing / augmentation handbook and the cited literature within the binding docstring._

### Mathematical principle

Gaussian smoothing jitter.

### Implementation

C ABI `n4m_aug_gauss_jitter_*` in libn4m (create / apply / destroy lifecycle), wrapped by `n4m.sklearn.GaussianJitter`. The same numerical kernel backs every language binding.

### Usage

```python
from n4m.sklearn import GaussianJitter
op = GaussianJitter()
X_transformed = op.fit_transform(X)
```

---

_See also_: [methods index](index.md) · [interactive dashboard](../landing/dashboard.md)