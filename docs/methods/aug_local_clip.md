# `aug_local_clip` — Local Clip

_Group_: **Augmentation** · _Binding_: `n4m.sklearn.LocalClip` · _C ABI_: `n4m_aug_local_clip_*`

## Description

Clip random local spectral regions.

### Parameters

| Name | Type | Default |
|------|------|---------|
| `n_regions` | `int` | `1` |
| `width_lo` | `int` | `5` |
| `width_hi` | `int` | `15` |
| `rng` | `Optional[PCG64]` | `None` |
| `seed` | `int` | `0` |

## Explanations

### Bibliographic source

_Standard spectroscopic operator — see the nirs4all preprocessing / augmentation handbook and the cited literature within the binding docstring._

### Mathematical principle

Clip random local spectral regions.

### Implementation

C ABI `n4m_aug_local_clip_*` in libn4m (create / apply / destroy lifecycle), wrapped by `n4m.sklearn.LocalClip`. The same numerical kernel backs every language binding.

### Usage

```python
from n4m.sklearn import LocalClip
op = LocalClip()
X_transformed = op.fit_transform(X)
```

---

_See also_: [methods index](index.md) · [interactive dashboard](../landing/dashboard.md)