# `aug_band_mask` — Band Masking

_Group_: **Augmentation** · _Binding_: `n4m.sklearn.BandMasking` · _C ABI_: `n4m_aug_band_mask_*`

## Description

Mask random spectral bands with zero-fill or interpolation.

### Parameters

| Name | Type | Default |
|------|------|---------|
| `n_bands_lo` | `int` | `1` |
| `n_bands_hi` | `int` | `3` |
| `bw_lo` | `int` | `5` |
| `bw_hi` | `int` | `15` |
| `mode` | `str | int` | `'zero'` |
| `rng` | `Optional[PCG64]` | `None` |
| `seed` | `int` | `0` |

## Explanations

### Bibliographic source

_Standard spectroscopic operator — see the nirs4all preprocessing / augmentation handbook and the cited literature within the binding docstring._

### Mathematical principle

Mask random spectral bands with zero-fill or interpolation.

### Implementation

C ABI `n4m_aug_band_mask_*` in libn4m (create / apply / destroy lifecycle), wrapped by `n4m.sklearn.BandMasking`. The same numerical kernel backs every language binding.

### Usage

```python
from n4m.sklearn import BandMasking
op = BandMasking()
X_transformed = op.fit_transform(X)
```

---

_See also_: [methods index](index.md) · [interactive dashboard](../landing/dashboard.md)