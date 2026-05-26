# `aug_band_perturb` — Band Perturbation Augmenter

_Group_: **Augmentation** · _Binding_: `n4m.sklearn.BandPerturbationAugmenter` · _C ABI_: `n4m_aug_band_perturb_*`

## Description

Random band-local gain and offset perturbations.

### Parameters

| Name | Type | Default |
|------|------|---------|
| `n_bands` | `int` | `3` |
| `bw_lo` | `int` | `5` |
| `bw_hi` | `int` | `15` |
| `gain_lo` | `float` | `0.9` |
| `gain_hi` | `float` | `1.1` |
| `offset_lo` | `float` | `-0.01` |
| `offset_hi` | `float` | `0.01` |
| `rng` | `Optional[PCG64]` | `None` |
| `seed` | `int` | `0` |

## Explanations

### Bibliographic source

_Standard spectroscopic operator — see the nirs4all preprocessing / augmentation handbook and the cited literature within the binding docstring._

### Mathematical principle

Random band-local gain and offset perturbations.

### Implementation

C ABI `n4m_aug_band_perturb_*` in libn4m (create / apply / destroy lifecycle), wrapped by `n4m.sklearn.BandPerturbationAugmenter`. The same numerical kernel backs every language binding.

### Usage

```python
from n4m.sklearn import BandPerturbationAugmenter
op = BandPerturbationAugmenter()
X_transformed = op.fit_transform(X)
```

---

_See also_: [methods index](index.md) · [interactive dashboard](../landing/dashboard.md)