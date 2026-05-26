# `aug_instrument_broaden` — Instrumental Broadening Augmenter

_Group_: **Augmentation** · _Binding_: `n4m.sklearn.InstrumentalBroadeningAugmenter` · _C ABI_: `n4m_aug_instrument_broaden_*`

## Description

Instrumental spectral broadening via Gaussian convolution.

### Parameters

| Name | Type | Default |
|------|------|---------|
| `fwhm` | `float` | `5.0` |
| `use_fwhm_range` | `bool` | `False` |
| `fwhm_low` | `float` | `3.0` |
| `fwhm_high` | `float` | `8.0` |
| `variation_scope` | `int` | `0` |
| `wavelengths` | `—` | `None` |
| `rng` | `Optional[PCG64]` | `None` |
| `seed` | `int` | `0` |

## Explanations

### Bibliographic source

_Standard spectroscopic operator — see the nirs4all preprocessing / augmentation handbook and the cited literature within the binding docstring._

### Mathematical principle

Instrumental spectral broadening via Gaussian convolution.

### Implementation

C ABI `n4m_aug_instrument_broaden_*` in libn4m (create / apply / destroy lifecycle), wrapped by `n4m.sklearn.InstrumentalBroadeningAugmenter`. The same numerical kernel backs every language binding.

### Usage

```python
from n4m.sklearn import InstrumentalBroadeningAugmenter
op = InstrumentalBroadeningAugmenter()
X_transformed = op.fit_transform(X)
```

---

_See also_: [methods index](index.md) · [interactive dashboard](../landing/dashboard.md)