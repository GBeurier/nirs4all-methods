# `aug_wavelength_shift` — Wavelength Shift

_Group_: **Augmentation** · _Binding_: `n4m.sklearn.WavelengthShift` · _C ABI_: `n4m_aug_wavelength_shift_*`

## Description

Random spectral shift with linear interpolation.

### Parameters

| Name | Type | Default |
|------|------|---------|
| `shift_lo` | `float` | `-1.0` |
| `shift_hi` | `float` | `1.0` |
| `wavelengths` | `—` | `None` |
| `rng` | `Optional[PCG64]` | `None` |
| `seed` | `int` | `0` |

## Explanations

### Bibliographic source

_Standard spectroscopic operator — see the nirs4all preprocessing / augmentation handbook and the cited literature within the binding docstring._

### Mathematical principle

Random spectral shift with linear interpolation.

### Implementation

C ABI `n4m_aug_wavelength_shift_*` in libn4m (create / apply / destroy lifecycle), wrapped by `n4m.sklearn.WavelengthShift`. The same numerical kernel backs every language binding.

### Usage

```python
from n4m.sklearn import WavelengthShift
op = WavelengthShift()
X_transformed = op.fit_transform(X)
```

---

_See also_: [methods index](index.md) · [interactive dashboard](../landing/dashboard.md)