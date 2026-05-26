# `aug_wavelength_stretch` — Wavelength Stretch

_Group_: **Augmentation** · _Binding_: `n4m.sklearn.WavelengthStretch` · _C ABI_: `n4m_aug_wavelength_stretch_*`

## Description

Random wavelength-axis stretching.

### Parameters

| Name | Type | Default |
|------|------|---------|
| `stretch_lo` | `float` | `0.99` |
| `stretch_hi` | `float` | `1.01` |
| `wavelengths` | `—` | `None` |
| `rng` | `Optional[PCG64]` | `None` |
| `seed` | `int` | `0` |

## Explanations

### Bibliographic source

_Standard spectroscopic operator — see the nirs4all preprocessing / augmentation handbook and the cited literature within the binding docstring._

### Mathematical principle

Random wavelength-axis stretching.

### Implementation

C ABI `n4m_aug_wavelength_stretch_*` in libn4m (create / apply / destroy lifecycle), wrapped by `n4m.sklearn.WavelengthStretch`. The same numerical kernel backs every language binding.

### Usage

```python
from n4m.sklearn import WavelengthStretch
op = WavelengthStretch()
X_transformed = op.fit_transform(X)
```

---

_See also_: [methods index](index.md) · [interactive dashboard](../landing/dashboard.md)