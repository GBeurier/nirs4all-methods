# `pp_localized_msc` — Localized M S C

_Group_: **Signal transforms** · _Binding_: `n4m.sklearn.LocalizedMSC` · _C ABI_: `n4m_pp_localized_msc_*`

## Description

Feature-wise MSC using a moving local wavelength window.

### Parameters

| Name | Type | Default |
|------|------|---------|
| `window_size` | `int` | `32` |
| `reference` | `—` | `None` |
| `eps` | `float` | `1e-12` |

## Explanations

### Bibliographic source

_Standard spectroscopic operator — see the nirs4all preprocessing / augmentation handbook and the cited literature within the binding docstring._

### Mathematical principle

Feature-wise MSC using a moving local wavelength window.

### Implementation

C ABI `n4m_pp_localized_msc_*` in libn4m (create / apply / destroy lifecycle), wrapped by `n4m.sklearn.LocalizedMSC`. The same numerical kernel backs every language binding.

### Usage

```python
from n4m.sklearn import LocalizedMSC
op = LocalizedMSC()
X_transformed = op.fit_transform(X)
```

---

_See also_: [methods index](index.md) · [interactive dashboard](../landing/dashboard.md)