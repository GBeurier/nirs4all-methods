# `aug_channel_dropout` — Channel Dropout

_Group_: **Augmentation** · _Binding_: `n4m.sklearn.ChannelDropout` · _C ABI_: `n4m_aug_channel_dropout_*`

## Description

Randomly drop individual wavelength channels.

### Parameters

| Name | Type | Default |
|------|------|---------|
| `dropout_prob` | `float` | `0.05` |
| `mode` | `str | int` | `'zero'` |
| `rng` | `Optional[PCG64]` | `None` |
| `seed` | `int` | `0` |

## Explanations

### Bibliographic source

_Standard spectroscopic operator — see the nirs4all preprocessing / augmentation handbook and the cited literature within the binding docstring._

### Mathematical principle

Randomly drop individual wavelength channels.

### Implementation

C ABI `n4m_aug_channel_dropout_*` in libn4m (create / apply / destroy lifecycle), wrapped by `n4m.sklearn.ChannelDropout`. The same numerical kernel backs every language binding.

### Usage

```python
from n4m.sklearn import ChannelDropout
op = ChannelDropout()
X_transformed = op.fit_transform(X)
```

---

_See also_: [methods index](index.md) · [interactive dashboard](../landing/dashboard.md)