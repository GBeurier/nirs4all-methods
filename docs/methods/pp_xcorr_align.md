# `pp_xcorr_align` — Cross Correlation Alignment

_Group_: **Signal transforms** · _Binding_: `n4m.sklearn.CrossCorrelationAlignment` · _C ABI_: `n4m_pp_xcorr_align_*`

## Description

Whole-spectrum integer shift chosen by maximum correlation.

### Parameters

| Name | Type | Default |
|------|------|---------|
| `reference` | `—` | `None` |
| `max_shift` | `int` | `5` |

## Explanations

### Bibliographic source

_Standard spectroscopic operator — see the nirs4all preprocessing / augmentation handbook and the cited literature within the binding docstring._

### Mathematical principle

Whole-spectrum integer shift chosen by maximum correlation.

### Implementation

C ABI `n4m_pp_xcorr_align_*` in libn4m (create / apply / destroy lifecycle), wrapped by `n4m.sklearn.CrossCorrelationAlignment`. The same numerical kernel backs every language binding.

### Usage

```python
from n4m.sklearn import CrossCorrelationAlignment
op = CrossCorrelationAlignment()
X_transformed = op.fit_transform(X)
```

---

_See also_: [methods index](index.md) · [interactive dashboard](../landing/dashboard.md)