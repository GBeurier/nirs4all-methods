# `pp_slope_bias` — Slope Bias Correction

_Group_: **Signal transforms** · _Binding_: `n4m.sklearn.SlopeBiasCorrection` · _C ABI_: `n4m_pp_slope_bias_*`

## Description

Linear slope/bias correction for transferred predictions.

### Parameters

_No constructor parameters._

## Explanations

### Bibliographic source

_Standard spectroscopic operator — see the nirs4all preprocessing / augmentation handbook and the cited literature within the binding docstring._

### Mathematical principle

Linear slope/bias correction for transferred predictions.

### Implementation

C ABI `n4m_pp_slope_bias_*` in libn4m (create / apply / destroy lifecycle), wrapped by `n4m.sklearn.SlopeBiasCorrection`. The same numerical kernel backs every language binding.

### Usage

```python
from n4m.sklearn import SlopeBiasCorrection
op = SlopeBiasCorrection()
X_transformed = op.fit_transform(X)
```

---

_See also_: [methods index](index.md) · [interactive dashboard](../landing/dashboard.md)