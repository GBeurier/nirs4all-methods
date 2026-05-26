# `interval_generator` — Interval Generator

_Group_: **Signal transforms** · _Binding_: `n4m.sklearn.IntervalGenerator` · _C ABI_: `n4m_interval_generator_*`

## Description

Generate fixed or overlapping wavelength intervals.

### Parameters

| Name | Type | Default |
|------|------|---------|
| `interval_size` | `int` | `32` |
| `step` | `int | None` | `None` |

## Explanations

### Bibliographic source

_Standard spectroscopic operator — see the nirs4all preprocessing / augmentation handbook and the cited literature within the binding docstring._

### Mathematical principle

Generate fixed or overlapping wavelength intervals.

### Implementation

C ABI `n4m_interval_generator_*` in libn4m (create / apply / destroy lifecycle), wrapped by `n4m.sklearn.IntervalGenerator`. The same numerical kernel backs every language binding.

### Usage

```python
from n4m.sklearn import IntervalGenerator
op = IntervalGenerator()
X_transformed = op.fit_transform(X)
```

---

_See also_: [methods index](index.md) · [interactive dashboard](../landing/dashboard.md)