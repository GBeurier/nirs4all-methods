# `aug_scatter_sim` — Scatter Simulation M S C

_Group_: **Augmentation** · _Binding_: `n4m.sklearn.ScatterSimulationMSC` · _C ABI_: `n4m_aug_scatter_sim_*`

## Description

MSC-style multiplicative/additive scatter simulation.

### Parameters

| Name | Type | Default |
|------|------|---------|
| `a_low` | `float` | `-0.05` |
| `a_high` | `float` | `0.05` |
| `b_low` | `float` | `0.9` |
| `b_high` | `float` | `1.1` |
| `rng` | `Optional[PCG64]` | `None` |
| `seed` | `int` | `0` |

## Explanations

### Bibliographic source

_Standard spectroscopic operator — see the nirs4all preprocessing / augmentation handbook and the cited literature within the binding docstring._

### Mathematical principle

MSC-style multiplicative/additive scatter simulation.

### Implementation

C ABI `n4m_aug_scatter_sim_*` in libn4m (create / apply / destroy lifecycle), wrapped by `n4m.sklearn.ScatterSimulationMSC`. The same numerical kernel backs every language binding.

### Usage

```python
from n4m.sklearn import ScatterSimulationMSC
op = ScatterSimulationMSC()
X_transformed = op.fit_transform(X)
```

---

_See also_: [methods index](index.md) · [interactive dashboard](../landing/dashboard.md)