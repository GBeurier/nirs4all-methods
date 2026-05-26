# `aug_mixup` — Mixup augmentation

_Group_: **Augmentation** · _Binding_: `n4m.sklearn.MixupAugmenter` · _C ABI_: `n4m_aug_mixup_*`

## Description

Batch-wise mixup augmentation.

### Parameters

| Name | Type | Default |
|------|------|---------|
| `alpha` | `float` | `0.2` |
| `rng` | `Optional[PCG64]` | `None` |
| `seed` | `int` | `0` |

## Explanations

### Bibliographic source

Zhang, H., Cisse, M., Dauphin, Y. N. & Lopez-Paz, D. (2018). *mixup: Beyond Empirical Risk Minimization*. ICLR 2018.

### Mathematical principle

Forms convex combinations of sample pairs, $\tilde{\mathbf{x}} = \lambda\mathbf{x}_i + (1-\lambda)\mathbf{x}_j$ and $\tilde{y} = \lambda y_i + (1-\lambda) y_j$ with $\lambda \sim \mathrm{Beta}(\alpha,\alpha)$, encouraging linear behaviour between training examples and regularising the calibration model.

### Implementation

C ABI `n4m_aug_mixup_*` in libn4m (create / apply / destroy lifecycle), wrapped by `n4m.sklearn.MixupAugmenter`. The same numerical kernel backs every language binding.

### Usage

```python
from n4m.sklearn import MixupAugmenter
op = MixupAugmenter()
X_transformed = op.fit_transform(X)
```

---

_See also_: [methods index](index.md) · [interactive dashboard](../landing/dashboard.md)