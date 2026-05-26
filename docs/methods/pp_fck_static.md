# `pp_fck_static` — F C K Static Transformer

_Group_: **Feature extraction** · _Binding_: `n4m.sklearn.FCKStaticTransformer` · _C ABI_: `n4m_pp_fck_static_*`

## Description

Static fractional convolutional kernel bank transformer.

### Parameters

| Name | Type | Default |
|------|------|---------|
| `kernel_size` | `int` | `—` |
| `alphas` | `Sequence[float] | None` | `None` |
| `sigmas` | `Sequence[float] | None` | `None` |

## Explanations

### Bibliographic source

_Standard spectroscopic operator — see the nirs4all preprocessing / augmentation handbook and the cited literature within the binding docstring._

### Mathematical principle

Static fractional convolutional kernel bank transformer.

### Implementation

C ABI `n4m_pp_fck_static_*` in libn4m (create / apply / destroy lifecycle), wrapped by `n4m.sklearn.FCKStaticTransformer`. The same numerical kernel backs every language binding.

### Usage

```python
from n4m.sklearn import FCKStaticTransformer
op = FCKStaticTransformer()
X_transformed = op.fit_transform(X)
```

---

_See also_: [methods index](index.md) · [interactive dashboard](../landing/dashboard.md)