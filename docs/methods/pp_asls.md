# `pp_asls` — Asymmetric least-squares baseline correction (AsLS)

_Group_: **Baseline correction** · _C ABI_: `n4m_transform_asls_*`

## Description

Asymmetric Least Squares (Eilers & Boelens 2005).

## Parameters

| Name | Type | Default |
|------|------|---------|
| `lam` | `float` | `1000000.0` |
| `p` | `float` | `0.01` |
| `max_iter` | `int` | `50` |
| `tol` | `float` | `0.001` |

## API and bindings

**C ABI (ABI 2):** [`n4m_transform_asls_create`](https://github.com/GBeurier/nirs4all-methods/blob/main/cpp/include/n4m/transform/baseline.h#L28) · [`n4m_transform_asls_destroy`](https://github.com/GBeurier/nirs4all-methods/blob/main/cpp/include/n4m/transform/baseline.h#L31) · [`n4m_transform_asls_transform`](https://github.com/GBeurier/nirs4all-methods/blob/main/cpp/include/n4m/transform/baseline.h#L32). Use the linked public header for the exact signature, configuration, and result handles.

**Python (verified public re-export):**

```python
from n4m.transform.baseline import AsLS
```

Source signature: [`AsLS(lam: float = 1000000.0, p: float = 0.01, max_iter: int = 50, tol: float = 0.001)`](https://github.com/GBeurier/nirs4all-methods/blob/main/bindings/python/src/n4m/_impl/baseline.py#L30).

**R:** no current source-verified entry point was found for this catalog method.

**MATLAB / Octave:** no current source-verified entry point was found for this catalog method.

## Explanations

### Bibliographic source

Eilers & Boelens (2005), *Baseline correction with asymmetric least squares smoothing*, Leiden University Medical Centre technical report; algorithmic source also summarized at https://doi.org/10.1039/C4AN01061B.

### Mathematical principle

AsLS alternates minimization of $\sum_j w_j(y_j-z_j)^2+\lambda\|D^2z\|^2$ with $w_j=p$ for positive residuals and $w_j=1-p$ otherwise. Subtracting the smooth $z$ preserves narrow peaks while suppressing background curvature.

### Appropriate uses

Baseline correction for Raman/NIR-like traces when the sign of analyte peaks is known and a controllable smoothness/asymmetry trade-off is wanted.

### Limits and validation

Both $\lambda$ and $p$ require tuning; broad peaks may enter the baseline and the binary residual weighting can converge slowly near ambiguous points.

### Implementation

`n4m.transform.baseline.AsLS` wraps `n4m_transform_asls_*`; the Whittaker system and weight updates live in `preprocessing/baselines/asls.c`.

The ABI-2 implementation is the `n4m_transform_asls_*` lifecycle in libn4m.

### Sources and provenance

https://github.com/GBeurier/nirs4all-methods/blob/main/cpp/src/core/preprocessing/baselines/asls.c; https://doi.org/10.1039/C4AN01061B


---

_See also_: [methods index](index.md) · [interactive dashboard](../landing/dashboard.md)