# `pp_airpls` — Adaptive iteratively reweighted penalized least squares (airPLS)

_Group_: **Baseline correction** · _C ABI_: `n4m_transform_airpls_*`

## Description

Adaptive iteratively reweighted PLS (Zhang 2010).

## Parameters

| Name | Type | Default |
|------|------|---------|
| `lam` | `float` | `1000000.0` |
| `max_iter` | `int` | `50` |
| `tol` | `float` | `0.001` |

## API and bindings

**C ABI (ABI 2):** [`n4m_transform_airpls_create`](https://github.com/GBeurier/nirs4all-methods/blob/main/cpp/include/n4m/transform/baseline.h#L41) · [`n4m_transform_airpls_destroy`](https://github.com/GBeurier/nirs4all-methods/blob/main/cpp/include/n4m/transform/baseline.h#L44) · [`n4m_transform_airpls_transform`](https://github.com/GBeurier/nirs4all-methods/blob/main/cpp/include/n4m/transform/baseline.h#L45). Use the linked public header for the exact signature, configuration, and result handles.

**Python (verified public re-export):**

```python
from n4m.transform.baseline import AirPLS
```

Source signature: [`AirPLS(lam: float = 1000000.0, max_iter: int = 50, tol: float = 0.001)`](https://github.com/GBeurier/nirs4all-methods/blob/main/bindings/python/src/n4m/_impl/baseline.py#L58).

**R:** no current source-verified entry point was found for this catalog method.

**MATLAB / Octave:** no current source-verified entry point was found for this catalog method.

## Explanations

### Bibliographic source

Zhang, Chen & Liang (2010), *Baseline correction using adaptive iteratively reweighted penalized least squares*, Analyst 135, 1138–1146, https://doi.org/10.1039/B922045C.

### Mathematical principle

For a spectrum $y$, airPLS repeatedly solves a Whittaker problem $\min_z \|W^{1/2}(y-z)\|_2^2+\lambda\|D^2z\|_2^2$. Points above the current baseline receive zero weight; negative residuals receive exponentially increasing weights until their mass is small or the iteration budget is spent.

### Appropriate uses

Removal of smooth fluorescence or background drift when peaks are expected to lie mainly above the baseline.

### Limits and validation

The result depends strongly on $\lambda$ and iteration stopping. Broad, dense, or negative bands can be mistaken for baseline; it is not a physical scatter model.

### Implementation

`n4m.transform.baseline.AirPLS` wraps the ABI-2 `n4m_transform_airpls_*` lifecycle; the numerical loop is in `cpp/src/core/preprocessing/baselines/airpls.c`.

The ABI-2 implementation is the `n4m_transform_airpls_*` lifecycle in libn4m.

### Sources and provenance

https://doi.org/10.1039/B922045C; https://github.com/GBeurier/nirs4all-methods/blob/main/cpp/src/core/preprocessing/baselines/airpls.c


---

_See also_: [methods index](index.md) · [interactive dashboard](../landing/dashboard.md)