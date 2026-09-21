# `pp_arpls` — Asymmetrically reweighted penalized least squares (arPLS)

_Group_: **Baseline correction** · _C ABI_: `n4m_transform_arpls_*`

## Description

Asymmetrically reweighted penalized least squares.

## Parameters

| Name | Type | Default |
|------|------|---------|
| `lam` | `float` | `100000.0` |
| `max_iter` | `int` | `50` |
| `tol` | `float` | `0.001` |

## API and bindings

**C ABI (ABI 2):** [`n4m_transform_arpls_create`](https://github.com/GBeurier/nirs4all-methods/blob/main/cpp/include/n4m/transform/baseline.h#L55) · [`n4m_transform_arpls_destroy`](https://github.com/GBeurier/nirs4all-methods/blob/main/cpp/include/n4m/transform/baseline.h#L58) · [`n4m_transform_arpls_transform`](https://github.com/GBeurier/nirs4all-methods/blob/main/cpp/include/n4m/transform/baseline.h#L59). Use the linked public header for the exact signature, configuration, and result handles.

**Python (verified public re-export):**

```python
from n4m.transform.baseline import ArPLS
```

Source signature: [`ArPLS(lam: float = 100000.0, max_iter: int = 50, tol: float = 0.001)`](https://github.com/GBeurier/nirs4all-methods/blob/main/bindings/python/src/n4m/_impl/baseline.py#L83).

**R:** no current source-verified entry point was found for this catalog method.

**MATLAB / Octave:** no current source-verified entry point was found for this catalog method.

## Explanations

### Bibliographic source

Baek et al. (2015), *Baseline correction using asymmetrically reweighted penalized least squares smoothing*, Analyst 140, 250–257, https://doi.org/10.1039/C4AN01061B.

### Mathematical principle

arPLS alternates a Whittaker smoothness solve with a logistic update of weights estimated from negative residual statistics. Positive peak residuals are progressively downweighted without requiring a fixed asymmetry parameter.

### Appropriate uses

Automatic baseline removal for spectra with positive peaks and slowly varying backgrounds, especially when a fixed AsLS asymmetry is hard to choose.

### Limits and validation

The negative-residual distribution must represent baseline noise. Very broad bands, negative peaks, or an ill-chosen smoothness penalty bias the estimate.

### Implementation

`n4m.transform.baseline.ArPLS` calls `n4m_transform_arpls_*`; the iterative reweighting and penalized solver are in `preprocessing/baselines/arpls.c`.

The ABI-2 implementation is the `n4m_transform_arpls_*` lifecycle in libn4m.

### Sources and provenance

https://doi.org/10.1039/C4AN01061B; https://github.com/GBeurier/nirs4all-methods/blob/main/cpp/src/core/preprocessing/baselines/arpls.c


---

_See also_: [methods index](index.md) · [interactive dashboard](../landing/dashboard.md)