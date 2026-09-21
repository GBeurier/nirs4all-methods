# `pp_iasls` — Improved asymmetric least-squares baseline correction (IAsLS)

_Group_: **Baseline correction** · _C ABI_: `n4m_transform_iasls_*`

## Description

Improved asymmetric least-squares baseline correction.

## Parameters

| Name | Type | Default |
|------|------|---------|
| `lam` | `float` | `1000000.0` |
| `p` | `float` | `0.01` |
| `lam_1` | `float` | `0.0001` |
| `polyorder` | `int` | `2` |
| `diff_order` | `int` | `2` |
| `max_iter` | `int` | `50` |
| `tol` | `float` | `0.001` |

## API and bindings

**C ABI (ABI 2):** [`n4m_transform_iasls_create`](https://github.com/GBeurier/nirs4all-methods/blob/main/cpp/include/n4m/transform/baseline.h#L129) · [`n4m_transform_iasls_create_ex`](https://github.com/GBeurier/nirs4all-methods/blob/main/cpp/include/n4m/transform/baseline.h#L133) · [`n4m_transform_iasls_destroy`](https://github.com/GBeurier/nirs4all-methods/blob/main/cpp/include/n4m/transform/baseline.h#L139) · [`n4m_transform_iasls_transform`](https://github.com/GBeurier/nirs4all-methods/blob/main/cpp/include/n4m/transform/baseline.h#L140). Use the linked public header for the exact signature, configuration, and result handles.

**Python (verified public re-export):**

```python
from n4m.transform.baseline import IAsLS
```

Source signature: [`IAsLS(lam: float = 1000000.0, p: float = 0.01, lam_1: float = 0.0001, polyorder: int = 2, diff_order: int = 2, max_iter: int = 50, tol: float = 0.001)`](https://github.com/GBeurier/nirs4all-methods/blob/main/bindings/python/src/n4m/_impl/baseline.py#L201).

**R:** no current source-verified entry point was found for this catalog method.

**MATLAB / Octave:** no current source-verified entry point was found for this catalog method.

## Explanations

### Bibliographic source

He et al. (2014), *Baseline correction for Raman spectra using an improved asymmetric least squares method*, Analytical Methods 6, 4402–4407, https://doi.org/10.1039/C4AY00068D.

### Mathematical principle

IAsLS augments asymmetric Whittaker fitting with a derivative fidelity penalty controlled by `lam_1`; it estimates an initial polynomial background, then iterates asymmetric weights while penalizing the configured baseline difference order.

### Appropriate uses

Baseline correction when ordinary AsLS leaves peak-dependent distortion and a derivative-aware smoothness term is useful.

### Limits and validation

The result is sensitive to two penalties, asymmetry, polynomial order, and stopping criteria. Published IAsLS variants differ, so this page describes the shipped kernel.

### Implementation

`n4m.transform.baseline.IAsLS` uses `n4m_transform_iasls_*`; the exact polynomial initialization and weighted solve are in `preprocessing/baselines/iasls.c`.

The ABI-2 implementation is the `n4m_transform_iasls_*` lifecycle in libn4m.

### Sources and provenance

https://doi.org/10.1039/C4AY00068D; https://github.com/GBeurier/nirs4all-methods/blob/main/cpp/src/core/preprocessing/baselines/iasls.c


---

_See also_: [methods index](index.md) · [interactive dashboard](../landing/dashboard.md)