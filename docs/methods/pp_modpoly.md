# `pp_modpoly` — Modified-polynomial baseline correction (ModPoly)

_Group_: **Baseline correction** · _C ABI_: `n4m_transform_modpoly_*`

## Description

Modified polynomial baseline correction.

## Parameters

| Name | Type | Default |
|------|------|---------|
| `polyorder` | `int` | `2` |
| `max_iter` | `int` | `250` |
| `tol` | `float` | `0.001` |

## API and bindings

**C ABI (ABI 2):** [`n4m_transform_modpoly_create`](https://github.com/GBeurier/nirs4all-methods/blob/main/cpp/include/n4m/transform/baseline.h#L68) · [`n4m_transform_modpoly_destroy`](https://github.com/GBeurier/nirs4all-methods/blob/main/cpp/include/n4m/transform/baseline.h#L71) · [`n4m_transform_modpoly_transform`](https://github.com/GBeurier/nirs4all-methods/blob/main/cpp/include/n4m/transform/baseline.h#L72). Use the linked public header for the exact signature, configuration, and result handles.

**Python (verified public re-export):**

```python
from n4m.transform.baseline import ModPoly
```

Source signature: [`ModPoly(polyorder: int = 2, max_iter: int = 250, tol: float = 0.001)`](https://github.com/GBeurier/nirs4all-methods/blob/main/bindings/python/src/n4m/_impl/baseline.py#L108).

**R:** no current source-verified entry point was found for this catalog method.

**MATLAB / Octave:** no current source-verified entry point was found for this catalog method.

## Explanations

### Bibliographic source

Lieber & Mahadevan-Jansen (2003), *Automated method for subtraction of fluorescence from biological Raman spectra*, Applied Spectroscopy 57, 1363–1367, https://doi.org/10.1366/000370203322554518.

### Mathematical principle

A polynomial baseline is fit iteratively. After each fit, samples above the fitted baseline are clipped to it in the working spectrum, so positive peaks have decreasing influence on the next polynomial estimate; iteration stops by tolerance or budget.

### Appropriate uses

Subtracting smooth fluorescence from spectra containing mostly positive, relatively narrow peaks.

### Limits and validation

Broad peaks can be clipped into the baseline and high polynomial orders are unstable at edges. Negative bands violate the one-sided clipping assumption.

### Implementation

`n4m.transform.baseline.ModPoly` wraps `n4m_transform_modpoly_*`; iterative clipping and polynomial fitting are in `preprocessing/baselines/modpoly.c`.

The ABI-2 implementation is the `n4m_transform_modpoly_*` lifecycle in libn4m.

### Sources and provenance

https://doi.org/10.1366/000370203322554518; https://github.com/GBeurier/nirs4all-methods/blob/main/cpp/src/core/preprocessing/baselines/modpoly.c


---

_See also_: [methods index](index.md) · [interactive dashboard](../landing/dashboard.md)