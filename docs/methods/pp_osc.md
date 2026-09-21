# `pp_osc` — Orthogonal signal correction (OSC)

_Group_: **Feature extraction** · _C ABI_: `n4m_transform_osc_*`

## Description

Orthogonal Signal Correction.

## Parameters

| Name | Type | Default |
|------|------|---------|
| `n_components` | `int` | `1` |
| `scale` | `bool` | `True` |

## API and bindings

**C ABI (ABI 2):** [`n4m_transform_osc_create`](https://github.com/GBeurier/nirs4all-methods/blob/main/cpp/include/n4m/transform/orthogonalization.h#L13) · [`n4m_transform_osc_destroy`](https://github.com/GBeurier/nirs4all-methods/blob/main/cpp/include/n4m/transform/orthogonalization.h#L15) · [`n4m_transform_osc_fit`](https://github.com/GBeurier/nirs4all-methods/blob/main/cpp/include/n4m/transform/orthogonalization.h#L16) · [`n4m_transform_osc_inverse_transform`](https://github.com/GBeurier/nirs4all-methods/blob/main/cpp/include/n4m/transform/orthogonalization.h#L22) · [`n4m_transform_osc_is_fitted`](https://github.com/GBeurier/nirs4all-methods/blob/main/cpp/include/n4m/transform/orthogonalization.h#L25) · [`n4m_transform_osc_transform`](https://github.com/GBeurier/nirs4all-methods/blob/main/cpp/include/n4m/transform/orthogonalization.h#L19). Use the linked public header for the exact signature, configuration, and result handles.

**Python (verified public re-export):**

```python
from n4m.transform.orthogonalization import OSC
```

Source signature: [`OSC(n_components: int = 1, scale: bool = True)`](https://github.com/GBeurier/nirs4all-methods/blob/main/bindings/python/src/n4m/_impl/feature_extraction.py#L27).

**R:** no current source-verified entry point was found for this catalog method.

**MATLAB / Octave:** no current source-verified entry point was found for this catalog method.

## Explanations

### Bibliographic source

Wold et al. (1998), *Orthogonal signal correction of near-infrared spectra*, Chemometrics and Intelligent Laboratory Systems 44, 175–185, https://doi.org/10.1016/S0169-7439(98)00109-9.

### Mathematical principle

OSC extracts directions in $X$ with high variance constrained to be orthogonal to the response $Y$, then removes their score-loading reconstructions from $X$. The native fit repeats this deflation for `n_components`, optionally after scaling.

### Appropriate uses

Supervised removal of structured spectral variation known to be unrelated to the calibration target.

### Limits and validation

OSC uses the response and therefore must be fit inside every validation fold. If the calibration set is small or confounded, it can remove transferable analyte signal; component count needs validation.

### Implementation

`n4m.transform.orthogonalization.OSC` wraps `n4m_transform_osc_*`; supervised component extraction and deflation are in `preprocessing/orthogonalization/osc.c`.

The ABI-2 implementation is the `n4m_transform_osc_*` lifecycle in libn4m.

### Sources and provenance

https://doi.org/10.1016/S0169-7439(98)00109-9; https://github.com/GBeurier/nirs4all-methods/blob/main/cpp/src/core/preprocessing/orthogonalization/osc.c


---

_See also_: [methods index](index.md) · [interactive dashboard](../landing/dashboard.md)