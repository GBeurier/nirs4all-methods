# `pp_emsc` — Extended multiplicative scatter correction (EMSC)

_Group_: **Preprocessing** · _C ABI_: `n4m_transform_emsc_*`

## Description

Extended Multiplicative Scatter Correction (polynomial).

## Parameters

| Name | Type | Default |
|------|------|---------|
| `degree` | `int` | `2` |

## API and bindings

**C ABI (ABI 2):** [`n4m_transform_emsc_create`](https://github.com/GBeurier/nirs4all-methods/blob/main/cpp/include/n4m/transform/scatter.h#L128) · [`n4m_transform_emsc_destroy`](https://github.com/GBeurier/nirs4all-methods/blob/main/cpp/include/n4m/transform/scatter.h#L130) · [`n4m_transform_emsc_fit`](https://github.com/GBeurier/nirs4all-methods/blob/main/cpp/include/n4m/transform/scatter.h#L131) · [`n4m_transform_emsc_get_reference`](https://github.com/GBeurier/nirs4all-methods/blob/main/cpp/include/n4m/transform/scatter.h#L140) · [`n4m_transform_emsc_is_fitted`](https://github.com/GBeurier/nirs4all-methods/blob/main/cpp/include/n4m/transform/scatter.h#L136) · [`n4m_transform_emsc_reference_size`](https://github.com/GBeurier/nirs4all-methods/blob/main/cpp/include/n4m/transform/scatter.h#L138) · [`n4m_transform_emsc_set_reference`](https://github.com/GBeurier/nirs4all-methods/blob/main/cpp/include/n4m/transform/scatter.h#L142) · [`n4m_transform_emsc_transform`](https://github.com/GBeurier/nirs4all-methods/blob/main/cpp/include/n4m/transform/scatter.h#L133). Use the linked public header for the exact signature, configuration, and result handles.

**Python (verified public re-export):**

```python
from n4m.transform.scatter import EMSC
```

Source signature: [`EMSC(degree: int = 2)`](https://github.com/GBeurier/nirs4all-methods/blob/main/bindings/python/src/n4m/_impl/preprocessing.py#L265).

**R (source-verified):** [`emsc_fit(X, degree = 2L)`](https://github.com/GBeurier/nirs4all-methods/blob/main/bindings/r/n4m/R/preprocessing.R).

```r
library(n4m)
result <- emsc_fit(X)
```

**MATLAB / Octave:** no current source-verified entry point was found for this catalog method.

## Explanations

### Bibliographic source

Martens & Stark (1991), *Extended multiplicative signal correction and spectral interference subtraction*, Journal of Pharmaceutical and Biomedical Analysis 9, 625–635, https://doi.org/10.1016/0731-7085(91)80188-F.

### Mathematical principle

Each spectrum is regressed on the fitted mean reference plus polynomial channel terms through the requested degree. Subtracting polynomial contributions and dividing by the reference coefficient extends MSC to smooth additive backgrounds.

### Appropriate uses

Correcting multiplicative scatter together with smooth baseline curvature while preserving the reference-shaped chemical contribution.

### Limits and validation

The polynomial basis can absorb broad analyte variation, and this implementation does not accept arbitrary constituent/interferent spectra. Fit the reference on training data only.

### Implementation

`n4m.transform.scatter.EMSC` uses `n4m_transform_emsc_*`; the training reference and per-row least-squares correction are in `preprocessing/scatter/emsc.c`.

The ABI-2 implementation is the `n4m_transform_emsc_*` lifecycle in libn4m.

### Sources and provenance

https://doi.org/10.1016/0731-7085(91)80188-F; https://github.com/GBeurier/nirs4all-methods/blob/main/cpp/src/core/preprocessing/scatter/emsc.c


---

_See also_: [methods index](index.md) · [interactive dashboard](../landing/dashboard.md)