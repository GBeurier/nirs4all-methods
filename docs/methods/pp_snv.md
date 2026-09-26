# `pp_snv` — Standard normal variate (SNV)

_Group_: **Preprocessing** · _C ABI_: `n4m_transform_snv_*`

## Description

Standard Normal Variate normalisation.

## Parameters

| Name | Type | Default |
|------|------|---------|
| `with_mean` | `bool` | `True` |
| `with_std` | `bool` | `True` |
| `ddof` | `int` | `0` |

## API and bindings

**C ABI (ABI 2):** [`n4m_transform_snv_create`](https://github.com/GBeurier/nirs4all-methods/blob/main/cpp/include/n4m/transform/scatter.h#L13) · [`n4m_transform_snv_destroy`](https://github.com/GBeurier/nirs4all-methods/blob/main/cpp/include/n4m/transform/scatter.h#L15) · [`n4m_transform_snv_transform`](https://github.com/GBeurier/nirs4all-methods/blob/main/cpp/include/n4m/transform/scatter.h#L16). Use the linked public header for the exact signature, configuration, and result handles.

**Python (verified public re-export):**

```python
from n4m.transform.scatter import SNV
```

Source signature: [`SNV(with_mean: bool = True, with_std: bool = True, ddof: int = 0)`](https://github.com/GBeurier/nirs4all-methods/blob/main/bindings/python/src/n4m/_impl/preprocessing.py#L21).

**R (source-verified):** [`snv_transform(X, with_mean = TRUE, with_std = TRUE, ddof = 0L)`](https://github.com/GBeurier/nirs4all-methods/blob/main/bindings/r/n4m/R/preprocessing.R).

```r
library(n4m)
result <- snv_transform(X)
```

**MATLAB / Octave (source-verified):** [`snv_transform(X, varargin)`](https://github.com/GBeurier/nirs4all-methods/blob/main/bindings/matlab/+n4m/snv_transform.m).

```matlab
addpath('bindings/matlab')
result = n4m.snv_transform(X);
```

## Explanations

### Bibliographic source

Barnes, Dhanoa & Lister (1989), *Standard Normal Variate Transformation and De-trending of Near-Infrared Diffuse Reflectance Spectra*, Applied Spectroscopy 43, 772–777, https://doi.org/10.1366/0003702894202201.

### Mathematical principle

Each spectrum is independently centered and scaled: $x'_i=(x_i-\bar{x}_i)/s_i$, with optional centering, scaling, and configurable `ddof`. Row statistics make the transform stateless across samples.

### Appropriate uses

Reducing additive and multiplicative scatter caused by particle size or optical path variation in diffuse-reflectance spectra.

### Limits and validation

Near-constant spectra have unstable scale, and SNV removes absolute row amplitude that may contain analyte information. It does not remove wavelength-dependent baselines.

### Implementation

`n4m.transform.scatter.SNV` wraps `n4m_transform_snv_*`; row mean/variance and flags are in `preprocessing/scatter/snv.c`.

The ABI-2 implementation is the `n4m_transform_snv_*` lifecycle in libn4m.

### Sources and provenance

https://doi.org/10.1366/0003702894202201; https://github.com/GBeurier/nirs4all-methods/blob/main/cpp/src/core/preprocessing/scatter/snv.c


---

_See also_: [methods index](index.md) · [interactive dashboard](../landing/dashboard.md)