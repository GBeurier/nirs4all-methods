# `pp_wavelet_svd` — Truncated SVD of multilevel wavelet coefficients

_Group_: **Augmentation** · _C ABI_: `n4m_transform_wavelet_svd_*`

## Description

DWT coefficient projection through SVD scores.

## Parameters

| Name | Type | Default |
|------|------|---------|
| `family` | `str` | `'haar'` |
| `mode` | `str` | `'periodization'` |
| `max_level` | `int` | `2` |
| `n_components` | `float` | `5.0` |

## API and bindings

**C ABI (ABI 2):** [`n4m_transform_wavelet_svd_create`](https://github.com/GBeurier/nirs4all-methods/blob/main/cpp/include/n4m/transform/wavelet.h#L116) · [`n4m_transform_wavelet_svd_destroy`](https://github.com/GBeurier/nirs4all-methods/blob/main/cpp/include/n4m/transform/wavelet.h#L122) · [`n4m_transform_wavelet_svd_fit`](https://github.com/GBeurier/nirs4all-methods/blob/main/cpp/include/n4m/transform/wavelet.h#L124) · [`n4m_transform_wavelet_svd_is_fitted`](https://github.com/GBeurier/nirs4all-methods/blob/main/cpp/include/n4m/transform/wavelet.h#L129) · [`n4m_transform_wavelet_svd_output_cols`](https://github.com/GBeurier/nirs4all-methods/blob/main/cpp/include/n4m/transform/wavelet.h#L131) · [`n4m_transform_wavelet_svd_transform`](https://github.com/GBeurier/nirs4all-methods/blob/main/cpp/include/n4m/transform/wavelet.h#L126). Use the linked public header for the exact signature, configuration, and result handles.

**Python (verified public re-export):**

```python
from n4m.transform.wavelet import WaveletSVD
```

Source signature: [`WaveletSVD(family: str = 'haar', mode: str = 'periodization', max_level: int = 2, n_components: float = 5.0)`](https://github.com/GBeurier/nirs4all-methods/blob/main/bindings/python/src/n4m/_impl/augmentation.py#L257).

**R:** no current source-verified entry point was found for this catalog method.

**MATLAB / Octave:** no current source-verified entry point was found for this catalog method.

## Explanations

### Bibliographic source

No unique paper defines this composite. It combines Mallat's DWT (https://doi.org/10.1109/34.192463) with Eckart–Young low-rank approximation (https://doi.org/10.1007/BF02288367).

### Mathematical principle

Rows are mapped to packed multilevel DWT coefficients and compact SVD is fitted without the explicit centering used by WaveletPCA. The operator keeps an integer rank or the smallest rank reaching the requested singular-value energy fraction and emits right-singular-vector scores.

### Appropriate uses

Low-rank multiscale features when retaining the coefficient-matrix mean direction is intentional.

### Limits and validation

Uncentered SVD can spend its first direction on mean offset. The learned basis is training-dependent, shift-sensitive through the DWT, and output rank may vary for a fraction threshold.

### Implementation

`n4m.transform.wavelet.WaveletSVD` calls `n4m_transform_wavelet_svd_*`; coefficient packing and uncentered SVD are in `wavelet_svd.c`.

The ABI-2 implementation is the `n4m_transform_wavelet_svd_*` lifecycle in libn4m.

### Sources and provenance

https://doi.org/10.1109/34.192463; https://doi.org/10.1007/BF02288367; https://github.com/GBeurier/nirs4all-methods/blob/main/cpp/src/core/preprocessing/wavelets/wavelet_svd.c


---

_See also_: [methods index](index.md) · [interactive dashboard](../landing/dashboard.md)