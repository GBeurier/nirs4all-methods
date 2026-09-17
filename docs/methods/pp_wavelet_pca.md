# `pp_wavelet_pca` — PCA of multilevel wavelet coefficients

_Group_: **Augmentation** · _C ABI_: `n4m_transform_wavelet_pca_*`

## Description

DWT coefficient projection through PCA scores.

## Parameters

| Name | Type | Default |
|------|------|---------|
| `family` | `str` | `'haar'` |
| `mode` | `str` | `'periodization'` |
| `max_level` | `int` | `2` |
| `n_components` | `float` | `5.0` |

## API and bindings

**C ABI (ABI 2):** [`n4m_transform_wavelet_pca_create`](https://github.com/GBeurier/nirs4all-methods/blob/main/cpp/include/n4m/transform/wavelet.h#L98) · [`n4m_transform_wavelet_pca_destroy`](https://github.com/GBeurier/nirs4all-methods/blob/main/cpp/include/n4m/transform/wavelet.h#L104) · [`n4m_transform_wavelet_pca_fit`](https://github.com/GBeurier/nirs4all-methods/blob/main/cpp/include/n4m/transform/wavelet.h#L106) · [`n4m_transform_wavelet_pca_is_fitted`](https://github.com/GBeurier/nirs4all-methods/blob/main/cpp/include/n4m/transform/wavelet.h#L111) · [`n4m_transform_wavelet_pca_output_cols`](https://github.com/GBeurier/nirs4all-methods/blob/main/cpp/include/n4m/transform/wavelet.h#L113) · [`n4m_transform_wavelet_pca_transform`](https://github.com/GBeurier/nirs4all-methods/blob/main/cpp/include/n4m/transform/wavelet.h#L108). Use the linked public header for the exact signature, configuration, and result handles.

**Python (verified public re-export):**

```python
from n4m.transform.wavelet import WaveletPCA
```

Source signature: [`WaveletPCA(family: str = 'haar', mode: str = 'periodization', max_level: int = 2, n_components: float = 5.0)`](https://github.com/GBeurier/nirs4all-methods/blob/main/bindings/python/src/n4m/_impl/augmentation.py#L247).

**R:** no current source-verified entry point was found for this catalog method.

**MATLAB / Octave:** no current source-verified entry point was found for this catalog method.

## Explanations

### Bibliographic source

No unique paper defines this composite. It combines Mallat's DWT (https://doi.org/10.1109/34.192463) with Pearson PCA (https://doi.org/10.1080/14786440109462720).

### Mathematical principle

Each row's multilevel DWT coefficients are packed into one vector. Fit centers that coefficient matrix, performs compact SVD, and retains an integer component count or the smallest count reaching the requested explained-variance fraction; transform projects centered coefficients onto those loadings.

### Appropriate uses

Joint multiscale denoising and dimensionality reduction when raw wavelet coefficients are too numerous for the downstream model.

### Limits and validation

Both DWT and PCA are scale/shift sensitive; interpretation mixes bands and fitting is training-dependent. A variance threshold can yield changing output dimension across folds.

### Implementation

`n4m.transform.wavelet.WaveletPCA` wraps `n4m_transform_wavelet_pca_*`; packing, centering, SVD, selection, and projection are in `wavelet_pca.c`.

The ABI-2 implementation is the `n4m_transform_wavelet_pca_*` lifecycle in libn4m.

### Sources and provenance

https://doi.org/10.1109/34.192463; https://doi.org/10.1080/14786440109462720; https://github.com/GBeurier/nirs4all-methods/blob/main/cpp/src/core/preprocessing/wavelets/wavelet_pca.c


---

_See also_: [methods index](index.md) · [interactive dashboard](../landing/dashboard.md)