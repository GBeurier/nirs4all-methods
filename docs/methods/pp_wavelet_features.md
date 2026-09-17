# `pp_wavelet_features` — Multilevel wavelet summary features

_Group_: **Augmentation** · _C ABI_: `n4m_transform_wavelet_features_*`

## Description

Multi-level DWT summary features.

## Parameters

| Name | Type | Default |
|------|------|---------|
| `family` | `str` | `'haar'` |
| `mode` | `str` | `'periodization'` |
| `max_level` | `int` | `3` |
| `entropy` | `str` | `'energy'` |

## API and bindings

**C ABI (ABI 2):** [`n4m_transform_wavelet_features_create`](https://github.com/GBeurier/nirs4all-methods/blob/main/cpp/include/n4m/transform/wavelet.h#L78) · [`n4m_transform_wavelet_features_create_ex`](https://github.com/GBeurier/nirs4all-methods/blob/main/cpp/include/n4m/transform/wavelet.h#L83) · [`n4m_transform_wavelet_features_destroy`](https://github.com/GBeurier/nirs4all-methods/blob/main/cpp/include/n4m/transform/wavelet.h#L89) · [`n4m_transform_wavelet_features_output_cols`](https://github.com/GBeurier/nirs4all-methods/blob/main/cpp/include/n4m/transform/wavelet.h#L91) · [`n4m_transform_wavelet_features_transform`](https://github.com/GBeurier/nirs4all-methods/blob/main/cpp/include/n4m/transform/wavelet.h#L94). Use the linked public header for the exact signature, configuration, and result handles.

**Python (verified public re-export):**

```python
from n4m.transform.wavelet import WaveletFeatures
```

Source signature: [`WaveletFeatures(family: str = 'haar', mode: str = 'periodization', max_level: int = 3, entropy: str = 'energy')`](https://github.com/GBeurier/nirs4all-methods/blob/main/bindings/python/src/n4m/_impl/augmentation.py#L165).

**R:** no current source-verified entry point was found for this catalog method.

**MATLAB / Octave:** no current source-verified entry point was found for this catalog method.

## Explanations

### Bibliographic source

No single canonical paper defines this exact four-statistic descriptor. It uses the multiresolution analysis of Mallat (1989), https://doi.org/10.1109/34.192463, with implementation-specific summaries.

### Mathematical principle

After multilevel DWT, every approximation/detail band contributes four values: mean, population standard deviation, energy $\sum c_i^2$, and either normalized-energy entropy or ten-bin histogram entropy. Output width is $4(L+1)$.

### Appropriate uses

Producing compact, fixed-size multiscale descriptors for classification or regression.

### Limits and validation

Band summaries discard coefficient position and sign structure beyond the mean. Entropy definitions are implementation-specific and depend on family, mode, and feasible decomposition depth.

### Implementation

`n4m.transform.wavelet.WaveletFeatures` uses `n4m_transform_wavelet_features_*`; all four summaries and entropy modes are in `preprocessing/wavelets/wavelet_features.c`.

The ABI-2 implementation is the `n4m_transform_wavelet_features_*` lifecycle in libn4m.

### Sources and provenance

https://doi.org/10.1109/34.192463; https://github.com/GBeurier/nirs4all-methods/blob/main/cpp/src/core/preprocessing/wavelets/wavelet_features.c


---

_See also_: [methods index](index.md) · [interactive dashboard](../landing/dashboard.md)