# `pp_wavelet` — Single-level discrete wavelet coefficient transform

_Group_: **Augmentation** · _C ABI_: `n4m_transform_wavelet_*`

## Description

Single-level DWT coefficient transform.

## Parameters

| Name | Type | Default |
|------|------|---------|
| `family` | `str` | `'haar'` |
| `mode` | `str` | `'periodization'` |

## API and bindings

**C ABI (ABI 2):** [`n4m_transform_wavelet_create`](https://github.com/GBeurier/nirs4all-methods/blob/main/cpp/include/n4m/transform/wavelet.h#L46) · [`n4m_transform_wavelet_destroy`](https://github.com/GBeurier/nirs4all-methods/blob/main/cpp/include/n4m/transform/wavelet.h#L49) · [`n4m_transform_wavelet_output_cols`](https://github.com/GBeurier/nirs4all-methods/blob/main/cpp/include/n4m/transform/wavelet.h#L50) · [`n4m_transform_wavelet_transform`](https://github.com/GBeurier/nirs4all-methods/blob/main/cpp/include/n4m/transform/wavelet.h#L53). Use the linked public header for the exact signature, configuration, and result handles.

**Python (verified public re-export):**

```python
from n4m.transform.wavelet import Wavelet
```

Source signature: [`Wavelet(family: str = 'haar', mode: str = 'periodization')`](https://github.com/GBeurier/nirs4all-methods/blob/main/bindings/python/src/n4m/_impl/augmentation.py#L32).

**R:** no current source-verified entry point was found for this catalog method.

**MATLAB / Octave:** no current source-verified entry point was found for this catalog method.

## Explanations

### Bibliographic source

Mallat (1989), *A theory for multiresolution signal decomposition: the wavelet representation*, IEEE TPAMI 11, 674–693, https://doi.org/10.1109/34.192463.

### Mathematical principle

A one-level analysis filter bank convolves each spectrum with the selected wavelet's low-pass and high-pass filters and downsamples by two. Approximation coefficients are concatenated with detail coefficients; the configured extension mode determines edge samples.

### Appropriate uses

Separating coarse spectral shape from fine-scale detail or supplying wavelet-domain features to a model.

### Limits and validation

The coefficients are shift-sensitive and their length/order depend on wavelet family, boundary mode, and input width. They are not on the original wavelength grid.

### Implementation

`n4m.transform.wavelet.Wavelet` wraps `n4m_transform_wavelet_*`; orchestration is in `preprocessing/wavelets/wavelet.c` and shared filters are in `wavelet_kernels.c`.

The ABI-2 implementation is the `n4m_transform_wavelet_*` lifecycle in libn4m.

### Sources and provenance

https://doi.org/10.1109/34.192463; https://github.com/GBeurier/nirs4all-methods/blob/main/cpp/src/core/preprocessing/wavelets/wavelet.c; https://github.com/GBeurier/nirs4all-methods/blob/main/cpp/src/core/common/wavelet_kernels.c


---

_See also_: [methods index](index.md) · [interactive dashboard](../landing/dashboard.md)