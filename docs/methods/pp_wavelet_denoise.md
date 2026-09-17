# `pp_wavelet_denoise` — Multilevel wavelet VisuShrink denoising

_Group_: **Augmentation** · _C ABI_: `n4m_transform_wavelet_denoise_*`

## Description

Multi-level DWT VisuShrink denoising.

<details>
<summary>Full binding docstring</summary>

```text
Multi-level DWT VisuShrink denoising.

Stateless: matches PyWavelets' ``waverec(threshold(coeffs))`` pipeline.
```
</details>

## Parameters

| Name | Type | Default |
|------|------|---------|
| `family` | `str` | `'db4'` |
| `mode` | `str` | `'periodization'` |
| `level` | `int` | `3` |
| `threshold_mode` | `str` | `'soft'` |
| `noise_estimator` | `str` | `'median'` |

## API and bindings

**C ABI (ABI 2):** [`n4m_transform_wavelet_denoise_create`](https://github.com/GBeurier/nirs4all-methods/blob/main/cpp/include/n4m/transform/wavelet.h#L65) · [`n4m_transform_wavelet_denoise_destroy`](https://github.com/GBeurier/nirs4all-methods/blob/main/cpp/include/n4m/transform/wavelet.h#L72) · [`n4m_transform_wavelet_denoise_transform`](https://github.com/GBeurier/nirs4all-methods/blob/main/cpp/include/n4m/transform/wavelet.h#L74). Use the linked public header for the exact signature, configuration, and result handles.

**Python (verified public re-export):**

```python
from n4m.transform.wavelet import WaveletDenoise
```

Source signature: [`WaveletDenoise(family: str = 'db4', mode: str = 'periodization', level: int = 3, threshold_mode: str = 'soft', noise_estimator: str = 'median')`](https://github.com/GBeurier/nirs4all-methods/blob/main/bindings/python/src/n4m/_impl/augmentation.py#L92).

**R:** no current source-verified entry point was found for this catalog method.

**MATLAB / Octave:** no current source-verified entry point was found for this catalog method.

## Explanations

### Bibliographic source

Donoho & Johnstone (1994), *Ideal spatial adaptation by wavelet shrinkage*, Biometrika 81, 425–455, https://doi.org/10.1093/biomet/81.3.425.

### Mathematical principle

Each row is decomposed to the requested feasible level. Noise scale is estimated from the finest detail by MAD/0.6745 or population standard deviation; the universal threshold $\sigma\sqrt{2\log p}$ is applied by hard or soft shrinkage to every detail band before inverse reconstruction.

### Appropriate uses

Removing approximately independent high-frequency noise while retaining multiscale spectral structure.

### Limits and validation

Universal thresholding can oversmooth weak bands, assumes a noise model, and is shift-sensitive. Results depend on family, level, boundary mode, estimator, and hard/soft choice.

### Implementation

`n4m.transform.wavelet.WaveletDenoise` calls `n4m_transform_wavelet_denoise_*`; the threshold and reconstruction pipeline is in `wavelet_denoise.c`.

The ABI-2 implementation is the `n4m_transform_wavelet_denoise_*` lifecycle in libn4m.

### Sources and provenance

https://doi.org/10.1093/biomet/81.3.425; https://github.com/GBeurier/nirs4all-methods/blob/main/cpp/src/core/preprocessing/wavelets/wavelet_denoise.c


---

_See also_: [methods index](index.md) · [interactive dashboard](../landing/dashboard.md)