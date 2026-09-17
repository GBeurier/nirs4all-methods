# `aug_gaussian_noise` — Gaussian additive noise scaled to each spectrum

_Group_: **Augmentation** · _C ABI_: `n4m_augmentation_gaussian_noise_*`

## Description

Add IID Gaussian noise to each element of ``X``.

## Parameters

| Name | Type | Default |
|------|------|---------|
| `sigma` | `float` | `0.01` |
| `rng` | `Optional[PCG64]` | `None` |
| `seed` | `int` | `0` |

## API and bindings

**C ABI (ABI 2):** [`n4m_augmentation_gaussian_noise_apply`](https://github.com/GBeurier/nirs4all-methods/blob/main/cpp/include/n4m/augmentation/noise.h#L21) · [`n4m_augmentation_gaussian_noise_create`](https://github.com/GBeurier/nirs4all-methods/blob/main/cpp/include/n4m/augmentation/noise.h#L16) · [`n4m_augmentation_gaussian_noise_destroy`](https://github.com/GBeurier/nirs4all-methods/blob/main/cpp/include/n4m/augmentation/noise.h#L20). Use the linked public header for the exact signature, configuration, and result handles.

**Python (verified public re-export):**

```python
from n4m.augmentation.noise import GaussianAdditiveNoise
```

Source signature: [`GaussianAdditiveNoise(sigma: float = 0.01, rng: Optional[PCG64] = None, seed: int = 0)`](https://github.com/GBeurier/nirs4all-methods/blob/main/bindings/python/src/n4m/_impl/augmentation.py#L381).

**R:** no current source-verified entry point was found for this catalog method.

**MATLAB / Octave:** no current source-verified entry point was found for this catalog method.

## Explanations

### Bibliographic source

There is no single canonical paper for additive-noise augmentation. Bjerrum, Glahder & Skov (2017), *Data Augmentation of Spectral Data for CNN Based Deep Chemometrics*, arXiv:1710.01927 (https://arxiv.org/abs/1710.01927), is a spectroscopy-specific precedent for perturbing spectra during training.

### Mathematical principle

For row $i$, the implementation computes the population standard deviation $s_i=\operatorname{std}(X_{i,:},\mathrm{ddof}=0)$ and returns $X'_{ij}=X_{ij}+\sigma s_i Z_{ij}$ with independent $Z_{ij}\sim\mathcal N(0,1)$. Thus `sigma` is a fraction of each spectrum's own spread; it is not an absolute noise standard deviation. `seed` or the supplied PCG64 stream fixes the draw sequence.

### Appropriate uses

Training-time robustness to approximately white detector or read noise when noise magnitude should follow the dynamic range of each spectrum.

### Limits and validation

Noise is independent across wavelengths and samples, with no smoothing or wavelength-dependent variance. Constant rows have $s_i=0$ and are unchanged; do not describe this operator as fixed-variance IID noise.

### Implementation

Python role API `n4m.augmentation.noise.GaussianAdditiveNoise`; ABI 2 family `n4m_augmentation_gaussian_noise_{create,apply,destroy}`. The formula is in `cpp/src/core/augmentation/noise/gaussian_noise.c`.

The ABI-2 implementation is the `n4m_augmentation_gaussian_noise_*` lifecycle in libn4m.

### Sources and provenance

https://github.com/GBeurier/nirs4all-methods/blob/main/cpp/src/core/augmentation/noise/gaussian_noise.h; https://arxiv.org/abs/1710.01927


---

_See also_: [methods index](index.md) · [interactive dashboard](../landing/dashboard.md)