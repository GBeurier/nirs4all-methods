# `pp_haar` — Single-level Haar discrete wavelet transform

_Group_: **Augmentation** · _C ABI_: `n4m_transform_haar_*`

## Description

Single-level Haar DWT coefficient transform.

## Parameters

_No constructor parameters._

## API and bindings

**C ABI (ABI 2):** [`n4m_transform_haar_create`](https://github.com/GBeurier/nirs4all-methods/blob/main/cpp/include/n4m/transform/wavelet.h#L57) · [`n4m_transform_haar_destroy`](https://github.com/GBeurier/nirs4all-methods/blob/main/cpp/include/n4m/transform/wavelet.h#L58) · [`n4m_transform_haar_output_cols`](https://github.com/GBeurier/nirs4all-methods/blob/main/cpp/include/n4m/transform/wavelet.h#L59) · [`n4m_transform_haar_transform`](https://github.com/GBeurier/nirs4all-methods/blob/main/cpp/include/n4m/transform/wavelet.h#L61). Use the linked public header for the exact signature, configuration, and result handles.

**Python (verified public re-export):**

```python
from n4m.transform.wavelet import Haar
```

Source signature: [`Haar()`](https://github.com/GBeurier/nirs4all-methods/blob/main/bindings/python/src/n4m/_impl/augmentation.py#L66).

**R:** no current source-verified entry point was found for this catalog method.

**MATLAB / Octave:** no current source-verified entry point was found for this catalog method.

## Explanations

### Bibliographic source

Haar (1910), *Zur Theorie der orthogonalen Funktionensysteme*, Mathematische Annalen 69, 331–371, https://doi.org/10.1007/BF01456326.

### Mathematical principle

Adjacent sample pairs are projected onto the Haar scaling and wavelet filters, producing approximation $(x_{2j}+x_{2j+1})/\sqrt2$ and detail $(x_{2j}-x_{2j+1})/\sqrt2$ coefficients under the configured endpoint convention.

### Appropriate uses

Compact separation of coarse spectral shape and one-channel-scale detail.

### Limits and validation

Haar is discontinuous and shift-sensitive; it can represent smooth bands less efficiently than longer wavelets. Odd-length boundary behaviour must be considered.

### Implementation

`n4m.transform.wavelet.Haar` wraps `n4m_transform_haar_*`; the single-level native transform is in `preprocessing/wavelets/haar.c`.

The ABI-2 implementation is the `n4m_transform_haar_*` lifecycle in libn4m.

### Sources and provenance

https://doi.org/10.1007/BF01456326; https://github.com/GBeurier/nirs4all-methods/blob/main/cpp/src/core/preprocessing/wavelets/haar.c


---

_See also_: [methods index](index.md) · [interactive dashboard](../landing/dashboard.md)