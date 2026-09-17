# `pp_cow_align` — Correlation optimized warping (COW)

_Group_: **Signal transforms** · _C ABI_: `n4m_transform_cow_align_*`

## Description

Segment-wise correlation optimized warping approximation.

## Parameters

| Name | Type | Default |
|------|------|---------|
| `reference` | `—` | `None` |
| `interval_size` | `int` | `32` |
| `max_shift` | `int` | `5` |

## API and bindings

**C ABI (ABI 2):** [`n4m_transform_cow_align_create`](https://github.com/GBeurier/nirs4all-methods/blob/main/cpp/include/n4m/transform/alignment.h#L49) · [`n4m_transform_cow_align_destroy`](https://github.com/GBeurier/nirs4all-methods/blob/main/cpp/include/n4m/transform/alignment.h#L52) · [`n4m_transform_cow_align_fit`](https://github.com/GBeurier/nirs4all-methods/blob/main/cpp/include/n4m/transform/alignment.h#L53) · [`n4m_transform_cow_align_is_fitted`](https://github.com/GBeurier/nirs4all-methods/blob/main/cpp/include/n4m/transform/alignment.h#L58) · [`n4m_transform_cow_align_transform`](https://github.com/GBeurier/nirs4all-methods/blob/main/cpp/include/n4m/transform/alignment.h#L55). Use the linked public header for the exact signature, configuration, and result handles.

**Python (verified public re-export):**

```python
from n4m.transform.alignment import CorrelationOptimizedWarping
```

Source signature: [`CorrelationOptimizedWarping(reference = None, interval_size: int = 32, max_shift: int = 5)`](https://github.com/GBeurier/nirs4all-methods/blob/main/bindings/python/src/n4m/_impl/advanced.py#L438).

**R:** no current source-verified entry point was found for this catalog method.

**MATLAB / Octave:** no current source-verified entry point was found for this catalog method.

## Explanations

### Bibliographic source

Nielsen et al. (1998), *Alignment of single and multiple wavelength chromatographic profiles for chemometric data analysis using correlation optimised warping*, Journal of Chemometrics 12, 521–538, https://doi.org/10.1002/(SICI)1099-128X(199811/12)12:6%3C521::AID-CEM515%3E3.0.CO;2-Z.

### Mathematical principle

The reference is divided into fixed intervals. A dynamic program moves internal sample boundaries within a slack derived from `max_shift`, tries admissible segment lengths, linearly resamples each candidate to the reference interval width, and maximizes cumulative centered correlation before backtracking the best warping path.

### Appropriate uses

Correcting local wavelength-axis displacement while preserving a common output grid, for example before multivariate calibration across instrument sessions.

### Limits and validation

This compact API couples slack to `max_shift` and fixes other classical COW choices. Piecewise linear resampling can smooth narrow peaks, short spectra fall back to global cross-correlation shifting, and a poor reference can induce an incorrect path.

### Implementation

`n4m.transform.alignment.CorrelationOptimizedWarping` uses `n4m_transform_cow_align_*`; the current algorithm is implemented in `cpp/src/c_api/c_api_advanced.cpp`.

The ABI-2 implementation is the `n4m_transform_cow_align_*` lifecycle in libn4m.

### Sources and provenance

https://doi.org/10.1002/(SICI)1099-128X(199811/12)12:6%3C521::AID-CEM515%3E3.0.CO;2-Z; https://github.com/GBeurier/nirs4all-methods/blob/main/cpp/src/c_api/c_api_advanced.cpp


---

_See also_: [methods index](index.md) · [interactive dashboard](../landing/dashboard.md)