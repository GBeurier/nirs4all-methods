# `pp_icoshift_align` — Interval correlation optimized shifting (icoshift-style)

_Group_: **Signal transforms** · _C ABI_: `n4m_transform_icoshift_align_*`

## Description

Interval correlation shifting with fixed-size intervals.

## Parameters

| Name | Type | Default |
|------|------|---------|
| `reference` | `—` | `None` |
| `interval_size` | `int` | `32` |
| `max_shift` | `int` | `5` |

## API and bindings

**C ABI (ABI 2):** [`n4m_transform_icoshift_align_create`](https://github.com/GBeurier/nirs4all-methods/blob/main/cpp/include/n4m/transform/alignment.h#L26) · [`n4m_transform_icoshift_align_destroy`](https://github.com/GBeurier/nirs4all-methods/blob/main/cpp/include/n4m/transform/alignment.h#L29) · [`n4m_transform_icoshift_align_fit`](https://github.com/GBeurier/nirs4all-methods/blob/main/cpp/include/n4m/transform/alignment.h#L31) · [`n4m_transform_icoshift_align_is_fitted`](https://github.com/GBeurier/nirs4all-methods/blob/main/cpp/include/n4m/transform/alignment.h#L36) · [`n4m_transform_icoshift_align_transform`](https://github.com/GBeurier/nirs4all-methods/blob/main/cpp/include/n4m/transform/alignment.h#L33). Use the linked public header for the exact signature, configuration, and result handles.

**Python (verified public re-export):**

```python
from n4m.transform.alignment import IcoshiftAlignment
```

Source signature: [`IcoshiftAlignment(reference = None, interval_size: int = 32, max_shift: int = 5)`](https://github.com/GBeurier/nirs4all-methods/blob/main/bindings/python/src/n4m/_impl/advanced.py#L414).

**R:** no current source-verified entry point was found for this catalog method.

**MATLAB / Octave:** no current source-verified entry point was found for this catalog method.

## Explanations

### Bibliographic source

Savorani, Tomasi & Engelsen (2010), *icoshift: A versatile tool for the rapid alignment of 1D NMR spectra*, Journal of Magnetic Resonance 202, 190–202, https://doi.org/10.1016/j.jmr.2009.11.012.

### Mathematical principle

Each fixed interval is compared with the same reference interval over integer lags within `max_shift`; the lag maximizing the centered cross-product score is applied independently with edge replication. Output length and interval positions are retained.

### Appropriate uses

Fast correction of piecewise-constant local channel shifts when peak order is stable.

### Limits and validation

This is a fixed-interval, bounded-shift implementation rather than every option in the original icoshift software. Independent intervals may create seams and weak intervals yield unstable correlations.

### Implementation

`n4m.transform.alignment.IcoshiftAlignment` uses `n4m_transform_icoshift_align_*`; interval search is implemented in `cpp/src/c_api/c_api_advanced.cpp`.

The ABI-2 implementation is the `n4m_transform_icoshift_align_*` lifecycle in libn4m.

### Sources and provenance

https://doi.org/10.1016/j.jmr.2009.11.012; https://github.com/GBeurier/nirs4all-methods/blob/main/cpp/src/c_api/c_api_advanced.cpp


---

_See also_: [methods index](index.md) · [interactive dashboard](../landing/dashboard.md)