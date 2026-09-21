# `pp_local_centering` — Source-to-target local centering

_Group_: **Signal transforms** · _C ABI_: `n4m_transform_local_centering_*`

## Description

Transfer by subtracting source mean and adding target mean.

## Parameters

_No constructor parameters._

## API and bindings

**C ABI (ABI 2):** [`n4m_transform_local_centering_create`](https://github.com/GBeurier/nirs4all-methods/blob/main/cpp/include/n4m/transform/scatter.h#L130) · [`n4m_transform_local_centering_destroy`](https://github.com/GBeurier/nirs4all-methods/blob/main/cpp/include/n4m/transform/scatter.h#L132) · [`n4m_transform_local_centering_fit`](https://github.com/GBeurier/nirs4all-methods/blob/main/cpp/include/n4m/transform/scatter.h#L134) · [`n4m_transform_local_centering_is_fitted`](https://github.com/GBeurier/nirs4all-methods/blob/main/cpp/include/n4m/transform/scatter.h#L140) · [`n4m_transform_local_centering_transform`](https://github.com/GBeurier/nirs4all-methods/blob/main/cpp/include/n4m/transform/scatter.h#L137). Use the linked public header for the exact signature, configuration, and result handles.

**Python (verified public re-export):**

```python
from n4m.transform.scatter import LocalCentering
```

Source signature: [`LocalCentering()`](https://github.com/GBeurier/nirs4all-methods/blob/main/bindings/python/src/n4m/_impl/advanced.py#L203).

**R:** no current source-verified entry point was found for this catalog method.

**MATLAB / Octave:** no current source-verified entry point was found for this catalog method.

## Explanations

### Bibliographic source

No unique canonical paper: this is a mean-shift domain-adaptation rule, fully specified by the implementation.

### Mathematical principle

Fit computes feature means $\mu_s$ and $\mu_t$ from source and target calibration sets. A source row is transferred as $x-\mu_s+\mu_t$, matching first moments while leaving covariance and higher-order structure unchanged.

### Appropriate uses

Correcting simple additive instrument or batch shifts when source and target samples share aligned wavelength columns.

### Limits and validation

Only mean shift is corrected. Target means estimated from few or nonrepresentative samples are noisy, and using evaluation target data during fit causes leakage.

### Implementation

`n4m.transform.scatter.LocalCentering` calls `n4m_transform_local_centering_*`; the paired-domain state and affine shift are in `cpp/src/c_api/c_api_advanced.cpp`.

The ABI-2 implementation is the `n4m_transform_local_centering_*` lifecycle in libn4m.

### Sources and provenance

https://github.com/GBeurier/nirs4all-methods/blob/main/cpp/src/c_api/c_api_advanced.cpp


---

_See also_: [methods index](index.md) · [interactive dashboard](../landing/dashboard.md)