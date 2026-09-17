# `pp_xcorr_align` — Whole-spectrum cross-correlation alignment

_Group_: **Signal transforms** · _C ABI_: `n4m_transform_xcorr_align_*`

## Description

Whole-spectrum integer shift chosen by maximum correlation.

## Parameters

| Name | Type | Default |
|------|------|---------|
| `reference` | `—` | `None` |
| `max_shift` | `int` | `5` |

## API and bindings

**C ABI (ABI 2):** [`n4m_transform_xcorr_align_create`](https://github.com/GBeurier/nirs4all-methods/blob/main/cpp/include/n4m/transform/alignment.h#L15) · [`n4m_transform_xcorr_align_destroy`](https://github.com/GBeurier/nirs4all-methods/blob/main/cpp/include/n4m/transform/alignment.h#L18) · [`n4m_transform_xcorr_align_fit`](https://github.com/GBeurier/nirs4all-methods/blob/main/cpp/include/n4m/transform/alignment.h#L19) · [`n4m_transform_xcorr_align_is_fitted`](https://github.com/GBeurier/nirs4all-methods/blob/main/cpp/include/n4m/transform/alignment.h#L24) · [`n4m_transform_xcorr_align_transform`](https://github.com/GBeurier/nirs4all-methods/blob/main/cpp/include/n4m/transform/alignment.h#L21). Use the linked public header for the exact signature, configuration, and result handles.

**Python (verified public re-export):**

```python
from n4m.transform.alignment import CrossCorrelationAlignment
```

Source signature: [`CrossCorrelationAlignment(reference = None, max_shift: int = 5)`](https://github.com/GBeurier/nirs4all-methods/blob/main/bindings/python/src/n4m/_impl/advanced.py#L403).

**R:** no current source-verified entry point was found for this catalog method.

**MATLAB / Octave:** no current source-verified entry point was found for this catalog method.

## Explanations

### Bibliographic source

No single canonical paper defines this bounded implementation. It applies the standard discrete cross-correlation lag estimator; the source defines padding and tie behaviour.

### Mathematical principle

For each row, every integer lag in `[-max_shift,max_shift]` is evaluated by the centered dot product with the fitted reference. The best lag is applied to the whole spectrum using edge-replicated samples.

### Appropriate uses

Correcting one global channel offset per spectrum before comparison or calibration.

### Limits and validation

It cannot model local stretching, and amplitude/shape changes can move the correlation maximum. Edge replication affects shifted ends; weak or periodic spectra can have ambiguous lags.

### Implementation

`n4m.transform.alignment.CrossCorrelationAlignment` wraps `n4m_transform_xcorr_align_*`; reference fitting and lag search are in `cpp/src/c_api/c_api_advanced.cpp`.

The ABI-2 implementation is the `n4m_transform_xcorr_align_*` lifecycle in libn4m.

### Sources and provenance

https://github.com/GBeurier/nirs4all-methods/blob/main/cpp/src/c_api/c_api_advanced.cpp#L798-L900


---

_See also_: [methods index](index.md) · [interactive dashboard](../landing/dashboard.md)