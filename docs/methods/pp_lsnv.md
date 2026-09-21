# `pp_lsnv` — Local standard normal variate (LSNV)

_Group_: **Preprocessing** · _C ABI_: `n4m_transform_local_snv_*`

## Description

Sliding-window (local) SNV.

## Parameters

| Name | Type | Default |
|------|------|---------|
| `window` | `int` | `11` |
| `pad_mode` | `str` | `'reflect'` |
| `constant_value` | `float` | `0.0` |

## API and bindings

**C ABI (ABI 2):** [`n4m_transform_local_snv_create`](https://github.com/GBeurier/nirs4all-methods/blob/main/cpp/include/n4m/transform/scatter.h#L28) · [`n4m_transform_local_snv_destroy`](https://github.com/GBeurier/nirs4all-methods/blob/main/cpp/include/n4m/transform/scatter.h#L31) · [`n4m_transform_local_snv_transform`](https://github.com/GBeurier/nirs4all-methods/blob/main/cpp/include/n4m/transform/scatter.h#L32). Use the linked public header for the exact signature, configuration, and result handles.

**Python (verified public re-export):**

```python
from n4m.transform.scatter import LSNV
```

Source signature: [`LSNV(window: int = 11, pad_mode: str = 'reflect', constant_value: float = 0.0)`](https://github.com/GBeurier/nirs4all-methods/blob/main/bindings/python/src/n4m/_impl/preprocessing.py#L43).

**R:** no current source-verified entry point was found for this catalog method.

**MATLAB / Octave:** no current source-verified entry point was found for this catalog method.

## Explanations

### Bibliographic source

No single canonical paper defines this sliding-window variant. It locally applies the SNV concept of Barnes, Dhanoa & Lister (1989), https://doi.org/10.1366/0003702894202201.

### Mathematical principle

For each wavelength, LSNV computes the mean and standard deviation in a centered window of the same spectrum and standardizes the center sample. Reflect, edge, or constant padding defines incomplete boundary windows.

### Appropriate uses

Removing wavelength-dependent local offset and scale variation that one global SNV cannot represent.

### Limits and validation

It can suppress broad chemical bands and amplify noise where local variance is small. Window width and padding materially affect the output, especially near edges.

### Implementation

`n4m.transform.scatter.LSNV` calls `n4m_transform_local_snv_*`; rolling statistics and padding modes are implemented in `preprocessing/scatter/local_snv.c`.

The ABI-2 implementation is the `n4m_transform_local_snv_*` lifecycle in libn4m.

### Sources and provenance

https://doi.org/10.1366/0003702894202201; https://github.com/GBeurier/nirs4all-methods/blob/main/cpp/src/core/preprocessing/scatter/local_snv.c


---

_See also_: [methods index](index.md) · [interactive dashboard](../landing/dashboard.md)