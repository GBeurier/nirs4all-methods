# `pp_baseline` — Training-set column mean centering

_Group_: **Preprocessing** · _C ABI_: `n4m_transform_baseline_center_*`

## Description

Column-mean baseline centering.

## Parameters

_No constructor parameters._

## API and bindings

**C ABI (ABI 2):** [`n4m_transform_baseline_center_create`](https://github.com/GBeurier/nirs4all-methods/blob/main/cpp/include/n4m/transform/scaling.h#L81) · [`n4m_transform_baseline_center_destroy`](https://github.com/GBeurier/nirs4all-methods/blob/main/cpp/include/n4m/transform/scaling.h#L82) · [`n4m_transform_baseline_center_fit`](https://github.com/GBeurier/nirs4all-methods/blob/main/cpp/include/n4m/transform/scaling.h#L83) · [`n4m_transform_baseline_center_inverse_transform`](https://github.com/GBeurier/nirs4all-methods/blob/main/cpp/include/n4m/transform/scaling.h#L89) · [`n4m_transform_baseline_center_is_fitted`](https://github.com/GBeurier/nirs4all-methods/blob/main/cpp/include/n4m/transform/scaling.h#L93) · [`n4m_transform_baseline_center_transform`](https://github.com/GBeurier/nirs4all-methods/blob/main/cpp/include/n4m/transform/scaling.h#L85). Use the linked public header for the exact signature, configuration, and result handles.

**Python (verified public re-export):**

```python
from n4m.transform.scaling import BaselineCenter
```

Source signature: [`BaselineCenter()`](https://github.com/GBeurier/nirs4all-methods/blob/main/bindings/python/src/n4m/_impl/preprocessing.py#L328).

**R:** no current source-verified entry point was found for this catalog method.

**MATLAB / Octave:** no current source-verified entry point was found for this catalog method.

## Explanations

### Bibliographic source

No single canonical paper: this is the standard centering operation used by PCA, PLS, and linear models. The implementation source is the normative definition.

### Mathematical principle

Fit stores $\mu_j=n^{-1}\sum_i X_{ij}$ for every wavelength; transform returns $X_{ij}-\mu_j$ and inverse transform adds the same training mean. It is feature centering across samples, not row-wise spectral baseline subtraction.

### Appropriate uses

Preparing calibration matrices for covariance-based models and reproducing the training coordinate system on validation or prediction data.

### Limits and validation

It only removes a constant offset per feature and is stateful: fitting before a data split leaks validation information. It does not correct per-spectrum drift.

### Implementation

`n4m.transform.scaling.BaselineCenter` uses `n4m_transform_baseline_center_*`; `scaling/baseline.c` stores and reuses the training column means.

The ABI-2 implementation is the `n4m_transform_baseline_center_*` lifecycle in libn4m.

### Sources and provenance

https://github.com/GBeurier/nirs4all-methods/blob/main/cpp/src/core/preprocessing/scaling/baseline.c


---

_See also_: [methods index](index.md) · [interactive dashboard](../landing/dashboard.md)