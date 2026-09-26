# `pp_rnv` — Robust normal variate (RNV)

_Group_: **Preprocessing** · _C ABI_: `n4m_transform_robust_snv_*`

## Description

Robust SNV using median + k * MAD.

## Parameters

| Name | Type | Default |
|------|------|---------|
| `with_center` | `bool` | `True` |
| `with_scale` | `bool` | `True` |
| `k` | `float` | `1.4826` |

## API and bindings

**C ABI (ABI 2):** [`n4m_transform_robust_snv_create`](https://github.com/GBeurier/nirs4all-methods/blob/main/cpp/include/n4m/transform/scatter.h#L38) · [`n4m_transform_robust_snv_destroy`](https://github.com/GBeurier/nirs4all-methods/blob/main/cpp/include/n4m/transform/scatter.h#L41) · [`n4m_transform_robust_snv_transform`](https://github.com/GBeurier/nirs4all-methods/blob/main/cpp/include/n4m/transform/scatter.h#L42). Use the linked public header for the exact signature, configuration, and result handles.

**Python (verified public re-export):**

```python
from n4m.transform.scatter import RNV
```

Source signature: [`RNV(with_center: bool = True, with_scale: bool = True, k: float = 1.4826)`](https://github.com/GBeurier/nirs4all-methods/blob/main/bindings/python/src/n4m/_impl/preprocessing.py#L77).

**R (source-verified):** [`robust_snv_transform(X, with_center = TRUE, with_scale = TRUE, k = 1.4826)`](https://github.com/GBeurier/nirs4all-methods/blob/main/bindings/r/n4m/R/preprocessing.R).

```r
library(n4m)
result <- robust_snv_transform(X)
```

**MATLAB / Octave:** no current source-verified entry point was found for this catalog method.

## Explanations

### Bibliographic source

Guo, Wu & Massart (1999), *The robust normal variate transform for pattern recognition with near-infrared data*, Analytica Chimica Acta 382, 87–103, https://doi.org/10.1016/S0003-2670(98)00737-5.

### Mathematical principle

For each row, RNV subtracts its median and divides by `k` times its median absolute deviation, according to the enabled centering/scaling flags. The default $k=1.4826$ makes MAD consistent for Gaussian scale.

### Appropriate uses

Scatter normalization when isolated spikes or intense bands would destabilize the mean and standard deviation used by SNV.

### Limits and validation

MAD is zero for sufficiently flat or tied spectra, requiring guarded behaviour, and robust row scaling can still erase meaningful absolute intensity. It does not model wavelength-local scatter.

### Implementation

`n4m.transform.scatter.RNV` uses `n4m_transform_robust_snv_*`; median/MAD selection and flags are implemented in `preprocessing/scatter/robust_snv.c`.

The ABI-2 implementation is the `n4m_transform_robust_snv_*` lifecycle in libn4m.

### Sources and provenance

https://doi.org/10.1016/S0003-2670(98)00737-5; https://github.com/GBeurier/nirs4all-methods/blob/main/cpp/src/core/preprocessing/scatter/robust_snv.c


---

_See also_: [methods index](index.md) · [interactive dashboard](../landing/dashboard.md)