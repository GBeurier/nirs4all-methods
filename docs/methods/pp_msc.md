# `pp_msc` — Multiplicative scatter correction (MSC)

_Group_: **Preprocessing** · _C ABI_: `n4m_transform_msc_*`

## Description

Multiplicative Scatter Correction.

<details>
<summary>Full binding docstring</summary>

```text
Multiplicative Scatter Correction.

Fit learns the mean reference spectrum from the training matrix. Transform
regresses each row against that reference and applies the conventional
row-wise MSC correction used by prospectr and pls.
```
</details>

## Parameters

_No constructor parameters._

## API and bindings

**C ABI (ABI 2):** [`n4m_transform_msc_create`](https://github.com/GBeurier/nirs4all-methods/blob/main/cpp/include/n4m/transform/scatter.h#L78) · [`n4m_transform_msc_destroy`](https://github.com/GBeurier/nirs4all-methods/blob/main/cpp/include/n4m/transform/scatter.h#L79) · [`n4m_transform_msc_fit`](https://github.com/GBeurier/nirs4all-methods/blob/main/cpp/include/n4m/transform/scatter.h#L80) · [`n4m_transform_msc_inverse_transform`](https://github.com/GBeurier/nirs4all-methods/blob/main/cpp/include/n4m/transform/scatter.h#L85) · [`n4m_transform_msc_is_fitted`](https://github.com/GBeurier/nirs4all-methods/blob/main/cpp/include/n4m/transform/scatter.h#L89) · [`n4m_transform_msc_transform`](https://github.com/GBeurier/nirs4all-methods/blob/main/cpp/include/n4m/transform/scatter.h#L82). Use the linked public header for the exact signature, configuration, and result handles.

**Python (verified public re-export):**

```python
from n4m.transform.scatter import MSC
```

Source signature: [`MSC()`](https://github.com/GBeurier/nirs4all-methods/blob/main/bindings/python/src/n4m/_impl/preprocessing.py#L194).

**R:** no current source-verified entry point was found for this catalog method.

**MATLAB / Octave:** no current source-verified entry point was found for this catalog method.

## Explanations

### Bibliographic source

Geladi, MacDougall & Martens (1985), *Linearization and Scatter-Correction for Near-Infrared Reflectance Spectra of Meat*, Applied Spectroscopy 39, 491–500, https://doi.org/10.1366/0003702854248656.

### Mathematical principle

Fit stores the training mean reference $r$. Each row is fit as $x_i=a_i+b_i r+e_i$ by ordinary least squares and corrected to $(x_i-a_i)/b_i$, removing row-wise additive and multiplicative scatter.

### Appropriate uses

Diffuse-reflectance calibration where particle size or path length mainly changes offset and scale.

### Limits and validation

MSC assumes a linear relation to a representative reference and becomes unstable when the fitted slope is near zero. Training-reference estimation must stay within folds, and chemical variation correlated with scatter can be altered.

### Implementation

`n4m.transform.scatter.MSC` uses `n4m_transform_msc_*`; reference fitting and row-wise OLS correction are in `preprocessing/scatter/msc.c`.

The ABI-2 implementation is the `n4m_transform_msc_*` lifecycle in libn4m.

### Sources and provenance

https://doi.org/10.1366/0003702854248656; https://github.com/GBeurier/nirs4all-methods/blob/main/cpp/src/core/preprocessing/scatter/msc.c


---

_See also_: [methods index](index.md) · [interactive dashboard](../landing/dashboard.md)