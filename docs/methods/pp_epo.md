# `pp_epo` — External parameter orthogonalization (EPO)

_Group_: **Feature extraction** · _C ABI_: `n4m_domain_adaptation_epo_*`

## Description

External Parameter Orthogonalisation.

## Parameters

| Name | Type | Default |
|------|------|---------|
| `scale` | `bool` | `True` |

## API and bindings

**C ABI (ABI 2):** [`n4m_domain_adaptation_epo_create`](https://github.com/GBeurier/nirs4all-methods/blob/main/cpp/include/n4m/domain_adaptation.h#L13) · [`n4m_domain_adaptation_epo_destroy`](https://github.com/GBeurier/nirs4all-methods/blob/main/cpp/include/n4m/domain_adaptation.h#L14) · [`n4m_domain_adaptation_epo_fit`](https://github.com/GBeurier/nirs4all-methods/blob/main/cpp/include/n4m/domain_adaptation.h#L15) · [`n4m_domain_adaptation_epo_inverse_transform`](https://github.com/GBeurier/nirs4all-methods/blob/main/cpp/include/n4m/domain_adaptation.h#L26) · [`n4m_domain_adaptation_epo_is_fitted`](https://github.com/GBeurier/nirs4all-methods/blob/main/cpp/include/n4m/domain_adaptation.h#L29) · [`n4m_domain_adaptation_epo_transform`](https://github.com/GBeurier/nirs4all-methods/blob/main/cpp/include/n4m/domain_adaptation.h#L18) · [`n4m_domain_adaptation_epo_transform_with_d`](https://github.com/GBeurier/nirs4all-methods/blob/main/cpp/include/n4m/domain_adaptation.h#L21). Use the linked public header for the exact signature, configuration, and result handles.

**Python (verified public re-export):**

```python
from n4m.domain_adaptation.orthogonalization import epo
```

Source signature: [`epo(X, d, scale: bool = True)`](https://github.com/GBeurier/nirs4all-methods/blob/main/bindings/python/src/n4m/_impl/native.py#L9634).

**R:** no current source-verified entry point was found for this catalog method.

**MATLAB / Octave:** no current source-verified entry point was found for this catalog method.

## Explanations

### Bibliographic source

Roger, Chauchard & Bellon-Maurel (2003), *EPO-PLS external parameter orthogonalisation of PLS application to temperature-independent measurement of sugar content of intact fruits*, Chemometrics and Intelligent Laboratory Systems 66, 191–204, https://doi.org/10.1016/S0169-7439(03)00051-0.

### Mathematical principle

From nuisance-variation spectra, EPO estimates leading singular vectors $P$ of the external-parameter subspace and projects data with $I-PP^T$. Optional scaling is learned before the projection and reused at transform time.

### Appropriate uses

Removing measured instrument, temperature, moisture, or batch variation before a calibration is fitted.

### Limits and validation

The nuisance experiment must span unwanted variation without confounding analyte signal. Removing too many components deletes predictive information; EPO is stateful and must be fit within validation folds.

### Implementation

`n4m.domain_adaptation.orthogonalization.EPO` wraps `n4m_domain_adaptation_epo_*`; SVD and projection are in `preprocessing/orthogonalization/epo.c`.

The ABI-2 implementation is the `n4m_domain_adaptation_epo_*` lifecycle in libn4m.

### Sources and provenance

https://doi.org/10.1016/S0169-7439(03)00051-0; https://github.com/GBeurier/nirs4all-methods/blob/main/cpp/src/core/preprocessing/orthogonalization/epo.c


---

_See also_: [methods index](index.md) · [interactive dashboard](../landing/dashboard.md)