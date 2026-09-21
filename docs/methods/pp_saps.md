# `pp_saps` — Score-augmented projection standardization (SAPS)

_Group_: **Signal transforms** · _C ABI_: `n4m_transform_saps_*`

## Description

Score-augmented projection standardization inspired by SA-PBS.

## Parameters

| Name | Type | Default |
|------|------|---------|
| `n_components` | `int` | `5` |
| `score_weight` | `float` | `1.0` |
| `fit_intercept` | `bool` | `True` |
| `ridge` | `float` | `0.0` |

## API and bindings

**C ABI (ABI 2):** [`n4m_transform_saps_create`](https://github.com/GBeurier/nirs4all-methods/blob/main/cpp/include/n4m/transform/baseline.h#L165) · [`n4m_transform_saps_destroy`](https://github.com/GBeurier/nirs4all-methods/blob/main/cpp/include/n4m/transform/baseline.h#L168) · [`n4m_transform_saps_fit`](https://github.com/GBeurier/nirs4all-methods/blob/main/cpp/include/n4m/transform/baseline.h#L169) · [`n4m_transform_saps_is_fitted`](https://github.com/GBeurier/nirs4all-methods/blob/main/cpp/include/n4m/transform/baseline.h#L175) · [`n4m_transform_saps_transform`](https://github.com/GBeurier/nirs4all-methods/blob/main/cpp/include/n4m/transform/baseline.h#L172). Use the linked public header for the exact signature, configuration, and result handles.

**Python (verified public re-export):**

```python
from n4m.domain_adaptation.standardization import ScoreAugmentedProjectionStandardization
```

Source signature: [`ScoreAugmentedProjectionStandardization(n_components: int = 5, score_weight: float = 1.0, fit_intercept: bool = True, ridge: float = 0.0)`](https://github.com/GBeurier/nirs4all-methods/blob/main/bindings/python/src/n4m/_impl/advanced.py#L137).

**R:** no current source-verified entry point was found for this catalog method.

**MATLAB / Octave:** no current source-verified entry point was found for this catalog method.

## Explanations

### Bibliographic source

No canonical publication defines this exact n4m transform. It is an internal score-augmented linear standardization inspired by projection-based calibration transfer; the source is the normative specification.

### Mathematical principle

Fit approximates leading source covariance eigenvectors by power iteration, computes centered source scores, appends `score_weight` times those scores to the original features, and fits a multi-output affine/ridge map to paired target spectra. Transform recomputes scores using the stored source mean and loadings before applying that map.

### Appropriate uses

Paired-instrument transfer where a direct linear map benefits from explicit source latent coordinates.

### Limits and validation

Power iteration is fixed at 30 steps per deflated component and is not a full robust PCA solver. It requires paired equal-shape data; component count, score weight, and ridge penalty can overfit.

### Implementation

`n4m.domain_adaptation.standardization.ScoreAugmentedProjectionStandardization` uses `n4m_transform_saps_*`; the exact augmentation and OLS/ridge fit are in `cpp/src/c_api/c_api_advanced.cpp`.

The ABI-2 implementation is the `n4m_transform_saps_*` lifecycle in libn4m.

### Sources and provenance

https://github.com/GBeurier/nirs4all-methods/blob/main/cpp/src/c_api/c_api_advanced.cpp#L356-L496


---

_See also_: [methods index](index.md) · [interactive dashboard](../landing/dashboard.md)