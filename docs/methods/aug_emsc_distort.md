# `aug_emsc_distort` — Random EMSC-like affine and polynomial distortion

_Group_: **Augmentation** · _C ABI_: `n4m_augmentation_emsc_distort_*`

## Description

Random EMSC-like multiplicative, additive and polynomial distortion.

## Parameters

| Name | Type | Default |
|------|------|---------|
| `mult_low` | `float` | `0.9` |
| `mult_high` | `float` | `1.1` |
| `add_low` | `float` | `-0.05` |
| `add_high` | `float` | `0.05` |
| `polynomial_order` | `int` | `2` |
| `polynomial_strength` | `float` | `0.02` |
| `correlation` | `float` | `0.3` |
| `wavelengths` | `—` | `None` |
| `rng` | `Optional[PCG64]` | `None` |
| `seed` | `int` | `0` |

## API and bindings

**C ABI (ABI 2):** [`n4m_augmentation_emsc_distort_apply`](https://github.com/GBeurier/nirs4all-methods/blob/main/cpp/include/n4m/augmentation/scattering.h#L56) · [`n4m_augmentation_emsc_distort_create`](https://github.com/GBeurier/nirs4all-methods/blob/main/cpp/include/n4m/augmentation/scattering.h#L47) · [`n4m_augmentation_emsc_distort_destroy`](https://github.com/GBeurier/nirs4all-methods/blob/main/cpp/include/n4m/augmentation/scattering.h#L59). Use the linked public header for the exact signature, configuration, and result handles.

**Python (verified public re-export):**

```python
from n4m.augmentation.scattering import EMSCDistortionAugmenter
```

Source signature: [`EMSCDistortionAugmenter(mult_low: float = 0.9, mult_high: float = 1.1, add_low: float = -0.05, add_high: float = 0.05, polynomial_order: int = 2, polynomial_strength: float = 0.02, correlation: float = 0.3, wavelengths = None, rng: Optional[PCG64] = None, seed: int = 0)`](https://github.com/GBeurier/nirs4all-methods/blob/main/bindings/python/src/n4m/_impl/augmentation.py#L805).

**R:** no current source-verified entry point was found for this catalog method.

**MATLAB / Octave:** no current source-verified entry point was found for this catalog method.

## Explanations

### Bibliographic source

Martens & Stark (1991), *Extended multiplicative signal correction and spectral interference subtraction*, Journal of Pharmaceutical and Biomedical Analysis 9, 625–635, DOI 10.1016/0731-7085(91)80188-F (https://doi.org/10.1016/0731-7085(91)80188-F). This operator generates EMSC-shaped distortions; it does not perform EMSC correction.

### Mathematical principle

Normalize wavelength to $t\in[-1,1]$. Draw clipped multiplicative $b_i$ and an additive $a_i$ whose mean is correlated with the standardized $b_i$ deviation. For orders $k=1..q$, draw $c_{ik}\sim N(0,s^2/k)$ and return $X'_i=a_i+b_iX_i+\sum_k c_{ik}t^k$.

### Appropriate uses

Training against correlated additive, multiplicative, and smooth baseline scatter effects.

### Limits and validation

Ranges are converted to mean and one-quarter-range standard deviations, then clipped; they are not uniform draws. Required wavelengths must span a nonzero range.

### Implementation

Python role API `n4m.augmentation.scattering.EMSCDistortionAugmenter`; ABI 2 family `n4m_augmentation_emsc_distort_{create,apply,destroy}`.

The ABI-2 implementation is the `n4m_augmentation_emsc_distort_*` lifecycle in libn4m.

### Sources and provenance

https://github.com/GBeurier/nirs4all-methods/blob/main/cpp/src/core/augmentation/scattering/emsc_distort.h; https://doi.org/10.1016/0731-7085(91)80188-F


---

_See also_: [methods index](index.md) · [interactive dashboard](../landing/dashboard.md)