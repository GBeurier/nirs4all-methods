# `aug_path_length` — Multiplicative path-length perturbation

_Group_: **Augmentation** · _C ABI_: `n4m_augmentation_path_length_*`

## Description

Simulate multiplicative path-length variation.

## Parameters

| Name | Type | Default |
|------|------|---------|
| `path_length_std` | `float` | `0.05` |
| `min_path_length` | `float` | `0.1` |
| `rng` | `Optional[PCG64]` | `None` |
| `seed` | `int` | `0` |

## API and bindings

**C ABI (ABI 2):** [`n4m_augmentation_path_length_apply`](https://github.com/GBeurier/nirs4all-methods/blob/main/cpp/include/n4m/augmentation/drift.h#L40) · [`n4m_augmentation_path_length_create`](https://github.com/GBeurier/nirs4all-methods/blob/main/cpp/include/n4m/augmentation/drift.h#L35) · [`n4m_augmentation_path_length_destroy`](https://github.com/GBeurier/nirs4all-methods/blob/main/cpp/include/n4m/augmentation/drift.h#L39). Use the linked public header for the exact signature, configuration, and result handles.

**Python (verified public re-export):**

```python
from n4m.augmentation.drift import PathLengthAugmenter
```

Source signature: [`PathLengthAugmenter(path_length_std: float = 0.05, min_path_length: float = 0.1, rng: Optional[PCG64] = None, seed: int = 0)`](https://github.com/GBeurier/nirs4all-methods/blob/main/bindings/python/src/n4m/_impl/augmentation.py#L496).

**R:** no current source-verified entry point was found for this catalog method.

**MATLAB / Octave:** no current source-verified entry point was found for this catalog method.

## Explanations

### Bibliographic source

There is no paper canonical to this random simulator. Its physical motivation is the multiplicative path-length/scatter term treated by Martens & Stark (1991), DOI 10.1016/0731-7085(91)80188-F (https://doi.org/10.1016/0731-7085(91)80188-F).

### Mathematical principle

For row $i$, draw $L_i=1+sZ_i$ with `path_length_std` $s$ and $Z_i\sim\mathcal N(0,1)$, clamp $L_i$ below by `min_path_length`, and return $X'_{ij}=L_iX_{ij}$.

### Appropriate uses

Training against moderate sample-thickness or optical path-length variability.

### Limits and validation

Every wavelength receives the same factor, so wavelength-dependent scattering and additive baselines are absent. The lower clamp makes the factor distribution non-Gaussian when variability is large.

### Implementation

Python role API `n4m.augmentation.drift.PathLengthAugmenter`; ABI 2 family `n4m_augmentation_path_length_{create,apply,destroy}`.

The ABI-2 implementation is the `n4m_augmentation_path_length_*` lifecycle in libn4m.

### Sources and provenance

https://github.com/GBeurier/nirs4all-methods/blob/main/cpp/src/core/augmentation/drift/path_length.h; https://doi.org/10.1016/0731-7085(91)80188-F


---

_See also_: [methods index](index.md) · [interactive dashboard](../landing/dashboard.md)