# `aug_local_mixup` — Nearest-neighbor constrained mixup

_Group_: **Augmentation** · _C ABI_: `n4m_augmentation_local_mixup_*`

## Description

Neighbor-constrained mixup augmentation.

## Parameters

| Name | Type | Default |
|------|------|---------|
| `alpha` | `float` | `0.2` |
| `k_neighbors` | `int` | `5` |
| `rng` | `Optional[PCG64]` | `None` |
| `seed` | `int` | `0` |

## API and bindings

**C ABI (ABI 2):** [`n4m_augmentation_local_mixup_apply`](https://github.com/GBeurier/nirs4all-methods/blob/main/cpp/include/n4m/augmentation/mixup.h#L27) · [`n4m_augmentation_local_mixup_create`](https://github.com/GBeurier/nirs4all-methods/blob/main/cpp/include/n4m/augmentation/mixup.h#L23) · [`n4m_augmentation_local_mixup_destroy`](https://github.com/GBeurier/nirs4all-methods/blob/main/cpp/include/n4m/augmentation/mixup.h#L30). Use the linked public header for the exact signature, configuration, and result handles.

**Python (verified public re-export):**

```python
from n4m.augmentation.mixup import LocalMixupAugmenter
```

Source signature: [`LocalMixupAugmenter(alpha: float = 0.2, k_neighbors: int = 5, rng: Optional[PCG64] = None, seed: int = 0)`](https://github.com/GBeurier/nirs4all-methods/blob/main/bindings/python/src/n4m/_impl/augmentation.py#L726).

**R:** no current source-verified entry point was found for this catalog method.

**MATLAB / Octave:** no current source-verified entry point was found for this catalog method.

## Explanations

### Bibliographic source

This is an internal local variant of Zhang et al.'s mixup (2018), arXiv:1710.09412 (https://arxiv.org/abs/1710.09412); no separate canonical paper defines the specific k-nearest-neighbor rule used here.

### Mathematical principle

Compute exact Euclidean neighbors in X. For each row choose uniformly among its `k_neighbors` nearest non-self rows, draw $\lambda_i\sim\operatorname{Beta}(\alpha,\alpha)$, and form $X'_i=\lambda_iX_i+(1-\lambda_i)X_{n(i)}$.

### Appropriate uses

Interpolating locally within a spectral manifold while avoiding arbitrary distant pairs.

### Limits and validation

Euclidean distance is scale- and preprocessing-dependent. Neighborhood search is quadratic in sample count, and, as for global mixup, y is not returned or mixed.

### Implementation

Python role API `n4m.augmentation.mixup.LocalMixupAugmenter`; ABI 2 family `n4m_augmentation_local_mixup_{create,apply,destroy}`.

The ABI-2 implementation is the `n4m_augmentation_local_mixup_*` lifecycle in libn4m.

### Sources and provenance

https://github.com/GBeurier/nirs4all-methods/blob/main/cpp/src/core/augmentation/mixup/local_mixup.h; https://arxiv.org/abs/1710.09412


---

_See also_: [methods index](index.md) · [interactive dashboard](../landing/dashboard.md)