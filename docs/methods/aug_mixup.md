# `aug_mixup` — Within-batch convex mixup of spectra

_Group_: **Augmentation** · _C ABI_: `n4m_augmentation_mixup_*`

## Description

Batch-wise mixup augmentation.

## Parameters

| Name | Type | Default |
|------|------|---------|
| `alpha` | `float` | `0.2` |
| `rng` | `Optional[PCG64]` | `None` |
| `seed` | `int` | `0` |

## API and bindings

**C ABI (ABI 2):** [`n4m_augmentation_mixup_apply`](https://github.com/GBeurier/nirs4all-methods/blob/main/cpp/include/n4m/augmentation/mixup.h#L16) · [`n4m_augmentation_mixup_create`](https://github.com/GBeurier/nirs4all-methods/blob/main/cpp/include/n4m/augmentation/mixup.h#L13) · [`n4m_augmentation_mixup_destroy`](https://github.com/GBeurier/nirs4all-methods/blob/main/cpp/include/n4m/augmentation/mixup.h#L19). Use the linked public header for the exact signature, configuration, and result handles.

**Python (verified public re-export):**

```python
from n4m.augmentation.mixup import MixupAugmenter
```

Source signature: [`MixupAugmenter(alpha: float = 0.2, rng: Optional[PCG64] = None, seed: int = 0)`](https://github.com/GBeurier/nirs4all-methods/blob/main/bindings/python/src/n4m/_impl/augmentation.py#L712).

**R:** no current source-verified entry point was found for this catalog method.

**MATLAB / Octave:** no current source-verified entry point was found for this catalog method.

## Explanations

### Bibliographic source

Zhang, Cisse, Dauphin & Lopez-Paz (2018), *mixup: Beyond Empirical Risk Minimization*, ICLR, arXiv:1710.09412 (https://arxiv.org/abs/1710.09412).

### Mathematical principle

A random permutation $\pi$ pairs rows and independent weights are drawn as $\lambda_i\sim\operatorname{Beta}(\alpha,\alpha)$. The operator returns $X'_i=\lambda_iX_i+(1-\lambda_i)X_{\pi(i)}$. `alpha` below one favors near-endpoint mixtures; larger values concentrate around one half.

### Appropriate uses

Regularizing models by filling linear neighborhoods between observed spectra.

### Limits and validation

This transformer returns only mixed X. It does not mix target labels, although the original mixup method requires the same convex combination of y; callers must preserve label consistency themselves. Pairing is restricted to the batch.

### Implementation

Python role API `n4m.augmentation.mixup.MixupAugmenter`; ABI 2 family `n4m_augmentation_mixup_{create,apply,destroy}`.

The ABI-2 implementation is the `n4m_augmentation_mixup_*` lifecycle in libn4m.

### Sources and provenance

https://github.com/GBeurier/nirs4all-methods/blob/main/cpp/src/core/augmentation/mixup/mixup.h; https://arxiv.org/abs/1710.09412


---

_See also_: [methods index](index.md) · [interactive dashboard](../landing/dashboard.md)