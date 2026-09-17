# `aug_unsharp_mask` — Unsharp spectral masking

_Group_: **Augmentation** · _C ABI_: `n4m_augmentation_unsharp_mask_*`

## Description

Random unsharp spectral mask.

## Parameters

| Name | Type | Default |
|------|------|---------|
| `amount_lo` | `float` | `0.1` |
| `amount_hi` | `float` | `0.5` |
| `sigma` | `float` | `1.0` |
| `kernel_width` | `int` | `11` |
| `rng` | `Optional[PCG64]` | `None` |
| `seed` | `int` | `0` |

## API and bindings

**C ABI (ABI 2):** [`n4m_augmentation_unsharp_mask_apply`](https://github.com/GBeurier/nirs4all-methods/blob/main/cpp/include/n4m/augmentation/spectral.h#L67) · [`n4m_augmentation_unsharp_mask_create`](https://github.com/GBeurier/nirs4all-methods/blob/main/cpp/include/n4m/augmentation/spectral.h#L62) · [`n4m_augmentation_unsharp_mask_destroy`](https://github.com/GBeurier/nirs4all-methods/blob/main/cpp/include/n4m/augmentation/spectral.h#L70). Use the linked public header for the exact signature, configuration, and result handles.

**Python (verified public re-export):**

```python
from n4m.augmentation.spectral import UnsharpMask
```

Source signature: [`UnsharpMask(amount_lo: float = 0.1, amount_hi: float = 0.5, sigma: float = 1.0, kernel_width: int = 11, rng: Optional[PCG64] = None, seed: int = 0)`](https://github.com/GBeurier/nirs4all-methods/blob/main/bindings/python/src/n4m/_impl/augmentation.py#L653).

**R:** no current source-verified entry point was found for this catalog method.

**MATLAB / Octave:** no current source-verified entry point was found for this catalog method.

## Explanations

### Bibliographic source

Unsharp masking is a classical signal/image sharpening construction, but no single NIR paper defines this stochastic parameterization; source code is normative.

### Mathematical principle

Build a Gaussian-smoothed row $S_i=G_{\sigma,w}*X_i$, draw $a_i\sim U(\text{amount_lo},\text{amount_hi})$, and compute $X'_i=X_i+a_i(X_i-S_i)$. `sigma` and odd `kernel_width` set the low-pass scale; `amount_*` controls sharpening.

### Appropriate uses

Varying apparent spectral resolution and sensitivity to narrow bands during training.

### Limits and validation

Sharpening amplifies high-frequency noise and can create overshoot. Reflect padding changes behavior near the first and last wavelengths.

### Implementation

Python role API `n4m.augmentation.spectral.UnsharpMask`; ABI 2 family `n4m_augmentation_unsharp_mask_{create,apply,destroy}`.

The ABI-2 implementation is the `n4m_augmentation_unsharp_mask_*` lifecycle in libn4m.

### Sources and provenance

https://github.com/GBeurier/nirs4all-methods/blob/main/cpp/src/core/augmentation/spectral/unsharp_mask.h


---

_See also_: [methods index](index.md) · [interactive dashboard](../landing/dashboard.md)