# `aug_rotate_translate` — Random hinged rotation-and-translation pattern

_Group_: **Augmentation** · _C ABI_: `n4m_augmentation_rotate_translate_*`

## Description

Random rotate/translate spectral augmentation.

## Parameters

| Name | Type | Default |
|------|------|---------|
| `p_range` | `float` | `2.0` |
| `y_factor` | `float` | `3.0` |
| `rng` | `Optional[PCG64]` | `None` |
| `seed` | `int` | `0` |

## API and bindings

**C ABI (ABI 2):** [`n4m_augmentation_rotate_translate_apply`](https://github.com/GBeurier/nirs4all-methods/blob/main/cpp/include/n4m/augmentation/mixup.h#L40) · [`n4m_augmentation_rotate_translate_create`](https://github.com/GBeurier/nirs4all-methods/blob/main/cpp/include/n4m/augmentation/mixup.h#L35) · [`n4m_augmentation_rotate_translate_destroy`](https://github.com/GBeurier/nirs4all-methods/blob/main/cpp/include/n4m/augmentation/mixup.h#L43). Use the linked public header for the exact signature, configuration, and result handles.

**Python (verified public re-export):**

```python
from n4m.augmentation.mixup import RotateTranslateAugmenter
```

Source signature: [`RotateTranslateAugmenter(p_range: float = 2.0, y_factor: float = 3.0, rng: Optional[PCG64] = None, seed: int = 0)`](https://github.com/GBeurier/nirs4all-methods/blob/main/bindings/python/src/n4m/_impl/augmentation.py#L1193).

**R:** no current source-verified entry point was found for this catalog method.

**MATLAB / Octave:** no current source-verified entry point was found for this catalog method.

## Explanations

### Bibliographic source

No canonical publication defines this piecewise-linear augmenter. Its exact hinge construction and scaling are internal nirs4all behavior.

### Mathematical principle

On a normalized channel axis, the engine samples a hinge and constructs two linear slopes meeting there. The resulting rotation/translation pattern is scaled by each row's standard deviation and by `p_range`/`y_factor`, then added to the spectrum.

### Appropriate uses

Perturbing global tilt and offset while allowing different left/right slopes.

### Limits and validation

This is a geometric heuristic rather than a literal coordinate rotation. Its amplitude vanishes for constant rows and depends on the row's scale.

### Implementation

Python role API `n4m.augmentation.mixup.RotateTranslateAugmenter`; ABI 2 family `n4m_augmentation_rotate_translate_{create,apply,destroy}`.

The ABI-2 implementation is the `n4m_augmentation_rotate_translate_*` lifecycle in libn4m.

### Sources and provenance

https://github.com/GBeurier/nirs4all-methods/blob/main/cpp/src/core/augmentation/random/rotate_translate.h


---

_See also_: [methods index](index.md) · [interactive dashboard](../landing/dashboard.md)