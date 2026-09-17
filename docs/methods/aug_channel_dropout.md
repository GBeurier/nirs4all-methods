# `aug_channel_dropout` — Independent spectral-channel dropout

_Group_: **Augmentation** · _C ABI_: `n4m_augmentation_channel_dropout_*`

## Description

Randomly drop individual wavelength channels.

## Parameters

| Name | Type | Default |
|------|------|---------|
| `dropout_prob` | `float` | `0.05` |
| `mode` | `str \| int` | `'zero'` |
| `rng` | `Optional[PCG64]` | `None` |
| `seed` | `int` | `0` |

## API and bindings

**C ABI (ABI 2):** [`n4m_augmentation_channel_dropout_apply`](https://github.com/GBeurier/nirs4all-methods/blob/main/cpp/include/n4m/augmentation/spectral.h#L47) · [`n4m_augmentation_channel_dropout_create`](https://github.com/GBeurier/nirs4all-methods/blob/main/cpp/include/n4m/augmentation/spectral.h#L42) · [`n4m_augmentation_channel_dropout_destroy`](https://github.com/GBeurier/nirs4all-methods/blob/main/cpp/include/n4m/augmentation/spectral.h#L50). Use the linked public header for the exact signature, configuration, and result handles.

**Python (verified public re-export):**

```python
from n4m.augmentation.spectral import ChannelDropout
```

Source signature: [`ChannelDropout(dropout_prob: float = 0.05, mode: str | int = 'zero', rng: Optional[PCG64] = None, seed: int = 0)`](https://github.com/GBeurier/nirs4all-methods/blob/main/bindings/python/src/n4m/_impl/augmentation.py#L617).

**R:** no current source-verified entry point was found for this catalog method.

**MATLAB / Octave:** no current source-verified entry point was found for this catalog method.

## Explanations

### Bibliographic source

No canonical NIR publication defines this cellwise dropout. The exact mask and replacement rules are implementation-defined.

### Mathematical principle

Each cell is marked independently with probability `dropout_prob`. Marked values are either set to zero or linearly interpolated from surviving channel indices in that row. The latter uses endpoint values beyond the first or last survivor.

### Appropriate uses

Simulating sporadic dead pixels/channels and discouraging reliance on single wavelengths.

### Limits and validation

Real sensor failures are often persistent or contiguous rather than IID. If too few channels survive, interpolation becomes poorly informative; zero mode is representation-dependent.

### Implementation

Python role API `n4m.augmentation.spectral.ChannelDropout`; ABI 2 family `n4m_augmentation_channel_dropout_{create,apply,destroy}`.

The ABI-2 implementation is the `n4m_augmentation_channel_dropout_*` lifecycle in libn4m.

### Sources and provenance

https://github.com/GBeurier/nirs4all-methods/blob/main/cpp/src/core/augmentation/spectral/channel_dropout.h


---

_See also_: [methods index](index.md) · [interactive dashboard](../landing/dashboard.md)