# `aug_spike_noise` — Sparse impulsive spike injection

_Group_: **Augmentation** · _C ABI_: `n4m_augmentation_spike_noise_*`

## Description

Inject random spike artifacts into spectra.

## Parameters

| Name | Type | Default |
|------|------|---------|
| `n_spikes_min` | `int` | `1` |
| `n_spikes_max` | `int` | `3` |
| `amplitude_min` | `float` | `-0.1` |
| `amplitude_max` | `float` | `0.1` |
| `rng` | `Optional[PCG64]` | `None` |
| `seed` | `int` | `0` |

## API and bindings

**C ABI (ABI 2):** [`n4m_augmentation_spike_noise_apply`](https://github.com/GBeurier/nirs4all-methods/blob/main/cpp/include/n4m/augmentation/noise.h#L41) · [`n4m_augmentation_spike_noise_create`](https://github.com/GBeurier/nirs4all-methods/blob/main/cpp/include/n4m/augmentation/noise.h#L35) · [`n4m_augmentation_spike_noise_destroy`](https://github.com/GBeurier/nirs4all-methods/blob/main/cpp/include/n4m/augmentation/noise.h#L40). Use the linked public header for the exact signature, configuration, and result handles.

**Python (verified public re-export):**

```python
from n4m.augmentation.noise import SpikeNoise
```

Source signature: [`SpikeNoise(n_spikes_min: int = 1, n_spikes_max: int = 3, amplitude_min: float = -0.1, amplitude_max: float = 0.1, rng: Optional[PCG64] = None, seed: int = 0)`](https://github.com/GBeurier/nirs4all-methods/blob/main/bindings/python/src/n4m/_impl/augmentation.py#L409).

**R:** no current source-verified entry point was found for this catalog method.

**MATLAB / Octave:** no current source-verified entry point was found for this catalog method.

## Explanations

### Bibliographic source

No canonical paper defines this exact spike simulator. It is an internal artifact model; the implementation source is the normative specification.

### Mathematical principle

For each row, an integer count $K_i$ is drawn uniformly from [`n_spikes_min`, `n_spikes_max`]. Candidate channel indices and amplitudes $a\sim U(a_{min},a_{max})$ are drawn, indices are sorted and deduplicated, and $X'_{i,j}=X_{i,j}+a$ at the retained locations. The deterministic draw order and the deduplication quirk are part of parity.

### Appropriate uses

Stress-testing models against isolated cosmic-ray, electronic, or acquisition spikes.

### Limits and validation

Coincident candidate indices reduce the realized spike count. Spikes affect one channel each and therefore do not model finite-width peaks or correlated glitches.

### Implementation

Python role API `n4m.augmentation.noise.SpikeNoise`; ABI 2 family `n4m_augmentation_spike_noise_{create,apply,destroy}`.

The ABI-2 implementation is the `n4m_augmentation_spike_noise_*` lifecycle in libn4m.

### Sources and provenance

https://github.com/GBeurier/nirs4all-methods/blob/main/cpp/src/core/augmentation/noise/spike_noise.h


---

_See also_: [methods index](index.md) · [interactive dashboard](../landing/dashboard.md)