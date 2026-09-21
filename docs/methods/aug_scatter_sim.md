# `aug_scatter_sim` — MSC-style affine scatter simulation

_Group_: **Augmentation** · _C ABI_: `n4m_augmentation_scatter_sim_msc_*`

## Description

MSC-style multiplicative/additive scatter simulation.

## Parameters

| Name | Type | Default |
|------|------|---------|
| `a_low` | `float` | `-0.05` |
| `a_high` | `float` | `0.05` |
| `b_low` | `float` | `0.9` |
| `b_high` | `float` | `1.1` |
| `rng` | `Optional[PCG64]` | `None` |
| `seed` | `int` | `0` |

## API and bindings

**C ABI (ABI 2):** [`n4m_augmentation_scatter_sim_msc_apply`](https://github.com/GBeurier/nirs4all-methods/blob/main/cpp/include/n4m/augmentation/scattering.h#L17) · [`n4m_augmentation_scatter_sim_msc_create`](https://github.com/GBeurier/nirs4all-methods/blob/main/cpp/include/n4m/augmentation/scattering.h#L13) · [`n4m_augmentation_scatter_sim_msc_destroy`](https://github.com/GBeurier/nirs4all-methods/blob/main/cpp/include/n4m/augmentation/scattering.h#L20). Use the linked public header for the exact signature, configuration, and result handles.

**Python (verified public re-export):**

```python
from n4m.augmentation.scattering import ScatterSimulationMSC
```

Source signature: [`ScatterSimulationMSC(a_low: float = -0.05, a_high: float = 0.05, b_low: float = 0.9, b_high: float = 1.1, rng: Optional[PCG64] = None, seed: int = 0)`](https://github.com/GBeurier/nirs4all-methods/blob/main/bindings/python/src/n4m/_impl/augmentation.py#L741).

**R:** no current source-verified entry point was found for this catalog method.

**MATLAB / Octave:** no current source-verified entry point was found for this catalog method.

## Explanations

### Bibliographic source

Geladi, MacDougall & Martens (1985), *Linearization and Scatter-Correction for Near-Infrared Reflectance Spectra of Meat*, Applied Spectroscopy 39, 491–500, DOI 10.1366/0003702854248656 (https://doi.org/10.1366/0003702854248656). The augmenter simulates the affine effects that MSC is designed to remove.

### Mathematical principle

For each spectrum draw $a_i\sim U(a_{low},a_{high})$ and $b_i\sim U(b_{low},b_{high})$, then compute $X'_i=a_i+b_iX_i$. This is the forward affine scatter model; it does not estimate or correct coefficients.

### Appropriate uses

Training robustness to additive baseline and multiplicative scatter variability.

### Limits and validation

Only wavelength-independent offset and gain are represented. The operator ignores wavelength input and does not implement a global-mean reference branch.

### Implementation

Python role API `n4m.augmentation.scattering.ScatterSimulationMSC`; ABI 2 family `n4m_augmentation_scatter_sim_msc_{create,apply,destroy}`.

The ABI-2 implementation is the `n4m_augmentation_scatter_sim_msc_*` lifecycle in libn4m.

### Sources and provenance

https://github.com/GBeurier/nirs4all-methods/blob/main/cpp/src/core/augmentation/scattering/scatter_sim_msc.h; https://doi.org/10.1366/0003702854248656


---

_See also_: [methods index](index.md) · [interactive dashboard](../landing/dashboard.md)