# `pp_beads` — Baseline estimation and denoising with sparsity (BEADS)

_Group_: **Baseline correction** · _C ABI_: `n4m_transform_beads_*`

## Description

Baseline estimation and denoising with sparsity.

## Parameters

| Name | Type | Default |
|------|------|---------|
| `lam_0` | `float` | `100.0` |
| `lam_1` | `float` | `0.5` |
| `lam_2` | `float` | `0.5` |
| `max_iter` | `int` | `50` |
| `tol` | `float` | `0.001` |

## API and bindings

**C ABI (ABI 2):** [`n4m_transform_beads_create`](https://github.com/GBeurier/nirs4all-methods/blob/main/cpp/include/n4m/transform/baseline.h#L155) · [`n4m_transform_beads_destroy`](https://github.com/GBeurier/nirs4all-methods/blob/main/cpp/include/n4m/transform/baseline.h#L159) · [`n4m_transform_beads_transform`](https://github.com/GBeurier/nirs4all-methods/blob/main/cpp/include/n4m/transform/baseline.h#L160). Use the linked public header for the exact signature, configuration, and result handles.

**Python (verified public re-export):**

```python
from n4m.transform.baseline import BEADS
```

Source signature: [`BEADS(lam_0: float = 100.0, lam_1: float = 0.5, lam_2: float = 0.5, max_iter: int = 50, tol: float = 0.001)`](https://github.com/GBeurier/nirs4all-methods/blob/main/bindings/python/src/n4m/_impl/baseline.py#L243).

**R:** no current source-verified entry point was found for this catalog method.

**MATLAB / Octave:** no current source-verified entry point was found for this catalog method.

## Explanations

### Bibliographic source

Ning, Selesnick & Duval (2014), *Chromatogram baseline estimation and denoising using sparsity (BEADS)*, Chemometrics and Intelligent Laboratory Systems 139, 156–167, https://doi.org/10.1016/j.chemolab.2014.09.014.

### Mathematical principle

BEADS decomposes a trace into a slowly varying baseline and a sparse signal by penalizing signal amplitude and first/second differences while constraining the baseline through a high-pass filter model. The native solver iterates this sparse penalized approximation with `lam_0`, `lam_1`, and `lam_2`.

### Appropriate uses

Spectra or chromatograms needing simultaneous smooth-background removal and sparse peak denoising.

### Limits and validation

Its sparsity assumptions suit isolated peaks better than broad overlapping bands; three penalties and the convergence budget need tuning. The native implementation is an iterative approximation of the published formulation.

### Implementation

`n4m.transform.baseline.BEADS` wraps `n4m_transform_beads_*`; the shipped solver is `cpp/src/core/preprocessing/baselines/beads.c`.

The ABI-2 implementation is the `n4m_transform_beads_*` lifecycle in libn4m.

### Sources and provenance

https://doi.org/10.1016/j.chemolab.2014.09.014; https://github.com/GBeurier/nirs4all-methods/blob/main/cpp/src/core/preprocessing/baselines/beads.c


---

_See also_: [methods index](index.md) · [interactive dashboard](../landing/dashboard.md)