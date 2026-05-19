# `aug_multiplicative_noise` — MultiplicativeNoise

_Group_: **Augmentation** · _Registry tolerance_: `rtol=1e-5`, `atol=1e-8` · _Source_: [`docs/algorithms/aug_multiplicative_noise.md`](../algorithms/aug_multiplicative_noise.md)

## Description

Applies per-sample multiplicative gain to spectra.

For each input row `X[i, :]`:

1. Draw `eps[i] = sigma_gain * standard_normal()` from the borrowed PCG64 handle (one draw per row, bulk-filled).
2. `out[i, j] = X[i, j] * (1 + eps[i])`.

Mirrors `nirs4all.operators.augmentation.spectral.MultiplicativeNoise` with `per_wavelength=False, variation_scope="sample"` (default). Per-wavelength and batch-scope variants are deferred to a follow-up phase.

### Parameters

- `sigma_gain: double` (`>= 0.0`) — standard deviation of the per-sample gain factor.

## Explanations

### Bibliographic source

- Frozen Python: `parity/python_generator/src/c4a_parity_augmenters_ref/multiplicative_noise.py`.
- Parity tolerance: 1e-15 abs.

### Mathematical principle

`aug_multiplicative_noise` follows the standard augmentation operator contract: an input matrix $\mathbf{X}\in\mathbb{R}^{n\times p}$ is transformed, scored, split, or filtered by the method-specific kernel while preserving the parity contract recorded by the benchmark snapshots. The precise numerical convention is the one exposed by the C ABI signatures and the registered external references below.

### Implementation

Implementation details are defined by the public C ABI and the generated binding wrappers.

### Implementations

| Layer | Entry point | Language | Contract |
|-------|-------------|----------|----------|
| C ABI | — | C/C++ | Stable libc4a entry point family. |

### Usage

Every chemometrics4all binding dispatches into the same C kernel. The registry references are listed in the parity card below.

::::{tab-set}
:class: chemometrics4all-bindings


:::{tab-item} C ABI · libc4a
:sync: c
:class-label: lang-c

```c
/* C ABI prefix */
c4a_*
```

:::

::::


**Registry parity references** 📐

:::{card}
:class-card: external-refs

- ℹ No external parity reference row is registered for this public helper; the page is generated from the in-tree API and algorithm documentation.
:::

### Benchmarks

No cross-binding timing row is currently registered for this method. The implementation table above is still generated from the public API surface.

---

_See also_: [benchmark overview](../benchmarks/overview.md) · [methods index](index.md) · [interactive dashboard](../landing/dashboard.md)
