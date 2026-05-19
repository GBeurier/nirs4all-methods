# `aug_gaussian_noise` — GaussianAdditiveNoise

_Group_: **Augmentation** · _Registry tolerance_: `rtol=1e-5`, `atol=1e-8` · _Source_: [`docs/algorithms/aug_gaussian_noise.md`](../algorithms/aug_gaussian_noise.md)

## Description

Adds per-sample std-scaled Gaussian noise to spectra.

For each input row `X[i, :]`:

1. Compute the row's biased standard deviation `stds[i] = std(X[i, :])` (ddof=0).
2. Draw a single bulk `noise = standard_normal(rows * cols)` from the borrowed PCG64 handle.
3. `out[i, j] = X[i, j] + noise[i*cols + j] * stds[i] * sigma`.

The single-call bulk draw fixes the RNG stream order, so the C reference matches the frozen NumPy reference at 1e-15 abs.

### Parameters

- `sigma: double` (`>= 0.0`) — relative noise scale; the per-row std is multiplied by `sigma` to get the actual noise stddev.

## Explanations

### Bibliographic source

- Frozen Python: `parity/python_generator/src/c4a_parity_augmenters_ref/gaussian_noise.py` (validated against `nirs4all.operators.augmentation.spectral.GaussianAdditiveNoise` with `variation_scope="sample"`).
- Parity tolerance: 1e-15 abs.

### Mathematical principle

`aug_gaussian_noise` follows the standard augmentation operator contract: an input matrix $\mathbf{X}\in\mathbb{R}^{n\times p}$ is transformed, scored, split, or filtered by the method-specific kernel while preserving the parity contract recorded by the benchmark snapshots. The precise numerical convention is the one exposed by the C ABI signatures and the registered external references below.

### Implementation

The augmenter does not own the RNG. Re-seed the RNG with `c4a_rng_pcg64_set_seed(rng, seed)` before each `_apply` to lock the output. Tested in `cpp/tests/test_augmenters_noise_drift.cpp::aug_gaussian_noise_*`.

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


**Registry parity references** ◆

:::{card}
:class-card: external-refs

- ℹ No external parity reference row is registered for this public helper; the page is generated from the in-tree API and algorithm documentation.
:::

### Benchmarks

No cross-binding timing row is currently registered for this method. The implementation table above is still generated from the public API surface.

---

_See also_: [benchmark overview](../benchmarks/overview.md) · [methods index](index.md) · [interactive dashboard](../landing/dashboard.md)
