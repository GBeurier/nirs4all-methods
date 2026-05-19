# `aug_linear_drift` — LinearBaselineDrift

_Group_: **Augmentation** · _Registry tolerance_: `rtol=1e-5`, `atol=1e-8` · _Source_: [`docs/algorithms/aug_linear_drift.md`](../algorithms/aug_linear_drift.md)

## Description

Adds a per-sample affine baseline drift (constant + linear-in-wavelength) to spectra.

Algorithm:

1. **Offsets** (`n_samples` uniforms in `[offset_min, offset_max)`):
   `offsets[i] = offset_min + (offset_max - offset_min) * u_off_i`.
2. **Slopes**  (`n_samples` uniforms in `[slope_min, slope_max)`):
   `slopes[i]  = slope_min  + (slope_max  - slope_min)  * u_slope_i`.
3. **Apply**: with `lambdas = arange(n_features)` centered at the mean,
   `out[i, j] = X[i, j] + offsets[i] + slopes[i] * (j - (n_features - 1)/2)`.

Mirrors `nirs4all.operators.augmentation.spectral.LinearBaselineDrift` with the implicit-wavelengths branch (no wavelength array supplied — the implicit branch uses `arange(n_features)`).

### Parameters

- `offset_min, offset_max: double` (`min <= max`) — uniform range for the per-sample offset term.
- `slope_min, slope_max: double` (`min <= max`) — uniform range for the per-sample slope.

## Explanations

### Bibliographic source

- Frozen Python: `parity/python_generator/src/c4a_parity_augmenters_ref/linear_drift.py`.
- Parity tolerance: 1e-15 abs.

### Mathematical principle

`aug_linear_drift` follows the standard augmentation operator contract: an input matrix $\mathbf{X}\in\mathbb{R}^{n\times p}$ is transformed, scored, split, or filtered by the method-specific kernel while preserving the parity contract recorded by the benchmark snapshots. The precise numerical convention is the one exposed by the C ABI signatures and the registered external references below.

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
