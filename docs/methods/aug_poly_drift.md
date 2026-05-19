# `aug_poly_drift` — PolynomialBaselineDrift

_Group_: **Augmentation** · _Registry tolerance_: `rtol=1e-5`, `atol=1e-8` · _Source_: [`docs/algorithms/aug_poly_drift.md`](../algorithms/aug_poly_drift.md)

## Description

Adds a per-sample polynomial baseline drift of configurable degree to spectra.

Algorithm (RNG draws are **order-major** to match nirs4all):

1. Build `lambdas = linspace(-1, 1, n_features)` (NumPy 1.26.4 conventions; for `n_features == 1` returns `[-1.0]`).
2. For each order `k` in `[0, degree]`, draw `n_samples` uniforms and form
   `coeffs[k, i] = coeff_min[k] + (coeff_max[k] - coeff_min[k]) * u_ik`.
3. `out[i, j] = X[i, j] + sum_k coeffs[k, i] * lambdas[j]^k`.

The polynomial accumulation uses the `lj_pow *= lj` recurrence rather than `np.polyval` so the multiplication order is bit-identical between the C reference and the Python ref.

### Parameters

- `degree: int32_t` (`>= 0`) — polynomial degree.
- `coeff_min, coeff_max: const double* (length degree + 1)` — per-order uniform ranges. `coeff_min[k]` and `coeff_max[k]` bracket the order-k coefficient for every sample.

## Explanations

### Bibliographic source

- Frozen Python: `parity/python_generator/src/c4a_parity_augmenters_ref/poly_drift.py`.
- Parity tolerance: 1e-15 abs.

### Mathematical principle

`aug_poly_drift` follows the standard augmentation operator contract: an input matrix $\mathbf{X}\in\mathbb{R}^{n\times p}$ is transformed, scored, split, or filtered by the method-specific kernel while preserving the parity contract recorded by the benchmark snapshots. The precise numerical convention is the one exposed by the C ABI signatures and the registered external references below.

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
