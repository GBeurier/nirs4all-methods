# `high_leverage` — HighLeverageFilter

_Group_: **Filter** · _Registry tolerance_: `rtol=1e-5`, `atol=1e-8` · _Source_: [`docs/algorithms/high_leverage.md`](../algorithms/high_leverage.md)

## Description

Identifies samples with high leverage (influence) on a linear regression fit. The leverage of sample i is the i-th diagonal of the hat matrix `H = X (X^T X)^{-1} X^T`. Samples whose leverage exceeds a threshold are excluded; the remaining samples form the training set for downstream models.

### Parameters

No public constructor parameters are required for the documented default call.

## Explanations

### Bibliographic source

- nirs4all 0.8.x — `nirs4all/operators/filters/high_leverage.py`
- `parity/python_generator/src/c4a_parity_filters_ref/high_leverage.py`

### Mathematical principle

`high_leverage` follows the standard filter operator contract: an input matrix $\mathbf{X}\in\mathbb{R}^{n\times p}$ is transformed, scored, split, or filtered by the method-specific kernel while preserving the parity contract recorded by the benchmark snapshots. The precise numerical convention is the one exposed by the C ABI signatures and the registered external references below.

### Implementation

- Tikhonov regularisation: `reg = 1e-10 * trace(X_c^T X_c) / cols`. Identical to the nirs4all reference.
- Hat path: Householder QR factorisation of `X_c^T X_c + reg * I` via the shared `c4a_householder_qr` helper; per-row leverage computed by `(R^{-1} Q^T x_i) . x_i` (back-substitution, no explicit inverse).
- PCA path: cyclic Jacobi eigendecomposition of `X_c^T X_c` (no scipy dependency). Sign convention matches sklearn (each component's maximum-absolute element is non-negative).
- Parity tolerance: `1e-9 abs / 1e-10 rel` against the frozen NumPy reference. Mask equality is exact at the contracted tolerance because the threshold lies far from any data point in the fixture.

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
