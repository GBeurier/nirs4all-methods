# `transfer_metrics` — TransferMetricsComputer

_Group_: **Metric** · _Registry tolerance_: `rtol=1e-5`, `atol=1e-8` · _Source_: [`docs/algorithms/transfer_metrics.md`](../algorithms/transfer_metrics.md)

## Description

Phase 20 introduces a single utility function that quantifies how well a *source* and a *target* dataset align in a shared low-dimensional space — useful for transfer-learning workflows where preprocessing aims to bring the two distributions closer before refitting a downstream model.

The utility mirrors `nirs4all.analysis.transfer_metrics.TransferMetricsComputer` (Python). One C ABI symbol — `c4a_transfer_metrics_compute` — returns a 9-field POD struct.

### Parameters

`c4a_transfer_metrics_compute` takes three scalars + a seed:

| Argument | Meaning | Constraint |
|---|---|---|
| `n_components` | PCA truncation rank | `>= 1`. Effective rank = `min(n_components, p_src, p_tgt, n_src, n_tgt, rank(X_s), rank(X_t))`. |
| `k_neighbors`  | Trustworthiness neighbour count | `>= 2`. Internally capped to `min(n_src, n_tgt) - 2`. |
| `seed`         | Spread-distance subsample seed | Any `uint64`. |

The two matrix views must be row-major contiguous F64. The wrapper returns:

- `C4A_OK` on success (every field of `*out` set).
- `C4A_ERR_NULL_POINTER` if `out` is NULL or a view has NULL data with `rows*cols > 0`.
- `C4A_ERR_INVALID_ARGUMENT` if either dataset has fewer than 2 rows or a scalar fails the constraint above.
- `C4A_ERR_DTYPE_MISMATCH` / `C4A_ERR_STRIDE_INVALID` on non-conforming views.
- `C4A_ERR_OUT_OF_MEMORY` on scratch allocation failure.
- `C4A_ERR_NUMERICAL_FAILURE` if the Jacobi sweep does not converge in the budgeted 100 sweeps (extremely unlikely on well-scaled NIR data; conditions producing this are typically a sign of an upstream bug).

## Explanations

### Bibliographic source

- nirs4all 0.8.11 — `nirs4all/analysis/transfer_metrics.py` (canonical Python implementation).
- Lin et al. 2019 — Centred Kernel Alignment for representation similarity.
- Robert & Escoufier 1976 — RV coefficient.
- Wong et al. 1967 — Grassmann distance / principal angles.
- scipy.spatial.procrustes documentation — Procrustes superposition disparity formula.
- Venna & Kaski 2001 — Trustworthiness formulation (also in `sklearn.manifold.trustworthiness`).

### Mathematical principle

`transfer_metrics` follows the standard metric operator contract: an input matrix $\mathbf{X}\in\mathbb{R}^{n\times p}$ is transformed, scored, split, or filtered by the method-specific kernel while preserving the parity contract recorded by the benchmark snapshots. The precise numerical convention is the one exposed by the C ABI signatures and the registered external references below.

### Implementation

C ABI entry points used by the language bindings:

```c
c4a_status_t c4a_transfer_metrics_compute( c4a_matrix_view_t X_source, c4a_matrix_view_t X_target, int32_t n_components, int32_t k_neighbors, uint64_t seed, c4a_transfer_metrics_t* out);
```

### Implementations

| Layer | Entry point | Language | Contract |
|-------|-------------|----------|----------|
| C ABI | `c4a_transfer_metrics_compute` | C/C++ | Stable libc4a entry point family. |

### Usage

Every chemometrics4all binding dispatches into the same C kernel. The registry references are listed in the parity card below.

::::{tab-set}
:class: chemometrics4all-bindings


:::{tab-item} C ABI · libc4a
:sync: c
:class-label: lang-c

```c
c4a_status_t c4a_transfer_metrics_compute( c4a_matrix_view_t X_source, c4a_matrix_view_t X_target, int32_t n_components, int32_t k_neighbors, uint64_t seed, c4a_transfer_metrics_t* out);
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
