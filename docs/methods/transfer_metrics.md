# `transfer_metrics` — Transfer metrics

_Group_: **Metric** · _Registry tolerance_: `rtol=1e-5`, `atol=1e-8` · _Source_: [`docs/algorithms/transfer_metrics.md`](../algorithms/transfer_metrics.md)

## Description

Phase 20 introduces a single utility function that quantifies how well a *source* and a *target* dataset align in a shared low-dimensional space — useful for transfer-learning workflows where preprocessing aims to bring the two distributions closer before refitting a downstream model.

The utility mirrors `nirs4all.analysis.transfer_metrics.TransferMetricsComputer` (Python). One C ABI symbol — `n4m_transfer_metrics_compute` — returns a 9-field POD struct.

### Parameters

`n4m_transfer_metrics_compute` takes three scalars + a seed:

| Argument | Meaning | Constraint |
|---|---|---|
| `n_components` | PCA truncation rank | `>= 1`. Effective rank = `min(n_components, p_src, p_tgt, n_src, n_tgt, rank(X_s), rank(X_t))`. |
| `k_neighbors`  | Trustworthiness neighbour count | `>= 2`. Internally capped to `min(n_src, n_tgt) - 2`. |
| `seed`         | Spread-distance subsample seed | Any `uint64`. |

The two matrix views must be row-major contiguous F64. The wrapper returns:

- `N4M_OK` on success (every field of `*out` set).
- `N4M_ERR_NULL_POINTER` if `out` is NULL or a view has NULL data with `rows*cols > 0`.
- `N4M_ERR_INVALID_ARGUMENT` if either dataset has fewer than 2 rows or a scalar fails the constraint above.
- `N4M_ERR_DTYPE_MISMATCH` / `N4M_ERR_STRIDE_INVALID` on non-conforming views.
- `N4M_ERR_OUT_OF_MEMORY` on scratch allocation failure.
- `N4M_ERR_NUMERICAL_FAILURE` if the Jacobi sweep does not converge in the budgeted 100 sweeps (extremely unlikely on well-scaled NIR data; conditions producing this are typically a sign of an upstream bug).

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
n4m_status_t n4m_transfer_metrics_compute( n4m_matrix_view_t X_source, n4m_matrix_view_t X_target, int32_t n_components, int32_t k_neighbors, uint64_t seed, n4m_transfer_metrics_t* out);
```

Benchmark comparator backends are registered in the matrix and stored as reproducible snapshots when they define the canonical contract.

### Implementations

| Layer | Entry point | Language | Contract |
|-------|-------------|----------|----------|
| C ABI | `n4m_transfer_metrics_compute` | C/C++ | Stable libn4m entry point family. |
| Python | `n4m.python.transfer_metrics` | Python | ABI-close function backed by ctypes. |
| ref.nirs4all | `nirs4all.TransferMetricsComputer` | Python | canonical/comparator |

### Usage

Every nirs4all-methods binding dispatches into the same C kernel. Registered comparator/source rows are listed in the benchmark card below.

::::{tab-set}
:class: nirs4all-methods-bindings


:::{tab-item} C ABI · libn4m
:sync: c
:class-label: lang-c

```c
n4m_status_t n4m_transfer_metrics_compute( n4m_matrix_view_t X_source, n4m_matrix_view_t X_target, int32_t n_components, int32_t k_neighbors, uint64_t seed, n4m_transfer_metrics_t* out);
```

:::

:::{tab-item} Python ABI · n4m.python
:sync: python-abi
:class-label: lang-python

```python
from n4m import python as n4m

Xt = n4m.transfer_metrics(X)
```

:::

::::


**Benchmark Comparators And Sources** ◆

:::{card}
:class-card: external-refs

- ◆ **`ref.nirs4all`** (Python · canonical) — `nirs4all.TransferMetricsComputer` · nirs4all@cd731a23+dirty
:::

### Validation contract

- Operation: `cross_binding_callable` · comparator: `default_allclose` · tolerance: `rtol=1e-05`, `atol=1e-08` · quality: **strict**
- Default validation dataset: `100×50` · seed `20260556`
- Suites: smoke `3` cells; benchmark `11` cells · Default C/Python/reference parity comparator.
- Metrics: `max_abs_diff`, `rel_l2_diff`, `rms_diff`, `shape_equal`
- Truth sources: cross-binding references declared directly in `benchmarks/cross_binding/orchestrator.py`.

| Backend | Library | Gate | Comparator | Note |
|---------|---------|------|------------|------|
| `ref.nirs4all` | `nirs4all.TransferMetricsComputer` | Python / parity | `default_allclose` |  |

### Benchmarks
Median wall-clock per cell from [`docs/_static/bench-data.json`](../benchmarks/overview.md). Divergence is the worst finite value over the visible sizes for each backend, preferring reference max-abs difference and falling back to binding max-abs difference when no reference comparison is recorded. Rows without a recorded comparison show `—`; the fastest backend per column is marked 🏆.
::::{tab-set}
:class: parity-tabs

:::{tab-item} 1 thread
:sync: threads-1

<div class="parity-table-wrap">
<table class="docutils parity-grouped">
<thead><tr><th>Backend</th><th>Divergence</th><th>100×50</th></tr></thead>
<tbody class="lang-band lang-cpp"><tr class="lang-band-row" data-lang="cpp"><th colspan="3" scope="rowgroup"><span class="lang-band-dot"></span>C++ native · libn4m</th></tr>
<tr class="bk-row"><td class="bk-name"><code>N4M.cpp</code></td><td class="parity parity-divergence parity-exact" title="worst reference max abs diff over visible sizes">2.2e-14</td><td class="ms">2.567 ms</td></tr>
</tbody>
<tbody class="lang-band lang-python"><tr class="lang-band-row" data-lang="python"><th colspan="3" scope="rowgroup"><span class="lang-band-dot"></span>Python · nirs4all-methods</th></tr>
<tr class="bk-row"><td class="bk-name"><code>N4M.python</code></td><td class="parity parity-divergence parity-exact" title="worst reference max abs diff over visible sizes">2.2e-14</td><td class="ms ms-best">🏆 2.458 ms</td></tr>
<tr class="bk-row"><td class="bk-name"><code>N4M.sklearn</code></td><td class="parity parity-divergence parity-exact" title="worst reference max abs diff over visible sizes">2.2e-14</td><td class="ms">2.481 ms</td></tr>
</tbody>
<tbody class="lang-band lang-python"><tr class="lang-band-row" data-lang="python"><th colspan="3" scope="rowgroup"><span class="lang-band-dot"></span>Python · external</th></tr>
<tr class="bk-row truth-source-strict"><td class="bk-name"><span class="truth-mark" title="Registry parity reference (Python): nirs4all.TransferMetricsComputer · nirs4all@cd731a23+dirty — canonical">◆</span><code>ref.nirs4all</code></td><td class="parity parity-divergence parity-exact" title="worst reference max abs diff over visible sizes">0</td><td class="ms">3.359 ms</td></tr>
</tbody>
</table>
</div>

:::
::::

---

_See also_: [benchmark overview](../benchmarks/overview.md) · [methods index](index.md) · [interactive dashboard](../landing/dashboard.md)
