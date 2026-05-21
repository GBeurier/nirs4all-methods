# `q_residuals` — Q residuals

_Group_: **Diagnostic** · _Registry tolerance_: `rtol=1e-5`, `atol=1e-8` · _Source_: [`docs/algorithms/q_residuals.md`](../algorithms/q_residuals.md)

## Description

Per-sample residual energy outside the first `n_components` PCA
directions. Complementary to Hotelling T²: T² measures variation inside
the model, Q-residuals measure variation outside.

### Parameters

No public constructor parameters are required for the documented default call.

## Explanations

### Bibliographic source

- Jackson, J. E., Mudholkar, G. S. (1979). "Control procedures for residuals
  associated with principal component analysis."
- Wichura, M. J. (1988). "Algorithm AS 241: The percentage points of the
  normal distribution." Applied Statistics 37, 477-484.
- Fixture: `parity/fixtures/q_residuals_v1.json`.

### Mathematical principle

For a `(n_samples × n_features)` matrix `X`:

1. Centre `X` by column means: `Xc = X - mean(X, axis=0)`.
2. Compact SVD: `Xc = U Σ V^T`.
3. Reconstruct using only the first `k = n_components` directions:
   `Xc_hat[i, :] = Σ_{j<k} U[i, j] · σ_j · V[:, j]^T`.
4. Per-sample residual:
   `Q[i] = Σ_d (Xc[i, d] − Xc_hat[i, d])²`.

### Implementation

- Same Jacobi SVD as Hotelling T² (see
  [hotelling_t2.md](hotelling_t2.md) for the SVD numerics).
- Per-sample Q parity vs LAPACK reference: **1e-10 abs / 1e-11 rel**.
- UCL parity vs `scipy.stats.norm.ppf`-based reference: **1e-6 abs /
  1e-5 rel**. The Wichura AS241 inverse-normal is accurate to ~7 digits
  in the central region (Φ⁻¹(0.95)) — sufficient for a statistical
  control limit but not bit-exact against the scipy LAPACK-quality
  inverter.

C ABI entry points used by the language bindings:

```c
n4m_status_t n4m_util_q_residuals(n4m_matrix_view_t X, int32_t n_components, double alpha, double* q_per_sample, int64_t n_samples, double* ucl_out);
```

Benchmark comparator backends are registered in the matrix and stored as reproducible snapshots when they define the canonical contract.

### Implementations

| Layer | Entry point | Language | Contract |
|-------|-------------|----------|----------|
| C ABI | `n4m_util_q_residuals` | C/C++ | Stable libn4m entry point family. |
| Python | `n4m.python.q_residuals` | Python | ABI-close function backed by ctypes. |
| ref.sklearn | `sklearn.PCA + scipy.stats.norm` | Python | canonical/comparator |

### Usage

Every nirs4all-methods binding dispatches into the same C kernel. Registered comparator/source rows are listed in the benchmark card below.

::::{tab-set}
:class: nirs4all-methods-bindings


:::{tab-item} C ABI · libn4m
:sync: c
:class-label: lang-c

```c
n4m_status_t n4m_util_q_residuals(n4m_matrix_view_t X, int32_t n_components, double alpha, double* q_per_sample, int64_t n_samples, double* ucl_out);
```

:::

:::{tab-item} Python ABI · n4m.python
:sync: python-abi
:class-label: lang-python

```python
from n4m import python as n4m

Xt = n4m.q_residuals(X)
```

:::

::::


**Benchmark Comparators And Sources** ◆

:::{card}
:class-card: external-refs

- ◆ **`ref.sklearn`** (Python · canonical) — `sklearn.PCA + scipy.stats.norm` · scikit-learn 1.8.0
:::

### Validation contract

- Operation: `cross_binding_callable` · comparator: `default_allclose` · tolerance: `rtol=1e-05`, `atol=1e-08` · quality: **strict**
- Default validation dataset: `100×50` · seed `20260556`
- Suites: smoke `3` cells; benchmark `11` cells · Default C/Python/reference parity comparator.
- Metrics: `max_abs_diff`, `rel_l2_diff`, `rms_diff`, `shape_equal`
- Truth sources: cross-binding references declared directly in `benchmarks/cross_binding/orchestrator.py`.

| Backend | Library | Gate | Comparator | Note |
|---------|---------|------|------------|------|
| `ref.sklearn` | `sklearn.PCA + scipy.stats.norm` | Python / parity | `default_allclose` |  |

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
<tr class="bk-row"><td class="bk-name"><code>N4M.cpp</code></td><td class="parity parity-divergence parity-exact" title="worst reference max abs diff over visible sizes">9.0e-13</td><td class="ms">1.467 ms</td></tr>
</tbody>
<tbody class="lang-band lang-python"><tr class="lang-band-row" data-lang="python"><th colspan="3" scope="rowgroup"><span class="lang-band-dot"></span>Python · nirs4all-methods</th></tr>
<tr class="bk-row"><td class="bk-name"><code>N4M.python</code></td><td class="parity parity-divergence parity-exact" title="worst reference max abs diff over visible sizes">9.0e-13</td><td class="ms">1.481 ms</td></tr>
<tr class="bk-row"><td class="bk-name"><code>N4M.sklearn</code></td><td class="parity parity-divergence parity-exact" title="worst reference max abs diff over visible sizes">9.0e-13</td><td class="ms">1.478 ms</td></tr>
</tbody>
<tbody class="lang-band lang-python"><tr class="lang-band-row" data-lang="python"><th colspan="3" scope="rowgroup"><span class="lang-band-dot"></span>Python · external</th></tr>
<tr class="bk-row truth-source-strict"><td class="bk-name"><span class="truth-mark" title="Registry parity reference (Python): sklearn.PCA + scipy.stats.norm · scikit-learn 1.8.0 — canonical">◆</span><code>ref.sklearn</code></td><td class="parity parity-divergence parity-exact" title="worst reference max abs diff over visible sizes">0</td><td class="ms ms-best">🏆 0.958 ms</td></tr>
</tbody>
</table>
</div>

:::
::::

---

_See also_: [benchmark overview](../benchmarks/overview.md) · [methods index](index.md) · [interactive dashboard](../landing/dashboard.md)
