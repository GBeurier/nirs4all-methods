# `hotelling_t2` — Hotelling T2

_Group_: **Diagnostic** · _Registry tolerance_: `rtol=1e-5`, `atol=1e-8` · _Source_: [`docs/algorithms/hotelling_t2.md`](../algorithms/hotelling_t2.md)

## Description

Multivariate outlier statistic on a PCA-reduced subspace.

### Parameters

No public constructor parameters are required for the documented default call.

## Explanations

### Bibliographic source

- Hotelling, H. (1931). "The Generalization of Student's Ratio".
- Fixture: `parity/fixtures/hotelling_t2_v1.json`.

### Mathematical principle

For a `(n_samples × n_features)` matrix `X`:

1. Centre `X` by column means: `Xc = X - mean(X, axis=0)`.
2. Compact SVD: `Xc = U Σ V^T` via the one-sided Jacobi solver in
   `core/common/svd.{c,h}`.
3. Take the first `k = n_components` columns of the score matrix
   `T = U Σ`.
4. The corresponding sample-covariance eigenvalues are
   `λ_j = σ_j² / (n_samples − 1)`.
5. Per-sample statistic:
   `T²[i] = Σ_{j<k} (T[i, j]² / λ_j)`.

### Implementation

- The PCA uses a Jacobi SVD with default tolerance `1e-14` and at most 50
  sweeps. For the typical NIRS PCA sizes (`p` up to a few hundred) this
  converges in 5–10 sweeps.
- F-quantile inverter: bisection to relative tolerance `1e-12`.
- Per-sample T² parity vs LAPACK reference (`numpy.linalg.svd`):
  **1e-10 abs / 1e-11 rel** — the structural gap between Jacobi rotations
  and Householder bidiagonal SVD.
- UCL parity vs `scipy.stats.f.ppf`: **1e-6 abs / 1e-6 rel** — the
  bisection inverter is intentionally simple and falls below the LAPACK
  / scipy approximation level.

C ABI entry points used by the language bindings:

```c
n4m_status_t n4m_util_hotelling_t2(n4m_matrix_view_t X, int32_t n_components, double alpha, double* t2_per_sample, int64_t n_samples, double* ucl_out);
```

Benchmark comparator backends are registered in the matrix and stored as reproducible snapshots when they define the canonical contract.

### Implementations

| Layer | Entry point | Language | Contract |
|-------|-------------|----------|----------|
| C ABI | `n4m_util_hotelling_t2` | C/C++ | Stable libn4m entry point family. |
| Python | `n4m.python.hotelling_t2` | Python | ABI-close function backed by ctypes. |
| ref.sklearn | `sklearn.PCA + scipy.stats.f` | Python | canonical/comparator |

### Usage

Every nirs4all-methods binding dispatches into the same C kernel. Registered comparator/source rows are listed in the benchmark card below.

::::{tab-set}
:class: nirs4all-methods-bindings


:::{tab-item} C ABI · libn4m
:sync: c
:class-label: lang-c

```c
n4m_status_t n4m_util_hotelling_t2(n4m_matrix_view_t X, int32_t n_components, double alpha, double* t2_per_sample, int64_t n_samples, double* ucl_out);
```

:::

:::{tab-item} Python ABI · n4m.python
:sync: python-abi
:class-label: lang-python

```python
from n4m import python as n4m

Xt = n4m.hotelling_t2(X)
```

:::

::::


**Benchmark Comparators And Sources** ◆

:::{card}
:class-card: external-refs

- ◆ **`ref.sklearn`** (Python · canonical) — `sklearn.PCA + scipy.stats.f` · scikit-learn 1.8.0
:::

### Validation contract

- Operation: `transform` · comparator: `default_allclose` · tolerance: `rtol=1e-05`, `atol=1e-08` · quality: **strict**
- Default validation dataset: `80×24` · seed `1234567897`
- Suites: smoke `3` cells; benchmark `11` cells · Default C/Python/reference parity comparator.
- Metrics: `finite_output`, `max_abs_diff`, `shape_equal`
- Truth sources:
  - `numpy_svd` — numpy.linalg.svd (numpy, 1.26.4)
  - `scipy_f_distribution` — scipy.stats.f (scipy, 1.17.1)

| Backend | Library | Gate | Comparator | Note |
|---------|---------|------|------------|------|
| `ref.sklearn` | `sklearn.PCA + scipy.stats.f` | Python / parity | `default_allclose` |  |

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
<tr class="bk-row"><td class="bk-name"><code>N4M.cpp</code></td><td class="parity parity-divergence parity-exact" title="worst reference max abs diff over visible sizes">6.0e-12</td><td class="ms">1.481 ms</td></tr>
</tbody>
<tbody class="lang-band lang-python"><tr class="lang-band-row" data-lang="python"><th colspan="3" scope="rowgroup"><span class="lang-band-dot"></span>Python · nirs4all-methods</th></tr>
<tr class="bk-row"><td class="bk-name"><code>N4M.python</code></td><td class="parity parity-divergence parity-exact" title="worst reference max abs diff over visible sizes">6.0e-12</td><td class="ms">1.459 ms</td></tr>
<tr class="bk-row"><td class="bk-name"><code>N4M.sklearn</code></td><td class="parity parity-divergence parity-exact" title="worst reference max abs diff over visible sizes">6.0e-12</td><td class="ms">1.508 ms</td></tr>
</tbody>
<tbody class="lang-band lang-python"><tr class="lang-band-row" data-lang="python"><th colspan="3" scope="rowgroup"><span class="lang-band-dot"></span>Python · external</th></tr>
<tr class="bk-row truth-source-strict"><td class="bk-name"><span class="truth-mark" title="Registry parity reference (Python): sklearn.PCA + scipy.stats.f · scikit-learn 1.8.0 — canonical">◆</span><code>ref.sklearn</code></td><td class="parity parity-divergence parity-exact" title="worst reference max abs diff over visible sizes">0</td><td class="ms ms-best">🏆 0.679 ms</td></tr>
</tbody>
</table>
</div>

:::
::::

---

_See also_: [benchmark overview](../benchmarks/overview.md) · [methods index](index.md) · [interactive dashboard](../landing/dashboard.md)
