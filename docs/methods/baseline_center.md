# `baseline_center` — Baseline center

_Group_: **Preprocessing** · _Registry tolerance_: `rtol=1e-5`, `atol=1e-8` · _Source_: [`docs/algorithms/baseline_center.md`](../algorithms/baseline_center.md)

## Description

From the `n4m.BaselineCenter` Python wrapper docstring:

> Column-mean baseline centering.

### Parameters

This operator takes no constructor parameters. Use `_create()`, then
`_fit(X)`, then `_transform(X)` / `_inverse_transform(X)`.

## Explanations

### Bibliographic source

Implementation matches `nirs4all.operators.transforms.signal.Baseline`.
The arithmetic is the same as scikit-learn's
`sklearn.preprocessing.StandardScaler(with_std=False)`.

### Mathematical principle

Per-column mean centering. Let $X \in \mathbb{R}^{n \times p}$ be the
training matrix.

**Fit** computes per-column means:

$$
\mu_j = \overline{X_{:, j}} = \frac{1}{n} \sum_{i=1}^{n} X_{i,j}.
$$

**Transform**:

$$
X'_{i,j} = X_{i,j} - \mu_j.
$$

**Inverse transform**:

$$
X_{i,j} = X'_{i,j} + \mu_j.
$$

### Implementation

* `_fit` requires `rows >= 1` and `cols >= 1`.
* `_transform` and `_inverse_transform` return `N4M_ERR_NOT_FITTED` before
  fit, and `N4M_ERR_SHAPE_MISMATCH` on column-count drift.
* Tolerance against nirs4all + numpy: 1e-12 absolute / 1e-13 relative
  (closed-form arithmetic — no least-squares involvement).

C ABI entry points used by the language bindings:

```c
n4m_status_t n4m_pp_baseline_create(n4m_pp_baseline_handle_t** out);
void n4m_pp_baseline_destroy(n4m_pp_baseline_handle_t* handle);
n4m_status_t n4m_pp_baseline_fit(n4m_pp_baseline_handle_t* handle, n4m_matrix_view_t X);
n4m_status_t n4m_pp_baseline_inverse_transform( const n4m_pp_baseline_handle_t* handle, n4m_matrix_view_t X, n4m_matrix_view_t out);
n4m_status_t n4m_pp_baseline_is_fitted( const n4m_pp_baseline_handle_t* handle, int* out_fitted);
n4m_status_t n4m_pp_baseline_transform( const n4m_pp_baseline_handle_t* handle, n4m_matrix_view_t X, n4m_matrix_view_t out);
```

Benchmark comparator backends are registered in the matrix and stored as reproducible snapshots when they define the canonical contract.

### Implementations

| Layer | Entry point | Language | Contract |
|-------|-------------|----------|----------|
| C ABI | `n4m_pp_baseline` | C/C++ | Stable libn4m entry point family. |
| Python | `n4m.python.baseline_center` | Python | ABI-close function backed by ctypes. |
| Python sklearn | `n4m.sklearn.BaselineCenter` | Python | scikit-learn-compatible estimator backed by ctypes. |
| R | `baseline_center(X, X_fit = X)` | R | Public package wrapper around the C ABI. |
| ref.nirs4all | `nirs4all.Baseline` | Python | canonical/comparator |

### Usage

Every nirs4all-methods binding dispatches into the same C kernel. Registered comparator/source rows are listed in the benchmark card below.

::::{tab-set}
:class: nirs4all-methods-bindings


:::{tab-item} C ABI · libn4m
:sync: c
:class-label: lang-c

```c
n4m_status_t n4m_pp_baseline_create(n4m_pp_baseline_handle_t** out);
void n4m_pp_baseline_destroy(n4m_pp_baseline_handle_t* handle);
n4m_status_t n4m_pp_baseline_fit(n4m_pp_baseline_handle_t* handle, n4m_matrix_view_t X);
n4m_status_t n4m_pp_baseline_inverse_transform( const n4m_pp_baseline_handle_t* handle, n4m_matrix_view_t X, n4m_matrix_view_t out);
n4m_status_t n4m_pp_baseline_is_fitted( const n4m_pp_baseline_handle_t* handle, int* out_fitted);
n4m_status_t n4m_pp_baseline_transform( const n4m_pp_baseline_handle_t* handle, n4m_matrix_view_t X, n4m_matrix_view_t out);
```

:::

:::{tab-item} Python ABI · n4m.python
:sync: python-abi
:class-label: lang-python

```python
from n4m import python as n4m

Xt = n4m.baseline_center(X)
```

:::

:::{tab-item} Python sklearn · n4m.sklearn
:sync: python-sklearn
:class-label: lang-python

```python
from n4m.sklearn import BaselineCenter

op = BaselineCenter()
Xt = op.fit_transform(X)
```

:::

:::{tab-item} R · nirs4all-methods
:sync: r
:class-label: lang-r

```r
library(n4m)
res <- baseline_center(X)
```

:::

::::


**Benchmark Comparators And Sources** ◆

:::{card}
:class-card: external-refs

- ◆ **`ref.nirs4all`** (Python · canonical) — `nirs4all.Baseline` · nirs4all@cd731a23+dirty
:::

### Validation contract

- Operation: `cross_binding_callable` · comparator: `default_allclose` · tolerance: `rtol=1e-05`, `atol=1e-08` · quality: **strict**
- Default validation dataset: `100×50` · seed `20260556`
- Suites: smoke `3` cells; benchmark `11` cells · Default C/Python/reference parity comparator.
- Metrics: `max_abs_diff`, `rel_l2_diff`, `rms_diff`, `shape_equal`
- Truth sources: cross-binding references declared directly in `benchmarks/cross_binding/orchestrator.py`.

| Backend | Library | Gate | Comparator | Note |
|---------|---------|------|------------|------|
| `ref.nirs4all` | `nirs4all.Baseline` | Python / parity | `default_allclose` |  |

### Benchmarks
Median wall-clock per cell from [`docs/_static/bench-data.json`](../benchmarks/overview.md). Divergence is the worst finite value over the visible sizes for each backend, preferring reference max-abs difference and falling back to binding max-abs difference when no reference comparison is recorded. Rows without a recorded comparison show `—`; the fastest backend per column is marked 🏆.
::::{tab-set}
:class: parity-tabs

:::{tab-item} 1 thread
:sync: threads-1

<div class="parity-table-wrap">
<table class="docutils parity-grouped">
<thead><tr><th>Backend</th><th>Divergence</th><th>100×50</th><th>100×500</th><th>100×2500</th></tr></thead>
<tbody class="lang-band lang-cpp"><tr class="lang-band-row" data-lang="cpp"><th colspan="5" scope="rowgroup"><span class="lang-band-dot"></span>C++ native · libn4m</th></tr>
<tr class="bk-row"><td class="bk-name"><code>N4M.cpp</code></td><td class="parity parity-divergence parity-exact" title="worst reference max abs diff over visible sizes">0</td><td class="ms ms-best">🏆 0.009 ms</td><td class="ms ms-best">🏆 0.030 ms</td><td class="ms">0.168 ms</td></tr>
</tbody>
<tbody class="lang-band lang-python"><tr class="lang-band-row" data-lang="python"><th colspan="5" scope="rowgroup"><span class="lang-band-dot"></span>Python · nirs4all-methods</th></tr>
<tr class="bk-row"><td class="bk-name"><code>N4M.python</code></td><td class="parity parity-divergence parity-exact" title="worst reference max abs diff over visible sizes">0</td><td class="ms">0.009 ms</td><td class="ms">0.031 ms</td><td class="ms ms-best">🏆 0.159 ms</td></tr>
<tr class="bk-row"><td class="bk-name"><code>N4M.sklearn</code></td><td class="parity parity-divergence parity-exact" title="worst reference max abs diff over visible sizes">0</td><td class="ms">0.011 ms</td><td class="ms">0.035 ms</td><td class="ms">0.177 ms</td></tr>
</tbody>
<tbody class="lang-band lang-r"><tr class="lang-band-row" data-lang="r"><th colspan="5" scope="rowgroup"><span class="lang-band-dot"></span>R · nirs4all-methods</th></tr>
<tr class="bk-row"><td class="bk-name"><code>N4M.R</code></td><td class="parity parity-divergence parity-exact" title="worst reference max abs diff over visible sizes">5.0e-16</td><td class="ms">0.030 ms</td><td class="ms">0.500 ms</td><td class="ms">3.031 ms</td></tr>
</tbody>
<tbody class="lang-band lang-python"><tr class="lang-band-row" data-lang="python"><th colspan="5" scope="rowgroup"><span class="lang-band-dot"></span>Python · external</th></tr>
<tr class="bk-row truth-source-strict"><td class="bk-name"><span class="truth-mark" title="Registry parity reference (Python): nirs4all.Baseline · nirs4all@cd731a23+dirty — canonical">◆</span><code>ref.nirs4all</code></td><td class="parity parity-divergence parity-exact" title="worst reference max abs diff over visible sizes">0</td><td class="ms">0.013 ms</td><td class="ms">0.060 ms</td><td class="ms">0.318 ms</td></tr>
</tbody>
</table>
</div>

:::
::::

---

_See also_: [benchmark overview](../benchmarks/overview.md) · [methods index](index.md) · [interactive dashboard](../landing/dashboard.md)
