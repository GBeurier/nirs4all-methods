# `normalize` — Normalize

_Group_: **Preprocessing** · _Registry tolerance_: `rtol=1e-5`, `atol=1e-8` · _Source_: [`docs/algorithms/normalize.md`](../algorithms/normalize.md)

## Description

From the `n4m.Normalize` Python wrapper docstring:

> Column-wise normalisation.

### Parameters

| Name | Type | Default | Notes |
|------|------|---------|-------|
| `feature_min` | `float` | `-1.0` |  |
| `feature_max` | `float` | `1.0` |  |

## Explanations

### Bibliographic source

Implementation matches `nirs4all.operators.transforms.scalers.Normalize`. Note this is **column-wise** — distinct from `sklearn.preprocessing.normalize`, which is row-wise.

### Mathematical principle

Column-wise (axis=0) normalisation with two modes selected by the desired output range $(\ell, u)$:

* **linalg-norm mode** (`(ell, u) == (-1, 1)`, the default in nirs4all): each column is divided by its L2 norm.
  $$
  n_j = \sqrt{\sum_{i=1}^{n} x_{i,j}^2}, \qquad
  x'_{i,j} = \frac{x_{i,j}}{n_j}
  $$

* **user-defined-range mode** (any other $(\ell, u)$): each column is min-max scaled into $[\ell, u]$.
  $$
  \mu_j = \min_i x_{i,j}, \;\; M_j = \max_i x_{i,j}, \;\;
  f_j = \frac{u - \ell}{M_j - \mu_j}, \;\;
  x'_{i,j} = \ell + f_j (x_{i,j} - \mu_j)
  $$

### Implementation

* Column sum-of-squares accumulates left-to-right with the same rounding sequence as `np.linalg.norm(X, axis=0)`.
* All divisions are evaluated per element (no precomputed reciprocal-multiplication), matching `X / linalg_norm_` in nirs4all.
* The user-defined-range arithmetic exactly mirrors `imin + f * (X - min_)` from nirs4all (single multiplication, single addition).

C ABI entry points used by the language bindings:

```c
n4m_status_t n4m_pp_normalize_create(n4m_pp_normalize_handle_t** out, double feature_min, double feature_max);
void n4m_pp_normalize_destroy(n4m_pp_normalize_handle_t* handle);
n4m_status_t n4m_pp_normalize_transform( const n4m_pp_normalize_handle_t* handle, n4m_matrix_view_t X, n4m_matrix_view_t out);
```

Benchmark comparator backends are registered in the matrix and stored as reproducible snapshots when they define the canonical contract.

### Implementations

| Layer | Entry point | Language | Contract |
|-------|-------------|----------|----------|
| C ABI | `n4m_pp_normalize` | C/C++ | Stable libn4m entry point family. |
| Python | `n4m.python.normalize` | Python | ABI-close function backed by ctypes. |
| Python sklearn | `n4m.sklearn.Normalize` | Python | scikit-learn-compatible estimator backed by ctypes. |
| R | `normalize(X, feature_min = -1.0, feature_max = 1.0)` | R | Public package wrapper around the C ABI. |
| ref.nirs4all | `nirs4all.Normalize` | Python | canonical/comparator |

### Usage

Every nirs4all-methods binding dispatches into the same C kernel. Registered comparator/source rows are listed in the benchmark card below.

::::{tab-set}
:class: nirs4all-methods-bindings


:::{tab-item} C ABI · libn4m
:sync: c
:class-label: lang-c

```c
n4m_status_t n4m_pp_normalize_create(n4m_pp_normalize_handle_t** out, double feature_min, double feature_max);
void n4m_pp_normalize_destroy(n4m_pp_normalize_handle_t* handle);
n4m_status_t n4m_pp_normalize_transform( const n4m_pp_normalize_handle_t* handle, n4m_matrix_view_t X, n4m_matrix_view_t out);
```

:::

:::{tab-item} Python ABI · n4m.python
:sync: python-abi
:class-label: lang-python

```python
from n4m import python as n4m

Xt = n4m.normalize(X)
```

:::

:::{tab-item} Python sklearn · n4m.sklearn
:sync: python-sklearn
:class-label: lang-python

```python
from n4m.sklearn import Normalize

op = Normalize(feature_min=-1.0, feature_max=1.0)
Xt = op.fit_transform(X)
```

:::

:::{tab-item} R · nirs4all-methods
:sync: r
:class-label: lang-r

```r
library(n4m)
res <- normalize(X, feature_min = -1, feature_max = 1)
```

:::

::::


**Benchmark Comparators And Sources** ◆

:::{card}
:class-card: external-refs

- ◆ **`ref.nirs4all`** (Python · canonical) — `nirs4all.Normalize` · nirs4all@cd731a23+dirty
:::

### Validation contract

- Operation: `cross_binding_callable` · comparator: `default_allclose` · tolerance: `rtol=1e-05`, `atol=1e-08` · quality: **strict**
- Default validation dataset: `100×50` · seed `20260556`
- Suites: smoke `3` cells; benchmark `11` cells · Default C/Python/reference parity comparator.
- Metrics: `max_abs_diff`, `rel_l2_diff`, `rms_diff`, `shape_equal`
- Truth sources: cross-binding references declared directly in `benchmarks/cross_binding/orchestrator.py`.

| Backend | Library | Gate | Comparator | Note |
|---------|---------|------|------------|------|
| `ref.nirs4all` | `nirs4all.Normalize` | Python / parity | `default_allclose` |  |

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
<tr class="bk-row"><td class="bk-name"><code>N4M.cpp</code></td><td class="parity parity-divergence parity-exact" title="worst reference max abs diff over visible sizes">2.8e-17</td><td class="ms ms-best">🏆 0.002 ms</td><td class="ms ms-best">🏆 0.019 ms</td><td class="ms ms-best">🏆 0.119 ms</td></tr>
</tbody>
<tbody class="lang-band lang-python"><tr class="lang-band-row" data-lang="python"><th colspan="5" scope="rowgroup"><span class="lang-band-dot"></span>Python · nirs4all-methods</th></tr>
<tr class="bk-row"><td class="bk-name"><code>N4M.python</code></td><td class="parity parity-divergence parity-exact" title="worst reference max abs diff over visible sizes">2.8e-17</td><td class="ms">0.007 ms</td><td class="ms">0.026 ms</td><td class="ms">0.132 ms</td></tr>
<tr class="bk-row"><td class="bk-name"><code>N4M.sklearn</code></td><td class="parity parity-divergence parity-exact" title="worst reference max abs diff over visible sizes">2.8e-17</td><td class="ms">0.010 ms</td><td class="ms">0.028 ms</td><td class="ms">0.155 ms</td></tr>
</tbody>
<tbody class="lang-band lang-r"><tr class="lang-band-row" data-lang="r"><th colspan="5" scope="rowgroup"><span class="lang-band-dot"></span>R · nirs4all-methods</th></tr>
<tr class="bk-row"><td class="bk-name"><code>N4M.R</code></td><td class="parity parity-divergence parity-exact" title="worst reference max abs diff over visible sizes">5.1e-16</td><td class="ms">0.028 ms</td><td class="ms">0.270 ms</td><td class="ms">1.953 ms</td></tr>
</tbody>
<tbody class="lang-band lang-python"><tr class="lang-band-row" data-lang="python"><th colspan="5" scope="rowgroup"><span class="lang-band-dot"></span>Python · external</th></tr>
<tr class="bk-row truth-source-strict"><td class="bk-name"><span class="truth-mark" title="Registry parity reference (Python): nirs4all.Normalize · nirs4all@cd731a23+dirty — canonical">◆</span><code>ref.nirs4all</code></td><td class="parity parity-divergence parity-exact" title="worst reference max abs diff over visible sizes">0</td><td class="ms">0.016 ms</td><td class="ms">0.073 ms</td><td class="ms">0.533 ms</td></tr>
</tbody>
</table>
</div>

:::
::::

---

_See also_: [benchmark overview](../benchmarks/overview.md) · [methods index](index.md) · [interactive dashboard](../landing/dashboard.md)
