# `simple_scale` — Simple scale

_Group_: **Preprocessing** · _Registry tolerance_: `rtol=1e-5`, `atol=1e-8` · _Source_: [`docs/algorithms/simple_scale.md`](../algorithms/simple_scale.md)

## Description

From the `n4m.SimpleScale` Python wrapper docstring:

> Column-wise min-max scaling to ``[0, 1]``.

### Parameters

None — the transform is fully data-driven.

## Explanations

### Bibliographic source

Implementation matches `nirs4all.operators.transforms.scalers.SimpleScale`.

### Mathematical principle

Column-wise (axis=0) min-max scaling to the $[0, 1]$ range. For each column $j$:

$$
\mu_j = \min_i x_{i,j}, \qquad
M_j = \max_i x_{i,j}, \qquad
x'_{i,j} = \frac{x_{i,j} - \mu_j}{M_j - \mu_j}
$$

This is the special case of [Normalize](normalize.md) with `feature_range = (0, 1)` written as a dedicated operator for performance and API symmetry.

### Implementation

* Per-column min/max found via a single left-to-right scan.
* Final scaling uses `(x - min) / (max - min)` — a single subtraction and a single division per element, matching nirs4all.

C ABI entry points used by the language bindings:

```c
n4m_status_t n4m_pp_simple_scale_create( n4m_pp_simple_scale_handle_t** out);
void n4m_pp_simple_scale_destroy( n4m_pp_simple_scale_handle_t* handle);
n4m_status_t n4m_pp_simple_scale_transform( const n4m_pp_simple_scale_handle_t* handle, n4m_matrix_view_t X, n4m_matrix_view_t out);
```

Benchmark comparator backends are registered in the matrix and stored as reproducible snapshots when they define the canonical contract.

### Implementations

| Layer | Entry point | Language | Contract |
|-------|-------------|----------|----------|
| C ABI | `n4m_pp_simple_scale` | C/C++ | Stable libn4m entry point family. |
| Python | `n4m.python.simple_scale` | Python | ABI-close function backed by ctypes. |
| Python sklearn | `n4m.sklearn.SimpleScale` | Python | scikit-learn-compatible estimator backed by ctypes. |
| R | `simple_scale(X)` | R | Public package wrapper around the C ABI. |
| ref.sklearn | `sklearn.preprocessing.MinMaxScaler` | Python | canonical/comparator |
| ref.nirs4all | `nirs4all.SimpleScale` | Python | canonical/comparator |

### Usage

Every nirs4all-methods binding dispatches into the same C kernel. Registered comparator/source rows are listed in the benchmark card below.

::::{tab-set}
:class: nirs4all-methods-bindings


:::{tab-item} C ABI · libn4m
:sync: c
:class-label: lang-c

```c
n4m_status_t n4m_pp_simple_scale_create( n4m_pp_simple_scale_handle_t** out);
void n4m_pp_simple_scale_destroy( n4m_pp_simple_scale_handle_t* handle);
n4m_status_t n4m_pp_simple_scale_transform( const n4m_pp_simple_scale_handle_t* handle, n4m_matrix_view_t X, n4m_matrix_view_t out);
```

:::

:::{tab-item} Python ABI · n4m.python
:sync: python-abi
:class-label: lang-python

```python
from n4m import python as n4m

Xt = n4m.simple_scale(X)
```

:::

:::{tab-item} Python sklearn · n4m.sklearn
:sync: python-sklearn
:class-label: lang-python

```python
from n4m.sklearn import SimpleScale

op = SimpleScale()
Xt = op.fit_transform(X)
```

:::

:::{tab-item} R · nirs4all-methods
:sync: r
:class-label: lang-r

```r
library(n4m)
res <- simple_scale(X)
```

:::

::::


**Benchmark Comparators And Sources** ◆

:::{card}
:class-card: external-refs

- ◆ **`ref.sklearn`** (Python · canonical) — `sklearn.preprocessing.MinMaxScaler` · scikit-learn 1.8.0
- ◆ **`ref.nirs4all`** (Python · comparator) — `nirs4all.SimpleScale` · nirs4all@cd731a23+dirty
:::

### Validation contract

- Operation: `cross_binding_callable` · comparator: `default_allclose` · tolerance: `rtol=1e-05`, `atol=1e-08` · quality: **strict**
- Default validation dataset: `100×50` · seed `20260556`
- Suites: smoke `3` cells; benchmark `11` cells · Default C/Python/reference parity comparator.
- Metrics: `max_abs_diff`, `rel_l2_diff`, `rms_diff`, `shape_equal`
- Truth sources: cross-binding references declared directly in `benchmarks/cross_binding/orchestrator.py`.

| Backend | Library | Gate | Comparator | Note |
|---------|---------|------|------------|------|
| `ref.sklearn` | `sklearn.preprocessing.MinMaxScaler` | Python / parity | `default_allclose` |  |
| `ref.nirs4all` | `nirs4all.SimpleScale` | Python / parity | `default_allclose` |  |

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
<tr class="bk-row"><td class="bk-name"><code>N4M.cpp</code></td><td class="parity parity-divergence parity-exact" title="worst reference max abs diff over visible sizes">4.4e-16</td><td class="ms ms-best">🏆 0.005 ms</td><td class="ms ms-best">🏆 0.044 ms</td><td class="ms ms-best">🏆 0.247 ms</td></tr>
</tbody>
<tbody class="lang-band lang-python"><tr class="lang-band-row" data-lang="python"><th colspan="5" scope="rowgroup"><span class="lang-band-dot"></span>Python · nirs4all-methods</th></tr>
<tr class="bk-row"><td class="bk-name"><code>N4M.python</code></td><td class="parity parity-divergence parity-exact" title="worst reference max abs diff over visible sizes">4.4e-16</td><td class="ms">0.010 ms</td><td class="ms">0.046 ms</td><td class="ms">0.260 ms</td></tr>
<tr class="bk-row"><td class="bk-name"><code>N4M.sklearn</code></td><td class="parity parity-divergence parity-exact" title="worst reference max abs diff over visible sizes">4.4e-16</td><td class="ms">0.013 ms</td><td class="ms">0.051 ms</td><td class="ms">0.253 ms</td></tr>
</tbody>
<tbody class="lang-band lang-r"><tr class="lang-band-row" data-lang="r"><th colspan="5" scope="rowgroup"><span class="lang-band-dot"></span>R · nirs4all-methods</th></tr>
<tr class="bk-row"><td class="bk-name"><code>N4M.R</code></td><td class="parity parity-divergence parity-exact" title="worst reference max abs diff over visible sizes">9.4e-16</td><td class="ms">0.031 ms</td><td class="ms">0.297 ms</td><td class="ms">2.062 ms</td></tr>
</tbody>
<tbody class="lang-band lang-python"><tr class="lang-band-row" data-lang="python"><th colspan="5" scope="rowgroup"><span class="lang-band-dot"></span>Python · external</th></tr>
<tr class="bk-row truth-source-relaxed"><td class="bk-name"><span class="truth-mark" title="Registry parity reference (Python): nirs4all.SimpleScale · nirs4all@cd731a23+dirty — comparator">◆</span><code>ref.nirs4all</code></td><td class="parity parity-divergence parity-exact" title="worst reference max abs diff over visible sizes">5.6e-16</td><td class="ms">0.021 ms</td><td class="ms">0.100 ms</td><td class="ms">0.585 ms</td></tr>
<tr class="bk-row truth-source-strict"><td class="bk-name"><span class="truth-mark" title="Registry parity reference (Python): sklearn.preprocessing.MinMaxScaler · scikit-learn 1.8.0 — canonical">◆</span><code>ref.sklearn</code></td><td class="parity parity-divergence parity-exact" title="worst reference max abs diff over visible sizes">0</td><td class="ms">0.124 ms</td><td class="ms">0.228 ms</td><td class="ms">0.663 ms</td></tr>
</tbody>
</table>
</div>

:::
::::

---

_See also_: [benchmark overview](../benchmarks/overview.md) · [methods index](index.md) · [interactive dashboard](../landing/dashboard.md)
