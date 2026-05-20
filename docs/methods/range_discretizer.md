# `range_discretizer` — Range discretizer

_Group_: **Resampling** · _Registry tolerance_: `rtol=1e-5`, `atol=1e-8`

## Description

From the `n4m.RangeDiscretizer` Python wrapper docstring:

> Integer binning against monotonic numeric edges.

### Parameters

| Name | Type | Default | Notes |
|------|------|---------|-------|
| `edges` | `Sequence[float] \| None` | `None` |  |
| `edges_csv` | `str \| None` | `None` |  |

## Explanations

### Bibliographic source

- `ref.nirs4all` — nirs4all.RangeDiscretizer (Python).

### Mathematical principle

`range_discretizer` follows the standard resampling operator contract: an input matrix $\mathbf{X}\in\mathbb{R}^{n\times p}$ is transformed, scored, split, or filtered by the method-specific kernel while preserving the parity contract recorded by the benchmark snapshots. The precise numerical convention is the one exposed by the C ABI signatures and the registered external references below.

### Implementation

C ABI entry points used by the language bindings:

```c
n4m_status_t n4m_pp_range_disc_create(n4m_pp_range_disc_handle_t** out, const double* bins, int64_t n_edges);
void n4m_pp_range_disc_destroy(n4m_pp_range_disc_handle_t* h);
n4m_status_t n4m_pp_range_disc_transform( const n4m_pp_range_disc_handle_t* h, n4m_matrix_view_t X, n4m_matrix_view_t out);
```

Benchmark comparator backends are registered in the matrix and stored as reproducible snapshots when they define the canonical contract.

### Implementations

| Layer | Entry point | Language | Contract |
|-------|-------------|----------|----------|
| C ABI | `n4m_pp_range_disc` | C/C++ | Stable libn4m entry point family. |
| Python | `n4m.python.range_discretizer` | Python | ABI-close function backed by ctypes. |
| Python sklearn | `n4m.sklearn.RangeDiscretizer` | Python | scikit-learn-compatible estimator backed by ctypes. |
| R | `range_discretizer(X, edges)` | R | Public package wrapper around the C ABI. |
| ref.nirs4all | `nirs4all.RangeDiscretizer` | Python | canonical/comparator |

### Usage

Every nirs4all-methods binding dispatches into the same C kernel. Registered comparator/source rows are listed in the benchmark card below.

::::{tab-set}
:class: nirs4all-methods-bindings


:::{tab-item} C ABI · libn4m
:sync: c
:class-label: lang-c

```c
n4m_status_t n4m_pp_range_disc_create(n4m_pp_range_disc_handle_t** out, const double* bins, int64_t n_edges);
void n4m_pp_range_disc_destroy(n4m_pp_range_disc_handle_t* h);
n4m_status_t n4m_pp_range_disc_transform( const n4m_pp_range_disc_handle_t* h, n4m_matrix_view_t X, n4m_matrix_view_t out);
```

:::

:::{tab-item} Python ABI · n4m.python
:sync: python-abi
:class-label: lang-python

```python
from n4m import python as n4m

Xi = n4m.range_discretizer(X, edges=[0.0, 0.25, 0.5, 1.0])
```

:::

:::{tab-item} Python sklearn · n4m.sklearn
:sync: python-sklearn
:class-label: lang-python

```python
from n4m.sklearn import RangeDiscretizer

op = RangeDiscretizer(edges=None, edges_csv=None)
Xt = op.fit_transform(X)
```

:::

:::{tab-item} R · nirs4all-methods
:sync: r
:class-label: lang-r

```r
library(n4m)
res <- range_discretizer(X, edges = c(0.25, 0.40, 0.55, 0.70))
```

:::

::::


**Benchmark Comparators And Sources** ◆

:::{card}
:class-card: external-refs

- ◆ **`ref.nirs4all`** (Python · canonical) — `nirs4all.RangeDiscretizer` · nirs4all@cd731a23+dirty
:::

### Validation contract

- Operation: `cross_binding_callable` · comparator: `default_allclose` · tolerance: `rtol=1e-05`, `atol=1e-08` · quality: **strict**
- Default validation dataset: `100×50` · seed `20260556`
- Suites: smoke `3` cells; benchmark `11` cells · Default C/Python/reference parity comparator.
- Metrics: `max_abs_diff`, `rel_l2_diff`, `rms_diff`, `shape_equal`
- Truth sources: cross-binding references declared directly in `benchmarks/cross_binding/orchestrator.py`.

| Backend | Library | Gate | Comparator | Note |
|---------|---------|------|------------|------|
| `ref.nirs4all` | `nirs4all.RangeDiscretizer` | Python / parity | `default_allclose` |  |

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
<tr class="bk-row"><td class="bk-name"><code>N4M.cpp</code></td><td class="parity parity-divergence parity-exact" title="worst reference max abs diff over visible sizes">0</td><td class="ms ms-best">🏆 0.019 ms</td><td class="ms">0.098 ms</td><td class="ms ms-best">🏆 0.427 ms</td></tr>
</tbody>
<tbody class="lang-band lang-python"><tr class="lang-band-row" data-lang="python"><th colspan="5" scope="rowgroup"><span class="lang-band-dot"></span>Python · nirs4all-methods</th></tr>
<tr class="bk-row"><td class="bk-name"><code>N4M.python</code></td><td class="parity parity-divergence parity-exact" title="worst reference max abs diff over visible sizes">0</td><td class="ms">0.021 ms</td><td class="ms ms-best">🏆 0.095 ms</td><td class="ms">0.436 ms</td></tr>
<tr class="bk-row"><td class="bk-name"><code>N4M.sklearn</code></td><td class="parity parity-divergence parity-exact" title="worst reference max abs diff over visible sizes">0</td><td class="ms">0.022 ms</td><td class="ms">0.104 ms</td><td class="ms">0.445 ms</td></tr>
</tbody>
<tbody class="lang-band lang-r"><tr class="lang-band-row" data-lang="r"><th colspan="5" scope="rowgroup"><span class="lang-band-dot"></span>R · nirs4all-methods</th></tr>
<tr class="bk-row"><td class="bk-name"><code>N4M.R</code></td><td class="parity parity-divergence parity-exact" title="worst reference max abs diff over visible sizes">0</td><td class="ms">0.049 ms</td><td class="ms">0.352 ms</td><td class="ms">1.828 ms</td></tr>
</tbody>
<tbody class="lang-band lang-python"><tr class="lang-band-row" data-lang="python"><th colspan="5" scope="rowgroup"><span class="lang-band-dot"></span>Python · external</th></tr>
<tr class="bk-row truth-source-strict"><td class="bk-name"><span class="truth-mark" title="Registry parity reference (Python): nirs4all.RangeDiscretizer · nirs4all@cd731a23+dirty — canonical">◆</span><code>ref.nirs4all</code></td><td class="parity parity-divergence parity-exact" title="worst reference max abs diff over visible sizes">0</td><td class="ms">0.022 ms</td><td class="ms">0.279 ms</td><td class="ms">1.487 ms</td></tr>
</tbody>
</table>
</div>

:::
::::

---

_See also_: [benchmark overview](../benchmarks/overview.md) · [methods index](index.md) · [interactive dashboard](../landing/dashboard.md)
