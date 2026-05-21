# `kbins_discretizer` — K-bins discretizer

_Group_: **Resampling** · _Registry tolerance_: `rtol=1e-5`, `atol=1e-8`

## Description

From the `n4m.IntegerKBinsDiscretizer` Python wrapper docstring:

> Per-column integer binning using uniform or quantile edges.

### Parameters

| Name | Type | Default | Notes |
|------|------|---------|-------|
| `n_bins` | `int` | `5` | Number of bins. |
| `strategy` | `str \| int` | `0` | Binning strategy. |

## Explanations

### Bibliographic source

- `ref.sklearn` — sklearn.preprocessing.KBinsDiscretizer (Python).
- `ref.nirs4all` — nirs4all.IntegerKBinsDiscretizer (Python).

### Mathematical principle

`kbins_discretizer` follows the standard resampling operator contract: an input matrix $\mathbf{X}\in\mathbb{R}^{n\times p}$ is transformed, scored, split, or filtered by the method-specific kernel while preserving the parity contract recorded by the benchmark snapshots. The precise numerical convention is the one exposed by the C ABI signatures and the registered external references below.

### Implementation

C ABI entry points used by the language bindings:

```c
n4m_status_t n4m_pp_kbins_disc_create(n4m_pp_kbins_disc_handle_t** out, int32_t n_bins, int32_t strategy);
void n4m_pp_kbins_disc_destroy(n4m_pp_kbins_disc_handle_t* h);
n4m_status_t n4m_pp_kbins_disc_fit(n4m_pp_kbins_disc_handle_t* h, n4m_matrix_view_t X);
n4m_status_t n4m_pp_kbins_disc_is_fitted( const n4m_pp_kbins_disc_handle_t* h, int* out_fitted);
n4m_status_t n4m_pp_kbins_disc_transform( const n4m_pp_kbins_disc_handle_t* h, n4m_matrix_view_t X, n4m_matrix_view_t out);
```

Benchmark comparator backends are registered in the matrix and stored as reproducible snapshots when they define the canonical contract.

### Implementations

| Layer | Entry point | Language | Contract |
|-------|-------------|----------|----------|
| C ABI | `n4m_pp_kbins_disc` | C/C++ | Stable libn4m entry point family. |
| Python | `n4m.python.kbins_discretizer` | Python | ABI-close function backed by ctypes. |
| Python sklearn | `n4m.sklearn.IntegerKBinsDiscretizer` | Python | scikit-learn-compatible estimator backed by ctypes. |
| R | `kbins_discretizer(X, n_bins = 5L, strategy = "uniform")` | R | Public package wrapper around the C ABI. |
| ref.sklearn | `sklearn.preprocessing.KBinsDiscretizer` | Python | canonical/comparator |
| ref.nirs4all | `nirs4all.IntegerKBinsDiscretizer` | Python | canonical/comparator |

### Usage

Every nirs4all-methods binding dispatches into the same C kernel. Registered comparator/source rows are listed in the benchmark card below.

::::{tab-set}
:class: nirs4all-methods-bindings


:::{tab-item} C ABI · libn4m
:sync: c
:class-label: lang-c

```c
n4m_status_t n4m_pp_kbins_disc_create(n4m_pp_kbins_disc_handle_t** out, int32_t n_bins, int32_t strategy);
void n4m_pp_kbins_disc_destroy(n4m_pp_kbins_disc_handle_t* h);
n4m_status_t n4m_pp_kbins_disc_fit(n4m_pp_kbins_disc_handle_t* h, n4m_matrix_view_t X);
n4m_status_t n4m_pp_kbins_disc_is_fitted( const n4m_pp_kbins_disc_handle_t* h, int* out_fitted);
n4m_status_t n4m_pp_kbins_disc_transform( const n4m_pp_kbins_disc_handle_t* h, n4m_matrix_view_t X, n4m_matrix_view_t out);
```

:::

:::{tab-item} Python ABI · n4m.python
:sync: python-abi
:class-label: lang-python

```python
from n4m import python as n4m

Xt = n4m.kbins_discretizer(X)
```

:::

:::{tab-item} Python sklearn · n4m.sklearn
:sync: python-sklearn
:class-label: lang-python

```python
from n4m.sklearn import IntegerKBinsDiscretizer

op = IntegerKBinsDiscretizer(n_bins=5, strategy=0)
Xt = op.fit_transform(X)
```

:::

:::{tab-item} R · nirs4all-methods
:sync: r
:class-label: lang-r

```r
library(n4m)
res <- kbins_discretizer(X, n_bins = 5L, strategy = 'uniform')
```

:::

::::


**Benchmark Comparators And Sources** ◆

:::{card}
:class-card: external-refs

- ◆ **`ref.sklearn`** (Python · canonical) — `sklearn.preprocessing.KBinsDiscretizer` · scikit-learn 1.8.0
- ◆ **`ref.nirs4all`** (Python · comparator) — `nirs4all.IntegerKBinsDiscretizer` · nirs4all@cd731a23+dirty
:::

### Validation contract

- Operation: `cross_binding_callable` · comparator: `default_allclose` · tolerance: `rtol=1e-05`, `atol=1e-08` · quality: **strict**
- Default validation dataset: `100×50` · seed `20260556`
- Suites: smoke `3` cells; benchmark `11` cells · Default C/Python/reference parity comparator.
- Metrics: `max_abs_diff`, `rel_l2_diff`, `rms_diff`, `shape_equal`
- Truth sources: cross-binding references declared directly in `benchmarks/cross_binding/orchestrator.py`.

| Backend | Library | Gate | Comparator | Note |
|---------|---------|------|------------|------|
| `ref.sklearn` | `sklearn.preprocessing.KBinsDiscretizer` | Python / parity | `default_allclose` |  |
| `ref.nirs4all` | `nirs4all.IntegerKBinsDiscretizer` | Python / parity | `default_allclose` |  |

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
<tr class="bk-row"><td class="bk-name"><code>N4M.cpp</code></td><td class="parity parity-divergence parity-exact" title="worst reference max abs diff over visible sizes">0</td><td class="ms">0.028 ms</td><td class="ms">0.344 ms</td><td class="ms ms-best">🏆 1.506 ms</td></tr>
</tbody>
<tbody class="lang-band lang-python"><tr class="lang-band-row" data-lang="python"><th colspan="5" scope="rowgroup"><span class="lang-band-dot"></span>Python · nirs4all-methods</th></tr>
<tr class="bk-row"><td class="bk-name"><code>N4M.python</code></td><td class="parity parity-divergence parity-exact" title="worst reference max abs diff over visible sizes">0</td><td class="ms ms-best">🏆 0.027 ms</td><td class="ms ms-best">🏆 0.336 ms</td><td class="ms">1.732 ms</td></tr>
<tr class="bk-row"><td class="bk-name"><code>N4M.sklearn</code></td><td class="parity parity-divergence parity-exact" title="worst reference max abs diff over visible sizes">0</td><td class="ms">0.035 ms</td><td class="ms">0.344 ms</td><td class="ms">1.601 ms</td></tr>
</tbody>
<tbody class="lang-band lang-r"><tr class="lang-band-row" data-lang="r"><th colspan="5" scope="rowgroup"><span class="lang-band-dot"></span>R · nirs4all-methods</th></tr>
<tr class="bk-row"><td class="bk-name"><code>N4M.R</code></td><td class="parity parity-divergence parity-exact" title="worst reference max abs diff over visible sizes">0</td><td class="ms">0.045 ms</td><td class="ms">0.629 ms</td><td class="ms">3.266 ms</td></tr>
</tbody>
<tbody class="lang-band lang-python"><tr class="lang-band-row" data-lang="python"><th colspan="5" scope="rowgroup"><span class="lang-band-dot"></span>Python · external</th></tr>
<tr class="bk-row truth-source-relaxed"><td class="bk-name"><span class="truth-mark" title="Registry parity reference (Python): nirs4all.IntegerKBinsDiscretizer · nirs4all@cd731a23+dirty — comparator">◆</span><code>ref.nirs4all</code></td><td class="parity parity-divergence parity-exact" title="worst reference max abs diff over visible sizes">0</td><td class="ms">0.500 ms</td><td class="ms">3.846 ms</td><td class="ms">15.810 ms</td></tr>
<tr class="bk-row truth-source-strict"><td class="bk-name"><span class="truth-mark" title="Registry parity reference (Python): sklearn.preprocessing.KBinsDiscretizer · scikit-learn 1.8.0 — canonical">◆</span><code>ref.sklearn</code></td><td class="parity parity-divergence parity-exact" title="worst reference max abs diff over visible sizes">0</td><td class="ms">0.410 ms</td><td class="ms">3.714 ms</td><td class="ms">16.892 ms</td></tr>
</tbody>
</table>
</div>

:::
::::

---

_See also_: [benchmark overview](../benchmarks/overview.md) · [methods index](index.md) · [interactive dashboard](../landing/dashboard.md)
