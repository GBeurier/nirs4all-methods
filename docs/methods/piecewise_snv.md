# `piecewise_snv` — Piecewise SNV

_Group_: **Preprocessing** · _Registry tolerance_: `rtol=1e-5`, `atol=1e-8`

## Description

From the `n4m.PiecewiseSNV` Python wrapper docstring:

> Apply SNV independently inside fixed wavelength intervals.

### Parameters

| Name | Type | Default | Notes |
|------|------|---------|-------|
| `window_size` | `int` | `32` |  |
| `ddof` | `int` | `0` |  |
| `eps` | `float` | `1e-12` |  |

## Explanations

### Bibliographic source

- `ref.nirs4all` — nirs4all.StandardNormalVariate(ddof=0) per interval (Python).

### Mathematical principle

`piecewise_snv` follows the standard preprocessing operator contract: an input matrix $\mathbf{X}\in\mathbb{R}^{n\times p}$ is transformed, scored, split, or filtered by the method-specific kernel while preserving the parity contract recorded by the benchmark snapshots. The precise numerical convention is the one exposed by the C ABI signatures and the registered external references below.

### Implementation

C ABI entry points used by the language bindings:

```c
n4m_status_t n4m_pp_piecewise_snv_create( n4m_pp_piecewise_snv_handle_t** out, int32_t window, int32_t ddof, double eps);
void n4m_pp_piecewise_snv_destroy(n4m_pp_piecewise_snv_handle_t* handle);
n4m_status_t n4m_pp_piecewise_snv_fit( n4m_pp_piecewise_snv_handle_t* handle, n4m_matrix_view_t X);
n4m_status_t n4m_pp_piecewise_snv_is_fitted( const n4m_pp_piecewise_snv_handle_t* handle, int* out_fitted);
n4m_status_t n4m_pp_piecewise_snv_transform( const n4m_pp_piecewise_snv_handle_t* handle, n4m_matrix_view_t X, n4m_matrix_view_t out);
```

Benchmark comparator backends are registered in the matrix and stored as reproducible snapshots when they define the canonical contract.

### Implementations

| Layer | Entry point | Language | Contract |
|-------|-------------|----------|----------|
| C ABI | `n4m_pp_piecewise_snv` | C/C++ | Stable libn4m entry point family. |
| Python | `n4m.python.piecewise_snv` | Python | ABI-close function backed by ctypes. |
| Python sklearn | `n4m.sklearn.PiecewiseSNV` | Python | scikit-learn-compatible estimator backed by ctypes. |
| R | `piecewise_snv(X, window = 16L, ddof = 0L, eps = 1e-12)` | R | Public package wrapper around the C ABI. |
| ref.nirs4all | `nirs4all.StandardNormalVariate(ddof=0) per interval` | Python | canonical/comparator; nirs4all supplies the SNV transform; the adapter applies it independently to N4M intervals |

### Usage

Every nirs4all-methods binding dispatches into the same C kernel. Registered comparator/source rows are listed in the benchmark card below.

::::{tab-set}
:class: nirs4all-methods-bindings


:::{tab-item} C ABI · libn4m
:sync: c
:class-label: lang-c

```c
n4m_status_t n4m_pp_piecewise_snv_create( n4m_pp_piecewise_snv_handle_t** out, int32_t window, int32_t ddof, double eps);
void n4m_pp_piecewise_snv_destroy(n4m_pp_piecewise_snv_handle_t* handle);
n4m_status_t n4m_pp_piecewise_snv_fit( n4m_pp_piecewise_snv_handle_t* handle, n4m_matrix_view_t X);
n4m_status_t n4m_pp_piecewise_snv_is_fitted( const n4m_pp_piecewise_snv_handle_t* handle, int* out_fitted);
n4m_status_t n4m_pp_piecewise_snv_transform( const n4m_pp_piecewise_snv_handle_t* handle, n4m_matrix_view_t X, n4m_matrix_view_t out);
```

:::

:::{tab-item} Python ABI · n4m.python
:sync: python-abi
:class-label: lang-python

```python
from n4m import python as n4m

Xt = n4m.piecewise_snv(X)
```

:::

:::{tab-item} Python sklearn · n4m.sklearn
:sync: python-sklearn
:class-label: lang-python

```python
from n4m.sklearn import PiecewiseSNV

op = PiecewiseSNV(window_size=32, ddof=0, eps=1e-12)
Xt = op.fit_transform(X)
```

:::

:::{tab-item} R · nirs4all-methods
:sync: r
:class-label: lang-r

```r
library(n4m)
res <- piecewise_snv(X, window = 16L, ddof = 0L, eps = 1e-12)
```

:::

::::


**Benchmark Comparators And Sources** ◆

:::{card}
:class-card: external-refs

- ◆ **`ref.nirs4all`** (Python · canonical) — `nirs4all.StandardNormalVariate(ddof=0) per interval` · nirs4all@cd731a23+dirty — nirs4all supplies the SNV transform; the adapter applies it independently to N4M intervals
:::

### Validation contract

- Operation: `fit_transform` · comparator: `default_allclose` · tolerance: `rtol=1e-05`, `atol=1e-08` · quality: **strict**
- Default validation dataset: `64×128` · seed `1234567906`
- Suites: smoke `3` cells; benchmark `11` cells · Default C/Python/reference parity comparator.
- Metrics: `finite_output`, `max_abs_diff`, `shape_equal`
- Truth sources:
  - `local_scatter_2026` — Demystifying Piecewise and Localized Scatter Correction Methods (doi:10.1002/cem.70113)

| Backend | Library | Gate | Comparator | Note |
|---------|---------|------|------------|------|
| `ref.nirs4all` | `nirs4all.StandardNormalVariate(ddof=0) per interval` | Python / parity | `default_allclose` | nirs4all supplies the SNV transform; the adapter applies it independently to N4M intervals |

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
<tr class="bk-row"><td class="bk-name"><code>N4M.cpp</code></td><td class="parity parity-divergence parity-exact" title="worst reference max abs diff over visible sizes">7.9e-14</td><td class="ms">0.012 ms</td><td class="ms">0.057 ms</td><td class="ms">0.263 ms</td></tr>
</tbody>
<tbody class="lang-band lang-python"><tr class="lang-band-row" data-lang="python"><th colspan="5" scope="rowgroup"><span class="lang-band-dot"></span>Python · nirs4all-methods</th></tr>
<tr class="bk-row"><td class="bk-name"><code>N4M.python</code></td><td class="parity parity-divergence parity-exact" title="worst reference max abs diff over visible sizes">7.9e-14</td><td class="ms ms-best">🏆 0.012 ms</td><td class="ms ms-best">🏆 0.056 ms</td><td class="ms ms-best">🏆 0.249 ms</td></tr>
<tr class="bk-row"><td class="bk-name"><code>N4M.sklearn</code></td><td class="parity parity-divergence parity-exact" title="worst reference max abs diff over visible sizes">7.9e-14</td><td class="ms">0.014 ms</td><td class="ms">0.061 ms</td><td class="ms">0.274 ms</td></tr>
</tbody>
<tbody class="lang-band lang-r"><tr class="lang-band-row" data-lang="r"><th colspan="5" scope="rowgroup"><span class="lang-band-dot"></span>R · nirs4all-methods</th></tr>
<tr class="bk-row"><td class="bk-name"><code>N4M.R</code></td><td class="parity parity-divergence parity-exact" title="worst reference max abs diff over visible sizes">8.1e-14</td><td class="ms">0.031 ms</td><td class="ms">0.320 ms</td><td class="ms">1.930 ms</td></tr>
</tbody>
<tbody class="lang-band lang-python"><tr class="lang-band-row" data-lang="python"><th colspan="5" scope="rowgroup"><span class="lang-band-dot"></span>Python · external</th></tr>
<tr class="bk-row truth-source-strict"><td class="bk-name"><span class="truth-mark" title="Registry parity reference (Python): nirs4all.StandardNormalVariate(ddof=0) per interval · nirs4all@cd731a23+dirty — canonical">◆</span><code>ref.nirs4all</code></td><td class="parity parity-divergence parity-exact" title="worst reference max abs diff over visible sizes">0</td><td class="ms">0.245 ms</td><td class="ms">2.064 ms</td><td class="ms">9.806 ms</td></tr>
</tbody>
</table>
</div>

:::
::::

---

_See also_: [benchmark overview](../benchmarks/overview.md) · [methods index](index.md) · [interactive dashboard](../landing/dashboard.md)
