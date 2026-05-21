# `correlation_optimized_warping` — Correlation optimized warping

_Group_: **Alignment** · _Registry tolerance_: `rtol=1e-5`, `atol=1e-8`

## Description

From the `n4m.CorrelationOptimizedWarping` Python wrapper docstring:

> Segment-wise correlation optimized warping approximation.

### Parameters

| Name | Type | Default | Notes |
|------|------|---------|-------|
| `reference` | `object` | `None` |  |
| `interval_size` | `int` | `32` |  |
| `max_shift` | `int` | `5` |  |

## Explanations

### Bibliographic source

- `ref.cowarp` — cowarp.warp(segment_length=16, slack=2) (Python).

### Mathematical principle

`correlation_optimized_warping` follows the standard alignment operator contract: an input matrix $\mathbf{X}\in\mathbb{R}^{n\times p}$ is transformed, scored, split, or filtered by the method-specific kernel while preserving the parity contract recorded by the benchmark snapshots. The precise numerical convention is the one exposed by the C ABI signatures and the registered external references below.

### Implementation

C ABI entry points used by the language bindings:

```c
n4m_status_t n4m_pp_cow_align_create( n4m_pp_cow_align_handle_t** out, const double* reference, int64_t n_reference, int32_t interval_size, int32_t max_shift);
void n4m_pp_cow_align_destroy(n4m_pp_cow_align_handle_t* handle);
n4m_status_t n4m_pp_cow_align_fit( n4m_pp_cow_align_handle_t* handle, n4m_matrix_view_t X);
n4m_status_t n4m_pp_cow_align_is_fitted( const n4m_pp_cow_align_handle_t* handle, int* out_fitted);
n4m_status_t n4m_pp_cow_align_transform( const n4m_pp_cow_align_handle_t* handle, n4m_matrix_view_t X, n4m_matrix_view_t out);
```

Benchmark comparator backends are registered in the matrix and stored as reproducible snapshots when they define the canonical contract.

### Implementations

| Layer | Entry point | Language | Contract |
|-------|-------------|----------|----------|
| C ABI | `n4m_pp_cow_align` | C/C++ | Stable libn4m entry point family. |
| Python | `n4m.python.correlation_optimized_warping` | Python | ABI-close function backed by ctypes. |
| Python sklearn | `n4m.sklearn.CorrelationOptimizedWarping` | Python | scikit-learn-compatible estimator backed by ctypes. |
| R | `correlation_optimized_warping(X, reference = NULL, interval_size = 16L, max_shift = 3L)` | R | Public package wrapper around the C ABI. |
| ref.cowarp | `cowarp.warp(segment_length=16, slack=2)` | Python | canonical/comparator; cowarp provides the external dynamic-programming COW contract used by this gate |

### Usage

Every nirs4all-methods binding dispatches into the same C kernel. Registered comparator/source rows are listed in the benchmark card below.

::::{tab-set}
:class: nirs4all-methods-bindings


:::{tab-item} C ABI · libn4m
:sync: c
:class-label: lang-c

```c
n4m_status_t n4m_pp_cow_align_create( n4m_pp_cow_align_handle_t** out, const double* reference, int64_t n_reference, int32_t interval_size, int32_t max_shift);
void n4m_pp_cow_align_destroy(n4m_pp_cow_align_handle_t* handle);
n4m_status_t n4m_pp_cow_align_fit( n4m_pp_cow_align_handle_t* handle, n4m_matrix_view_t X);
n4m_status_t n4m_pp_cow_align_is_fitted( const n4m_pp_cow_align_handle_t* handle, int* out_fitted);
n4m_status_t n4m_pp_cow_align_transform( const n4m_pp_cow_align_handle_t* handle, n4m_matrix_view_t X, n4m_matrix_view_t out);
```

:::

:::{tab-item} Python ABI · n4m.python
:sync: python-abi
:class-label: lang-python

```python
from n4m import python as n4m

Xt = n4m.correlation_optimized_warping(X)
```

:::

:::{tab-item} Python sklearn · n4m.sklearn
:sync: python-sklearn
:class-label: lang-python

```python
from n4m.sklearn import CorrelationOptimizedWarping

op = CorrelationOptimizedWarping(reference=None, interval_size=32, max_shift=5)
Xt = op.fit_transform(X)
```

:::

:::{tab-item} R · nirs4all-methods
:sync: r
:class-label: lang-r

```r
library(n4m)
res <- correlation_optimized_warping(X, reference = NULL, interval_size = 16L, max_shift = 2L)
```

:::

::::


**Benchmark Comparators And Sources** ◆

:::{card}
:class-card: external-refs

- ◆ **`ref.cowarp`** (Python · canonical) — `cowarp.warp(segment_length=16, slack=2)` · cowarp 0.2.1 — cowarp provides the external dynamic-programming COW contract used by this gate
:::

### Validation contract

- Operation: `fit_transform` · comparator: `default_allclose` · tolerance: `rtol=1e-05`, `atol=1e-08` · quality: **strict**
- Default validation dataset: `32×96` · seed `1234567910`
- Suites: smoke `3` cells; benchmark `11` cells · Default C/Python/reference parity comparator.
- Metrics: `finite_output`, `max_abs_diff`, `shape_equal`
- Truth sources:
  - `tomasi_cow_dtw_2004` — Correlation optimized warping and dynamic time warping as preprocessing methods for chromatographic data (doi:10.1002/cem.859)

| Backend | Library | Gate | Comparator | Note |
|---------|---------|------|------------|------|
| `ref.cowarp` | `cowarp.warp(segment_length=16, slack=2)` | Python / parity | `default_allclose` | cowarp provides the external dynamic-programming COW contract used by this gate |

### Benchmarks
Median wall-clock per cell from [`docs/_static/bench-data.json`](../benchmarks/overview.md). Divergence is the worst finite value over the visible sizes for each backend, preferring reference max-abs difference and falling back to binding max-abs difference when no reference comparison is recorded. Rows without a recorded comparison show `—`; the fastest backend per column is marked 🏆.
::::{tab-set}
:class: parity-tabs

:::{tab-item} 1 thread
:sync: threads-1

<div class="parity-table-wrap">
<table class="docutils parity-grouped">
<thead><tr><th>Backend</th><th>Divergence</th><th>100×50</th><th>100×500</th></tr></thead>
<tbody class="lang-band lang-cpp"><tr class="lang-band-row" data-lang="cpp"><th colspan="4" scope="rowgroup"><span class="lang-band-dot"></span>C++ native · libn4m</th></tr>
<tr class="bk-row"><td class="bk-name"><code>N4M.cpp</code></td><td class="parity parity-divergence parity-exact" title="worst reference max abs diff over visible sizes">1.1e-16</td><td class="ms ms-best">🏆 0.178 ms</td><td class="ms">22.626 ms</td></tr>
</tbody>
<tbody class="lang-band lang-python"><tr class="lang-band-row" data-lang="python"><th colspan="4" scope="rowgroup"><span class="lang-band-dot"></span>Python · nirs4all-methods</th></tr>
<tr class="bk-row"><td class="bk-name"><code>N4M.python</code></td><td class="parity parity-divergence parity-exact" title="worst reference max abs diff over visible sizes">1.1e-16</td><td class="ms">0.179 ms</td><td class="ms ms-best">🏆 22.594 ms</td></tr>
<tr class="bk-row"><td class="bk-name"><code>N4M.sklearn</code></td><td class="parity parity-divergence parity-exact" title="worst reference max abs diff over visible sizes">1.1e-16</td><td class="ms">0.191 ms</td><td class="ms">23.003 ms</td></tr>
</tbody>
<tbody class="lang-band lang-r"><tr class="lang-band-row" data-lang="r"><th colspan="4" scope="rowgroup"><span class="lang-band-dot"></span>R · nirs4all-methods</th></tr>
<tr class="bk-row"><td class="bk-name"><code>N4M.R</code></td><td class="parity parity-divergence parity-exact" title="worst reference max abs diff over visible sizes">5.6e-16</td><td class="ms">0.201 ms</td><td class="ms">29.500 ms</td></tr>
</tbody>
<tbody class="lang-band lang-python"><tr class="lang-band-row" data-lang="python"><th colspan="4" scope="rowgroup"><span class="lang-band-dot"></span>Python · external</th></tr>
<tr class="bk-row truth-source-strict"><td class="bk-name"><span class="truth-mark" title="Registry parity reference (Python): cowarp.warp(segment_length=16, slack=2) · cowarp 0.2.1 — canonical">◆</span><code>ref.cowarp</code></td><td class="parity parity-divergence parity-exact" title="worst reference max abs diff over visible sizes">0</td><td class="ms">34.693 ms</td><td class="ms">2844.264 ms</td></tr>
</tbody>
</table>
</div>

:::
::::

---

_See also_: [benchmark overview](../benchmarks/overview.md) · [methods index](index.md) · [interactive dashboard](../landing/dashboard.md)
