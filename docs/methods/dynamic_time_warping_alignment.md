# `dynamic_time_warping_alignment` — Dynamic time warping alignment

_Group_: **Alignment** · _Registry tolerance_: `rtol=1e-5`, `atol=1e-8`

## Description

From the `chemometrics4all.DynamicTimeWarpingAlignment` Python wrapper docstring:

> Dynamic-time-warping alignment to a fixed-length reference.

### Parameters

| Name | Type | Default | Notes |
|------|------|---------|-------|
| `reference` | `object` | `None` |  |

## Explanations

### Bibliographic source

- `ref.dtw_python` — dtw-python.dtw(step_pattern=symmetric1) (Python).

### Mathematical principle

`dynamic_time_warping_alignment` follows the standard alignment operator contract: an input matrix $\mathbf{X}\in\mathbb{R}^{n\times p}$ is transformed, scored, split, or filtered by the method-specific kernel while preserving the parity contract recorded by the benchmark snapshots. The precise numerical convention is the one exposed by the C ABI signatures and the registered external references below.

### Implementation

C ABI entry points used by the language bindings:

```c
c4a_status_t c4a_pp_dtw_align_create( c4a_pp_dtw_align_handle_t** out, const double* reference, int64_t n_reference, int32_t interval_size, int32_t max_shift);
void c4a_pp_dtw_align_destroy(c4a_pp_dtw_align_handle_t* handle);
c4a_status_t c4a_pp_dtw_align_fit( c4a_pp_dtw_align_handle_t* handle, c4a_matrix_view_t X);
c4a_status_t c4a_pp_dtw_align_is_fitted( const c4a_pp_dtw_align_handle_t* handle, int* out_fitted);
c4a_status_t c4a_pp_dtw_align_transform( const c4a_pp_dtw_align_handle_t* handle, c4a_matrix_view_t X, c4a_matrix_view_t out);
```

Benchmark comparator backends are registered in the matrix and stored as reproducible snapshots when they define the canonical contract.

### Implementations

| Layer | Entry point | Language | Contract |
|-------|-------------|----------|----------|
| C ABI | `c4a_pp_dtw_align` | C/C++ | Stable libc4a entry point family. |
| Python | `chemometrics4all.python.dynamic_time_warping_alignment` | Python | ABI-close function backed by ctypes. |
| Python sklearn | `chemometrics4all.sklearn.DynamicTimeWarpingAlignment` | Python | scikit-learn-compatible estimator backed by ctypes. |
| R | `dynamic_time_warping_alignment(X, reference = NULL, interval_size = 16L, max_shift = 3L)` | R | Public package wrapper around the C ABI. |
| ref.dtw_python | `dtw-python.dtw(step_pattern=symmetric1)` | Python | canonical/comparator; dtw-python provides the DTW path; the parity adapter maps that external path to c4a's fixed-length output by averaging query samples assigned to each reference index |

### Usage

Every chemometrics4all binding dispatches into the same C kernel. Registered comparator/source rows are listed in the benchmark card below.

::::{tab-set}
:class: chemometrics4all-bindings


:::{tab-item} C ABI · libc4a
:sync: c
:class-label: lang-c

```c
c4a_status_t c4a_pp_dtw_align_create( c4a_pp_dtw_align_handle_t** out, const double* reference, int64_t n_reference, int32_t interval_size, int32_t max_shift);
void c4a_pp_dtw_align_destroy(c4a_pp_dtw_align_handle_t* handle);
c4a_status_t c4a_pp_dtw_align_fit( c4a_pp_dtw_align_handle_t* handle, c4a_matrix_view_t X);
c4a_status_t c4a_pp_dtw_align_is_fitted( const c4a_pp_dtw_align_handle_t* handle, int* out_fitted);
c4a_status_t c4a_pp_dtw_align_transform( const c4a_pp_dtw_align_handle_t* handle, c4a_matrix_view_t X, c4a_matrix_view_t out);
```

:::

:::{tab-item} Python ABI · chemometrics4all.python
:sync: python-abi
:class-label: lang-python

```python
from chemometrics4all import python as c4a

Xt = c4a.dynamic_time_warping_alignment(X)
```

:::

:::{tab-item} Python sklearn · chemometrics4all.sklearn
:sync: python-sklearn
:class-label: lang-python

```python
from chemometrics4all.sklearn import DynamicTimeWarpingAlignment

op = DynamicTimeWarpingAlignment(reference=None)
Xt = op.fit_transform(X)
```

:::

:::{tab-item} R · chemometrics4all
:sync: r
:class-label: lang-r

```r
library(chemometrics4all)
res <- dynamic_time_warping_alignment(X, reference = NULL, interval_size = 0L, max_shift = 0L)
```

:::

::::


**Benchmark Comparators And Sources** ◆

:::{card}
:class-card: external-refs

- ◆ **`ref.dtw_python`** (Python · canonical) — `dtw-python.dtw(step_pattern=symmetric1)` · dtw-python 1.7.4 — dtw-python provides the DTW path; the parity adapter maps that external path to c4a's fixed-length output by averaging query samples assigned to each reference index
:::

### Validation contract

- Operation: `fit_transform` · comparator: `default_allclose` · tolerance: `rtol=1e-05`, `atol=1e-08` · quality: **strict**
- Default validation dataset: `32×96` · seed `1234567911`
- Suites: smoke `3` cells; benchmark `11` cells · Default C/Python/reference parity comparator.
- Metrics: `finite_output`, `max_abs_diff`, `shape_equal`
- Truth sources:
  - `tomasi_cow_dtw_2004` — Correlation optimized warping and dynamic time warping as preprocessing methods for chromatographic data (doi:10.1002/cem.859)

| Backend | Library | Gate | Comparator | Note |
|---------|---------|------|------------|------|
| `ref.dtw_python` | `dtw-python.dtw(step_pattern=symmetric1)` | Python / parity | `default_allclose` | dtw-python provides the DTW path; the parity adapter maps that external path to c4a's fixed-length output by averaging query samples assigned to each reference index |

### Benchmarks
Median wall-clock per cell from [`docs/_static/bench-data.json`](../benchmarks/overview.md). Divergence is the worst finite value over the visible sizes for each backend, preferring reference max-abs difference and falling back to binding max-abs difference when no reference comparison is recorded. Rows without a recorded comparison show `—`; the fastest backend per column is marked 🏆.
::::{tab-set}
:class: parity-tabs

:::{tab-item} 1 thread
:sync: threads-1

<div class="parity-table-wrap">
<table class="docutils parity-grouped">
<thead><tr><th>Backend</th><th>Divergence</th><th>100×50</th></tr></thead>
<tbody class="lang-band lang-cpp"><tr class="lang-band-row" data-lang="cpp"><th colspan="3" scope="rowgroup"><span class="lang-band-dot"></span>C++ native · libc4a</th></tr>
<tr class="bk-row"><td class="bk-name"><code>C4A.cpp</code></td><td class="parity parity-divergence parity-exact" title="worst reference max abs diff over visible sizes">2.2e-16</td><td class="ms ms-best">🏆 0.381 ms</td></tr>
</tbody>
<tbody class="lang-band lang-python"><tr class="lang-band-row" data-lang="python"><th colspan="3" scope="rowgroup"><span class="lang-band-dot"></span>Python · chemometrics4all</th></tr>
<tr class="bk-row"><td class="bk-name"><code>C4A.python</code></td><td class="parity parity-divergence parity-exact" title="worst reference max abs diff over visible sizes">2.2e-16</td><td class="ms">0.383 ms</td></tr>
<tr class="bk-row"><td class="bk-name"><code>C4A.sklearn</code></td><td class="parity parity-divergence parity-exact" title="worst reference max abs diff over visible sizes">2.2e-16</td><td class="ms">0.406 ms</td></tr>
</tbody>
<tbody class="lang-band lang-r"><tr class="lang-band-row" data-lang="r"><th colspan="3" scope="rowgroup"><span class="lang-band-dot"></span>R · chemometrics4all</th></tr>
<tr class="bk-row"><td class="bk-name"><code>C4A.R</code></td><td class="parity parity-divergence parity-exact" title="worst reference max abs diff over visible sizes">5.6e-16</td><td class="ms">1.234 ms</td></tr>
</tbody>
<tbody class="lang-band lang-python"><tr class="lang-band-row" data-lang="python"><th colspan="3" scope="rowgroup"><span class="lang-band-dot"></span>Python · external</th></tr>
<tr class="bk-row truth-source-strict"><td class="bk-name"><span class="truth-mark" title="Registry parity reference (Python): dtw-python.dtw(step_pattern=symmetric1) · dtw-python 1.7.4 — canonical">◆</span><code>ref.dtw_python</code></td><td class="parity parity-divergence parity-exact" title="worst reference max abs diff over visible sizes">0</td><td class="ms">14.412 ms</td></tr>
</tbody>
</table>
</div>

:::
::::

---

_See also_: [benchmark overview](../benchmarks/overview.md) · [methods index](index.md) · [interactive dashboard](../landing/dashboard.md)
