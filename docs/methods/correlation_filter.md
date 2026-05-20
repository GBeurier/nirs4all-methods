# `correlation_filter` — Correlation filter

_Group_: **Feature Selection** · _Registry tolerance_: `rtol=1e-5`, `atol=1e-8`

## Description

`correlation_filter` is a chemometrics4all feature selection method exposed through the C ABI and the generated language bindings.

### Parameters

No public constructor parameters are required for the documented default call.

## Explanations

### Bibliographic source

- `ref.sklearn` — sklearn.feature_selection.r_regression (Python).

### Mathematical principle

`correlation_filter` follows the standard feature selection operator contract: an input matrix $\mathbf{X}\in\mathbb{R}^{n\times p}$ is transformed, scored, split, or filtered by the method-specific kernel while preserving the parity contract recorded by the benchmark snapshots. The precise numerical convention is the one exposed by the C ABI signatures and the registered external references below.

### Implementation

C ABI entry points used by the language bindings:

```c
c4a_status_t c4a_filter_correlation_create( c4a_filter_correlation_handle_t** out, double threshold, int32_t top_k);
void c4a_filter_correlation_destroy( c4a_filter_correlation_handle_t* handle);
c4a_status_t c4a_filter_correlation_fit( c4a_filter_correlation_handle_t* handle, c4a_matrix_view_t X, const double* y, int64_t n_y);
c4a_status_t c4a_filter_correlation_is_fitted( const c4a_filter_correlation_handle_t* handle, int* out_fitted);
c4a_status_t c4a_filter_correlation_output_cols( const c4a_filter_correlation_handle_t* handle, int64_t* out_cols);
c4a_status_t c4a_filter_correlation_transform( const c4a_filter_correlation_handle_t* handle, c4a_matrix_view_t X, c4a_matrix_view_t out);
```

Benchmark comparator backends are registered in the matrix and stored as reproducible snapshots when they define the canonical contract.

### Implementations

| Layer | Entry point | Language | Contract |
|-------|-------------|----------|----------|
| C ABI | `c4a_filter_correlation` | C/C++ | Stable libc4a entry point family. |
| Python | `chemometrics4all.python.correlation_filter` | Python | ABI-close function backed by ctypes. |
| R | `correlation_filter(X, y, threshold = 0.0, top_k = 0L)` | R | Public package wrapper around the C ABI. |
| ref.sklearn | `sklearn.feature_selection.r_regression` | Python | canonical/comparator |

### Usage

Every chemometrics4all binding dispatches into the same C kernel. Registered comparator/source rows are listed in the benchmark card below.

::::{tab-set}
:class: chemometrics4all-bindings


:::{tab-item} C ABI · libc4a
:sync: c
:class-label: lang-c

```c
c4a_status_t c4a_filter_correlation_create( c4a_filter_correlation_handle_t** out, double threshold, int32_t top_k);
void c4a_filter_correlation_destroy( c4a_filter_correlation_handle_t* handle);
c4a_status_t c4a_filter_correlation_fit( c4a_filter_correlation_handle_t* handle, c4a_matrix_view_t X, const double* y, int64_t n_y);
c4a_status_t c4a_filter_correlation_is_fitted( const c4a_filter_correlation_handle_t* handle, int* out_fitted);
c4a_status_t c4a_filter_correlation_output_cols( const c4a_filter_correlation_handle_t* handle, int64_t* out_cols);
c4a_status_t c4a_filter_correlation_transform( const c4a_filter_correlation_handle_t* handle, c4a_matrix_view_t X, c4a_matrix_view_t out);
```

:::

:::{tab-item} Python ABI · chemometrics4all.python
:sync: python-abi
:class-label: lang-python

```python
from chemometrics4all import python as c4a

Xt = c4a.correlation_filter(X, y, top_k=5)
```

:::

:::{tab-item} R · chemometrics4all
:sync: r
:class-label: lang-r

```r
library(chemometrics4all)
res <- correlation_filter(X, y, threshold = 0.0, top_k = min(5L, ncol(X)))
```

:::

::::


**Benchmark Comparators And Sources** ◆

:::{card}
:class-card: external-refs

- ◆ **`ref.sklearn`** (Python · canonical) — `sklearn.feature_selection.r_regression` · scikit-learn 1.8.0
:::

### Validation contract

- Operation: `fit_transform` · comparator: `default_allclose` · tolerance: `rtol=1e-05`, `atol=1e-08` · quality: **strict**
- Default validation dataset: `64×128` · seed `1234567914`
- Suites: smoke `3` cells; benchmark `11` cells · Default C/Python/reference parity comparator.
- Metrics: `finite_output`, `shape_equal`, `support_equal`
- Truth sources:
  - `nir_variable_selection_review_2010` — Variables selection methods in near-infrared spectroscopy (doi:10.1016/j.aca.2010.03.048)

| Backend | Library | Gate | Comparator | Note |
|---------|---------|------|------------|------|
| `ref.sklearn` | `sklearn.feature_selection.r_regression` | Python / parity | `default_allclose` |  |

### Benchmarks
Median wall-clock per cell from [`docs/_static/bench-data.json`](../benchmarks/overview.md). Divergence is the worst finite value over the visible sizes for each backend, preferring reference max-abs difference and falling back to binding max-abs difference when no reference comparison is recorded. Rows without a recorded comparison show `—`; the fastest backend per column is marked 🏆.
::::{tab-set}
:class: parity-tabs

:::{tab-item} 1 thread
:sync: threads-1

<div class="parity-table-wrap">
<table class="docutils parity-grouped">
<thead><tr><th>Backend</th><th>Divergence</th><th>100×50</th><th>100×500</th><th>100×2500</th></tr></thead>
<tbody class="lang-band lang-cpp"><tr class="lang-band-row" data-lang="cpp"><th colspan="5" scope="rowgroup"><span class="lang-band-dot"></span>C++ native · libc4a</th></tr>
<tr class="bk-row"><td class="bk-name"><code>C4A.cpp</code></td><td class="parity parity-divergence parity-exact" title="worst reference max abs diff over visible sizes">0</td><td class="ms">0.015 ms</td><td class="ms ms-best">🏆 0.064 ms</td><td class="ms">0.366 ms</td></tr>
</tbody>
<tbody class="lang-band lang-python"><tr class="lang-band-row" data-lang="python"><th colspan="5" scope="rowgroup"><span class="lang-band-dot"></span>Python · chemometrics4all</th></tr>
<tr class="bk-row"><td class="bk-name"><code>C4A.python</code></td><td class="parity parity-divergence parity-exact" title="worst reference max abs diff over visible sizes">0</td><td class="ms ms-best">🏆 0.015 ms</td><td class="ms">0.065 ms</td><td class="ms ms-best">🏆 0.312 ms</td></tr>
<tr class="bk-row"><td class="bk-name"><code>C4A.sklearn</code></td><td class="parity parity-divergence parity-exact" title="worst reference max abs diff over visible sizes">0</td><td class="ms">0.021 ms</td><td class="ms">0.068 ms</td><td class="ms">0.333 ms</td></tr>
</tbody>
<tbody class="lang-band lang-r"><tr class="lang-band-row" data-lang="r"><th colspan="5" scope="rowgroup"><span class="lang-band-dot"></span>R · chemometrics4all</th></tr>
<tr class="bk-row"><td class="bk-name"><code>C4A.R</code></td><td class="parity parity-divergence parity-exact" title="worst reference max abs diff over visible sizes">5.6e-16</td><td class="ms">0.042 ms</td><td class="ms">0.239 ms</td><td class="ms">1.398 ms</td></tr>
</tbody>
<tbody class="lang-band lang-python"><tr class="lang-band-row" data-lang="python"><th colspan="5" scope="rowgroup"><span class="lang-band-dot"></span>Python · external</th></tr>
<tr class="bk-row truth-source-strict"><td class="bk-name"><span class="truth-mark" title="Registry parity reference (Python): sklearn.feature_selection.r_regression · scikit-learn 1.8.0 — canonical">◆</span><code>ref.sklearn</code></td><td class="parity parity-divergence parity-exact" title="worst reference max abs diff over visible sizes">0</td><td class="ms">0.132 ms</td><td class="ms">0.240 ms</td><td class="ms">0.818 ms</td></tr>
</tbody>
</table>
</div>

:::
::::

---

_See also_: [benchmark overview](../benchmarks/overview.md) · [methods index](index.md) · [interactive dashboard](../landing/dashboard.md)
