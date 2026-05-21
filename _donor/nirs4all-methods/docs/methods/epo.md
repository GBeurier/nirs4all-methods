# `epo` — EPO

_Group_: **Feature extraction** · _Registry tolerance_: `rtol=1e-5`, `atol=1e-8` · _Source_: [`docs/algorithms/epo.md`](../algorithms/epo.md)

## Description

From the `n4m.EPO` Python wrapper docstring:

> External Parameter Orthogonalisation.

### Parameters

| Name | Type | Default | Notes |
|------|------|---------|-------|
| `scale` | `bool` | `True` | Whether to standardize internal variables before fitting. |

## Explanations

### Bibliographic source

1. Roger, J. M., Chauchard, F., Bellon-Maurel, V. (2003). EPO–PLS:
   external parameter orthogonalisation of PLS. Application to
   temperature-independent measurement of sugar content of intact
   fruits. *Chemometrics and Intelligent Laboratory Systems*, 66(2),
   191-204.

### Mathematical principle

For a univariate external parameter `d` (length `n_samples`):

1. Compute calibration means (when `scale != 0`):
   `X_mean = mean(X, axis=0)`,  `d_mean = mean(d)`.
   When `scale == 0` both means are taken as zero.
2. Solve the per-feature scalar regression
   `B[j] = (d − d_mean)^T (X[:, j] − X_mean[j]) / (d − d_mean)^T (d − d_mean)`,
   for `j = 0..cols-1`.

At fit time the n4m engine stores `X_mean`, `d_mean`, and `B`. The
training-time output `X - outer(d - d_mean, B)` is what nirs4all's
`fit_transform(X, d)` returns.

### Implementation

C ABI entry points used by the language bindings:

```c
n4m_status_t n4m_pp_epo_create(n4m_pp_epo_handle_t** out, int scale);
void n4m_pp_epo_destroy(n4m_pp_epo_handle_t* handle);
n4m_status_t n4m_pp_epo_fit(n4m_pp_epo_handle_t* handle, n4m_matrix_view_t X, const double* d, int64_t d_len);
n4m_status_t n4m_pp_epo_inverse_transform( const n4m_pp_epo_handle_t* handle, n4m_matrix_view_t X, n4m_matrix_view_t out);
n4m_status_t n4m_pp_epo_is_fitted(const n4m_pp_epo_handle_t* handle, int* out_fitted);
n4m_status_t n4m_pp_epo_transform(const n4m_pp_epo_handle_t* handle, n4m_matrix_view_t X, n4m_matrix_view_t out);
n4m_status_t n4m_pp_epo_transform_with_d( const n4m_pp_epo_handle_t* handle, n4m_matrix_view_t X, const double* d, int64_t d_len, n4m_matrix_view_t out);
```

Benchmark comparator backends are registered in the matrix and stored as reproducible snapshots when they define the canonical contract.

### Implementations

| Layer | Entry point | Language | Contract |
|-------|-------------|----------|----------|
| C ABI | `n4m_pp_epo` | C/C++ | Stable libn4m entry point family. |
| Python | `n4m.python.epo` | Python | ABI-close function backed by ctypes. |
| Python sklearn | `n4m.sklearn.EPO` | Python | scikit-learn-compatible estimator backed by ctypes. |
| R | `epo(X, d, scale = TRUE)` | R | Public package wrapper around the C ABI. |
| ref.nirs4all | `nirs4all.EPO` | Python | canonical/comparator |

### Usage

Every nirs4all-methods binding dispatches into the same C kernel. Registered comparator/source rows are listed in the benchmark card below.

::::{tab-set}
:class: nirs4all-methods-bindings


:::{tab-item} C ABI · libn4m
:sync: c
:class-label: lang-c

```c
n4m_status_t n4m_pp_epo_create(n4m_pp_epo_handle_t** out, int scale);
void n4m_pp_epo_destroy(n4m_pp_epo_handle_t* handle);
n4m_status_t n4m_pp_epo_fit(n4m_pp_epo_handle_t* handle, n4m_matrix_view_t X, const double* d, int64_t d_len);
n4m_status_t n4m_pp_epo_inverse_transform( const n4m_pp_epo_handle_t* handle, n4m_matrix_view_t X, n4m_matrix_view_t out);
n4m_status_t n4m_pp_epo_is_fitted(const n4m_pp_epo_handle_t* handle, int* out_fitted);
n4m_status_t n4m_pp_epo_transform(const n4m_pp_epo_handle_t* handle, n4m_matrix_view_t X, n4m_matrix_view_t out);
n4m_status_t n4m_pp_epo_transform_with_d( const n4m_pp_epo_handle_t* handle, n4m_matrix_view_t X, const double* d, int64_t d_len, n4m_matrix_view_t out);
```

:::

:::{tab-item} Python ABI · n4m.python
:sync: python-abi
:class-label: lang-python

```python
from n4m import python as n4m

Xt = n4m.epo(X, d)
```

:::

:::{tab-item} Python sklearn · n4m.sklearn
:sync: python-sklearn
:class-label: lang-python

```python
from n4m.sklearn import EPO

op = EPO(scale=True)
Xt = op.fit_transform(X, d)
```

:::

:::{tab-item} R · nirs4all-methods
:sync: r
:class-label: lang-r

```r
library(n4m)
res <- epo(X, d, scale = TRUE)
```

:::

::::


**Benchmark Comparators And Sources** ◆

:::{card}
:class-card: external-refs

- ◆ **`ref.nirs4all`** (Python · canonical) — `nirs4all.EPO` · nirs4all@cd731a23+dirty
:::

### Validation contract

- Operation: `cross_binding_callable` · comparator: `default_allclose` · tolerance: `rtol=1e-05`, `atol=1e-08` · quality: **strict**
- Default validation dataset: `100×50` · seed `20260556`
- Suites: smoke `3` cells; benchmark `11` cells · Default C/Python/reference parity comparator.
- Metrics: `max_abs_diff`, `rel_l2_diff`, `rms_diff`, `shape_equal`
- Truth sources: cross-binding references declared directly in `benchmarks/cross_binding/orchestrator.py`.

| Backend | Library | Gate | Comparator | Note |
|---------|---------|------|------------|------|
| `ref.nirs4all` | `nirs4all.EPO` | Python / parity | `default_allclose` |  |

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
<tr class="bk-row"><td class="bk-name"><code>N4M.cpp</code></td><td class="parity parity-divergence parity-exact" title="worst reference max abs diff over visible sizes">1.1e-16</td><td class="ms ms-best">🏆 0.015 ms</td><td class="ms">0.058 ms</td><td class="ms ms-best">🏆 0.271 ms</td></tr>
</tbody>
<tbody class="lang-band lang-python"><tr class="lang-band-row" data-lang="python"><th colspan="5" scope="rowgroup"><span class="lang-band-dot"></span>Python · nirs4all-methods</th></tr>
<tr class="bk-row"><td class="bk-name"><code>N4M.python</code></td><td class="parity parity-divergence parity-exact" title="worst reference max abs diff over visible sizes">1.1e-16</td><td class="ms">0.016 ms</td><td class="ms ms-best">🏆 0.057 ms</td><td class="ms">0.405 ms</td></tr>
<tr class="bk-row"><td class="bk-name"><code>N4M.sklearn</code></td><td class="parity parity-divergence parity-exact" title="worst reference max abs diff over visible sizes">1.1e-16</td><td class="ms">0.018 ms</td><td class="ms">0.063 ms</td><td class="ms">0.271 ms</td></tr>
</tbody>
<tbody class="lang-band lang-r"><tr class="lang-band-row" data-lang="r"><th colspan="5" scope="rowgroup"><span class="lang-band-dot"></span>R · nirs4all-methods</th></tr>
<tr class="bk-row"><td class="bk-name"><code>N4M.R</code></td><td class="parity parity-divergence parity-exact" title="worst reference max abs diff over visible sizes">6.7e-16</td><td class="ms">0.031 ms</td><td class="ms">0.318 ms</td><td class="ms">2.266 ms</td></tr>
</tbody>
<tbody class="lang-band lang-python"><tr class="lang-band-row" data-lang="python"><th colspan="5" scope="rowgroup"><span class="lang-band-dot"></span>Python · external</th></tr>
<tr class="bk-row truth-source-strict"><td class="bk-name"><span class="truth-mark" title="Registry parity reference (Python): nirs4all.EPO · nirs4all@cd731a23+dirty — canonical">◆</span><code>ref.nirs4all</code></td><td class="parity parity-divergence parity-exact" title="worst reference max abs diff over visible sizes">0</td><td class="ms">0.061 ms</td><td class="ms">0.362 ms</td><td class="ms">2.249 ms</td></tr>
</tbody>
</table>
</div>

:::
::::

---

_See also_: [benchmark overview](../benchmarks/overview.md) · [methods index](index.md) · [interactive dashboard](../landing/dashboard.md)
