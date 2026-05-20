# `piecewise_direct_standardization` — Piecewise direct standardization

_Group_: **Calibration Transfer** · _Registry tolerance_: `rtol=1e-5`, `atol=1e-8`

## Description

From the `n4m.PiecewiseDirectStandardization` Python wrapper docstring:

> PDS: local regressions mapping source windows to target wavelengths.

### Parameters

| Name | Type | Default | Notes |
|------|------|---------|-------|
| `window_size` | `int` | `5` |  |
| `fit_intercept` | `bool` | `True` |  |
| `ridge` | `float` | `0.0` |  |

## Explanations

### Bibliographic source

- `ref.sklearn` — sklearn.linear_model.LinearRegression(local windows) (Python).
- `ref.pycaltransfer` — pycaltransfer.pds_pls_transfer_fit (Python).

### Mathematical principle

`piecewise_direct_standardization` follows the standard calibration transfer operator contract: an input matrix $\mathbf{X}\in\mathbb{R}^{n\times p}$ is transformed, scored, split, or filtered by the method-specific kernel while preserving the parity contract recorded by the benchmark snapshots. The precise numerical convention is the one exposed by the C ABI signatures and the registered external references below.

### Implementation

C ABI entry points used by the language bindings:

```c
n4m_status_t n4m_pp_piecewise_direct_standardization_create( n4m_pp_piecewise_direct_standardization_handle_t** out, int32_t window_size, int fit_intercept, double ridge);
void n4m_pp_piecewise_direct_standardization_destroy( n4m_pp_piecewise_direct_standardization_handle_t* handle);
n4m_status_t n4m_pp_piecewise_direct_standardization_fit( n4m_pp_piecewise_direct_standardization_handle_t* handle, n4m_matrix_view_t source, n4m_matrix_view_t target);
n4m_status_t n4m_pp_piecewise_direct_standardization_is_fitted( const n4m_pp_piecewise_direct_standardization_handle_t* handle, int* out_fitted);
n4m_status_t n4m_pp_piecewise_direct_standardization_transform( const n4m_pp_piecewise_direct_standardization_handle_t* handle, n4m_matrix_view_t X, n4m_matrix_view_t out);
```

Benchmark comparator backends are registered in the matrix and stored as reproducible snapshots when they define the canonical contract.

### Implementations

| Layer | Entry point | Language | Contract |
|-------|-------------|----------|----------|
| C ABI | `n4m_pp_piecewise_direct_standardization` | C/C++ | Stable libn4m entry point family. |
| Python | `n4m.python.piecewise_direct_standardization` | Python | ABI-close function backed by ctypes. |
| Python sklearn | `n4m.sklearn.PiecewiseDirectStandardization` | Python | scikit-learn-compatible estimator backed by ctypes. |
| R | `{target <- 1.05 * X + 0.1; piecewise_direct_standardization(X, target, X, window_size = 5L, fit_intercept = TRUE, ridge = 0.0)}` | R | Public package wrapper around the C ABI. |
| ref.sklearn | `sklearn.linear_model.LinearRegression(local windows)` | Python | canonical/comparator; sklearn supplies each local least-squares transfer model for the N4M PDS contract |
| ref.pycaltransfer | `pycaltransfer.pds_pls_transfer_fit` | Python | context only; pycaltransfer exposes the PDS-PLS variant; N4M's current PiecewiseDirectStandardization contract is local OLS and is gated by sklearn above |

### Usage

Every nirs4all-methods binding dispatches into the same C kernel. Registered comparator/source rows are listed in the benchmark card below.

::::{tab-set}
:class: nirs4all-methods-bindings


:::{tab-item} C ABI · libn4m
:sync: c
:class-label: lang-c

```c
n4m_status_t n4m_pp_piecewise_direct_standardization_create( n4m_pp_piecewise_direct_standardization_handle_t** out, int32_t window_size, int fit_intercept, double ridge);
void n4m_pp_piecewise_direct_standardization_destroy( n4m_pp_piecewise_direct_standardization_handle_t* handle);
n4m_status_t n4m_pp_piecewise_direct_standardization_fit( n4m_pp_piecewise_direct_standardization_handle_t* handle, n4m_matrix_view_t source, n4m_matrix_view_t target);
n4m_status_t n4m_pp_piecewise_direct_standardization_is_fitted( const n4m_pp_piecewise_direct_standardization_handle_t* handle, int* out_fitted);
n4m_status_t n4m_pp_piecewise_direct_standardization_transform( const n4m_pp_piecewise_direct_standardization_handle_t* handle, n4m_matrix_view_t X, n4m_matrix_view_t out);
```

:::

:::{tab-item} Python ABI · n4m.python
:sync: python-abi
:class-label: lang-python

```python
from n4m import python as n4m

Xt = n4m.piecewise_direct_standardization(X_source, X_target, X)
```

:::

:::{tab-item} Python sklearn · n4m.sklearn
:sync: python-sklearn
:class-label: lang-python

```python
from n4m.sklearn import PiecewiseDirectStandardization

op = PiecewiseDirectStandardization(window_size=5, fit_intercept=True, ridge=0.0)
Xt = op.fit_transform(X)
```

:::

:::{tab-item} R · nirs4all-methods
:sync: r
:class-label: lang-r

```r
library(n4m)
res <- {target <- 1.05 * X + 0.1; piecewise_direct_standardization(X, target, X, window_size = 5L, fit_intercept = TRUE, ridge = 0.0)}
```

:::

::::


**Benchmark Comparators And Sources** ◆

:::{card}
:class-card: external-refs

- ◆ **`ref.sklearn`** (Python · canonical) — `sklearn.linear_model.LinearRegression(local windows)` · scikit-learn 1.8.0 — sklearn supplies each local least-squares transfer model for the N4M PDS contract
- ℹ **`ref.pycaltransfer`** (Python · context) — `pycaltransfer.pds_pls_transfer_fit` · pycaltransfer 0.1.8 — pycaltransfer exposes the PDS-PLS variant; N4M's current PiecewiseDirectStandardization contract is local OLS and is gated by sklearn above
:::

### Validation contract

- Operation: `fit_transform_pair` · comparator: `default_allclose` · tolerance: `rtol=1e-05`, `atol=1e-08` · quality: **strict**
- Default validation dataset: `48×96` · seed `1234567899`
- Suites: smoke `3` cells; benchmark `11` cells · Default C/Python/reference parity comparator.
- Metrics: `finite_output`, `max_abs_diff`, `shape_equal`
- Truth sources:
  - `workman_2018_calibration_transfer` — A Review of Calibration Transfer Practices and Instrument Differences in Spectroscopy (doi:10.1177/0003702817736064)

| Backend | Library | Gate | Comparator | Note |
|---------|---------|------|------------|------|
| `ref.sklearn` | `sklearn.linear_model.LinearRegression(local windows)` | Python / parity | `default_allclose` | sklearn supplies each local least-squares transfer model for the N4M PDS contract |
| `ref.pycaltransfer` | `pycaltransfer.pds_pls_transfer_fit` | Python / context | `default_allclose` | pycaltransfer exposes the PDS-PLS variant; N4M's current PiecewiseDirectStandardization contract is local OLS and is gated by sklearn above |

### Benchmarks
Median wall-clock per cell from [`docs/_static/bench-data.json`](../benchmarks/overview.md). Divergence is the worst finite value over the visible sizes for each backend, preferring reference max-abs difference and falling back to binding max-abs difference when no reference comparison is recorded. Rows without a recorded comparison show `—`; the fastest backend per column is marked 🏆.
::::{tab-set}
:class: parity-tabs

:::{tab-item} 1 thread
:sync: threads-1

<div class="parity-table-wrap">
<table class="docutils parity-grouped">
<thead><tr><th>Backend</th><th>Divergence</th><th>100×50</th></tr></thead>
<tbody class="lang-band lang-cpp"><tr class="lang-band-row" data-lang="cpp"><th colspan="3" scope="rowgroup"><span class="lang-band-dot"></span>C++ native · libn4m</th></tr>
<tr class="bk-row"><td class="bk-name"><code>N4M.cpp</code></td><td class="parity parity-divergence parity-exact" title="worst reference max abs diff over visible sizes">1.8e-13</td><td class="ms">0.133 ms</td></tr>
</tbody>
<tbody class="lang-band lang-python"><tr class="lang-band-row" data-lang="python"><th colspan="3" scope="rowgroup"><span class="lang-band-dot"></span>Python · nirs4all-methods</th></tr>
<tr class="bk-row"><td class="bk-name"><code>N4M.python</code></td><td class="parity parity-divergence parity-exact" title="worst reference max abs diff over visible sizes">1.8e-13</td><td class="ms ms-best">🏆 0.133 ms</td></tr>
<tr class="bk-row"><td class="bk-name"><code>N4M.sklearn</code></td><td class="parity parity-divergence parity-exact" title="worst reference max abs diff over visible sizes">1.8e-13</td><td class="ms">0.136 ms</td></tr>
</tbody>
<tbody class="lang-band lang-r"><tr class="lang-band-row" data-lang="r"><th colspan="3" scope="rowgroup"><span class="lang-band-dot"></span>R · nirs4all-methods</th></tr>
<tr class="bk-row"><td class="bk-name"><code>N4M.R</code></td><td class="parity parity-divergence parity-exact" title="worst reference max abs diff over visible sizes">1.9e-13</td><td class="ms">0.212 ms</td></tr>
</tbody>
<tbody class="lang-band lang-python"><tr class="lang-band-row" data-lang="python"><th colspan="3" scope="rowgroup"><span class="lang-band-dot"></span>Python · external</th></tr>
<tr class="bk-row truth-source-relaxed"><td class="bk-name"><span class="truth-mark" title="Registry parity reference (Python): pycaltransfer.pds_pls_transfer_fit · pycaltransfer 0.1.8 — context">◆</span><code>ref.pycaltransfer</code></td><td class="parity parity-divergence parity-context" title="no divergence recorded">—</td><td class="ms">20.281 ms</td></tr>
<tr class="bk-row truth-source-strict"><td class="bk-name"><span class="truth-mark" title="Registry parity reference (Python): sklearn.linear_model.LinearRegression(local windows) · scikit-learn 1.8.0 — canonical">◆</span><code>ref.sklearn</code></td><td class="parity parity-divergence parity-exact" title="worst reference max abs diff over visible sizes">0</td><td class="ms">12.633 ms</td></tr>
</tbody>
</table>
</div>

:::
::::

---

_See also_: [benchmark overview](../benchmarks/overview.md) · [methods index](index.md) · [interactive dashboard](../landing/dashboard.md)
