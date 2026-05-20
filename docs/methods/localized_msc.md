# `localized_msc` — Localized MSC

_Group_: **Preprocessing** · _Registry tolerance_: `rtol=1e-5`, `atol=1e-8`

## Description

From the `n4m.LocalizedMSC` Python wrapper docstring:

> Feature-wise MSC using a moving local wavelength window.

### Parameters

No public constructor parameters are required for the documented default call.

## Explanations

### Bibliographic source

- `ref.sklearn` — sklearn.linear_model.LinearRegression(sliding local MSC) (Python).
- `ref.r.prospectr` — prospectr.msc sliding interval (R).
- `ref.scipy` — scipy.ndimage.convolve1d(local MSC window sums) (Python).

### Mathematical principle

`localized_msc` follows the standard preprocessing operator contract: an input matrix $\mathbf{X}\in\mathbb{R}^{n\times p}$ is transformed, scored, split, or filtered by the method-specific kernel while preserving the parity contract recorded by the benchmark snapshots. The precise numerical convention is the one exposed by the C ABI signatures and the registered external references below.

### Implementation

C ABI entry points used by the language bindings:

```c
n4m_status_t n4m_pp_localized_msc_create( n4m_pp_localized_msc_handle_t** out, const double* reference, int64_t n_reference, int32_t window, double eps);
void n4m_pp_localized_msc_destroy( n4m_pp_localized_msc_handle_t* handle);
n4m_status_t n4m_pp_localized_msc_fit( n4m_pp_localized_msc_handle_t* handle, n4m_matrix_view_t X);
n4m_status_t n4m_pp_localized_msc_is_fitted( const n4m_pp_localized_msc_handle_t* handle, int* out_fitted);
n4m_status_t n4m_pp_localized_msc_transform( const n4m_pp_localized_msc_handle_t* handle, n4m_matrix_view_t X, n4m_matrix_view_t out);
```

Benchmark comparator backends are registered in the matrix and stored as reproducible snapshots when they define the canonical contract.

### Implementations

| Layer | Entry point | Language | Contract |
|-------|-------------|----------|----------|
| C ABI | `n4m_pp_localized_msc` | C/C++ | Stable libn4m entry point family. |
| Python | `n4m.python.localized_msc` | Python | ABI-close function backed by ctypes. |
| Python sklearn | `n4m.sklearn.LocalizedMSC` | Python | scikit-learn-compatible estimator backed by ctypes. |
| R | `localized_msc(X, reference = NULL, window = 16L, eps = 1e-12)` | R | Public package wrapper around the C ABI. |
| ref.sklearn | `sklearn.linear_model.LinearRegression(sliding local MSC)` | Python | canonical/comparator; sklearn supplies the per-window least-squares fit for N4M localized MSC |
| ref.r.prospectr | `prospectr.msc sliding interval` | R | canonical/comparator; prospectr supplies the MSC transform over each local N4M window |
| ref.scipy | `scipy.ndimage.convolve1d(local MSC window sums)` | Python | context only; SciPy supplies moving-window sums, but the raw-sum contract is numerically unstable on near-singular windows; sklearn is the strict gate |

### Usage

Every nirs4all-methods binding dispatches into the same C kernel. Registered comparator/source rows are listed in the benchmark card below.

::::{tab-set}
:class: nirs4all-methods-bindings


:::{tab-item} C ABI · libn4m
:sync: c
:class-label: lang-c

```c
n4m_status_t n4m_pp_localized_msc_create( n4m_pp_localized_msc_handle_t** out, const double* reference, int64_t n_reference, int32_t window, double eps);
void n4m_pp_localized_msc_destroy( n4m_pp_localized_msc_handle_t* handle);
n4m_status_t n4m_pp_localized_msc_fit( n4m_pp_localized_msc_handle_t* handle, n4m_matrix_view_t X);
n4m_status_t n4m_pp_localized_msc_is_fitted( const n4m_pp_localized_msc_handle_t* handle, int* out_fitted);
n4m_status_t n4m_pp_localized_msc_transform( const n4m_pp_localized_msc_handle_t* handle, n4m_matrix_view_t X, n4m_matrix_view_t out);
```

:::

:::{tab-item} Python ABI · n4m.python
:sync: python-abi
:class-label: lang-python

```python
from n4m import python as n4m

Xt = n4m.localized_msc(X)
```

:::

:::{tab-item} Python sklearn · n4m.sklearn
:sync: python-sklearn
:class-label: lang-python

```python
from n4m.sklearn import LocalizedMSC

op = LocalizedMSC()
Xt = op.fit_transform(X)
```

:::

:::{tab-item} R · nirs4all-methods
:sync: r
:class-label: lang-r

```r
library(n4m)
res <- localized_msc(X, reference = NULL, window = 17L, eps = 1e-12)
```

:::

::::


**Benchmark Comparators And Sources** ◆

:::{card}
:class-card: external-refs

- ◆ **`ref.sklearn`** (Python · canonical) — `sklearn.linear_model.LinearRegression(sliding local MSC)` · scikit-learn 1.8.0 — sklearn supplies the per-window least-squares fit for N4M localized MSC
- ◆ **`ref.r.prospectr`** (R · comparator) — `prospectr.msc sliding interval` · prospectr 0.2.8 — prospectr supplies the MSC transform over each local N4M window
- ℹ **`ref.scipy`** (Python · context) — `scipy.ndimage.convolve1d(local MSC window sums)` · scipy 1.17.1 — SciPy supplies moving-window sums, but the raw-sum contract is numerically unstable on near-singular windows; sklearn is the strict gate
:::

### Validation contract

- Operation: `fit_transform` · comparator: `default_allclose` · tolerance: `rtol=1e-05`, `atol=1e-08` · quality: **strict**
- Default validation dataset: `64×128` · seed `1234567908`
- Suites: smoke `3` cells; benchmark `11` cells · Default C/Python/reference parity comparator.
- Metrics: `finite_output`, `max_abs_diff`, `shape_equal`
- Truth sources:
  - `local_scatter_2026` — Demystifying Piecewise and Localized Scatter Correction Methods (doi:10.1002/cem.70113)

| Backend | Library | Gate | Comparator | Note |
|---------|---------|------|------------|------|
| `ref.sklearn` | `sklearn.linear_model.LinearRegression(sliding local MSC)` | Python / parity | `default_allclose` | sklearn supplies the per-window least-squares fit for N4M localized MSC |
| `ref.r.prospectr` | `prospectr.msc sliding interval` | R / parity | `default_allclose` | prospectr supplies the MSC transform over each local N4M window |
| `ref.scipy` | `scipy.ndimage.convolve1d(local MSC window sums)` | Python / context | `default_allclose` | SciPy supplies moving-window sums, but the raw-sum contract is numerically unstable on near-singular windows; sklearn is the strict gate |

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
<tr class="bk-row"><td class="bk-name"><code>N4M.cpp</code></td><td class="parity parity-divergence parity-exact" title="worst reference max abs diff over visible sizes">1.8e-08</td><td class="ms ms-best">🏆 0.097 ms</td><td class="ms">0.972 ms</td><td class="ms">4.909 ms</td></tr>
</tbody>
<tbody class="lang-band lang-python"><tr class="lang-band-row" data-lang="python"><th colspan="5" scope="rowgroup"><span class="lang-band-dot"></span>Python · nirs4all-methods</th></tr>
<tr class="bk-row"><td class="bk-name"><code>N4M.python</code></td><td class="parity parity-divergence parity-exact" title="worst reference max abs diff over visible sizes">1.8e-08</td><td class="ms">0.099 ms</td><td class="ms">0.934 ms</td><td class="ms">4.795 ms</td></tr>
<tr class="bk-row"><td class="bk-name"><code>N4M.sklearn</code></td><td class="parity parity-divergence parity-exact" title="worst reference max abs diff over visible sizes">1.8e-08</td><td class="ms">0.098 ms</td><td class="ms">0.961 ms</td><td class="ms">4.858 ms</td></tr>
</tbody>
<tbody class="lang-band lang-r"><tr class="lang-band-row" data-lang="r"><th colspan="5" scope="rowgroup"><span class="lang-band-dot"></span>R · nirs4all-methods</th></tr>
<tr class="bk-row"><td class="bk-name"><code>N4M.R</code></td><td class="parity parity-divergence parity-exact" title="worst reference max abs diff over visible sizes">1.8e-08</td><td class="ms">0.108 ms</td><td class="ms">1.180 ms</td><td class="ms">7.000 ms</td></tr>
</tbody>
<tbody class="lang-band lang-python"><tr class="lang-band-row" data-lang="python"><th colspan="5" scope="rowgroup"><span class="lang-band-dot"></span>Python · external</th></tr>
<tr class="bk-row truth-source-relaxed"><td class="bk-name"><span class="truth-mark" title="Registry parity reference (Python): scipy.ndimage.convolve1d(local MSC window sums) · scipy 1.17.1 — context">◆</span><code>ref.scipy</code></td><td class="parity parity-divergence parity-context" title="no divergence recorded">—</td><td class="ms">0.097 ms</td><td class="ms ms-best">🏆 0.721 ms</td><td class="ms ms-best">🏆 4.674 ms</td></tr>
<tr class="bk-row truth-source-strict"><td class="bk-name"><span class="truth-mark" title="Registry parity reference (Python): sklearn.linear_model.LinearRegression(sliding local MSC) · scikit-learn 1.8.0 — canonical">◆</span><code>ref.sklearn</code></td><td class="parity parity-divergence parity-exact" title="worst reference max abs diff over visible sizes">0</td><td class="ms">11.525 ms</td><td class="ms">123.901 ms</td><td class="ms">690.762 ms</td></tr>
</tbody>
<tbody class="lang-band lang-r"><tr class="lang-band-row" data-lang="r"><th colspan="5" scope="rowgroup"><span class="lang-band-dot"></span>R · external</th></tr>
<tr class="bk-row truth-source-relaxed"><td class="bk-name"><span class="truth-mark" title="Registry parity reference (R): prospectr.msc sliding interval · prospectr 0.2.8 — comparator">◆</span><code>ref.r.prospectr</code></td><td class="parity parity-divergence parity-exact" title="worst reference max abs diff over visible sizes">3.6e-13</td><td class="ms">2.844 ms</td><td class="ms">—</td><td class="ms">—</td></tr>
</tbody>
</table>
</div>

:::
::::

---

_See also_: [benchmark overview](../benchmarks/overview.md) · [methods index](index.md) · [interactive dashboard](../landing/dashboard.md)
