# `icoshift_alignment` — Icoshift alignment

_Group_: **Alignment** · _Registry tolerance_: `rtol=1e-5`, `atol=1e-8`

## Description

From the `chemometrics4all.IcoshiftAlignment` Python wrapper docstring:

> Interval correlation shifting with fixed-size intervals.

### Parameters

| Name | Type | Default | Notes |
|------|------|---------|-------|
| `reference` | `object` | `None` |  |
| `interval_size` | `int` | `32` |  |
| `max_shift` | `int` | `5` |  |

## Explanations

### Bibliographic source

- `ref.scipy` — scipy.ndimage.shift(interval-correlation shift search) (Python).

### Mathematical principle

`icoshift_alignment` follows the standard alignment operator contract: an input matrix $\mathbf{X}\in\mathbb{R}^{n\times p}$ is transformed, scored, split, or filtered by the method-specific kernel while preserving the parity contract recorded by the benchmark snapshots. The precise numerical convention is the one exposed by the C ABI signatures and the registered external references below.

### Implementation

C ABI entry points used by the language bindings:

```c
c4a_status_t c4a_pp_icoshift_align_create( c4a_pp_icoshift_align_handle_t** out, const double* reference, int64_t n_reference, int32_t interval_size, int32_t max_shift);
void c4a_pp_icoshift_align_destroy( c4a_pp_icoshift_align_handle_t* handle);
c4a_status_t c4a_pp_icoshift_align_fit( c4a_pp_icoshift_align_handle_t* handle, c4a_matrix_view_t X);
c4a_status_t c4a_pp_icoshift_align_is_fitted( const c4a_pp_icoshift_align_handle_t* handle, int* out_fitted);
c4a_status_t c4a_pp_icoshift_align_transform( const c4a_pp_icoshift_align_handle_t* handle, c4a_matrix_view_t X, c4a_matrix_view_t out);
```

Benchmark comparator backends are registered in the matrix and stored as reproducible snapshots when they define the canonical contract.

### Implementations

| Layer | Entry point | Language | Contract |
|-------|-------------|----------|----------|
| C ABI | `c4a_pp_icoshift_align` | C/C++ | Stable libc4a entry point family. |
| Python | `chemometrics4all.python.icoshift_alignment` | Python | ABI-close function backed by ctypes. |
| Python sklearn | `chemometrics4all.sklearn.IcoshiftAlignment` | Python | scikit-learn-compatible estimator backed by ctypes. |
| R | `icoshift_alignment(X, reference = NULL, interval_size = 16L, max_shift = 3L)` | R | Public package wrapper around the C ABI. |
| ref.scipy | `scipy.ndimage.shift(interval-correlation shift search)` | Python | canonical/comparator; SciPy supplies the edge-clamped interpolation used inside each C4A interval |

### Usage

Every chemometrics4all binding dispatches into the same C kernel. Registered comparator/source rows are listed in the benchmark card below.

::::{tab-set}
:class: chemometrics4all-bindings


:::{tab-item} C ABI · libc4a
:sync: c
:class-label: lang-c

```c
c4a_status_t c4a_pp_icoshift_align_create( c4a_pp_icoshift_align_handle_t** out, const double* reference, int64_t n_reference, int32_t interval_size, int32_t max_shift);
void c4a_pp_icoshift_align_destroy( c4a_pp_icoshift_align_handle_t* handle);
c4a_status_t c4a_pp_icoshift_align_fit( c4a_pp_icoshift_align_handle_t* handle, c4a_matrix_view_t X);
c4a_status_t c4a_pp_icoshift_align_is_fitted( const c4a_pp_icoshift_align_handle_t* handle, int* out_fitted);
c4a_status_t c4a_pp_icoshift_align_transform( const c4a_pp_icoshift_align_handle_t* handle, c4a_matrix_view_t X, c4a_matrix_view_t out);
```

:::

:::{tab-item} Python ABI · chemometrics4all.python
:sync: python-abi
:class-label: lang-python

```python
from chemometrics4all import python as c4a

Xt = c4a.icoshift_alignment(X)
```

:::

:::{tab-item} Python sklearn · chemometrics4all.sklearn
:sync: python-sklearn
:class-label: lang-python

```python
from chemometrics4all.sklearn import IcoshiftAlignment

op = IcoshiftAlignment(reference=None, interval_size=32, max_shift=5)
Xt = op.fit_transform(X)
```

:::

:::{tab-item} R · chemometrics4all
:sync: r
:class-label: lang-r

```r
library(chemometrics4all)
res <- icoshift_alignment(X, reference = NULL, interval_size = 16L, max_shift = 2L)
```

:::

::::


**Benchmark Comparators And Sources** ◆

:::{card}
:class-card: external-refs

- ◆ **`ref.scipy`** (Python · canonical) — `scipy.ndimage.shift(interval-correlation shift search)` · scipy 1.17.1 — SciPy supplies the edge-clamped interpolation used inside each C4A interval
:::

### Validation contract

- Operation: `fit_transform` · comparator: `default_allclose` · tolerance: `rtol=1e-05`, `atol=1e-08` · quality: **strict**
- Default validation dataset: `32×96` · seed `1234567912`
- Suites: smoke `3` cells; benchmark `11` cells · Default C/Python/reference parity comparator.
- Metrics: `finite_output`, `max_abs_diff`, `shape_equal`
- Truth sources:
  - `savorani_icoshift_2010` — icoshift: A versatile tool for the rapid alignment of 1D NMR spectra (doi:10.1016/j.jmr.2009.11.012)

| Backend | Library | Gate | Comparator | Note |
|---------|---------|------|------------|------|
| `ref.scipy` | `scipy.ndimage.shift(interval-correlation shift search)` | Python / parity | `default_allclose` | SciPy supplies the edge-clamped interpolation used inside each C4A interval |

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
<tr class="bk-row"><td class="bk-name"><code>C4A.cpp</code></td><td class="parity parity-divergence parity-exact" title="worst reference max abs diff over visible sizes">0</td><td class="ms ms-best">🏆 0.085 ms</td><td class="ms">0.852 ms</td><td class="ms ms-best">🏆 4.077 ms</td></tr>
</tbody>
<tbody class="lang-band lang-python"><tr class="lang-band-row" data-lang="python"><th colspan="5" scope="rowgroup"><span class="lang-band-dot"></span>Python · chemometrics4all</th></tr>
<tr class="bk-row"><td class="bk-name"><code>C4A.python</code></td><td class="parity parity-divergence parity-exact" title="worst reference max abs diff over visible sizes">0</td><td class="ms">0.086 ms</td><td class="ms">0.853 ms</td><td class="ms">4.153 ms</td></tr>
<tr class="bk-row"><td class="bk-name"><code>C4A.sklearn</code></td><td class="parity parity-divergence parity-exact" title="worst reference max abs diff over visible sizes">0</td><td class="ms">0.094 ms</td><td class="ms ms-best">🏆 0.836 ms</td><td class="ms">4.282 ms</td></tr>
</tbody>
<tbody class="lang-band lang-r"><tr class="lang-band-row" data-lang="r"><th colspan="5" scope="rowgroup"><span class="lang-band-dot"></span>R · chemometrics4all</th></tr>
<tr class="bk-row"><td class="bk-name"><code>C4A.R</code></td><td class="parity parity-divergence parity-exact" title="worst reference max abs diff over visible sizes">5.6e-16</td><td class="ms">0.107 ms</td><td class="ms">1.156 ms</td><td class="ms">7.188 ms</td></tr>
</tbody>
<tbody class="lang-band lang-python"><tr class="lang-band-row" data-lang="python"><th colspan="5" scope="rowgroup"><span class="lang-band-dot"></span>Python · external</th></tr>
<tr class="bk-row truth-source-strict"><td class="bk-name"><span class="truth-mark" title="Registry parity reference (Python): scipy.ndimage.shift(interval-correlation shift search) · scipy 1.17.1 — canonical">◆</span><code>ref.scipy</code></td><td class="parity parity-divergence parity-exact" title="worst reference max abs diff over visible sizes">0</td><td class="ms">16.574 ms</td><td class="ms">146.984 ms</td><td class="ms">693.895 ms</td></tr>
</tbody>
</table>
</div>

:::
::::

---

_See also_: [benchmark overview](../benchmarks/overview.md) · [methods index](index.md) · [interactive dashboard](../landing/dashboard.md)
