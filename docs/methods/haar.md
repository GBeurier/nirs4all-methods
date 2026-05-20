# `haar` — Haar

_Group_: **Wavelet** · _Registry tolerance_: `rtol=1e-5`, `atol=1e-8` · _Source_: [`docs/algorithms/haar.md`](../algorithms/haar.md)

## Description

From the `chemometrics4all.Haar` Python wrapper docstring:

> Single-level Haar DWT coefficient transform.

### Parameters

None.

## Explanations

### Bibliographic source

`nirs4all.operators.transforms.nirs.Haar`.

### Mathematical principle

Convenience wrapper around the `Wavelet` operator with the
`family = haar` and `mode = periodization` defaults.  Matches
`nirs4all.operators.transforms.nirs.Haar` (which itself is a
zero-argument subclass of `Wavelet`).

The Haar filter bank is the simplest orthogonal wavelet ($L = 2$):

$$
h_{\text{lo}} = (1/\sqrt 2,\; 1/\sqrt 2), \qquad
h_{\text{hi}} = (-1/\sqrt 2,\; 1/\sqrt 2).
$$

For an input row $x \in \mathbb{R}^n$ (with $n$ even, or
internally padded to even), the periodization Haar DWT reduces to:

$$
c_A[k] = \frac{x[2k] + x[2k+1]}{\sqrt 2}, \qquad
c_D[k] = \frac{x[2k+1] - x[2k]}{\sqrt 2}.
$$

Output is the concatenation $[c_A \,\Vert\, c_D]$ of length $2m$ with
$m = \lceil n / 2 \rceil$.

### Implementation

Stateless: `_create / _transform / _destroy`.  `_output_cols(input_cols)`
returns the row width $2m$.

C ABI entry points used by the language bindings:

```c
c4a_status_t c4a_pp_haar_create(c4a_pp_haar_handle_t** out);
void c4a_pp_haar_destroy(c4a_pp_haar_handle_t* handle);
c4a_status_t c4a_pp_haar_output_cols(int64_t input_cols, int64_t* out_cols);
c4a_status_t c4a_pp_haar_transform(const c4a_pp_haar_handle_t* handle, c4a_matrix_view_t X, c4a_matrix_view_t out);
```

Benchmark comparator backends are registered in the matrix and stored as reproducible snapshots when they define the canonical contract.

### Implementations

| Layer | Entry point | Language | Contract |
|-------|-------------|----------|----------|
| C ABI | `c4a_pp_haar` | C/C++ | Stable libc4a entry point family. |
| Python | `chemometrics4all.python.haar` | Python | ABI-close function backed by ctypes. |
| Python sklearn | `chemometrics4all.sklearn.Haar` | Python | scikit-learn-compatible estimator backed by ctypes. |
| R | `haar(X)` | R | Public package wrapper around the C ABI. |
| ref.pywavelets | `PyWavelets.dwt(haar, periodization)` | Python | canonical/comparator |
| ref.nirs4all | `nirs4all.Haar(detail-resampled)` | Python | context only; nirs4all returns resampled detail coefficients only; c4a/PyWavelets gate the concatenated approximation+detail coefficients |

### Usage

Every chemometrics4all binding dispatches into the same C kernel. Registered comparator/source rows are listed in the benchmark card below.

::::{tab-set}
:class: chemometrics4all-bindings


:::{tab-item} C ABI · libc4a
:sync: c
:class-label: lang-c

```c
c4a_status_t c4a_pp_haar_create(c4a_pp_haar_handle_t** out);
void c4a_pp_haar_destroy(c4a_pp_haar_handle_t* handle);
c4a_status_t c4a_pp_haar_output_cols(int64_t input_cols, int64_t* out_cols);
c4a_status_t c4a_pp_haar_transform(const c4a_pp_haar_handle_t* handle, c4a_matrix_view_t X, c4a_matrix_view_t out);
```

:::

:::{tab-item} Python ABI · chemometrics4all.python
:sync: python-abi
:class-label: lang-python

```python
from chemometrics4all import python as c4a

Xt = c4a.haar(X)
```

:::

:::{tab-item} Python sklearn · chemometrics4all.sklearn
:sync: python-sklearn
:class-label: lang-python

```python
from chemometrics4all.sklearn import Haar

op = Haar()
Xt = op.fit_transform(X)
```

:::

:::{tab-item} R · chemometrics4all
:sync: r
:class-label: lang-r

```r
library(chemometrics4all)
res <- haar(X)
```

:::

::::


**Benchmark Comparators And Sources** ◆

:::{card}
:class-card: external-refs

- ◆ **`ref.pywavelets`** (Python · canonical) — `PyWavelets.dwt(haar, periodization)` · PyWavelets 1.9.0
- ℹ **`ref.nirs4all`** (Python · context) — `nirs4all.Haar(detail-resampled)` · nirs4all@cd731a23+dirty — nirs4all returns resampled detail coefficients only; c4a/PyWavelets gate the concatenated approximation+detail coefficients
:::

### Validation contract

- Operation: `cross_binding_callable` · comparator: `default_allclose` · tolerance: `rtol=1e-05`, `atol=1e-08` · quality: **strict**
- Default validation dataset: `100×50` · seed `20260556`
- Suites: smoke `3` cells; benchmark `11` cells · Default C/Python/reference parity comparator.
- Metrics: `max_abs_diff`, `rel_l2_diff`, `rms_diff`, `shape_equal`
- Truth sources: cross-binding references declared directly in `benchmarks/cross_binding/orchestrator.py`.

| Backend | Library | Gate | Comparator | Note |
|---------|---------|------|------------|------|
| `ref.pywavelets` | `PyWavelets.dwt(haar, periodization)` | Python / parity | `default_allclose` |  |
| `ref.nirs4all` | `nirs4all.Haar(detail-resampled)` | Python / context | `default_allclose` | nirs4all returns resampled detail coefficients only; c4a/PyWavelets gate the concatenated approximation+detail coefficients |

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
<tr class="bk-row"><td class="bk-name"><code>C4A.cpp</code></td><td class="parity parity-divergence parity-exact" title="worst reference max abs diff over visible sizes">2.2e-16</td><td class="ms ms-best">🏆 0.002 ms</td><td class="ms ms-best">🏆 0.015 ms</td><td class="ms ms-best">🏆 0.096 ms</td></tr>
</tbody>
<tbody class="lang-band lang-python"><tr class="lang-band-row" data-lang="python"><th colspan="5" scope="rowgroup"><span class="lang-band-dot"></span>Python · chemometrics4all</th></tr>
<tr class="bk-row"><td class="bk-name"><code>C4A.python</code></td><td class="parity parity-divergence parity-exact" title="worst reference max abs diff over visible sizes">2.2e-16</td><td class="ms">0.007 ms</td><td class="ms">0.021 ms</td><td class="ms">0.099 ms</td></tr>
<tr class="bk-row"><td class="bk-name"><code>C4A.sklearn</code></td><td class="parity parity-divergence parity-exact" title="worst reference max abs diff over visible sizes">2.2e-16</td><td class="ms">0.009 ms</td><td class="ms">0.030 ms</td><td class="ms">0.097 ms</td></tr>
</tbody>
<tbody class="lang-band lang-r"><tr class="lang-band-row" data-lang="r"><th colspan="5" scope="rowgroup"><span class="lang-band-dot"></span>R · chemometrics4all</th></tr>
<tr class="bk-row"><td class="bk-name"><code>C4A.R</code></td><td class="parity parity-divergence parity-exact" title="worst reference max abs diff over visible sizes">5.1e-15</td><td class="ms">0.029 ms</td><td class="ms">0.268 ms</td><td class="ms">1.539 ms</td></tr>
</tbody>
<tbody class="lang-band lang-python"><tr class="lang-band-row" data-lang="python"><th colspan="5" scope="rowgroup"><span class="lang-band-dot"></span>Python · external</th></tr>
<tr class="bk-row truth-source-relaxed"><td class="bk-name"><span class="truth-mark" title="Registry parity reference (Python): nirs4all.Haar(detail-resampled) · nirs4all@cd731a23+dirty — context">◆</span><code>ref.nirs4all</code></td><td class="parity parity-divergence parity-context" title="no divergence recorded">—</td><td class="ms">0.054 ms</td><td class="ms">0.297 ms</td><td class="ms">1.789 ms</td></tr>
<tr class="bk-row truth-source-strict"><td class="bk-name"><span class="truth-mark" title="Registry parity reference (Python): PyWavelets.dwt(haar, periodization) · PyWavelets 1.9.0 — canonical">◆</span><code>ref.pywavelets</code></td><td class="parity parity-divergence parity-exact" title="worst reference max abs diff over visible sizes">0</td><td class="ms">0.013 ms</td><td class="ms">0.064 ms</td><td class="ms">0.318 ms</td></tr>
</tbody>
</table>
</div>

:::
::::

---

_See also_: [benchmark overview](../benchmarks/overview.md) · [methods index](index.md) · [interactive dashboard](../landing/dashboard.md)
