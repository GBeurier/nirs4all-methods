# `detrend` — Detrend

_Group_: **Baseline** · _Registry tolerance_: `rtol=1e-5`, `atol=1e-8` · _Source_: [`docs/algorithms/detrend.md`](../algorithms/detrend.md)

## Description

Per-row polynomial detrending. For each row of `X` of length `p`:

1. Build positions `t = [0, 1, …, p-1]`.
2. Fit a polynomial of degree `polyorder` to `(t, X[i, :])` via least-squares (Householder QR).
3. Evaluate the fitted polynomial at the same positions.
4. Subtract: `out[i, j] = X[i, j] - polyval(coefs, t)[j]`.

From the `chemometrics4all.Detrend` Python wrapper docstring:

> Polynomial baseline subtraction.

### Parameters

| Name | Type | Default | Notes |
|------|------|---------|-------|
| `polyorder` | `int` | `1` | Polynomial order. |

## Explanations

### Bibliographic source

- Standard polynomial detrending; see `numpy.polyfit` / `numpy.polyval` documentation.

### Mathematical principle

`detrend` follows the standard baseline operator contract: an input matrix $\mathbf{X}\in\mathbb{R}^{n\times p}$ is transformed, scored, split, or filtered by the method-specific kernel while preserving the parity contract recorded by the benchmark snapshots. The precise numerical convention is the one exposed by the C ABI signatures and the registered external references below.

### Implementation

- Closed-form least-squares via the shared Householder QR helper in `core/common/linalg.{c,h}`.
- True per-element division and subtraction; no reciprocal-multiplication shortcut.
- Parity tolerance vs internal parity fixture: `1e-11 abs / 1e-12 rel`.
- Internal parity fixture: `parity/python_generator/src/c4a_parity_pybaselines_ref/detrend.py` (uses `np.polyfit`/`np.polyval`).

C ABI entry points used by the language bindings:

```c
c4a_status_t c4a_pp_detrend_create(c4a_pp_detrend_handle_t** out, int32_t polyorder);
void c4a_pp_detrend_destroy(c4a_pp_detrend_handle_t* handle);
c4a_status_t c4a_pp_detrend_transform( const c4a_pp_detrend_handle_t* handle, c4a_matrix_view_t X, c4a_matrix_view_t out);
```

Benchmark comparator backends are registered in the matrix and stored as reproducible snapshots when they define the canonical contract.

### Implementations

| Layer | Entry point | Language | Contract |
|-------|-------------|----------|----------|
| C ABI | `c4a_pp_detrend` | C/C++ | Stable libc4a entry point family. |
| Python | `chemometrics4all.python.detrend` | Python | ABI-close function backed by ctypes. |
| Python sklearn | `chemometrics4all.sklearn.Detrend` | Python | scikit-learn-compatible estimator backed by ctypes. |
| R | `detrend(X, polyorder = 1L)` | R | Public package wrapper around the C ABI. |
| ref.scipy | `scipy.signal.detrend` | Python | canonical/comparator |
| ref.nirs4all | `nirs4all.Detrend` | Python | canonical/comparator |
| ref.r.prospectr | `prospectr.detrend` | R | context only; prospectr combines the NIR detrend convention with its own wavelength argument contract |

### Usage

Every chemometrics4all binding dispatches into the same C kernel. Registered comparator/source rows are listed in the benchmark card below.

::::{tab-set}
:class: chemometrics4all-bindings


:::{tab-item} C ABI · libc4a
:sync: c
:class-label: lang-c

```c
c4a_status_t c4a_pp_detrend_create(c4a_pp_detrend_handle_t** out, int32_t polyorder);
void c4a_pp_detrend_destroy(c4a_pp_detrend_handle_t* handle);
c4a_status_t c4a_pp_detrend_transform( const c4a_pp_detrend_handle_t* handle, c4a_matrix_view_t X, c4a_matrix_view_t out);
```

:::

:::{tab-item} Python ABI · chemometrics4all.python
:sync: python-abi
:class-label: lang-python

```python
from chemometrics4all import python as c4a

Xt = c4a.detrend(X)
```

:::

:::{tab-item} Python sklearn · chemometrics4all.sklearn
:sync: python-sklearn
:class-label: lang-python

```python
from chemometrics4all.sklearn import Detrend

op = Detrend(polyorder=1)
Xt = op.fit_transform(X)
```

:::

:::{tab-item} R · chemometrics4all
:sync: r
:class-label: lang-r

```r
library(chemometrics4all)
res <- detrend(X, polyorder = 1L)
```

:::

::::


**Benchmark Comparators And Sources** ◆

:::{card}
:class-card: external-refs

- ◆ **`ref.scipy`** (Python · canonical) — `scipy.signal.detrend` · scipy 1.17.1
- ◆ **`ref.nirs4all`** (Python · comparator) — `nirs4all.Detrend` · nirs4all@cd731a23+dirty
- ℹ **`ref.r.prospectr`** (R · context) — `prospectr.detrend` · prospectr 0.2.8 — prospectr combines the NIR detrend convention with its own wavelength argument contract
:::

### Validation contract

- Operation: `cross_binding_callable` · comparator: `default_allclose` · tolerance: `rtol=1e-05`, `atol=1e-08` · quality: **strict**
- Default validation dataset: `100×50` · seed `20260556`
- Suites: smoke `3` cells; benchmark `11` cells · Default C/Python/reference parity comparator.
- Metrics: `max_abs_diff`, `rel_l2_diff`, `rms_diff`, `shape_equal`
- Truth sources: cross-binding references declared directly in `benchmarks/cross_binding/orchestrator.py`.

| Backend | Library | Gate | Comparator | Note |
|---------|---------|------|------------|------|
| `ref.scipy` | `scipy.signal.detrend` | Python / parity | `default_allclose` |  |
| `ref.nirs4all` | `nirs4all.Detrend` | Python / parity | `default_allclose` |  |
| `ref.r.prospectr` | `prospectr.detrend` | R / context | `default_allclose` | prospectr combines the NIR detrend convention with its own wavelength argument contract |

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
<tr class="bk-row"><td class="bk-name"><code>C4A.cpp</code></td><td class="parity parity-divergence parity-exact" title="worst reference max abs diff over visible sizes">3.0e-15</td><td class="ms ms-best">🏆 0.011 ms</td><td class="ms ms-best">🏆 0.108 ms</td><td class="ms ms-best">🏆 0.598 ms</td></tr>
</tbody>
<tbody class="lang-band lang-python"><tr class="lang-band-row" data-lang="python"><th colspan="5" scope="rowgroup"><span class="lang-band-dot"></span>Python · chemometrics4all</th></tr>
<tr class="bk-row"><td class="bk-name"><code>C4A.python</code></td><td class="parity parity-divergence parity-exact" title="worst reference max abs diff over visible sizes">3.0e-15</td><td class="ms">0.016 ms</td><td class="ms">0.122 ms</td><td class="ms">0.676 ms</td></tr>
<tr class="bk-row"><td class="bk-name"><code>C4A.sklearn</code></td><td class="parity parity-divergence parity-exact" title="worst reference max abs diff over visible sizes">3.0e-15</td><td class="ms">0.020 ms</td><td class="ms">0.150 ms</td><td class="ms">0.762 ms</td></tr>
</tbody>
<tbody class="lang-band lang-r"><tr class="lang-band-row" data-lang="r"><th colspan="5" scope="rowgroup"><span class="lang-band-dot"></span>R · chemometrics4all</th></tr>
<tr class="bk-row"><td class="bk-name"><code>C4A.R</code></td><td class="parity parity-divergence parity-exact" title="worst reference max abs diff over visible sizes">3.4e-15</td><td class="ms">0.034 ms</td><td class="ms">0.268 ms</td><td class="ms">2.219 ms</td></tr>
</tbody>
<tbody class="lang-band lang-python"><tr class="lang-band-row" data-lang="python"><th colspan="5" scope="rowgroup"><span class="lang-band-dot"></span>Python · external</th></tr>
<tr class="bk-row truth-source-relaxed"><td class="bk-name"><span class="truth-mark" title="Registry parity reference (Python): nirs4all.Detrend · nirs4all@cd731a23+dirty — comparator">◆</span><code>ref.nirs4all</code></td><td class="parity parity-divergence parity-exact" title="worst reference max abs diff over visible sizes">0</td><td class="ms">0.083 ms</td><td class="ms">0.317 ms</td><td class="ms">3.037 ms</td></tr>
<tr class="bk-row truth-source-strict"><td class="bk-name"><span class="truth-mark" title="Registry parity reference (Python): scipy.signal.detrend · scipy 1.17.1 — canonical">◆</span><code>ref.scipy</code></td><td class="parity parity-divergence parity-exact" title="worst reference max abs diff over visible sizes">0</td><td class="ms">0.085 ms</td><td class="ms">0.280 ms</td><td class="ms">2.673 ms</td></tr>
</tbody>
<tbody class="lang-band lang-r"><tr class="lang-band-row" data-lang="r"><th colspan="5" scope="rowgroup"><span class="lang-band-dot"></span>R · external</th></tr>
<tr class="bk-row truth-source-relaxed"><td class="bk-name"><span class="truth-mark" title="Registry parity reference (R): prospectr.detrend · prospectr 0.2.8 — context">◆</span><code>ref.r.prospectr</code></td><td class="parity parity-divergence parity-context" title="worst reference max abs diff over visible sizes">1.685</td><td class="ms">0.859 ms</td><td class="ms">2.281 ms</td><td class="ms">11.062 ms</td></tr>
</tbody>
</table>
</div>

:::
::::

---

_See also_: [benchmark overview](../benchmarks/overview.md) · [methods index](index.md) · [interactive dashboard](../landing/dashboard.md)
