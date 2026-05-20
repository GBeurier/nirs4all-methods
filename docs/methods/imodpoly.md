# `imodpoly` — IModPoly

_Group_: **Baseline** · _Registry tolerance_: `rtol=1e-5`, `atol=1e-8` · _Source_: [`docs/algorithms/imodpoly.md`](../algorithms/imodpoly.md)

## Description

Gan, Ruan & Mo 2006 σ-stopping variant of ModPoly. For each row `y` of length `n`:

1. Fit polynomial baseline `z`; compute `devr = std(y - z, ddof = 0)`.
2. Repeat up to `max_iter`:
   - Clip peaks: `y[i] := y[i] if y[i] < z[i] + devr else (z[i] + devr)`.
   - Re-fit polynomial -> new `z`.
   - `devr_new = std(y - z, ddof = 0)`.
   - `conv = |devr_new - devr| / max(devr_new, DBL_MIN)`.
   - `devr = devr_new`.
   - If `conv < tol`, break.
3. Output: `out = y_original - z`.

From the `n4m.IModPoly` Python wrapper docstring:

> Improved modified polynomial baseline correction.

### Parameters

| Name | Type | Default | Notes |
|------|------|---------|-------|
| `polyorder` | `int` | `2` | Polynomial order. |
| `max_iter` | `int` | `250` | Maximum number of solver or reweighting iterations. |
| `tol` | `float` | `0.001` | Convergence tolerance. |

## Explanations

### Bibliographic source

- Gan, F., Ruan, G. & Mo, J. (2006). "Baseline correction by improved iterative polynomial fitting with automatic threshold." Chemometrics and Intelligent Laboratory Systems 82(1-2), 59-65.
- Internal parity fixture: `parity/python_generator/src/n4m_parity_pybaselines_ref/imodpoly.py`.

### Mathematical principle

`imodpoly` follows the standard baseline operator contract: an input matrix $\mathbf{X}\in\mathbb{R}^{n\times p}$ is transformed, scored, split, or filtered by the method-specific kernel while preserving the parity contract recorded by the benchmark snapshots. The precise numerical convention is the one exposed by the C ABI signatures and the registered external references below.

### Implementation

- Same Householder-QR Vandermonde factorisation as ModPoly (shared from `core/common/linalg.h`).
- Residual stdev uses ddof = 0 (population); the convergence check protects the denominator with `max(devr_new, DBL_MIN)`.
- `mask_initial_peaks` (pybaselines' optional pre-threshold step) is disabled; this matches `num_std = 1` default behaviour on clean spectra and keeps the parity surface compact.
- On `max_iter` reached without convergence: silently returns the last iterate.
- Parity tolerance vs internal parity fixture: `1e-9 abs / 1e-10 rel`.

C ABI entry points used by the language bindings:

```c
n4m_status_t n4m_pp_imodpoly_create(n4m_pp_imodpoly_handle_t** out, int32_t polyorder, int32_t max_iter, double tol);
void n4m_pp_imodpoly_destroy(n4m_pp_imodpoly_handle_t* handle);
n4m_status_t n4m_pp_imodpoly_transform( const n4m_pp_imodpoly_handle_t* handle, n4m_matrix_view_t X, n4m_matrix_view_t out);
```

Benchmark comparator backends are registered in the matrix and stored as reproducible snapshots when they define the canonical contract.

### Implementations

| Layer | Entry point | Language | Contract |
|-------|-------------|----------|----------|
| C ABI | `n4m_pp_imodpoly` | C/C++ | Stable libn4m entry point family. |
| Python | `n4m.python.imodpoly` | Python | ABI-close function backed by ctypes. |
| Python sklearn | `n4m.sklearn.IModPoly` | Python | scikit-learn-compatible estimator backed by ctypes. |
| R | `imodpoly(X, polyorder = 2L, max_iter = 250L, tol = 1e-3)` | R | Public package wrapper around the C ABI. |
| ref.pybaselines | `pybaselines.imodpoly(mask_initial_peaks=False)` | Python | canonical/comparator |

### Usage

Every nirs4all-methods binding dispatches into the same C kernel. Registered comparator/source rows are listed in the benchmark card below.

::::{tab-set}
:class: nirs4all-methods-bindings


:::{tab-item} C ABI · libn4m
:sync: c
:class-label: lang-c

```c
n4m_status_t n4m_pp_imodpoly_create(n4m_pp_imodpoly_handle_t** out, int32_t polyorder, int32_t max_iter, double tol);
void n4m_pp_imodpoly_destroy(n4m_pp_imodpoly_handle_t* handle);
n4m_status_t n4m_pp_imodpoly_transform( const n4m_pp_imodpoly_handle_t* handle, n4m_matrix_view_t X, n4m_matrix_view_t out);
```

:::

:::{tab-item} Python ABI · n4m.python
:sync: python-abi
:class-label: lang-python

```python
from n4m import python as n4m

Xt = n4m.imodpoly(X)
```

:::

:::{tab-item} Python sklearn · n4m.sklearn
:sync: python-sklearn
:class-label: lang-python

```python
from n4m.sklearn import IModPoly

op = IModPoly(polyorder=2, max_iter=250, tol=0.001)
Xt = op.fit_transform(X)
```

:::

:::{tab-item} R · nirs4all-methods
:sync: r
:class-label: lang-r

```r
library(n4m)
res <- imodpoly(X, polyorder = 2L, max_iter = 50L)
```

:::

::::


**Benchmark Comparators And Sources** ◆

:::{card}
:class-card: external-refs

- ◆ **`ref.pybaselines`** (Python · canonical) — `pybaselines.imodpoly(mask_initial_peaks=False)` · pybaselines 1.2.1
:::

### Validation contract

- Operation: `cross_binding_callable` · comparator: `default_allclose` · tolerance: `rtol=1e-05`, `atol=1e-08` · quality: **strict**
- Default validation dataset: `100×50` · seed `20260556`
- Suites: smoke `3` cells; benchmark `11` cells · Default C/Python/reference parity comparator.
- Metrics: `max_abs_diff`, `rel_l2_diff`, `rms_diff`, `shape_equal`
- Truth sources: cross-binding references declared directly in `benchmarks/cross_binding/orchestrator.py`.

| Backend | Library | Gate | Comparator | Note |
|---------|---------|------|------------|------|
| `ref.pybaselines` | `pybaselines.imodpoly(mask_initial_peaks=False)` | Python / parity | `default_allclose` |  |

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
<tr class="bk-row"><td class="bk-name"><code>N4M.cpp</code></td><td class="parity parity-divergence parity-exact" title="worst reference max abs diff over visible sizes">1.3e-14</td><td class="ms ms-best">🏆 0.531 ms</td><td class="ms ms-best">🏆 4.941 ms</td><td class="ms">25.764 ms</td></tr>
</tbody>
<tbody class="lang-band lang-python"><tr class="lang-band-row" data-lang="python"><th colspan="5" scope="rowgroup"><span class="lang-band-dot"></span>Python · nirs4all-methods</th></tr>
<tr class="bk-row"><td class="bk-name"><code>N4M.python</code></td><td class="parity parity-divergence parity-exact" title="worst reference max abs diff over visible sizes">1.3e-14</td><td class="ms">0.558 ms</td><td class="ms">5.017 ms</td><td class="ms">25.931 ms</td></tr>
<tr class="bk-row"><td class="bk-name"><code>N4M.sklearn</code></td><td class="parity parity-divergence parity-exact" title="worst reference max abs diff over visible sizes">1.3e-14</td><td class="ms">0.546 ms</td><td class="ms">5.080 ms</td><td class="ms ms-best">🏆 25.567 ms</td></tr>
</tbody>
<tbody class="lang-band lang-r"><tr class="lang-band-row" data-lang="r"><th colspan="5" scope="rowgroup"><span class="lang-band-dot"></span>R · nirs4all-methods</th></tr>
<tr class="bk-row"><td class="bk-name"><code>N4M.R</code></td><td class="parity parity-divergence parity-exact" title="worst reference max abs diff over visible sizes">1.3e-14</td><td class="ms">0.609 ms</td><td class="ms">5.188 ms</td><td class="ms">27.500 ms</td></tr>
</tbody>
<tbody class="lang-band lang-python"><tr class="lang-band-row" data-lang="python"><th colspan="5" scope="rowgroup"><span class="lang-band-dot"></span>Python · external</th></tr>
<tr class="bk-row truth-source-strict"><td class="bk-name"><span class="truth-mark" title="Registry parity reference (Python): pybaselines.imodpoly(mask_initial_peaks=False) · pybaselines 1.2.1 — canonical">◆</span><code>ref.pybaselines</code></td><td class="parity parity-divergence parity-exact" title="worst reference max abs diff over visible sizes">0</td><td class="ms">26.417 ms</td><td class="ms">29.760 ms</td><td class="ms">44.267 ms</td></tr>
</tbody>
</table>
</div>

:::
::::

---

_See also_: [benchmark overview](../benchmarks/overview.md) · [methods index](index.md) · [interactive dashboard](../landing/dashboard.md)
