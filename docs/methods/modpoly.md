# `modpoly` — ModPoly

_Group_: **Baseline** · _Registry tolerance_: `rtol=1e-5`, `atol=1e-8` · _Source_: [`docs/algorithms/modpoly.md`](../algorithms/modpoly.md)

## Description

Lieber & Mahadevan-Jansen 2003 iterative polynomial baseline. For each row `y` of length `n`:

1. Fit polynomial of degree `polyorder` to `(positions = [0..n-1], y)`; evaluate at positions to get baseline `z`.
2. Repeat up to `max_iter`:
   - Clip peaks: `y[i] = min(y[i], z[i])`.
   - Re-fit polynomial -> new baseline `z_new`.
   - If `rel_l2_diff(z, z_new) < tol`, break.
   - `z = z_new`.
3. Output: `out[i] = y_original[i] - z[i]`.

From the `n4m.ModPoly` Python wrapper docstring:

> Modified polynomial baseline correction.

### Parameters

| Name | Type | Default | Notes |
|------|------|---------|-------|
| `polyorder` | `int` | `2` | Polynomial order. |
| `max_iter` | `int` | `250` | Maximum number of solver or reweighting iterations. |
| `tol` | `float` | `0.001` | Convergence tolerance. |

## Explanations

### Bibliographic source

- Lieber, C. A. & Mahadevan-Jansen, A. (2003). "Automated method for subtraction of fluorescence from biological raman spectra." Applied Spectroscopy 57(11), 1363-1367.
- Internal parity fixture: `parity/python_generator/src/n4m_parity_pybaselines_ref/modpoly.py`.

### Mathematical principle

`modpoly` follows the standard baseline operator contract: an input matrix $\mathbf{X}\in\mathbb{R}^{n\times p}$ is transformed, scored, split, or filtered by the method-specific kernel while preserving the parity contract recorded by the benchmark snapshots. The precise numerical convention is the one exposed by the C ABI signatures and the registered external references below.

### Implementation

- Per-row Householder QR factorisation of the `(n, polyorder + 1)` Vandermonde matrix in descending powers (shared with `Detrend` / `IModPoly` / `IAsLS` via `core/common/linalg.{c,h}`). Factor once per row; re-apply `Q^T` against the clipped y at each iteration.
- Convergence uses `relative_difference(z_old, z_new)` from `banded_solver.h` (L2 norms, `max(||z_old||, DBL_MIN)` denominator).
- On `max_iter` reached without convergence: silently returns the last iterate.
- Parity tolerance vs internal parity fixture: `1e-9 abs / 1e-10 rel`.

C ABI entry points used by the language bindings:

```c
n4m_status_t n4m_pp_modpoly_create(n4m_pp_modpoly_handle_t** out, int32_t polyorder, int32_t max_iter, double tol);
void n4m_pp_modpoly_destroy(n4m_pp_modpoly_handle_t* handle);
n4m_status_t n4m_pp_modpoly_transform( const n4m_pp_modpoly_handle_t* handle, n4m_matrix_view_t X, n4m_matrix_view_t out);
```

Benchmark comparator backends are registered in the matrix and stored as reproducible snapshots when they define the canonical contract.

### Implementations

| Layer | Entry point | Language | Contract |
|-------|-------------|----------|----------|
| C ABI | `n4m_pp_modpoly` | C/C++ | Stable libn4m entry point family. |
| Python | `n4m.python.modpoly` | Python | ABI-close function backed by ctypes. |
| Python sklearn | `n4m.sklearn.ModPoly` | Python | scikit-learn-compatible estimator backed by ctypes. |
| R | `modpoly(X, polyorder = 2L, max_iter = 250L, tol = 1e-3)` | R | Public package wrapper around the C ABI. |
| ref.pybaselines | `pybaselines.modpoly` | Python | canonical/comparator |

### Usage

Every nirs4all-methods binding dispatches into the same C kernel. Registered comparator/source rows are listed in the benchmark card below.

::::{tab-set}
:class: nirs4all-methods-bindings


:::{tab-item} C ABI · libn4m
:sync: c
:class-label: lang-c

```c
n4m_status_t n4m_pp_modpoly_create(n4m_pp_modpoly_handle_t** out, int32_t polyorder, int32_t max_iter, double tol);
void n4m_pp_modpoly_destroy(n4m_pp_modpoly_handle_t* handle);
n4m_status_t n4m_pp_modpoly_transform( const n4m_pp_modpoly_handle_t* handle, n4m_matrix_view_t X, n4m_matrix_view_t out);
```

:::

:::{tab-item} Python ABI · n4m.python
:sync: python-abi
:class-label: lang-python

```python
from n4m import python as n4m

Xt = n4m.modpoly(X)
```

:::

:::{tab-item} Python sklearn · n4m.sklearn
:sync: python-sklearn
:class-label: lang-python

```python
from n4m.sklearn import ModPoly

op = ModPoly(polyorder=2, max_iter=250, tol=0.001)
Xt = op.fit_transform(X)
```

:::

:::{tab-item} R · nirs4all-methods
:sync: r
:class-label: lang-r

```r
library(n4m)
res <- modpoly(X, polyorder = 2L, max_iter = 50L)
```

:::

::::


**Benchmark Comparators And Sources** ◆

:::{card}
:class-card: external-refs

- ◆ **`ref.pybaselines`** (Python · canonical) — `pybaselines.modpoly` · pybaselines 1.2.1
:::

### Validation contract

- Operation: `cross_binding_callable` · comparator: `default_allclose` · tolerance: `rtol=1e-05`, `atol=1e-08` · quality: **strict**
- Default validation dataset: `100×50` · seed `20260556`
- Suites: smoke `3` cells; benchmark `11` cells · Default C/Python/reference parity comparator.
- Metrics: `max_abs_diff`, `rel_l2_diff`, `rms_diff`, `shape_equal`
- Truth sources: cross-binding references declared directly in `benchmarks/cross_binding/orchestrator.py`.

| Backend | Library | Gate | Comparator | Note |
|---------|---------|------|------------|------|
| `ref.pybaselines` | `pybaselines.modpoly` | Python / parity | `default_allclose` |  |

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
<tr class="bk-row"><td class="bk-name"><code>N4M.cpp</code></td><td class="parity parity-divergence parity-exact" title="worst reference max abs diff over visible sizes">7.5e-14</td><td class="ms ms-best">🏆 0.367 ms</td><td class="ms ms-best">🏆 3.588 ms</td><td class="ms">19.163 ms</td></tr>
</tbody>
<tbody class="lang-band lang-python"><tr class="lang-band-row" data-lang="python"><th colspan="5" scope="rowgroup"><span class="lang-band-dot"></span>Python · nirs4all-methods</th></tr>
<tr class="bk-row"><td class="bk-name"><code>N4M.python</code></td><td class="parity parity-divergence parity-exact" title="worst reference max abs diff over visible sizes">7.5e-14</td><td class="ms">0.392 ms</td><td class="ms">3.635 ms</td><td class="ms">19.025 ms</td></tr>
<tr class="bk-row"><td class="bk-name"><code>N4M.sklearn</code></td><td class="parity parity-divergence parity-exact" title="worst reference max abs diff over visible sizes">7.5e-14</td><td class="ms">0.377 ms</td><td class="ms">3.598 ms</td><td class="ms ms-best">🏆 18.899 ms</td></tr>
</tbody>
<tbody class="lang-band lang-r"><tr class="lang-band-row" data-lang="r"><th colspan="5" scope="rowgroup"><span class="lang-band-dot"></span>R · nirs4all-methods</th></tr>
<tr class="bk-row"><td class="bk-name"><code>N4M.R</code></td><td class="parity parity-divergence parity-exact" title="worst reference max abs diff over visible sizes">7.6e-14</td><td class="ms">0.434 ms</td><td class="ms">3.844 ms</td><td class="ms">20.250 ms</td></tr>
</tbody>
<tbody class="lang-band lang-python"><tr class="lang-band-row" data-lang="python"><th colspan="5" scope="rowgroup"><span class="lang-band-dot"></span>Python · external</th></tr>
<tr class="bk-row truth-source-strict"><td class="bk-name"><span class="truth-mark" title="Registry parity reference (Python): pybaselines.modpoly · pybaselines 1.2.1 — canonical">◆</span><code>ref.pybaselines</code></td><td class="parity parity-divergence parity-exact" title="worst reference max abs diff over visible sizes">0</td><td class="ms">10.377 ms</td><td class="ms">12.014 ms</td><td class="ms">20.224 ms</td></tr>
</tbody>
</table>
</div>

:::
::::

---

_See also_: [benchmark overview](../benchmarks/overview.md) · [methods index](index.md) · [interactive dashboard](../landing/dashboard.md)
