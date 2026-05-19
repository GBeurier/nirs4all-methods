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

From the `chemometrics4all.IModPoly` Python wrapper docstring:

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
- Frozen Python reference: `parity/python_generator/src/c4a_parity_pybaselines_ref/imodpoly.py`.

### Mathematical principle

`imodpoly` follows the standard baseline operator contract: an input matrix $\mathbf{X}\in\mathbb{R}^{n\times p}$ is transformed, scored, split, or filtered by the method-specific kernel while preserving the parity contract recorded by the benchmark snapshots. The precise numerical convention is the one exposed by the C ABI signatures and the registered external references below.

### Implementation

- Same Householder-QR Vandermonde factorisation as ModPoly (shared from `core/common/linalg.h`).
- Residual stdev uses ddof = 0 (population); the convergence check protects the denominator with `max(devr_new, DBL_MIN)`.
- `mask_initial_peaks` (pybaselines' optional pre-threshold step) is disabled; this matches `num_std = 1` default behaviour on clean spectra and keeps the parity surface compact.
- On `max_iter` reached without convergence: silently returns the last iterate.
- Parity tolerance vs frozen NumPy reference: `1e-9 abs / 1e-10 rel`.

C ABI entry points used by the language bindings:

```c
c4a_status_t c4a_pp_imodpoly_create(c4a_pp_imodpoly_handle_t** out, int32_t polyorder, int32_t max_iter, double tol);
void c4a_pp_imodpoly_destroy(c4a_pp_imodpoly_handle_t* handle);
c4a_status_t c4a_pp_imodpoly_transform( const c4a_pp_imodpoly_handle_t* handle, c4a_matrix_view_t X, c4a_matrix_view_t out);
```

Reference backends are registered in the benchmark matrix and stored as reproducible snapshots when they define the canonical contract.

### Implementations

| Layer | Entry point | Language | Contract |
|-------|-------------|----------|----------|
| C ABI | `c4a_pp_imodpoly` | C/C++ | Stable libc4a entry point family. |
| Python | `chemometrics4all.IModPoly` | Python | sklearn-style wrapper backed by ctypes. |
| R | `imodpoly(X, polyorder = 2L, max_iter = 250L, tol = 1e-3)` | R | Public package wrapper around the C ABI. |
| ref.pybaselines | `pybaselines.imodpoly(mask_initial_peaks=False)` | Python | canonical/comparator |

### Usage

Every chemometrics4all binding dispatches into the same C kernel. The registry references are listed in the parity card below.

::::{tab-set}
:class: chemometrics4all-bindings


:::{tab-item} C ABI · libc4a
:sync: c
:class-label: lang-c

```c
c4a_status_t c4a_pp_imodpoly_create(c4a_pp_imodpoly_handle_t** out, int32_t polyorder, int32_t max_iter, double tol);
void c4a_pp_imodpoly_destroy(c4a_pp_imodpoly_handle_t* handle);
c4a_status_t c4a_pp_imodpoly_transform( const c4a_pp_imodpoly_handle_t* handle, c4a_matrix_view_t X, c4a_matrix_view_t out);
```

:::

:::{tab-item} Python · chemometrics4all
:sync: python
:class-label: lang-python

```python
from chemometrics4all import IModPoly

op = IModPoly(polyorder=2, max_iter=250, tol=0.001)
Xt = op.fit_transform(X)
```

:::

:::{tab-item} R · chemometrics4all
:sync: r
:class-label: lang-r

```r
library(chemometrics4all)
res <- imodpoly(X, polyorder = 2L, max_iter = 50L)
```

:::

::::


**Registry parity references** 📐

:::{card}
:class-card: external-refs

- 📐 **`ref.pybaselines`** (Python · canonical) — `pybaselines.imodpoly(mask_initial_peaks=False)` · pybaselines 1.2.1
:::

### Benchmarks
Median wall-clock per cell from [`docs/_static/bench-data.json`](../benchmarks/overview.md). Verdict legend: ✓ exact · ≈ context/drift · ✗ divergent · ⊘ not available · — not run · ⚠ error.
::::{tab-set}
:class: parity-tabs

:::{tab-item} 1 thread
:sync: threads-1

<div class="parity-table-wrap">
<table class="docutils parity-grouped">
<thead><tr><th>Backend</th><th>Parity</th><th>100×50</th><th>100×500</th><th>100×2500</th></tr></thead>
<tbody class="lang-band lang-cpp"><tr class="lang-band-row" data-lang="cpp"><th colspan="5" scope="rowgroup"><span class="lang-band-dot"></span>C++ native · libc4a</th></tr>
<tr class="bk-row"><td class="bk-name"><code>C4A.cpp</code></td><td class="parity parity-exact">✓ exact</td><td class="ms ms-best">🏆 0.532 ms</td><td class="ms">4.867 ms</td><td class="ms ms-best">🏆 25.205 ms</td></tr>
</tbody>
<tbody class="lang-band lang-python"><tr class="lang-band-row" data-lang="python"><th colspan="5" scope="rowgroup"><span class="lang-band-dot"></span>Python · chemometrics4all</th></tr>
<tr class="bk-row"><td class="bk-name"><code>C4A.sklearn</code></td><td class="parity parity-exact">✓ bind</td><td class="ms">0.540 ms</td><td class="ms ms-best">🏆 4.806 ms</td><td class="ms">25.831 ms</td></tr>
</tbody>
<tbody class="lang-band lang-r"><tr class="lang-band-row" data-lang="r"><th colspan="5" scope="rowgroup"><span class="lang-band-dot"></span>R · chemometrics4all</th></tr>
<tr class="bk-row"><td class="bk-name"><code>C4A.R</code></td><td class="parity parity-exact">✓ bind</td><td class="ms">0.594 ms</td><td class="ms">5.094 ms</td><td class="ms">26.500 ms</td></tr>
</tbody>
<tbody class="lang-band lang-python"><tr class="lang-band-row" data-lang="python"><th colspan="5" scope="rowgroup"><span class="lang-band-dot"></span>Python · external</th></tr>
<tr class="bk-row truth-source-strict"><td class="bk-name"><span class="truth-mark" title="Registry parity reference (Python): pybaselines.imodpoly(mask_initial_peaks=False) · pybaselines 1.2.1 — canonical">📐</span><code>ref.pybaselines</code></td><td class="parity parity-exact">✓ ref</td><td class="ms">27.198 ms</td><td class="ms">30.619 ms</td><td class="ms">56.555 ms</td></tr>
</tbody>
</table>
</div>

:::
::::

---

_See also_: [benchmark overview](../benchmarks/overview.md) · [methods index](index.md) · [interactive dashboard](../landing/dashboard.md)
