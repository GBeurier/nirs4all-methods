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

From the `chemometrics4all.ModPoly` Python wrapper docstring:

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
- Frozen Python reference: `parity/python_generator/src/c4a_parity_pybaselines_ref/modpoly.py`.

### Mathematical principle

`modpoly` follows the standard baseline operator contract: an input matrix $\mathbf{X}\in\mathbb{R}^{n\times p}$ is transformed, scored, split, or filtered by the method-specific kernel while preserving the parity contract recorded by the benchmark snapshots. The precise numerical convention is the one exposed by the C ABI signatures and the registered external references below.

### Implementation

- Per-row Householder QR factorisation of the `(n, polyorder + 1)` Vandermonde matrix in descending powers (shared with `Detrend` / `IModPoly` / `IAsLS` via `core/common/linalg.{c,h}`). Factor once per row; re-apply `Q^T` against the clipped y at each iteration.
- Convergence uses `relative_difference(z_old, z_new)` from `banded_solver.h` (L2 norms, `max(||z_old||, DBL_MIN)` denominator).
- On `max_iter` reached without convergence: silently returns the last iterate.
- Parity tolerance vs frozen NumPy reference: `1e-9 abs / 1e-10 rel`.

C ABI entry points used by the language bindings:

```c
c4a_status_t c4a_pp_modpoly_create(c4a_pp_modpoly_handle_t** out, int32_t polyorder, int32_t max_iter, double tol);
void c4a_pp_modpoly_destroy(c4a_pp_modpoly_handle_t* handle);
c4a_status_t c4a_pp_modpoly_transform( const c4a_pp_modpoly_handle_t* handle, c4a_matrix_view_t X, c4a_matrix_view_t out);
```

Reference backends are registered in the benchmark matrix and stored as reproducible snapshots when they define the canonical contract.

### Implementations

| Layer | Entry point | Language | Contract |
|-------|-------------|----------|----------|
| C ABI | `c4a_pp_modpoly` | C/C++ | Stable libc4a entry point family. |
| Python | `chemometrics4all.ModPoly` | Python | sklearn-style wrapper backed by ctypes. |
| R | `modpoly(X, polyorder = 2L, max_iter = 250L, tol = 1e-3)` | R | Public package wrapper around the C ABI. |
| ref.pybaselines | `pybaselines.modpoly` | Python | canonical/comparator |

### Usage

Every chemometrics4all binding dispatches into the same C kernel. The registry references are listed in the parity card below.

::::{tab-set}
:class: chemometrics4all-bindings


:::{tab-item} C ABI · libc4a
:sync: c
:class-label: lang-c

```c
c4a_status_t c4a_pp_modpoly_create(c4a_pp_modpoly_handle_t** out, int32_t polyorder, int32_t max_iter, double tol);
void c4a_pp_modpoly_destroy(c4a_pp_modpoly_handle_t* handle);
c4a_status_t c4a_pp_modpoly_transform( const c4a_pp_modpoly_handle_t* handle, c4a_matrix_view_t X, c4a_matrix_view_t out);
```

:::

:::{tab-item} Python · chemometrics4all
:sync: python
:class-label: lang-python

```python
from chemometrics4all import ModPoly

op = ModPoly(polyorder=2, max_iter=250, tol=0.001)
Xt = op.fit_transform(X)
```

:::

:::{tab-item} R · chemometrics4all
:sync: r
:class-label: lang-r

```r
library(chemometrics4all)
res <- modpoly(X, polyorder = 2L, max_iter = 50L)
```

:::

::::


**Registry parity references** ◆

:::{card}
:class-card: external-refs

- ◆ **`ref.pybaselines`** (Python · canonical) — `pybaselines.modpoly` · pybaselines 1.2.1
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
<tr class="bk-row"><td class="bk-name"><code>C4A.cpp</code></td><td class="parity parity-exact">✓ exact</td><td class="ms ms-best">🏆 0.379 ms</td><td class="ms ms-best">🏆 3.459 ms</td><td class="ms">18.787 ms</td></tr>
</tbody>
<tbody class="lang-band lang-python"><tr class="lang-band-row" data-lang="python"><th colspan="5" scope="rowgroup"><span class="lang-band-dot"></span>Python · chemometrics4all</th></tr>
<tr class="bk-row"><td class="bk-name"><code>C4A.sklearn</code></td><td class="parity parity-exact">✓ bind</td><td class="ms">0.393 ms</td><td class="ms">3.671 ms</td><td class="ms ms-best">🏆 18.674 ms</td></tr>
</tbody>
<tbody class="lang-band lang-r"><tr class="lang-band-row" data-lang="r"><th colspan="5" scope="rowgroup"><span class="lang-band-dot"></span>R · chemometrics4all</th></tr>
<tr class="bk-row"><td class="bk-name"><code>C4A.R</code></td><td class="parity parity-exact">✓ bind</td><td class="ms">0.402 ms</td><td class="ms">3.719 ms</td><td class="ms">20.125 ms</td></tr>
</tbody>
<tbody class="lang-band lang-python"><tr class="lang-band-row" data-lang="python"><th colspan="5" scope="rowgroup"><span class="lang-band-dot"></span>Python · external</th></tr>
<tr class="bk-row truth-source-strict"><td class="bk-name"><span class="truth-mark" title="Registry parity reference (Python): pybaselines.modpoly · pybaselines 1.2.1 — canonical">◆</span><code>ref.pybaselines</code></td><td class="parity parity-exact">✓ ref</td><td class="ms">9.349 ms</td><td class="ms">11.927 ms</td><td class="ms">20.557 ms</td></tr>
</tbody>
</table>
</div>

:::
::::

---

_See also_: [benchmark overview](../benchmarks/overview.md) · [methods index](index.md) · [interactive dashboard](../landing/dashboard.md)
