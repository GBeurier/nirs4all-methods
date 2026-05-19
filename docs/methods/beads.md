# `beads` — BEADS

_Group_: **Baseline** · _Registry tolerance_: `rtol=1e-5`, `atol=1e-8` · _Source_: [`docs/algorithms/beads.md`](../algorithms/beads.md)

## Description

Ning & Selesnick 2014 BEADS. **Phase 5b ships the simplified pentadiagonal variant**; see "Simplification" below for what is deferred.

For each row `y` of length `n`:

```
w := ones(n)
z := zeros(n)
for iter in 0..max_iter:
    A := diag(lam_0 * w) + (lam_1 + lam_2) * D_2^T D_2
    rhs := lam_0 * w * y
    z_new := A^{-1} rhs                              # pentadiagonal LDLT
    d := y - z_new
    w[i] := 1 / (|d[i]| + eps)                       # reweighted L2 sparsity
    w := w * n / sum(w)                              # renormalise to mean 1
    if rel_l2_diff(z, z_new) < tol: break
    z := z_new
out := y - z
```

`eps = 1e-3` prevents division by zero on perfectly matched residuals.

From the `chemometrics4all.BEADS` Python wrapper docstring:

> Baseline estimation and denoising with sparsity.

### Parameters

| Name | Type | Default | Notes |
|------|------|---------|-------|
| `lam_0` | `float` | `100.0` |  |
| `lam_1` | `float` | `0.5` |  |
| `lam_2` | `float` | `0.5` |  |
| `max_iter` | `int` | `50` | Maximum number of solver or reweighting iterations. |
| `tol` | `float` | `0.001` | Convergence tolerance. |

## Explanations

### Bibliographic source

- Ning, X., Selesnick, I. W. & Duval, L. (2014). "Chromatogram baseline estimation and denoising using sparsity (BEADS)." Chemometrics and Intelligent Laboratory Systems 139, 156-167.
- Frozen Python reference (simplified variant): `parity/python_generator/src/c4a_parity_pybaselines_ref/beads.py`.

### Mathematical principle

`beads` follows the standard baseline operator contract: an input matrix $\mathbf{X}\in\mathbb{R}^{n\times p}$ is transformed, scored, split, or filtered by the method-specific kernel while preserving the parity contract recorded by the benchmark snapshots. The precise numerical convention is the one exposed by the C ABI signatures and the registered external references below.

### Implementation

- Pentadiagonal LDLT (`core/common/banded_solver.h`) with row-scope `L` / `D` scratch buffers.
- Penalty: `(lam_1 + lam_2) * D_2^T D_2` (the simplification — see below).
- Convergence on `rel_l2_diff(z_old, z_new)`. Renormalising `w` to mean 1 keeps the effective lam_0 stable across iterations.
- On `max_iter` reached without convergence: silently returns the last iterate.
- Parity tolerance vs frozen NumPy reference: `1e-6 abs / 1e-7 rel`.

C ABI entry points used by the language bindings:

```c
c4a_status_t c4a_pp_beads_create(c4a_pp_beads_handle_t** out, double lam_0, double lam_1, double lam_2, int32_t max_iter, double tol);
void c4a_pp_beads_destroy(c4a_pp_beads_handle_t* handle);
c4a_status_t c4a_pp_beads_transform(const c4a_pp_beads_handle_t* handle, c4a_matrix_view_t X, c4a_matrix_view_t out);
```

Reference backends are registered in the benchmark matrix and stored as reproducible snapshots when they define the canonical contract.

### Implementations

| Layer | Entry point | Language | Contract |
|-------|-------------|----------|----------|
| C ABI | `c4a_pp_beads` | C/C++ | Stable libc4a entry point family. |
| Python | `chemometrics4all.BEADS` | Python | sklearn-style wrapper backed by ctypes. |
| R | `beads(X, lam_0 = 100.0, lam_1 = 0.5, lam_2 = 0.5, max_iter = 50L, tol = 1e-3)` | R | Public package wrapper around the C ABI. |
| ref.frozen | `c4a frozen BEADS(simplified)` | Python | canonical/comparator |
| ref.pybaselines | `pybaselines.beads(full)` | Python | context only; pybaselines.beads is the full BEADS algorithm; c4a currently ships the documented simplified reweighted-L2 variant |

### Usage

Every chemometrics4all binding dispatches into the same C kernel. The registry references are listed in the parity card below.

::::{tab-set}
:class: chemometrics4all-bindings


:::{tab-item} C ABI · libc4a
:sync: c
:class-label: lang-c

```c
c4a_status_t c4a_pp_beads_create(c4a_pp_beads_handle_t** out, double lam_0, double lam_1, double lam_2, int32_t max_iter, double tol);
void c4a_pp_beads_destroy(c4a_pp_beads_handle_t* handle);
c4a_status_t c4a_pp_beads_transform(const c4a_pp_beads_handle_t* handle, c4a_matrix_view_t X, c4a_matrix_view_t out);
```

:::

:::{tab-item} Python · chemometrics4all
:sync: python
:class-label: lang-python

```python
from chemometrics4all import BEADS

op = BEADS(lam_0=100.0, lam_1=0.5, lam_2=0.5, max_iter=50, tol=0.001)
Xt = op.fit_transform(X)
```

:::

:::{tab-item} R · chemometrics4all
:sync: r
:class-label: lang-r

```r
library(chemometrics4all)
res <- beads(X, max_iter = 20L)
```

:::

::::


**Registry parity references** 📐

:::{card}
:class-card: external-refs

- 📐 **`ref.frozen`** (Python · canonical) — `c4a frozen BEADS(simplified)` · chemometrics4all frozen reference
- ℹ **`ref.pybaselines`** (Python · context) — `pybaselines.beads(full)` · pybaselines 1.2.1 — pybaselines.beads is the full BEADS algorithm; c4a currently ships the documented simplified reweighted-L2 variant
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
<tr class="bk-row"><td class="bk-name"><code>C++</code></td><td class="parity parity-exact">✓ exact</td><td class="ms ms-best">🏆 0.118 ms</td><td class="ms ms-best">🏆 1.277 ms</td><td class="ms ms-best">🏆 6.819 ms</td></tr>
</tbody>
<tbody class="lang-band lang-python"><tr class="lang-band-row" data-lang="python"><th colspan="5" scope="rowgroup"><span class="lang-band-dot"></span>Python · chemometrics4all</th></tr>
<tr class="bk-row"><td class="bk-name"><code>Py</code></td><td class="parity parity-exact">✓ bind</td><td class="ms">0.124 ms</td><td class="ms">1.312 ms</td><td class="ms">6.897 ms</td></tr>
</tbody>
<tbody class="lang-band lang-r"><tr class="lang-band-row" data-lang="r"><th colspan="5" scope="rowgroup"><span class="lang-band-dot"></span>R · chemometrics4all</th></tr>
<tr class="bk-row"><td class="bk-name"><code>R</code></td><td class="parity parity-exact">✓ bind</td><td class="ms">0.146 ms</td><td class="ms">1.516 ms</td><td class="ms">7.875 ms</td></tr>
</tbody>
<tbody class="lang-band lang-python"><tr class="lang-band-row" data-lang="python"><th colspan="5" scope="rowgroup"><span class="lang-band-dot"></span>Python · external</th></tr>
<tr class="bk-row truth-source-strict"><td class="bk-name"><span class="truth-mark" title="Registry parity reference (Python): c4a frozen BEADS(simplified) · chemometrics4all frozen reference — canonical">📐</span><code>ref.frozen</code></td><td class="parity parity-exact">✓ ref</td><td class="ms">30.549 ms</td><td class="ms">51.746 ms</td><td class="ms">215.855 ms</td></tr>
<tr class="bk-row truth-source-relaxed"><td class="bk-name"><span class="truth-mark" title="Registry parity reference (Python): pybaselines.beads(full) · pybaselines 1.2.1 — context">📐</span><code>ref.pybaselines</code></td><td class="parity parity-context">✓ ref</td><td class="ms">249.644 ms</td><td class="ms">360.442 ms</td><td class="ms">1041.274 ms</td></tr>
</tbody>
</table>
</div>

:::
::::

---

_See also_: [benchmark overview](../benchmarks/overview.md) · [methods index](index.md) · [interactive dashboard](../landing/dashboard.md)
