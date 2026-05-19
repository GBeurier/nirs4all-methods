# `asls` — AsLS

_Group_: **Baseline** · _Registry tolerance_: `rtol=1e-5`, `atol=1e-8` · _Source_: [`docs/algorithms/asls.md`](../algorithms/asls.md)

## Description

Eilers & Boelens 2005 asymmetric least squares baseline estimation. For each row `y` of length `n`:

1. Initialize weights `w = 1`.
2. Repeat up to `max_iter`:
   - Solve `(diag(w) + λ D²ᵀD²) z = w · y` (pentadiagonal LDLT).
   - Compute new weights: `new_w[i] = p` if `y[i] > z[i]` else `1 - p`.
   - If `||new_w - w||_2 / max(||w||_2, ε) < tol`, break.
   - `w ← new_w`.
3. Output: `out[i] = y[i] - z[i]` (baseline-subtracted spectrum).

From the `chemometrics4all.AsLS` Python wrapper docstring:

> Asymmetric Least Squares (Eilers & Boelens 2005).

### Parameters

| Name | Type | Default | Notes |
|------|------|---------|-------|
| `lam` | `float` | `1000000.0` | Smoothness penalty; larger values produce a smoother baseline. |
| `p` | `float` | `0.01` | Asymmetry parameter in penalized least-squares baseline correction. |
| `max_iter` | `int` | `50` | Maximum number of solver or reweighting iterations. |
| `tol` | `float` | `0.001` | Convergence tolerance. |

## Explanations

### Bibliographic source

- Eilers, P. H. C. & Boelens, H. F. M. (2005). "Baseline Correction with Asymmetric Least Squares Smoothing." Leiden University Medical Centre Report.
- Frozen Python reference: `parity/python_generator/src/c4a_parity_pybaselines_ref/asls.py` (historically validated against `pybaselines==1.1.4`); current upstream `pybaselines` drift is tracked by the reference parity gate.

### Mathematical principle

`asls` follows the standard baseline operator contract: an input matrix $\mathbf{X}\in\mathbb{R}^{n\times p}$ is transformed, scored, split, or filtered by the method-specific kernel while preserving the parity contract recorded by the benchmark snapshots. The precise numerical convention is the one exposed by the C ABI signatures and the registered external references below.

### Implementation

- Pentadiagonal symmetric LDLT solver in `core/common/banded_solver.{c,h}` (O(n) per iteration, vs pls4all's dense O(n²)).
- Per-row scratch buffers reused across all iterations to minimize allocator churn.
- Convergence uses `relative_difference(w, new_w)` matching pybaselines's `_safe_relative_difference` (L2 norms with `max(||w||, DBL_MIN)` denominator).
- On `max_iter` reached without convergence: silently returns the last iterate (matches pybaselines behaviour).
- Parity tolerance vs frozen NumPy reference: `1e-7 abs / 1e-8 rel`.

C ABI entry points used by the language bindings:

```c
c4a_status_t c4a_pp_asls_create(c4a_pp_asls_handle_t** out, double lam, double p, int32_t max_iter, double tol);
void c4a_pp_asls_destroy(c4a_pp_asls_handle_t* handle);
c4a_status_t c4a_pp_asls_transform(const c4a_pp_asls_handle_t* handle, c4a_matrix_view_t X, c4a_matrix_view_t out);
```

Reference backends are registered in the benchmark matrix and stored as reproducible snapshots when they define the canonical contract.

### Implementations

| Layer | Entry point | Language | Contract |
|-------|-------------|----------|----------|
| C ABI | `c4a_pp_asls` | C/C++ | Stable libc4a entry point family. |
| Python | `chemometrics4all.AsLS` | Python | sklearn-style wrapper backed by ctypes. |
| R | `asls(X, lam = 1e6, p = 0.01, max_iter = 50L, tol = 1e-3)` | R | Public package wrapper around the C ABI. |
| ref.pybaselines | `pybaselines.asls` | Python | canonical/comparator |

### Usage

Every chemometrics4all binding dispatches into the same C kernel. The registry references are listed in the parity card below.

::::{tab-set}
:class: chemometrics4all-bindings


:::{tab-item} C ABI · libc4a
:sync: c
:class-label: lang-c

```c
c4a_status_t c4a_pp_asls_create(c4a_pp_asls_handle_t** out, double lam, double p, int32_t max_iter, double tol);
void c4a_pp_asls_destroy(c4a_pp_asls_handle_t* handle);
c4a_status_t c4a_pp_asls_transform(const c4a_pp_asls_handle_t* handle, c4a_matrix_view_t X, c4a_matrix_view_t out);
```

:::

:::{tab-item} Python · chemometrics4all
:sync: python
:class-label: lang-python

```python
from chemometrics4all import AsLS

op = AsLS(lam=1000000.0, p=0.01, max_iter=50, tol=0.001)
Xt = op.fit_transform(X)
```

:::

:::{tab-item} R · chemometrics4all
:sync: r
:class-label: lang-r

```r
library(chemometrics4all)
res <- asls(X, lam = 1e5, p = 0.01, max_iter = 20L)
```

:::

::::


**Registry parity references** 📐

:::{card}
:class-card: external-refs

- 📐 **`ref.pybaselines`** (Python · canonical) — `pybaselines.asls` · pybaselines 1.2.1
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
<tr class="bk-row"><td class="bk-name"><code>C4A.cpp</code></td><td class="parity parity-exact">✓ exact</td><td class="ms ms-best">🏆 0.237 ms</td><td class="ms ms-best">🏆 2.986 ms</td><td class="ms ms-best">🏆 15.069 ms</td></tr>
</tbody>
<tbody class="lang-band lang-python"><tr class="lang-band-row" data-lang="python"><th colspan="5" scope="rowgroup"><span class="lang-band-dot"></span>Python · chemometrics4all</th></tr>
<tr class="bk-row"><td class="bk-name"><code>C4A.sklearn</code></td><td class="parity parity-exact">✓ bind</td><td class="ms">0.242 ms</td><td class="ms">3.039 ms</td><td class="ms">15.128 ms</td></tr>
</tbody>
<tbody class="lang-band lang-r"><tr class="lang-band-row" data-lang="r"><th colspan="5" scope="rowgroup"><span class="lang-band-dot"></span>R · chemometrics4all</th></tr>
<tr class="bk-row"><td class="bk-name"><code>C4A.R</code></td><td class="parity parity-exact">✓ bind</td><td class="ms">0.252 ms</td><td class="ms">3.219 ms</td><td class="ms">16.375 ms</td></tr>
</tbody>
<tbody class="lang-band lang-python"><tr class="lang-band-row" data-lang="python"><th colspan="5" scope="rowgroup"><span class="lang-band-dot"></span>Python · external</th></tr>
<tr class="bk-row truth-source-strict"><td class="bk-name"><span class="truth-mark" title="Registry parity reference (Python): pybaselines.asls · pybaselines 1.2.1 — canonical">📐</span><code>ref.pybaselines</code></td><td class="parity parity-exact">✓ ref</td><td class="ms">9.170 ms</td><td class="ms">21.905 ms</td><td class="ms">78.647 ms</td></tr>
</tbody>
</table>
</div>

:::
::::

---

_See also_: [benchmark overview](../benchmarks/overview.md) · [methods index](index.md) · [interactive dashboard](../landing/dashboard.md)
