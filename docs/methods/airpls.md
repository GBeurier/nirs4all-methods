# `airpls` — AirPLS

_Group_: **Baseline** · _Registry tolerance_: `rtol=1e-5`, `atol=1e-8` · _Source_: [`docs/algorithms/airpls.md`](../algorithms/airpls.md)

## Description

Zhang 2010 baseline correction. Same skeleton as AsLS but with an exponential weight update that concentrates on negative residuals.

For each row `y`:

1. Initialize `w = 1`, `y_l1 = sum(|y|)`.
2. Repeat up to `max_iter` (iteration index `i`, starting at 1):
   - Solve `(diag(w) + λ D²ᵀD²) z = w · y` (pentadiagonal LDLT).
   - `d = y - z`; `d_neg = -min(d, 0)` (magnitude of negative residuals).
   - `sum_neg = sum(d_neg)`.
   - If `y_l1 > 0` and `sum_neg / y_l1 < tol`: break.
   - If `count(d_neg > 0) < 2`: break (no negative residuals to fit).
   - `t = min(i, 50)` (clip exponent multiplier).
   - `new_w[i] = exp(clip(t * d_neg / sum_neg, [0, log(DBL_MAX) - spacing]))` on negative-residual indices, `0` elsewhere.
   - `w ← new_w`.
3. Output: `out[i] = y[i] - z[i]`.

From the `chemometrics4all.AirPLS` Python wrapper docstring:

> Adaptive iteratively reweighted PLS (Zhang 2010).

### Parameters

| Name | Type | Default | Notes |
|------|------|---------|-------|
| `lam` | `float` | `1000000.0` | Smoothness penalty; larger values produce a smoother baseline. |
| `max_iter` | `int` | `50` | Maximum number of solver or reweighting iterations. |
| `tol` | `float` | `0.001` | Convergence tolerance. |

## Explanations

### Bibliographic source

- Zhang, Z.-M.; Chen, S.; Liang, Y.-Z. (2010). "Baseline Correction Using Adaptive Iteratively Reweighted Penalized Least Squares." *Analyst*, 135 (5), 1138–1146.
- Frozen Python reference: `parity/python_generator/src/c4a_parity_pybaselines_ref/airpls.py`.

### Mathematical principle

`airpls` follows the standard baseline operator contract: an input matrix $\mathbf{X}\in\mathbb{R}^{n\times p}$ is transformed, scored, split, or filtered by the method-specific kernel while preserving the parity contract recorded by the benchmark snapshots. The precise numerical convention is the one exposed by the C ABI signatures and the registered external references below.

### Implementation

- Pentadiagonal symmetric LDLT solver.
- Exp argument clipped to `[0, log(DBL_MAX) - spacing(log(DBL_MAX))]` to avoid `+inf` from large iterates (matches pybaselines `_airpls_loop` overflow guard).
- Convergence metric `sum(d_neg) / ||y||_1` evaluated BEFORE weight update (matches pybaselines).
- Parity tolerance vs frozen NumPy reference: `1e-7 abs / 1e-8 rel` for typical regularisation; `1e-6 abs / 1e-7 rel` for stiff regularisation (`lam >= 1e7`) where LDLT-vs-spsolve ULP differences compound across iterations.

C ABI entry points used by the language bindings:

```c
c4a_status_t c4a_pp_airpls_create(c4a_pp_airpls_handle_t** out, double lam, int32_t max_iter, double tol);
void c4a_pp_airpls_destroy(c4a_pp_airpls_handle_t* handle);
c4a_status_t c4a_pp_airpls_transform( const c4a_pp_airpls_handle_t* handle, c4a_matrix_view_t X, c4a_matrix_view_t out);
```

Reference backends are registered in the benchmark matrix and stored as reproducible snapshots when they define the canonical contract.

### Implementations

| Layer | Entry point | Language | Contract |
|-------|-------------|----------|----------|
| C ABI | `c4a_pp_airpls` | C/C++ | Stable libc4a entry point family. |
| Python | `chemometrics4all.AirPLS` | Python | sklearn-style wrapper backed by ctypes. |
| R | `airpls(X, lam = 1e6, max_iter = 50L, tol = 1e-3)` | R | Public package wrapper around the C ABI. |
| ref.pybaselines | `pybaselines.airpls` | Python | canonical/comparator |

### Usage

Every chemometrics4all binding dispatches into the same C kernel. The registry references are listed in the parity card below.

::::{tab-set}
:class: chemometrics4all-bindings


:::{tab-item} C ABI · libc4a
:sync: c
:class-label: lang-c

```c
c4a_status_t c4a_pp_airpls_create(c4a_pp_airpls_handle_t** out, double lam, int32_t max_iter, double tol);
void c4a_pp_airpls_destroy(c4a_pp_airpls_handle_t* handle);
c4a_status_t c4a_pp_airpls_transform( const c4a_pp_airpls_handle_t* handle, c4a_matrix_view_t X, c4a_matrix_view_t out);
```

:::

:::{tab-item} Python · chemometrics4all
:sync: python
:class-label: lang-python

```python
from chemometrics4all import AirPLS

op = AirPLS(lam=1000000.0, max_iter=50, tol=0.001)
Xt = op.fit_transform(X)
```

:::

:::{tab-item} R · chemometrics4all
:sync: r
:class-label: lang-r

```r
library(chemometrics4all)
res <- airpls(X, lam = 1e5, max_iter = 20L)
```

:::

::::


**Registry parity references** ◆

:::{card}
:class-card: external-refs

- ◆ **`ref.pybaselines`** (Python · canonical) — `pybaselines.airpls` · pybaselines 1.2.1
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
<tr class="bk-row"><td class="bk-name"><code>C4A.cpp</code></td><td class="parity parity-exact">✓ exact</td><td class="ms ms-best">🏆 0.253 ms</td><td class="ms ms-best">🏆 2.708 ms</td><td class="ms">13.565 ms</td></tr>
</tbody>
<tbody class="lang-band lang-python"><tr class="lang-band-row" data-lang="python"><th colspan="5" scope="rowgroup"><span class="lang-band-dot"></span>Python · chemometrics4all</th></tr>
<tr class="bk-row"><td class="bk-name"><code>C4A.sklearn</code></td><td class="parity parity-exact">✓ bind</td><td class="ms">0.277 ms</td><td class="ms">2.781 ms</td><td class="ms ms-best">🏆 13.543 ms</td></tr>
</tbody>
<tbody class="lang-band lang-r"><tr class="lang-band-row" data-lang="r"><th colspan="5" scope="rowgroup"><span class="lang-band-dot"></span>R · chemometrics4all</th></tr>
<tr class="bk-row"><td class="bk-name"><code>C4A.R</code></td><td class="parity parity-exact">✓ bind</td><td class="ms">0.270 ms</td><td class="ms">2.844 ms</td><td class="ms">14.375 ms</td></tr>
</tbody>
<tbody class="lang-band lang-python"><tr class="lang-band-row" data-lang="python"><th colspan="5" scope="rowgroup"><span class="lang-band-dot"></span>Python · external</th></tr>
<tr class="bk-row truth-source-strict"><td class="bk-name"><span class="truth-mark" title="Registry parity reference (Python): pybaselines.airpls · pybaselines 1.2.1 — canonical">◆</span><code>ref.pybaselines</code></td><td class="parity parity-exact">✓ ref</td><td class="ms">8.884 ms</td><td class="ms">14.428 ms</td><td class="ms">39.311 ms</td></tr>
</tbody>
</table>
</div>

:::
::::

---

_See also_: [benchmark overview](../benchmarks/overview.md) · [methods index](index.md) · [interactive dashboard](../landing/dashboard.md)
