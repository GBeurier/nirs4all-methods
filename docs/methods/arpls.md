# `arpls` — ArPLS

_Group_: **Baseline** · _Registry tolerance_: `rtol=1e-5`, `atol=1e-8` · _Source_: [`docs/algorithms/arpls.md`](../algorithms/arpls.md)

## Description

Baek et al. 2015. Same skeleton as AsLS / AirPLS but with a logistic weight update centred on the negative-residual statistics.

For each row `y`:

1. Initialize `w = 1`.
2. Repeat up to `max_iter`:
   - Solve `(diag(w) + λ D²ᵀD²) z = w · y` (pentadiagonal LDLT).
   - `d = y - z`; `d_neg = d[d < 0]` (negative residuals).
   - `μ⁻ = mean(d_neg)`, `σ⁻ = std(d_neg, ddof=1)` (sample std; clamped to `DBL_MIN`).
   - `new_w[i] = 1 / (1 + exp(2 * (d[i] - (2σ⁻ - μ⁻)) / σ⁻))`  (numerically stable two-branch logistic).
   - If `||new_w - w||_2 / max(||w||_2, ε) < tol`: break.
   - `w ← new_w`.
3. Output: `out[i] = y[i] - z[i]`.

From the `chemometrics4all.ArPLS` Python wrapper docstring:

> Asymmetrically reweighted penalized least squares.

### Parameters

| Name | Type | Default | Notes |
|------|------|---------|-------|
| `lam` | `float` | `100000.0` | Smoothness penalty; larger values produce a smoother baseline. |
| `max_iter` | `int` | `50` | Maximum number of solver or reweighting iterations. |
| `tol` | `float` | `0.001` | Convergence tolerance. |

## Explanations

### Bibliographic source

- Baek, S.-J.; Park, A.; Ahn, Y.-J.; Choo, J. (2015). "Baseline Correction Using Asymmetrically Reweighted Penalized Least Squares Smoothing." *Analyst*, 140 (1), 250–257.
- Frozen Python reference: `parity/python_generator/src/c4a_parity_pybaselines_ref/arpls.py`.

### Mathematical principle

`arpls` follows the standard baseline operator contract: an input matrix $\mathbf{X}\in\mathbb{R}^{n\times p}$ is transformed, scored, split, or filtered by the method-specific kernel while preserving the parity contract recorded by the benchmark snapshots. The precise numerical convention is the one exposed by the C ABI signatures and the registered external references below.

### Implementation

- Pentadiagonal symmetric LDLT solver.
- Sample std uses `ddof=1` (matches `numpy.std(d_neg, ddof=1)`).
- Std clamped to `DBL_MIN` to avoid divide-by-zero on degenerate cases (all-zero residuals).
- Logistic stabilised via two-branch computation: for `arg >= 0` use `exp(-arg) / (1 + exp(-arg))`; for `arg < 0` use `1 / (1 + exp(arg))`. Avoids overflow for `|arg| >> 700`.
- Parity tolerance vs frozen NumPy reference: `1e-7 abs / 1e-8 rel`.

C ABI entry points used by the language bindings:

```c
c4a_status_t c4a_pp_arpls_create(c4a_pp_arpls_handle_t** out, double lam, int32_t max_iter, double tol);
void c4a_pp_arpls_destroy(c4a_pp_arpls_handle_t* handle);
c4a_status_t c4a_pp_arpls_transform(const c4a_pp_arpls_handle_t* handle, c4a_matrix_view_t X, c4a_matrix_view_t out);
```

Reference backends are registered in the benchmark matrix and stored as reproducible snapshots when they define the canonical contract.

### Implementations

| Layer | Entry point | Language | Contract |
|-------|-------------|----------|----------|
| C ABI | `c4a_pp_arpls` | C/C++ | Stable libc4a entry point family. |
| Python | `chemometrics4all.ArPLS` | Python | sklearn-style wrapper backed by ctypes. |
| R | `arpls(X, lam = 1e5, max_iter = 50L, tol = 1e-3)` | R | Public package wrapper around the C ABI. |
| ref.pybaselines | `pybaselines.arpls` | Python | canonical/comparator |

### Usage

Every chemometrics4all binding dispatches into the same C kernel. The registry references are listed in the parity card below.

::::{tab-set}
:class: chemometrics4all-bindings


:::{tab-item} C ABI · libc4a
:sync: c
:class-label: lang-c

```c
c4a_status_t c4a_pp_arpls_create(c4a_pp_arpls_handle_t** out, double lam, int32_t max_iter, double tol);
void c4a_pp_arpls_destroy(c4a_pp_arpls_handle_t* handle);
c4a_status_t c4a_pp_arpls_transform(const c4a_pp_arpls_handle_t* handle, c4a_matrix_view_t X, c4a_matrix_view_t out);
```

:::

:::{tab-item} Python · chemometrics4all
:sync: python
:class-label: lang-python

```python
from chemometrics4all import ArPLS

op = ArPLS(lam=100000.0, max_iter=50, tol=0.001)
Xt = op.fit_transform(X)
```

:::

:::{tab-item} R · chemometrics4all
:sync: r
:class-label: lang-r

```r
library(chemometrics4all)
res <- arpls(X, lam = 1e5, max_iter = 20L)
```

:::

::::


**Registry parity references** 📐

:::{card}
:class-card: external-refs

- 📐 **`ref.pybaselines`** (Python · canonical) — `pybaselines.arpls` · pybaselines 1.2.1
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
<tr class="bk-row"><td class="bk-name"><code>C4A.cpp</code></td><td class="parity parity-exact">✓ exact</td><td class="ms">1.494 ms</td><td class="ms">8.628 ms</td><td class="ms">30.171 ms</td></tr>
</tbody>
<tbody class="lang-band lang-python"><tr class="lang-band-row" data-lang="python"><th colspan="5" scope="rowgroup"><span class="lang-band-dot"></span>Python · chemometrics4all</th></tr>
<tr class="bk-row"><td class="bk-name"><code>C4A.sklearn</code></td><td class="parity parity-exact">✓ bind</td><td class="ms">1.559 ms</td><td class="ms ms-best">🏆 8.585 ms</td><td class="ms ms-best">🏆 29.220 ms</td></tr>
</tbody>
<tbody class="lang-band lang-r"><tr class="lang-band-row" data-lang="r"><th colspan="5" scope="rowgroup"><span class="lang-band-dot"></span>R · chemometrics4all</th></tr>
<tr class="bk-row"><td class="bk-name"><code>C4A.R</code></td><td class="parity parity-exact">✓ bind</td><td class="ms ms-best">🏆 1.469 ms</td><td class="ms">8.625 ms</td><td class="ms">29.250 ms</td></tr>
</tbody>
<tbody class="lang-band lang-python"><tr class="lang-band-row" data-lang="python"><th colspan="5" scope="rowgroup"><span class="lang-band-dot"></span>Python · external</th></tr>
<tr class="bk-row truth-source-strict"><td class="bk-name"><span class="truth-mark" title="Registry parity reference (Python): pybaselines.arpls · pybaselines 1.2.1 — canonical">📐</span><code>ref.pybaselines</code></td><td class="parity parity-exact">✓ ref</td><td class="ms">47.134 ms</td><td class="ms">45.326 ms</td><td class="ms">74.300 ms</td></tr>
</tbody>
</table>
</div>

:::
::::

---

_See also_: [benchmark overview](../benchmarks/overview.md) · [methods index](index.md) · [interactive dashboard](../landing/dashboard.md)
