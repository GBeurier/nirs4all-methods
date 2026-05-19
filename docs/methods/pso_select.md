# `pso_select` — PSO-PLS — Particle Swarm Optimisation

_Group_: **Variable selector** · _Registry tolerance_: `1.4`

## Description

PSO-PLS — Binary Particle Swarm variable selection (§48)

From the `pls4all.sklearn.PSOSelector` docstring:

> Binary Particle Swarm Optimization selector.

> **Registry note** — Python `pyswarms 1.3.0` Binary PSO with PLS-CV-RMSE fitness. RNG diverges from pls4all splitmix64; parity is on algorithm family, not bit-exact selection.

### Parameters

| Name | Type | Default | Notes |
|------|------|---------|-------|
| `n_components` | `int` | `2` | Number of latent components extracted (k). |
| `n_swarm` | `int` | `30` | Number of particles in the binary PSO swarm. |
| `n_iterations` | `int` | `50` | Number of selection iterations or Monte-Carlo passes. |
| `w` | `float` | `0.729` | PSO inertia weight on the previous-velocity term. |
| `c1` | `float` | `1.494` | PSO cognitive (personal-best) acceleration coefficient. |
| `c2` | `float` | `1.494` | PSO social (global-best) acceleration coefficient. |
| `v_max` | `float` | `4.0` | Velocity clipping bound for binary PSO. |
| `n_folds` | `int` | `3` | Number of cross-validation folds used inside the selector. |
| `seed` | `int` | `0` | Random seed for reproducible sampling/initialization. |

## Explanations

### Bibliographic source

Kennedy, J. & Eberhart, R. (1995). *Particle swarm optimization*. IEEE ICNN 1995, vol. 4, 1942–1948. — binary PSO variant used for variable selection.

### Mathematical principle

Binary PSO maintains a swarm of particles where each particle's position is a $p$-bit feature mask. Velocity updates blend the particle's personal best, the swarm's global best, and an inertia term; positions are stochastically rounded to 0/1 via a sigmoid.

Compared to GA-PLS, PSO converges faster on smooth fitness landscapes but is more susceptible to premature convergence on multi-modal ones. The two methods are complementary: PSO for quick reconnaissance, GA for final polish.

Recommended swarm size: 20–50. The fitness is again PLS CV-RMSE on the masked subset.

### Implementation

`p4a_pso_select`. Reference: Python `pyswarms` for the PSO core, wrapped against PLS CV-RMSE.

R roxygen note (`methods_extra.R::pso_select`):

> PSO-PLS (Binary Particle Swarm Optimization).
> @param n_components Integer. Number of latent components.
> @param n_swarm Method-specific parameter. See the underlying `*_fit()` function for the exact semantics.
> @param n_iterations Integer >= 1. Number of outer-loop iterations.
> @param w Method-specific parameter. See the underlying `*_fit()` function for the exact semantics.
> @param c1 Method-specific parameter. See the underlying `*_fit()` function for the exact semantics.
> @param c2 Method-specific parameter. See the underlying `*_fit()` function for the exact semantics.
> @param v_max Method-specific parameter. See the underlying `*_fit()` function for the exact semantics.
> @param seed Integer. Random seed for reproducibility.
> @param X Numeric matrix of predictors (rows = samples, cols = features).
> @param Y Numeric matrix or vector of responses, with one row per sample.
> @export

### Usage

All four pls4all bindings dispatch into the same C kernel; the external libraries on the right are the parity references registered in `benchmarks.parity_timing.registry`. Switch tabs to read the same fit in your language.

**pls4all bindings**

::::{tab-set}
:class: pls4all-bindings

:::{tab-item} C ABI · libp4a
:sync: c
:class-label: lang-c

```c
/* C ABI — libp4a */
p4a_context_t* ctx = p4a_context_create();
p4a_config_t*  cfg = p4a_config_create();
p4a_method_result_t* res = NULL;
p4a_pso_select(ctx, cfg, &x_view, &y_view, /* hyperparams */, &res);
/* … read coefficients / mask / scores via */
/* p4a_method_result_get_double_matrix / vector / scalar … */
p4a_method_result_destroy(res);
p4a_config_destroy(cfg);
p4a_context_destroy(ctx);
```

:::

:::{tab-item} Python · pls4all (raw)
:sync: python-raw
:class-label: lang-python

```python
import pls4all
from pls4all._methods import pso_select
with pls4all.Context() as ctx, pls4all.Config() as cfg:
    res = pso_select(ctx, cfg, X, y, n_components=3)
# then: res.matrix("predictions"), res.matrix("coefficients"),
# res.vector("mask"), res.scalar("intercept"), …
```

:::

:::{tab-item} Python · pls4all.sklearn
:sync: python-sklearn
:class-label: lang-python

```python
from pls4all.sklearn import PSOSelector
mdl = PSOSelector(n_components=2, n_swarm=30, n_iterations=50, w=0.729, c1=1.494, c2=1.494, v_max=4.0, n_folds=3, seed=0)
mdl.fit(X, y)
y_hat = mdl.predict(X_test)
```

:::

:::{tab-item} R · pls4all_method()
:sync: r-dispatcher
:class-label: lang-r

```r
library(pls4all)
# Unified low-level dispatcher (May 2026 R cleanup):
res <- pls4all_method("pso_select", X, y,
                      n_components = 3L, params = list(n_swarm = 10L, n_iterations = 12L, w = 0.729, c1 = 1.494, c2 = 1.494, v_max = 4.0, seed = 42L))
# res is a named list with MethodResult arrays/scalars.
# selected_indices / top_k_intervals are 1-based.
```

:::

:::{tab-item} R · pls4all (raw fn)
:sync: r-raw
:class-label: lang-r

```r
library(pls4all)
res  <- pso_select(X, Y, n_components,
            n_swarm = 30L, n_iterations = 50L,
            w = 0.729, c1 = 1.494, c2 = 1.494, v_max = 4.0,
            seed = 0L)
yhat <- pls4all_predict(res, X_test)
```

:::

:::{tab-item} MATLAB · pls4all (MEX)
:sync: matlab-mex
:class-label: lang-matlab

```matlab
res  = pls4all.fit("pso_select", X, y, "NumComponents", 3);
yhat = predict(res, Xtest);
```

:::

:::{tab-item} MATLAB · pls4all (classdef)
:sync: matlab-classdef
:class-label: lang-matlab

_No idiomatic classdef wrapper — invoke `pls4all.fit("pso_select", X, y, …)` directly from the unified MEX factory._

:::

::::


**Registry parity references** 📐

:::{card}
:class-card: external-refs

- 📐 **`ref.python_pyswarms`** (python · python) — `pyswarms` 1.3.0 · qualitative (rmse_rel ≤ 1e+00) — pyswarms Binary PSO with PLS-CV fitness. RNG diverges from pls4all's splitmix64; parity is on algorithm family, not bit-exact masks. Mask RMSE-rel ~0 = perfect, ~1 = half disagree.
:::

### Benchmarks

Median wall-clock per cell from [`benchmarks/cross_binding/results/full_matrix.csv`](../benchmarks/overview.md). Verdict legend: ✓ exact · ≈ drift · ✗ divergent · ⊘ not available in lib · — not run · ⚠ error. The fastest backend per column is marked with a 🏆 medal. Rows tagged with **📐** are *also* declared in [`benchmarks/parity_timing/registry.py`](../benchmarks/methodology.md) as the canonical parity references for this method (`python_reference` / `r_reference` / `extra_references`). C++ and external rows show reference parity; pls4all language bindings show binding parity against the C++ backend. The 📐 icon points at the *library-of-record* the parity gate ultimately answers to. Hover the icon to see the role and tolerance band.

::::{tab-set}
:class: parity-tabs

:::{tab-item} 1 thread
:sync: threads-1

<div class="parity-table-wrap">
<table class="docutils parity-grouped">
<thead><tr><th scope="col">Backend</th><th scope="col">Parity</th><th class="size-col" scope="col">100×50 (ms)</th></tr></thead>
<tbody class="lang-band lang-cpp"><tr class="lang-band-row" data-lang="cpp"><th colspan="3" scope="rowgroup"><span class="lang-band-dot"></span>C++ native · libp4a</th></tr>
<tr class="bk-row"><td class="bk-name"><code>pls4all.cpp.blas</code></td><td class="parity parity-exact">✓ 1e+00</td><td class="ms ms-best">4.62 ms<span class="medal" title="fastest">🏆</span></td></tr>
<tr class="bk-row"><td class="bk-name"><code>pls4all.cpp.blas+omp</code></td><td class="parity parity-exact">✓ 1e+00</td><td class="ms">4.87 ms</td></tr>
<tr class="bk-row"><td class="bk-name"><code>pls4all.cpp.omp</code></td><td class="parity parity-exact">✓ 1e+00</td><td class="ms">5.87 ms</td></tr>
<tr class="bk-row"><td class="bk-name"><code>pls4all.cpp.ref</code></td><td class="parity parity-exact">✓ 1e+00</td><td class="ms">5.80 ms</td></tr>
</tbody>
<tbody class="lang-band lang-python"><tr class="lang-band-row" data-lang="python"><th colspan="3" scope="rowgroup"><span class="lang-band-dot"></span>Python · pls4all</th></tr>
<tr class="bk-row"><td class="bk-name"><code>pls4all.python</code></td><td class="parity parity-exact">✓ bind</td><td class="ms">4.82 ms</td></tr>
<tr class="bk-row"><td class="bk-name"><code>pls4all.registry</code></td><td class="parity parity-exact">✓ bind</td><td class="ms">4.93 ms</td></tr>
<tr class="bk-row"><td class="bk-name"><code>pls4all.sklearn</code></td><td class="parity parity-exact">✓ bind</td><td class="ms">5.14 ms</td></tr>
</tbody>
<tbody class="lang-band lang-r"><tr class="lang-band-row" data-lang="r"><th colspan="3" scope="rowgroup"><span class="lang-band-dot"></span>R · pls4all</th></tr>
<tr class="bk-row"><td class="bk-name"><code>pls4all.R</code></td><td class="parity parity-exact">✓ bind</td><td class="ms">7.00 ms</td></tr>
<tr class="bk-row"><td class="bk-name"><code>pls4all.R.formula</code></td><td class="parity parity-exact">✓ bind</td><td class="ms">10.0 ms</td></tr>
</tbody>
<tbody class="lang-band lang-matlab"><tr class="lang-band-row" data-lang="matlab"><th colspan="3" scope="rowgroup"><span class="lang-band-dot"></span>MATLAB · pls4all</th></tr>
<tr class="bk-row"><td class="bk-name"><code>pls4all.matlab</code></td><td class="parity parity-exact">✓ bind</td><td class="ms">6.51 ms</td></tr>
<tr class="bk-row"><td class="bk-name"><code>pls4all.matlab.classdef</code></td><td class="parity parity-exact">✓ bind</td><td class="ms">9.49 ms</td></tr>
</tbody>
<tbody class="lang-band lang-python"><tr class="lang-band-row" data-lang="python"><th colspan="3" scope="rowgroup"><span class="lang-band-dot"></span>Python · external</th></tr>
<tr class="bk-row"><td class="bk-name"><code>ikpls</code></td><td class="parity parity-not_available">⊘</td><td class="ms">—</td></tr>
<tr class="bk-row truth-source truth-source-qualitative"><td class="bk-name"><span class="truth-mark" title="Registry parity reference (python): pyswarms 1.3.0 — qualitative (rmse_rel ≤ 1e+00)">📐</span><code>ref.python_pyswarms</code></td><td class="parity parity-exact">✓ ref</td><td class="ms">174.6 ms</td></tr>
<tr class="bk-row"><td class="bk-name"><code>sklearn</code></td><td class="parity parity-not_available">⊘</td><td class="ms">—</td></tr>
</tbody>
<tbody class="lang-band lang-r"><tr class="lang-band-row" data-lang="r"><th colspan="3" scope="rowgroup"><span class="lang-band-dot"></span>R · external</th></tr>
<tr class="bk-row"><td class="bk-name"><code>mixOmics</code></td><td class="parity parity-not_available">⊘</td><td class="ms">—</td></tr>
<tr class="bk-row"><td class="bk-name"><code>pls</code></td><td class="parity parity-not_available">⊘</td><td class="ms">—</td></tr>
<tr class="bk-row"><td class="bk-name"><code>ropls</code></td><td class="parity parity-not_available">⊘</td><td class="ms">—</td></tr>
</tbody>
<tbody class="lang-band lang-matlab"><tr class="lang-band-row" data-lang="matlab"><th colspan="3" scope="rowgroup"><span class="lang-band-dot"></span>MATLAB · external</th></tr>
<tr class="bk-row"><td class="bk-name"><code>plsregress</code></td><td class="parity parity-not_run">—</td><td class="ms">—</td></tr>
</tbody>
</table>
</div>

:::

::::


---

_See also_: [benchmark overview](../benchmarks/overview.md) · [methods index](index.md) · [interactive dashboard](../landing/dashboard.md)