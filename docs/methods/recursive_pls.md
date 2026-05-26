# `recursive_pls` — Recursive (moving-window) PLS

_Group_: **Core PLS** · _Registry tolerance_: `0.1`

## Description

Recursive (moving-window) PLS

From the `pls4all.sklearn.RecursivePLSRegression` docstring:

> Moving-window recursive PLS.

### Parameters

| Name | Type | Default | Notes |
|------|------|---------|-------|
| `n_components` | `int` | `2` | Number of latent components extracted (k). |
| `window_size` | `int` | `50` | Length of the moving window for recursive / interval-random-frog models. |

## Explanations

### Bibliographic source

Helland, K., Berntsen, H. E., Borgen, O. S. & Martens, H. (1992). *Recursive algorithm for partial least squares regression*. Chemometrics and Intelligent Laboratory Systems 14(1–3), 129–137.

### Mathematical principle

Process-analytical instruments produce streams of spectra under drifting conditions (changing humidity, instrument warm-up, fouling). Recursive PLS maintains a fitted model that **adapts as new samples arrive** by re-fitting on a sliding window of the most recent $w$ samples.

At time step $t$, the model is fit on $\{(\mathbf{x}_{t-w+1}, y_{t-w+1}), \ldots, (\mathbf{x}_t, y_t)\}$ and applied to incoming $\mathbf{x}_{t+1}$. Computational cost is $O(wpk)$ per step. The window width $w$ controls a stability/adaptability trade-off: short windows track drift aggressively but are noisier; long windows are stable but lag.

More sophisticated recursive variants (Qin 1998) use exponential forgetting factors instead of a hard window. pls4all's variant uses the hard-window form for deterministic parity with R `pls` rolling refits.

### Implementation

`n4m_recursive_pls_run` (returns predictions only — no global coefficient export, since the model changes per step). The Python sklearn wrapper is an in-sample-only estimator.

MATLAB header (`bindings/matlab/+pls4all/RecursivePlsRegression.m`):

```text
pls4all.RecursivePlsRegression  Moving-window recursive PLS.
 In-sample only: result holds in-window predictions; no global coefficient
 matrix. `predict(X)` returns the stored predictions for the training X
 (length-preserved; warmup samples are 0).
```

### Usage

Every pls4all binding tab dispatches into the same C kernel; the external libraries listed at the bottom of the page are the parity references registered in `benchmarks.parity_timing.registry`. Switch tabs to read the same fit in your language. The R package now ships drop-in-compatible facades for the CRAN `pls` package (`plsr`, `pcr`, `mvr`) and for the `mdatools::pls(x, y, ...)` matrix idiom — those tabs appear only on the methods that have a meaningful equivalence.

**pls4all bindings**

::::{tab-set}
:class: pls4all-bindings

:::{tab-item} C ABI · libn4m
:sync: c
:class-label: lang-c

```c
/* C ABI — libn4m */
n4m_context_t* ctx = n4m_context_create();
n4m_config_t*  cfg = n4m_config_create();
n4m_method_result_t* res = NULL;
n4m_recursive_pls_run(ctx, cfg, &x_view, &y_view, /* hyperparams */, &res);
/* … read coefficients / mask / scores via */
/* n4m_method_result_get_double_matrix / vector / scalar … */
n4m_method_result_destroy(res);
n4m_config_destroy(cfg);
n4m_context_destroy(ctx);
```

:::

:::{tab-item} Python · pls4all (raw)
:sync: python-raw
:class-label: lang-python

```python
import pls4all
from pls4all._methods import recursive_pls_run
with pls4all.Context() as ctx, pls4all.Config() as cfg:
    res = recursive_pls_run(ctx, cfg, X, y, n_components=3)
# then: res.matrix("predictions"), res.matrix("coefficients"),
# res.vector("mask"), res.scalar("intercept"), …
```

:::

:::{tab-item} Python · pls4all.sklearn
:sync: python-sklearn
:class-label: lang-python

```python
from pls4all.sklearn import RecursivePLSRegression
mdl = RecursivePLSRegression(n_components=2, window_size=50)
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
res <- pls4all_method("recursive_pls", X, y,
                      n_components = 3L, params = list(window_size = 60L))
# res is a named list with MethodResult arrays/scalars.
# selected_indices / top_k_intervals are 1-based.
```

:::

:::{tab-item} MATLAB · pls4all (MEX)
:sync: matlab-mex
:class-label: lang-matlab

```matlab
res = pls4all.recursive_pls(X, y, 3);
% see header of bindings/matlab/+pls4all/recursive_pls.m for full
% parameter surface:
%   res = recursive_pls(X, Y, n_components, window_size)
yhat = predict(res, Xtest);
```

:::

:::{tab-item} MATLAB · pls4all (classdef)
:sync: matlab-classdef
:class-label: lang-matlab

```matlab
mdl  = pls4all.fit("recursive_pls", X, y, "NumComponents", 3);
yhat = predict(mdl, Xtest);
```

:::

::::


**Registry parity references** 📐

:::{card}
:class-card: external-refs

- 📐 **`ref.python_scikit_learn`** (python · python) — `scikit-learn` 1.4.2 · qualitative (rmse_rel ≤ 1e-01) — Moving-window refit using sklearn PLSRegression (NIPALS).
- 📐 **`ref.r_pls`** (R · r) — `pls` 2.8.5 · qualitative (rmse_rel ≤ 1e-01) — Moving-window refit using R `pls::plsr` (SIMPLS by default).
:::

### Benchmarks

Adaptive wall-clock per cell measured against [`full_matrix.csv`](../benchmarks/overview.md). Only backends that implement this method are listed; libraries without the method are omitted.

**Verdict** &nbsp;·&nbsp; ✓ ref / ≈ ref / ~ shape mark a reference-gate pass at strict / relaxed / qualitative tolerance &nbsp;·&nbsp; ✓ bind = pls4all binding agrees with the C++ baseline &nbsp;·&nbsp; ✗ divergent &nbsp;·&nbsp; ⚠ error &nbsp;·&nbsp; — not run. The fastest backend per column is marked 🏆.

**Reference gate**: qualitative — shape/smoke comparison only. The external library and pls4all do not produce numerically equivalent output for this method (see the MethodSpec notes); the `rmse_rel_tol ≤ 1e-01` budget is set wide on purpose. Treat ~ shape as *“we ran both, both finished”*, not as numerical agreement.

Rows tagged with **📐** are the canonical parity references for this method (declared in [`parity_timing.registry`](../benchmarks/methodology.md)). C++ and external rows show reference parity; pls4all language bindings show binding parity against the C++ backend. Hover the icon for role and tolerance band.

::::{tab-set}
:class: parity-tabs

:::{tab-item} 1 thread
:sync: threads-1

<div class="parity-table-wrap">
<table class="docutils parity-grouped">
<thead><tr><th scope="col">Backend</th><th scope="col">Parity</th><th class="size-col" scope="col">50×250 (ms)</th><th class="size-col" scope="col">100×50 (ms)</th><th class="size-col" scope="col">100×500 (ms)</th><th class="size-col" scope="col">100×2500 (ms)</th><th class="size-col" scope="col">200×50 (ms)</th><th class="size-col" scope="col">250×50 (ms)</th><th class="size-col" scope="col">500×50 (ms)</th><th class="size-col" scope="col">500×500 (ms)</th><th class="size-col" scope="col">500×2500 (ms)</th><th class="size-col" scope="col">2500×50 (ms)</th><th class="size-col" scope="col">2500×500 (ms)</th><th class="size-col" scope="col">2500×2500 (ms)</th><th class="size-col" scope="col">10000×50 (ms)</th><th class="size-col" scope="col">10000×500 (ms)</th></tr></thead>
<tbody class="lang-band lang-cpp"><tr class="lang-band-row" data-lang="cpp"><th colspan="16" scope="rowgroup"><span class="lang-band-dot"></span>C++ native · libn4m</th></tr>
<tr class="bk-row"><td class="bk-name"><code>pls4all.cpp.blas</code></td><td class="parity parity-error">⚠ +2e-15</td><td class="ms">—</td><td class="ms ms-best">1.81 ms<span class="medal" title="fastest">🏆</span></td><td class="ms">22.0 ms</td><td class="ms">117.5 ms</td><td class="ms ms-best">3.04 ms<span class="medal" title="fastest">🏆</span></td><td class="ms">5.04 ms</td><td class="ms ms-best">15.5 ms<span class="medal" title="fastest">🏆</span></td><td class="ms ms-best">140.6 ms<span class="medal" title="fastest">🏆</span></td><td class="ms">837.9 ms</td><td class="ms">100.5 ms</td><td class="ms">700.0 ms</td><td class="ms">5.8 s</td><td class="ms ms-best">244.1 ms<span class="medal" title="fastest">🏆</span></td><td class="ms ms-best">2.4 s<span class="medal" title="fastest">🏆</span></td></tr>
<tr class="bk-row"><td class="bk-name"><code>pls4all.cpp.blas+omp</code></td><td class="parity parity-error">⚠ +2e-15</td><td class="ms">—</td><td class="ms">3.42 ms</td><td class="ms ms-best">18.7 ms<span class="medal" title="fastest">🏆</span></td><td class="ms ms-best">116.0 ms<span class="medal" title="fastest">🏆</span></td><td class="ms">3.12 ms</td><td class="ms">4.92 ms</td><td class="ms">16.5 ms</td><td class="ms">158.4 ms</td><td class="ms ms-best">752.2 ms<span class="medal" title="fastest">🏆</span></td><td class="ms ms-best">92.4 ms<span class="medal" title="fastest">🏆</span></td><td class="ms ms-best">662.4 ms<span class="medal" title="fastest">🏆</span></td><td class="ms ms-best">5.3 s<span class="medal" title="fastest">🏆</span></td><td class="ms">250.1 ms</td><td class="ms">2.4 s</td></tr>
<tr class="bk-row"><td class="bk-name"><code>pls4all.cpp.omp</code></td><td class="parity parity-error">⚠ +2e-15</td><td class="ms">—</td><td class="ms">2.99 ms</td><td class="ms">26.9 ms</td><td class="ms">132.2 ms</td><td class="ms">3.60 ms</td><td class="ms">5.99 ms</td><td class="ms">19.2 ms</td><td class="ms">202.5 ms</td><td class="ms">996.6 ms</td><td class="ms">103.0 ms</td><td class="ms">926.0 ms</td><td class="ms">6.8 s</td><td class="ms">324.9 ms</td><td class="ms">3.4 s</td></tr>
<tr class="bk-row"><td class="bk-name"><code>pls4all.cpp.ref</code></td><td class="parity parity-error">⚠ +2e-15</td><td class="ms">—</td><td class="ms">3.55 ms</td><td class="ms">26.3 ms</td><td class="ms">135.6 ms</td><td class="ms">3.73 ms</td><td class="ms">5.83 ms</td><td class="ms">18.6 ms</td><td class="ms">214.1 ms</td><td class="ms">1.1 s</td><td class="ms">115.9 ms</td><td class="ms">1.1 s</td><td class="ms">7.3 s</td><td class="ms">335.6 ms</td><td class="ms">3.5 s</td></tr>
</tbody>
<tbody class="lang-band lang-python"><tr class="lang-band-row" data-lang="python"><th colspan="16" scope="rowgroup"><span class="lang-band-dot"></span>Python · pls4all</th></tr>
<tr class="bk-row"><td class="bk-name"><code>pls4all.python</code></td><td class="parity parity-error">⚠</td><td class="ms">—</td><td class="ms ms-empty">—</td><td class="ms ms-empty">—</td><td class="ms ms-empty">—</td><td class="ms">3.24 ms</td><td class="ms">4.89 ms</td><td class="ms ms-empty">—</td><td class="ms ms-empty">—</td><td class="ms ms-empty">—</td><td class="ms ms-empty">—</td><td class="ms ms-empty">—</td><td class="ms ms-empty">—</td><td class="ms ms-empty">—</td><td class="ms ms-empty">—</td></tr>
<tr class="bk-row"><td class="bk-name"><code>pls4all.sklearn</code></td><td class="parity parity-error">⚠</td><td class="ms">—</td><td class="ms ms-empty">—</td><td class="ms ms-empty">—</td><td class="ms ms-empty">—</td><td class="ms">5.12 ms</td><td class="ms ms-best">4.85 ms<span class="medal" title="fastest">🏆</span></td><td class="ms ms-empty">—</td><td class="ms ms-empty">—</td><td class="ms ms-empty">—</td><td class="ms ms-empty">—</td><td class="ms ms-empty">—</td><td class="ms ms-empty">—</td><td class="ms ms-empty">—</td><td class="ms ms-empty">—</td></tr>
</tbody>
<tbody class="lang-band lang-r"><tr class="lang-band-row" data-lang="r"><th colspan="16" scope="rowgroup"><span class="lang-band-dot"></span>R · pls4all</th></tr>
<tr class="bk-row"><td class="bk-name"><code>pls4all.R</code></td><td class="parity parity-exact">✓ 9e-15</td><td class="ms">—</td><td class="ms ms-empty">—</td><td class="ms ms-empty">—</td><td class="ms ms-empty">—</td><td class="ms">8.73 ms</td><td class="ms">14.1 ms</td><td class="ms ms-empty">—</td><td class="ms ms-empty">—</td><td class="ms ms-empty">—</td><td class="ms ms-empty">—</td><td class="ms ms-empty">—</td><td class="ms ms-empty">—</td><td class="ms ms-empty">—</td><td class="ms ms-empty">—</td></tr>
<tr class="bk-row"><td class="bk-name"><code>pls4all.R.formula</code></td><td class="parity parity-exact">✓ 9e-15</td><td class="ms">—</td><td class="ms ms-empty">—</td><td class="ms ms-empty">—</td><td class="ms ms-empty">—</td><td class="ms">10.6 ms</td><td class="ms">11.0 ms</td><td class="ms ms-empty">—</td><td class="ms ms-empty">—</td><td class="ms ms-empty">—</td><td class="ms ms-empty">—</td><td class="ms ms-empty">—</td><td class="ms ms-empty">—</td><td class="ms ms-empty">—</td><td class="ms ms-empty">—</td></tr>
<tr class="bk-row"><td class="bk-name"><code>pls4all.R.mdatools</code></td><td class="parity parity-exact">✓ 9e-15</td><td class="ms">—</td><td class="ms ms-empty">—</td><td class="ms ms-empty">—</td><td class="ms ms-empty">—</td><td class="ms">9.76 ms</td><td class="ms">12.1 ms</td><td class="ms ms-empty">—</td><td class="ms ms-empty">—</td><td class="ms ms-empty">—</td><td class="ms ms-empty">—</td><td class="ms ms-empty">—</td><td class="ms ms-empty">—</td><td class="ms ms-empty">—</td><td class="ms ms-empty">—</td></tr>
<tr class="bk-row"><td class="bk-name"><code>pls4all.R.pls</code></td><td class="parity parity-exact">✓ 9e-15</td><td class="ms">—</td><td class="ms ms-empty">—</td><td class="ms ms-empty">—</td><td class="ms ms-empty">—</td><td class="ms">11.3 ms</td><td class="ms">12.6 ms</td><td class="ms ms-empty">—</td><td class="ms ms-empty">—</td><td class="ms ms-empty">—</td><td class="ms ms-empty">—</td><td class="ms ms-empty">—</td><td class="ms ms-empty">—</td><td class="ms ms-empty">—</td><td class="ms ms-empty">—</td></tr>
</tbody>
<tbody class="lang-band lang-matlab"><tr class="lang-band-row" data-lang="matlab"><th colspan="16" scope="rowgroup"><span class="lang-band-dot"></span>MATLAB · pls4all</th></tr>
<tr class="bk-row"><td class="bk-name"><code>pls4all.matlab</code></td><td class="parity parity-divergent">✗ +1e+01</td><td class="ms">—</td><td class="ms ms-empty">—</td><td class="ms ms-empty">—</td><td class="ms ms-empty">—</td><td class="ms">4.35 ms</td><td class="ms">6.37 ms</td><td class="ms ms-empty">—</td><td class="ms ms-empty">—</td><td class="ms ms-empty">—</td><td class="ms ms-empty">—</td><td class="ms ms-empty">—</td><td class="ms ms-empty">—</td><td class="ms ms-empty">—</td><td class="ms ms-empty">—</td></tr>
<tr class="bk-row"><td class="bk-name"><code>pls4all.matlab.classdef</code></td><td class="parity parity-divergent">✗ +1e+01</td><td class="ms">—</td><td class="ms ms-empty">—</td><td class="ms ms-empty">—</td><td class="ms ms-empty">—</td><td class="ms">5.16 ms</td><td class="ms">7.59 ms</td><td class="ms ms-empty">—</td><td class="ms ms-empty">—</td><td class="ms ms-empty">—</td><td class="ms ms-empty">—</td><td class="ms ms-empty">—</td><td class="ms ms-empty">—</td><td class="ms ms-empty">—</td><td class="ms ms-empty">—</td></tr>
</tbody>
<tbody class="lang-band lang-python"><tr class="lang-band-row" data-lang="python"><th colspan="16" scope="rowgroup"><span class="lang-band-dot"></span>Python · external</th></tr>
<tr class="bk-row truth-source truth-source-qualitative"><td class="bk-name"><span class="truth-mark" title="Registry parity reference (python): scikit-learn 1.4.2 — qualitative (rmse_rel ≤ 1e-01)">📐</span><code>ref.python_scikit_learn</code></td><td class="parity parity-ref-source">source</td><td class="ms ms-best">2.50 ms<span class="medal" title="fastest">🏆</span></td><td class="ms ms-empty">—</td><td class="ms ms-empty">—</td><td class="ms ms-empty">—</td><td class="ms">70.0 ms</td><td class="ms">93.0 ms</td><td class="ms ms-empty">—</td><td class="ms ms-empty">—</td><td class="ms ms-empty">—</td><td class="ms ms-empty">—</td><td class="ms ms-empty">—</td><td class="ms ms-empty">—</td><td class="ms ms-empty">—</td><td class="ms ms-empty">—</td></tr>
</tbody>
<tbody class="lang-band lang-r"><tr class="lang-band-row" data-lang="r"><th colspan="16" scope="rowgroup"><span class="lang-band-dot"></span>R · external</th></tr>
<tr class="bk-row truth-source truth-source-qualitative"><td class="bk-name"><span class="truth-mark" title="Registry parity reference (r): pls 2.8.5 — qualitative (rmse_rel ≤ 1e-01)">📐</span><code>ref.r_pls</code></td><td class="parity parity-error">⚠ +2e-14</td><td class="ms">—</td><td class="ms ms-empty">—</td><td class="ms ms-empty">—</td><td class="ms ms-empty">—</td><td class="ms">216.4 ms</td><td class="ms">206.2 ms</td><td class="ms ms-empty">—</td><td class="ms ms-empty">—</td><td class="ms ms-empty">—</td><td class="ms ms-empty">—</td><td class="ms ms-empty">—</td><td class="ms ms-empty">—</td><td class="ms ms-empty">—</td><td class="ms ms-empty">—</td></tr>
</tbody>
</table>
</div>

:::

:::{tab-item} 3 threads
:sync: threads-3

<div class="parity-table-wrap">
<table class="docutils parity-grouped">
<thead><tr><th scope="col">Backend</th><th scope="col">Parity</th><th class="size-col" scope="col">50×250 (ms)</th><th class="size-col" scope="col">100×50 (ms)</th><th class="size-col" scope="col">100×500 (ms)</th><th class="size-col" scope="col">100×2500 (ms)</th><th class="size-col" scope="col">200×50 (ms)</th><th class="size-col" scope="col">250×50 (ms)</th><th class="size-col" scope="col">500×50 (ms)</th><th class="size-col" scope="col">500×500 (ms)</th><th class="size-col" scope="col">500×2500 (ms)</th><th class="size-col" scope="col">2500×50 (ms)</th><th class="size-col" scope="col">2500×500 (ms)</th><th class="size-col" scope="col">2500×2500 (ms)</th><th class="size-col" scope="col">10000×50 (ms)</th><th class="size-col" scope="col">10000×500 (ms)</th></tr></thead>
<tbody class="lang-band lang-cpp"><tr class="lang-band-row" data-lang="cpp"><th colspan="16" scope="rowgroup"><span class="lang-band-dot"></span>C++ native · libn4m</th></tr>
<tr class="bk-row"><td class="bk-name"><code>pls4all.cpp.blas</code></td><td class="parity parity-ref-qualitative">~ shape 9e-16</td><td class="ms ms-empty">—</td><td class="ms ms-empty">—</td><td class="ms ms-empty">—</td><td class="ms ms-empty">—</td><td class="ms ms-best">3.08 ms<span class="medal" title="fastest">🏆</span></td><td class="ms ms-empty">—</td><td class="ms ms-empty">—</td><td class="ms ms-empty">—</td><td class="ms ms-empty">—</td><td class="ms ms-empty">—</td><td class="ms ms-empty">—</td><td class="ms ms-empty">—</td><td class="ms ms-empty">—</td><td class="ms ms-empty">—</td></tr>
<tr class="bk-row"><td class="bk-name"><code>pls4all.cpp.blas+omp</code></td><td class="parity parity-ref-qualitative">~ shape 9e-16</td><td class="ms ms-empty">—</td><td class="ms ms-empty">—</td><td class="ms ms-empty">—</td><td class="ms ms-empty">—</td><td class="ms">3.36 ms</td><td class="ms ms-empty">—</td><td class="ms ms-empty">—</td><td class="ms ms-empty">—</td><td class="ms ms-empty">—</td><td class="ms ms-empty">—</td><td class="ms ms-empty">—</td><td class="ms ms-empty">—</td><td class="ms ms-empty">—</td><td class="ms ms-empty">—</td></tr>
<tr class="bk-row"><td class="bk-name"><code>pls4all.cpp.omp</code></td><td class="parity parity-ref-qualitative">~ shape 1e-15</td><td class="ms ms-empty">—</td><td class="ms ms-empty">—</td><td class="ms ms-empty">—</td><td class="ms ms-empty">—</td><td class="ms">5.70 ms</td><td class="ms ms-empty">—</td><td class="ms ms-empty">—</td><td class="ms ms-empty">—</td><td class="ms ms-empty">—</td><td class="ms ms-empty">—</td><td class="ms ms-empty">—</td><td class="ms ms-empty">—</td><td class="ms ms-empty">—</td><td class="ms ms-empty">—</td></tr>
<tr class="bk-row"><td class="bk-name"><code>pls4all.cpp.ref</code></td><td class="parity parity-ref-qualitative">~ shape 1e-15</td><td class="ms ms-empty">—</td><td class="ms ms-empty">—</td><td class="ms ms-empty">—</td><td class="ms ms-empty">—</td><td class="ms">4.10 ms</td><td class="ms ms-empty">—</td><td class="ms ms-empty">—</td><td class="ms ms-empty">—</td><td class="ms ms-empty">—</td><td class="ms ms-empty">—</td><td class="ms ms-empty">—</td><td class="ms ms-empty">—</td><td class="ms ms-empty">—</td><td class="ms ms-empty">—</td></tr>
</tbody>
<tbody class="lang-band lang-python"><tr class="lang-band-row" data-lang="python"><th colspan="16" scope="rowgroup"><span class="lang-band-dot"></span>Python · pls4all</th></tr>
<tr class="bk-row"><td class="bk-name"><code>pls4all.python</code></td><td class="parity parity-exact">✓ bind</td><td class="ms ms-empty">—</td><td class="ms ms-empty">—</td><td class="ms ms-empty">—</td><td class="ms ms-empty">—</td><td class="ms">3.22 ms</td><td class="ms ms-empty">—</td><td class="ms ms-empty">—</td><td class="ms ms-empty">—</td><td class="ms ms-empty">—</td><td class="ms ms-empty">—</td><td class="ms ms-empty">—</td><td class="ms ms-empty">—</td><td class="ms ms-empty">—</td><td class="ms ms-empty">—</td></tr>
<tr class="bk-row"><td class="bk-name"><code>pls4all.sklearn</code></td><td class="parity parity-exact">✓ bind</td><td class="ms ms-empty">—</td><td class="ms ms-empty">—</td><td class="ms ms-empty">—</td><td class="ms ms-empty">—</td><td class="ms">3.54 ms</td><td class="ms ms-empty">—</td><td class="ms ms-empty">—</td><td class="ms ms-empty">—</td><td class="ms ms-empty">—</td><td class="ms ms-empty">—</td><td class="ms ms-empty">—</td><td class="ms ms-empty">—</td><td class="ms ms-empty">—</td><td class="ms ms-empty">—</td></tr>
</tbody>
<tbody class="lang-band lang-r"><tr class="lang-band-row" data-lang="r"><th colspan="16" scope="rowgroup"><span class="lang-band-dot"></span>R · pls4all</th></tr>
<tr class="bk-row"><td class="bk-name"><code>pls4all.R</code></td><td class="parity parity-exact">✓ 4e-15</td><td class="ms ms-empty">—</td><td class="ms ms-empty">—</td><td class="ms ms-empty">—</td><td class="ms ms-empty">—</td><td class="ms">8.26 ms</td><td class="ms ms-empty">—</td><td class="ms ms-empty">—</td><td class="ms ms-empty">—</td><td class="ms ms-empty">—</td><td class="ms ms-empty">—</td><td class="ms ms-empty">—</td><td class="ms ms-empty">—</td><td class="ms ms-empty">—</td><td class="ms ms-empty">—</td></tr>
<tr class="bk-row"><td class="bk-name"><code>pls4all.R.formula</code></td><td class="parity parity-exact">✓ 4e-15</td><td class="ms ms-empty">—</td><td class="ms ms-empty">—</td><td class="ms ms-empty">—</td><td class="ms ms-empty">—</td><td class="ms">9.96 ms</td><td class="ms ms-empty">—</td><td class="ms ms-empty">—</td><td class="ms ms-empty">—</td><td class="ms ms-empty">—</td><td class="ms ms-empty">—</td><td class="ms ms-empty">—</td><td class="ms ms-empty">—</td><td class="ms ms-empty">—</td><td class="ms ms-empty">—</td></tr>
<tr class="bk-row"><td class="bk-name"><code>pls4all.R.mdatools</code></td><td class="parity parity-exact">✓ 4e-15</td><td class="ms ms-empty">—</td><td class="ms ms-empty">—</td><td class="ms ms-empty">—</td><td class="ms ms-empty">—</td><td class="ms">9.54 ms</td><td class="ms ms-empty">—</td><td class="ms ms-empty">—</td><td class="ms ms-empty">—</td><td class="ms ms-empty">—</td><td class="ms ms-empty">—</td><td class="ms ms-empty">—</td><td class="ms ms-empty">—</td><td class="ms ms-empty">—</td><td class="ms ms-empty">—</td></tr>
<tr class="bk-row"><td class="bk-name"><code>pls4all.R.pls</code></td><td class="parity parity-exact">✓ 4e-15</td><td class="ms ms-empty">—</td><td class="ms ms-empty">—</td><td class="ms ms-empty">—</td><td class="ms ms-empty">—</td><td class="ms">9.47 ms</td><td class="ms ms-empty">—</td><td class="ms ms-empty">—</td><td class="ms ms-empty">—</td><td class="ms ms-empty">—</td><td class="ms ms-empty">—</td><td class="ms ms-empty">—</td><td class="ms ms-empty">—</td><td class="ms ms-empty">—</td><td class="ms ms-empty">—</td></tr>
</tbody>
<tbody class="lang-band lang-matlab"><tr class="lang-band-row" data-lang="matlab"><th colspan="16" scope="rowgroup"><span class="lang-band-dot"></span>MATLAB · pls4all</th></tr>
<tr class="bk-row"><td class="bk-name"><code>pls4all.matlab</code></td><td class="parity parity-divergent">✗ +1e+01</td><td class="ms ms-empty">—</td><td class="ms ms-empty">—</td><td class="ms ms-empty">—</td><td class="ms ms-empty">—</td><td class="ms">4.51 ms</td><td class="ms ms-empty">—</td><td class="ms ms-empty">—</td><td class="ms ms-empty">—</td><td class="ms ms-empty">—</td><td class="ms ms-empty">—</td><td class="ms ms-empty">—</td><td class="ms ms-empty">—</td><td class="ms ms-empty">—</td><td class="ms ms-empty">—</td></tr>
<tr class="bk-row"><td class="bk-name"><code>pls4all.matlab.classdef</code></td><td class="parity parity-divergent">✗ +1e+01</td><td class="ms ms-empty">—</td><td class="ms ms-empty">—</td><td class="ms ms-empty">—</td><td class="ms ms-empty">—</td><td class="ms">4.76 ms</td><td class="ms ms-empty">—</td><td class="ms ms-empty">—</td><td class="ms ms-empty">—</td><td class="ms ms-empty">—</td><td class="ms ms-empty">—</td><td class="ms ms-empty">—</td><td class="ms ms-empty">—</td><td class="ms ms-empty">—</td><td class="ms ms-empty">—</td></tr>
</tbody>
<tbody class="lang-band lang-python"><tr class="lang-band-row" data-lang="python"><th colspan="16" scope="rowgroup"><span class="lang-band-dot"></span>Python · external</th></tr>
<tr class="bk-row truth-source truth-source-qualitative"><td class="bk-name"><span class="truth-mark" title="Registry parity reference (python): scikit-learn 1.4.2 — qualitative (rmse_rel ≤ 1e-01)">📐</span><code>ref.python_scikit_learn</code></td><td class="parity parity-ref-source">source</td><td class="ms ms-empty">—</td><td class="ms ms-empty">—</td><td class="ms ms-empty">—</td><td class="ms ms-empty">—</td><td class="ms">60.4 ms</td><td class="ms ms-empty">—</td><td class="ms ms-empty">—</td><td class="ms ms-empty">—</td><td class="ms ms-empty">—</td><td class="ms ms-empty">—</td><td class="ms ms-empty">—</td><td class="ms ms-empty">—</td><td class="ms ms-empty">—</td><td class="ms ms-empty">—</td></tr>
</tbody>
<tbody class="lang-band lang-r"><tr class="lang-band-row" data-lang="r"><th colspan="16" scope="rowgroup"><span class="lang-band-dot"></span>R · external</th></tr>
<tr class="bk-row truth-source truth-source-qualitative"><td class="bk-name"><span class="truth-mark" title="Registry parity reference (r): pls 2.8.5 — qualitative (rmse_rel ≤ 1e-01)">📐</span><code>ref.r_pls</code></td><td class="parity parity-ref-qualitative">~ shape 4e-15</td><td class="ms ms-empty">—</td><td class="ms ms-empty">—</td><td class="ms ms-empty">—</td><td class="ms ms-empty">—</td><td class="ms">195.9 ms</td><td class="ms ms-empty">—</td><td class="ms ms-empty">—</td><td class="ms ms-empty">—</td><td class="ms ms-empty">—</td><td class="ms ms-empty">—</td><td class="ms ms-empty">—</td><td class="ms ms-empty">—</td><td class="ms ms-empty">—</td><td class="ms ms-empty">—</td></tr>
</tbody>
</table>
</div>

:::

:::{tab-item} 10 threads
:sync: threads-10

<div class="parity-table-wrap">
<table class="docutils parity-grouped">
<thead><tr><th scope="col">Backend</th><th scope="col">Parity</th><th class="size-col" scope="col">50×250 (ms)</th><th class="size-col" scope="col">100×50 (ms)</th><th class="size-col" scope="col">100×500 (ms)</th><th class="size-col" scope="col">100×2500 (ms)</th><th class="size-col" scope="col">200×50 (ms)</th><th class="size-col" scope="col">250×50 (ms)</th><th class="size-col" scope="col">500×50 (ms)</th><th class="size-col" scope="col">500×500 (ms)</th><th class="size-col" scope="col">500×2500 (ms)</th><th class="size-col" scope="col">2500×50 (ms)</th><th class="size-col" scope="col">2500×500 (ms)</th><th class="size-col" scope="col">2500×2500 (ms)</th><th class="size-col" scope="col">10000×50 (ms)</th><th class="size-col" scope="col">10000×500 (ms)</th></tr></thead>
<tbody class="lang-band lang-cpp"><tr class="lang-band-row" data-lang="cpp"><th colspan="16" scope="rowgroup"><span class="lang-band-dot"></span>C++ native · libn4m</th></tr>
<tr class="bk-row"><td class="bk-name"><code>pls4all.cpp.blas</code></td><td class="parity parity-ref-qualitative">~ shape 9e-16</td><td class="ms ms-empty">—</td><td class="ms ms-empty">—</td><td class="ms ms-empty">—</td><td class="ms ms-empty">—</td><td class="ms ms-best">2.80 ms<span class="medal" title="fastest">🏆</span></td><td class="ms ms-empty">—</td><td class="ms ms-empty">—</td><td class="ms ms-empty">—</td><td class="ms ms-empty">—</td><td class="ms ms-empty">—</td><td class="ms ms-empty">—</td><td class="ms ms-empty">—</td><td class="ms ms-empty">—</td><td class="ms ms-empty">—</td></tr>
<tr class="bk-row"><td class="bk-name"><code>pls4all.cpp.blas+omp</code></td><td class="parity parity-ref-qualitative">~ shape 9e-16</td><td class="ms ms-empty">—</td><td class="ms ms-empty">—</td><td class="ms ms-empty">—</td><td class="ms ms-empty">—</td><td class="ms">2.84 ms</td><td class="ms ms-empty">—</td><td class="ms ms-empty">—</td><td class="ms ms-empty">—</td><td class="ms ms-empty">—</td><td class="ms ms-empty">—</td><td class="ms ms-empty">—</td><td class="ms ms-empty">—</td><td class="ms ms-empty">—</td><td class="ms ms-empty">—</td></tr>
<tr class="bk-row"><td class="bk-name"><code>pls4all.cpp.omp</code></td><td class="parity parity-ref-qualitative">~ shape 1e-15</td><td class="ms ms-empty">—</td><td class="ms ms-empty">—</td><td class="ms ms-empty">—</td><td class="ms ms-empty">—</td><td class="ms">3.37 ms</td><td class="ms ms-empty">—</td><td class="ms ms-empty">—</td><td class="ms ms-empty">—</td><td class="ms ms-empty">—</td><td class="ms ms-empty">—</td><td class="ms ms-empty">—</td><td class="ms ms-empty">—</td><td class="ms ms-empty">—</td><td class="ms ms-empty">—</td></tr>
<tr class="bk-row"><td class="bk-name"><code>pls4all.cpp.ref</code></td><td class="parity parity-ref-qualitative">~ shape 1e-15</td><td class="ms ms-empty">—</td><td class="ms ms-empty">—</td><td class="ms ms-empty">—</td><td class="ms ms-empty">—</td><td class="ms">3.45 ms</td><td class="ms ms-empty">—</td><td class="ms ms-empty">—</td><td class="ms ms-empty">—</td><td class="ms ms-empty">—</td><td class="ms ms-empty">—</td><td class="ms ms-empty">—</td><td class="ms ms-empty">—</td><td class="ms ms-empty">—</td><td class="ms ms-empty">—</td></tr>
</tbody>
<tbody class="lang-band lang-python"><tr class="lang-band-row" data-lang="python"><th colspan="16" scope="rowgroup"><span class="lang-band-dot"></span>Python · pls4all</th></tr>
<tr class="bk-row"><td class="bk-name"><code>pls4all.python</code></td><td class="parity parity-exact">✓ 4e-15</td><td class="ms ms-empty">—</td><td class="ms ms-empty">—</td><td class="ms ms-empty">—</td><td class="ms ms-empty">—</td><td class="ms">2.91 ms</td><td class="ms ms-empty">—</td><td class="ms ms-empty">—</td><td class="ms ms-empty">—</td><td class="ms ms-empty">—</td><td class="ms ms-empty">—</td><td class="ms ms-empty">—</td><td class="ms ms-empty">—</td><td class="ms ms-empty">—</td><td class="ms ms-empty">—</td></tr>
<tr class="bk-row"><td class="bk-name"><code>pls4all.sklearn</code></td><td class="parity parity-exact">✓ 4e-15</td><td class="ms ms-empty">—</td><td class="ms ms-empty">—</td><td class="ms ms-empty">—</td><td class="ms ms-empty">—</td><td class="ms">2.98 ms</td><td class="ms ms-empty">—</td><td class="ms ms-empty">—</td><td class="ms ms-empty">—</td><td class="ms ms-empty">—</td><td class="ms ms-empty">—</td><td class="ms ms-empty">—</td><td class="ms ms-empty">—</td><td class="ms ms-empty">—</td><td class="ms ms-empty">—</td></tr>
</tbody>
<tbody class="lang-band lang-r"><tr class="lang-band-row" data-lang="r"><th colspan="16" scope="rowgroup"><span class="lang-band-dot"></span>R · pls4all</th></tr>
<tr class="bk-row"><td class="bk-name"><code>pls4all.R</code></td><td class="parity parity-exact">✓ bind</td><td class="ms ms-empty">—</td><td class="ms ms-empty">—</td><td class="ms ms-empty">—</td><td class="ms ms-empty">—</td><td class="ms">6.38 ms</td><td class="ms ms-empty">—</td><td class="ms ms-empty">—</td><td class="ms ms-empty">—</td><td class="ms ms-empty">—</td><td class="ms ms-empty">—</td><td class="ms ms-empty">—</td><td class="ms ms-empty">—</td><td class="ms ms-empty">—</td><td class="ms ms-empty">—</td></tr>
<tr class="bk-row"><td class="bk-name"><code>pls4all.R.formula</code></td><td class="parity parity-exact">✓ bind</td><td class="ms ms-empty">—</td><td class="ms ms-empty">—</td><td class="ms ms-empty">—</td><td class="ms ms-empty">—</td><td class="ms">7.69 ms</td><td class="ms ms-empty">—</td><td class="ms ms-empty">—</td><td class="ms ms-empty">—</td><td class="ms ms-empty">—</td><td class="ms ms-empty">—</td><td class="ms ms-empty">—</td><td class="ms ms-empty">—</td><td class="ms ms-empty">—</td><td class="ms ms-empty">—</td></tr>
<tr class="bk-row"><td class="bk-name"><code>pls4all.R.mdatools</code></td><td class="parity parity-exact">✓ bind</td><td class="ms ms-empty">—</td><td class="ms ms-empty">—</td><td class="ms ms-empty">—</td><td class="ms ms-empty">—</td><td class="ms">7.80 ms</td><td class="ms ms-empty">—</td><td class="ms ms-empty">—</td><td class="ms ms-empty">—</td><td class="ms ms-empty">—</td><td class="ms ms-empty">—</td><td class="ms ms-empty">—</td><td class="ms ms-empty">—</td><td class="ms ms-empty">—</td><td class="ms ms-empty">—</td></tr>
<tr class="bk-row"><td class="bk-name"><code>pls4all.R.pls</code></td><td class="parity parity-exact">✓ bind</td><td class="ms ms-empty">—</td><td class="ms ms-empty">—</td><td class="ms ms-empty">—</td><td class="ms ms-empty">—</td><td class="ms">7.58 ms</td><td class="ms ms-empty">—</td><td class="ms ms-empty">—</td><td class="ms ms-empty">—</td><td class="ms ms-empty">—</td><td class="ms ms-empty">—</td><td class="ms ms-empty">—</td><td class="ms ms-empty">—</td><td class="ms ms-empty">—</td><td class="ms ms-empty">—</td></tr>
</tbody>
<tbody class="lang-band lang-matlab"><tr class="lang-band-row" data-lang="matlab"><th colspan="16" scope="rowgroup"><span class="lang-band-dot"></span>MATLAB · pls4all</th></tr>
<tr class="bk-row"><td class="bk-name"><code>pls4all.matlab</code></td><td class="parity parity-divergent">✗ +1e+01</td><td class="ms ms-empty">—</td><td class="ms ms-empty">—</td><td class="ms ms-empty">—</td><td class="ms ms-empty">—</td><td class="ms">4.05 ms</td><td class="ms ms-empty">—</td><td class="ms ms-empty">—</td><td class="ms ms-empty">—</td><td class="ms ms-empty">—</td><td class="ms ms-empty">—</td><td class="ms ms-empty">—</td><td class="ms ms-empty">—</td><td class="ms ms-empty">—</td><td class="ms ms-empty">—</td></tr>
<tr class="bk-row"><td class="bk-name"><code>pls4all.matlab.classdef</code></td><td class="parity parity-divergent">✗ +1e+01</td><td class="ms ms-empty">—</td><td class="ms ms-empty">—</td><td class="ms ms-empty">—</td><td class="ms ms-empty">—</td><td class="ms">4.35 ms</td><td class="ms ms-empty">—</td><td class="ms ms-empty">—</td><td class="ms ms-empty">—</td><td class="ms ms-empty">—</td><td class="ms ms-empty">—</td><td class="ms ms-empty">—</td><td class="ms ms-empty">—</td><td class="ms ms-empty">—</td><td class="ms ms-empty">—</td></tr>
</tbody>
<tbody class="lang-band lang-python"><tr class="lang-band-row" data-lang="python"><th colspan="16" scope="rowgroup"><span class="lang-band-dot"></span>Python · external</th></tr>
<tr class="bk-row truth-source truth-source-qualitative"><td class="bk-name"><span class="truth-mark" title="Registry parity reference (python): scikit-learn 1.4.2 — qualitative (rmse_rel ≤ 1e-01)">📐</span><code>ref.python_scikit_learn</code></td><td class="parity parity-ref-source">source</td><td class="ms ms-empty">—</td><td class="ms ms-empty">—</td><td class="ms ms-empty">—</td><td class="ms ms-empty">—</td><td class="ms">52.4 ms</td><td class="ms ms-empty">—</td><td class="ms ms-empty">—</td><td class="ms ms-empty">—</td><td class="ms ms-empty">—</td><td class="ms ms-empty">—</td><td class="ms ms-empty">—</td><td class="ms ms-empty">—</td><td class="ms ms-empty">—</td><td class="ms ms-empty">—</td></tr>
</tbody>
<tbody class="lang-band lang-r"><tr class="lang-band-row" data-lang="r"><th colspan="16" scope="rowgroup"><span class="lang-band-dot"></span>R · external</th></tr>
<tr class="bk-row truth-source truth-source-qualitative"><td class="bk-name"><span class="truth-mark" title="Registry parity reference (r): pls 2.8.5 — qualitative (rmse_rel ≤ 1e-01)">📐</span><code>ref.r_pls</code></td><td class="parity parity-ref-qualitative">~ shape 4e-15</td><td class="ms ms-empty">—</td><td class="ms ms-empty">—</td><td class="ms ms-empty">—</td><td class="ms ms-empty">—</td><td class="ms">147.7 ms</td><td class="ms ms-empty">—</td><td class="ms ms-empty">—</td><td class="ms ms-empty">—</td><td class="ms ms-empty">—</td><td class="ms ms-empty">—</td><td class="ms ms-empty">—</td><td class="ms ms-empty">—</td><td class="ms ms-empty">—</td><td class="ms ms-empty">—</td></tr>
</tbody>
</table>
</div>

:::

::::


---

_See also_: [benchmark overview](../benchmarks/overview.md) · [methods index](index.md) · [interactive dashboard](../landing/dashboard.md)