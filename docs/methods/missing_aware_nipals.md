# `missing_aware_nipals` — Missing-aware NIPALS

_Group_: **Missing data** · _Registry tolerance_: `10.0`

## Description

Missing-aware NIPALS PLS (§13)

From the `pls4all.sklearn.MissingAwareNipalsRegression` docstring:

> Nelson 1996 missing-aware NIPALS PLS.

> **Registry note** — R `softImpute` + `pls::plsr` reference: matrix completion + SIMPLS. Different imputation algorithm than Nelson 1996 NIPALS-missing; same goal. Cell has no missing values so softImpute reduces to mean-fill; widened tolerance flags ref presence.

### Parameters

| Name | Type | Default | Notes |
|------|------|---------|-------|
| `n_components` | `int` | `2` | Number of latent components extracted (k). |

## Explanations

### Bibliographic source

Walczak, B. & Massart, D. L. (2001). *Dealing with missing data: part I & II*. Chemometrics and Intelligent Laboratory Systems 58(1), 15–27 & 29–42. — applied to the NIPALS PLS algorithm.

### Mathematical principle

Standard PLS cannot ingest `NaN` values: the matrix operations propagate NaN. Missing-aware NIPALS treats missing entries as **excluded from the inner products** in the iterative loading-weight calculation. Concretely, the score $t_i = \sum_{j \in \mathcal{O}_i} x_{ij} w_j$ and the loading $p_j = \sum_{i \in \mathcal{O}_j} x_{ij} t_i / \sum_{i \in \mathcal{O}_j} t_i^2$ are computed only over the observed indices $\mathcal{O}_{i}, \mathcal{O}_{j}$.

Convergence is slower than vanilla NIPALS and the result is sensitive to the missingness pattern, but the alternative (delete rows with any NaN or impute by column means) usually performs much worse on spectroscopic data. Acceptable up to ~10 % missing entries; beyond that low-rank matrix completion (softImpute) is a better front-end.

Note that missing-aware NIPALS produces a regression model that can still predict on **complete** new $\mathbf{x}$; only the training step tolerates missing data.

### Implementation

`n4m_missing_aware_nipals_fit`. Reference: R `softImpute 1.4.3` for the imputation step; the PLS fitting itself is in-tree.

MATLAB header (`bindings/matlab/+pls4all/MissingAwareNipalsRegression.m`):

```text
pls4all.MissingAwareNipalsRegression  Nelson 1996 missing-aware NIPALS PLS.
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
n4m_missing_aware_nipals_fit(ctx, cfg, &x_view, &y_view, /* hyperparams */, &res);
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
from pls4all._methods import missing_aware_nipals_fit
with pls4all.Context() as ctx, pls4all.Config() as cfg:
    res = missing_aware_nipals_fit(ctx, cfg, X, y, n_components=4)
# then: res.matrix("predictions"), res.matrix("coefficients"),
# res.vector("mask"), res.scalar("intercept"), …
```

:::

:::{tab-item} Python · pls4all.sklearn
:sync: python-sklearn
:class-label: lang-python

```python
from pls4all.sklearn import MissingAwareNipalsRegression
mdl = MissingAwareNipalsRegression(n_components=2)
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
res <- pls4all_method("missing_aware_nipals", X, y,
                      n_components = 4L)
# res is a named list with MethodResult arrays/scalars.
# selected_indices / top_k_intervals are 1-based.
```

:::

:::{tab-item} MATLAB · pls4all (MEX)
:sync: matlab-mex
:class-label: lang-matlab

```matlab
res = pls4all.missing_aware_nipals(X, y, 4);
% see header of bindings/matlab/+pls4all/missing_aware_nipals.m for full
% parameter surface:
%   res = missing_aware_nipals(X, Y, n_components)
yhat = predict(res, Xtest);
```

:::

:::{tab-item} MATLAB · pls4all (classdef)
:sync: matlab-classdef
:class-label: lang-matlab

```matlab
mdl  = pls4all.fit("missing_aware_nipals", X, y, "NumComponents", 4);
yhat = predict(mdl, Xtest);
```

:::

::::


**Registry parity references** 📐

:::{card}
:class-card: external-refs

- 📐 **`ref.r_softimpute`** (R · r) — `softImpute` 1.4-1 · qualitative (rmse_rel ≤ 1e+01) — R `softImpute::softImpute` followed by `pls::plsr` on the completed (X, Y). Different imputation algorithm than Nelson 1996 NIPALS-missing; same goal.
:::

### Benchmarks

Adaptive wall-clock per cell measured against [`full_matrix.csv`](../benchmarks/overview.md). Only backends that implement this method are listed; libraries without the method are omitted.

**Verdict** &nbsp;·&nbsp; ✓ ref / ≈ ref / ~ shape mark a reference-gate pass at strict / relaxed / qualitative tolerance &nbsp;·&nbsp; ✓ bind = pls4all binding agrees with the C++ baseline &nbsp;·&nbsp; ✗ divergent &nbsp;·&nbsp; ⚠ error &nbsp;·&nbsp; — not run. The fastest backend per column is marked 🏆.

**Reference gate**: qualitative — shape/smoke comparison only. The external library and pls4all do not produce numerically equivalent output for this method (see the MethodSpec notes); the `rmse_rel_tol ≤ 1e+01` budget is set wide on purpose. Treat ~ shape as *“we ran both, both finished”*, not as numerical agreement.

Rows tagged with **📐** are the canonical parity references for this method (declared in [`parity_timing.registry`](../benchmarks/methodology.md)). C++ and external rows show reference parity; pls4all language bindings show binding parity against the C++ backend. Hover the icon for role and tolerance band.

::::{tab-set}
:class: parity-tabs

:::{tab-item} 1 thread
:sync: threads-1

<div class="parity-table-wrap">
<table class="docutils parity-grouped">
<thead><tr><th scope="col">Backend</th><th scope="col">Parity</th><th class="size-col" scope="col">50×250 (ms)</th><th class="size-col" scope="col">100×50 (ms)</th><th class="size-col" scope="col">100×500 (ms)</th><th class="size-col" scope="col">100×2500 (ms)</th><th class="size-col" scope="col">200×30 (ms)</th><th class="size-col" scope="col">250×50 (ms)</th><th class="size-col" scope="col">500×50 (ms)</th><th class="size-col" scope="col">500×500 (ms)</th><th class="size-col" scope="col">500×2500 (ms)</th><th class="size-col" scope="col">2500×50 (ms)</th><th class="size-col" scope="col">2500×500 (ms)</th><th class="size-col" scope="col">2500×2500 (ms)</th><th class="size-col" scope="col">10000×50 (ms)</th><th class="size-col" scope="col">10000×500 (ms)</th></tr></thead>
<tbody class="lang-band lang-cpp"><tr class="lang-band-row" data-lang="cpp"><th colspan="16" scope="rowgroup"><span class="lang-band-dot"></span>C++ native · libn4m</th></tr>
<tr class="bk-row"><td class="bk-name"><code>pls4all.cpp.blas</code></td><td class="parity parity-drift">≈ +6e-14</td><td class="ms">2.97 ms</td><td class="ms">2.06 ms</td><td class="ms">12.5 ms</td><td class="ms">68.0 ms</td><td class="ms">1.24 ms</td><td class="ms">2.93 ms</td><td class="ms">6.60 ms</td><td class="ms ms-best">62.0 ms<span class="medal" title="fastest">🏆</span></td><td class="ms">333.1 ms</td><td class="ms">33.9 ms</td><td class="ms">350.2 ms</td><td class="ms ms-best">1.6 s<span class="medal" title="fastest">🏆</span></td><td class="ms ms-best">113.5 ms<span class="medal" title="fastest">🏆</span></td><td class="ms">1.4 s</td></tr>
<tr class="bk-row"><td class="bk-name"><code>pls4all.cpp.blas+omp</code></td><td class="parity parity-drift">≈ +6e-14</td><td class="ms">3.28 ms</td><td class="ms">1.90 ms</td><td class="ms ms-best">11.4 ms<span class="medal" title="fastest">🏆</span></td><td class="ms ms-best">61.7 ms<span class="medal" title="fastest">🏆</span></td><td class="ms">1.99 ms</td><td class="ms ms-best">2.55 ms<span class="medal" title="fastest">🏆</span></td><td class="ms ms-best">5.05 ms<span class="medal" title="fastest">🏆</span></td><td class="ms">65.7 ms</td><td class="ms">339.3 ms</td><td class="ms">34.7 ms</td><td class="ms ms-best">346.6 ms<span class="medal" title="fastest">🏆</span></td><td class="ms">1.7 s</td><td class="ms">115.0 ms</td><td class="ms">1.5 s</td></tr>
<tr class="bk-row"><td class="bk-name"><code>pls4all.cpp.omp</code></td><td class="parity parity-drift">≈ +6e-14</td><td class="ms">2.83 ms</td><td class="ms">1.68 ms</td><td class="ms">12.4 ms</td><td class="ms">64.4 ms</td><td class="ms ms-best">1.21 ms<span class="medal" title="fastest">🏆</span></td><td class="ms">2.59 ms</td><td class="ms">7.98 ms</td><td class="ms">66.2 ms</td><td class="ms ms-best">330.8 ms<span class="medal" title="fastest">🏆</span></td><td class="ms ms-best">32.8 ms<span class="medal" title="fastest">🏆</span></td><td class="ms">367.1 ms</td><td class="ms">1.7 s</td><td class="ms">113.7 ms</td><td class="ms">1.4 s</td></tr>
<tr class="bk-row"><td class="bk-name"><code>pls4all.cpp.ref</code></td><td class="parity parity-drift">≈ +6e-14</td><td class="ms ms-best">2.60 ms<span class="medal" title="fastest">🏆</span></td><td class="ms ms-best">1.34 ms<span class="medal" title="fastest">🏆</span></td><td class="ms">13.5 ms</td><td class="ms">64.6 ms</td><td class="ms">1.25 ms</td><td class="ms">2.64 ms</td><td class="ms">7.16 ms</td><td class="ms">67.4 ms</td><td class="ms">336.3 ms</td><td class="ms">33.9 ms</td><td class="ms">350.5 ms</td><td class="ms">1.7 s</td><td class="ms">114.0 ms</td><td class="ms ms-best">1.4 s<span class="medal" title="fastest">🏆</span></td></tr>
</tbody>
<tbody class="lang-band lang-python"><tr class="lang-band-row" data-lang="python"><th colspan="16" scope="rowgroup"><span class="lang-band-dot"></span>Python · pls4all</th></tr>
<tr class="bk-row"><td class="bk-name"><code>pls4all.python</code></td><td class="parity parity-exact">✓ bind</td><td class="ms">2.72 ms</td><td class="ms ms-empty">—</td><td class="ms ms-empty">—</td><td class="ms ms-empty">—</td><td class="ms">1.37 ms</td><td class="ms">2.88 ms</td><td class="ms ms-empty">—</td><td class="ms ms-empty">—</td><td class="ms ms-empty">—</td><td class="ms ms-empty">—</td><td class="ms ms-empty">—</td><td class="ms ms-empty">—</td><td class="ms ms-empty">—</td><td class="ms ms-empty">—</td></tr>
<tr class="bk-row"><td class="bk-name"><code>pls4all.sklearn</code></td><td class="parity parity-exact">✓ bind</td><td class="ms">3.28 ms</td><td class="ms ms-empty">—</td><td class="ms ms-empty">—</td><td class="ms ms-empty">—</td><td class="ms">1.45 ms</td><td class="ms">3.06 ms</td><td class="ms ms-empty">—</td><td class="ms ms-empty">—</td><td class="ms ms-empty">—</td><td class="ms ms-empty">—</td><td class="ms ms-empty">—</td><td class="ms ms-empty">—</td><td class="ms ms-empty">—</td><td class="ms ms-empty">—</td></tr>
</tbody>
<tbody class="lang-band lang-r"><tr class="lang-band-row" data-lang="r"><th colspan="16" scope="rowgroup"><span class="lang-band-dot"></span>R · pls4all</th></tr>
<tr class="bk-row"><td class="bk-name"><code>pls4all.R</code></td><td class="parity parity-exact">✓ bind</td><td class="ms">13.8 ms</td><td class="ms ms-empty">—</td><td class="ms ms-empty">—</td><td class="ms ms-empty">—</td><td class="ms">3.97 ms</td><td class="ms">13.7 ms</td><td class="ms ms-empty">—</td><td class="ms ms-empty">—</td><td class="ms ms-empty">—</td><td class="ms ms-empty">—</td><td class="ms ms-empty">—</td><td class="ms ms-empty">—</td><td class="ms ms-empty">—</td><td class="ms ms-empty">—</td></tr>
<tr class="bk-row"><td class="bk-name"><code>pls4all.R.formula</code></td><td class="parity parity-exact">✓ bind</td><td class="ms">24.3 ms</td><td class="ms ms-empty">—</td><td class="ms ms-empty">—</td><td class="ms ms-empty">—</td><td class="ms">5.26 ms</td><td class="ms">9.00 ms</td><td class="ms ms-empty">—</td><td class="ms ms-empty">—</td><td class="ms ms-empty">—</td><td class="ms ms-empty">—</td><td class="ms ms-empty">—</td><td class="ms ms-empty">—</td><td class="ms ms-empty">—</td><td class="ms ms-empty">—</td></tr>
<tr class="bk-row"><td class="bk-name"><code>pls4all.R.mdatools</code></td><td class="parity parity-exact">✓ bind</td><td class="ms">25.5 ms</td><td class="ms ms-empty">—</td><td class="ms ms-empty">—</td><td class="ms ms-empty">—</td><td class="ms">5.14 ms</td><td class="ms">12.4 ms</td><td class="ms ms-empty">—</td><td class="ms ms-empty">—</td><td class="ms ms-empty">—</td><td class="ms ms-empty">—</td><td class="ms ms-empty">—</td><td class="ms ms-empty">—</td><td class="ms ms-empty">—</td><td class="ms ms-empty">—</td></tr>
<tr class="bk-row"><td class="bk-name"><code>pls4all.R.pls</code></td><td class="parity parity-exact">✓ bind</td><td class="ms">29.8 ms</td><td class="ms ms-empty">—</td><td class="ms ms-empty">—</td><td class="ms ms-empty">—</td><td class="ms">6.02 ms</td><td class="ms">11.4 ms</td><td class="ms ms-empty">—</td><td class="ms ms-empty">—</td><td class="ms ms-empty">—</td><td class="ms ms-empty">—</td><td class="ms ms-empty">—</td><td class="ms ms-empty">—</td><td class="ms ms-empty">—</td><td class="ms ms-empty">—</td></tr>
</tbody>
<tbody class="lang-band lang-matlab"><tr class="lang-band-row" data-lang="matlab"><th colspan="16" scope="rowgroup"><span class="lang-band-dot"></span>MATLAB · pls4all</th></tr>
<tr class="bk-row"><td class="bk-name"><code>pls4all.matlab</code></td><td class="parity parity-divergent">✗ +9e+00</td><td class="ms">4.33 ms</td><td class="ms ms-empty">—</td><td class="ms ms-empty">—</td><td class="ms ms-empty">—</td><td class="ms">2.17 ms</td><td class="ms">4.50 ms</td><td class="ms ms-empty">—</td><td class="ms ms-empty">—</td><td class="ms ms-empty">—</td><td class="ms ms-empty">—</td><td class="ms ms-empty">—</td><td class="ms ms-empty">—</td><td class="ms ms-empty">—</td><td class="ms ms-empty">—</td></tr>
<tr class="bk-row"><td class="bk-name"><code>pls4all.matlab.classdef</code></td><td class="parity parity-divergent">✗ +9e+00</td><td class="ms">6.78 ms</td><td class="ms ms-empty">—</td><td class="ms ms-empty">—</td><td class="ms ms-empty">—</td><td class="ms">2.38 ms</td><td class="ms">5.90 ms</td><td class="ms ms-empty">—</td><td class="ms ms-empty">—</td><td class="ms ms-empty">—</td><td class="ms ms-empty">—</td><td class="ms ms-empty">—</td><td class="ms ms-empty">—</td><td class="ms ms-empty">—</td><td class="ms ms-empty">—</td></tr>
</tbody>
<tbody class="lang-band lang-r"><tr class="lang-band-row" data-lang="r"><th colspan="16" scope="rowgroup"><span class="lang-band-dot"></span>R · external</th></tr>
<tr class="bk-row truth-source truth-source-qualitative"><td class="bk-name"><span class="truth-mark" title="Registry parity reference (r): softImpute 1.4-1 — qualitative (rmse_rel ≤ 1e+01)">📐</span><code>ref.r_softimpute</code></td><td class="parity parity-ref-source">source</td><td class="ms">32.7 ms</td><td class="ms ms-empty">—</td><td class="ms ms-empty">—</td><td class="ms ms-empty">—</td><td class="ms">16.7 ms</td><td class="ms">20.2 ms</td><td class="ms ms-empty">—</td><td class="ms ms-empty">—</td><td class="ms ms-empty">—</td><td class="ms ms-empty">—</td><td class="ms ms-empty">—</td><td class="ms ms-empty">—</td><td class="ms ms-empty">—</td><td class="ms ms-empty">—</td></tr>
</tbody>
</table>
</div>

:::

:::{tab-item} 3 threads
:sync: threads-3

<div class="parity-table-wrap">
<table class="docutils parity-grouped">
<thead><tr><th scope="col">Backend</th><th scope="col">Parity</th><th class="size-col" scope="col">50×250 (ms)</th><th class="size-col" scope="col">100×50 (ms)</th><th class="size-col" scope="col">100×500 (ms)</th><th class="size-col" scope="col">100×2500 (ms)</th><th class="size-col" scope="col">200×30 (ms)</th><th class="size-col" scope="col">250×50 (ms)</th><th class="size-col" scope="col">500×50 (ms)</th><th class="size-col" scope="col">500×500 (ms)</th><th class="size-col" scope="col">500×2500 (ms)</th><th class="size-col" scope="col">2500×50 (ms)</th><th class="size-col" scope="col">2500×500 (ms)</th><th class="size-col" scope="col">2500×2500 (ms)</th><th class="size-col" scope="col">10000×50 (ms)</th><th class="size-col" scope="col">10000×500 (ms)</th></tr></thead>
<tbody class="lang-band lang-cpp"><tr class="lang-band-row" data-lang="cpp"><th colspan="16" scope="rowgroup"><span class="lang-band-dot"></span>C++ native · libn4m</th></tr>
<tr class="bk-row"><td class="bk-name"><code>pls4all.cpp.blas</code></td><td class="parity parity-ref-qualitative">~ shape 8e-15</td><td class="ms ms-empty">—</td><td class="ms ms-empty">—</td><td class="ms ms-empty">—</td><td class="ms ms-empty">—</td><td class="ms">1.63 ms</td><td class="ms ms-empty">—</td><td class="ms ms-empty">—</td><td class="ms ms-empty">—</td><td class="ms ms-empty">—</td><td class="ms ms-empty">—</td><td class="ms ms-empty">—</td><td class="ms ms-empty">—</td><td class="ms ms-empty">—</td><td class="ms ms-empty">—</td></tr>
<tr class="bk-row"><td class="bk-name"><code>pls4all.cpp.blas+omp</code></td><td class="parity parity-ref-qualitative">~ shape 8e-15</td><td class="ms ms-empty">—</td><td class="ms ms-empty">—</td><td class="ms ms-empty">—</td><td class="ms ms-empty">—</td><td class="ms">1.20 ms</td><td class="ms ms-empty">—</td><td class="ms ms-empty">—</td><td class="ms ms-empty">—</td><td class="ms ms-empty">—</td><td class="ms ms-empty">—</td><td class="ms ms-empty">—</td><td class="ms ms-empty">—</td><td class="ms ms-empty">—</td><td class="ms ms-empty">—</td></tr>
<tr class="bk-row"><td class="bk-name"><code>pls4all.cpp.omp</code></td><td class="parity parity-ref-qualitative">~ shape 8e-15</td><td class="ms ms-empty">—</td><td class="ms ms-empty">—</td><td class="ms ms-empty">—</td><td class="ms ms-empty">—</td><td class="ms">1.90 ms</td><td class="ms ms-empty">—</td><td class="ms ms-empty">—</td><td class="ms ms-empty">—</td><td class="ms ms-empty">—</td><td class="ms ms-empty">—</td><td class="ms ms-empty">—</td><td class="ms ms-empty">—</td><td class="ms ms-empty">—</td><td class="ms ms-empty">—</td></tr>
<tr class="bk-row"><td class="bk-name"><code>pls4all.cpp.ref</code></td><td class="parity parity-ref-qualitative">~ shape 8e-15</td><td class="ms ms-empty">—</td><td class="ms ms-empty">—</td><td class="ms ms-empty">—</td><td class="ms ms-empty">—</td><td class="ms ms-best">1.18 ms<span class="medal" title="fastest">🏆</span></td><td class="ms ms-empty">—</td><td class="ms ms-empty">—</td><td class="ms ms-empty">—</td><td class="ms ms-empty">—</td><td class="ms ms-empty">—</td><td class="ms ms-empty">—</td><td class="ms ms-empty">—</td><td class="ms ms-empty">—</td><td class="ms ms-empty">—</td></tr>
</tbody>
<tbody class="lang-band lang-python"><tr class="lang-band-row" data-lang="python"><th colspan="16" scope="rowgroup"><span class="lang-band-dot"></span>Python · pls4all</th></tr>
<tr class="bk-row"><td class="bk-name"><code>pls4all.python</code></td><td class="parity parity-exact">✓ bind</td><td class="ms ms-empty">—</td><td class="ms ms-empty">—</td><td class="ms ms-empty">—</td><td class="ms ms-empty">—</td><td class="ms">1.83 ms</td><td class="ms ms-empty">—</td><td class="ms ms-empty">—</td><td class="ms ms-empty">—</td><td class="ms ms-empty">—</td><td class="ms ms-empty">—</td><td class="ms ms-empty">—</td><td class="ms ms-empty">—</td><td class="ms ms-empty">—</td><td class="ms ms-empty">—</td></tr>
<tr class="bk-row"><td class="bk-name"><code>pls4all.sklearn</code></td><td class="parity parity-exact">✓ bind</td><td class="ms ms-empty">—</td><td class="ms ms-empty">—</td><td class="ms ms-empty">—</td><td class="ms ms-empty">—</td><td class="ms">1.46 ms</td><td class="ms ms-empty">—</td><td class="ms ms-empty">—</td><td class="ms ms-empty">—</td><td class="ms ms-empty">—</td><td class="ms ms-empty">—</td><td class="ms ms-empty">—</td><td class="ms ms-empty">—</td><td class="ms ms-empty">—</td><td class="ms ms-empty">—</td></tr>
</tbody>
<tbody class="lang-band lang-r"><tr class="lang-band-row" data-lang="r"><th colspan="16" scope="rowgroup"><span class="lang-band-dot"></span>R · pls4all</th></tr>
<tr class="bk-row"><td class="bk-name"><code>pls4all.R</code></td><td class="parity parity-exact">✓ bind</td><td class="ms ms-empty">—</td><td class="ms ms-empty">—</td><td class="ms ms-empty">—</td><td class="ms ms-empty">—</td><td class="ms">4.37 ms</td><td class="ms ms-empty">—</td><td class="ms ms-empty">—</td><td class="ms ms-empty">—</td><td class="ms ms-empty">—</td><td class="ms ms-empty">—</td><td class="ms ms-empty">—</td><td class="ms ms-empty">—</td><td class="ms ms-empty">—</td><td class="ms ms-empty">—</td></tr>
<tr class="bk-row"><td class="bk-name"><code>pls4all.R.formula</code></td><td class="parity parity-exact">✓ bind</td><td class="ms ms-empty">—</td><td class="ms ms-empty">—</td><td class="ms ms-empty">—</td><td class="ms ms-empty">—</td><td class="ms">5.03 ms</td><td class="ms ms-empty">—</td><td class="ms ms-empty">—</td><td class="ms ms-empty">—</td><td class="ms ms-empty">—</td><td class="ms ms-empty">—</td><td class="ms ms-empty">—</td><td class="ms ms-empty">—</td><td class="ms ms-empty">—</td><td class="ms ms-empty">—</td></tr>
<tr class="bk-row"><td class="bk-name"><code>pls4all.R.mdatools</code></td><td class="parity parity-exact">✓ bind</td><td class="ms ms-empty">—</td><td class="ms ms-empty">—</td><td class="ms ms-empty">—</td><td class="ms ms-empty">—</td><td class="ms">4.40 ms</td><td class="ms ms-empty">—</td><td class="ms ms-empty">—</td><td class="ms ms-empty">—</td><td class="ms ms-empty">—</td><td class="ms ms-empty">—</td><td class="ms ms-empty">—</td><td class="ms ms-empty">—</td><td class="ms ms-empty">—</td><td class="ms ms-empty">—</td></tr>
<tr class="bk-row"><td class="bk-name"><code>pls4all.R.pls</code></td><td class="parity parity-exact">✓ bind</td><td class="ms ms-empty">—</td><td class="ms ms-empty">—</td><td class="ms ms-empty">—</td><td class="ms ms-empty">—</td><td class="ms">5.04 ms</td><td class="ms ms-empty">—</td><td class="ms ms-empty">—</td><td class="ms ms-empty">—</td><td class="ms ms-empty">—</td><td class="ms ms-empty">—</td><td class="ms ms-empty">—</td><td class="ms ms-empty">—</td><td class="ms ms-empty">—</td><td class="ms ms-empty">—</td></tr>
</tbody>
<tbody class="lang-band lang-matlab"><tr class="lang-band-row" data-lang="matlab"><th colspan="16" scope="rowgroup"><span class="lang-band-dot"></span>MATLAB · pls4all</th></tr>
<tr class="bk-row"><td class="bk-name"><code>pls4all.matlab</code></td><td class="parity parity-divergent">✗ +9e+00</td><td class="ms ms-empty">—</td><td class="ms ms-empty">—</td><td class="ms ms-empty">—</td><td class="ms ms-empty">—</td><td class="ms">1.94 ms</td><td class="ms ms-empty">—</td><td class="ms ms-empty">—</td><td class="ms ms-empty">—</td><td class="ms ms-empty">—</td><td class="ms ms-empty">—</td><td class="ms ms-empty">—</td><td class="ms ms-empty">—</td><td class="ms ms-empty">—</td><td class="ms ms-empty">—</td></tr>
<tr class="bk-row"><td class="bk-name"><code>pls4all.matlab.classdef</code></td><td class="parity parity-divergent">✗ +9e+00</td><td class="ms ms-empty">—</td><td class="ms ms-empty">—</td><td class="ms ms-empty">—</td><td class="ms ms-empty">—</td><td class="ms">3.98 ms</td><td class="ms ms-empty">—</td><td class="ms ms-empty">—</td><td class="ms ms-empty">—</td><td class="ms ms-empty">—</td><td class="ms ms-empty">—</td><td class="ms ms-empty">—</td><td class="ms ms-empty">—</td><td class="ms ms-empty">—</td><td class="ms ms-empty">—</td></tr>
</tbody>
<tbody class="lang-band lang-r"><tr class="lang-band-row" data-lang="r"><th colspan="16" scope="rowgroup"><span class="lang-band-dot"></span>R · external</th></tr>
<tr class="bk-row truth-source truth-source-qualitative"><td class="bk-name"><span class="truth-mark" title="Registry parity reference (r): softImpute 1.4-1 — qualitative (rmse_rel ≤ 1e+01)">📐</span><code>ref.r_softimpute</code></td><td class="parity parity-ref-source">source</td><td class="ms ms-empty">—</td><td class="ms ms-empty">—</td><td class="ms ms-empty">—</td><td class="ms ms-empty">—</td><td class="ms">17.8 ms</td><td class="ms ms-empty">—</td><td class="ms ms-empty">—</td><td class="ms ms-empty">—</td><td class="ms ms-empty">—</td><td class="ms ms-empty">—</td><td class="ms ms-empty">—</td><td class="ms ms-empty">—</td><td class="ms ms-empty">—</td><td class="ms ms-empty">—</td></tr>
</tbody>
</table>
</div>

:::

:::{tab-item} 10 threads
:sync: threads-10

<div class="parity-table-wrap">
<table class="docutils parity-grouped">
<thead><tr><th scope="col">Backend</th><th scope="col">Parity</th><th class="size-col" scope="col">50×250 (ms)</th><th class="size-col" scope="col">100×50 (ms)</th><th class="size-col" scope="col">100×500 (ms)</th><th class="size-col" scope="col">100×2500 (ms)</th><th class="size-col" scope="col">200×30 (ms)</th><th class="size-col" scope="col">250×50 (ms)</th><th class="size-col" scope="col">500×50 (ms)</th><th class="size-col" scope="col">500×500 (ms)</th><th class="size-col" scope="col">500×2500 (ms)</th><th class="size-col" scope="col">2500×50 (ms)</th><th class="size-col" scope="col">2500×500 (ms)</th><th class="size-col" scope="col">2500×2500 (ms)</th><th class="size-col" scope="col">10000×50 (ms)</th><th class="size-col" scope="col">10000×500 (ms)</th></tr></thead>
<tbody class="lang-band lang-cpp"><tr class="lang-band-row" data-lang="cpp"><th colspan="16" scope="rowgroup"><span class="lang-band-dot"></span>C++ native · libn4m</th></tr>
<tr class="bk-row"><td class="bk-name"><code>pls4all.cpp.blas</code></td><td class="parity parity-ref-qualitative">~ shape 8e-15</td><td class="ms ms-empty">—</td><td class="ms ms-empty">—</td><td class="ms ms-empty">—</td><td class="ms ms-empty">—</td><td class="ms">1.14 ms</td><td class="ms ms-empty">—</td><td class="ms ms-empty">—</td><td class="ms ms-empty">—</td><td class="ms ms-empty">—</td><td class="ms ms-empty">—</td><td class="ms ms-empty">—</td><td class="ms ms-empty">—</td><td class="ms ms-empty">—</td><td class="ms ms-empty">—</td></tr>
<tr class="bk-row"><td class="bk-name"><code>pls4all.cpp.blas+omp</code></td><td class="parity parity-ref-qualitative">~ shape 8e-15</td><td class="ms ms-empty">—</td><td class="ms ms-empty">—</td><td class="ms ms-empty">—</td><td class="ms ms-empty">—</td><td class="ms">2.01 ms</td><td class="ms ms-empty">—</td><td class="ms ms-empty">—</td><td class="ms ms-empty">—</td><td class="ms ms-empty">—</td><td class="ms ms-empty">—</td><td class="ms ms-empty">—</td><td class="ms ms-empty">—</td><td class="ms ms-empty">—</td><td class="ms ms-empty">—</td></tr>
<tr class="bk-row"><td class="bk-name"><code>pls4all.cpp.omp</code></td><td class="parity parity-ref-qualitative">~ shape 8e-15</td><td class="ms ms-empty">—</td><td class="ms ms-empty">—</td><td class="ms ms-empty">—</td><td class="ms ms-empty">—</td><td class="ms ms-best">1.14 ms<span class="medal" title="fastest">🏆</span></td><td class="ms ms-empty">—</td><td class="ms ms-empty">—</td><td class="ms ms-empty">—</td><td class="ms ms-empty">—</td><td class="ms ms-empty">—</td><td class="ms ms-empty">—</td><td class="ms ms-empty">—</td><td class="ms ms-empty">—</td><td class="ms ms-empty">—</td></tr>
<tr class="bk-row"><td class="bk-name"><code>pls4all.cpp.ref</code></td><td class="parity parity-ref-qualitative">~ shape 8e-15</td><td class="ms ms-empty">—</td><td class="ms ms-empty">—</td><td class="ms ms-empty">—</td><td class="ms ms-empty">—</td><td class="ms">1.15 ms</td><td class="ms ms-empty">—</td><td class="ms ms-empty">—</td><td class="ms ms-empty">—</td><td class="ms ms-empty">—</td><td class="ms ms-empty">—</td><td class="ms ms-empty">—</td><td class="ms ms-empty">—</td><td class="ms ms-empty">—</td><td class="ms ms-empty">—</td></tr>
</tbody>
<tbody class="lang-band lang-python"><tr class="lang-band-row" data-lang="python"><th colspan="16" scope="rowgroup"><span class="lang-band-dot"></span>Python · pls4all</th></tr>
<tr class="bk-row"><td class="bk-name"><code>pls4all.python</code></td><td class="parity parity-exact">✓ bind</td><td class="ms ms-empty">—</td><td class="ms ms-empty">—</td><td class="ms ms-empty">—</td><td class="ms ms-empty">—</td><td class="ms">2.09 ms</td><td class="ms ms-empty">—</td><td class="ms ms-empty">—</td><td class="ms ms-empty">—</td><td class="ms ms-empty">—</td><td class="ms ms-empty">—</td><td class="ms ms-empty">—</td><td class="ms ms-empty">—</td><td class="ms ms-empty">—</td><td class="ms ms-empty">—</td></tr>
<tr class="bk-row"><td class="bk-name"><code>pls4all.sklearn</code></td><td class="parity parity-exact">✓ bind</td><td class="ms ms-empty">—</td><td class="ms ms-empty">—</td><td class="ms ms-empty">—</td><td class="ms ms-empty">—</td><td class="ms">1.27 ms</td><td class="ms ms-empty">—</td><td class="ms ms-empty">—</td><td class="ms ms-empty">—</td><td class="ms ms-empty">—</td><td class="ms ms-empty">—</td><td class="ms ms-empty">—</td><td class="ms ms-empty">—</td><td class="ms ms-empty">—</td><td class="ms ms-empty">—</td></tr>
</tbody>
<tbody class="lang-band lang-r"><tr class="lang-band-row" data-lang="r"><th colspan="16" scope="rowgroup"><span class="lang-band-dot"></span>R · pls4all</th></tr>
<tr class="bk-row"><td class="bk-name"><code>pls4all.R</code></td><td class="parity parity-exact">✓ bind</td><td class="ms ms-empty">—</td><td class="ms ms-empty">—</td><td class="ms ms-empty">—</td><td class="ms ms-empty">—</td><td class="ms">2.97 ms</td><td class="ms ms-empty">—</td><td class="ms ms-empty">—</td><td class="ms ms-empty">—</td><td class="ms ms-empty">—</td><td class="ms ms-empty">—</td><td class="ms ms-empty">—</td><td class="ms ms-empty">—</td><td class="ms ms-empty">—</td><td class="ms ms-empty">—</td></tr>
<tr class="bk-row"><td class="bk-name"><code>pls4all.R.formula</code></td><td class="parity parity-exact">✓ bind</td><td class="ms ms-empty">—</td><td class="ms ms-empty">—</td><td class="ms ms-empty">—</td><td class="ms ms-empty">—</td><td class="ms">3.74 ms</td><td class="ms ms-empty">—</td><td class="ms ms-empty">—</td><td class="ms ms-empty">—</td><td class="ms ms-empty">—</td><td class="ms ms-empty">—</td><td class="ms ms-empty">—</td><td class="ms ms-empty">—</td><td class="ms ms-empty">—</td><td class="ms ms-empty">—</td></tr>
<tr class="bk-row"><td class="bk-name"><code>pls4all.R.mdatools</code></td><td class="parity parity-exact">✓ bind</td><td class="ms ms-empty">—</td><td class="ms ms-empty">—</td><td class="ms ms-empty">—</td><td class="ms ms-empty">—</td><td class="ms">3.56 ms</td><td class="ms ms-empty">—</td><td class="ms ms-empty">—</td><td class="ms ms-empty">—</td><td class="ms ms-empty">—</td><td class="ms ms-empty">—</td><td class="ms ms-empty">—</td><td class="ms ms-empty">—</td><td class="ms ms-empty">—</td><td class="ms ms-empty">—</td></tr>
<tr class="bk-row"><td class="bk-name"><code>pls4all.R.pls</code></td><td class="parity parity-exact">✓ bind</td><td class="ms ms-empty">—</td><td class="ms ms-empty">—</td><td class="ms ms-empty">—</td><td class="ms ms-empty">—</td><td class="ms">3.72 ms</td><td class="ms ms-empty">—</td><td class="ms ms-empty">—</td><td class="ms ms-empty">—</td><td class="ms ms-empty">—</td><td class="ms ms-empty">—</td><td class="ms ms-empty">—</td><td class="ms ms-empty">—</td><td class="ms ms-empty">—</td><td class="ms ms-empty">—</td></tr>
</tbody>
<tbody class="lang-band lang-matlab"><tr class="lang-band-row" data-lang="matlab"><th colspan="16" scope="rowgroup"><span class="lang-band-dot"></span>MATLAB · pls4all</th></tr>
<tr class="bk-row"><td class="bk-name"><code>pls4all.matlab</code></td><td class="parity parity-divergent">✗ +9e+00</td><td class="ms ms-empty">—</td><td class="ms ms-empty">—</td><td class="ms ms-empty">—</td><td class="ms ms-empty">—</td><td class="ms">1.87 ms</td><td class="ms ms-empty">—</td><td class="ms ms-empty">—</td><td class="ms ms-empty">—</td><td class="ms ms-empty">—</td><td class="ms ms-empty">—</td><td class="ms ms-empty">—</td><td class="ms ms-empty">—</td><td class="ms ms-empty">—</td><td class="ms ms-empty">—</td></tr>
<tr class="bk-row"><td class="bk-name"><code>pls4all.matlab.classdef</code></td><td class="parity parity-divergent">✗ +9e+00</td><td class="ms ms-empty">—</td><td class="ms ms-empty">—</td><td class="ms ms-empty">—</td><td class="ms ms-empty">—</td><td class="ms">2.17 ms</td><td class="ms ms-empty">—</td><td class="ms ms-empty">—</td><td class="ms ms-empty">—</td><td class="ms ms-empty">—</td><td class="ms ms-empty">—</td><td class="ms ms-empty">—</td><td class="ms ms-empty">—</td><td class="ms ms-empty">—</td><td class="ms ms-empty">—</td></tr>
</tbody>
<tbody class="lang-band lang-r"><tr class="lang-band-row" data-lang="r"><th colspan="16" scope="rowgroup"><span class="lang-band-dot"></span>R · external</th></tr>
<tr class="bk-row truth-source truth-source-qualitative"><td class="bk-name"><span class="truth-mark" title="Registry parity reference (r): softImpute 1.4-1 — qualitative (rmse_rel ≤ 1e+01)">📐</span><code>ref.r_softimpute</code></td><td class="parity parity-ref-source">source</td><td class="ms ms-empty">—</td><td class="ms ms-empty">—</td><td class="ms ms-empty">—</td><td class="ms ms-empty">—</td><td class="ms">10.6 ms</td><td class="ms ms-empty">—</td><td class="ms ms-empty">—</td><td class="ms ms-empty">—</td><td class="ms ms-empty">—</td><td class="ms ms-empty">—</td><td class="ms ms-empty">—</td><td class="ms ms-empty">—</td><td class="ms ms-empty">—</td><td class="ms ms-empty">—</td></tr>
</tbody>
</table>
</div>

:::

::::


---

_See also_: [benchmark overview](../benchmarks/overview.md) · [methods index](index.md) · [interactive dashboard](../landing/dashboard.md)