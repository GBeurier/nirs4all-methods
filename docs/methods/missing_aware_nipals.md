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

R roxygen note (`sklearn_extra.R::missing_aware_nipals`):

> Missing-aware NIPALS — formula entry point.

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

:::{tab-item} R · pls4all (raw fn)
:sync: r-raw
:class-label: lang-r

```r
library(pls4all)
res  <- missing_aware_nipals_fit(X, Y, n_components)
yhat <- pls4all_predict(res, X_test)
```

:::

:::{tab-item} R · pls4all (formula+S3)
:sync: r-formula
:class-label: lang-r

```r
library(pls4all)
fit  <- missing_aware_nipals(y ~ ., data = train, ncomp = 4L)
yhat <- predict(fit, newdata = test)
summary(fit)
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

Median wall-clock per cell from [`benchmarks/cross_binding/results/full_matrix.csv`](../benchmarks/overview.md). Verdict legend: ✓ exact · ≈ drift · ✗ divergent · ⊘ not available in lib · — not run · ⚠ error. The fastest backend per column is marked with a 🏆 medal. Rows tagged with **📐** are *also* declared in [`benchmarks/parity_timing/registry.py`](../benchmarks/methodology.md) as the canonical parity references for this method (`python_reference` / `r_reference` / `extra_references`). C++ and external rows show reference parity; pls4all language bindings show binding parity against the C++ backend. The 📐 icon points at the *library-of-record* the parity gate ultimately answers to. Hover the icon to see the role and tolerance band.

::::{tab-set}
:class: parity-tabs

:::{tab-item} 1 thread
:sync: threads-1

<div class="parity-table-wrap">
<table class="docutils parity-grouped">
<thead><tr><th scope="col">Backend</th><th scope="col">Parity</th><th class="size-col" scope="col">100×50 (ms)</th></tr></thead>
<tbody class="lang-band lang-cpp"><tr class="lang-band-row" data-lang="cpp"><th colspan="3" scope="rowgroup"><span class="lang-band-dot"></span>C++ native · libn4m</th></tr>
<tr class="bk-row"><td class="bk-name"><code>pls4all.cpp.blas</code></td><td class="parity parity-exact">✓ 7e-15</td><td class="ms">1.06 ms</td></tr>
<tr class="bk-row"><td class="bk-name"><code>pls4all.cpp.blas+omp</code></td><td class="parity parity-exact">✓ 7e-15</td><td class="ms">1.01 ms</td></tr>
<tr class="bk-row"><td class="bk-name"><code>pls4all.cpp.omp</code></td><td class="parity parity-exact">✓ 7e-15</td><td class="ms">1.03 ms</td></tr>
<tr class="bk-row"><td class="bk-name"><code>pls4all.cpp.ref</code></td><td class="parity parity-exact">✓ 7e-15</td><td class="ms">1.05 ms</td></tr>
</tbody>
<tbody class="lang-band lang-python"><tr class="lang-band-row" data-lang="python"><th colspan="3" scope="rowgroup"><span class="lang-band-dot"></span>Python · pls4all</th></tr>
<tr class="bk-row"><td class="bk-name"><code>pls4all.python</code></td><td class="parity parity-exact">✓ bind</td><td class="ms">1.04 ms</td></tr>
<tr class="bk-row"><td class="bk-name"><code>pls4all.registry</code></td><td class="parity parity-exact">✓ bind</td><td class="ms ms-best">0.97 ms<span class="medal" title="fastest">🏆</span></td></tr>
<tr class="bk-row"><td class="bk-name"><code>pls4all.sklearn</code></td><td class="parity parity-exact">✓ bind</td><td class="ms">1.15 ms</td></tr>
</tbody>
<tbody class="lang-band lang-r"><tr class="lang-band-row" data-lang="r"><th colspan="3" scope="rowgroup"><span class="lang-band-dot"></span>R · pls4all</th></tr>
<tr class="bk-row"><td class="bk-name"><code>pls4all.R</code></td><td class="parity parity-exact">✓ bind</td><td class="ms">2.66 ms</td></tr>
<tr class="bk-row"><td class="bk-name"><code>pls4all.R.formula</code></td><td class="parity parity-exact">✓ bind</td><td class="ms">3.65 ms</td></tr>
</tbody>
<tbody class="lang-band lang-matlab"><tr class="lang-band-row" data-lang="matlab"><th colspan="3" scope="rowgroup"><span class="lang-band-dot"></span>MATLAB · pls4all</th></tr>
<tr class="bk-row"><td class="bk-name"><code>pls4all.matlab</code></td><td class="parity parity-exact">✓ bind</td><td class="ms">1.90 ms</td></tr>
<tr class="bk-row"><td class="bk-name"><code>pls4all.matlab.classdef</code></td><td class="parity parity-exact">✓ 5e-15</td><td class="ms">1.96 ms</td></tr>
</tbody>
<tbody class="lang-band lang-python"><tr class="lang-band-row" data-lang="python"><th colspan="3" scope="rowgroup"><span class="lang-band-dot"></span>Python · external</th></tr>
<tr class="bk-row"><td class="bk-name"><code>ikpls</code></td><td class="parity parity-not_available">⊘</td><td class="ms">—</td></tr>
<tr class="bk-row"><td class="bk-name"><code>sklearn</code></td><td class="parity parity-not_available">⊘</td><td class="ms">—</td></tr>
</tbody>
<tbody class="lang-band lang-r"><tr class="lang-band-row" data-lang="r"><th colspan="3" scope="rowgroup"><span class="lang-band-dot"></span>R · external</th></tr>
<tr class="bk-row"><td class="bk-name"><code>mixOmics</code></td><td class="parity parity-not_available">⊘</td><td class="ms">—</td></tr>
<tr class="bk-row"><td class="bk-name"><code>pls</code></td><td class="parity parity-not_available">⊘</td><td class="ms">—</td></tr>
<tr class="bk-row truth-source truth-source-qualitative"><td class="bk-name"><span class="truth-mark" title="Registry parity reference (r): softImpute 1.4-1 — qualitative (rmse_rel ≤ 1e+01)">📐</span><code>ref.r_softimpute</code></td><td class="parity parity-exact">✓ ref</td><td class="ms">599.1 ms</td></tr>
<tr class="bk-row"><td class="bk-name"><code>ropls</code></td><td class="parity parity-not_available">⊘</td><td class="ms">—</td></tr>
</tbody>
<tbody class="lang-band lang-matlab"><tr class="lang-band-row" data-lang="matlab"><th colspan="3" scope="rowgroup"><span class="lang-band-dot"></span>MATLAB · external</th></tr>
<tr class="bk-row"><td class="bk-name"><code>plsregress</code></td><td class="parity parity-not_run">—</td><td class="ms">—</td></tr>
</tbody>
<tbody class="lang-band lang-ext"><tr class="lang-band-row" data-lang="ext"><th colspan="3" scope="rowgroup"><span class="lang-band-dot"></span>Other</th></tr>
<tr class="bk-row"><td class="bk-name"><code>r_mdatools_compat</code></td><td class="parity parity-exact">✓ bind</td><td class="ms">3.66 ms</td></tr>
<tr class="bk-row"><td class="bk-name"><code>r_pls_compat</code></td><td class="parity parity-exact">✓ bind</td><td class="ms">3.49 ms</td></tr>
</tbody>
</table>
</div>

:::

::::


---

_See also_: [benchmark overview](../benchmarks/overview.md) · [methods index](index.md) · [interactive dashboard](../landing/dashboard.md)