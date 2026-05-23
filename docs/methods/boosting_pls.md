# `boosting_pls` — Boosting PLS

_Group_: **Ensemble** · _Registry tolerance_: `10.0`

## Description

Boosting PLS (§20)

From the `pls4all.sklearn.BoostingPLSRegression` docstring:

> Boosted PLS regression.

> **Registry note** — R `mboost::glmboost(family=Gaussian())` boosts linear weak learners while pls4all boosts PLS weak learners; same family, different decision surfaces — widened tol.

### Parameters

| Name | Type | Default | Notes |
|------|------|---------|-------|
| `n_components` | `int` | `2` | Number of latent components extracted (k). |
| `n_estimators` | `int` | `50` | Number of base PLS sub-models in the ensemble. |
| `learning_rate` | `float` | `0.1` | Boosting shrinkage applied to each successive base learner. |

## Explanations

### Bibliographic source

Friedman, J. H. (2001). *Greedy function approximation: a gradient boosting machine*. Annals of Statistics 29(5), 1189–1232. — adapted for PLS as a base learner.

### Mathematical principle

Gradient boosting builds an additive predictor $F_M(\mathbf{x}) = \sum_{m=1}^M \eta\, h_m(\mathbf{x})$ where each weak learner $h_m$ is fit on the negative gradient (the residuals, for squared-error loss) of the current ensemble. With PLS as the weak learner, each $h_m$ is a small ($k$-component) PLS fitted on the pseudo-response $r_i^{(m)} = y_i - F_{m-1}(\mathbf{x}_i)$.

The learning rate $\eta$ (typically 0.05–0.1) and the number of boosting iterations $M$ are the key hyperparameters; their product roughly controls the effective number of latent dimensions explored. Because boosting reduces bias, it can recover non-linear $Y$–$X$ relationships even with linear PLS base learners — at the cost of much higher computational cost than a single PLS.

### Implementation

`n4m_boosting_pls_fit`. Reference: CRAN `mboost::glmboost` with a PLS base learner (`mboost 2.9.11`).

R roxygen note (`sklearn_extra.R::boosting_pls`):

> Boosting PLS — formula entry point.

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
n4m_boosting_pls_fit(ctx, cfg, &x_view, &y_view, /* hyperparams */, &res);
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
from pls4all._methods import boosting_pls_fit
with pls4all.Context() as ctx, pls4all.Config() as cfg:
    res = boosting_pls_fit(ctx, cfg, X, y, n_components=4)
# then: res.matrix("predictions"), res.matrix("coefficients"),
# res.vector("mask"), res.scalar("intercept"), …
```

:::

:::{tab-item} Python · pls4all.sklearn
:sync: python-sklearn
:class-label: lang-python

```python
from pls4all.sklearn import BoostingPLSRegression
mdl = BoostingPLSRegression(n_components=2, n_estimators=50, learning_rate=0.1)
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
res <- pls4all_method("boosting_pls", X, y,
                      n_components = 4L, params = list(n_estimators = 10L, learning_rate = 0.1))
# res is a named list with MethodResult arrays/scalars.
# selected_indices / top_k_intervals are 1-based.
```

:::

:::{tab-item} R · pls4all (raw fn)
:sync: r-raw
:class-label: lang-r

```r
library(pls4all)
res  <- boosting_pls_fit(X, Y, n_components,
                  n_estimators = 50L, learning_rate = 0.1)
yhat <- pls4all_predict(res, X_test)
```

:::

:::{tab-item} R · pls4all (formula+S3)
:sync: r-formula
:class-label: lang-r

```r
library(pls4all)
fit  <- boosting_pls(y ~ ., data = train, ncomp = 4L)
yhat <- predict(fit, newdata = test)
summary(fit)
```

:::

:::{tab-item} MATLAB · pls4all (MEX)
:sync: matlab-mex
:class-label: lang-matlab

```matlab
res = pls4all.boosting_pls(X, y, 4);
% see header of bindings/matlab/+pls4all/boosting_pls.m for full
% parameter surface:
%   res = boosting_pls(X, Y, n_components, n_estimators, learning_rate)
yhat = predict(res, Xtest);
```

:::

:::{tab-item} MATLAB · pls4all (classdef)
:sync: matlab-classdef
:class-label: lang-matlab

```matlab
mdl  = pls4all.fit("boosting_pls", X, y, "NumComponents", 4);
yhat = predict(mdl, Xtest);
```

:::

::::


**Registry parity references** 📐

:::{card}
:class-card: external-refs

- 📐 **`ref.r_mboost`** (R · r) — `mboost` 2.9-11 · qualitative (rmse_rel ≤ 1e+01) — R `mboost::glmboost(family=Gaussian())`. pls4all boosts PLS weak learners; mboost boosts linear weak learners. Qualitative parity (same algorithm family).
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
<tr class="bk-row"><td class="bk-name"><code>pls4all.cpp.blas</code></td><td class="parity parity-exact">✓ 5e-01</td><td class="ms">1.44 ms</td></tr>
<tr class="bk-row"><td class="bk-name"><code>pls4all.cpp.blas+omp</code></td><td class="parity parity-exact">✓ 5e-01</td><td class="ms">1.22 ms</td></tr>
<tr class="bk-row"><td class="bk-name"><code>pls4all.cpp.omp</code></td><td class="parity parity-exact">✓ 5e-01</td><td class="ms">1.26 ms</td></tr>
<tr class="bk-row"><td class="bk-name"><code>pls4all.cpp.ref</code></td><td class="parity parity-exact">✓ 5e-01</td><td class="ms ms-best">1.14 ms<span class="medal" title="fastest">🏆</span></td></tr>
</tbody>
<tbody class="lang-band lang-python"><tr class="lang-band-row" data-lang="python"><th colspan="3" scope="rowgroup"><span class="lang-band-dot"></span>Python · pls4all</th></tr>
<tr class="bk-row"><td class="bk-name"><code>pls4all.python</code></td><td class="parity parity-exact">✓ bind</td><td class="ms">1.22 ms</td></tr>
<tr class="bk-row"><td class="bk-name"><code>pls4all.registry</code></td><td class="parity parity-exact">✓ bind</td><td class="ms">1.20 ms</td></tr>
<tr class="bk-row"><td class="bk-name"><code>pls4all.sklearn</code></td><td class="parity parity-exact">✓ bind</td><td class="ms">1.33 ms</td></tr>
</tbody>
<tbody class="lang-band lang-r"><tr class="lang-band-row" data-lang="r"><th colspan="3" scope="rowgroup"><span class="lang-band-dot"></span>R · pls4all</th></tr>
<tr class="bk-row"><td class="bk-name"><code>pls4all.R</code></td><td class="parity parity-exact">✓ bind</td><td class="ms">2.72 ms</td></tr>
<tr class="bk-row"><td class="bk-name"><code>pls4all.R.formula</code></td><td class="parity parity-exact">✓ bind</td><td class="ms">3.98 ms</td></tr>
</tbody>
<tbody class="lang-band lang-matlab"><tr class="lang-band-row" data-lang="matlab"><th colspan="3" scope="rowgroup"><span class="lang-band-dot"></span>MATLAB · pls4all</th></tr>
<tr class="bk-row"><td class="bk-name"><code>pls4all.matlab</code></td><td class="parity parity-exact">✓ bind</td><td class="ms">1.88 ms</td></tr>
<tr class="bk-row"><td class="bk-name"><code>pls4all.matlab.classdef</code></td><td class="parity parity-exact">✓ 2e-15</td><td class="ms">2.32 ms</td></tr>
</tbody>
<tbody class="lang-band lang-python"><tr class="lang-band-row" data-lang="python"><th colspan="3" scope="rowgroup"><span class="lang-band-dot"></span>Python · external</th></tr>
<tr class="bk-row"><td class="bk-name"><code>ikpls</code></td><td class="parity parity-not_available">⊘</td><td class="ms">—</td></tr>
<tr class="bk-row"><td class="bk-name"><code>sklearn</code></td><td class="parity parity-not_available">⊘</td><td class="ms">—</td></tr>
</tbody>
<tbody class="lang-band lang-r"><tr class="lang-band-row" data-lang="r"><th colspan="3" scope="rowgroup"><span class="lang-band-dot"></span>R · external</th></tr>
<tr class="bk-row"><td class="bk-name"><code>mixOmics</code></td><td class="parity parity-not_available">⊘</td><td class="ms">—</td></tr>
<tr class="bk-row"><td class="bk-name"><code>pls</code></td><td class="parity parity-not_available">⊘</td><td class="ms">—</td></tr>
<tr class="bk-row truth-source truth-source-qualitative"><td class="bk-name"><span class="truth-mark" title="Registry parity reference (r): mboost 2.9-11 — qualitative (rmse_rel ≤ 1e+01)">📐</span><code>ref.r_mboost</code></td><td class="parity parity-exact">✓ ref</td><td class="ms">633.3 ms</td></tr>
<tr class="bk-row"><td class="bk-name"><code>ropls</code></td><td class="parity parity-not_available">⊘</td><td class="ms">—</td></tr>
</tbody>
<tbody class="lang-band lang-matlab"><tr class="lang-band-row" data-lang="matlab"><th colspan="3" scope="rowgroup"><span class="lang-band-dot"></span>MATLAB · external</th></tr>
<tr class="bk-row"><td class="bk-name"><code>plsregress</code></td><td class="parity parity-not_run">—</td><td class="ms">—</td></tr>
</tbody>
<tbody class="lang-band lang-ext"><tr class="lang-band-row" data-lang="ext"><th colspan="3" scope="rowgroup"><span class="lang-band-dot"></span>Other</th></tr>
<tr class="bk-row"><td class="bk-name"><code>r_mdatools_compat</code></td><td class="parity parity-exact">✓ bind</td><td class="ms">3.50 ms</td></tr>
<tr class="bk-row"><td class="bk-name"><code>r_pls_compat</code></td><td class="parity parity-exact">✓ bind</td><td class="ms">3.41 ms</td></tr>
</tbody>
</table>
</div>

:::

::::


---

_See also_: [benchmark overview](../benchmarks/overview.md) · [methods index](index.md) · [interactive dashboard](../landing/dashboard.md)