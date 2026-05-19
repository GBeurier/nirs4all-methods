# `robust_pls` — Robust PLS (Partial Robust M-regression)

_Group_: **Robust / weighted** · _Registry tolerance_: `2.0`

## Description

Robust PLS (Huber IRLS over weighted SIMPLS)

From the `pls4all.sklearn.RobustPLSRegression` docstring:

> Robust PLS via Huber IRLS over weighted SIMPLS.

> **Registry note** — R `chemometrics::prm` (Serneels et al. 2005) — Partial Robust M-regression. pls4all uses Huber IRLS over weighted SIMPLS; chemometrics uses M-estimator weights on SIMPLS. Same family, different weight function; widened tol.

### Parameters

| Name | Type | Default | Notes |
|------|------|---------|-------|
| `n_components` | `int` | `2` | Number of latent components extracted (k). |
| `huber_k` | `float` | `1.345` | Huber threshold (in residual-stdev units) controlling IRLS reweighting; smaller = more robust. |
| `max_irls_iter` | `int` | `20` | Maximum IRLS reweighting iterations. |

## Explanations

### Bibliographic source

Serneels, S., Croux, C., Filzmoser, P. & Van Espen, P. J. (2005). *Partial Robust M-Regression*. Chemometrics and Intelligent Laboratory Systems 79(1–2), 55–64.

### Mathematical principle

Robust PLS performs a sequence of weighted PLS fits where the weights $w_i$ are reduced for samples with large current residuals — the iteratively-reweighted least-squares (IRLS) recipe.

At each iteration, the weight of sample $i$ is set from Huber's $\psi$-function applied to the standardised residual: $w_i = \psi(r_i / s) / r_i$ where $\psi(z) = z$ for $|z| \le k$ and $\psi(z) = k\,\operatorname{sign}(z)$ otherwise. $k = 1.345$ gives 95 % asymptotic efficiency at the Gaussian. The robust scale $s$ is typically the MAD of the residuals.

Convergence is rapid: 3–5 iterations typically suffice. Robust PLS down-weights — rather than removes — outliers, which is desirable when outlier-ness is a continuous concept (mild spectral artefacts) rather than binary (broken samples).

Compared to median-PLS variants, the M-regression form preserves the analytic structure of SIMPLS and offers smooth weighting; it also generalises to leverage-based weights (PRM with x-weights).

### Implementation

`p4a_robust_pls_fit`. Reference: CRAN `chemometrics::prm` (Serneels et al. authors). The exact weight schedule and scale estimator differ from `prm` so RMSE-rel parity is widened to ~2.0 to flag presence rather than enforce exact agreement.

R roxygen note (`sklearn_extra.R::robust_pls`):

> Robust PLS — formula entry point.

MATLAB header (`bindings/matlab/+pls4all/RobustPlsRegression.m`):

```text
pls4all.RobustPlsRegression  Robust PLS via Huber IRLS.
```

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
p4a_robust_pls_fit(ctx, cfg, &x_view, &y_view, /* hyperparams */, &res);
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
from pls4all._methods import robust_pls_fit
with pls4all.Context() as ctx, pls4all.Config() as cfg:
    res = robust_pls_fit(ctx, cfg, X, y, n_components=4)
# then: res.matrix("predictions"), res.matrix("coefficients"),
# res.vector("mask"), res.scalar("intercept"), …
```

:::

:::{tab-item} Python · pls4all.sklearn
:sync: python-sklearn
:class-label: lang-python

```python
from pls4all.sklearn import RobustPLSRegression
mdl = RobustPLSRegression(n_components=2, huber_k=1.345, max_irls_iter=20)
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
res <- pls4all_method("robust_pls", X, y,
                      n_components = 4L, params = list(huber_k = 1.345, max_irls_iter = 5L))
# res is a named list with MethodResult arrays/scalars.
# selected_indices / top_k_intervals are 1-based.
```

:::

:::{tab-item} R · pls4all (raw fn)
:sync: r-raw
:class-label: lang-r

```r
library(pls4all)
res  <- robust_pls_fit(X, Y, n_components,
                huber_k = 1.345, max_irls_iter = 20L)
yhat <- pls4all_predict(res, X_test)
```

:::

:::{tab-item} R · pls4all (formula+S3)
:sync: r-formula
:class-label: lang-r

```r
library(pls4all)
fit  <- robust_pls(y ~ ., data = train, ncomp = 4L)
yhat <- predict(fit, newdata = test)
summary(fit)
```

:::

:::{tab-item} R · parsnip / mlr3
:sync: r-meta
:class-label: lang-r

```r
# parsnip / tidymodels
library(tidymodels)
pls4all::register_parsnip()
spec <- pls_pls4all_reg(num_comp = 4) %>%
    set_engine("pls4all", algorithm = "robust_pls") %>%
    set_mode("regression")
wflow <- workflow() %>% add_model(spec) %>% add_recipe(rec)
fit <- fit(wflow, data = train)

# mlr3
library(mlr3)
pls4all::register_mlr3()
lrn <- lrn("regr.pls4all", algorithm = "robust_pls", n_components = 4L)
lrn$train(task)
```

:::

:::{tab-item} MATLAB · pls4all (MEX)
:sync: matlab-mex
:class-label: lang-matlab

```matlab
res = pls4all.robust_pls(X, y, 4);
% see header of bindings/matlab/+pls4all/robust_pls.m for full
% parameter surface:
%   res = robust_pls(X, Y, n_components, huber_k, max_irls_iter)
yhat = predict(res, Xtest);
```

:::

:::{tab-item} MATLAB · pls4all (classdef)
:sync: matlab-classdef
:class-label: lang-matlab

```matlab
mdl  = pls4all.fit("robust_pls", X, y, "NumComponents", 4);
yhat = predict(mdl, Xtest);
```

:::

::::


**Registry parity references** 📐

:::{card}
:class-card: external-refs

- 📐 **`ref.r_chemometrics`** (R · r) — `chemometrics` 0.7.x · qualitative (rmse_rel ≤ 2e+00) — R `chemometrics::prm` (Partial Robust M-regression). pls4all uses Huber IRLS over weighted SIMPLS; this is an M-estimator variant from the same family. Predictions diverge by O(0.5).
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
<tr class="bk-row"><td class="bk-name"><code>pls4all.cpp.blas</code></td><td class="parity parity-exact">✓ 6e-02</td><td class="ms ms-best">1.09 ms<span class="medal" title="fastest">🏆</span></td></tr>
<tr class="bk-row"><td class="bk-name"><code>pls4all.cpp.blas+omp</code></td><td class="parity parity-exact">✓ 6e-02</td><td class="ms">1.25 ms</td></tr>
<tr class="bk-row"><td class="bk-name"><code>pls4all.cpp.omp</code></td><td class="parity parity-exact">✓ 6e-02</td><td class="ms">1.25 ms</td></tr>
<tr class="bk-row"><td class="bk-name"><code>pls4all.cpp.ref</code></td><td class="parity parity-exact">✓ 6e-02</td><td class="ms">1.14 ms</td></tr>
</tbody>
<tbody class="lang-band lang-python"><tr class="lang-band-row" data-lang="python"><th colspan="3" scope="rowgroup"><span class="lang-band-dot"></span>Python · pls4all</th></tr>
<tr class="bk-row"><td class="bk-name"><code>pls4all.python</code></td><td class="parity parity-exact">✓ bind</td><td class="ms">1.13 ms</td></tr>
<tr class="bk-row"><td class="bk-name"><code>pls4all.registry</code></td><td class="parity parity-exact">✓ bind</td><td class="ms">1.13 ms</td></tr>
<tr class="bk-row"><td class="bk-name"><code>pls4all.sklearn</code></td><td class="parity parity-exact">✓ bind</td><td class="ms">1.28 ms</td></tr>
</tbody>
<tbody class="lang-band lang-r"><tr class="lang-band-row" data-lang="r"><th colspan="3" scope="rowgroup"><span class="lang-band-dot"></span>R · pls4all</th></tr>
<tr class="bk-row"><td class="bk-name"><code>pls4all.R</code></td><td class="parity parity-exact">✓ bind</td><td class="ms">3.50 ms</td></tr>
<tr class="bk-row"><td class="bk-name"><code>pls4all.R.formula</code></td><td class="parity parity-exact">✓ 3e-15</td><td class="ms">5.50 ms</td></tr>
</tbody>
<tbody class="lang-band lang-matlab"><tr class="lang-band-row" data-lang="matlab"><th colspan="3" scope="rowgroup"><span class="lang-band-dot"></span>MATLAB · pls4all</th></tr>
<tr class="bk-row"><td class="bk-name"><code>pls4all.matlab</code></td><td class="parity parity-exact">✓ bind</td><td class="ms">1.67 ms</td></tr>
<tr class="bk-row"><td class="bk-name"><code>pls4all.matlab.classdef</code></td><td class="parity parity-exact">✓ 3e-15</td><td class="ms">3.60 ms</td></tr>
</tbody>
<tbody class="lang-band lang-python"><tr class="lang-band-row" data-lang="python"><th colspan="3" scope="rowgroup"><span class="lang-band-dot"></span>Python · external</th></tr>
<tr class="bk-row"><td class="bk-name"><code>ikpls</code></td><td class="parity parity-not_available">⊘</td><td class="ms">—</td></tr>
<tr class="bk-row"><td class="bk-name"><code>sklearn</code></td><td class="parity parity-not_available">⊘</td><td class="ms">—</td></tr>
</tbody>
<tbody class="lang-band lang-r"><tr class="lang-band-row" data-lang="r"><th colspan="3" scope="rowgroup"><span class="lang-band-dot"></span>R · external</th></tr>
<tr class="bk-row"><td class="bk-name"><code>mixOmics</code></td><td class="parity parity-not_available">⊘</td><td class="ms">—</td></tr>
<tr class="bk-row"><td class="bk-name"><code>pls</code></td><td class="parity parity-not_available">⊘</td><td class="ms">—</td></tr>
<tr class="bk-row truth-source truth-source-qualitative"><td class="bk-name"><span class="truth-mark" title="Registry parity reference (r): chemometrics 0.7.x — qualitative (rmse_rel ≤ 2e+00)">📐</span><code>ref.r_chemometrics</code></td><td class="parity parity-exact">✓ ref</td><td class="ms">162.0 ms</td></tr>
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