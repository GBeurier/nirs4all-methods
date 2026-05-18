# `lw_pls` — Locally-Weighted PLS (LW-PLS)

_Group_: **Nonlinear / local** · _Registry tolerance_: `5.0`

## Description

LW-PLS — Locally-weighted PLS (§17 Phase 4)

From the `pls4all.sklearn.LWPLSRegression` docstring:

> Locally-weighted PLS (Næs & Centner 1998).

> **Registry note** — In-tree `nirs4all.operators.models.sklearn.lwpls.LWPLS` is the sanctioned external reference. nirs4all weights neighbours via a kernel bandwidth, pls4all uses a k-NN cut — both are valid Naes-Centner LW-PLS variants.

### Parameters

| Name | Type | Default | Notes |
|------|------|---------|-------|
| `n_components` | `int` | `2` | Number of latent components extracted (k). |
| `n_neighbors` | `int` | `30` | Number of training neighbours used for each local prediction (LW-PLS). |

## Explanations

### Bibliographic source

Centner, V. & Massart, D. L. (1998). *Optimisation in locally weighted regression*. Analytical Chemistry 70(19), 4206–4211.

### Mathematical principle

Instead of fitting a single global PLS, LW-PLS refits a *per-prediction-point* local PLS using only the $k$-nearest calibration samples (in $\mathbf{X}$-space distance). This adapts the model to the local geometry around each query point and is effective on calibration sets that span heterogeneous regimes (e.g. a single instrument calibrated across several product classes).

The neighbourhood weight typically combines distance (Gaussian or tricube kernel on the Euclidean / Mahalanobis distance) with the inverse residual variance from a preliminary global fit. The local PLS uses few components (typically 2–4) because the neighbourhood is small.

Prediction cost is $O(n)$ for the neighbour search plus $O(k_{\mathrm{nn}} \cdot p \cdot k_{\mathrm{pls}})$ for the local fit, per query. KD-tree / ball-tree indices accelerate the neighbour search; pls4all uses an exhaustive scan because $p \gg n$ defeats most spatial indices for NIR data anyway.

### Implementation

`p4a_lw_pls_fit`. Reference: sanctioned git-pinned port `nirs4all.operators.models.sklearn.lwpls`.

R roxygen note (`methods_extra.R::lw_pls_fit`):

> Locally-weighted PLS (Næs & Centner 1998).
> @param n_components Integer. Number of latent components.
> @param n_neighbors Method-specific parameter. See the underlying `*_fit()` function for the exact semantics.
> @param X Numeric matrix of predictors (rows = samples, cols = features).
> @param Y Numeric matrix or vector of responses, with one row per sample.
> @export

MATLAB header (`bindings/matlab/+pls4all/lw_pls.m`):

```text
pls4all.lw_pls  Locally-weighted PLS (Næs & Centner 1998).
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
p4a_lw_pls_fit(ctx, cfg, &x_view, &y_view, /* hyperparams */, &res);
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
from pls4all._methods import lw_pls_fit
with pls4all.Context() as ctx, pls4all.Config() as cfg:
    res = lw_pls_fit(ctx, cfg, X, y, n_components=3)
# then: res.matrix("predictions"), res.matrix("coefficients"),
# res.vector("mask"), res.scalar("intercept"), …
```

:::

:::{tab-item} Python · pls4all.sklearn
:sync: python-sklearn
:class-label: lang-python

```python
from pls4all.sklearn import LWPLSRegression
mdl = LWPLSRegression(n_components=2, n_neighbors=30)
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
res <- pls4all_method("lw_pls", X, y,
                      n_components = 3L, params = list(n_neighbors = 30L))
# res is a named list with MethodResult arrays/scalars.
# selected_indices / top_k_intervals are 1-based.
```

:::

:::{tab-item} R · pls4all (raw fn)
:sync: r-raw
:class-label: lang-r

```r
library(pls4all)
res  <- lw_pls_fit(X, Y, n_components, n_neighbors = 30L)
yhat <- pls4all_predict(res, X_test)
```

:::

:::{tab-item} R · parsnip / mlr3
:sync: r-meta
:class-label: lang-r

```r
# parsnip / tidymodels
library(tidymodels)
pls4all::register_parsnip()
spec <- pls_pls4all_reg(num_comp = 3) %>%
    set_engine("pls4all", algorithm = "lw_pls") %>%
    set_mode("regression")
wflow <- workflow() %>% add_model(spec) %>% add_recipe(rec)
fit <- fit(wflow, data = train)

# mlr3
library(mlr3)
pls4all::register_mlr3()
lrn <- lrn("regr.pls4all", algorithm = "lw_pls", n_components = 3L)
lrn$train(task)
```

:::

:::{tab-item} MATLAB · pls4all (MEX)
:sync: matlab-mex
:class-label: lang-matlab

```matlab
res = pls4all.lw_pls(X, y, 3);
% see header of bindings/matlab/+pls4all/lw_pls.m for full
% parameter surface:
%   res = lw_pls(X, Y, n_components, n_neighbors)
yhat = predict(res, Xtest);
```

:::

:::{tab-item} MATLAB · pls4all (classdef)
:sync: matlab-classdef
:class-label: lang-matlab

_No idiomatic classdef wrapper — invoke `pls4all.fit("lw_pls", X, y, …)` directly from the unified MEX factory._

:::

::::


**Registry parity references** 📐

:::{card}
:class-card: external-refs

- 📐 **`ref.python_nirs4all_operators_models_sklearn_lwpls`** (python · python) — `nirs4all.operators.models.sklearn.lwpls` in-tree · qualitative (rmse_rel ≤ 5e+00) — In-tree Python LW-PLS (sanctioned external reference). Locally-weighted PLS (Naes 1990 / Centner 1998). The kernel-bandwidth (`lambda_in_similarity`) on the nirs4all side controls neighbour weighting differently from the k-NN cut-off used by pls4all — parity is qualitative.
:::

### Benchmarks

Median wall-clock per cell from [`benchmarks/cross_binding/results/full_matrix.csv`](../benchmarks/overview.md). Verdict legend: ✓ exact · ≈ drift · ✗ divergent · ⊘ not available in lib · — not run · ⚠ error. The fastest backend per column is marked with a 🏆 medal. Rows tagged with **📐** are *also* declared in [`benchmarks/parity_timing/registry.py`](../benchmarks/methodology.md) as the canonical parity references for this method (`python_reference` / `r_reference` / `extra_references`). Cell parity in this table is measured against the cross-binding reference (`pls4all.cpp` blas-omp, 1 thread); the 📐 icon points at the *library-of-record* the parity gate ultimately answers to. Hover the icon to see the role and tolerance band.

::::{tab-set}
:class: parity-tabs

:::{tab-item} 1 thread
:sync: threads-1

<div class="parity-table-wrap">
<table class="docutils parity-grouped">
<thead><tr><th scope="col">Backend</th><th scope="col">Parity</th><th class="size-col" scope="col">100×50 (ms)</th><th class="size-col" scope="col">200×40 (ms)</th><th class="size-col" scope="col">500×50 (ms)</th></tr></thead>
<tbody class="lang-band lang-cpp"><tr class="lang-band-row" data-lang="cpp"><th colspan="5" scope="rowgroup"><span class="lang-band-dot"></span>C++ native · libp4a</th></tr>
<tr class="bk-row"><td class="bk-name"><code>pls4all.cpp.blas</code></td><td class="parity parity-drift">≈</td><td class="ms">2.46 ms</td><td class="ms ms-empty">—</td><td class="ms">21.7 ms</td></tr>
<tr class="bk-row"><td class="bk-name"><code>pls4all.cpp.blas+omp</code></td><td class="parity parity-drift">≈</td><td class="ms">2.31 ms</td><td class="ms ms-empty">—</td><td class="ms">22.0 ms</td></tr>
<tr class="bk-row"><td class="bk-name"><code>pls4all.cpp.omp</code></td><td class="parity parity-drift">≈</td><td class="ms ms-best">2.25 ms<span class="medal" title="fastest">🏆</span></td><td class="ms ms-empty">—</td><td class="ms">22.5 ms</td></tr>
<tr class="bk-row"><td class="bk-name"><code>pls4all.cpp.ref</code></td><td class="parity parity-drift">≈</td><td class="ms">2.43 ms</td><td class="ms ms-empty">—</td><td class="ms ms-best">18.8 ms<span class="medal" title="fastest">🏆</span></td></tr>
</tbody>
<tbody class="lang-band lang-python"><tr class="lang-band-row" data-lang="python"><th colspan="5" scope="rowgroup"><span class="lang-band-dot"></span>Python · pls4all</th></tr>
<tr class="bk-row"><td class="bk-name"><code>pls4all.python</code></td><td class="parity parity-drift">≈</td><td class="ms">2.72 ms</td><td class="ms ms-empty">—</td><td class="ms">22.2 ms</td></tr>
<tr class="bk-row"><td class="bk-name"><code>pls4all.registry</code></td><td class="parity parity-drift">≈</td><td class="ms">2.53 ms</td><td class="ms ms-best">170.1 ms<span class="medal" title="fastest">🏆</span></td><td class="ms">20.3 ms</td></tr>
<tr class="bk-row"><td class="bk-name"><code>pls4all.sklearn</code></td><td class="parity parity-not_available">⊘</td><td class="ms">—</td><td class="ms ms-empty">—</td><td class="ms">—</td></tr>
</tbody>
<tbody class="lang-band lang-r"><tr class="lang-band-row" data-lang="r"><th colspan="5" scope="rowgroup"><span class="lang-band-dot"></span>R · pls4all</th></tr>
<tr class="bk-row"><td class="bk-name"><code>pls4all.R</code></td><td class="parity parity-drift">≈</td><td class="ms">4.50 ms</td><td class="ms ms-empty">—</td><td class="ms">34.5 ms</td></tr>
<tr class="bk-row"><td class="bk-name"><code>pls4all.R.formula</code></td><td class="parity parity-not_run">—</td><td class="ms">—</td><td class="ms ms-empty">—</td><td class="ms">—</td></tr>
</tbody>
<tbody class="lang-band lang-matlab"><tr class="lang-band-row" data-lang="matlab"><th colspan="5" scope="rowgroup"><span class="lang-band-dot"></span>MATLAB · pls4all</th></tr>
<tr class="bk-row"><td class="bk-name"><code>pls4all.matlab</code></td><td class="parity parity-not_run">—</td><td class="ms">—</td><td class="ms ms-empty">—</td><td class="ms">—</td></tr>
<tr class="bk-row"><td class="bk-name"><code>pls4all.matlab.classdef</code></td><td class="parity parity-not_run">—</td><td class="ms">—</td><td class="ms ms-empty">—</td><td class="ms">—</td></tr>
</tbody>
<tbody class="lang-band lang-python"><tr class="lang-band-row" data-lang="python"><th colspan="5" scope="rowgroup"><span class="lang-band-dot"></span>Python · external</th></tr>
<tr class="bk-row"><td class="bk-name"><code>ikpls</code></td><td class="parity parity-not_available">⊘</td><td class="ms">—</td><td class="ms ms-empty">—</td><td class="ms">—</td></tr>
<tr class="bk-row truth-source truth-source-qualitative"><td class="bk-name"><span class="truth-mark" title="Registry parity reference (python): nirs4all.operators.models.sklearn.lwpls in-tree — qualitative (rmse_rel ≤ 5e+00)">📐</span><code>ref.python_nirs4all_operators_models_sklearn_lwpls</code></td><td class="parity parity-drift">≈</td><td class="ms">14.4 ms</td><td class="ms">596.1 ms</td><td class="ms">159.4 ms</td></tr>
<tr class="bk-row"><td class="bk-name"><code>sklearn</code></td><td class="parity parity-not_available">⊘</td><td class="ms">—</td><td class="ms ms-empty">—</td><td class="ms">—</td></tr>
</tbody>
<tbody class="lang-band lang-r"><tr class="lang-band-row" data-lang="r"><th colspan="5" scope="rowgroup"><span class="lang-band-dot"></span>R · external</th></tr>
<tr class="bk-row"><td class="bk-name"><code>mixOmics</code></td><td class="parity parity-not_available">⊘</td><td class="ms">—</td><td class="ms ms-empty">—</td><td class="ms">—</td></tr>
<tr class="bk-row"><td class="bk-name"><code>pls</code></td><td class="parity parity-not_available">⊘</td><td class="ms">—</td><td class="ms ms-empty">—</td><td class="ms">—</td></tr>
<tr class="bk-row"><td class="bk-name"><code>ropls</code></td><td class="parity parity-not_available">⊘</td><td class="ms">—</td><td class="ms ms-empty">—</td><td class="ms">—</td></tr>
</tbody>
<tbody class="lang-band lang-matlab"><tr class="lang-band-row" data-lang="matlab"><th colspan="5" scope="rowgroup"><span class="lang-band-dot"></span>MATLAB · external</th></tr>
<tr class="bk-row"><td class="bk-name"><code>plsregress</code></td><td class="parity parity-not_run">—</td><td class="ms">—</td><td class="ms ms-empty">—</td><td class="ms">—</td></tr>
</tbody>
</table>
</div>

:::

::::


---

_See also_: [benchmark overview](../benchmarks/overview.md) · [methods index](index.md) · [interactive dashboard](../landing/dashboard.md)