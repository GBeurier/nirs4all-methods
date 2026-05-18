# `sparse_simpls` — Sparse SIMPLS (Chun & Keleş 2010)

_Group_: **Sparse** · _Registry tolerance_: `1.0`

## Description

Sparse SIMPLS with soft-threshold lambda

From the `pls4all.sklearn.SparseSimplsRegression` docstring:

> Sparse SIMPLS with soft-thresholded weights (Chun & Keles 2010).

> **Registry note** — R `spls` 2.3.2 is the canonical Chun & Keles reference. No widely installable Python port for sparse SIMPLS with this exact normalization.

### Parameters

| Name | Type | Default | Notes |
|------|------|---------|-------|
| `n_components` | `int` | `2` | Number of latent components extracted (k). |
| `sparsity_lambda` | `float` | `0.05` | L1 soft-threshold magnitude applied to the PLS weight vectors. |

## Explanations

### Bibliographic source

Chun, H. & Keleş, S. (2010). *Sparse partial least squares regression for simultaneous dimension reduction and variable selection*. JRSS B 72(1), 3–25.

### Mathematical principle

Sparse PLS adds a soft-thresholding step to each SIMPLS loading weight so that the latent direction is supported on only a small subset of features. Mathematically, after the un-thresholded weight $\mathbf{w}$ is computed, we solve $\mathbf{w}^{\star} = \arg\min_{\|\mathbf{c}\|=1} \|\mathbf{c} - \mathbf{w}\|_2^2 + \lambda \|\mathbf{c}\|_1$, which has the closed-form soft-threshold solution $c_j = \operatorname{sign}(w_j)\,(|w_j| - \lambda/2)_+$ followed by re-normalisation.

The penalty $\lambda$ controls sparsity: small $\lambda$ approaches standard PLS, large $\lambda$ zeroes most weights. In high-dimensional ($p \gg n$) spectroscopy or omics data, sparse PLS simultaneously builds the latent predictive direction *and* selects the variables that support it — a much cleaner story than running PLS then thresholding coefficients post-hoc.

The Chun & Keleş formulation differs subtly from the earlier Lê Cao 2008 sPLS (used in mixOmics): Chun & Keleş threshold the un-deflated weight while Lê Cao threshold the deflated weight at each iteration. pls4all implements the Chun & Keleş formulation.

### Implementation

`p4a_sparse_simpls_fit`. Reference: CRAN `spls 2.3.2` (Chun & Keleş authors). No widely installable Python port exists with this exact normalisation convention.

R roxygen note (`sklearn_methods.R::sparse_pls`):

> Sparse SIMPLS — formula entry point.

MATLAB header (`bindings/matlab/+pls4all/SparsePlsRegression.m`):

```text
pls4all.SparsePlsRegression — Sparse SIMPLS (Chun & Keles 2010)
 as a tier-2 classdef. Construct via the factory:

   mdl = pls4all.fitrsparsepls(X, y, "NumComponents", 5, "Lambda", 0.05)

 or directly:
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
p4a_sparse_simpls_fit(ctx, cfg, &x_view, &y_view, /* hyperparams */, &res);
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
from pls4all._methods import sparse_simpls_fit
with pls4all.Context() as ctx, pls4all.Config() as cfg:
    res = sparse_simpls_fit(ctx, cfg, X, y, n_components=4)
# then: res.matrix("predictions"), res.matrix("coefficients"),
# res.vector("mask"), res.scalar("intercept"), …
```

:::

:::{tab-item} Python · pls4all.sklearn
:sync: python-sklearn
:class-label: lang-python

```python
from pls4all.sklearn import SparseSimplsRegression
mdl = SparseSimplsRegression(n_components=2, sparsity_lambda=0.05)
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
res <- pls4all_method("sparse_simpls", X, y,
                      n_components = 4L, params = list(sparsity_lambda = 0.05))
# res is a named list with MethodResult arrays/scalars.
# selected_indices / top_k_intervals are 1-based.
```

:::

:::{tab-item} R · pls4all (raw fn)
:sync: r-raw
:class-label: lang-r

```r
library(pls4all)
res  <- sparse_simpls_fit(X, Y, n_components, sparsity_lambda = 0.05)
yhat <- pls4all_predict(res, X_test)
```

:::

:::{tab-item} R · pls4all (formula+S3)
:sync: r-formula
:class-label: lang-r

```r
library(pls4all)
fit  <- sparse_pls(y ~ ., data = train, ncomp = 4L)
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
    set_engine("pls4all", algorithm = "sparse_simpls") %>%
    set_mode("regression")
wflow <- workflow() %>% add_model(spec) %>% add_recipe(rec)
fit <- fit(wflow, data = train)

# mlr3
library(mlr3)
pls4all::register_mlr3()
lrn <- lrn("regr.pls4all", algorithm = "sparse_simpls", n_components = 4L)
lrn$train(task)
```

:::

:::{tab-item} MATLAB · pls4all (MEX)
:sync: matlab-mex
:class-label: lang-matlab

```matlab
res = pls4all.sparse_simpls(X, y, 4);
% see header of bindings/matlab/+pls4all/sparse_simpls.m for full
% parameter surface:
%   res = sparse_simpls(X, Y, n_components, sparsity_lambda)
yhat = predict(res, Xtest);
```

:::

:::{tab-item} MATLAB · pls4all (classdef)
:sync: matlab-classdef
:class-label: lang-matlab

```matlab
mdl  = pls4all.fit("sparse_simpls", X, y, "NumComponents", 4);
yhat = predict(mdl, Xtest);
```

:::

::::


**Registry parity references** 📐

:::{card}
:class-card: external-refs

- 📐 **`ref.r_spls`** (R · r) — `spls` 2.3.2 · qualitative (rmse_rel ≤ 1e+00) — R `spls` 2.3.2 (Chun & Keles). Predicts via the regression coefficient matrix from sparse-thresholded SIMPLS.
:::

### Benchmarks

Median wall-clock per cell from [`benchmarks/cross_binding/results/full_matrix.csv`](../benchmarks/overview.md). Verdict legend: ✓ exact · ≈ drift · ✗ divergent · ⊘ not available in lib · — not run · ⚠ error. The fastest backend per column is marked with a 🏆 medal. Rows tagged with **📐** are *also* declared in [`benchmarks/parity_timing/registry.py`](../benchmarks/methodology.md) as the canonical parity references for this method (`python_reference` / `r_reference` / `extra_references`). Cell parity in this table is measured against the cross-binding reference (`pls4all.cpp` blas-omp, 1 thread); the 📐 icon points at the *library-of-record* the parity gate ultimately answers to. Hover the icon to see the role and tolerance band.

::::{tab-set}
:class: parity-tabs

:::{tab-item} 1 thread
:sync: threads-1

<div class="parity-table-wrap">
<table class="docutils parity-grouped">
<thead><tr><th scope="col">Backend</th><th scope="col">Parity</th><th class="size-col" scope="col">100×50 (ms)</th><th class="size-col" scope="col">200×50 (ms)</th><th class="size-col" scope="col">500×50 (ms)</th></tr></thead>
<tbody class="lang-band lang-cpp"><tr class="lang-band-row" data-lang="cpp"><th colspan="5" scope="rowgroup"><span class="lang-band-dot"></span>C++ native · libp4a</th></tr>
<tr class="bk-row"><td class="bk-name"><code>pls4all.cpp.blas</code></td><td class="parity parity-drift">≈</td><td class="ms">0.94 ms</td><td class="ms ms-empty">—</td><td class="ms">5.60 ms</td></tr>
<tr class="bk-row"><td class="bk-name"><code>pls4all.cpp.blas+omp</code></td><td class="parity parity-drift">≈</td><td class="ms">1.00 ms</td><td class="ms ms-empty">—</td><td class="ms">5.29 ms</td></tr>
<tr class="bk-row"><td class="bk-name"><code>pls4all.cpp.omp</code></td><td class="parity parity-drift">≈</td><td class="ms ms-best">0.90 ms<span class="medal" title="fastest">🏆</span></td><td class="ms ms-empty">—</td><td class="ms ms-best">4.60 ms<span class="medal" title="fastest">🏆</span></td></tr>
<tr class="bk-row"><td class="bk-name"><code>pls4all.cpp.ref</code></td><td class="parity parity-drift">≈</td><td class="ms">0.99 ms</td><td class="ms ms-empty">—</td><td class="ms">4.69 ms</td></tr>
</tbody>
<tbody class="lang-band lang-python"><tr class="lang-band-row" data-lang="python"><th colspan="5" scope="rowgroup"><span class="lang-band-dot"></span>Python · pls4all</th></tr>
<tr class="bk-row"><td class="bk-name"><code>pls4all.python</code></td><td class="parity parity-drift">≈</td><td class="ms">0.95 ms</td><td class="ms ms-empty">—</td><td class="ms">5.67 ms</td></tr>
<tr class="bk-row"><td class="bk-name"><code>pls4all.registry</code></td><td class="parity parity-drift">≈</td><td class="ms">1.28 ms</td><td class="ms ms-best">165.1 ms<span class="medal" title="fastest">🏆</span></td><td class="ms">5.83 ms</td></tr>
<tr class="bk-row"><td class="bk-name"><code>pls4all.sklearn</code></td><td class="parity parity-drift">≈</td><td class="ms">1.11 ms</td><td class="ms ms-empty">—</td><td class="ms">6.64 ms</td></tr>
</tbody>
<tbody class="lang-band lang-r"><tr class="lang-band-row" data-lang="r"><th colspan="5" scope="rowgroup"><span class="lang-band-dot"></span>R · pls4all</th></tr>
<tr class="bk-row"><td class="bk-name"><code>pls4all.R</code></td><td class="parity parity-drift">≈</td><td class="ms">4.00 ms</td><td class="ms ms-empty">—</td><td class="ms">23.0 ms</td></tr>
<tr class="bk-row"><td class="bk-name"><code>pls4all.R.formula</code></td><td class="parity parity-drift">≈</td><td class="ms">7.50 ms</td><td class="ms ms-empty">—</td><td class="ms">27.0 ms</td></tr>
</tbody>
<tbody class="lang-band lang-matlab"><tr class="lang-band-row" data-lang="matlab"><th colspan="5" scope="rowgroup"><span class="lang-band-dot"></span>MATLAB · pls4all</th></tr>
<tr class="bk-row"><td class="bk-name"><code>pls4all.matlab</code></td><td class="parity parity-drift">≈</td><td class="ms">2.67 ms</td><td class="ms ms-empty">—</td><td class="ms">9.31 ms</td></tr>
<tr class="bk-row"><td class="bk-name"><code>pls4all.matlab.classdef</code></td><td class="parity parity-drift">≈</td><td class="ms">5.32 ms</td><td class="ms ms-empty">—</td><td class="ms">12.2 ms</td></tr>
</tbody>
<tbody class="lang-band lang-python"><tr class="lang-band-row" data-lang="python"><th colspan="5" scope="rowgroup"><span class="lang-band-dot"></span>Python · external</th></tr>
<tr class="bk-row"><td class="bk-name"><code>ikpls</code></td><td class="parity parity-not_available">⊘</td><td class="ms">—</td><td class="ms ms-empty">—</td><td class="ms">—</td></tr>
<tr class="bk-row"><td class="bk-name"><code>sklearn</code></td><td class="parity parity-not_available">⊘</td><td class="ms">—</td><td class="ms ms-empty">—</td><td class="ms">—</td></tr>
</tbody>
<tbody class="lang-band lang-r"><tr class="lang-band-row" data-lang="r"><th colspan="5" scope="rowgroup"><span class="lang-band-dot"></span>R · external</th></tr>
<tr class="bk-row"><td class="bk-name"><code>mixOmics</code></td><td class="parity parity-drift">≈</td><td class="ms">11.0 ms</td><td class="ms ms-empty">—</td><td class="ms">42.5 ms</td></tr>
<tr class="bk-row"><td class="bk-name"><code>pls</code></td><td class="parity parity-not_available">⊘</td><td class="ms">—</td><td class="ms ms-empty">—</td><td class="ms">—</td></tr>
<tr class="bk-row truth-source truth-source-qualitative"><td class="bk-name"><span class="truth-mark" title="Registry parity reference (r): spls 2.3.2 — qualitative (rmse_rel ≤ 1e+00)">📐</span><code>ref.r_spls</code></td><td class="parity parity-drift">≈</td><td class="ms">190.1 ms</td><td class="ms">135.3 ms</td><td class="ms">250.2 ms</td></tr>
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