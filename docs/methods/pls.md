# `pls` — PLS regression (SIMPLS)

_Group_: **Core PLS** · _Registry tolerance_: `0.1`

## Description

SIMPLS PLS regression baseline

From the `pls4all.sklearn.PLSRegression` docstring:

> Partial Least Squares regression backed by `pls4all`'s C core.

<details>
<summary>Full Python <code>sklearn</code>-wrapper docstring</summary>

```text
Partial Least Squares regression backed by `pls4all`'s C core.

Drop-in replacement for `sklearn.cross_decomposition.PLSRegression`
with two distinguishing knobs:

* `solver` selects the inner algorithm (NIPALS, SIMPLS, SVD, …)
  directly — sklearn only exposes 'nipals' / 'svd'.
* Round-trip via `pickle.dumps` is bit-exact, backed by the C ABI
  `.n4a` bundle (`p4a_model_export_to_buffer`).

Parameters
----------
n_components : int, default=2
    Number of latent components.
solver : str, default='simpls'
    One of 'nipals', 'simpls', 'orthogonal-scores', 'kernel',
    'wide-kernel', 'svd', 'power', 'randomized-svd'.
center_x, scale_x : bool, default=True
    Standardize X columns to zero mean / unit variance.
center_y : bool, default=True
    Center y to zero mean.
scale_y : bool, default=False
    Standardize y columns to unit variance.
tol : float, default=1e-6
    Convergence tolerance for iterative solvers.
max_iter : int, default=500
    Max NIPALS iterations.
store_scores : bool, default=False
    Keep the latent score matrices (`x_scores_`) after fit.
```

</details>

> **Registry note** — Baseline SIMPLS cell. sklearn uses NIPALS and ikpls uses improved-kernel PLS, so exact bit parity is not expected; the row exists to anchor timing comparisons.

### Parameters

| Name | Type | Default | Notes |
|------|------|---------|-------|
| `n_components` | `int` | `2` | Number of latent components extracted (k). |
| `solver` | `str` | `'simpls'` | Inner algorithm: 'nipals', 'simpls', 'svd', 'kernel', 'orthogonal-scores', 'power', 'randomized-svd', 'wide-kernel'. |
| `center_x` | `bool` | `True` | Subtract the column mean of X before fitting. |
| `scale_x` | `bool` | `True` | Standardize X columns to unit variance before fitting. |
| `center_y` | `bool` | `True` | Subtract the column mean of y before fitting. |
| `scale_y` | `bool` | `False` | Standardize y columns to unit variance before fitting. |
| `tol` | `float` | `1e-06` | Convergence tolerance for iterative solvers (NIPALS / power-iteration). |
| `max_iter` | `int` | `500` | Maximum iterations for iterative solvers. |
| `store_scores` | `bool` | `False` | If True, keep the latent score matrix (`x_scores_`) after fit. |

## Explanations

### Bibliographic source

de Jong, S. (1993). *SIMPLS: an alternative approach to partial least squares regression*. Chemometrics and Intelligent Laboratory Systems 18(3), 251–263.

### Mathematical principle

Partial Least Squares regression seeks a set of latent directions in the predictor space that maximise the *covariance* with the response, in contrast to PCA which maximises only the variance of $\mathbf{X}$.

Given centred $\mathbf{X} \in \mathbb{R}^{n\times p}$ and $\mathbf{Y} \in \mathbb{R}^{n\times q}$, the first PLS component is the unit-norm direction $\mathbf{w}_1$ maximising $\operatorname{Cov}(\mathbf{X}\mathbf{w}_1, \mathbf{Y})$. Closed form: $\mathbf{w}_1 \propto \mathbf{X}^{\top}\mathbf{Y}$ (or its dominant left singular vector when $q>1$). Subsequent components are extracted from the deflated residual matrix so the resulting scores $\mathbf{T} = \mathbf{X}\mathbf{W}$ are orthogonal.

**SIMPLS** (de Jong 1993) is algebraically equivalent to NIPALS but computes the loading weights directly from the cross-product $\mathbf{S} = \mathbf{X}^{\top}\mathbf{Y}$ without re-deflating $\mathbf{X}$ at each step. This avoids accumulating floating-point error from iterative deflation and runs in roughly half the time of NIPALS for the same number of components. SIMPLS is the variant exposed by MATLAB's `plsregress`.

Once $k$ latent scores have been extracted the regression coefficients are reconstructed as $\mathbf{B} = \mathbf{W}(\mathbf{P}^{\top}\mathbf{W})^{-1}\mathbf{Q}^{\top}$, where $\mathbf{P}, \mathbf{Q}$ are the X- and Y-loadings. Predictions on new $\mathbf{X}^{\star}$ follow $\hat{\mathbf{Y}} = \mathbf{X}^{\star}\mathbf{B} + \bar{\mathbf{y}}$. The choice of $k$ trades bias and variance: use cross-validated PRESS or the one-SE rule of Hastie et al. (2009) to select it.

### Implementation

Dispatched through `Algorithm.PLS_REGRESSION` + `Solver.SIMPLS` in libp4a (the `p4a_pls_fit` C entry point). The same `Model.fit` / `Model.predict` surface is used by every binding. NIPALS, SVD, power-iteration, randomised-SVD, orthogonal-scores, kernel and wide-kernel solver variants are all available — see the `Solver` enum.

R roxygen note (`sklearn.R::pls`):

> Formula-based PLS regression wrapper around the pls4all C ABI.

MATLAB header (`bindings/matlab/+pls4all/Regression.m`):

```text
pls4all.Regression — Statistics Toolbox-style class for PLS regression.

 Tier-2 idiomatic MATLAB / Octave wrapper around the tier-1
 pls4all.pls_fit(X, Y, n_components) primitive. Mirrors the shape
 of MATLAB's built-in RegressionPartialLeastSquares: object-oriented
 properties + methods, factory function `pls4all.fitrpls`, and a
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
p4a_pls_fit(ctx, cfg, &x_view, &y_view, /* hyperparams */, &res);
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
from pls4all._methods import pls_fit
with pls4all.Context() as ctx, pls4all.Config() as cfg:
    res = pls_fit(ctx, cfg, X, y, n_components=4)
# then: res.matrix("predictions"), res.matrix("coefficients"),
# res.vector("mask"), res.scalar("intercept"), …
```

:::

:::{tab-item} Python · pls4all.sklearn
:sync: python-sklearn
:class-label: lang-python

```python
from pls4all.sklearn import PLSRegression
mdl = PLSRegression(n_components=2, solver='simpls', center_x=True, scale_x=True, center_y=True, scale_y=False, tol=1e-06, max_iter=500, store_scores=False)
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
res <- pls4all_method("pls", X, y,
                      n_components = 4L)
# res is a named list with MethodResult arrays/scalars.
# selected_indices / top_k_intervals are 1-based.
```

:::

:::{tab-item} R · pls4all (formula+S3)
:sync: r-formula
:class-label: lang-r

```r
library(pls4all)
fit  <- pls(y ~ ., data = train, ncomp = 4L)
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
    set_engine("pls4all", algorithm = "pls") %>%
    set_mode("regression")
wflow <- workflow() %>% add_model(spec) %>% add_recipe(rec)
fit <- fit(wflow, data = train)

# mlr3
library(mlr3)
pls4all::register_mlr3()
lrn <- lrn("regr.pls4all", algorithm = "pls", n_components = 4L)
lrn$train(task)
```

:::

:::{tab-item} MATLAB · pls4all (MEX)
:sync: matlab-mex
:class-label: lang-matlab

```matlab
res = pls4all.pls_fit(X, y, 4);
% see header of bindings/matlab/+pls4all/pls_fit.m for full
% parameter surface:
%   [coefs, x_mean, y_mean, predictions] = pls_fit(X, Y, n_components)
yhat = predict(res, Xtest);
```

:::

:::{tab-item} MATLAB · pls4all (classdef)
:sync: matlab-classdef
:class-label: lang-matlab

```matlab
mdl  = pls4all.fit("pls", X, y, "NumComponents", 4);
yhat = predict(mdl, Xtest);
```

:::

::::


**Registry parity references** 📐

:::{card}
:class-card: external-refs

- 📐 **`ref.python_ikpls`** (python · ikpls) — `ikpls` 4.0.1.post1 · qualitative (rmse_rel ≤ 1e-01) — ikpls.numpy_ikpls.PLS algorithm 1.
- 📐 **`ref.python_scikit_learn`** (python · python) — `scikit-learn` 1.8.0 · qualitative (rmse_rel ≤ 1e-01) — sklearn.cross_decomposition.PLSRegression(scale=False).
- 📐 **`ref.r_mixomics`** (R · mixOmics) — `mixOmics` 6.26.0 · qualitative (rmse_rel ≤ 1e-01) — Bioconductor mixOmics::pls(mode='regression', scale=FALSE).
- 📐 **`ref.r_pls`** (R · r) — `pls` 2.8.5 · qualitative (rmse_rel ≤ 1e-01) — R pls::plsr(method='simpls', scale=FALSE).
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
<tr class="bk-row"><td class="bk-name"><code>pls4all.cpp.blas</code></td><td class="parity parity-exact">✓ 9e-16</td><td class="ms ms-best">0.95 ms<span class="medal" title="fastest">🏆</span></td></tr>
<tr class="bk-row"><td class="bk-name"><code>pls4all.cpp.blas+omp</code></td><td class="parity parity-exact">✓ 9e-16</td><td class="ms">1.02 ms</td></tr>
<tr class="bk-row"><td class="bk-name"><code>pls4all.cpp.omp</code></td><td class="parity parity-exact">✓ 1e-15</td><td class="ms">1.02 ms</td></tr>
<tr class="bk-row"><td class="bk-name"><code>pls4all.cpp.ref</code></td><td class="parity parity-exact">✓ 1e-15</td><td class="ms">0.98 ms</td></tr>
</tbody>
<tbody class="lang-band lang-python"><tr class="lang-band-row" data-lang="python"><th colspan="3" scope="rowgroup"><span class="lang-band-dot"></span>Python · pls4all</th></tr>
<tr class="bk-row"><td class="bk-name"><code>pls4all.python</code></td><td class="parity parity-exact">✓ bind</td><td class="ms">0.97 ms</td></tr>
<tr class="bk-row"><td class="bk-name"><code>pls4all.registry</code></td><td class="parity parity-exact">✓ bind</td><td class="ms">1.10 ms</td></tr>
<tr class="bk-row"><td class="bk-name"><code>pls4all.sklearn</code></td><td class="parity parity-exact">✓ bind</td><td class="ms">1.22 ms</td></tr>
</tbody>
<tbody class="lang-band lang-r"><tr class="lang-band-row" data-lang="r"><th colspan="3" scope="rowgroup"><span class="lang-band-dot"></span>R · pls4all</th></tr>
<tr class="bk-row"><td class="bk-name"><code>pls4all.R</code></td><td class="parity parity-exact">✓ 1e-14</td><td class="ms">2.00 ms</td></tr>
<tr class="bk-row"><td class="bk-name"><code>pls4all.R.formula</code></td><td class="parity parity-exact">✓ 1e-14</td><td class="ms">4.00 ms</td></tr>
</tbody>
<tbody class="lang-band lang-matlab"><tr class="lang-band-row" data-lang="matlab"><th colspan="3" scope="rowgroup"><span class="lang-band-dot"></span>MATLAB · pls4all</th></tr>
<tr class="bk-row"><td class="bk-name"><code>pls4all.matlab</code></td><td class="parity parity-exact">✓ 1e-14</td><td class="ms">1.76 ms</td></tr>
<tr class="bk-row"><td class="bk-name"><code>pls4all.matlab.classdef</code></td><td class="parity parity-exact">✓ 9e-15</td><td class="ms">4.63 ms</td></tr>
</tbody>
<tbody class="lang-band lang-python"><tr class="lang-band-row" data-lang="python"><th colspan="3" scope="rowgroup"><span class="lang-band-dot"></span>Python · external</th></tr>
<tr class="bk-row"><td class="bk-name"><code>ikpls</code></td><td class="parity parity-exact">✓ 2e-02</td><td class="ms">1.08 ms</td></tr>
<tr class="bk-row truth-source truth-source-qualitative"><td class="bk-name"><span class="truth-mark" title="Registry parity reference (ikpls): ikpls 4.0.1.post1 — qualitative (rmse_rel ≤ 1e-01)">📐</span><code>ref.python_ikpls</code></td><td class="parity parity-exact">✓ 2e-02</td><td class="ms">1.30 ms</td></tr>
<tr class="bk-row truth-source truth-source-qualitative"><td class="bk-name"><span class="truth-mark" title="Registry parity reference (python): scikit-learn 1.8.0 — qualitative (rmse_rel ≤ 1e-01)">📐</span><code>ref.python_scikit_learn</code></td><td class="parity parity-exact">✓ ref</td><td class="ms">1.45 ms</td></tr>
<tr class="bk-row"><td class="bk-name"><code>sklearn</code></td><td class="parity parity-exact">✓ ref</td><td class="ms">1.32 ms</td></tr>
</tbody>
<tbody class="lang-band lang-r"><tr class="lang-band-row" data-lang="r"><th colspan="3" scope="rowgroup"><span class="lang-band-dot"></span>R · external</th></tr>
<tr class="bk-row"><td class="bk-name"><code>mixOmics</code></td><td class="parity parity-exact">✓ 6e-16</td><td class="ms">7.50 ms</td></tr>
<tr class="bk-row"><td class="bk-name"><code>pls</code></td><td class="parity parity-exact">✓ 1e-15</td><td class="ms">5.50 ms</td></tr>
<tr class="bk-row truth-source truth-source-qualitative"><td class="bk-name"><span class="truth-mark" title="Registry parity reference (mixOmics): mixOmics 6.26.0 — qualitative (rmse_rel ≤ 1e-01)">📐</span><code>ref.r_mixomics</code></td><td class="parity parity-exact">✓ 6e-16</td><td class="ms">8.00 ms</td></tr>
<tr class="bk-row truth-source truth-source-qualitative"><td class="bk-name"><span class="truth-mark" title="Registry parity reference (r): pls 2.8.5 — qualitative (rmse_rel ≤ 1e-01)">📐</span><code>ref.r_pls</code></td><td class="parity parity-exact">✓ 1e-15</td><td class="ms">7.00 ms</td></tr>
<tr class="bk-row"><td class="bk-name"><code>ropls</code></td><td class="parity parity-not_available">⊘</td><td class="ms">—</td></tr>
</tbody>
<tbody class="lang-band lang-matlab"><tr class="lang-band-row" data-lang="matlab"><th colspan="3" scope="rowgroup"><span class="lang-band-dot"></span>MATLAB · external</th></tr>
<tr class="bk-row"><td class="bk-name"><code>plsregress</code></td><td class="parity parity-exact">✓ 7e-15</td><td class="ms">2.21 ms</td></tr>
</tbody>
</table>
</div>

:::

::::


---

_See also_: [benchmark overview](../benchmarks/overview.md) · [methods index](index.md) · [interactive dashboard](../landing/dashboard.md)