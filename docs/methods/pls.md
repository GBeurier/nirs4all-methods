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

Every pls4all binding tab dispatches into the same C kernel; the external libraries listed at the bottom of the page are the parity references registered in `benchmarks.parity_timing.registry`. Switch tabs to read the same fit in your language. The R package now ships drop-in-compatible facades for the CRAN `pls` package (`plsr`, `pcr`, `mvr`) and for the `mdatools::pls(x, y, ...)` matrix idiom — those tabs appear only on the methods that have a meaningful equivalence.

**pls4all bindings**

::::{tab-set}
:class: pls4all-bindings

:::{tab-item} C ABI · libp4a
:sync: c
:class-label: lang-c

```c
/* C ABI — libp4a (Model.fit path) */
p4a_context_t* ctx = p4a_context_create();
p4a_config_t*  cfg = p4a_config_create();
p4a_config_set_algorithm(cfg, P4A_ALGORITHM_PLS_REGRESSION);
p4a_config_set_solver   (cfg, P4A_SOLVER_SIMPLS);
p4a_config_set_n_components(cfg, 4);
p4a_model_t* mdl = NULL;
p4a_model_fit(ctx, cfg, &x_view, &y_view, &mdl);
p4a_model_predict(ctx, mdl, &x_test_view, &y_hat_view);
p4a_model_destroy(mdl);
p4a_config_destroy(cfg);
p4a_context_destroy(ctx);
```

:::

:::{tab-item} Python · pls4all (raw)
:sync: python-raw
:class-label: lang-python

```python
import pls4all
from pls4all import Algorithm, Solver
with pls4all.Context() as ctx, pls4all.Config() as cfg:
    cfg.algorithm = Algorithm.PLS_REGRESSION
    cfg.solver = Solver.SIMPLS
    cfg.n_components = 4
    with pls4all.Model.fit(ctx, cfg, X, y) as mdl:
        y_hat = mdl.predict(X_test)
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

:::{tab-item} R · `pls` package compat
:sync: r-pls-compat
:class-label: lang-r

```r
library(pls4all)
# Drop-in for CRAN `pls::plsr` (same signature).
fit  <- plsr(y ~ ., ncomp = 4L, data = train,
                           validation = "CV", segments = 10L)
yhat <- predict(fit, newdata = test, ncomp = 4L)
RMSEP(fit)
```

:::

:::{tab-item} R · `mdatools` compat
:sync: r-mdatools
:class-label: lang-r

```r
library(pls4all)
# Drop-in for `mdatools::pls(x, y, ncomp, method = "simpls")`.
fit  <- pls_mdatools(X, y, ncomp = 4L, method = "simpls",
               center = TRUE, scale = FALSE)
yhat <- predict(fit, newdata = X_test, ncomp = 4L)
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

### Validation contract

Structural validation contract for `pls`, exported by the declarative validation framework (`benchmarks/validation/registry/`) from `benchmarks.parity_timing.registry`. Regenerated by `python -m benchmarks.validation.export_registry --write`.

| Field | Value |
|------|-------|
| Reference kind | `external` |
| Canonical reference role | `python` |
| Declared references | Python, R, ikpls, mixOmics |
| Comparator | `reference_parity` (external-reference validity gate) |
| Comparator metric | `rmse_rel` |
| Reference tolerance | `1e-01` (qualitative) |
| Binding tolerance | `1e-06` |
| Prediction key | `predictions` |
| Required data flags | none |

**Benchmark cell parameters**

| Parameter | Value |
|-----------|-------|
| `n_components` | `4` |
| `n_features` | `50` |
| `n_samples` | `200` |

**Validation suites**

- **`benchmark`** — Full cross-binding sweep - every MethodSpec across every default dataset size. Mirrors the cross-binding orchestrator's `DEFAULT_SIZES` surface. Datasets: `10000x50`, `10000x500`, `100x2500`, `100x50`, `100x500`, `2500x2500`, `2500x50`, `2500x500`, `500x2500`, `500x50`, `500x500`. Comparators: `binding_parity`, `reference_parity`.
- **`smoke`** — Fastest cross-binding cells for every MethodSpec; used by pre-commit / CI smoke gates. Datasets: `100x50`, `100x500`. Comparators: `binding_parity`, `reference_parity`.

**Dataset cells referenced by these suites**

| Dataset | n | p |
|---------|---|---|
| `100x50` | 100 | 50 |
| `100x500` | 100 | 500 |
| `100x2500` | 100 | 2500 |
| `500x50` | 500 | 50 |
| `500x500` | 500 | 500 |
| `500x2500` | 500 | 2500 |
| `2500x50` | 2500 | 50 |
| `2500x500` | 2500 | 500 |
| `2500x2500` | 2500 | 2500 |
| `10000x50` | 10000 | 50 |
| `10000x500` | 10000 | 500 |

### Benchmarks

Adaptive wall-clock per cell measured against [`full_matrix.csv`](../benchmarks/overview.md). Only backends that implement this method are listed; libraries without the method are omitted.

**Divergence** is the worst finite value over the visible sizes for each backend: reference-gate rows show `rmse_rel`, binding-gate rows show `max_diff` against the C++ baseline. Rows without a recorded comparison show `—`; the fastest backend per column is marked 🏆.

**Reference gate**: non-release diagnostic — shape/smoke comparison only (`rmse_rel ≤ 1e-01`). Treat it as evidence that both implementations ran, not as numerical agreement.

Rows tagged with **📐** are the canonical parity references for this method (declared in [`parity_timing.registry`](../benchmarks/methodology.md)). C++ and external rows show reference parity; pls4all language bindings show binding parity against the C++ backend. Hover the icon for role and tolerance band.

::::{tab-set}
:class: parity-tabs

:::{tab-item} 1 thread
:sync: threads-1

<div class="parity-table-wrap">
<table class="docutils parity-grouped">
<thead><tr><th scope="col">Backend</th><th scope="col">Divergence</th><th class="size-col" scope="col">100×50 (ms)</th><th class="size-col" scope="col">100×500 (ms)</th><th class="size-col" scope="col">100×2500 (ms)</th><th class="size-col" scope="col">1000×500 (ms)</th></tr></thead>
<tbody class="lang-band lang-cpp"><tr class="lang-band-row" data-lang="cpp"><th colspan="6" scope="rowgroup"><span class="lang-band-dot"></span>C++ native · libp4a</th></tr>
<tr class="bk-row"><td class="bk-name"><code>pls4all.cpp.blas</code></td><td class="parity parity-divergence parity-ref-qualitative">3e-15</td><td class="ms">0.99 ms</td><td class="ms">8.35 ms</td><td class="ms ms-best">40.3 ms<span class="medal" title="fastest">🏆</span></td><td class="ms">182.3 ms</td></tr>
<tr class="bk-row"><td class="bk-name"><code>pls4all.cpp.blas+omp</code></td><td class="parity parity-divergence parity-ref-qualitative">3e-15</td><td class="ms">0.99 ms</td><td class="ms">8.07 ms</td><td class="ms">40.3 ms</td><td class="ms">99.6 ms</td></tr>
<tr class="bk-row"><td class="bk-name"><code>pls4all.cpp.omp</code></td><td class="parity parity-divergence parity-ref-qualitative">3e-15</td><td class="ms">0.99 ms</td><td class="ms">8.36 ms</td><td class="ms">40.8 ms</td><td class="ms">81.8 ms</td></tr>
<tr class="bk-row"><td class="bk-name"><code>pls4all.cpp.ref</code></td><td class="parity parity-divergence parity-ref-qualitative">3e-15</td><td class="ms ms-best">0.98 ms<span class="medal" title="fastest">🏆</span></td><td class="ms">8.24 ms</td><td class="ms">41.0 ms</td><td class="ms">83.1 ms</td></tr>
</tbody>
<tbody class="lang-band lang-python"><tr class="lang-band-row" data-lang="python"><th colspan="6" scope="rowgroup"><span class="lang-band-dot"></span>Python · pls4all</th></tr>
<tr class="bk-row"><td class="bk-name"><code>pls4all.python</code></td><td class="parity parity-divergence parity-ref-qualitative">3e-15</td><td class="ms">1.01 ms</td><td class="ms ms-best">8.03 ms<span class="medal" title="fastest">🏆</span></td><td class="ms">41.0 ms</td><td class="ms">84.4 ms</td></tr>
<tr class="bk-row"><td class="bk-name"><code>pls4all.sklearn</code></td><td class="parity parity-divergence parity-ref-qualitative">3e-15</td><td class="ms">1.22 ms</td><td class="ms">8.49 ms</td><td class="ms">42.0 ms</td><td class="ms ms-best">78.5 ms<span class="medal" title="fastest">🏆</span></td></tr>
</tbody>
<tbody class="lang-band lang-r"><tr class="lang-band-row" data-lang="r"><th colspan="6" scope="rowgroup"><span class="lang-band-dot"></span>R · pls4all</th></tr>
<tr class="bk-row"><td class="bk-name"><code>pls4all.R</code></td><td class="parity parity-divergence parity-ref-qualitative">2e-15</td><td class="ms">3.36 ms</td><td class="ms">47.7 ms</td><td class="ms">374.0 ms</td><td class="ms">449.9 ms</td></tr>
<tr class="bk-row"><td class="bk-name"><code>pls4all.R.formula</code></td><td class="parity parity-divergence parity-ref-qualitative">2e-15</td><td class="ms">4.90 ms</td><td class="ms">58.5 ms</td><td class="ms">563.8 ms</td><td class="ms">611.1 ms</td></tr>
<tr class="bk-row"><td class="bk-name"><code>pls4all.R.mdatools</code></td><td class="parity parity-divergence parity-ref-qualitative">2e-15</td><td class="ms">5.12 ms</td><td class="ms">61.6 ms</td><td class="ms">570.8 ms</td><td class="ms">547.9 ms</td></tr>
<tr class="bk-row"><td class="bk-name"><code>pls4all.R.pls</code></td><td class="parity parity-divergence parity-ref-qualitative">2e-15</td><td class="ms">8.92 ms</td><td class="ms">103.8 ms</td><td class="ms">958.8 ms</td><td class="ms">695.5 ms</td></tr>
</tbody>
<tbody class="lang-band lang-matlab"><tr class="lang-band-row" data-lang="matlab"><th colspan="6" scope="rowgroup"><span class="lang-band-dot"></span>MATLAB · pls4all</th></tr>
<tr class="bk-row"><td class="bk-name"><code>pls4all.matlab</code></td><td class="parity parity-divergence parity-ref-qualitative">3e-15</td><td class="ms">1.72 ms</td><td class="ms">13.0 ms</td><td class="ms">65.6 ms</td><td class="ms">154.4 ms</td></tr>
<tr class="bk-row"><td class="bk-name"><code>pls4all.matlab.classdef</code></td><td class="parity parity-divergence parity-ref-qualitative">3e-15</td><td class="ms">2.16 ms</td><td class="ms">13.6 ms</td><td class="ms">67.6 ms</td><td class="ms">162.6 ms</td></tr>
</tbody>
<tbody class="lang-band lang-python"><tr class="lang-band-row" data-lang="python"><th colspan="6" scope="rowgroup"><span class="lang-band-dot"></span>Python · external</th></tr>
<tr class="bk-row truth-source truth-source-qualitative"><td class="bk-name"><span class="truth-mark" title="Registry parity reference (ikpls): ikpls 4.0.1.post1 — qualitative (rmse_rel ≤ 1e-01)">📐</span><code>ref.python_ikpls</code></td><td class="parity parity-divergence parity-divergent">2.3e-02</td><td class="ms">1.54 ms</td><td class="ms">8.31 ms</td><td class="ms">42.6 ms</td><td class="ms">82.8 ms</td></tr>
<tr class="bk-row truth-source truth-source-qualitative"><td class="bk-name"><span class="truth-mark" title="Registry parity reference (python): scikit-learn 1.8.0 — qualitative (rmse_rel ≤ 1e-01)">📐</span><code>ref.python_scikit_learn</code></td><td class="parity parity-divergence parity-ref-source">0</td><td class="ms">1.55 ms</td><td class="ms">8.82 ms</td><td class="ms">46.0 ms</td><td class="ms">84.7 ms</td></tr>
</tbody>
<tbody class="lang-band lang-r"><tr class="lang-band-row" data-lang="r"><th colspan="6" scope="rowgroup"><span class="lang-band-dot"></span>R · external</th></tr>
<tr class="bk-row truth-source truth-source-qualitative"><td class="bk-name"><span class="truth-mark" title="Registry parity reference (mixOmics): mixOmics 6.26.0 — qualitative (rmse_rel ≤ 1e-01)">📐</span><code>ref.r_mixomics</code></td><td class="parity parity-divergence parity-ref-qualitative">1e-15</td><td class="ms">8.73 ms</td><td class="ms">56.7 ms</td><td class="ms">519.5 ms</td><td class="ms">770.5 ms</td></tr>
<tr class="bk-row truth-source truth-source-qualitative"><td class="bk-name"><span class="truth-mark" title="Registry parity reference (r): pls 2.8.5 — qualitative (rmse_rel ≤ 1e-01)">📐</span><code>ref.r_pls</code></td><td class="parity parity-divergence parity-ref-qualitative">2e-14</td><td class="ms">6.69 ms</td><td class="ms">88.0 ms</td><td class="ms">724.1 ms</td><td class="ms">651.7 ms</td></tr>
</tbody>
<tbody class="lang-band lang-matlab"><tr class="lang-band-row" data-lang="matlab"><th colspan="6" scope="rowgroup"><span class="lang-band-dot"></span>MATLAB · external</th></tr>
<tr class="bk-row"><td class="bk-name"><code>plsregress</code></td><td class="parity parity-divergence parity-ref-qualitative">2e-13</td><td class="ms">2.29 ms</td><td class="ms">14.4 ms</td><td class="ms">71.0 ms</td><td class="ms">177.2 ms</td></tr>
</tbody>
</table>
</div>

:::

:::{tab-item} 8 threads
:sync: threads-8

<div class="parity-table-wrap">
<table class="docutils parity-grouped">
<thead><tr><th scope="col">Backend</th><th scope="col">Divergence</th><th class="size-col" scope="col">100×50 (ms)</th><th class="size-col" scope="col">100×500 (ms)</th><th class="size-col" scope="col">100×2500 (ms)</th><th class="size-col" scope="col">1000×500 (ms)</th></tr></thead>
<tbody class="lang-band lang-cpp"><tr class="lang-band-row" data-lang="cpp"><th colspan="6" scope="rowgroup"><span class="lang-band-dot"></span>C++ native · libp4a</th></tr>
<tr class="bk-row"><td class="bk-name"><code>pls4all.cpp.blas</code></td><td class="parity parity-divergence parity-ref-qualitative">3e-15</td><td class="ms ms-best">0.98 ms<span class="medal" title="fastest">🏆</span></td><td class="ms ms-best">7.97 ms<span class="medal" title="fastest">🏆</span></td><td class="ms ms-best">40.4 ms<span class="medal" title="fastest">🏆</span></td><td class="ms">243.2 ms</td></tr>
<tr class="bk-row"><td class="bk-name"><code>pls4all.cpp.blas+omp</code></td><td class="parity parity-divergence parity-ref-qualitative">3e-15</td><td class="ms">1.02 ms</td><td class="ms">8.25 ms</td><td class="ms">41.0 ms</td><td class="ms">103.7 ms</td></tr>
<tr class="bk-row"><td class="bk-name"><code>pls4all.cpp.omp</code></td><td class="parity parity-divergence parity-ref-qualitative">3e-15</td><td class="ms">0.99 ms</td><td class="ms">8.06 ms</td><td class="ms">40.7 ms</td><td class="ms">95.1 ms</td></tr>
<tr class="bk-row"><td class="bk-name"><code>pls4all.cpp.ref</code></td><td class="parity parity-divergence parity-ref-qualitative">3e-15</td><td class="ms">1.00 ms</td><td class="ms">8.14 ms</td><td class="ms">41.1 ms</td><td class="ms ms-best">83.0 ms<span class="medal" title="fastest">🏆</span></td></tr>
</tbody>
<tbody class="lang-band lang-python"><tr class="lang-band-row" data-lang="python"><th colspan="6" scope="rowgroup"><span class="lang-band-dot"></span>Python · pls4all</th></tr>
<tr class="bk-row"><td class="bk-name"><code>pls4all.python</code></td><td class="parity parity-divergence parity-ref-qualitative">3e-15</td><td class="ms">1.20 ms</td><td class="ms">8.01 ms</td><td class="ms">40.5 ms</td><td class="ms">103.1 ms</td></tr>
<tr class="bk-row"><td class="bk-name"><code>pls4all.sklearn</code></td><td class="parity parity-divergence parity-ref-qualitative">3e-15</td><td class="ms">1.27 ms</td><td class="ms">8.78 ms</td><td class="ms">41.8 ms</td><td class="ms">108.8 ms</td></tr>
</tbody>
<tbody class="lang-band lang-r"><tr class="lang-band-row" data-lang="r"><th colspan="6" scope="rowgroup"><span class="lang-band-dot"></span>R · pls4all</th></tr>
<tr class="bk-row"><td class="bk-name"><code>pls4all.R</code></td><td class="parity parity-divergence parity-ref-qualitative">2e-15</td><td class="ms">3.39 ms</td><td class="ms">44.8 ms</td><td class="ms">374.3 ms</td><td class="ms">457.6 ms</td></tr>
<tr class="bk-row"><td class="bk-name"><code>pls4all.R.formula</code></td><td class="parity parity-divergence parity-ref-qualitative">2e-15</td><td class="ms">4.82 ms</td><td class="ms">60.0 ms</td><td class="ms">552.5 ms</td><td class="ms">628.4 ms</td></tr>
<tr class="bk-row"><td class="bk-name"><code>pls4all.R.mdatools</code></td><td class="parity parity-divergence parity-ref-qualitative">2e-15</td><td class="ms">5.23 ms</td><td class="ms">61.0 ms</td><td class="ms">574.7 ms</td><td class="ms">543.8 ms</td></tr>
<tr class="bk-row"><td class="bk-name"><code>pls4all.R.pls</code></td><td class="parity parity-divergence parity-ref-qualitative">2e-15</td><td class="ms">9.49 ms</td><td class="ms">100.2 ms</td><td class="ms">938.2 ms</td><td class="ms">701.9 ms</td></tr>
</tbody>
<tbody class="lang-band lang-matlab"><tr class="lang-band-row" data-lang="matlab"><th colspan="6" scope="rowgroup"><span class="lang-band-dot"></span>MATLAB · pls4all</th></tr>
<tr class="bk-row"><td class="bk-name"><code>pls4all.matlab</code></td><td class="parity parity-divergence parity-ref-qualitative">3e-15</td><td class="ms">1.62 ms</td><td class="ms">12.8 ms</td><td class="ms">65.2 ms</td><td class="ms">128.9 ms</td></tr>
<tr class="bk-row"><td class="bk-name"><code>pls4all.matlab.classdef</code></td><td class="parity parity-divergence parity-ref-qualitative">3e-15</td><td class="ms">2.22 ms</td><td class="ms">13.9 ms</td><td class="ms">66.1 ms</td><td class="ms">150.3 ms</td></tr>
</tbody>
<tbody class="lang-band lang-python"><tr class="lang-band-row" data-lang="python"><th colspan="6" scope="rowgroup"><span class="lang-band-dot"></span>Python · external</th></tr>
<tr class="bk-row truth-source truth-source-qualitative"><td class="bk-name"><span class="truth-mark" title="Registry parity reference (ikpls): ikpls 4.0.1.post1 — qualitative (rmse_rel ≤ 1e-01)">📐</span><code>ref.python_ikpls</code></td><td class="parity parity-divergence parity-divergent">2.3e-02</td><td class="ms">1.32 ms</td><td class="ms">8.42 ms</td><td class="ms">42.7 ms</td><td class="ms">107.1 ms</td></tr>
<tr class="bk-row truth-source truth-source-qualitative"><td class="bk-name"><span class="truth-mark" title="Registry parity reference (python): scikit-learn 1.8.0 — qualitative (rmse_rel ≤ 1e-01)">📐</span><code>ref.python_scikit_learn</code></td><td class="parity parity-divergence parity-ref-source">0</td><td class="ms">1.51 ms</td><td class="ms">8.89 ms</td><td class="ms">42.1 ms</td><td class="ms">114.5 ms</td></tr>
</tbody>
<tbody class="lang-band lang-r"><tr class="lang-band-row" data-lang="r"><th colspan="6" scope="rowgroup"><span class="lang-band-dot"></span>R · external</th></tr>
<tr class="bk-row truth-source truth-source-qualitative"><td class="bk-name"><span class="truth-mark" title="Registry parity reference (mixOmics): mixOmics 6.26.0 — qualitative (rmse_rel ≤ 1e-01)">📐</span><code>ref.r_mixomics</code></td><td class="parity parity-divergence parity-ref-qualitative">1e-15</td><td class="ms">8.48 ms</td><td class="ms">56.1 ms</td><td class="ms">536.0 ms</td><td class="ms">771.8 ms</td></tr>
<tr class="bk-row truth-source truth-source-qualitative"><td class="bk-name"><span class="truth-mark" title="Registry parity reference (r): pls 2.8.5 — qualitative (rmse_rel ≤ 1e-01)">📐</span><code>ref.r_pls</code></td><td class="parity parity-divergence parity-ref-qualitative">4e-14</td><td class="ms">6.76 ms</td><td class="ms">82.3 ms</td><td class="ms">710.4 ms</td><td class="ms">649.4 ms</td></tr>
</tbody>
<tbody class="lang-band lang-matlab"><tr class="lang-band-row" data-lang="matlab"><th colspan="6" scope="rowgroup"><span class="lang-band-dot"></span>MATLAB · external</th></tr>
<tr class="bk-row"><td class="bk-name"><code>plsregress</code></td><td class="parity parity-divergence parity-ref-qualitative">2e-13</td><td class="ms">2.50 ms</td><td class="ms">14.1 ms</td><td class="ms">75.4 ms</td><td class="ms">169.5 ms</td></tr>
</tbody>
</table>
</div>

:::

::::


---

_See also_: [benchmark overview](../benchmarks/overview.md) · [methods index](index.md) · [interactive dashboard](../landing/dashboard.md)