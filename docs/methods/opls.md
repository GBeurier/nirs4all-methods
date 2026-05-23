# `opls` — Orthogonal PLS (OPLS)

_Group_: **Core PLS** · _Registry tolerance_: `0.001`

## Description

Orthogonal PLS (Trygg & Wold 2002)

From the `pls4all.sklearn.OPLSRegression` docstring:

> Orthogonal PLS regression (Trygg & Wold 2002).

> **Registry note** — Bioconductor `ropls::opls` is the external OPLS reference; convergence and orthogonal-component conventions may differ.

### Parameters

| Name | Type | Default | Notes |
|------|------|---------|-------|
| `n_components` | `int` | `2` | Number of latent components extracted (k). |
| `solver` | `str` | `'nipals'` | Inner algorithm: 'nipals', 'simpls', 'svd', 'kernel', 'orthogonal-scores', 'power', 'randomized-svd', 'wide-kernel'. |
| `center_x` | `bool` | `True` | Subtract the column mean of X before fitting. |
| `scale_x` | `bool` | `True` | Standardize X columns to unit variance before fitting. |
| `center_y` | `bool` | `True` | Subtract the column mean of y before fitting. |
| `scale_y` | `bool` | `False` | Standardize y columns to unit variance before fitting. |
| `tol` | `float` | `1e-06` | Convergence tolerance for iterative solvers (NIPALS / power-iteration). |
| `max_iter` | `int` | `500` | Maximum iterations for iterative solvers. |
| `store_scores` | `bool` | `False` | If True, keep the latent score matrix (`x_scores_`) after fit. |

## Explanations

### Bibliographic source

Trygg, J. & Wold, S. (2002). *Orthogonal projections to latent structures (O-PLS)*. Journal of Chemometrics 16(3), 119–128.

### Mathematical principle

OPLS rotates the standard PLS latent space so that a single direction captures all $\mathbf{Y}$-correlated variation while the remaining components capture $\mathbf{Y}$-orthogonal structural variation in $\mathbf{X}$. The resulting decomposition $\mathbf{X} = \mathbf{t}_p\mathbf{p}_p^{\top} + \mathbf{T}_o\mathbf{P}_o^{\top} + \mathbf{E}$ separates the **predictive component** $\mathbf{t}_p$ from the orthogonal block $\mathbf{T}_o$, which absorbs spectroscopic baselines, scatter and other nuisance factors that confound interpretation of the predictive loading.

Numerically OPLS proceeds by NIPALS-deflating $\mathbf{X}$ against directions orthogonal to $\mathbf{X}^{\top}\mathbf{y}$ before each new predictive component is extracted. Predictions are identical to those of a one-component PLS on the orthogonal-corrected $\mathbf{X}$; the value is in **the interpretation of the loadings**, not in better predictions per se.

OPLS shines in metabolomics and process spectroscopy where the spectra carry strong systematic but non-predictive variation; in those settings the single-vector predictive loading is far easier to relate to biology / chemistry than a multi-component PLS loading matrix.

### Implementation

`Algorithm.OPLS` + `Solver.NIPALS` + `Deflation.ORTHOGONAL`. Reference: Bioconductor `ropls::opls`. Note: orthogonal-component ordering and the criterion that stops orthogonal extraction differ between implementations — exact bit parity is not expected, but RMSE-rel parity within ~1e-3 is.

R roxygen note (`sklearn.R::opls`):

> Formula-based OPLS regression wrapper around the pls4all C ABI.

MATLAB header (`bindings/matlab/+pls4all/OplsRegression.m`):

```text
pls4all.OplsRegression — Orthogonal Partial Least Squares Regression model.

 Example:

     mdl = pls4all.OplsRegression(X, y, 5);
     yhat = predict(mdl, Xnew);
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
/* C ABI — libn4m (Model.fit path) */
n4m_context_t* ctx = n4m_context_create();
n4m_config_t*  cfg = n4m_config_create();
n4m_config_set_algorithm(cfg, N4M_ALGORITHM_OPLS);
n4m_config_set_solver   (cfg, N4M_SOLVER_NIPALS);
n4m_config_set_n_components(cfg, 4);
n4m_model_t* mdl = NULL;
n4m_model_fit(ctx, cfg, &x_view, &y_view, &mdl);
n4m_model_predict(ctx, mdl, &x_test_view, &y_hat_view);
n4m_model_destroy(mdl);
n4m_config_destroy(cfg);
n4m_context_destroy(ctx);
```

:::

:::{tab-item} Python · pls4all (raw)
:sync: python-raw
:class-label: lang-python

```python
import pls4all
from pls4all import Algorithm, Solver
with pls4all.Context() as ctx, pls4all.Config() as cfg:
    cfg.algorithm = Algorithm.OPLS
    cfg.solver = Solver.NIPALS
    cfg.n_components = 4
    with pls4all.Model.fit(ctx, cfg, X, y) as mdl:
        y_hat = mdl.predict(X_test)
```

:::

:::{tab-item} Python · pls4all.sklearn
:sync: python-sklearn
:class-label: lang-python

```python
from pls4all.sklearn import OPLSRegression
mdl = OPLSRegression(n_components=2, solver='nipals', center_x=True, scale_x=True, center_y=True, scale_y=False, tol=1e-06, max_iter=500, store_scores=False)
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
res <- pls4all_method("opls", X, y,
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
fit  <- opls(y ~ ., data = train, ncomp = 4L)
yhat <- predict(fit, newdata = test)
summary(fit)
```

:::

:::{tab-item} MATLAB · pls4all (MEX)
:sync: matlab-mex
:class-label: lang-matlab

```matlab
res = pls4all.opls(X, y, 4);
% see header of bindings/matlab/+pls4all/opls.m for full
% parameter surface:
%   [coefs, x_mean, y_mean, predictions] = opls(X, Y, n_components)
yhat = predict(res, Xtest);
```

:::

:::{tab-item} MATLAB · pls4all (classdef)
:sync: matlab-classdef
:class-label: lang-matlab

```matlab
mdl  = pls4all.fit("opls", X, y, "NumComponents", 4);
yhat = predict(mdl, Xtest);
```

:::

::::


**Registry parity references** 📐

:::{card}
:class-card: external-refs

- 📐 **`ref.r_ropls`** (R · r) — `ropls` Bioc · relaxed (rmse_rel ≤ 1e-03) — Bioconductor `ropls::opls` — OPLS / OPLS-DA reference.
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
<tr class="bk-row"><td class="bk-name"><code>pls4all.cpp.blas</code></td><td class="parity parity-exact">✓ 1e-14</td><td class="ms">1.13 ms</td></tr>
<tr class="bk-row"><td class="bk-name"><code>pls4all.cpp.blas+omp</code></td><td class="parity parity-exact">✓ 2e-15</td><td class="ms">1.06 ms</td></tr>
<tr class="bk-row"><td class="bk-name"><code>pls4all.cpp.omp</code></td><td class="parity parity-exact">✓ 2e-15</td><td class="ms">1.03 ms</td></tr>
<tr class="bk-row"><td class="bk-name"><code>pls4all.cpp.ref</code></td><td class="parity parity-exact">✓ 2e-15</td><td class="ms">1.03 ms</td></tr>
</tbody>
<tbody class="lang-band lang-python"><tr class="lang-band-row" data-lang="python"><th colspan="3" scope="rowgroup"><span class="lang-band-dot"></span>Python · pls4all</th></tr>
<tr class="bk-row"><td class="bk-name"><code>pls4all.python</code></td><td class="parity parity-exact">✓ bind</td><td class="ms">1.07 ms</td></tr>
<tr class="bk-row"><td class="bk-name"><code>pls4all.registry</code></td><td class="parity parity-exact">✓ bind</td><td class="ms ms-best">0.98 ms<span class="medal" title="fastest">🏆</span></td></tr>
<tr class="bk-row"><td class="bk-name"><code>pls4all.sklearn</code></td><td class="parity parity-exact">✓ bind</td><td class="ms">1.20 ms</td></tr>
</tbody>
<tbody class="lang-band lang-r"><tr class="lang-band-row" data-lang="r"><th colspan="3" scope="rowgroup"><span class="lang-band-dot"></span>R · pls4all</th></tr>
<tr class="bk-row"><td class="bk-name"><code>pls4all.R</code></td><td class="parity parity-exact">✓ 3e-14</td><td class="ms">2.49 ms</td></tr>
<tr class="bk-row"><td class="bk-name"><code>pls4all.R.formula</code></td><td class="parity parity-exact">✓ 3e-14</td><td class="ms">3.74 ms</td></tr>
</tbody>
<tbody class="lang-band lang-matlab"><tr class="lang-band-row" data-lang="matlab"><th colspan="3" scope="rowgroup"><span class="lang-band-dot"></span>MATLAB · pls4all</th></tr>
<tr class="bk-row"><td class="bk-name"><code>pls4all.matlab</code></td><td class="parity parity-exact">✓ 3e-14</td><td class="ms">1.78 ms</td></tr>
<tr class="bk-row"><td class="bk-name"><code>pls4all.matlab.classdef</code></td><td class="parity parity-exact">✓ 3e-14</td><td class="ms">1.92 ms</td></tr>
</tbody>
<tbody class="lang-band lang-python"><tr class="lang-band-row" data-lang="python"><th colspan="3" scope="rowgroup"><span class="lang-band-dot"></span>Python · external</th></tr>
<tr class="bk-row"><td class="bk-name"><code>ikpls</code></td><td class="parity parity-not_available">⊘</td><td class="ms">—</td></tr>
<tr class="bk-row"><td class="bk-name"><code>sklearn</code></td><td class="parity parity-not_available">⊘</td><td class="ms">—</td></tr>
</tbody>
<tbody class="lang-band lang-r"><tr class="lang-band-row" data-lang="r"><th colspan="3" scope="rowgroup"><span class="lang-band-dot"></span>R · external</th></tr>
<tr class="bk-row"><td class="bk-name"><code>mixOmics</code></td><td class="parity parity-not_available">⊘</td><td class="ms">—</td></tr>
<tr class="bk-row"><td class="bk-name"><code>pls</code></td><td class="parity parity-not_available">⊘</td><td class="ms">—</td></tr>
<tr class="bk-row truth-source truth-source-relaxed"><td class="bk-name"><span class="truth-mark" title="Registry parity reference (r): ropls Bioc — relaxed (rmse_rel ≤ 1e-03)">📐</span><code>ref.r_ropls</code></td><td class="parity parity-exact">✓ ref</td><td class="ms">132.7 ms</td></tr>
<tr class="bk-row"><td class="bk-name"><code>ropls</code></td><td class="parity parity-not_run">—</td><td class="ms">—</td></tr>
</tbody>
<tbody class="lang-band lang-matlab"><tr class="lang-band-row" data-lang="matlab"><th colspan="3" scope="rowgroup"><span class="lang-band-dot"></span>MATLAB · external</th></tr>
<tr class="bk-row"><td class="bk-name"><code>plsregress</code></td><td class="parity parity-not_run">—</td><td class="ms">—</td></tr>
</tbody>
<tbody class="lang-band lang-ext"><tr class="lang-band-row" data-lang="ext"><th colspan="3" scope="rowgroup"><span class="lang-band-dot"></span>Other</th></tr>
<tr class="bk-row"><td class="bk-name"><code>r_mdatools_compat</code></td><td class="parity parity-exact">✓ bind</td><td class="ms">3.29 ms</td></tr>
<tr class="bk-row"><td class="bk-name"><code>r_pls_compat</code></td><td class="parity parity-exact">✓ bind</td><td class="ms">3.37 ms</td></tr>
</tbody>
</table>
</div>

:::

::::


---

_See also_: [benchmark overview](../benchmarks/overview.md) · [methods index](index.md) · [interactive dashboard](../landing/dashboard.md)