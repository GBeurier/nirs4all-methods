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

:::{tab-item} C ABI · libp4a
:sync: c
:class-label: lang-c

```c
/* C ABI — libp4a (Model.fit path) */
p4a_context_t* ctx = p4a_context_create();
p4a_config_t*  cfg = p4a_config_create();
p4a_config_set_algorithm(cfg, P4A_ALGORITHM_OPLS);
p4a_config_set_solver   (cfg, P4A_SOLVER_NIPALS);
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

- 📐 **`ref.r_ropls`** (R · r) — `ropls` Bioc · relaxed (rmse_rel ≤ 1e-03) — Bioconductor `ropls::opls` — OPLS reference. Permutations and plotting are disabled in benchmark timing; ropls still requires crossvalI >= 1 for a finite Q2 path.
:::

### Validation contract

Structural validation contract for `opls`, exported by the declarative validation framework (`benchmarks/validation/registry/`) from `benchmarks.parity_timing.registry`. Regenerated by `python -m benchmarks.validation.export_registry --write`.

| Field | Value |
|------|-------|
| Reference kind | `external` |
| Canonical reference role | `r` |
| Declared references | R |
| Comparator | `reference_parity` (external-reference validity gate) |
| Comparator metric | `rmse_rel` |
| Reference tolerance | `1e-03` (relaxed) |
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

**Reference gate**: strict release — `rmse_rel ≤ 1e-03` is required; `rmse_rel ≤ 1e-06` is displayed as exact.

Rows tagged with **📐** are the canonical parity references for this method (declared in [`parity_timing.registry`](../benchmarks/methodology.md)). C++ and external rows show reference parity; pls4all language bindings show binding parity against the C++ backend. Hover the icon for role and tolerance band.

::::{tab-set}
:class: parity-tabs

:::{tab-item} 1 thread
:sync: threads-1

<div class="parity-table-wrap">
<table class="docutils parity-grouped">
<thead><tr><th scope="col">Backend</th><th scope="col">Divergence</th><th class="size-col" scope="col">100×50 (ms)</th><th class="size-col" scope="col">100×500 (ms)</th><th class="size-col" scope="col">100×2500 (ms)</th><th class="size-col" scope="col">1000×500 (ms)</th></tr></thead>
<tbody class="lang-band lang-cpp"><tr class="lang-band-row" data-lang="cpp"><th colspan="6" scope="rowgroup"><span class="lang-band-dot"></span>C++ native · libp4a</th></tr>
<tr class="bk-row"><td class="bk-name"><code>pls4all.cpp.blas</code></td><td class="parity parity-divergence parity-ref-strict">3e-14</td><td class="ms">1.05 ms</td><td class="ms">9.33 ms</td><td class="ms">91.4 ms</td><td class="ms">88.1 ms</td></tr>
<tr class="bk-row"><td class="bk-name"><code>pls4all.cpp.blas+omp</code></td><td class="parity parity-divergence parity-ref-strict">3e-14</td><td class="ms">1.05 ms</td><td class="ms ms-best">9.11 ms<span class="medal" title="fastest">🏆</span></td><td class="ms ms-best">91.4 ms<span class="medal" title="fastest">🏆</span></td><td class="ms">89.0 ms</td></tr>
<tr class="bk-row"><td class="bk-name"><code>pls4all.cpp.omp</code></td><td class="parity parity-divergence parity-ref-strict">1e-13</td><td class="ms">1.09 ms</td><td class="ms">9.32 ms</td><td class="ms">91.8 ms</td><td class="ms">89.5 ms</td></tr>
<tr class="bk-row"><td class="bk-name"><code>pls4all.cpp.ref</code></td><td class="parity parity-divergence parity-ref-strict">1e-13</td><td class="ms ms-best">1.02 ms<span class="medal" title="fastest">🏆</span></td><td class="ms">9.38 ms</td><td class="ms">91.5 ms</td><td class="ms">89.0 ms</td></tr>
</tbody>
<tbody class="lang-band lang-python"><tr class="lang-band-row" data-lang="python"><th colspan="6" scope="rowgroup"><span class="lang-band-dot"></span>Python · pls4all</th></tr>
<tr class="bk-row"><td class="bk-name"><code>pls4all.python</code></td><td class="parity parity-divergence parity-ref-strict">3e-14</td><td class="ms">1.35 ms</td><td class="ms">9.16 ms</td><td class="ms">93.2 ms</td><td class="ms ms-best">87.8 ms<span class="medal" title="fastest">🏆</span></td></tr>
<tr class="bk-row"><td class="bk-name"><code>pls4all.sklearn</code></td><td class="parity parity-divergence parity-ref-strict">8e-13</td><td class="ms">1.24 ms</td><td class="ms">10.0 ms</td><td class="ms">94.6 ms</td><td class="ms">88.6 ms</td></tr>
</tbody>
<tbody class="lang-band lang-r"><tr class="lang-band-row" data-lang="r"><th colspan="6" scope="rowgroup"><span class="lang-band-dot"></span>R · pls4all</th></tr>
<tr class="bk-row"><td class="bk-name"><code>pls4all.R</code></td><td class="parity parity-divergence parity-ref-strict">2e-13</td><td class="ms">3.12 ms</td><td class="ms">48.5 ms</td><td class="ms">478.5 ms</td><td class="ms">518.2 ms</td></tr>
<tr class="bk-row"><td class="bk-name"><code>pls4all.R.formula</code></td><td class="parity parity-divergence parity-ref-strict">2e-13</td><td class="ms">4.64 ms</td><td class="ms">63.0 ms</td><td class="ms">616.2 ms</td><td class="ms">559.6 ms</td></tr>
<tr class="bk-row"><td class="bk-name"><code>pls4all.R.mdatools</code></td><td class="parity parity-divergence parity-ref-strict">2e-13</td><td class="ms">4.50 ms</td><td class="ms">62.2 ms</td><td class="ms">611.9 ms</td><td class="ms">577.1 ms</td></tr>
<tr class="bk-row"><td class="bk-name"><code>pls4all.R.pls</code></td><td class="parity parity-divergence parity-ref-strict">2e-13</td><td class="ms">4.60 ms</td><td class="ms">57.0 ms</td><td class="ms">607.7 ms</td><td class="ms">628.3 ms</td></tr>
</tbody>
<tbody class="lang-band lang-matlab"><tr class="lang-band-row" data-lang="matlab"><th colspan="6" scope="rowgroup"><span class="lang-band-dot"></span>MATLAB · pls4all</th></tr>
<tr class="bk-row"><td class="bk-name"><code>pls4all.matlab</code></td><td class="parity parity-divergence parity-ref-strict">2e-13</td><td class="ms">1.83 ms</td><td class="ms">14.3 ms</td><td class="ms">124.9 ms</td><td class="ms">138.3 ms</td></tr>
<tr class="bk-row"><td class="bk-name"><code>pls4all.matlab.classdef</code></td><td class="parity parity-divergence parity-ref-strict">2e-13</td><td class="ms">2.08 ms</td><td class="ms">14.7 ms</td><td class="ms">117.4 ms</td><td class="ms">138.3 ms</td></tr>
</tbody>
<tbody class="lang-band lang-r"><tr class="lang-band-row" data-lang="r"><th colspan="6" scope="rowgroup"><span class="lang-band-dot"></span>R · external</th></tr>
<tr class="bk-row truth-source truth-source-relaxed"><td class="bk-name"><span class="truth-mark" title="Registry parity reference (r): ropls Bioc — relaxed (rmse_rel ≤ 1e-03)">📐</span><code>ref.r_ropls</code></td><td class="parity parity-divergence parity-ref-source">0</td><td class="ms">14.6 ms</td><td class="ms">83.9 ms</td><td class="ms">598.7 ms</td><td class="ms">933.6 ms</td></tr>
</tbody>
</table>
</div>

:::

::::


---

_See also_: [benchmark overview](../benchmarks/overview.md) · [methods index](index.md) · [interactive dashboard](../landing/dashboard.md)