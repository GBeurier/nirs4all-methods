# `pcr` — Principal Components Regression

_Group_: **Core PLS** · _Registry tolerance_: `1e-06`

## Description

Principal Components Regression

From the `pls4all.sklearn.PCR` docstring:

> Principal Components Regression — fits a least-squares regression
on the SVD of X.

> **Registry note** — PCR via SVD on X then linear regression; references are sklearn Pipeline(PCA + LinearRegression) and R `pls::pcr`.

### Parameters

| Name | Type | Default | Notes |
|------|------|---------|-------|
| `n_components` | `int` | `2` | Number of latent components extracted (k). |
| `center_x` | `bool` | `True` | Subtract the column mean of X before fitting. |
| `scale_x` | `bool` | `True` | Standardize X columns to unit variance before fitting. |
| `center_y` | `bool` | `True` | Subtract the column mean of y before fitting. |
| `scale_y` | `bool` | `False` | Standardize y columns to unit variance before fitting. |
| `tol` | `float` | `1e-06` | Convergence tolerance for iterative solvers (NIPALS / power-iteration). |
| `max_iter` | `int` | `500` | Maximum iterations for iterative solvers. |
| `store_scores` | `bool` | `False` | If True, keep the latent score matrix (`x_scores_`) after fit. |

## Explanations

### Bibliographic source

Massy, W. F. (1965). *Principal Components Regression in Exploratory Statistical Research*. JASA 60(309), 234–256.

### Mathematical principle

PCR sidesteps the multicollinearity of $\mathbf{X}$ by regressing on its orthogonal principal-component scores rather than on the raw columns. The factorisation $\mathbf{X} = \mathbf{U}\boldsymbol{\Sigma}\mathbf{V}^{\top}$ (SVD) yields scores $\mathbf{T}_k = \mathbf{U}_k\boldsymbol{\Sigma}_k$ for the top $k$ components, and the regression $\mathbf{Y} = \mathbf{T}_k\mathbf{Q}_k + \mathbf{E}$ is fit by ordinary least squares.

Unlike PLS, PCR is **unsupervised in its dimensionality reduction**: the first $k$ directions maximise the variance of $\mathbf{X}$ regardless of how relevant they are to $\mathbf{Y}$. This makes PCR a useful baseline for diagnosing whether the predictive directions in a calibration set really do coincide with the high-variance directions (in which case PCR ≈ PLS) or not (in which case PLS is strictly preferable at the same $k$).

Coefficients in the original feature scale are recovered as $\mathbf{B} = \mathbf{V}_k \boldsymbol{\Sigma}_k^{-1} \mathbf{T}_k^{\top}\mathbf{Y}$. Total cost is dominated by the partial SVD: $O(np\min(n,p))$ for a full decomposition, or $O(npk)$ with a truncated method (Lanczos, randomised SVD).

### Implementation

`Algorithm.PCR` + `Solver.SVD` in libp4a. Reference implementations are scikit-learn's `Pipeline(PCA(n_components=k), LinearRegression())` and R `pls::pcr`.

MATLAB header (`bindings/matlab/+pls4all/PcrRegression.m`):

```text
pls4all.PcrRegression — Principal Component Regression model.

 Example:

     mdl = pls4all.PcrRegression(X, y, 5);
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
p4a_config_set_algorithm(cfg, P4A_ALGORITHM_PCR);
p4a_config_set_solver   (cfg, P4A_SOLVER_SVD);
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
    cfg.algorithm = Algorithm.PCR
    cfg.solver = Solver.SVD
    cfg.n_components = 4
    with pls4all.Model.fit(ctx, cfg, X, y) as mdl:
        y_hat = mdl.predict(X_test)
```

:::

:::{tab-item} Python · pls4all.sklearn
:sync: python-sklearn
:class-label: lang-python

```python
from pls4all.sklearn import PCR
mdl = PCR(n_components=2, center_x=True, scale_x=True, center_y=True, scale_y=False, tol=1e-06, max_iter=500, store_scores=False)
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
res <- pls4all_method("pcr", X, y,
                      n_components = 4L)
# res is a named list with MethodResult arrays/scalars.
# selected_indices / top_k_intervals are 1-based.
```

:::

:::{tab-item} R · `pls` package compat
:sync: r-pls-compat
:class-label: lang-r

```r
library(pls4all)
# Drop-in for CRAN `pls::pcr` (same signature).
fit  <- pcr(y ~ ., ncomp = 4L, data = train,
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
# Drop-in for `mdatools::pls(x, y, ncomp, method = "pcr")`.
fit  <- pls_mdatools(X, y, ncomp = 4L, method = "pcr",
               center = TRUE, scale = FALSE)
yhat <- predict(fit, newdata = X_test, ncomp = 4L)
```

:::

:::{tab-item} MATLAB · pls4all (MEX)
:sync: matlab-mex
:class-label: lang-matlab

```matlab
res = pls4all.pcr(X, y, 4);
% see header of bindings/matlab/+pls4all/pcr.m for full
% parameter surface:
%   [coefs, x_mean, y_mean, predictions] = pcr(X, Y, n_components)
yhat = predict(res, Xtest);
```

:::

:::{tab-item} MATLAB · pls4all (classdef)
:sync: matlab-classdef
:class-label: lang-matlab

```matlab
mdl  = pls4all.fit("pcr", X, y, "NumComponents", 4);
yhat = predict(mdl, Xtest);
```

:::

::::


**Registry parity references** 📐

:::{card}
:class-card: external-refs

- 📐 **`ref.python_scikit_learn`** (python · python) — `scikit-learn` 1.8.0 · strict (rmse_rel ≤ 1e-06) — sklearn Pipeline(PCA + LinearRegression).
- 📐 **`ref.r_pls`** (R · r) — `pls` 2.8.5 · strict (rmse_rel ≤ 1e-06) — R pls::pcr(scale=FALSE).
:::

### Validation contract

Structural validation contract for `pcr`, exported by the declarative validation framework (`benchmarks/validation/registry/`) from `benchmarks.parity_timing.registry`. Regenerated by `python -m benchmarks.validation.export_registry --write`.

| Field | Value |
|------|-------|
| Reference kind | `external` |
| Canonical reference role | `python` |
| Declared references | Python, R |
| Comparator | `reference_parity` (external-reference validity gate) |
| Comparator metric | `rmse_rel` |
| Reference tolerance | `1e-06` (strict) |
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

**Reference gate**: strict release — `rmse_rel ≤ 1e-06` is required; `rmse_rel ≤ 1e-06` is displayed as exact.

Rows tagged with **📐** are the canonical parity references for this method (declared in [`parity_timing.registry`](../benchmarks/methodology.md)). C++ and external rows show reference parity; pls4all language bindings show binding parity against the C++ backend. Hover the icon for role and tolerance band.

::::{tab-set}
:class: parity-tabs

:::{tab-item} 1 thread
:sync: threads-1

<div class="parity-table-wrap">
<table class="docutils parity-grouped">
<thead><tr><th scope="col">Backend</th><th scope="col">Divergence</th><th class="size-col" scope="col">100×50 (ms)</th><th class="size-col" scope="col">100×500 (ms)</th><th class="size-col" scope="col">100×2500 (ms)</th></tr></thead>
<tbody class="lang-band lang-cpp"><tr class="lang-band-row" data-lang="cpp"><th colspan="5" scope="rowgroup"><span class="lang-band-dot"></span>C++ native · libp4a</th></tr>
<tr class="bk-row"><td class="bk-name"><code>pls4all.cpp.blas</code></td><td class="parity parity-divergence parity-ref-strict">5e-12</td><td class="ms">1.85 ms</td><td class="ms">992.7 ms</td><td class="ms">233.8 s</td></tr>
<tr class="bk-row"><td class="bk-name"><code>pls4all.cpp.blas+omp</code></td><td class="parity parity-divergence parity-ref-strict">5e-12</td><td class="ms">1.89 ms</td><td class="ms">1.0 s</td><td class="ms">230.6 s</td></tr>
<tr class="bk-row"><td class="bk-name"><code>pls4all.cpp.omp</code></td><td class="parity parity-divergence parity-ref-strict">5e-12</td><td class="ms ms-best">1.79 ms<span class="medal" title="fastest">🏆</span></td><td class="ms">1.1 s</td><td class="ms">224.3 s</td></tr>
<tr class="bk-row"><td class="bk-name"><code>pls4all.cpp.ref</code></td><td class="parity parity-divergence parity-ref-strict">5e-12</td><td class="ms">1.94 ms</td><td class="ms">1.0 s</td><td class="ms ms-best">202.7 s<span class="medal" title="fastest">🏆</span></td></tr>
</tbody>
<tbody class="lang-band lang-python"><tr class="lang-band-row" data-lang="python"><th colspan="5" scope="rowgroup"><span class="lang-band-dot"></span>Python · pls4all</th></tr>
<tr class="bk-row"><td class="bk-name"><code>pls4all.python</code></td><td class="parity parity-divergence parity-drift">5e-12</td><td class="ms">1.87 ms</td><td class="ms">985.8 ms</td><td class="ms">276.4 s</td></tr>
<tr class="bk-row"><td class="bk-name"><code>pls4all.sklearn</code></td><td class="parity parity-divergence parity-drift">5e-12</td><td class="ms">2.12 ms</td><td class="ms">991.5 ms</td><td class="ms">247.6 s</td></tr>
</tbody>
<tbody class="lang-band lang-r"><tr class="lang-band-row" data-lang="r"><th colspan="5" scope="rowgroup"><span class="lang-band-dot"></span>R · pls4all</th></tr>
<tr class="bk-row"><td class="bk-name"><code>pls4all.R</code></td><td class="parity parity-divergence parity-drift">2e-12</td><td class="ms">21.7 ms</td><td class="ms">43.4 s</td><td class="ms">12316.3 s</td></tr>
<tr class="bk-row"><td class="bk-name"><code>pls4all.R.formula</code></td><td class="parity parity-divergence parity-ref-strict">2e-12</td><td class="ms">23.9 ms</td><td class="ms">45.3 s</td><td class="ms ms-empty">—</td></tr>
<tr class="bk-row"><td class="bk-name"><code>pls4all.R.mdatools</code></td><td class="parity parity-divergence parity-drift">—</td><td class="ms">24.9 ms</td><td class="ms">44.7 s</td><td class="ms ms-empty">—</td></tr>
<tr class="bk-row"><td class="bk-name"><code>pls4all.R.pls</code></td><td class="parity parity-divergence parity-not_available">—</td><td class="ms">27.5 ms</td><td class="ms">45.2 s</td><td class="ms ms-empty">—</td></tr>
</tbody>
<tbody class="lang-band lang-matlab"><tr class="lang-band-row" data-lang="matlab"><th colspan="5" scope="rowgroup"><span class="lang-band-dot"></span>MATLAB · pls4all</th></tr>
<tr class="bk-row"><td class="bk-name"><code>pls4all.matlab</code></td><td class="parity parity-divergence parity-ref-strict">5e-12</td><td class="ms">2.65 ms</td><td class="ms">1.0 s</td><td class="ms ms-empty">—</td></tr>
<tr class="bk-row"><td class="bk-name"><code>pls4all.matlab.classdef</code></td><td class="parity parity-divergence parity-ref-strict">5e-12</td><td class="ms">3.07 ms</td><td class="ms">1.0 s</td><td class="ms ms-empty">—</td></tr>
</tbody>
<tbody class="lang-band lang-python"><tr class="lang-band-row" data-lang="python"><th colspan="5" scope="rowgroup"><span class="lang-band-dot"></span>Python · external</th></tr>
<tr class="bk-row truth-source truth-source-strict"><td class="bk-name"><span class="truth-mark" title="Registry parity reference (python): scikit-learn 1.8.0 — strict (rmse_rel ≤ 1e-06)">📐</span><code>ref.python_scikit_learn</code></td><td class="parity parity-divergence parity-ref-source">0</td><td class="ms">2.12 ms</td><td class="ms ms-best">15.9 ms<span class="medal" title="fastest">🏆</span></td><td class="ms ms-empty">—</td></tr>
</tbody>
<tbody class="lang-band lang-r"><tr class="lang-band-row" data-lang="r"><th colspan="5" scope="rowgroup"><span class="lang-band-dot"></span>R · external</th></tr>
<tr class="bk-row truth-source truth-source-strict"><td class="bk-name"><span class="truth-mark" title="Registry parity reference (r): pls 2.8.5 — strict (rmse_rel ≤ 1e-06)">📐</span><code>ref.r_pls</code></td><td class="parity parity-divergence parity-ref-strict">4e-15</td><td class="ms">8.13 ms</td><td class="ms">90.7 ms</td><td class="ms ms-empty">—</td></tr>
</tbody>
</table>
</div>

:::

:::{tab-item} 8 threads
:sync: threads-8

<div class="parity-table-wrap">
<table class="docutils parity-grouped">
<thead><tr><th scope="col">Backend</th><th scope="col">Divergence</th><th class="size-col" scope="col">100×50 (ms)</th><th class="size-col" scope="col">100×500 (ms)</th><th class="size-col" scope="col">100×2500 (ms)</th></tr></thead>
<tbody class="lang-band lang-cpp"><tr class="lang-band-row" data-lang="cpp"><th colspan="5" scope="rowgroup"><span class="lang-band-dot"></span>C++ native · libp4a</th></tr>
<tr class="bk-row"><td class="bk-name"><code>pls4all.cpp.blas</code></td><td class="parity parity-divergence parity-ref-strict">5e-12</td><td class="ms">1.98 ms</td><td class="ms">1.1 s</td><td class="ms">204.3 s</td></tr>
<tr class="bk-row"><td class="bk-name"><code>pls4all.cpp.blas+omp</code></td><td class="parity parity-divergence parity-ref-strict">5e-12</td><td class="ms">1.94 ms</td><td class="ms">1.0 s</td><td class="ms">244.1 s</td></tr>
<tr class="bk-row"><td class="bk-name"><code>pls4all.cpp.omp</code></td><td class="parity parity-divergence parity-ref-strict">5e-12</td><td class="ms">1.95 ms</td><td class="ms">1.1 s</td><td class="ms">194.5 s</td></tr>
<tr class="bk-row"><td class="bk-name"><code>pls4all.cpp.ref</code></td><td class="parity parity-divergence parity-ref-strict">5e-12</td><td class="ms ms-best">1.93 ms<span class="medal" title="fastest">🏆</span></td><td class="ms">1.1 s</td><td class="ms ms-best">170.7 s<span class="medal" title="fastest">🏆</span></td></tr>
</tbody>
<tbody class="lang-band lang-python"><tr class="lang-band-row" data-lang="python"><th colspan="5" scope="rowgroup"><span class="lang-band-dot"></span>Python · pls4all</th></tr>
<tr class="bk-row"><td class="bk-name"><code>pls4all.python</code></td><td class="parity parity-divergence parity-drift">5e-12</td><td class="ms">2.18 ms</td><td class="ms">1.1 s</td><td class="ms">272.4 s</td></tr>
<tr class="bk-row"><td class="bk-name"><code>pls4all.sklearn</code></td><td class="parity parity-divergence parity-drift">5e-12</td><td class="ms">2.32 ms</td><td class="ms">1.1 s</td><td class="ms">274.2 s</td></tr>
</tbody>
<tbody class="lang-band lang-r"><tr class="lang-band-row" data-lang="r"><th colspan="5" scope="rowgroup"><span class="lang-band-dot"></span>R · pls4all</th></tr>
<tr class="bk-row"><td class="bk-name"><code>pls4all.R</code></td><td class="parity parity-divergence parity-drift">2e-12</td><td class="ms">21.9 ms</td><td class="ms">43.9 s</td><td class="ms">12348.2 s</td></tr>
<tr class="bk-row"><td class="bk-name"><code>pls4all.R.formula</code></td><td class="parity parity-divergence parity-drift">2e-12</td><td class="ms">23.0 ms</td><td class="ms">43.2 s</td><td class="ms ms-empty">—</td></tr>
<tr class="bk-row"><td class="bk-name"><code>pls4all.R.mdatools</code></td><td class="parity parity-divergence parity-drift">2e-12</td><td class="ms">23.5 ms</td><td class="ms">43.5 s</td><td class="ms ms-empty">—</td></tr>
<tr class="bk-row"><td class="bk-name"><code>pls4all.R.pls</code></td><td class="parity parity-divergence parity-drift">2e-12</td><td class="ms">27.3 ms</td><td class="ms">43.4 s</td><td class="ms ms-empty">—</td></tr>
</tbody>
<tbody class="lang-band lang-matlab"><tr class="lang-band-row" data-lang="matlab"><th colspan="5" scope="rowgroup"><span class="lang-band-dot"></span>MATLAB · pls4all</th></tr>
<tr class="bk-row"><td class="bk-name"><code>pls4all.matlab</code></td><td class="parity parity-divergence parity-ref-strict">5e-12</td><td class="ms">2.62 ms</td><td class="ms">1.0 s</td><td class="ms ms-empty">—</td></tr>
<tr class="bk-row"><td class="bk-name"><code>pls4all.matlab.classdef</code></td><td class="parity parity-divergence parity-ref-strict">5e-12</td><td class="ms">3.08 ms</td><td class="ms">1.0 s</td><td class="ms ms-empty">—</td></tr>
</tbody>
<tbody class="lang-band lang-python"><tr class="lang-band-row" data-lang="python"><th colspan="5" scope="rowgroup"><span class="lang-band-dot"></span>Python · external</th></tr>
<tr class="bk-row truth-source truth-source-strict"><td class="bk-name"><span class="truth-mark" title="Registry parity reference (python): scikit-learn 1.8.0 — strict (rmse_rel ≤ 1e-06)">📐</span><code>ref.python_scikit_learn</code></td><td class="parity parity-divergence parity-ref-source">0</td><td class="ms">3.35 ms</td><td class="ms ms-best">18.0 ms<span class="medal" title="fastest">🏆</span></td><td class="ms ms-empty">—</td></tr>
</tbody>
<tbody class="lang-band lang-r"><tr class="lang-band-row" data-lang="r"><th colspan="5" scope="rowgroup"><span class="lang-band-dot"></span>R · external</th></tr>
<tr class="bk-row truth-source truth-source-strict"><td class="bk-name"><span class="truth-mark" title="Registry parity reference (r): pls 2.8.5 — strict (rmse_rel ≤ 1e-06)">📐</span><code>ref.r_pls</code></td><td class="parity parity-divergence parity-ref-strict">4e-15</td><td class="ms">10.5 ms</td><td class="ms">109.8 ms</td><td class="ms ms-empty">—</td></tr>
</tbody>
</table>
</div>

:::

::::


---

_See also_: [benchmark overview](../benchmarks/overview.md) · [methods index](index.md) · [interactive dashboard](../landing/dashboard.md)