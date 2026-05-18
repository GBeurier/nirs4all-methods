# `pls_qda` — PLS-QDA

_Group_: **Classification & GLM** · _Registry tolerance_: `10.0`

## Description

PLS-QDA (§5) — quadratic discriminant on PLS scores

From the `pls4all.sklearn.PLSQDAClassifier` docstring:

> PLS-QDA on PLS scores (in-sample only).

> **Registry note** — sklearn `PLSRegression -> QuadraticDiscriminantAnalysis` pipeline is the closest external reference for PLS-QDA. pls4all returns discriminant scores (centered class responses) whereas sklearn QDA returns probabilities — the score scales differ wildly, hence the loose tolerance.

### Parameters

| Name | Type | Default | Notes |
|------|------|---------|-------|
| `n_components` | `int` | `2` | Number of latent components extracted (k). |
| `n_classes` | `int` | `3` | registry benchmark cell value |

## Explanations

### Bibliographic source

Pérez-Enciso, M. & Tenenhaus, M. (2003). *Prediction of clinical outcome with microarray data: a partial least squares discriminant analysis (PLS-DA) approach*. Human Genetics 112(5–6), 581–592.

### Mathematical principle

Replace LDA with QDA in the second stage of PLS-LDA: instead of assuming a shared covariance across classes, fit a per-class covariance $\boldsymbol{\Sigma}_c$ on the latent scores. The resulting decision rule $\hat{c}(\mathbf{x}) = \arg\min_c (\mathbf{t}(\mathbf{x}) - \boldsymbol{\mu}_c)^{\top} \boldsymbol{\Sigma}_c^{-1} (\mathbf{t}(\mathbf{x}) - \boldsymbol{\mu}_c) + \log|\boldsymbol{\Sigma}_c|$ is **quadratic** in the latent scores.

QDA needs at least $k + 1$ samples per class to estimate $\boldsymbol{\Sigma}_c$ stably, but otherwise gives more flexible decision boundaries than LDA. Worth trying whenever the LDA boundary visibly under-fits in a 2-D latent score plot.

Class probabilities follow from the Mahalanobis distance via the Bayes rule with uniform priors (or user-supplied priors).

### Implementation

`p4a_pls_qda_fit`. Reference: composite PLSRegression + sklearn `QuadraticDiscriminantAnalysis` on the scores.

R roxygen note (`methods_extra.R::pls_qda_fit`):

> PLS-QDA (Quadratic Discriminant Analysis on PLS scores).
> @param y_labels Integer vector. Class labels.
> @param n_components Integer. Number of latent components.
> @param X Numeric matrix of predictors (rows = samples, cols = features).
> @export

MATLAB header (`bindings/matlab/+pls4all/pls_qda.m`):

```text
pls4all.pls_qda  Quadratic Discriminant Analysis on PLS scores.
 y_labels: integer class IDs in {0, …, n_classes-1}.
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
p4a_PLSRegression+QDA(ctx, cfg, &x_view, &y_view, /* hyperparams */, &res);
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
from pls4all._methods import PLSRegression+QDA
with pls4all.Context() as ctx, pls4all.Config() as cfg:
    res = PLSRegression+QDA(ctx, cfg, X, y, n_components=4, y_labels=y_labels)
# then: res.matrix("predictions"), res.matrix("coefficients"),
# res.vector("mask"), res.scalar("intercept"), …
```

:::

:::{tab-item} Python · pls4all.sklearn
:sync: python-sklearn
:class-label: lang-python

```python
from pls4all.sklearn import PLSQDAClassifier
mdl = PLSQDAClassifier(n_components=2)
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
res <- pls4all_method("pls_qda", X, y,
                      n_components = 4L, params = list(n_classes = 3L))
# res is a named list with MethodResult arrays/scalars.
# selected_indices / top_k_intervals are 1-based.
```

:::

:::{tab-item} R · pls4all (raw fn)
:sync: r-raw
:class-label: lang-r

```r
library(pls4all)
res  <- pls_qda_fit(X, y_labels, n_components)
yhat <- pls4all_predict(res, X_test)
```

:::

:::{tab-item} MATLAB · pls4all (MEX)
:sync: matlab-mex
:class-label: lang-matlab

```matlab
res = pls4all.pls_qda(X, y, 4);
% see header of bindings/matlab/+pls4all/pls_qda.m for full
% parameter surface:
%   res = pls_qda(X, y_labels, n_components)
yhat = predict(res, Xtest);
```

:::

:::{tab-item} MATLAB · pls4all (classdef)
:sync: matlab-classdef
:class-label: lang-matlab

_No idiomatic classdef wrapper — invoke `pls4all.fit("pls_qda", X, y, …)` directly from the unified MEX factory._

:::

::::


**Registry parity references** 📐

:::{card}
:class-card: external-refs

- 📐 **`ref.python_scikit_learn`** (python · python) — `scikit-learn` 1.8.0 · qualitative (rmse_rel ≤ 1e+01) — sklearn `PLSRegression -> QuadraticDiscriminantAnalysis` pipeline. pls4all wires the QDA covariance estimation internally; sklearn does it externally on the PLS scores. Class-decision boundaries align but the soft scores differ.
:::

### Benchmarks

Median wall-clock per cell from [`benchmarks/cross_binding/results/full_matrix.csv`](../benchmarks/overview.md). Verdict legend: ✓ exact · ≈ drift · ✗ divergent · ⊘ not available in lib · — not run · ⚠ error. The fastest backend per column is marked with a 🏆 medal. Rows tagged with **📐** are *also* declared in [`benchmarks/parity_timing/registry.py`](../benchmarks/methodology.md) as the canonical parity references for this method (`python_reference` / `r_reference` / `extra_references`). Cell parity in this table is measured against the cross-binding reference (`pls4all.cpp` blas-omp, 1 thread); the 📐 icon points at the *library-of-record* the parity gate ultimately answers to. Hover the icon to see the role and tolerance band.

::::{tab-set}
:class: parity-tabs

:::{tab-item} 1 thread
:sync: threads-1

<div class="parity-table-wrap">
<table class="docutils parity-grouped">
<thead><tr><th scope="col">Backend</th><th scope="col">Parity</th><th class="size-col" scope="col">100×50 (ms)</th><th class="size-col" scope="col">200×30 (ms)</th><th class="size-col" scope="col">500×50 (ms)</th></tr></thead>
<tbody class="lang-band lang-cpp"><tr class="lang-band-row" data-lang="cpp"><th colspan="5" scope="rowgroup"><span class="lang-band-dot"></span>C++ native · libp4a</th></tr>
<tr class="bk-row"><td class="bk-name"><code>pls4all.cpp.blas</code></td><td class="parity parity-drift">≈</td><td class="ms">1.05 ms</td><td class="ms ms-empty">—</td><td class="ms">4.83 ms</td></tr>
<tr class="bk-row"><td class="bk-name"><code>pls4all.cpp.blas+omp</code></td><td class="parity parity-drift">≈</td><td class="ms ms-best">0.97 ms<span class="medal" title="fastest">🏆</span></td><td class="ms ms-empty">—</td><td class="ms">4.62 ms</td></tr>
<tr class="bk-row"><td class="bk-name"><code>pls4all.cpp.omp</code></td><td class="parity parity-drift">≈</td><td class="ms">1.02 ms</td><td class="ms ms-empty">—</td><td class="ms">4.94 ms</td></tr>
<tr class="bk-row"><td class="bk-name"><code>pls4all.cpp.ref</code></td><td class="parity parity-drift">≈</td><td class="ms">2.03 ms</td><td class="ms ms-empty">—</td><td class="ms">6.39 ms</td></tr>
</tbody>
<tbody class="lang-band lang-python"><tr class="lang-band-row" data-lang="python"><th colspan="5" scope="rowgroup"><span class="lang-band-dot"></span>Python · pls4all</th></tr>
<tr class="bk-row"><td class="bk-name"><code>pls4all.python</code></td><td class="parity parity-drift">≈</td><td class="ms">1.07 ms</td><td class="ms ms-empty">—</td><td class="ms ms-best">4.47 ms<span class="medal" title="fastest">🏆</span></td></tr>
<tr class="bk-row"><td class="bk-name"><code>pls4all.registry</code></td><td class="parity parity-drift">≈</td><td class="ms">1.43 ms</td><td class="ms ms-best">164.7 ms<span class="medal" title="fastest">🏆</span></td><td class="ms">5.28 ms</td></tr>
<tr class="bk-row"><td class="bk-name"><code>pls4all.sklearn</code></td><td class="parity parity-not_available">⊘</td><td class="ms">—</td><td class="ms ms-empty">—</td><td class="ms">—</td></tr>
</tbody>
<tbody class="lang-band lang-r"><tr class="lang-band-row" data-lang="r"><th colspan="5" scope="rowgroup"><span class="lang-band-dot"></span>R · pls4all</th></tr>
<tr class="bk-row"><td class="bk-name"><code>pls4all.R</code></td><td class="parity parity-not_run">—</td><td class="ms">—</td><td class="ms ms-empty">—</td><td class="ms">—</td></tr>
<tr class="bk-row"><td class="bk-name"><code>pls4all.R.formula</code></td><td class="parity parity-not_run">—</td><td class="ms">—</td><td class="ms ms-empty">—</td><td class="ms">—</td></tr>
</tbody>
<tbody class="lang-band lang-matlab"><tr class="lang-band-row" data-lang="matlab"><th colspan="5" scope="rowgroup"><span class="lang-band-dot"></span>MATLAB · pls4all</th></tr>
<tr class="bk-row"><td class="bk-name"><code>pls4all.matlab</code></td><td class="parity parity-not_run">—</td><td class="ms">—</td><td class="ms ms-empty">—</td><td class="ms">—</td></tr>
<tr class="bk-row"><td class="bk-name"><code>pls4all.matlab.classdef</code></td><td class="parity parity-not_run">—</td><td class="ms">—</td><td class="ms ms-empty">—</td><td class="ms">—</td></tr>
</tbody>
<tbody class="lang-band lang-python"><tr class="lang-band-row" data-lang="python"><th colspan="5" scope="rowgroup"><span class="lang-band-dot"></span>Python · external</th></tr>
<tr class="bk-row"><td class="bk-name"><code>ikpls</code></td><td class="parity parity-not_available">⊘</td><td class="ms">—</td><td class="ms ms-empty">—</td><td class="ms">—</td></tr>
<tr class="bk-row truth-source truth-source-qualitative"><td class="bk-name"><span class="truth-mark" title="Registry parity reference (python): scikit-learn 1.8.0 — qualitative (rmse_rel ≤ 1e+01)">📐</span><code>ref.python_scikit_learn</code></td><td class="parity parity-drift">≈</td><td class="ms">2.49 ms</td><td class="ms">584.5 ms</td><td class="ms">6.43 ms</td></tr>
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