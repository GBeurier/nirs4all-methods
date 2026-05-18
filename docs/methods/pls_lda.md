# `pls_lda` — PLS-LDA

_Group_: **Classification & GLM** · _Registry tolerance_: `5.0`

## Description

PLS-LDA — LDA on PLS scores (§17 Phase 4)

From the `pls4all.sklearn.PLSLDAClassifier` docstring:

> PLS-LDA on PLS scores (in-sample only).

> **Registry note** — sklearn `PLSRegression -> LinearDiscriminantAnalysis` pipeline is the closest installable reference. R `plsVarSel::lda_from_pls` exists but its return shape differs from pls4all's `decision_scores`.

### Parameters

| Name | Type | Default | Notes |
|------|------|---------|-------|
| `n_components` | `int` | `2` | Number of latent components extracted (k). |
| `n_classes` | `int` | `3` | registry benchmark cell value |

## Explanations

### Bibliographic source

Barker, M. & Rayens, W. (2003). *Partial least squares for discrimination*. Journal of Chemometrics 17(3), 166–173.

### Mathematical principle

PLS-LDA is a two-stage classifier: first project $\mathbf{X}$ into the PLS latent space using one-hot encoded class labels as $\mathbf{Y}$, then fit Linear Discriminant Analysis on the resulting scores $\mathbf{T} = \mathbf{X}\mathbf{W}$.

LDA in the latent space is well-conditioned (the score matrix has $k \ll p$ columns by construction), and the PLS projection has already aligned the latent axes with the class separation direction. This is more robust than applying LDA directly to high-dimensional $\mathbf{X}$, where the within-class covariance is singular.

The decision boundary is **linear in the latent space** (and therefore also linear in the original feature space via $\mathbf{W}$). For non-linear class boundaries use PLS-QDA or PLS-logistic.

### Implementation

`p4a_pls_lda_fit`. The reference is composite (sklearn `PLSRegression` + sklearn `LinearDiscriminantAnalysis`); no library exposes a single PLS-LDA call.

R roxygen note (`methods_extra.R::pls_lda_fit`):

> PLS-LDA — Linear Discriminant Analysis on PLS scores.
> @param y_labels Integer vector. Class labels.
> @param n_components Integer. Number of latent components.
> @param n_classes Integer >= 2. Number of class labels.
> @param X Numeric matrix of predictors (rows = samples, cols = features).
> @export

MATLAB header (`bindings/matlab/+pls4all/pls_lda.m`):

```text
pls4all.pls_lda  Linear Discriminant Analysis on PLS scores.
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
p4a_PLSRegression+LDA(ctx, cfg, &x_view, &y_view, /* hyperparams */, &res);
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
from pls4all._methods import PLSRegression+LDA
with pls4all.Context() as ctx, pls4all.Config() as cfg:
    res = PLSRegression+LDA(ctx, cfg, X, y, n_components=3, y_labels=y_labels)
# then: res.matrix("predictions"), res.matrix("coefficients"),
# res.vector("mask"), res.scalar("intercept"), …
```

:::

:::{tab-item} Python · pls4all.sklearn
:sync: python-sklearn
:class-label: lang-python

```python
from pls4all.sklearn import PLSLDAClassifier
mdl = PLSLDAClassifier(n_components=2)
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
res <- pls4all_method("pls_lda", X, y,
                      n_components = 3L, params = list(n_classes = 3L))
# res is a named list with MethodResult arrays/scalars.
# selected_indices / top_k_intervals are 1-based.
```

:::

:::{tab-item} R · pls4all (raw fn)
:sync: r-raw
:class-label: lang-r

```r
library(pls4all)
res  <- pls_lda_fit(X, y_labels, n_components, n_classes = NULL)
yhat <- pls4all_predict(res, X_test)
```

:::

:::{tab-item} MATLAB · pls4all (MEX)
:sync: matlab-mex
:class-label: lang-matlab

```matlab
res = pls4all.pls_lda(X, y, 3);
% see header of bindings/matlab/+pls4all/pls_lda.m for full
% parameter surface:
%   res = pls_lda(X, y_labels, n_components, n_classes)
yhat = predict(res, Xtest);
```

:::

:::{tab-item} MATLAB · pls4all (classdef)
:sync: matlab-classdef
:class-label: lang-matlab

_No idiomatic classdef wrapper — invoke `pls4all.fit("pls_lda", X, y, …)` directly from the unified MEX factory._

:::

::::


**Registry parity references** 📐

:::{card}
:class-card: external-refs

- 📐 **`ref.python_scikit_learn`** (python · python) — `scikit-learn` 1.8.0 · qualitative (rmse_rel ≤ 5e+00) — sklearn `PLSRegression -> LinearDiscriminantAnalysis` pipeline. pls4all's PLS-LDA uses a single SIMPLS pass with an internal LDA head; sklearn fits PLS on dummy-encoded targets and feeds the scores into LDA — both are LDA on PLS scores but the latent bases diverge. We compare class boundaries via one-hot decision scores.
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
<tr class="bk-row"><td class="bk-name"><code>pls4all.cpp.blas</code></td><td class="parity parity-error">⚠</td><td class="ms">—</td><td class="ms ms-empty">—</td><td class="ms">—</td></tr>
<tr class="bk-row"><td class="bk-name"><code>pls4all.cpp.blas+omp</code></td><td class="parity parity-error">⚠</td><td class="ms">—</td><td class="ms ms-empty">—</td><td class="ms">—</td></tr>
<tr class="bk-row"><td class="bk-name"><code>pls4all.cpp.omp</code></td><td class="parity parity-error">⚠</td><td class="ms">—</td><td class="ms ms-empty">—</td><td class="ms">—</td></tr>
<tr class="bk-row"><td class="bk-name"><code>pls4all.cpp.ref</code></td><td class="parity parity-error">⚠</td><td class="ms">—</td><td class="ms ms-empty">—</td><td class="ms">—</td></tr>
</tbody>
<tbody class="lang-band lang-python"><tr class="lang-band-row" data-lang="python"><th colspan="5" scope="rowgroup"><span class="lang-band-dot"></span>Python · pls4all</th></tr>
<tr class="bk-row"><td class="bk-name"><code>pls4all.python</code></td><td class="parity parity-error">⚠</td><td class="ms">—</td><td class="ms ms-empty">—</td><td class="ms">—</td></tr>
<tr class="bk-row"><td class="bk-name"><code>pls4all.registry</code></td><td class="parity parity-drift">≈</td><td class="ms ms-best">1.83 ms<span class="medal" title="fastest">🏆</span></td><td class="ms ms-best">185.5 ms<span class="medal" title="fastest">🏆</span></td><td class="ms ms-best">4.95 ms<span class="medal" title="fastest">🏆</span></td></tr>
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
<tr class="bk-row truth-source truth-source-qualitative"><td class="bk-name"><span class="truth-mark" title="Registry parity reference (python): scikit-learn 1.8.0 — qualitative (rmse_rel ≤ 5e+00)">📐</span><code>ref.python_scikit_learn</code></td><td class="parity parity-drift">≈</td><td class="ms">2.51 ms</td><td class="ms">565.0 ms</td><td class="ms">6.17 ms</td></tr>
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