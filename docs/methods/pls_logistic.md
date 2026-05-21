# `pls_logistic` — PLS-logistic regression

_Group_: **Classification & GLM** · _Registry tolerance_: `5.0`

## Description

PLS-Logistic — Logistic regression on PLS scores

From the `pls4all.sklearn.PLSLogisticClassifier` docstring:

> PLS-Logistic: PLS scores fed into multinomial softmax IRLS.

> **Registry note** — sklearn `PLSRegression -> LogisticRegression` pipeline vs pls4all's single-pass PLS + softmax IRLS. Latent decompositions differ; parity is qualitative.

### Parameters

| Name | Type | Default | Notes |
|------|------|---------|-------|
| `n_components` | `int` | `2` | Number of latent components extracted (k). |
| `n_classes` | `int` | `3` | registry benchmark cell value |

## Explanations

### Bibliographic source

Bastien, P., Esposito Vinzi, V. & Tenenhaus, M. (2005). *PLS generalised linear regression*. Computational Statistics & Data Analysis 48(1), 17–46.

### Mathematical principle

Iteratively-reweighted-least-squares PLS with a logit link function. At each iteration the current predictor is converted to a working response via $z_i = \eta_i + (y_i - p_i) / (p_i(1 - p_i))$ where $p_i = 1/(1 + e^{-\eta_i})$, a PLS fit is run on $(\mathbf{X}, \mathbf{z})$ with weights $p_i(1 - p_i)$, and the linear predictor is updated.

This is the natural extension of PLS to binary / multinomial classification when class probabilities (rather than hard labels or class scores) are needed, and it generalises smoothly to GLM families beyond Bernoulli (Poisson — see `pls_glm`). The multinomial case extends to $K$ classes via a softmax link.

Convergence is typically reached in 5–10 IRLS iterations. The Bastien et al. variant is closely related to Marx 1996's *Iteratively Reweighted PLS* but differs in the deflation convention.

### Implementation

`p4a_pls_logistic_fit` (in-sample only). Reference: R `plsRglm 1.7.0`.

R roxygen note (`methods_extra.R::pls_logistic_fit`):

> Multinomial logistic regression on PLS scores.
> @param y_labels Integer vector. Class labels.
> @param n_components Integer. Number of latent components.
> @param n_classes Integer >= 2. Number of class labels.
> @param X Numeric matrix of predictors (rows = samples, cols = features).
> @export

MATLAB header (`bindings/matlab/+pls4all/pls_logistic.m`):

```text
pls4all.pls_logistic  Multinomial logistic regression on PLS scores.
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
/* C ABI — libp4a */
p4a_context_t* ctx = p4a_context_create();
p4a_config_t*  cfg = p4a_config_create();
p4a_method_result_t* res = NULL;
p4a_pls_logistic_fit(ctx, cfg, &x_view, &y_view, /* hyperparams */, &res);
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
from pls4all._methods import pls_logistic_fit
with pls4all.Context() as ctx, pls4all.Config() as cfg:
    res = pls_logistic_fit(ctx, cfg, X, y, n_components=3, y_labels=y_labels)
# then: res.matrix("predictions"), res.matrix("coefficients"),
# res.vector("mask"), res.scalar("intercept"), …
```

:::

:::{tab-item} Python · pls4all.sklearn
:sync: python-sklearn
:class-label: lang-python

```python
from pls4all.sklearn import PLSLogisticClassifier
mdl = PLSLogisticClassifier(n_components=2)
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
res <- pls4all_method("pls_logistic", X, y,
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
res  <- pls_logistic_fit(X, y_labels, n_components, n_classes = NULL)
yhat <- pls4all_predict(res, X_test)
```

:::

:::{tab-item} MATLAB · pls4all (MEX)
:sync: matlab-mex
:class-label: lang-matlab

```matlab
res = pls4all.pls_logistic(X, y, 3);
% see header of bindings/matlab/+pls4all/pls_logistic.m for full
% parameter surface:
%   res = pls_logistic(X, y_labels, n_components, n_classes)
yhat = predict(res, Xtest);
```

:::

:::{tab-item} MATLAB · pls4all (classdef)
:sync: matlab-classdef
:class-label: lang-matlab

_No idiomatic classdef wrapper — invoke `pls4all.fit("pls_logistic", X, y, …)` directly from the unified MEX factory._

:::

::::


**Registry parity references** 📐

:::{card}
:class-card: external-refs

- 📐 **`ref.python_scikit_learn`** (python · python) — `scikit-learn` 1.8.0 · qualitative (rmse_rel ≤ 5e+00) — sklearn `PLSRegression -> LogisticRegression` pipeline. pls4all's PLS-Logistic does a single PLS + softmax IRLS in C; sklearn fits PLS on one-hot Y, then a multinomial LogisticRegression on the scores. Both are valid PLS-logistic pipelines but the latent decompositions differ; parity is on the decision-score shape.
:::

### Validation contract

Structural validation contract for `pls_logistic`, exported by the declarative validation framework (`benchmarks/validation/registry/`) from `benchmarks.parity_timing.registry`. Regenerated by `python -m benchmarks.validation.export_registry --write`.

| Field | Value |
|------|-------|
| Reference kind | `external` |
| Canonical reference role | `python` |
| Declared references | Python |
| Comparator | `reference_parity` (external-reference validity gate) |
| Comparator metric | `rmse_rel` |
| Reference tolerance | `5e+00` (qualitative) |
| Binding tolerance | `1e-06` |
| Prediction key | `decision_scores` |
| Required data flags | `y_labels` |

**Benchmark cell parameters**

| Parameter | Value |
|-----------|-------|
| `n_classes` | `3` |
| `n_components` | `3` |
| `n_features` | `40` |
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

**Reference gate**: non-release diagnostic — shape/smoke comparison only (`rmse_rel ≤ 5e+00`). Treat it as evidence that both implementations ran, not as numerical agreement.

Rows tagged with **📐** are the canonical parity references for this method (declared in [`parity_timing.registry`](../benchmarks/methodology.md)). C++ and external rows show reference parity; pls4all language bindings show binding parity against the C++ backend. Hover the icon for role and tolerance band.

::::{tab-set}
:class: parity-tabs

:::{tab-item} 1 thread
:sync: threads-1

<div class="parity-table-wrap">
<table class="docutils parity-grouped">
<thead><tr><th scope="col">Backend</th><th scope="col">Divergence</th><th class="size-col" scope="col">100×50 (ms)</th><th class="size-col" scope="col">100×500 (ms)</th><th class="size-col" scope="col">100×2500 (ms)</th><th class="size-col" scope="col">1000×500 (ms)</th></tr></thead>
<tbody class="lang-band lang-cpp"><tr class="lang-band-row" data-lang="cpp"><th colspan="6" scope="rowgroup"><span class="lang-band-dot"></span>C++ native · libp4a</th></tr>
<tr class="bk-row"><td class="bk-name"><code>pls4all.cpp.blas</code></td><td class="parity parity-divergence parity-divergent">0.87</td><td class="ms ms-best">1.08 ms<span class="medal" title="fastest">🏆</span></td><td class="ms">8.30 ms</td><td class="ms">41.3 ms</td><td class="ms">85.2 ms</td></tr>
<tr class="bk-row"><td class="bk-name"><code>pls4all.cpp.blas+omp</code></td><td class="parity parity-divergence parity-divergent">0.87</td><td class="ms">1.12 ms</td><td class="ms">8.35 ms</td><td class="ms">41.1 ms</td><td class="ms">85.3 ms</td></tr>
<tr class="bk-row"><td class="bk-name"><code>pls4all.cpp.omp</code></td><td class="parity parity-divergence parity-divergent">0.87</td><td class="ms">1.09 ms</td><td class="ms">8.45 ms</td><td class="ms ms-best">40.8 ms<span class="medal" title="fastest">🏆</span></td><td class="ms ms-best">84.2 ms<span class="medal" title="fastest">🏆</span></td></tr>
<tr class="bk-row"><td class="bk-name"><code>pls4all.cpp.ref</code></td><td class="parity parity-divergence parity-divergent">0.87</td><td class="ms">1.14 ms</td><td class="ms">8.40 ms</td><td class="ms">41.4 ms</td><td class="ms">85.5 ms</td></tr>
</tbody>
<tbody class="lang-band lang-python"><tr class="lang-band-row" data-lang="python"><th colspan="6" scope="rowgroup"><span class="lang-band-dot"></span>Python · pls4all</th></tr>
<tr class="bk-row"><td class="bk-name"><code>pls4all.python</code></td><td class="parity parity-divergence parity-divergent">0.87</td><td class="ms">1.11 ms</td><td class="ms ms-best">8.28 ms<span class="medal" title="fastest">🏆</span></td><td class="ms">40.9 ms</td><td class="ms">84.3 ms</td></tr>
<tr class="bk-row"><td class="bk-name"><code>pls4all.sklearn</code></td><td class="parity parity-divergence parity-divergent">0.87</td><td class="ms">1.29 ms</td><td class="ms">8.77 ms</td><td class="ms">41.7 ms</td><td class="ms">85.5 ms</td></tr>
</tbody>
<tbody class="lang-band lang-r"><tr class="lang-band-row" data-lang="r"><th colspan="6" scope="rowgroup"><span class="lang-band-dot"></span>R · pls4all</th></tr>
<tr class="bk-row"><td class="bk-name"><code>pls4all.R</code></td><td class="parity parity-divergence parity-divergent">0.87</td><td class="ms">3.68 ms</td><td class="ms">43.8 ms</td><td class="ms">409.0 ms</td><td class="ms">549.6 ms</td></tr>
<tr class="bk-row"><td class="bk-name"><code>pls4all.R.formula</code></td><td class="parity parity-divergence parity-divergent">0.87</td><td class="ms">5.56 ms</td><td class="ms">68.2 ms</td><td class="ms">610.2 ms</td><td class="ms">720.7 ms</td></tr>
<tr class="bk-row"><td class="bk-name"><code>pls4all.R.mdatools</code></td><td class="parity parity-divergence parity-divergent">0.87</td><td class="ms">5.71 ms</td><td class="ms">73.8 ms</td><td class="ms">625.8 ms</td><td class="ms">710.4 ms</td></tr>
<tr class="bk-row"><td class="bk-name"><code>pls4all.R.pls</code></td><td class="parity parity-divergence parity-divergent">0.87</td><td class="ms">5.58 ms</td><td class="ms">70.2 ms</td><td class="ms">606.8 ms</td><td class="ms">692.2 ms</td></tr>
</tbody>
<tbody class="lang-band lang-matlab"><tr class="lang-band-row" data-lang="matlab"><th colspan="6" scope="rowgroup"><span class="lang-band-dot"></span>MATLAB · pls4all</th></tr>
<tr class="bk-row"><td class="bk-name"><code>pls4all.matlab</code></td><td class="parity parity-divergence parity-divergent">0.87</td><td class="ms">1.89 ms</td><td class="ms">13.4 ms</td><td class="ms">64.9 ms</td><td class="ms">129.8 ms</td></tr>
<tr class="bk-row"><td class="bk-name"><code>pls4all.matlab.classdef</code></td><td class="parity parity-divergence parity-divergent">0.87</td><td class="ms">2.35 ms</td><td class="ms">13.9 ms</td><td class="ms">66.5 ms</td><td class="ms">133.7 ms</td></tr>
</tbody>
<tbody class="lang-band lang-python"><tr class="lang-band-row" data-lang="python"><th colspan="6" scope="rowgroup"><span class="lang-band-dot"></span>Python · external</th></tr>
<tr class="bk-row truth-source truth-source-qualitative"><td class="bk-name"><span class="truth-mark" title="Registry parity reference (python): scikit-learn 1.8.0 — qualitative (rmse_rel ≤ 5e+00)">📐</span><code>ref.python_scikit_learn</code></td><td class="parity parity-divergence parity-ref-source">0</td><td class="ms">2.40 ms</td><td class="ms">10.0 ms</td><td class="ms">41.4 ms</td><td class="ms">86.1 ms</td></tr>
</tbody>
</table>
</div>

:::

::::


---

_See also_: [benchmark overview](../benchmarks/overview.md) · [methods index](index.md) · [interactive dashboard](../landing/dashboard.md)