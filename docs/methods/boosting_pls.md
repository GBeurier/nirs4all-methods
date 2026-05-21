# `boosting_pls` — Boosting PLS

_Group_: **Ensemble** · _Registry tolerance_: `10.0`

## Description

Boosting PLS (§20)

From the `pls4all.sklearn.BoostingPLSRegression` docstring:

> Boosted PLS regression.

> **Registry note** — R `mboost::glmboost(family=Gaussian())` boosts linear weak learners while pls4all boosts PLS weak learners; same family, different decision surfaces — widened tol.

### Parameters

| Name | Type | Default | Notes |
|------|------|---------|-------|
| `n_components` | `int` | `2` | Number of latent components extracted (k). |
| `n_estimators` | `int` | `50` | Number of base PLS sub-models in the ensemble. |
| `learning_rate` | `float` | `0.1` | Boosting shrinkage applied to each successive base learner. |

## Explanations

### Bibliographic source

Friedman, J. H. (2001). *Greedy function approximation: a gradient boosting machine*. Annals of Statistics 29(5), 1189–1232. — adapted for PLS as a base learner.

### Mathematical principle

Gradient boosting builds an additive predictor $F_M(\mathbf{x}) = \sum_{m=1}^M \eta\, h_m(\mathbf{x})$ where each weak learner $h_m$ is fit on the negative gradient (the residuals, for squared-error loss) of the current ensemble. With PLS as the weak learner, each $h_m$ is a small ($k$-component) PLS fitted on the pseudo-response $r_i^{(m)} = y_i - F_{m-1}(\mathbf{x}_i)$.

The learning rate $\eta$ (typically 0.05–0.1) and the number of boosting iterations $M$ are the key hyperparameters; their product roughly controls the effective number of latent dimensions explored. Because boosting reduces bias, it can recover non-linear $Y$–$X$ relationships even with linear PLS base learners — at the cost of much higher computational cost than a single PLS.

### Implementation

`p4a_boosting_pls_fit`. Reference: CRAN `mboost::glmboost` with a PLS base learner (`mboost 2.9.11`).

R roxygen note (`sklearn_extra.R::boosting_pls`):

> Boosting PLS — formula entry point.

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
p4a_boosting_pls_fit(ctx, cfg, &x_view, &y_view, /* hyperparams */, &res);
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
from pls4all._methods import boosting_pls_fit
with pls4all.Context() as ctx, pls4all.Config() as cfg:
    res = boosting_pls_fit(ctx, cfg, X, y, n_components=4)
# then: res.matrix("predictions"), res.matrix("coefficients"),
# res.vector("mask"), res.scalar("intercept"), …
```

:::

:::{tab-item} Python · pls4all.sklearn
:sync: python-sklearn
:class-label: lang-python

```python
from pls4all.sklearn import BoostingPLSRegression
mdl = BoostingPLSRegression(n_components=2, n_estimators=50, learning_rate=0.1)
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
res <- pls4all_method("boosting_pls", X, y,
                      n_components = 4L, params = list(n_estimators = 10L, learning_rate = 0.1))
# res is a named list with MethodResult arrays/scalars.
# selected_indices / top_k_intervals are 1-based.
```

:::

:::{tab-item} R · pls4all (raw fn)
:sync: r-raw
:class-label: lang-r

```r
library(pls4all)
res  <- boosting_pls_fit(X, Y, n_components,
                  n_estimators = 50L, learning_rate = 0.1)
yhat <- pls4all_predict(res, X_test)
```

:::

:::{tab-item} R · pls4all (formula+S3)
:sync: r-formula
:class-label: lang-r

```r
library(pls4all)
fit  <- boosting_pls(y ~ ., data = train, ncomp = 4L)
yhat <- predict(fit, newdata = test)
summary(fit)
```

:::

:::{tab-item} MATLAB · pls4all (MEX)
:sync: matlab-mex
:class-label: lang-matlab

```matlab
res = pls4all.boosting_pls(X, y, 4);
% see header of bindings/matlab/+pls4all/boosting_pls.m for full
% parameter surface:
%   res = boosting_pls(X, Y, n_components, n_estimators, learning_rate)
yhat = predict(res, Xtest);
```

:::

:::{tab-item} MATLAB · pls4all (classdef)
:sync: matlab-classdef
:class-label: lang-matlab

```matlab
mdl  = pls4all.fit("boosting_pls", X, y, "NumComponents", 4);
yhat = predict(mdl, Xtest);
```

:::

::::


**Registry parity references** 📐

:::{card}
:class-card: external-refs

- 📐 **`ref.r_mboost`** (R · r) — `mboost` 2.9-11 · qualitative (rmse_rel ≤ 1e+01) — R `mboost::glmboost(family=Gaussian())`. pls4all boosts PLS weak learners; mboost boosts linear weak learners. Qualitative parity (same algorithm family).
:::

### Validation contract

Structural validation contract for `boosting_pls`, exported by the declarative validation framework (`benchmarks/validation/registry/`) from `benchmarks.parity_timing.registry`. Regenerated by `python -m benchmarks.validation.export_registry --write`.

| Field | Value |
|------|-------|
| Reference kind | `external` |
| Canonical reference role | `r` |
| Declared references | R |
| Comparator | `reference_parity` (external-reference validity gate) |
| Comparator metric | `rmse_rel` |
| Reference tolerance | `1e+01` (qualitative) |
| Binding tolerance | `1e-06` |
| Prediction key | `predictions` |
| Required data flags | none |

**Benchmark cell parameters**

| Parameter | Value |
|-----------|-------|
| `learning_rate` | `0.1` |
| `n_components` | `4` |
| `n_estimators` | `10` |
| `n_features` | `30` |
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

**Reference gate**: non-release diagnostic — shape/smoke comparison only (`rmse_rel ≤ 1e+01`). Treat it as evidence that both implementations ran, not as numerical agreement.

Rows tagged with **📐** are the canonical parity references for this method (declared in [`parity_timing.registry`](../benchmarks/methodology.md)). C++ and external rows show reference parity; pls4all language bindings show binding parity against the C++ backend. Hover the icon for role and tolerance band.

::::{tab-set}
:class: parity-tabs

:::{tab-item} 1 thread
:sync: threads-1

<div class="parity-table-wrap">
<table class="docutils parity-grouped">
<thead><tr><th scope="col">Backend</th><th scope="col">Divergence</th><th class="size-col" scope="col">100×50 (ms)</th><th class="size-col" scope="col">100×500 (ms)</th><th class="size-col" scope="col">100×2500 (ms)</th><th class="size-col" scope="col">1000×500 (ms)</th></tr></thead>
<tbody class="lang-band lang-cpp"><tr class="lang-band-row" data-lang="cpp"><th colspan="6" scope="rowgroup"><span class="lang-band-dot"></span>C++ native · libp4a</th></tr>
<tr class="bk-row"><td class="bk-name"><code>pls4all.cpp.blas</code></td><td class="parity parity-divergence parity-divergent">0.49</td><td class="ms">1.25 ms</td><td class="ms ms-best">11.0 ms<span class="medal" title="fastest">🏆</span></td><td class="ms ms-best">56.5 ms<span class="medal" title="fastest">🏆</span></td><td class="ms ms-best">119.4 ms<span class="medal" title="fastest">🏆</span></td></tr>
<tr class="bk-row"><td class="bk-name"><code>pls4all.cpp.blas+omp</code></td><td class="parity parity-divergence parity-divergent">0.49</td><td class="ms">1.30 ms</td><td class="ms">11.4 ms</td><td class="ms">59.5 ms</td><td class="ms">122.1 ms</td></tr>
<tr class="bk-row"><td class="bk-name"><code>pls4all.cpp.omp</code></td><td class="parity parity-divergence parity-divergent">0.49</td><td class="ms ms-best">1.25 ms<span class="medal" title="fastest">🏆</span></td><td class="ms">13.8 ms</td><td class="ms">57.6 ms</td><td class="ms">119.9 ms</td></tr>
<tr class="bk-row"><td class="bk-name"><code>pls4all.cpp.ref</code></td><td class="parity parity-divergence parity-divergent">0.49</td><td class="ms">1.29 ms</td><td class="ms">13.6 ms</td><td class="ms">58.7 ms</td><td class="ms">121.7 ms</td></tr>
</tbody>
<tbody class="lang-band lang-python"><tr class="lang-band-row" data-lang="python"><th colspan="6" scope="rowgroup"><span class="lang-band-dot"></span>Python · pls4all</th></tr>
<tr class="bk-row"><td class="bk-name"><code>pls4all.python</code></td><td class="parity parity-divergence parity-divergent">0.49</td><td class="ms">1.44 ms</td><td class="ms">11.4 ms</td><td class="ms">57.8 ms</td><td class="ms">122.3 ms</td></tr>
<tr class="bk-row"><td class="bk-name"><code>pls4all.sklearn</code></td><td class="parity parity-divergence parity-divergent">0.49</td><td class="ms">1.57 ms</td><td class="ms">11.5 ms</td><td class="ms">59.1 ms</td><td class="ms">125.0 ms</td></tr>
</tbody>
<tbody class="lang-band lang-r"><tr class="lang-band-row" data-lang="r"><th colspan="6" scope="rowgroup"><span class="lang-band-dot"></span>R · pls4all</th></tr>
<tr class="bk-row"><td class="bk-name"><code>pls4all.R</code></td><td class="parity parity-divergence parity-divergent">0.49</td><td class="ms">3.97 ms</td><td class="ms">48.7 ms</td><td class="ms">430.1 ms</td><td class="ms">584.5 ms</td></tr>
<tr class="bk-row"><td class="bk-name"><code>pls4all.R.formula</code></td><td class="parity parity-divergence parity-divergent">0.49</td><td class="ms">5.43 ms</td><td class="ms">64.9 ms</td><td class="ms">648.1 ms</td><td class="ms">744.8 ms</td></tr>
<tr class="bk-row"><td class="bk-name"><code>pls4all.R.mdatools</code></td><td class="parity parity-divergence parity-divergent">0.47</td><td class="ms">5.51 ms</td><td class="ms">64.2 ms</td><td class="ms">640.4 ms</td><td class="ms">704.2 ms</td></tr>
<tr class="bk-row"><td class="bk-name"><code>pls4all.R.pls</code></td><td class="parity parity-divergence parity-divergent">0.47</td><td class="ms">5.30 ms</td><td class="ms">63.4 ms</td><td class="ms">634.3 ms</td><td class="ms">703.5 ms</td></tr>
</tbody>
<tbody class="lang-band lang-matlab"><tr class="lang-band-row" data-lang="matlab"><th colspan="6" scope="rowgroup"><span class="lang-band-dot"></span>MATLAB · pls4all</th></tr>
<tr class="bk-row"><td class="bk-name"><code>pls4all.matlab</code></td><td class="parity parity-divergence parity-divergent">0.49</td><td class="ms">3.21 ms</td><td class="ms">15.8 ms</td><td class="ms">86.5 ms</td><td class="ms">172.3 ms</td></tr>
<tr class="bk-row"><td class="bk-name"><code>pls4all.matlab.classdef</code></td><td class="parity parity-divergence parity-divergent">0.49</td><td class="ms">3.90 ms</td><td class="ms">16.7 ms</td><td class="ms">85.1 ms</td><td class="ms">174.2 ms</td></tr>
</tbody>
<tbody class="lang-band lang-r"><tr class="lang-band-row" data-lang="r"><th colspan="6" scope="rowgroup"><span class="lang-band-dot"></span>R · external</th></tr>
<tr class="bk-row truth-source truth-source-qualitative"><td class="bk-name"><span class="truth-mark" title="Registry parity reference (r): mboost 2.9-11 — qualitative (rmse_rel ≤ 1e+01)">📐</span><code>ref.r_mboost</code></td><td class="parity parity-divergence parity-ref-source">0</td><td class="ms">17.5 ms</td><td class="ms">113.3 ms</td><td class="ms">1.1 s</td><td class="ms">927.1 ms</td></tr>
</tbody>
</table>
</div>

:::

::::


---

_See also_: [benchmark overview](../benchmarks/overview.md) · [methods index](index.md) · [interactive dashboard](../landing/dashboard.md)