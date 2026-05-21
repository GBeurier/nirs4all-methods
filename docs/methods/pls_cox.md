# `pls_cox` — PLS-Cox (survival regression)

_Group_: **Classification & GLM** · _Registry tolerance_: `5.0`

## Description

PLS-Cox (§5) — Cox PH on PLS scores

From the `pls4all.sklearn.PLSCoxRegressor` docstring:

> PLS + Cox proportional-hazards regression on PLS scores.

> **Registry note** — R `plsRcox::coxsplsDR` (Bastien 2008) — Deviance Residuals based PLS for censored data. pls4all uses a simplified PLS-then-Cox variant; widened tolerance to flag external reference presence.

### Parameters

| Name | Type | Default | Notes |
|------|------|---------|-------|
| `n_components` | `int` | `2` | Number of latent components extracted (k). |
| `n_classes` | `int` | `2` | registry benchmark cell value |

## Explanations

### Bibliographic source

Bastien, P., Bertrand, F., Meyer, N. & Maumy-Bertrand, M. (2015). *Deviance residuals-based sparse PLS and sparse kernel PLS regression for censored data*. Bioinformatics 31(3), 397–404.

### Mathematical principle

Cox proportional-hazards regression with PLS-based dimensionality reduction. The Cox model $\lambda(t \mid \mathbf{x}) = \lambda_0(t)\exp(\mathbf{x}^{\top}\boldsymbol{\beta})$ is degenerate in $p \gg n$ because the partial likelihood loses identifiability. PLS-Cox replaces the $p$-dimensional $\boldsymbol{\beta}$ with a $k$-dimensional latent representation by extracting PLS scores from the **deviance residuals** of a null Cox model.

Required inputs are survival times and event indicators (0 = censored, 1 = event observed). The output is a fitted Cox model on the latent scores; risk scores for new samples are computed by first projecting them into the latent space and then evaluating $\mathbf{t}^{\top}\boldsymbol{\beta}$.

This is the canonical method in high-dimensional biomarker survival studies (RNA-seq, MALDI-TOF) where a direct Cox model is infeasible.

### Implementation

`p4a_pls_cox_fit`. Reference: R `plsRcox 1.8.2`.

R roxygen note (`methods_extra.R::pls_cox_fit`):

> PLS-Cox proportional hazards.
> @param n_components Integer. Number of latent components.
> @param survival_times Method-specific parameter. See the underlying `*_fit()` function for the exact semantics.
> @param event_indicators Method-specific parameter. See the underlying `*_fit()` function for the exact semantics.
> @param X Numeric matrix of predictors (rows = samples, cols = features).
> @export

MATLAB header (`bindings/matlab/+pls4all/pls_cox.m`):

```text
pls4all.pls_cox  PLS-Cox proportional hazards (Breslow baseline hazard).
 survival_times: numeric vector of length size(X, 1).
 event_indicators: 0/1 integer vector of length size(X, 1).
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
p4a_pls_cox_fit(ctx, cfg, &x_view, &y_view, /* hyperparams */, &res);
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
from pls4all._methods import pls_cox_fit
with pls4all.Context() as ctx, pls4all.Config() as cfg:
    res = pls_cox_fit(ctx, cfg, X, y, n_components=4, sample_weights=sample_w, y_labels=y_labels)
# then: res.matrix("predictions"), res.matrix("coefficients"),
# res.vector("mask"), res.scalar("intercept"), …
```

:::

:::{tab-item} Python · pls4all.sklearn
:sync: python-sklearn
:class-label: lang-python

```python
from pls4all.sklearn import PLSCoxRegressor
mdl = PLSCoxRegressor(n_components=2)
mdl.fit(X, y, sample_weight=sample_w)
y_hat = mdl.predict(X_test)
```

:::

:::{tab-item} R · pls4all_method()
:sync: r-dispatcher
:class-label: lang-r

```r
library(pls4all)
# Unified low-level dispatcher (May 2026 R cleanup):
res <- pls4all_method("pls_cox", X, y,
                      n_components = 4L, params = list(n_classes = 2L))
# res is a named list with MethodResult arrays/scalars.
# selected_indices / top_k_intervals are 1-based.
```

:::

:::{tab-item} R · pls4all (raw fn)
:sync: r-raw
:class-label: lang-r

```r
library(pls4all)
res  <- pls_cox_fit(X, n_components, survival_times, event_indicators)
yhat <- pls4all_predict(res, X_test)
```

:::

:::{tab-item} MATLAB · pls4all (MEX)
:sync: matlab-mex
:class-label: lang-matlab

```matlab
res = pls4all.pls_cox(X, y, 4);
% see header of bindings/matlab/+pls4all/pls_cox.m for full
% parameter surface:
%   res = pls_cox(X, n_components, survival_times, event_indicators)
yhat = predict(res, Xtest);
```

:::

:::{tab-item} MATLAB · pls4all (classdef)
:sync: matlab-classdef
:class-label: lang-matlab

_No idiomatic classdef wrapper — invoke `pls4all.fit("pls_cox", X, y, …)` directly from the unified MEX factory._

:::

::::


**Registry parity references** 📐

:::{card}
:class-card: external-refs

- 📐 **`ref.r_plsrcox`** (R · r) — `plsRcox` 1.8.2 · qualitative (rmse_rel ≤ 5e+00) — R `plsRcox::coxsplsDR(Xplan, time, event, ncomp=K)`. pls4all uses a simplified PLS-then-Cox; widened tolerance.
:::

### Validation contract

Structural validation contract for `pls_cox`, exported by the declarative validation framework (`benchmarks/validation/registry/`) from `benchmarks.parity_timing.registry`. Regenerated by `python -m benchmarks.validation.export_registry --write`.

| Field | Value |
|------|-------|
| Reference kind | `external` |
| Canonical reference role | `r` |
| Declared references | R |
| Comparator | `reference_parity` (external-reference validity gate) |
| Comparator metric | `rmse_rel` |
| Reference tolerance | `5e+00` (qualitative) |
| Binding tolerance | `1e-06` |
| Prediction key | `predictions` |
| Required data flags | `sample_weights`, `y_labels` |

**Benchmark cell parameters**

| Parameter | Value |
|-----------|-------|
| `n_classes` | `2` |
| `n_components` | `4` |
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
<tr class="bk-row"><td class="bk-name"><code>pls4all.cpp.blas</code></td><td class="parity parity-divergence parity-divergent">0.92</td><td class="ms">1.01 ms</td><td class="ms">8.29 ms</td><td class="ms">43.6 ms</td><td class="ms">85.2 ms</td></tr>
<tr class="bk-row"><td class="bk-name"><code>pls4all.cpp.blas+omp</code></td><td class="parity parity-divergence parity-divergent">0.92</td><td class="ms">1.01 ms</td><td class="ms">8.47 ms</td><td class="ms">43.6 ms</td><td class="ms">87.6 ms</td></tr>
<tr class="bk-row"><td class="bk-name"><code>pls4all.cpp.omp</code></td><td class="parity parity-divergence parity-divergent">0.92</td><td class="ms ms-best">0.99 ms<span class="medal" title="fastest">🏆</span></td><td class="ms ms-best">8.29 ms<span class="medal" title="fastest">🏆</span></td><td class="ms">43.5 ms</td><td class="ms ms-best">84.7 ms<span class="medal" title="fastest">🏆</span></td></tr>
<tr class="bk-row"><td class="bk-name"><code>pls4all.cpp.ref</code></td><td class="parity parity-divergence parity-divergent">0.92</td><td class="ms">1.55 ms</td><td class="ms">8.60 ms</td><td class="ms ms-best">43.1 ms<span class="medal" title="fastest">🏆</span></td><td class="ms">85.6 ms</td></tr>
</tbody>
<tbody class="lang-band lang-python"><tr class="lang-band-row" data-lang="python"><th colspan="6" scope="rowgroup"><span class="lang-band-dot"></span>Python · pls4all</th></tr>
<tr class="bk-row"><td class="bk-name"><code>pls4all.python</code></td><td class="parity parity-divergence parity-divergent">0.92</td><td class="ms">1.01 ms</td><td class="ms">8.46 ms</td><td class="ms">43.6 ms</td><td class="ms">93.7 ms</td></tr>
<tr class="bk-row"><td class="bk-name"><code>pls4all.sklearn</code></td><td class="parity parity-divergence parity-divergent">0.92</td><td class="ms">1.14 ms</td><td class="ms">8.71 ms</td><td class="ms">44.4 ms</td><td class="ms">86.9 ms</td></tr>
</tbody>
<tbody class="lang-band lang-r"><tr class="lang-band-row" data-lang="r"><th colspan="6" scope="rowgroup"><span class="lang-band-dot"></span>R · pls4all</th></tr>
<tr class="bk-row"><td class="bk-name"><code>pls4all.R</code></td><td class="parity parity-divergence parity-divergent">0.92</td><td class="ms">3.43 ms</td><td class="ms">43.0 ms</td><td class="ms">447.1 ms</td><td class="ms">509.4 ms</td></tr>
<tr class="bk-row"><td class="bk-name"><code>pls4all.R.formula</code></td><td class="parity parity-divergence parity-divergent">0.92</td><td class="ms">5.28 ms</td><td class="ms">64.3 ms</td><td class="ms">660.3 ms</td><td class="ms">699.2 ms</td></tr>
<tr class="bk-row"><td class="bk-name"><code>pls4all.R.mdatools</code></td><td class="parity parity-divergence parity-drift">—</td><td class="ms">5.22 ms</td><td class="ms">70.2 ms</td><td class="ms">639.2 ms</td><td class="ms">685.2 ms</td></tr>
<tr class="bk-row"><td class="bk-name"><code>pls4all.R.pls</code></td><td class="parity parity-divergence parity-drift">—</td><td class="ms">4.84 ms</td><td class="ms">67.8 ms</td><td class="ms">628.1 ms</td><td class="ms">548.1 ms</td></tr>
</tbody>
<tbody class="lang-band lang-matlab"><tr class="lang-band-row" data-lang="matlab"><th colspan="6" scope="rowgroup"><span class="lang-band-dot"></span>MATLAB · pls4all</th></tr>
<tr class="bk-row"><td class="bk-name"><code>pls4all.matlab</code></td><td class="parity parity-divergence parity-divergent">0.92</td><td class="ms">1.87 ms</td><td class="ms">13.5 ms</td><td class="ms">67.5 ms</td><td class="ms">133.5 ms</td></tr>
<tr class="bk-row"><td class="bk-name"><code>pls4all.matlab.classdef</code></td><td class="parity parity-divergence parity-divergent">0.92</td><td class="ms">2.60 ms</td><td class="ms">13.9 ms</td><td class="ms">67.0 ms</td><td class="ms">133.6 ms</td></tr>
</tbody>
<tbody class="lang-band lang-r"><tr class="lang-band-row" data-lang="r"><th colspan="6" scope="rowgroup"><span class="lang-band-dot"></span>R · external</th></tr>
<tr class="bk-row truth-source truth-source-qualitative"><td class="bk-name"><span class="truth-mark" title="Registry parity reference (r): plsRcox 1.8.2 — qualitative (rmse_rel ≤ 5e+00)">📐</span><code>ref.r_plsrcox</code></td><td class="parity parity-divergence parity-ref-source">0</td><td class="ms">34.7 ms</td><td class="ms">142.0 ms</td><td class="ms">1.3 s</td><td class="ms">1.3 s</td></tr>
</tbody>
</table>
</div>

:::

::::


---

_See also_: [benchmark overview](../benchmarks/overview.md) · [methods index](index.md) · [interactive dashboard](../landing/dashboard.md)