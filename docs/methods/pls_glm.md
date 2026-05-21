# `pls_glm` — PLS-GLM (Generalised Linear Model PLS)

_Group_: **Classification & GLM** · _Registry tolerance_: `0.5`

## Description

PLS-GLM (§5) — softmax/Poisson IRLS on PLS scores

From the `pls4all.sklearn.PLSGLMRegressor` docstring:

> PLS + Generalised Linear Model head (Bastien 2005).

> **Registry note** — R `plsRglm::plsRglm` (Bastien et al. 2005) with the Bastien IRLS algorithm. Fit per-target since plsRglm is univariate; predictions stacked. pls4all uses a simplified PLS-then-link variant — tolerance widened to 5e-1 to admit the expected algorithmic divergence.

### Parameters

| Name | Type | Default | Notes |
|------|------|---------|-------|
| `n_components` | `int` | `2` | Number of latent components extracted (k). |
| `poisson` | `bool` | `False` | If True, fit a Poisson-deviance PLS-GLM (default Gaussian link). |
| `n_targets` | `int` | `3` | registry benchmark cell value |

## Explanations

### Bibliographic source

Marx, B. D. (1996). *Iteratively reweighted partial least squares estimation for generalized linear regression*. Technometrics 38(4), 374–381.

### Mathematical principle

PLS-GLM generalises PLS-logistic to any GLM family. The IRLS recipe is identical — derive a working response from the current linear predictor, fit PLS with the GLM weights, iterate — but the link function varies: identity for Gaussian, log for Poisson, logit for Bernoulli/binomial.

pls4all currently supports Gaussian and Poisson families (controlled by the `poisson` flag). The Poisson case is useful for count regression on spectroscopy data where the response is an integer abundance (cell counts, particle counts) rather than a continuous concentration.

Compared to running a vanilla PLS on $\log(y+1)$, the true Poisson formulation correctly handles the mean–variance relationship and is less biased for low counts.

### Implementation

`p4a_pls_glm_fit`. Reference: R `plsRglm 1.7.0`.

R roxygen note (`sklearn_methods.R::pls_glm`):

> PLS-GLM — formula entry point. Default is Gaussian; set
> `family = "poisson"` for Poisson IRLS.

MATLAB header (`bindings/matlab/+pls4all/GlmRegression.m`):

```text
pls4all.GlmRegression — PLS-GLM (Gaussian / Poisson IRLS).
 Like MB-PLS, uses the stored intercept directly.
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
p4a_pls_glm_fit(ctx, cfg, &x_view, &y_view, /* hyperparams */, &res);
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
from pls4all._methods import pls_glm_fit
with pls4all.Context() as ctx, pls4all.Config() as cfg:
    res = pls_glm_fit(ctx, cfg, X, y, n_components=4)
# then: res.matrix("predictions"), res.matrix("coefficients"),
# res.vector("mask"), res.scalar("intercept"), …
```

:::

:::{tab-item} Python · pls4all.sklearn
:sync: python-sklearn
:class-label: lang-python

```python
from pls4all.sklearn import PLSGLMRegressor
mdl = PLSGLMRegressor(n_components=2, poisson=False)
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
res <- pls4all_method("pls_glm", X, y,
                      n_components = 4L, params = list(n_targets = 3L, poisson = 0L))
# res is a named list with MethodResult arrays/scalars.
# selected_indices / top_k_intervals are 1-based.
```

:::

:::{tab-item} R · pls4all (raw fn)
:sync: r-raw
:class-label: lang-r

```r
library(pls4all)
res  <- pls_glm_fit(X, Y, n_components, poisson = FALSE)
yhat <- pls4all_predict(res, X_test)
```

:::

:::{tab-item} R · pls4all (formula+S3)
:sync: r-formula
:class-label: lang-r

```r
library(pls4all)
fit  <- pls_glm(y ~ ., data = train, ncomp = 4L)
yhat <- predict(fit, newdata = test)
summary(fit)
```

:::

:::{tab-item} MATLAB · pls4all (MEX)
:sync: matlab-mex
:class-label: lang-matlab

```matlab
res = pls4all.pls_glm(X, y, 4);
% see header of bindings/matlab/+pls4all/pls_glm.m for full
% parameter surface:
%   res = pls_glm(X, Y, n_components, family)
yhat = predict(res, Xtest);
```

:::

:::{tab-item} MATLAB · pls4all (classdef)
:sync: matlab-classdef
:class-label: lang-matlab

```matlab
mdl  = pls4all.fit("pls_glm", X, y, "NumComponents", 4);
yhat = predict(mdl, Xtest);
```

:::

::::


**Registry parity references** 📐

:::{card}
:class-card: external-refs

- 📐 **`ref.r_plsrglm`** (R · r) — `plsRglm` 1.5.1 · qualitative (rmse_rel ≤ 5e-01) — R `plsRglm::plsRglm` (Bastien, Vinzi & Tenenhaus 2005) with the `pls-glm-gaussian` / `pls-glm-poisson` family. pls4all implements a simpler PLS-then-link variant so predictions diverge substantially; the parity check is a presence flag for the external reference.
:::

### Validation contract

Structural validation contract for `pls_glm`, exported by the declarative validation framework (`benchmarks/validation/registry/`) from `benchmarks.parity_timing.registry`. Regenerated by `python -m benchmarks.validation.export_registry --write`.

| Field | Value |
|------|-------|
| Reference kind | `external` |
| Canonical reference role | `r` |
| Declared references | R |
| Comparator | `reference_parity` (external-reference validity gate) |
| Comparator metric | `rmse_rel` |
| Reference tolerance | `5e-01` (qualitative) |
| Binding tolerance | `1e-06` |
| Prediction key | `predictions` |
| Required data flags | none |

**Benchmark cell parameters**

| Parameter | Value |
|-----------|-------|
| `n_components` | `4` |
| `n_features` | `30` |
| `n_samples` | `200` |
| `n_targets` | `3` |
| `poisson` | `0` |

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

**Reference gate**: non-release diagnostic — shape/smoke comparison only (`rmse_rel ≤ 5e-01`). Treat it as evidence that both implementations ran, not as numerical agreement.

Rows tagged with **📐** are the canonical parity references for this method (declared in [`parity_timing.registry`](../benchmarks/methodology.md)). C++ and external rows show reference parity; pls4all language bindings show binding parity against the C++ backend. Hover the icon for role and tolerance band.

::::{tab-set}
:class: parity-tabs

:::{tab-item} 1 thread
:sync: threads-1

<div class="parity-table-wrap">
<table class="docutils parity-grouped">
<thead><tr><th scope="col">Backend</th><th scope="col">Divergence</th><th class="size-col" scope="col">100×50 (ms)</th><th class="size-col" scope="col">100×500 (ms)</th><th class="size-col" scope="col">100×2500 (ms)</th><th class="size-col" scope="col">1000×500 (ms)</th></tr></thead>
<tbody class="lang-band lang-cpp"><tr class="lang-band-row" data-lang="cpp"><th colspan="6" scope="rowgroup"><span class="lang-band-dot"></span>C++ native · libp4a</th></tr>
<tr class="bk-row"><td class="bk-name"><code>pls4all.cpp.blas</code></td><td class="parity parity-divergence parity-not_available">—</td><td class="ms">1.02 ms</td><td class="ms">8.32 ms</td><td class="ms">44.0 ms</td><td class="ms">86.2 ms</td></tr>
<tr class="bk-row"><td class="bk-name"><code>pls4all.cpp.blas+omp</code></td><td class="parity parity-divergence parity-divergent">1.36</td><td class="ms">1.05 ms</td><td class="ms">8.39 ms</td><td class="ms">43.5 ms</td><td class="ms">86.7 ms</td></tr>
<tr class="bk-row"><td class="bk-name"><code>pls4all.cpp.omp</code></td><td class="parity parity-divergence parity-not_available">—</td><td class="ms ms-best">1.00 ms<span class="medal" title="fastest">🏆</span></td><td class="ms">8.36 ms</td><td class="ms">44.3 ms</td><td class="ms">88.5 ms</td></tr>
<tr class="bk-row"><td class="bk-name"><code>pls4all.cpp.ref</code></td><td class="parity parity-divergence parity-not_available">—</td><td class="ms">1.01 ms</td><td class="ms">8.44 ms</td><td class="ms">43.7 ms</td><td class="ms ms-best">84.9 ms<span class="medal" title="fastest">🏆</span></td></tr>
</tbody>
<tbody class="lang-band lang-python"><tr class="lang-band-row" data-lang="python"><th colspan="6" scope="rowgroup"><span class="lang-band-dot"></span>Python · pls4all</th></tr>
<tr class="bk-row"><td class="bk-name"><code>pls4all.python</code></td><td class="parity parity-divergence parity-divergent">1.36</td><td class="ms">1.00 ms</td><td class="ms ms-best">8.31 ms<span class="medal" title="fastest">🏆</span></td><td class="ms ms-best">43.3 ms<span class="medal" title="fastest">🏆</span></td><td class="ms">86.4 ms</td></tr>
<tr class="bk-row"><td class="bk-name"><code>pls4all.sklearn</code></td><td class="parity parity-divergence parity-divergent">1.36</td><td class="ms">1.20 ms</td><td class="ms">8.87 ms</td><td class="ms">44.1 ms</td><td class="ms">87.4 ms</td></tr>
</tbody>
<tbody class="lang-band lang-r"><tr class="lang-band-row" data-lang="r"><th colspan="6" scope="rowgroup"><span class="lang-band-dot"></span>R · pls4all</th></tr>
<tr class="bk-row"><td class="bk-name"><code>pls4all.R</code></td><td class="parity parity-divergence parity-divergent">1.36</td><td class="ms">3.23 ms</td><td class="ms">41.0 ms</td><td class="ms">395.1 ms</td><td class="ms">536.0 ms</td></tr>
<tr class="bk-row"><td class="bk-name"><code>pls4all.R.formula</code></td><td class="parity parity-divergence parity-divergent">1.36</td><td class="ms">4.81 ms</td><td class="ms">66.1 ms</td><td class="ms">580.6 ms</td><td class="ms">535.1 ms</td></tr>
<tr class="bk-row"><td class="bk-name"><code>pls4all.R.mdatools</code></td><td class="parity parity-divergence parity-drift">—</td><td class="ms">4.62 ms</td><td class="ms">64.5 ms</td><td class="ms">574.8 ms</td><td class="ms">626.4 ms</td></tr>
<tr class="bk-row"><td class="bk-name"><code>pls4all.R.pls</code></td><td class="parity parity-divergence parity-drift">—</td><td class="ms">4.77 ms</td><td class="ms">68.6 ms</td><td class="ms">560.2 ms</td><td class="ms">638.6 ms</td></tr>
</tbody>
<tbody class="lang-band lang-matlab"><tr class="lang-band-row" data-lang="matlab"><th colspan="6" scope="rowgroup"><span class="lang-band-dot"></span>MATLAB · pls4all</th></tr>
<tr class="bk-row"><td class="bk-name"><code>pls4all.matlab</code></td><td class="parity parity-divergence parity-divergent">1.36</td><td class="ms">1.69 ms</td><td class="ms">14.0 ms</td><td class="ms">67.3 ms</td><td class="ms">136.8 ms</td></tr>
<tr class="bk-row"><td class="bk-name"><code>pls4all.matlab.classdef</code></td><td class="parity parity-divergence parity-divergent">1.36</td><td class="ms">2.05 ms</td><td class="ms">14.6 ms</td><td class="ms">68.4 ms</td><td class="ms">137.5 ms</td></tr>
</tbody>
<tbody class="lang-band lang-r"><tr class="lang-band-row" data-lang="r"><th colspan="6" scope="rowgroup"><span class="lang-band-dot"></span>R · external</th></tr>
<tr class="bk-row truth-source truth-source-qualitative"><td class="bk-name"><span class="truth-mark" title="Registry parity reference (r): plsRglm 1.5.1 — qualitative (rmse_rel ≤ 5e-01)">📐</span><code>ref.r_plsrglm</code></td><td class="parity parity-divergence parity-ref-source">0</td><td class="ms">135.0 ms</td><td class="ms">1.3 s</td><td class="ms">6.1 s</td><td class="ms">5.2 s</td></tr>
</tbody>
</table>
</div>

:::

::::


---

_See also_: [benchmark overview](../benchmarks/overview.md) · [methods index](index.md) · [interactive dashboard](../landing/dashboard.md)