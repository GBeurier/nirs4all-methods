# `ecr` — ECR — Elastic Component Regression

_Group_: **Calibration transfer** · _Registry tolerance_: `0.001`

## Description

Elastic Component Regression (Phase 50)

From the `pls4all.sklearn.ECRegression` docstring:

> Elastic Component Regression (Liu 2013) — interpolates PCR (α=0)
and PLS (α=1).

> **Registry note** — Octave-bridged libPLS 1.95 `ecr(X, y, A, 'center', alpha)`. Deterministic algorithm; small numerical differences arise only from the power-method tolerance and FP accumulation order.

### Parameters

| Name | Type | Default | Notes |
|------|------|---------|-------|
| `n_components` | `int` | `2` | Number of latent components extracted (k). |
| `alpha` | `float` | `0.5` | Elastic-net mixing weight (0 = pure L2, 1 = pure L1) applied to the PLS coefficient path. |

## Explanations

### Bibliographic source

Liu, Y., Zhang, B. & Hu, J. (2013). *Elastic Component Regression*. Chemometrics and Intelligent Laboratory Systems 124, 73–79. — adapted in pls4all as a continuum/elastic blend.

### Mathematical principle

ECR interpolates between PCR and PLS via a single parameter $\alpha \in [0, 1]$ that mixes the two loading-weight criteria. The latent direction is $\mathbf{w} \propto (1-\alpha)\mathbf{X}^{\top}\mathbf{X}\mathbf{w} + \alpha \mathbf{X}^{\top}\mathbf{y}$, which recovers PCR at $\alpha = 0$ (the leading eigenvector of $\mathbf{X}^{\top}\mathbf{X}$) and PLS at $\alpha = 1$ (proportional to $\mathbf{X}^{\top}\mathbf{y}$). Intermediate $\alpha$ blends variance and covariance criteria; the optimum is typically located by cross-validation.

ECR is closely related to continuum regression with a different parameterisation, and in practice serves a similar purpose: when neither PCR nor PLS dominates RMSE on a given dataset, an interpolating method often wins by a small margin and offers a smooth tunable spectrum.

### Implementation

`p4a_ecr_fit`. No widely installable reference; treated as `paper_only` in the registry.

R roxygen note (`sklearn_extra.R::ecr`):

> Elastic Component Regression — formula entry point.

MATLAB header (`bindings/matlab/+pls4all/EcrRegression.m`):

```text
pls4all.EcrRegression  Elastic Component Regression (Liu 2009).
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
p4a_ecr_fit(ctx, cfg, &x_view, &y_view, /* hyperparams */, &res);
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
from pls4all._methods import ecr_fit
with pls4all.Context() as ctx, pls4all.Config() as cfg:
    res = ecr_fit(ctx, cfg, X, y, n_components=4)
# then: res.matrix("predictions"), res.matrix("coefficients"),
# res.vector("mask"), res.scalar("intercept"), …
```

:::

:::{tab-item} Python · pls4all.sklearn
:sync: python-sklearn
:class-label: lang-python

```python
from pls4all.sklearn import ECRegression
mdl = ECRegression(n_components=2, alpha=0.5)
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
res <- pls4all_method("ecr", X, y,
                      n_components = 4L, params = list(alpha = 0.5))
# res is a named list with MethodResult arrays/scalars.
# selected_indices / top_k_intervals are 1-based.
```

:::

:::{tab-item} R · pls4all (raw fn)
:sync: r-raw
:class-label: lang-r

```r
library(pls4all)
res  <- ecr_fit(X, Y, n_components, alpha = 0.5)
yhat <- pls4all_predict(res, X_test)
```

:::

:::{tab-item} R · pls4all (formula+S3)
:sync: r-formula
:class-label: lang-r

```r
library(pls4all)
fit  <- ecr(y ~ ., data = train, ncomp = 4L)
yhat <- predict(fit, newdata = test)
summary(fit)
```

:::

:::{tab-item} MATLAB · pls4all (MEX)
:sync: matlab-mex
:class-label: lang-matlab

```matlab
res = pls4all.ecr(X, y, 4);
% see header of bindings/matlab/+pls4all/ecr.m for full
% parameter surface:
%   res = ecr(X, Y, n_components, alpha)
yhat = predict(res, Xtest);
```

:::

:::{tab-item} MATLAB · pls4all (classdef)
:sync: matlab-classdef
:class-label: lang-matlab

```matlab
mdl  = pls4all.fit("ecr", X, y, "NumComponents", 4);
yhat = predict(mdl, Xtest);
```

:::

::::


**Registry parity references** 📐

:::{card}
:class-card: external-refs

- 📐 **`ref.matlab_libpls`** (matlab · python) — `libPLS` 1.95 · relaxed (rmse_rel ≤ 1e-03) — Octave-bridged libPLS 1.95 `ecr(X, y, A, 'center', alpha)`. Predictions computed as X_predict @ B + y_mean using the fitted coefficient matrix and centring parameters.
:::

### Validation contract

Structural validation contract for `ecr`, exported by the declarative validation framework (`benchmarks/validation/registry/`) from `benchmarks.parity_timing.registry`. Regenerated by `python -m benchmarks.validation.export_registry --write`.

| Field | Value |
|------|-------|
| Reference kind | `external` |
| Canonical reference role | `python` |
| Declared references | Python |
| Comparator | `reference_parity` (external-reference validity gate) |
| Comparator metric | `rmse_rel` |
| Reference tolerance | `1e-03` (relaxed) |
| Binding tolerance | `1e-06` |
| Prediction key | `predictions` |
| Required data flags | none |

**Benchmark cell parameters**

| Parameter | Value |
|-----------|-------|
| `alpha` | `0.5` |
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
<thead><tr><th scope="col">Backend</th><th scope="col">Divergence</th><th class="size-col" scope="col">100×50 (ms)</th></tr></thead>
<tbody class="lang-band lang-cpp"><tr class="lang-band-row" data-lang="cpp"><th colspan="3" scope="rowgroup"><span class="lang-band-dot"></span>C++ native · libp4a</th></tr>
<tr class="bk-row"><td class="bk-name"><code>pls4all.cpp.blas+omp</code></td><td class="parity parity-divergence parity-ref-strict">7e-07</td><td class="ms">1.29 ms</td></tr>
</tbody>
<tbody class="lang-band lang-python"><tr class="lang-band-row" data-lang="python"><th colspan="3" scope="rowgroup"><span class="lang-band-dot"></span>Python · pls4all</th></tr>
<tr class="bk-row"><td class="bk-name"><code>pls4all.python</code></td><td class="parity parity-divergence parity-ref-strict">7e-07</td><td class="ms ms-best">1.24 ms<span class="medal" title="fastest">🏆</span></td></tr>
<tr class="bk-row"><td class="bk-name"><code>pls4all.sklearn</code></td><td class="parity parity-divergence parity-ref-strict">7e-07</td><td class="ms">1.40 ms</td></tr>
</tbody>
<tbody class="lang-band lang-r"><tr class="lang-band-row" data-lang="r"><th colspan="3" scope="rowgroup"><span class="lang-band-dot"></span>R · pls4all</th></tr>
<tr class="bk-row"><td class="bk-name"><code>pls4all.R</code></td><td class="parity parity-divergence parity-ref-strict">7e-07</td><td class="ms">2.62 ms</td></tr>
<tr class="bk-row"><td class="bk-name"><code>pls4all.R.formula</code></td><td class="parity parity-divergence parity-ref-strict">7e-07</td><td class="ms">3.87 ms</td></tr>
</tbody>
<tbody class="lang-band lang-matlab"><tr class="lang-band-row" data-lang="matlab"><th colspan="3" scope="rowgroup"><span class="lang-band-dot"></span>MATLAB · pls4all</th></tr>
<tr class="bk-row"><td class="bk-name"><code>pls4all.matlab</code></td><td class="parity parity-divergence parity-ref-strict">7e-07</td><td class="ms">1.87 ms</td></tr>
<tr class="bk-row"><td class="bk-name"><code>pls4all.matlab.classdef</code></td><td class="parity parity-divergence parity-ref-strict">7e-07</td><td class="ms">2.30 ms</td></tr>
</tbody>
<tbody class="lang-band lang-matlab"><tr class="lang-band-row" data-lang="matlab"><th colspan="3" scope="rowgroup"><span class="lang-band-dot"></span>MATLAB · external</th></tr>
<tr class="bk-row truth-source truth-source-relaxed"><td class="bk-name"><span class="truth-mark" title="Registry parity reference (python): libPLS 1.95 — relaxed (rmse_rel ≤ 1e-03)">📐</span><code>ref.matlab_libpls</code></td><td class="parity parity-divergence parity-ref-source">0</td><td class="ms">11.2 ms</td></tr>
</tbody>
</table>
</div>

:::

::::


---

_See also_: [benchmark overview](../benchmarks/overview.md) · [methods index](index.md) · [interactive dashboard](../landing/dashboard.md)