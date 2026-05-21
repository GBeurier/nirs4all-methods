# `weighted_pls` — Sample-weighted PLS

_Group_: **Robust / weighted** · _Registry tolerance_: `0.1`

## Description

Sample-weighted PLS (sqrt(w)-prescaled SIMPLS)

From the `pls4all.sklearn.WeightedPLSRegression` docstring:

> Sample-weighted PLS (sqrt(w)-prescaled SIMPLS).

> **Registry note** — sklearn PLSRegression on the sqrt(w)-prescaled centered data is mathematically equivalent to weighted PLS. sklearn vs pls4all use NIPALS vs SIMPLS so tolerance is widened to ~1e-1.

### Parameters

| Name | Type | Default | Notes |
|------|------|---------|-------|
| `n_components` | `int` | `2` | Number of latent components extracted (k). |
| `sample_weight` | `vector (required)` | `` | fit-time extra (not part of `__init__`) |

## Explanations

### Bibliographic source

Martens, H. & Næs, T. (1989). *Multivariate Calibration*. Wiley. §4.5 'Weighted regression for non-i.i.d. errors'.

### Mathematical principle

When the residual variance is not constant across samples — typical when calibration spectra are aggregated across instruments, sites or operators — a weighted least squares fit can dramatically improve generalisation. Weighted PLS prescales centred rows by $\sqrt{w_i}$ before extracting the SIMPLS components: $\tilde{\mathbf{X}} = \operatorname{diag}(\sqrt{w})\,\mathbf{X}_c, \quad \tilde{\mathbf{Y}} = \operatorname{diag}(\sqrt{w})\,\mathbf{Y}_c$, then runs vanilla SIMPLS on $(\tilde{\mathbf{X}}, \tilde{\mathbf{Y}})$.

Weights $w_i > 0$ encode any known per-sample reliability: inverse residual variance from a previous fit, instrument noise estimates, sample replicate counts. The weighted fit is mathematically equivalent to running standard PLS on a duplicated dataset where each row appears $w_i$ times.

This is a building block for robust PLS (IRLS over a weighted fit) and for incorporating known measurement noise into the calibration.

### Implementation

`p4a_weighted_pls_fit` (in-sample only — no global coefficient export, since the weighted fit's $\bar{\mathbf{x}}, \bar{\mathbf{y}}$ depend on the weights). Python reference: sklearn `PLSRegression` on the prescaled matrices.

R roxygen note (`sklearn_methods.R::weighted_pls`):

> Sample-weighted PLS — formula entry point.
> @param weights Numeric vector of length nrow(data) with sample weights.
> @inheritParams pls
> @export

MATLAB header (`bindings/matlab/+pls4all/WeightedPlsRegression.m`):

```text
pls4all.WeightedPlsRegression — sqrt(w)-prescaled SIMPLS.
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
p4a_weighted_pls_fit(ctx, cfg, &x_view, &y_view, /* hyperparams */, &res);
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
from pls4all._methods import weighted_pls_fit
with pls4all.Context() as ctx, pls4all.Config() as cfg:
    res = weighted_pls_fit(ctx, cfg, X, y, n_components=4, sample_weights=sample_w)
# then: res.matrix("predictions"), res.matrix("coefficients"),
# res.vector("mask"), res.scalar("intercept"), …
```

:::

:::{tab-item} Python · pls4all.sklearn
:sync: python-sklearn
:class-label: lang-python

```python
from pls4all.sklearn import WeightedPLSRegression
mdl = WeightedPLSRegression(n_components=2)
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
res <- pls4all_method("weighted_pls", X, y,
                      n_components = 4L)
# res is a named list with MethodResult arrays/scalars.
# selected_indices / top_k_intervals are 1-based.
```

:::

:::{tab-item} R · pls4all (raw fn)
:sync: r-raw
:class-label: lang-r

```r
library(pls4all)
res  <- weighted_pls_fit(X, Y, n_components, sample_weights)
yhat <- pls4all_predict(res, X_test)
```

:::

:::{tab-item} R · pls4all (formula+S3)
:sync: r-formula
:class-label: lang-r

```r
library(pls4all)
fit  <- weighted_pls(y ~ ., data = train, ncomp = 4L)
yhat <- predict(fit, newdata = test)
summary(fit)
```

:::

:::{tab-item} MATLAB · pls4all (MEX)
:sync: matlab-mex
:class-label: lang-matlab

```matlab
res = pls4all.weighted_pls(X, y, 4);
% see header of bindings/matlab/+pls4all/weighted_pls.m for full
% parameter surface:
%   res = weighted_pls(X, Y, n_components, sample_weights)
yhat = predict(res, Xtest);
```

:::

:::{tab-item} MATLAB · pls4all (classdef)
:sync: matlab-classdef
:class-label: lang-matlab

```matlab
mdl  = pls4all.fit("weighted_pls", X, y, "NumComponents", 4);
yhat = predict(mdl, Xtest);
```

:::

::::


**Registry parity references** 📐

:::{card}
:class-card: external-refs

- 📐 **`ref.python_scikit_learn`** (python · python) — `scikit-learn` 1.4.2 · qualitative (rmse_rel ≤ 1e-01) — Weighted PLS computed via sklearn PLSRegression on the sqrt(w)-prescaled centered (X, Y). sklearn is the external PLS engine; the row-scaling is a standard preconditioning step that is mathematically equivalent to weighted PLS.
:::

### Validation contract

Structural validation contract for `weighted_pls`, exported by the declarative validation framework (`benchmarks/validation/registry/`) from `benchmarks.parity_timing.registry`. Regenerated by `python -m benchmarks.validation.export_registry --write`.

| Field | Value |
|------|-------|
| Reference kind | `external` |
| Canonical reference role | `python` |
| Declared references | Python |
| Comparator | `reference_parity` (external-reference validity gate) |
| Comparator metric | `rmse_rel` |
| Reference tolerance | `1e-01` (qualitative) |
| Binding tolerance | `1e-06` |
| Prediction key | `predictions` |
| Required data flags | `sample_weights` |

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

**Reference gate**: non-release diagnostic — shape/smoke comparison only (`rmse_rel ≤ 1e-01`). Treat it as evidence that both implementations ran, not as numerical agreement.

Rows tagged with **📐** are the canonical parity references for this method (declared in [`parity_timing.registry`](../benchmarks/methodology.md)). C++ and external rows show reference parity; pls4all language bindings show binding parity against the C++ backend. Hover the icon for role and tolerance band.

::::{tab-set}
:class: parity-tabs

:::{tab-item} 1 thread
:sync: threads-1

<div class="parity-table-wrap">
<table class="docutils parity-grouped">
<thead><tr><th scope="col">Backend</th><th scope="col">Divergence</th><th class="size-col" scope="col">100×50 (ms)</th><th class="size-col" scope="col">100×500 (ms)</th><th class="size-col" scope="col">100×2500 (ms)</th><th class="size-col" scope="col">1000×500 (ms)</th></tr></thead>
<tbody class="lang-band lang-cpp"><tr class="lang-band-row" data-lang="cpp"><th colspan="6" scope="rowgroup"><span class="lang-band-dot"></span>C++ native · libp4a</th></tr>
<tr class="bk-row"><td class="bk-name"><code>pls4all.cpp.blas</code></td><td class="parity parity-divergence parity-divergent">2.2e-03</td><td class="ms">1.03 ms</td><td class="ms">8.58 ms</td><td class="ms ms-best">41.2 ms<span class="medal" title="fastest">🏆</span></td><td class="ms">84.5 ms</td></tr>
<tr class="bk-row"><td class="bk-name"><code>pls4all.cpp.blas+omp</code></td><td class="parity parity-divergence parity-divergent">2.2e-03</td><td class="ms">1.06 ms</td><td class="ms ms-best">8.42 ms<span class="medal" title="fastest">🏆</span></td><td class="ms">42.4 ms</td><td class="ms">85.9 ms</td></tr>
<tr class="bk-row"><td class="bk-name"><code>pls4all.cpp.omp</code></td><td class="parity parity-divergence parity-divergent">2.2e-03</td><td class="ms ms-best">1.00 ms<span class="medal" title="fastest">🏆</span></td><td class="ms">11.3 ms</td><td class="ms">42.9 ms</td><td class="ms">85.5 ms</td></tr>
<tr class="bk-row"><td class="bk-name"><code>pls4all.cpp.ref</code></td><td class="parity parity-divergence parity-divergent">2.2e-03</td><td class="ms">1.04 ms</td><td class="ms">8.48 ms</td><td class="ms">44.1 ms</td><td class="ms">86.5 ms</td></tr>
</tbody>
<tbody class="lang-band lang-python"><tr class="lang-band-row" data-lang="python"><th colspan="6" scope="rowgroup"><span class="lang-band-dot"></span>Python · pls4all</th></tr>
<tr class="bk-row"><td class="bk-name"><code>pls4all.python</code></td><td class="parity parity-divergence parity-divergent">2.2e-03</td><td class="ms">1.01 ms</td><td class="ms">8.63 ms</td><td class="ms">41.7 ms</td><td class="ms">85.1 ms</td></tr>
<tr class="bk-row"><td class="bk-name"><code>pls4all.sklearn</code></td><td class="parity parity-divergence parity-divergent">2.2e-03</td><td class="ms">1.29 ms</td><td class="ms">8.72 ms</td><td class="ms">42.5 ms</td><td class="ms">88.7 ms</td></tr>
</tbody>
<tbody class="lang-band lang-r"><tr class="lang-band-row" data-lang="r"><th colspan="6" scope="rowgroup"><span class="lang-band-dot"></span>R · pls4all</th></tr>
<tr class="bk-row"><td class="bk-name"><code>pls4all.R</code></td><td class="parity parity-divergence parity-divergent">2.2e-03</td><td class="ms">3.17 ms</td><td class="ms">52.6 ms</td><td class="ms">396.4 ms</td><td class="ms">491.4 ms</td></tr>
<tr class="bk-row"><td class="bk-name"><code>pls4all.R.formula</code></td><td class="parity parity-divergence parity-divergent">2.2e-03</td><td class="ms">6.68 ms</td><td class="ms">77.6 ms</td><td class="ms">570.8 ms</td><td class="ms">688.3 ms</td></tr>
<tr class="bk-row"><td class="bk-name"><code>pls4all.R.mdatools</code></td><td class="parity parity-divergence parity-divergent">1.2e-03</td><td class="ms">5.11 ms</td><td class="ms">65.7 ms</td><td class="ms">568.7 ms</td><td class="ms">681.4 ms</td></tr>
<tr class="bk-row"><td class="bk-name"><code>pls4all.R.pls</code></td><td class="parity parity-divergence parity-divergent">1.2e-03</td><td class="ms">4.87 ms</td><td class="ms">69.1 ms</td><td class="ms">574.2 ms</td><td class="ms">667.5 ms</td></tr>
</tbody>
<tbody class="lang-band lang-matlab"><tr class="lang-band-row" data-lang="matlab"><th colspan="6" scope="rowgroup"><span class="lang-band-dot"></span>MATLAB · pls4all</th></tr>
<tr class="bk-row"><td class="bk-name"><code>pls4all.matlab</code></td><td class="parity parity-divergence parity-divergent">2.2e-03</td><td class="ms">1.81 ms</td><td class="ms">13.8 ms</td><td class="ms">64.8 ms</td><td class="ms">130.2 ms</td></tr>
<tr class="bk-row"><td class="bk-name"><code>pls4all.matlab.classdef</code></td><td class="parity parity-divergence parity-divergent">2.2e-03</td><td class="ms">2.61 ms</td><td class="ms">13.7 ms</td><td class="ms">65.8 ms</td><td class="ms">131.9 ms</td></tr>
</tbody>
<tbody class="lang-band lang-python"><tr class="lang-band-row" data-lang="python"><th colspan="6" scope="rowgroup"><span class="lang-band-dot"></span>Python · external</th></tr>
<tr class="bk-row truth-source truth-source-qualitative"><td class="bk-name"><span class="truth-mark" title="Registry parity reference (python): scikit-learn 1.4.2 — qualitative (rmse_rel ≤ 1e-01)">📐</span><code>ref.python_scikit_learn</code></td><td class="parity parity-divergence parity-ref-source">0</td><td class="ms">1.58 ms</td><td class="ms">9.08 ms</td><td class="ms">41.9 ms</td><td class="ms ms-best">83.4 ms<span class="medal" title="fastest">🏆</span></td></tr>
</tbody>
</table>
</div>

:::

::::


---

_See also_: [benchmark overview](../benchmarks/overview.md) · [methods index](index.md) · [interactive dashboard](../landing/dashboard.md)