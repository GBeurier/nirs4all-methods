# `continuum_regression` — Continuum Regression (Stone & Brooks 1990)

_Group_: **Nonlinear / local** · _Registry tolerance_: `0.2`

## Description

Continuum regression (interpolates PLS / OLS)

From the `pls4all.sklearn.ContinuumRegression` docstring:

> Continuum regression τ ∈ [0, 1] interpolates PLS (1) / OLS (0).

> **Registry note** — R `JICO::continuum` (Stone & Brooks 1990). Different parameterization (lambda, gamma, om) than pls4all's single-τ knob. The adapter reconstructs fitted values by regressing Y on JICO's latent scores, leaving a small but real parameterization gap.

### Parameters

| Name | Type | Default | Notes |
|------|------|---------|-------|
| `n_components` | `int` | `2` | Number of latent components extracted (k). |
| `tau` | `float` | `0.5` | Continuum mixing parameter in [0, 1]; 0 ≈ PLS, 1 ≈ whitened OLS / PCR-like. |

## Explanations

### Bibliographic source

Stone, M. & Brooks, R. J. (1990). *Continuum regression: cross-validated sequentially constructed prediction embracing ordinary least squares, partial least squares and principal components regression*. JRSS B 52(2), 237–269.

### Mathematical principle

Continuum regression introduces a single shape parameter $\tau \in [0, 1]$ that selects the loading-weight criterion: $\mathbf{w} \propto \operatorname{Cov}(\mathbf{X}\mathbf{w}, \mathbf{y})^{\tau} \cdot \operatorname{Var}(\mathbf{X}\mathbf{w})^{1-\tau}$. Special cases: $\tau = 0$ gives PCR (variance-maximising), $\tau = 1/2$ gives PLS (covariance-maximising), $\tau = 1$ gives OLS (correlation-maximising, in the appropriate limit).

Cross-validating $\tau$ on a fine grid often improves RMSE over the discrete PLS / PCR choices — the optimum is rarely exactly at $\tau = 0.5$. Stone & Brooks' original treatment also cross-validates the number of components $k$ jointly with $\tau$, producing a 2-D grid.

Implementation note: numerically stable continuum regression uses the parameterised power method on the matrix $\mathbf{X}^{\top}\mathbf{y}\mathbf{y}^{\top}\mathbf{X} / (\mathbf{X}^{\top}\mathbf{X})^{1-\tau}$, which avoids forming the rank-1 outer product explicitly and is what pls4all uses.

### Implementation

`p4a_continuum_regression_fit` (in-sample only).

R roxygen note (`sklearn_extra.R::continuum_regression`):

> Continuum regression — formula entry point.

MATLAB header (`bindings/matlab/+pls4all/ContinuumRegression.m`):

```text
pls4all.ContinuumRegression  Continuum regression (tau ∈ [0, 1]).
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
p4a_continuum_regression_fit(ctx, cfg, &x_view, &y_view, /* hyperparams */, &res);
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
from pls4all._methods import continuum_regression_fit
with pls4all.Context() as ctx, pls4all.Config() as cfg:
    res = continuum_regression_fit(ctx, cfg, X, y, n_components=4)
# then: res.matrix("predictions"), res.matrix("coefficients"),
# res.vector("mask"), res.scalar("intercept"), …
```

:::

:::{tab-item} Python · pls4all.sklearn
:sync: python-sklearn
:class-label: lang-python

```python
from pls4all.sklearn import ContinuumRegression
mdl = ContinuumRegression(n_components=2, tau=0.5)
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
res <- pls4all_method("continuum_regression", X, y,
                      n_components = 4L, params = list(tau = 0.5))
# res is a named list with MethodResult arrays/scalars.
# selected_indices / top_k_intervals are 1-based.
```

:::

:::{tab-item} R · pls4all (raw fn)
:sync: r-raw
:class-label: lang-r

```r
library(pls4all)
res  <- continuum_regression_fit(X, Y, n_components, tau = 0.5)
yhat <- pls4all_predict(res, X_test)
```

:::

:::{tab-item} R · pls4all (formula+S3)
:sync: r-formula
:class-label: lang-r

```r
library(pls4all)
fit  <- continuum_regression(y ~ ., data = train, ncomp = 4L)
yhat <- predict(fit, newdata = test)
summary(fit)
```

:::

:::{tab-item} MATLAB · pls4all (MEX)
:sync: matlab-mex
:class-label: lang-matlab

```matlab
res = pls4all.continuum_regression(X, y, 4);
% see header of bindings/matlab/+pls4all/continuum_regression.m for full
% parameter surface:
%   res = continuum_regression(X, Y, n_components, tau)
yhat = predict(res, Xtest);
```

:::

:::{tab-item} MATLAB · pls4all (classdef)
:sync: matlab-classdef
:class-label: lang-matlab

```matlab
mdl  = pls4all.fit("continuum_regression", X, y, "NumComponents", 4);
yhat = predict(mdl, Xtest);
```

:::

::::


**Registry parity references** 📐

:::{card}
:class-card: external-refs

- 📐 **`ref.r_jico`** (R · r) — `JICO` 0.1 · qualitative (rmse_rel ≤ 2e-01) — R `JICO::continuum` (Stone & Brooks 1990). Different parameterization than pls4all — JICO uses (lambda, gamma, om) while pls4all maps a single τ. Predictions are reconstructed by regressing Y on JICO's latent scores.
:::

### Validation contract

Structural validation contract for `continuum_regression`, exported by the declarative validation framework (`benchmarks/validation/registry/`) from `benchmarks.parity_timing.registry`. Regenerated by `python -m benchmarks.validation.export_registry --write`.

| Field | Value |
|------|-------|
| Reference kind | `external` |
| Canonical reference role | `r` |
| Declared references | R |
| Comparator | `reference_parity` (external-reference validity gate) |
| Comparator metric | `rmse_rel` |
| Reference tolerance | `2e-01` (qualitative) |
| Binding tolerance | `1e-06` |
| Prediction key | `predictions` |
| Required data flags | none |

**Benchmark cell parameters**

| Parameter | Value |
|-----------|-------|
| `n_components` | `4` |
| `n_features` | `50` |
| `n_samples` | `200` |
| `tau` | `0.5` |

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

**Reference gate**: non-release diagnostic — shape/smoke comparison only (`rmse_rel ≤ 2e-01`). Treat it as evidence that both implementations ran, not as numerical agreement.

Rows tagged with **📐** are the canonical parity references for this method (declared in [`parity_timing.registry`](../benchmarks/methodology.md)). C++ and external rows show reference parity; pls4all language bindings show binding parity against the C++ backend. Hover the icon for role and tolerance band.

::::{tab-set}
:class: parity-tabs

:::{tab-item} 1 thread
:sync: threads-1

<div class="parity-table-wrap">
<table class="docutils parity-grouped">
<thead><tr><th scope="col">Backend</th><th scope="col">Divergence</th><th class="size-col" scope="col">100×50 (ms)</th><th class="size-col" scope="col">100×500 (ms)</th><th class="size-col" scope="col">100×2500 (ms)</th><th class="size-col" scope="col">1000×500 (ms)</th></tr></thead>
<tbody class="lang-band lang-cpp"><tr class="lang-band-row" data-lang="cpp"><th colspan="6" scope="rowgroup"><span class="lang-band-dot"></span>C++ native · libp4a</th></tr>
<tr class="bk-row"><td class="bk-name"><code>pls4all.cpp.blas</code></td><td class="parity parity-divergence parity-divergent">6.2e-02</td><td class="ms">1.05 ms</td><td class="ms">8.75 ms</td><td class="ms">44.5 ms</td><td class="ms">86.8 ms</td></tr>
<tr class="bk-row"><td class="bk-name"><code>pls4all.cpp.blas+omp</code></td><td class="parity parity-divergence parity-divergent">6.2e-02</td><td class="ms">1.07 ms</td><td class="ms">8.64 ms</td><td class="ms">44.1 ms</td><td class="ms ms-best">86.7 ms<span class="medal" title="fastest">🏆</span></td></tr>
<tr class="bk-row"><td class="bk-name"><code>pls4all.cpp.omp</code></td><td class="parity parity-divergence parity-divergent">6.2e-02</td><td class="ms">1.06 ms</td><td class="ms">8.73 ms</td><td class="ms ms-best">43.7 ms<span class="medal" title="fastest">🏆</span></td><td class="ms">87.0 ms</td></tr>
<tr class="bk-row"><td class="bk-name"><code>pls4all.cpp.ref</code></td><td class="parity parity-divergence parity-divergent">6.2e-02</td><td class="ms">1.13 ms</td><td class="ms ms-best">8.60 ms<span class="medal" title="fastest">🏆</span></td><td class="ms">44.0 ms</td><td class="ms">87.9 ms</td></tr>
</tbody>
<tbody class="lang-band lang-python"><tr class="lang-band-row" data-lang="python"><th colspan="6" scope="rowgroup"><span class="lang-band-dot"></span>Python · pls4all</th></tr>
<tr class="bk-row"><td class="bk-name"><code>pls4all.python</code></td><td class="parity parity-divergence parity-divergent">6.2e-02</td><td class="ms ms-best">1.03 ms<span class="medal" title="fastest">🏆</span></td><td class="ms">8.63 ms</td><td class="ms">44.1 ms</td><td class="ms">87.1 ms</td></tr>
<tr class="bk-row"><td class="bk-name"><code>pls4all.sklearn</code></td><td class="parity parity-divergence parity-divergent">6.2e-02</td><td class="ms">1.20 ms</td><td class="ms">9.10 ms</td><td class="ms">45.4 ms</td><td class="ms">92.6 ms</td></tr>
</tbody>
<tbody class="lang-band lang-r"><tr class="lang-band-row" data-lang="r"><th colspan="6" scope="rowgroup"><span class="lang-band-dot"></span>R · pls4all</th></tr>
<tr class="bk-row"><td class="bk-name"><code>pls4all.R</code></td><td class="parity parity-divergence parity-divergent">6.2e-02</td><td class="ms">3.19 ms</td><td class="ms">42.3 ms</td><td class="ms">401.2 ms</td><td class="ms">459.9 ms</td></tr>
<tr class="bk-row"><td class="bk-name"><code>pls4all.R.formula</code></td><td class="parity parity-divergence parity-divergent">6.2e-02</td><td class="ms">4.55 ms</td><td class="ms">56.5 ms</td><td class="ms">578.6 ms</td><td class="ms">598.5 ms</td></tr>
<tr class="bk-row"><td class="bk-name"><code>pls4all.R.mdatools</code></td><td class="parity parity-divergence parity-drift">—</td><td class="ms">4.71 ms</td><td class="ms">67.7 ms</td><td class="ms">573.8 ms</td><td class="ms">595.6 ms</td></tr>
<tr class="bk-row"><td class="bk-name"><code>pls4all.R.pls</code></td><td class="parity parity-divergence parity-drift">—</td><td class="ms">4.40 ms</td><td class="ms">56.9 ms</td><td class="ms">546.8 ms</td><td class="ms">577.0 ms</td></tr>
</tbody>
<tbody class="lang-band lang-matlab"><tr class="lang-band-row" data-lang="matlab"><th colspan="6" scope="rowgroup"><span class="lang-band-dot"></span>MATLAB · pls4all</th></tr>
<tr class="bk-row"><td class="bk-name"><code>pls4all.matlab</code></td><td class="parity parity-divergence parity-divergent">6.2e-02</td><td class="ms">1.71 ms</td><td class="ms">13.8 ms</td><td class="ms">68.5 ms</td><td class="ms">133.5 ms</td></tr>
<tr class="bk-row"><td class="bk-name"><code>pls4all.matlab.classdef</code></td><td class="parity parity-divergence parity-divergent">6.2e-02</td><td class="ms">2.09 ms</td><td class="ms">14.0 ms</td><td class="ms">68.9 ms</td><td class="ms">137.7 ms</td></tr>
</tbody>
<tbody class="lang-band lang-r"><tr class="lang-band-row" data-lang="r"><th colspan="6" scope="rowgroup"><span class="lang-band-dot"></span>R · external</th></tr>
<tr class="bk-row truth-source truth-source-qualitative"><td class="bk-name"><span class="truth-mark" title="Registry parity reference (r): JICO 0.1 — qualitative (rmse_rel ≤ 2e-01)">📐</span><code>ref.r_jico</code></td><td class="parity parity-divergence parity-ref-source">0</td><td class="ms">32.9 ms</td><td class="ms">150.5 ms</td><td class="ms">1.2 s</td><td class="ms">3.2 s</td></tr>
</tbody>
</table>
</div>

:::

::::


---

_See also_: [benchmark overview](../benchmarks/overview.md) · [methods index](index.md) · [interactive dashboard](../landing/dashboard.md)