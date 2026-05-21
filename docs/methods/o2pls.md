# `o2pls` — O2-PLS (two-way orthogonal)

_Group_: **Multi-block / cross-modal** · _Registry tolerance_: `1.0`

## Description

O2PLS — bi-directional OPLS (Trygg & Wold 2003)

From the `pls4all.sklearn.O2PLSRegression` docstring:

> O2-PLS (bi-directional OPLS, Trygg & Wold 2003).

> **Registry note** — R `OmicsPLS::o2m` 2.1.0 implements a joint iterative SVD O2PLS variant — differs from pls4all's peel-then-PLS algorithm (Trygg & Wold 2003 §3.2). Both are valid O2PLS formulations; predictions diverge by ~45% RMS so exact parity is not possible. Tolerance widened to 1.0 to flag the R ref is present.

### Parameters

| Name | Type | Default | Notes |
|------|------|---------|-------|
| `n_predictive` | `int` | `2` | Number of joint (predictive) components shared by X and Y in O2-PLS. |
| `n_x_orthogonal` | `int` | `1` | Number of X-orthogonal components (Y-irrelevant structure in X). |
| `n_y_orthogonal` | `int` | `1` | Number of Y-orthogonal components (X-irrelevant structure in Y). |
| `n_targets` | `int` | `4` | registry benchmark cell value |

## Explanations

### Bibliographic source

Trygg, J. & Wold, S. (2003). *O2-PLS, a two-block (X–Y) latent variable regression method with an integral OSC filter*. Journal of Chemometrics 17(1), 53–64.

### Mathematical principle

O2-PLS extends OPLS symmetrically to both $\mathbf{X}$ and $\mathbf{Y}$: it decomposes each block into a joint predictive component plus block-orthogonal components. Unlike OPLS, which is asymmetric ($\mathbf{Y}$ drives the decomposition of $\mathbf{X}$), O2-PLS treats both matrices as observation blocks of equal status.

Required hyperparameters: $n_{\mathrm{pred}}$ (joint components), $n_{\mathrm{X,ortho}}$ (X-unique orthogonal), $n_{\mathrm{Y,ortho}}$ (Y-unique orthogonal). Choosing all three by cross-validation is a 3-D grid which can be expensive; a common compromise fixes the orthogonal counts at 1 and tunes only $n_{\mathrm{pred}}$.

O2-PLS is dominant in metabolomics ↔ transcriptomics integration where the analyst wants to disentangle platform-specific orthogonal variation from biology that is consistent across platforms.

### Implementation

`p4a_o2pls_fit`. Reference: CRAN `OmicsPLS 2.1.0`.

R roxygen note (`sklearn_extra.R::o2pls`):

> O2-PLS — formula entry point (uses n_predictive for component count).

MATLAB header (`bindings/matlab/+pls4all/O2plsRegression.m`):

```text
pls4all.O2plsRegression  O2-PLS (bi-directional OPLS).
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
p4a_o2pls_fit(ctx, cfg, &x_view, &y_view, /* hyperparams */, &res);
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
from pls4all._methods import o2pls_fit
with pls4all.Context() as ctx, pls4all.Config() as cfg:
    res = o2pls_fit(ctx, cfg, X, y)
# then: res.matrix("predictions"), res.matrix("coefficients"),
# res.vector("mask"), res.scalar("intercept"), …
```

:::

:::{tab-item} Python · pls4all.sklearn
:sync: python-sklearn
:class-label: lang-python

```python
from pls4all.sklearn import O2PLSRegression
mdl = O2PLSRegression(n_predictive=2, n_x_orthogonal=1, n_y_orthogonal=1)
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
res <- pls4all_method("o2pls", X, y,
                      n_components = 2L, params = list(n_targets = 4L, n_predictive = 2L, n_x_orthogonal = 1L, n_y_orthogonal = 1L))
# res is a named list with MethodResult arrays/scalars.
# selected_indices / top_k_intervals are 1-based.
```

:::

:::{tab-item} R · pls4all (raw fn)
:sync: r-raw
:class-label: lang-r

```r
library(pls4all)
res  <- o2pls_fit(X, Y, n_predictive = 2L,
           n_x_orthogonal = 1L, n_y_orthogonal = 1L)
yhat <- pls4all_predict(res, X_test)
```

:::

:::{tab-item} R · pls4all (formula+S3)
:sync: r-formula
:class-label: lang-r

```r
library(pls4all)
fit  <- o2pls(y ~ ., data = train, ncomp = 2L)
yhat <- predict(fit, newdata = test)
summary(fit)
```

:::

:::{tab-item} MATLAB · pls4all (MEX)
:sync: matlab-mex
:class-label: lang-matlab

```matlab
res = pls4all.o2pls(X, y, 2);
% see header of bindings/matlab/+pls4all/o2pls.m for full
% parameter surface:
%   res = o2pls(X, Y, n_predictive, n_x_orthogonal, n_y_orthogonal)
yhat = predict(res, Xtest);
```

:::

:::{tab-item} MATLAB · pls4all (classdef)
:sync: matlab-classdef
:class-label: lang-matlab

```matlab
mdl  = pls4all.fit("o2pls", X, y, "NumComponents", 2);
yhat = predict(mdl, Xtest);
```

:::

::::


**Registry parity references** 📐

:::{card}
:class-card: external-refs

- 📐 **`ref.r_omicspls`** (R · r) — `OmicsPLS` 2.1.0 · qualitative (rmse_rel ≤ 1e+00) — R `OmicsPLS::o2m` (Bouhaddani 2018) implements a joint iterative SVD O2PLS variant — differs from pls4all's peel-then-PLS algorithm. Tolerance widened to admit the expected ~45% RMS divergence.
:::

### Validation contract

Structural validation contract for `o2pls`, exported by the declarative validation framework (`benchmarks/validation/registry/`) from `benchmarks.parity_timing.registry`. Regenerated by `python -m benchmarks.validation.export_registry --write`.

| Field | Value |
|------|-------|
| Reference kind | `external` |
| Canonical reference role | `r` |
| Declared references | R |
| Comparator | `reference_parity` (external-reference validity gate) |
| Comparator metric | `rmse_rel` |
| Reference tolerance | `1e+00` (qualitative) |
| Binding tolerance | `1e-06` |
| Prediction key | `predictions` |
| Required data flags | none |

**Benchmark cell parameters**

| Parameter | Value |
|-----------|-------|
| `n_features` | `30` |
| `n_predictive` | `2` |
| `n_samples` | `200` |
| `n_targets` | `4` |
| `n_x_orthogonal` | `1` |
| `n_y_orthogonal` | `1` |

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

**Reference gate**: non-release diagnostic — shape/smoke comparison only (`rmse_rel ≤ 1e+00`). Treat it as evidence that both implementations ran, not as numerical agreement.

Rows tagged with **📐** are the canonical parity references for this method (declared in [`parity_timing.registry`](../benchmarks/methodology.md)). C++ and external rows show reference parity; pls4all language bindings show binding parity against the C++ backend. Hover the icon for role and tolerance band.

::::{tab-set}
:class: parity-tabs

:::{tab-item} 1 thread
:sync: threads-1

<div class="parity-table-wrap">
<table class="docutils parity-grouped">
<thead><tr><th scope="col">Backend</th><th scope="col">Divergence</th><th class="size-col" scope="col">100×50 (ms)</th></tr></thead>
<tbody class="lang-band lang-cpp"><tr class="lang-band-row" data-lang="cpp"><th colspan="3" scope="rowgroup"><span class="lang-band-dot"></span>C++ native · libp4a</th></tr>
<tr class="bk-row"><td class="bk-name"><code>pls4all.cpp.blas+omp</code></td><td class="parity parity-divergence parity-divergent">0.12</td><td class="ms ms-best">1.05 ms<span class="medal" title="fastest">🏆</span></td></tr>
</tbody>
<tbody class="lang-band lang-python"><tr class="lang-band-row" data-lang="python"><th colspan="3" scope="rowgroup"><span class="lang-band-dot"></span>Python · pls4all</th></tr>
<tr class="bk-row"><td class="bk-name"><code>pls4all.python</code></td><td class="parity parity-divergence parity-divergent">0.12</td><td class="ms">1.12 ms</td></tr>
<tr class="bk-row"><td class="bk-name"><code>pls4all.sklearn</code></td><td class="parity parity-divergence parity-divergent">0.12</td><td class="ms">1.26 ms</td></tr>
</tbody>
<tbody class="lang-band lang-r"><tr class="lang-band-row" data-lang="r"><th colspan="3" scope="rowgroup"><span class="lang-band-dot"></span>R · pls4all</th></tr>
<tr class="bk-row"><td class="bk-name"><code>pls4all.R</code></td><td class="parity parity-divergence parity-divergent">0.12</td><td class="ms">2.56 ms</td></tr>
<tr class="bk-row"><td class="bk-name"><code>pls4all.R.formula</code></td><td class="parity parity-divergence parity-divergent">0.12</td><td class="ms">3.61 ms</td></tr>
</tbody>
<tbody class="lang-band lang-matlab"><tr class="lang-band-row" data-lang="matlab"><th colspan="3" scope="rowgroup"><span class="lang-band-dot"></span>MATLAB · pls4all</th></tr>
<tr class="bk-row"><td class="bk-name"><code>pls4all.matlab</code></td><td class="parity parity-divergence parity-divergent">0.12</td><td class="ms">1.73 ms</td></tr>
<tr class="bk-row"><td class="bk-name"><code>pls4all.matlab.classdef</code></td><td class="parity parity-divergence parity-divergent">0.12</td><td class="ms">2.23 ms</td></tr>
</tbody>
<tbody class="lang-band lang-r"><tr class="lang-band-row" data-lang="r"><th colspan="3" scope="rowgroup"><span class="lang-band-dot"></span>R · external</th></tr>
<tr class="bk-row truth-source truth-source-qualitative"><td class="bk-name"><span class="truth-mark" title="Registry parity reference (r): OmicsPLS 2.1.0 — qualitative (rmse_rel ≤ 1e+00)">📐</span><code>ref.r_omicspls</code></td><td class="parity parity-divergence parity-ref-source">0</td><td class="ms">948.8 ms</td></tr>
</tbody>
</table>
</div>

:::

::::


---

_See also_: [benchmark overview](../benchmarks/overview.md) · [methods index](index.md) · [interactive dashboard](../landing/dashboard.md)