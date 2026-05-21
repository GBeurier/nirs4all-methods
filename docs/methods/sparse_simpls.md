# `sparse_simpls` — Sparse SIMPLS (Chun & Keleş 2010)

_Group_: **Sparse** · _Registry tolerance_: `1.0`

## Description

Sparse SIMPLS with soft-threshold lambda

From the `pls4all.sklearn.SparseSimplsRegression` docstring:

> Sparse SIMPLS with soft-thresholded weights (Chun & Keles 2010).

> **Registry note** — R `spls` 2.3.2 is the canonical Chun & Keles reference. No widely installable Python port for sparse SIMPLS with this exact normalization.

### Parameters

| Name | Type | Default | Notes |
|------|------|---------|-------|
| `n_components` | `int` | `2` | Number of latent components extracted (k). |
| `sparsity_lambda` | `float` | `0.05` | L1 soft-threshold magnitude applied to the PLS weight vectors. |

## Explanations

### Bibliographic source

Chun, H. & Keleş, S. (2010). *Sparse partial least squares regression for simultaneous dimension reduction and variable selection*. JRSS B 72(1), 3–25.

### Mathematical principle

Sparse PLS adds a soft-thresholding step to each SIMPLS loading weight so that the latent direction is supported on only a small subset of features. Mathematically, after the un-thresholded weight $\mathbf{w}$ is computed, we solve $\mathbf{w}^{\star} = \arg\min_{\|\mathbf{c}\|=1} \|\mathbf{c} - \mathbf{w}\|_2^2 + \lambda \|\mathbf{c}\|_1$, which has the closed-form soft-threshold solution $c_j = \operatorname{sign}(w_j)\,(|w_j| - \lambda/2)_+$ followed by re-normalisation.

The penalty $\lambda$ controls sparsity: small $\lambda$ approaches standard PLS, large $\lambda$ zeroes most weights. In high-dimensional ($p \gg n$) spectroscopy or omics data, sparse PLS simultaneously builds the latent predictive direction *and* selects the variables that support it — a much cleaner story than running PLS then thresholding coefficients post-hoc.

The Chun & Keleş formulation differs subtly from the earlier Lê Cao 2008 sPLS (used in mixOmics): Chun & Keleş threshold the un-deflated weight while Lê Cao threshold the deflated weight at each iteration. pls4all implements the Chun & Keleş formulation.

### Implementation

`p4a_sparse_simpls_fit`. Reference: CRAN `spls 2.3.2` (Chun & Keleş authors). No widely installable Python port exists with this exact normalisation convention.

R roxygen note (`sklearn_methods.R::sparse_pls`):

> Sparse SIMPLS — formula entry point.

MATLAB header (`bindings/matlab/+pls4all/SparsePlsRegression.m`):

```text
pls4all.SparsePlsRegression — Sparse SIMPLS (Chun & Keles 2010)
 as a tier-2 classdef. Construct via the factory:

   mdl = pls4all.fitrsparsepls(X, y, "NumComponents", 5, "Lambda", 0.05)

 or directly:
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
p4a_sparse_simpls_fit(ctx, cfg, &x_view, &y_view, /* hyperparams */, &res);
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
from pls4all._methods import sparse_simpls_fit
with pls4all.Context() as ctx, pls4all.Config() as cfg:
    res = sparse_simpls_fit(ctx, cfg, X, y, n_components=4)
# then: res.matrix("predictions"), res.matrix("coefficients"),
# res.vector("mask"), res.scalar("intercept"), …
```

:::

:::{tab-item} Python · pls4all.sklearn
:sync: python-sklearn
:class-label: lang-python

```python
from pls4all.sklearn import SparseSimplsRegression
mdl = SparseSimplsRegression(n_components=2, sparsity_lambda=0.05)
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
res <- pls4all_method("sparse_simpls", X, y,
                      n_components = 4L, params = list(sparsity_lambda = 0.05))
# res is a named list with MethodResult arrays/scalars.
# selected_indices / top_k_intervals are 1-based.
```

:::

:::{tab-item} R · pls4all (raw fn)
:sync: r-raw
:class-label: lang-r

```r
library(pls4all)
res  <- sparse_simpls_fit(X, Y, n_components, sparsity_lambda = 0.05)
yhat <- pls4all_predict(res, X_test)
```

:::

:::{tab-item} R · pls4all (formula+S3)
:sync: r-formula
:class-label: lang-r

```r
library(pls4all)
fit  <- sparse_pls(y ~ ., data = train, ncomp = 4L)
yhat <- predict(fit, newdata = test)
summary(fit)
```

:::

:::{tab-item} MATLAB · pls4all (MEX)
:sync: matlab-mex
:class-label: lang-matlab

```matlab
res = pls4all.sparse_simpls(X, y, 4);
% see header of bindings/matlab/+pls4all/sparse_simpls.m for full
% parameter surface:
%   res = sparse_simpls(X, Y, n_components, sparsity_lambda)
yhat = predict(res, Xtest);
```

:::

:::{tab-item} MATLAB · pls4all (classdef)
:sync: matlab-classdef
:class-label: lang-matlab

```matlab
mdl  = pls4all.fit("sparse_simpls", X, y, "NumComponents", 4);
yhat = predict(mdl, Xtest);
```

:::

::::


**Registry parity references** 📐

:::{card}
:class-card: external-refs

- 📐 **`ref.r_spls`** (R · r) — `spls` 2.3.2 · qualitative (rmse_rel ≤ 1e+00) — R `spls` 2.3.2 (Chun & Keles). Predicts via the regression coefficient matrix from sparse-thresholded SIMPLS.
:::

### Validation contract

Structural validation contract for `sparse_simpls`, exported by the declarative validation framework (`benchmarks/validation/registry/`) from `benchmarks.parity_timing.registry`. Regenerated by `python -m benchmarks.validation.export_registry --write`.

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
| `n_components` | `4` |
| `n_features` | `50` |
| `n_samples` | `200` |
| `sparsity_lambda` | `0.05` |

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
<thead><tr><th scope="col">Backend</th><th scope="col">Divergence</th><th class="size-col" scope="col">20×6 (ms)</th><th class="size-col" scope="col">100×50 (ms)</th><th class="size-col" scope="col">100×500 (ms)</th><th class="size-col" scope="col">100×2500 (ms)</th><th class="size-col" scope="col">1000×500 (ms)</th></tr></thead>
<tbody class="lang-band lang-cpp"><tr class="lang-band-row" data-lang="cpp"><th colspan="7" scope="rowgroup"><span class="lang-band-dot"></span>C++ native · libp4a</th></tr>
<tr class="bk-row"><td class="bk-name"><code>pls4all.cpp.blas</code></td><td class="parity parity-divergence parity-divergent">3.4e-03</td><td class="ms ms-empty">—</td><td class="ms">1.00 ms</td><td class="ms">8.36 ms</td><td class="ms ms-best">40.6 ms<span class="medal" title="fastest">🏆</span></td><td class="ms">83.1 ms</td></tr>
<tr class="bk-row"><td class="bk-name"><code>pls4all.cpp.blas+omp</code></td><td class="parity parity-divergence parity-divergent">2.2e-02</td><td class="ms ms-empty">—</td><td class="ms">1.46 ms</td><td class="ms">8.12 ms</td><td class="ms">40.8 ms</td><td class="ms ms-best">82.6 ms<span class="medal" title="fastest">🏆</span></td></tr>
<tr class="bk-row"><td class="bk-name"><code>pls4all.cpp.omp</code></td><td class="parity parity-divergence parity-divergent">3.4e-03</td><td class="ms ms-empty">—</td><td class="ms">1.02 ms</td><td class="ms">8.23 ms</td><td class="ms">40.6 ms</td><td class="ms">83.4 ms</td></tr>
<tr class="bk-row"><td class="bk-name"><code>pls4all.cpp.ref</code></td><td class="parity parity-divergence parity-divergent">3.4e-03</td><td class="ms ms-empty">—</td><td class="ms ms-best">1.00 ms<span class="medal" title="fastest">🏆</span></td><td class="ms">8.32 ms</td><td class="ms">40.8 ms</td><td class="ms">83.4 ms</td></tr>
</tbody>
<tbody class="lang-band lang-python"><tr class="lang-band-row" data-lang="python"><th colspan="7" scope="rowgroup"><span class="lang-band-dot"></span>Python · pls4all</th></tr>
<tr class="bk-row"><td class="bk-name"><code>pls4all.python</code></td><td class="parity parity-divergence parity-divergent">2.2e-02</td><td class="ms ms-empty">—</td><td class="ms">1.02 ms</td><td class="ms ms-best">8.03 ms<span class="medal" title="fastest">🏆</span></td><td class="ms">41.1 ms</td><td class="ms">84.3 ms</td></tr>
<tr class="bk-row"><td class="bk-name"><code>pls4all.sklearn</code></td><td class="parity parity-divergence parity-divergent">2.2e-02</td><td class="ms ms-empty">—</td><td class="ms">1.24 ms</td><td class="ms">8.53 ms</td><td class="ms">42.4 ms</td><td class="ms">83.5 ms</td></tr>
</tbody>
<tbody class="lang-band lang-r"><tr class="lang-band-row" data-lang="r"><th colspan="7" scope="rowgroup"><span class="lang-band-dot"></span>R · pls4all</th></tr>
<tr class="bk-row"><td class="bk-name"><code>pls4all.R</code></td><td class="parity parity-divergence parity-divergent">2.2e-02</td><td class="ms ms-empty">—</td><td class="ms">3.41 ms</td><td class="ms">40.3 ms</td><td class="ms">390.2 ms</td><td class="ms">475.7 ms</td></tr>
<tr class="bk-row"><td class="bk-name"><code>pls4all.R.formula</code></td><td class="parity parity-divergence parity-divergent">2.2e-02</td><td class="ms ms-empty">—</td><td class="ms">5.01 ms</td><td class="ms">58.5 ms</td><td class="ms">559.8 ms</td><td class="ms">609.2 ms</td></tr>
<tr class="bk-row"><td class="bk-name"><code>pls4all.R.mdatools</code></td><td class="parity parity-divergence parity-divergent">9.2e-03</td><td class="ms ms-empty">—</td><td class="ms">4.91 ms</td><td class="ms">58.3 ms</td><td class="ms">562.4 ms</td><td class="ms">641.5 ms</td></tr>
<tr class="bk-row"><td class="bk-name"><code>pls4all.R.pls</code></td><td class="parity parity-divergence parity-divergent">9.2e-03</td><td class="ms ms-empty">—</td><td class="ms">4.64 ms</td><td class="ms">58.7 ms</td><td class="ms">549.4 ms</td><td class="ms">582.7 ms</td></tr>
</tbody>
<tbody class="lang-band lang-matlab"><tr class="lang-band-row" data-lang="matlab"><th colspan="7" scope="rowgroup"><span class="lang-band-dot"></span>MATLAB · pls4all</th></tr>
<tr class="bk-row"><td class="bk-name"><code>pls4all.matlab</code></td><td class="parity parity-divergence parity-divergent">2.2e-02</td><td class="ms ms-empty">—</td><td class="ms">1.71 ms</td><td class="ms">12.9 ms</td><td class="ms">65.7 ms</td><td class="ms">127.1 ms</td></tr>
<tr class="bk-row"><td class="bk-name"><code>pls4all.matlab.classdef</code></td><td class="parity parity-divergence parity-divergent">2.2e-02</td><td class="ms ms-empty">—</td><td class="ms">2.23 ms</td><td class="ms">13.6 ms</td><td class="ms">66.1 ms</td><td class="ms">133.4 ms</td></tr>
</tbody>
<tbody class="lang-band lang-r"><tr class="lang-band-row" data-lang="r"><th colspan="7" scope="rowgroup"><span class="lang-band-dot"></span>R · external</th></tr>
<tr class="bk-row"><td class="bk-name"><code>mixOmics</code></td><td class="parity parity-divergence parity-divergent">2.4e-02</td><td class="ms ms-empty">—</td><td class="ms">9.79 ms</td><td class="ms">60.3 ms</td><td class="ms">500.9 ms</td><td class="ms">705.2 ms</td></tr>
<tr class="bk-row truth-source truth-source-qualitative"><td class="bk-name"><span class="truth-mark" title="Registry parity reference (r): spls 2.3.2 — qualitative (rmse_rel ≤ 1e+00)">📐</span><code>ref.r_spls</code></td><td class="parity parity-divergence parity-ref-source">0</td><td class="ms ms-empty">—</td><td class="ms">10.5 ms</td><td class="ms">104.8 ms</td><td class="ms">1.1 s</td><td class="ms">1.0 s</td></tr>
</tbody>
<tbody class="lang-band lang-ext"><tr class="lang-band-row" data-lang="ext"><th colspan="7" scope="rowgroup"><span class="lang-band-dot"></span>Other</th></tr>
<tr class="bk-row"><td class="bk-name"><code>js_wasm</code></td><td class="parity parity-divergence parity-not_available">—</td><td class="ms ms-best">0.30 ms<span class="medal" title="fastest">🏆</span></td><td class="ms ms-empty">—</td><td class="ms ms-empty">—</td><td class="ms ms-empty">—</td><td class="ms ms-empty">—</td></tr>
</tbody>
</table>
</div>

:::

::::


---

_See also_: [benchmark overview](../benchmarks/overview.md) · [methods index](index.md) · [interactive dashboard](../landing/dashboard.md)