# `mb_pls` — Multi-block PLS (Westerhuis 1998)

_Group_: **Multi-block / cross-modal** · _Registry tolerance_: `0.001`

## Description

MB-PLS — Multi-block PLS (§17 Phase 4)

From the `pls4all.sklearn.MBPLSRegression` docstring:

> Multi-block PLS (Westerhuis 1998).

> **Registry note** — In-tree `nirs4all.operators.models.sklearn.mbpls.MBPLS` is the sanctioned external reference (the mbpls PyPI package is broken against sklearn 1.8). The C++ kernel mirrors the in-tree multiblock NIPALS implementation.

### Parameters

| Name | Type | Default | Notes |
|------|------|---------|-------|
| `n_components` | `int` | `2` | Number of latent components extracted (k). |
| `block_sizes` | `—` | `None` | Sequence of contiguous block widths defining the X-block partition (columns of X). |
| `n_blocks` | `int` | `3` | registry benchmark cell value |

## Explanations

### Bibliographic source

Westerhuis, J. A., Kourti, T. & MacGregor, J. F. (1998). *Analysis of multiblock and hierarchical PCA and PLS models*. Journal of Chemometrics 12(5), 301–321.

### Mathematical principle

When predictors come from several distinct sources — NIR, MIR, Raman, process tags, lab assays — concatenating them into one wide matrix lets the block with the most variance dominate. Multi-block PLS instead **block-scales** each $\mathbf{X}_b$ so blocks contribute proportionally to their information content rather than their dimensionality.

Formally, each block is centred and autoscaled, then scaled by $1 / \sqrt{p_b}$ so its total variance is unit-normalised. PLS then runs on the concatenated $[\mathbf{X}_1, \ldots, \mathbf{X}_B]$ with optional per-block weights. Block-level *importance* statistics (block-VIP, block-RMSE) are recovered from the loadings by restriction to each block's columns.

Compared to plain concatenation, MB-PLS gives interpretable per-block contributions and is the standard approach in process spectroscopy.

### Implementation

`p4a_mb_pls_fit` — requires a `block_sizes` integer vector summing to $p$. The C ABI materialises the intercept directly (no separate $\bar{\mathbf{y}}$ key) because the block scaling changes the centring semantics. Reference: sanctioned git-pinned port `nirs4all.operators.models.sklearn.mbpls`.

R roxygen note (`sklearn_methods.R::mb_pls`):

> Multi-block PLS — formula entry point.
> @param block_sizes Integer vector summing to the number of predictors.
> @inheritParams pls
> @export

MATLAB header (`bindings/matlab/+pls4all/MbPlsRegression.m`):

```text
pls4all.MbPlsRegression — Multi-block PLS.
 predict uses the stored intercept directly (coefficients are already
 in original X scale + intercept folds in y_mean - x_mean @ coef).
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
p4a_mb_pls_fit(ctx, cfg, &x_view, &y_view, /* hyperparams */, &res);
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
from pls4all._methods import mb_pls_fit
with pls4all.Context() as ctx, pls4all.Config() as cfg:
    res = mb_pls_fit(ctx, cfg, X, y, n_components=3)
# then: res.matrix("predictions"), res.matrix("coefficients"),
# res.vector("mask"), res.scalar("intercept"), …
```

:::

:::{tab-item} Python · pls4all.sklearn
:sync: python-sklearn
:class-label: lang-python

```python
from pls4all.sklearn import MBPLSRegression
mdl = MBPLSRegression(n_components=2, block_sizes=None)
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
res <- pls4all_method("mb_pls", X, y,
                      n_components = 3L, params = list(n_blocks = 3L))
# res is a named list with MethodResult arrays/scalars.
# selected_indices / top_k_intervals are 1-based.
```

:::

:::{tab-item} R · pls4all (raw fn)
:sync: r-raw
:class-label: lang-r

```r
library(pls4all)
res  <- mb_pls_fit(X, Y, n_components, block_sizes)
yhat <- pls4all_predict(res, X_test)
```

:::

:::{tab-item} R · pls4all (formula+S3)
:sync: r-formula
:class-label: lang-r

```r
library(pls4all)
fit  <- mb_pls(y ~ ., data = train, ncomp = 3L)
yhat <- predict(fit, newdata = test)
summary(fit)
```

:::

:::{tab-item} MATLAB · pls4all (MEX)
:sync: matlab-mex
:class-label: lang-matlab

```matlab
res = pls4all.mb_pls(X, y, 3);
% see header of bindings/matlab/+pls4all/mb_pls.m for full
% parameter surface:
%   res = mb_pls(X, Y, n_components, block_sizes)
yhat = predict(res, Xtest);
```

:::

:::{tab-item} MATLAB · pls4all (classdef)
:sync: matlab-classdef
:class-label: lang-matlab

```matlab
mdl  = pls4all.fit("mb_pls", X, y, "NumComponents", 3);
yhat = predict(mdl, Xtest);
```

:::

::::


**Registry parity references** 📐

:::{card}
:class-card: external-refs

- 📐 **`nirs4all`** (python · python) — `nirs4all` in-tree · relaxed (rmse_rel ≤ 1e-03) — In-tree Python MB-PLS (sanctioned external reference). The mbpls PyPI package is broken against sklearn 1.8 (uses the deprecated `force_all_finite` kwarg). nirs4all's implementation is a clean re-derivation of Westerhuis 1998.
:::

### Validation contract

Structural validation contract for `mb_pls`, exported by the declarative validation framework (`benchmarks/validation/registry/`) from `benchmarks.parity_timing.registry`. Regenerated by `python -m benchmarks.validation.export_registry --write`.

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
| `n_blocks` | `3` |
| `n_components` | `3` |
| `n_features` | `60` |
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

**Reference gate**: non-release diagnostic — shape/smoke comparison only (`rmse_rel ≤ 2e+00`). Treat it as evidence that both implementations ran, not as numerical agreement.

Rows tagged with **📐** are the canonical parity references for this method (declared in [`parity_timing.registry`](../benchmarks/methodology.md)). C++ and external rows show reference parity; pls4all language bindings show binding parity against the C++ backend. Hover the icon for role and tolerance band.

::::{tab-set}
:class: parity-tabs

:::{tab-item} 1 thread
:sync: threads-1

<div class="parity-table-wrap">
<table class="docutils parity-grouped">
<thead><tr><th scope="col">Backend</th><th scope="col">Divergence</th><th class="size-col" scope="col">100×50 (ms)</th><th class="size-col" scope="col">100×500 (ms)</th><th class="size-col" scope="col">100×2500 (ms)</th><th class="size-col" scope="col">1000×500 (ms)</th></tr></thead>
<tbody class="lang-band lang-cpp"><tr class="lang-band-row" data-lang="cpp"><th colspan="6" scope="rowgroup"><span class="lang-band-dot"></span>C++ native · libp4a</th></tr>
<tr class="bk-row"><td class="bk-name"><code>pls4all.cpp.blas</code></td><td class="parity parity-divergence parity-divergent">6.1e-02</td><td class="ms">1.04 ms</td><td class="ms">8.50 ms</td><td class="ms">46.4 ms</td><td class="ms ms-best">85.1 ms<span class="medal" title="fastest">🏆</span></td></tr>
<tr class="bk-row"><td class="bk-name"><code>pls4all.cpp.blas+omp</code></td><td class="parity parity-divergence parity-divergent">6.1e-02</td><td class="ms">1.03 ms</td><td class="ms">8.51 ms</td><td class="ms">45.8 ms</td><td class="ms">86.0 ms</td></tr>
<tr class="bk-row"><td class="bk-name"><code>pls4all.cpp.omp</code></td><td class="parity parity-divergence parity-divergent">6.1e-02</td><td class="ms ms-best">1.03 ms<span class="medal" title="fastest">🏆</span></td><td class="ms">8.43 ms</td><td class="ms">45.4 ms</td><td class="ms">86.2 ms</td></tr>
<tr class="bk-row"><td class="bk-name"><code>pls4all.cpp.ref</code></td><td class="parity parity-divergence parity-divergent">6.1e-02</td><td class="ms">1.04 ms</td><td class="ms">8.64 ms</td><td class="ms">45.6 ms</td><td class="ms">86.4 ms</td></tr>
</tbody>
<tbody class="lang-band lang-python"><tr class="lang-band-row" data-lang="python"><th colspan="6" scope="rowgroup"><span class="lang-band-dot"></span>Python · pls4all</th></tr>
<tr class="bk-row"><td class="bk-name"><code>pls4all.python</code></td><td class="parity parity-divergence parity-divergent">6.1e-02</td><td class="ms">1.04 ms</td><td class="ms ms-best">8.22 ms<span class="medal" title="fastest">🏆</span></td><td class="ms">45.7 ms</td><td class="ms">88.1 ms</td></tr>
<tr class="bk-row"><td class="bk-name"><code>pls4all.sklearn</code></td><td class="parity parity-divergence parity-divergent">6.1e-02</td><td class="ms">1.26 ms</td><td class="ms">8.73 ms</td><td class="ms ms-best">43.6 ms<span class="medal" title="fastest">🏆</span></td><td class="ms">85.6 ms</td></tr>
</tbody>
<tbody class="lang-band lang-r"><tr class="lang-band-row" data-lang="r"><th colspan="6" scope="rowgroup"><span class="lang-band-dot"></span>R · pls4all</th></tr>
<tr class="bk-row"><td class="bk-name"><code>pls4all.R</code></td><td class="parity parity-divergence parity-divergent">6.1e-02</td><td class="ms">3.35 ms</td><td class="ms">38.9 ms</td><td class="ms">466.6 ms</td><td class="ms">475.9 ms</td></tr>
<tr class="bk-row"><td class="bk-name"><code>pls4all.R.formula</code></td><td class="parity parity-divergence parity-divergent">6.1e-02</td><td class="ms">4.63 ms</td><td class="ms">58.2 ms</td><td class="ms">663.9 ms</td><td class="ms">571.6 ms</td></tr>
<tr class="bk-row"><td class="bk-name"><code>pls4all.R.mdatools</code></td><td class="parity parity-divergence parity-divergent">3.8e-03</td><td class="ms">4.87 ms</td><td class="ms">55.9 ms</td><td class="ms">649.5 ms</td><td class="ms">660.4 ms</td></tr>
<tr class="bk-row"><td class="bk-name"><code>pls4all.R.pls</code></td><td class="parity parity-divergence parity-divergent">3.8e-03</td><td class="ms">4.87 ms</td><td class="ms">61.4 ms</td><td class="ms">616.7 ms</td><td class="ms">580.4 ms</td></tr>
</tbody>
<tbody class="lang-band lang-matlab"><tr class="lang-band-row" data-lang="matlab"><th colspan="6" scope="rowgroup"><span class="lang-band-dot"></span>MATLAB · pls4all</th></tr>
<tr class="bk-row"><td class="bk-name"><code>pls4all.matlab</code></td><td class="parity parity-divergence parity-divergent">6.1e-02</td><td class="ms">1.81 ms</td><td class="ms">13.0 ms</td><td class="ms">67.1 ms</td><td class="ms">134.5 ms</td></tr>
<tr class="bk-row"><td class="bk-name"><code>pls4all.matlab.classdef</code></td><td class="parity parity-divergence parity-divergent">6.1e-02</td><td class="ms">2.48 ms</td><td class="ms">13.7 ms</td><td class="ms">69.7 ms</td><td class="ms">136.1 ms</td></tr>
</tbody>
<tbody class="lang-band lang-python"><tr class="lang-band-row" data-lang="python"><th colspan="6" scope="rowgroup"><span class="lang-band-dot"></span>Python · external</th></tr>
<tr class="bk-row truth-source truth-source-relaxed"><td class="bk-name"><span class="truth-mark" title="Registry parity reference (python): nirs4all in-tree — relaxed (rmse_rel ≤ 1e-03)">📐</span><code>nirs4all</code></td><td class="parity parity-divergence parity-ref-source">0</td><td class="ms">1.61 ms</td><td class="ms">9.30 ms</td><td class="ms">50.3 ms</td><td class="ms">85.8 ms</td></tr>
</tbody>
</table>
</div>

:::

::::


---

_See also_: [benchmark overview](../benchmarks/overview.md) · [methods index](index.md) · [interactive dashboard](../landing/dashboard.md)