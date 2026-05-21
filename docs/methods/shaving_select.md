# `shaving_select` — Shaving (recursive elimination)

_Group_: **Variable selector** · _Registry tolerance_: `0.8`

## Description

Shaving iterative variable trimming

From the `pls4all.sklearn.ShavingSelector` docstring:

> Iterative SR-shaving variable elimination.

> **Registry note** — R `plsVarSel::shaving(method='SR')` iterative trimming. The pls4all trajectory is step-count based, so the parity cell uses a deeper trajectory and three components to match the compact R survivor set. Mask RMSE-rel ~0=perfect, ~1=half disagree, ~1.41=disjoint; tolerance 0.8 admits the residual trajectory-vs-package cutoff difference.

### Parameters

| Name | Type | Default | Notes |
|------|------|---------|-------|
| `n_components` | `int` | `2` | Number of latent components extracted (k). |
| `n_steps` | `int` | `10` | Number of elimination passes performed. |
| `min_features` | `int | None` | `None` | Minimum number of variables the selector is allowed to keep (defaults to `n_components`). |
| `shave_fraction` | `float` | `0.2` | Fraction of the worst-ranked variables removed at each shaving step. |
| `n_folds` | `int` | `3` | Number of cross-validation folds used inside the selector. |

## Explanations

### Bibliographic source

Mehmood, T., Liland, K. H., Snipen, L. & Sæbø, S. (2012). *A review of variable selection methods in partial least squares regression*. Chemometrics and Intelligent Laboratory Systems 118, 62–69 (§3.2 Shaving).

### Mathematical principle

Shaving recursively eliminates a fraction $\rho \in (0, 1)$ of the lowest-scoring features at each step, refits PLS, and tracks the CV-RMSE of the shrinking subset. The subset with the lowest recorded CV-RMSE across the whole shaving trajectory is returned.

Compared to backward variable elimination (BVE — see next), shaving removes a **batch** of features per step instead of one, which is faster ($O(\log p)$ steps vs $O(p)$) but more aggressive — a single bad shave removes many useful features irrecoverably. Recommended $\rho \le 0.2$ to keep shave granularity reasonable.

### Implementation

`p4a_shaving_select`.

R roxygen note (`methods_extra.R::shaving_select`):

> Shaving selector.
> @param n_components Integer. Number of latent components.
> @param n_steps Method-specific parameter. See the underlying `*_fit()` function for the exact semantics.
> @param min_features Method-specific parameter. See the underlying `*_fit()` function for the exact semantics.
> @param shave_fraction Method-specific parameter. See the underlying `*_fit()` function for the exact semantics.
> @param X Numeric matrix of predictors (rows = samples, cols = features).
> @param Y Numeric matrix or vector of responses, with one row per sample.
> @export

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
p4a_shaving_select(ctx, cfg, &x_view, &y_view, /* hyperparams */, &res);
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
from pls4all._methods import shaving_select
with pls4all.Context() as ctx, pls4all.Config() as cfg:
    res = shaving_select(ctx, cfg, X, y, n_components=3)
# then: res.matrix("predictions"), res.matrix("coefficients"),
# res.vector("mask"), res.scalar("intercept"), …
```

:::

:::{tab-item} Python · pls4all.sklearn
:sync: python-sklearn
:class-label: lang-python

```python
from pls4all.sklearn import ShavingSelector
mdl = ShavingSelector(n_components=2, n_steps=10, min_features=None, shave_fraction=0.2, n_folds=3)
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
res <- pls4all_method("shaving_select", X, y,
                      n_components = 3L, params = list(n_steps = 12L, min_features = 3L, shave_fraction = 0.2))
# res is a named list with MethodResult arrays/scalars.
# selected_indices / top_k_intervals are 1-based.
```

:::

:::{tab-item} R · pls4all (raw fn)
:sync: r-raw
:class-label: lang-r

```r
library(pls4all)
res  <- shaving_select(X, Y, n_components,
                n_steps = 10L, min_features = 5L,
                shave_fraction = 0.1)
yhat <- pls4all_predict(res, X_test)
```

:::

:::{tab-item} MATLAB · pls4all (MEX)
:sync: matlab-mex
:class-label: lang-matlab

```matlab
res = pls4all.shaving_select(X, y, 3);
% see header of bindings/matlab/+pls4all/shaving_select.m for full
% parameter surface:
%   res = shaving_select(X, Y, n_components, n_steps, min_features, shave_fraction)
yhat = predict(res, Xtest);
```

:::

:::{tab-item} MATLAB · pls4all (classdef)
:sync: matlab-classdef
:class-label: lang-matlab

_No idiomatic classdef wrapper — invoke `pls4all.fit("shaving_select", X, y, …)` directly from the unified MEX factory._

:::

::::


**Registry parity references** 📐

:::{card}
:class-card: external-refs

- 📐 **`ref.r_plsvarsel`** (R · r) — `plsVarSel` 0.10.0 · qualitative (rmse_rel ≤ 8e-01) — R `plsVarSel::shaving(method='SR')` — iterative SR-shaving of low-importance features.
:::

### Validation contract

Structural validation contract for `shaving_select`, exported by the declarative validation framework (`benchmarks/validation/registry/`) from `benchmarks.parity_timing.registry`. Regenerated by `python -m benchmarks.validation.export_registry --write`.

| Field | Value |
|------|-------|
| Reference kind | `external` |
| Canonical reference role | `r` |
| Declared references | R |
| Comparator | `reference_parity` (external-reference validity gate) |
| Comparator metric | `rmse_rel` |
| Reference tolerance | `8e-01` (qualitative) |
| Binding tolerance | `1e-06` |
| Prediction key | `mask` |
| Required data flags | none |

**Benchmark cell parameters**

| Parameter | Value |
|-----------|-------|
| `min_features` | `3` |
| `n_components` | `3` |
| `n_features` | `40` |
| `n_samples` | `200` |
| `n_steps` | `12` |
| `shave_fraction` | `0.2` |

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

**Reference gate**: non-release diagnostic — shape/smoke comparison only (`rmse_rel ≤ 8e-01`). Treat it as evidence that both implementations ran, not as numerical agreement.

Rows tagged with **📐** are the canonical parity references for this method (declared in [`parity_timing.registry`](../benchmarks/methodology.md)). C++ and external rows show reference parity; pls4all language bindings show binding parity against the C++ backend. Hover the icon for role and tolerance band.

::::{tab-set}
:class: parity-tabs

:::{tab-item} 1 thread
:sync: threads-1

<div class="parity-table-wrap">
<table class="docutils parity-grouped">
<thead><tr><th scope="col">Backend</th><th scope="col">Divergence</th><th class="size-col" scope="col">100×50 (ms)</th><th class="size-col" scope="col">100×500 (ms)</th><th class="size-col" scope="col">100×2500 (ms)</th><th class="size-col" scope="col">1000×500 (ms)</th></tr></thead>
<tbody class="lang-band lang-cpp"><tr class="lang-band-row" data-lang="cpp"><th colspan="6" scope="rowgroup"><span class="lang-band-dot"></span>C++ native · libp4a</th></tr>
<tr class="bk-row"><td class="bk-name"><code>pls4all.cpp.blas</code></td><td class="parity parity-divergence parity-divergent">3.56</td><td class="ms">1.48 ms</td><td class="ms">10.9 ms</td><td class="ms">64.7 ms</td><td class="ms">138.7 ms</td></tr>
<tr class="bk-row"><td class="bk-name"><code>pls4all.cpp.blas+omp</code></td><td class="parity parity-divergence parity-divergent">3.56</td><td class="ms">1.80 ms</td><td class="ms ms-best">10.8 ms<span class="medal" title="fastest">🏆</span></td><td class="ms">64.2 ms</td><td class="ms">143.1 ms</td></tr>
<tr class="bk-row"><td class="bk-name"><code>pls4all.cpp.omp</code></td><td class="parity parity-divergence parity-divergent">3.56</td><td class="ms">1.65 ms</td><td class="ms">11.7 ms</td><td class="ms">69.4 ms</td><td class="ms">145.8 ms</td></tr>
<tr class="bk-row"><td class="bk-name"><code>pls4all.cpp.ref</code></td><td class="parity parity-divergence parity-divergent">3.56</td><td class="ms">1.80 ms</td><td class="ms">11.9 ms</td><td class="ms">69.8 ms</td><td class="ms">144.6 ms</td></tr>
</tbody>
<tbody class="lang-band lang-python"><tr class="lang-band-row" data-lang="python"><th colspan="6" scope="rowgroup"><span class="lang-band-dot"></span>Python · pls4all</th></tr>
<tr class="bk-row"><td class="bk-name"><code>pls4all.python</code></td><td class="parity parity-divergence parity-divergent">3.56</td><td class="ms ms-best">1.46 ms<span class="medal" title="fastest">🏆</span></td><td class="ms">10.9 ms</td><td class="ms ms-best">61.1 ms<span class="medal" title="fastest">🏆</span></td><td class="ms ms-best">137.9 ms<span class="medal" title="fastest">🏆</span></td></tr>
<tr class="bk-row"><td class="bk-name"><code>pls4all.sklearn</code></td><td class="parity parity-divergence parity-divergent">3.56</td><td class="ms">1.97 ms</td><td class="ms">11.1 ms</td><td class="ms">64.0 ms</td><td class="ms">140.9 ms</td></tr>
</tbody>
<tbody class="lang-band lang-r"><tr class="lang-band-row" data-lang="r"><th colspan="6" scope="rowgroup"><span class="lang-band-dot"></span>R · pls4all</th></tr>
<tr class="bk-row"><td class="bk-name"><code>pls4all.R</code></td><td class="parity parity-divergence parity-divergent">10.30</td><td class="ms">4.43 ms</td><td class="ms">48.0 ms</td><td class="ms">419.9 ms</td><td class="ms">589.8 ms</td></tr>
<tr class="bk-row"><td class="bk-name"><code>pls4all.R.formula</code></td><td class="parity parity-divergence parity-divergent">10.30</td><td class="ms">5.63 ms</td><td class="ms">67.2 ms</td><td class="ms">617.1 ms</td><td class="ms">773.9 ms</td></tr>
<tr class="bk-row"><td class="bk-name"><code>pls4all.R.mdatools</code></td><td class="parity parity-divergence parity-divergent">10.30</td><td class="ms">5.51 ms</td><td class="ms">67.6 ms</td><td class="ms">622.9 ms</td><td class="ms">787.9 ms</td></tr>
<tr class="bk-row"><td class="bk-name"><code>pls4all.R.pls</code></td><td class="parity parity-divergence parity-divergent">10.30</td><td class="ms">6.17 ms</td><td class="ms">68.6 ms</td><td class="ms">612.7 ms</td><td class="ms">714.3 ms</td></tr>
</tbody>
<tbody class="lang-band lang-matlab"><tr class="lang-band-row" data-lang="matlab"><th colspan="6" scope="rowgroup"><span class="lang-band-dot"></span>MATLAB · pls4all</th></tr>
<tr class="bk-row"><td class="bk-name"><code>pls4all.matlab</code></td><td class="parity parity-divergence parity-divergent">3.56</td><td class="ms">2.27 ms</td><td class="ms">16.7 ms</td><td class="ms">94.5 ms</td><td class="ms">189.9 ms</td></tr>
<tr class="bk-row"><td class="bk-name"><code>pls4all.matlab.classdef</code></td><td class="parity parity-divergence parity-divergent">3.56</td><td class="ms">2.64 ms</td><td class="ms">17.2 ms</td><td class="ms">95.8 ms</td><td class="ms">191.1 ms</td></tr>
</tbody>
<tbody class="lang-band lang-r"><tr class="lang-band-row" data-lang="r"><th colspan="6" scope="rowgroup"><span class="lang-band-dot"></span>R · external</th></tr>
<tr class="bk-row truth-source truth-source-qualitative"><td class="bk-name"><span class="truth-mark" title="Registry parity reference (r): plsVarSel 0.10.0 — qualitative (rmse_rel ≤ 8e-01)">📐</span><code>ref.r_plsvarsel</code></td><td class="parity parity-divergence parity-ref-source">0</td><td class="ms">32.5 ms</td><td class="ms">134.2 ms</td><td class="ms">794.7 ms</td><td class="ms">928.1 ms</td></tr>
</tbody>
</table>
</div>

:::

::::


---

_See also_: [benchmark overview](../benchmarks/overview.md) · [methods index](index.md) · [interactive dashboard](../landing/dashboard.md)