# `ipw_select` — IPW — Iterative Predictor Weighting

_Group_: **Variable selector** · _Registry tolerance_: `1.0`

## Description

IPW-PLS iterative predictor weighting (§18 Phase 5t)

From the `pls4all.sklearn.IPWSelector` docstring:

> Iterative Predictor Weighting PLS selector.

> **Registry note** — R `plsVarSel::ipw_pls` iterative predictor weighting. pls4all uses a top-k cutoff over its iterative score path; the parity cell uses top_k=4 to match the compact R survivor set. Mask RMSE-rel ~0=perfect, ~1=half disagree, ~1.41=disjoint.

### Parameters

| Name | Type | Default | Notes |
|------|------|---------|-------|
| `top_k` | `int` | `None` | Number of features to retain. |
| `n_components` | `int` | `2` | Number of latent components extracted (k). |
| `n_iterations` | `int` | `20` | Number of selection iterations or Monte-Carlo passes. |
| `damping` | `float` | `0.5` | Exponential moving-average factor mixing previous and current weights in IPW. |
| `weight_floor` | `float` | `1e-06` | Lower bound applied to per-feature weights to prevent zero-trapping. |
| `n_folds` | `int` | `3` | Number of cross-validation folds used inside the selector. |
| `seed` | `int` | `0` | Random seed for reproducible sampling/initialization. |

## Explanations

### Bibliographic source

Forina, M., Casolino, C. & Pizarro Millán, C. (1999). *Iterative predictor weighting (IPW) PLS: a technique for the elimination of useless predictors in regression problems*. Journal of Chemometrics 13(2), 165–184. https://doi.org/10.1002/(SICI)1099-128X(199903/04)13:2<165::AID-CEM535>3.0.CO;2-Y

### Mathematical principle

IPW iteratively re-weights features in $\mathbf{X}$ by their importance, refits PLS on the re-weighted data, and tracks the score path. Weights are derived from coefficient magnitude after each fit; the iteration converges to a stable importance ranking.

Compared to single-fit coefficient ranking, IPW's iterative refinement gives more stable rankings when the calibration set is small or noisy. Exposes both the score path (for diagnostic) and the weight path (for interpretation).

### Implementation

`p4a_ipw_select`.

R roxygen note (`methods_extra.R::ipw_select`):

> IPW-PLS.
> @param n_components Integer. Number of latent components.
> @param n_iterations Integer >= 1. Number of outer-loop iterations.
> @param top_k Method-specific parameter. See the underlying `*_fit()` function for the exact semantics.
> @param damping Method-specific parameter. See the underlying `*_fit()` function for the exact semantics.
> @param weight_floor Method-specific parameter. See the underlying `*_fit()` function for the exact semantics.
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
p4a_ipw_select(ctx, cfg, &x_view, &y_view, /* hyperparams */, &res);
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
from pls4all._methods import ipw_select
with pls4all.Context() as ctx, pls4all.Config() as cfg:
    res = ipw_select(ctx, cfg, X, y, n_components=4)
# then: res.matrix("predictions"), res.matrix("coefficients"),
# res.vector("mask"), res.scalar("intercept"), …
```

:::

:::{tab-item} Python · pls4all.sklearn
:sync: python-sklearn
:class-label: lang-python

```python
from pls4all.sklearn import IPWSelector
mdl = IPWSelector(top_k, n_components=2, n_iterations=20, damping=0.5, weight_floor=1e-06, n_folds=3, seed=0)
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
res <- pls4all_method("ipw_select", X, y,
                      n_components = 4L, params = list(n_iterations = 5L, top_k = 4L, damping = 0.5, weight_floor = 0.01))
# res is a named list with MethodResult arrays/scalars.
# selected_indices / top_k_intervals are 1-based.
```

:::

:::{tab-item} R · pls4all (raw fn)
:sync: r-raw
:class-label: lang-r

```r
library(pls4all)
res  <- ipw_select(X, Y, n_components,
            n_iterations = 10L, top_k = 10L,
            damping = 0.5, weight_floor = 1e-6)
yhat <- pls4all_predict(res, X_test)
```

:::

:::{tab-item} MATLAB · pls4all (MEX)
:sync: matlab-mex
:class-label: lang-matlab

```matlab
res = pls4all.ipw_select(X, y, 4);
% see header of bindings/matlab/+pls4all/ipw_select.m for full
% parameter surface:
%   res = ipw_select(X, Y, n_components, n_iterations, top_k, damping, weight_floor)
yhat = predict(res, Xtest);
```

:::

:::{tab-item} MATLAB · pls4all (classdef)
:sync: matlab-classdef
:class-label: lang-matlab

_No idiomatic classdef wrapper — invoke `pls4all.fit("ipw_select", X, y, …)` directly from the unified MEX factory._

:::

::::


**Registry parity references** 📐

:::{card}
:class-card: external-refs

- 📐 **`ref.r_plsvarsel`** (R · r) — `plsVarSel` 0.10.0 · qualitative (rmse_rel ≤ 1e+00) — R `plsVarSel::ipw_pls` — iterative predictor weighting with the RC filter.
:::

### Validation contract

Structural validation contract for `ipw_select`, exported by the declarative validation framework (`benchmarks/validation/registry/`) from `benchmarks.parity_timing.registry`. Regenerated by `python -m benchmarks.validation.export_registry --write`.

| Field | Value |
|------|-------|
| Reference kind | `external` |
| Canonical reference role | `r` |
| Declared references | R |
| Comparator | `reference_parity` (external-reference validity gate) |
| Comparator metric | `rmse_rel` |
| Reference tolerance | `1e+00` (qualitative) |
| Binding tolerance | `1e-06` |
| Prediction key | `mask` |
| Required data flags | none |

**Benchmark cell parameters**

| Parameter | Value |
|-----------|-------|
| `damping` | `0.5` |
| `n_components` | `4` |
| `n_features` | `40` |
| `n_iterations` | `5` |
| `n_samples` | `200` |
| `top_k` | `4` |
| `weight_floor` | `0.01` |

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
<thead><tr><th scope="col">Backend</th><th scope="col">Divergence</th><th class="size-col" scope="col">100×50 (ms)</th><th class="size-col" scope="col">100×500 (ms)</th><th class="size-col" scope="col">100×2500 (ms)</th><th class="size-col" scope="col">1000×500 (ms)</th></tr></thead>
<tbody class="lang-band lang-cpp"><tr class="lang-band-row" data-lang="cpp"><th colspan="6" scope="rowgroup"><span class="lang-band-dot"></span>C++ native · libp4a</th></tr>
<tr class="bk-row"><td class="bk-name"><code>pls4all.cpp.blas</code></td><td class="parity parity-divergence parity-divergent">1.00</td><td class="ms">1.16 ms</td><td class="ms">8.39 ms</td><td class="ms ms-best">42.0 ms<span class="medal" title="fastest">🏆</span></td><td class="ms">84.2 ms</td></tr>
<tr class="bk-row"><td class="bk-name"><code>pls4all.cpp.blas+omp</code></td><td class="parity parity-divergence parity-divergent">1.00</td><td class="ms">1.20 ms</td><td class="ms">8.43 ms</td><td class="ms">42.1 ms</td><td class="ms ms-best">82.7 ms<span class="medal" title="fastest">🏆</span></td></tr>
<tr class="bk-row"><td class="bk-name"><code>pls4all.cpp.omp</code></td><td class="parity parity-divergence parity-divergent">1.00</td><td class="ms ms-best">1.14 ms<span class="medal" title="fastest">🏆</span></td><td class="ms">8.46 ms</td><td class="ms">42.3 ms</td><td class="ms">83.5 ms</td></tr>
<tr class="bk-row"><td class="bk-name"><code>pls4all.cpp.ref</code></td><td class="parity parity-divergence parity-divergent">1.00</td><td class="ms">1.18 ms</td><td class="ms">8.58 ms</td><td class="ms">42.4 ms</td><td class="ms">83.2 ms</td></tr>
</tbody>
<tbody class="lang-band lang-python"><tr class="lang-band-row" data-lang="python"><th colspan="6" scope="rowgroup"><span class="lang-band-dot"></span>Python · pls4all</th></tr>
<tr class="bk-row"><td class="bk-name"><code>pls4all.python</code></td><td class="parity parity-divergence parity-divergent">1.00</td><td class="ms">1.17 ms</td><td class="ms ms-best">8.35 ms<span class="medal" title="fastest">🏆</span></td><td class="ms">42.5 ms</td><td class="ms">83.1 ms</td></tr>
<tr class="bk-row"><td class="bk-name"><code>pls4all.sklearn</code></td><td class="parity parity-divergence parity-divergent">1.00</td><td class="ms">1.39 ms</td><td class="ms">8.67 ms</td><td class="ms">42.3 ms</td><td class="ms">83.3 ms</td></tr>
</tbody>
<tbody class="lang-band lang-r"><tr class="lang-band-row" data-lang="r"><th colspan="6" scope="rowgroup"><span class="lang-band-dot"></span>R · pls4all</th></tr>
<tr class="bk-row"><td class="bk-name"><code>pls4all.R</code></td><td class="parity parity-divergence parity-divergent">1.73</td><td class="ms">3.52 ms</td><td class="ms">46.4 ms</td><td class="ms">406.7 ms</td><td class="ms">543.5 ms</td></tr>
<tr class="bk-row"><td class="bk-name"><code>pls4all.R.formula</code></td><td class="parity parity-divergence parity-divergent">1.73</td><td class="ms">6.06 ms</td><td class="ms">62.2 ms</td><td class="ms">601.0 ms</td><td class="ms">726.6 ms</td></tr>
<tr class="bk-row"><td class="bk-name"><code>pls4all.R.mdatools</code></td><td class="parity parity-divergence parity-divergent">1.73</td><td class="ms">5.07 ms</td><td class="ms">63.4 ms</td><td class="ms">607.3 ms</td><td class="ms">743.8 ms</td></tr>
<tr class="bk-row"><td class="bk-name"><code>pls4all.R.pls</code></td><td class="parity parity-divergence parity-divergent">1.73</td><td class="ms">5.69 ms</td><td class="ms">62.5 ms</td><td class="ms">585.6 ms</td><td class="ms">680.9 ms</td></tr>
</tbody>
<tbody class="lang-band lang-matlab"><tr class="lang-band-row" data-lang="matlab"><th colspan="6" scope="rowgroup"><span class="lang-band-dot"></span>MATLAB · pls4all</th></tr>
<tr class="bk-row"><td class="bk-name"><code>pls4all.matlab</code></td><td class="parity parity-divergence parity-divergent">1.00</td><td class="ms">2.48 ms</td><td class="ms">13.3 ms</td><td class="ms">66.5 ms</td><td class="ms">133.0 ms</td></tr>
<tr class="bk-row"><td class="bk-name"><code>pls4all.matlab.classdef</code></td><td class="parity parity-divergence parity-divergent">1.00</td><td class="ms">2.49 ms</td><td class="ms">13.9 ms</td><td class="ms">67.2 ms</td><td class="ms">133.5 ms</td></tr>
</tbody>
<tbody class="lang-band lang-r"><tr class="lang-band-row" data-lang="r"><th colspan="6" scope="rowgroup"><span class="lang-band-dot"></span>R · external</th></tr>
<tr class="bk-row truth-source truth-source-qualitative"><td class="bk-name"><span class="truth-mark" title="Registry parity reference (r): plsVarSel 0.10.0 — qualitative (rmse_rel ≤ 1e+00)">📐</span><code>ref.r_plsvarsel</code></td><td class="parity parity-divergence parity-ref-source">0</td><td class="ms">18.1 ms</td><td class="ms">106.6 ms</td><td class="ms">653.8 ms</td><td class="ms">1.1 s</td></tr>
</tbody>
</table>
</div>

:::

::::


---

_See also_: [benchmark overview](../benchmarks/overview.md) · [methods index](index.md) · [interactive dashboard](../landing/dashboard.md)