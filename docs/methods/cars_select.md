# `cars_select` — CARS — Competitive Adaptive Reweighted Sampling

_Group_: **Variable selector** · _Registry tolerance_: `1.4`

## Description

CARS competitive adaptive reweighted sampling

From the `pls4all.sklearn.CARSSelector` docstring:

> Competitive Adaptive Reweighted Sampling (Li 2009).

> **Registry note** — R `enpls::enpls.fs(method='mc')` is the closest installable analog of CARS — different RNG and sampling policy. Mask RMSE-rel ~0=perfect, ~1=half disagree, ~1.41=disjoint; tolerance admits the known Monte-Carlo sampling-policy divergence.

### Parameters

| Name | Type | Default | Notes |
|------|------|---------|-------|
| `n_components` | `int` | `2` | Number of latent components extracted (k). |
| `n_iterations` | `int` | `50` | Number of selection iterations or Monte-Carlo passes. |
| `min_features` | `int | None` | `None` | Minimum number of variables the selector is allowed to keep (defaults to `n_components`). |
| `n_folds` | `int` | `3` | Number of cross-validation folds used inside the selector. |
| `seed` | `int` | `0` | Random seed for reproducible sampling/initialization. |

## Explanations

### Bibliographic source

Li, H., Liang, Y., Xu, Q. & Cao, D. (2009). *Key wavelengths screening using competitive adaptive reweighted sampling method for multivariate calibration*. Analytica Chimica Acta 648(1), 77–84.

### Mathematical principle

CARS is one of the most widely-used spectroscopic variable selectors. It runs $M$ iterations of: (1) draw a Monte-Carlo subsample, (2) fit PLS, (3) compute coefficient weights $w_j = |b_j| / \sum |b_j|$, (4) keep a shrinking fraction of features ranked by weighted competitive sampling — features compete stochastically with probability proportional to $w_j$.

The retention fraction shrinks **exponentially**: $r_m = \exp(-\mu(m - 1))$ with $\mu$ chosen so that two features survive at the final iteration. The iteration whose surviving subset minimises CV-RMSE is returned.

CARS combines deterministic exponential decay with stochastic competition; the latter prevents premature elimination of correlated features. Practically very robust to noise.

### Implementation

`p4a_cars_select`. Reference: R `enpls 6.1.1` (`enpls.fs(method='mc')` is the closest analogue).

R roxygen note (`selectors.R::cars_select`):

> CARS — Competitive Adaptive Reweighted Sampling.

MATLAB header (`bindings/matlab/+pls4all/cars_select.m`):

```text
pls4all.cars_select  Competitive Adaptive Reweighted Sampling.

   res = pls4all.cars_select(X, Y, K, n_iter, min_feats)

 Uses the default (NULL) ValidationPlan on the C side (5-fold fallback).
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
p4a_cars_select(ctx, cfg, &x_view, &y_view, /* hyperparams */, &res);
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
from pls4all._methods import cars_select
with pls4all.Context() as ctx, pls4all.Config() as cfg:
    res = cars_select(ctx, cfg, X, y, n_components=4)
# then: res.matrix("predictions"), res.matrix("coefficients"),
# res.vector("mask"), res.scalar("intercept"), …
```

:::

:::{tab-item} Python · pls4all.sklearn
:sync: python-sklearn
:class-label: lang-python

```python
from pls4all.sklearn import CARSSelector
mdl = CARSSelector(n_components=2, n_iterations=50, min_features=None, n_folds=3, seed=0)
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
res <- pls4all_method("cars_select", X, y,
                      n_components = 4L, params = list(n_iterations = 8L, min_features = 5L))
# res is a named list with MethodResult arrays/scalars.
# selected_indices / top_k_intervals are 1-based.
```

:::

:::{tab-item} R · pls4all (raw fn)
:sync: r-raw
:class-label: lang-r

```r
library(pls4all)
res  <- cars_select(X, Y, n_components, n_iterations = 50L,
             min_features = 5L)
yhat <- pls4all_predict(res, X_test)
```

:::

:::{tab-item} MATLAB · pls4all (MEX)
:sync: matlab-mex
:class-label: lang-matlab

```matlab
res = pls4all.cars_select(X, y, 4);
% see header of bindings/matlab/+pls4all/cars_select.m for full
% parameter surface:
%   res = cars_select(X, Y, n_components, n_iterations, min_features)
yhat = predict(res, Xtest);
```

:::

:::{tab-item} MATLAB · pls4all (classdef)
:sync: matlab-classdef
:class-label: lang-matlab

_No idiomatic classdef wrapper — invoke `pls4all.fit("cars_select", X, y, …)` directly from the unified MEX factory._

:::

::::


**Registry parity references** 📐

:::{card}
:class-card: external-refs

- 📐 **`ref.r_enpls`** (R · r) — `enpls` 6.1 · qualitative (rmse_rel ≤ 1e+00) — R `enpls::enpls.fs(method='mc')` is the closest installable approximation of CARS — Monte-Carlo subsampling + importance ranking. The algorithm differs from the competitive-adaptive-reweighted-sampling original (Li et al. 2009), so set overlap is qualitative.
:::

### Validation contract

Structural validation contract for `cars_select`, exported by the declarative validation framework (`benchmarks/validation/registry/`) from `benchmarks.parity_timing.registry`. Regenerated by `python -m benchmarks.validation.export_registry --write`.

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
| `min_features` | `5` |
| `n_components` | `4` |
| `n_features` | `40` |
| `n_iterations` | `8` |
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
<tr class="bk-row"><td class="bk-name"><code>pls4all.cpp.blas</code></td><td class="parity parity-divergence parity-divergent">0.82</td><td class="ms ms-best">1.42 ms<span class="medal" title="fastest">🏆</span></td><td class="ms ms-best">9.73 ms<span class="medal" title="fastest">🏆</span></td><td class="ms">53.0 ms</td><td class="ms">117.9 ms</td></tr>
<tr class="bk-row"><td class="bk-name"><code>pls4all.cpp.blas+omp</code></td><td class="parity parity-divergence parity-divergent">0.82</td><td class="ms">1.49 ms</td><td class="ms">9.81 ms</td><td class="ms">52.0 ms</td><td class="ms">116.6 ms</td></tr>
<tr class="bk-row"><td class="bk-name"><code>pls4all.cpp.omp</code></td><td class="parity parity-divergence parity-divergent">0.82</td><td class="ms">1.53 ms</td><td class="ms">10.5 ms</td><td class="ms">55.0 ms</td><td class="ms">123.3 ms</td></tr>
<tr class="bk-row"><td class="bk-name"><code>pls4all.cpp.ref</code></td><td class="parity parity-divergence parity-divergent">0.82</td><td class="ms">1.57 ms</td><td class="ms">10.5 ms</td><td class="ms">55.2 ms</td><td class="ms">121.2 ms</td></tr>
</tbody>
<tbody class="lang-band lang-python"><tr class="lang-band-row" data-lang="python"><th colspan="6" scope="rowgroup"><span class="lang-band-dot"></span>Python · pls4all</th></tr>
<tr class="bk-row"><td class="bk-name"><code>pls4all.python</code></td><td class="parity parity-divergence parity-divergent">0.82</td><td class="ms">1.55 ms</td><td class="ms">9.89 ms</td><td class="ms ms-best">51.9 ms<span class="medal" title="fastest">🏆</span></td><td class="ms ms-best">116.0 ms<span class="medal" title="fastest">🏆</span></td></tr>
<tr class="bk-row"><td class="bk-name"><code>pls4all.sklearn</code></td><td class="parity parity-divergence parity-divergent">0.82</td><td class="ms">1.77 ms</td><td class="ms">10.1 ms</td><td class="ms">55.9 ms</td><td class="ms">116.6 ms</td></tr>
</tbody>
<tbody class="lang-band lang-r"><tr class="lang-band-row" data-lang="r"><th colspan="6" scope="rowgroup"><span class="lang-band-dot"></span>R · pls4all</th></tr>
<tr class="bk-row"><td class="bk-name"><code>pls4all.R</code></td><td class="parity parity-divergence parity-divergent">0.82</td><td class="ms">3.96 ms</td><td class="ms">44.5 ms</td><td class="ms">413.6 ms</td><td class="ms">585.0 ms</td></tr>
<tr class="bk-row"><td class="bk-name"><code>pls4all.R.formula</code></td><td class="parity parity-divergence parity-divergent">0.82</td><td class="ms">5.85 ms</td><td class="ms">61.4 ms</td><td class="ms">642.6 ms</td><td class="ms">761.8 ms</td></tr>
<tr class="bk-row"><td class="bk-name"><code>pls4all.R.mdatools</code></td><td class="parity parity-divergence parity-not_available">—</td><td class="ms">5.41 ms</td><td class="ms">63.0 ms</td><td class="ms">604.0 ms</td><td class="ms">737.5 ms</td></tr>
<tr class="bk-row"><td class="bk-name"><code>pls4all.R.pls</code></td><td class="parity parity-divergence parity-not_available">—</td><td class="ms">5.75 ms</td><td class="ms">62.5 ms</td><td class="ms">596.0 ms</td><td class="ms">664.4 ms</td></tr>
</tbody>
<tbody class="lang-band lang-matlab"><tr class="lang-band-row" data-lang="matlab"><th colspan="6" scope="rowgroup"><span class="lang-band-dot"></span>MATLAB · pls4all</th></tr>
<tr class="bk-row"><td class="bk-name"><code>pls4all.matlab</code></td><td class="parity parity-divergence parity-divergent">0.82</td><td class="ms">2.25 ms</td><td class="ms">15.4 ms</td><td class="ms">80.2 ms</td><td class="ms">170.2 ms</td></tr>
<tr class="bk-row"><td class="bk-name"><code>pls4all.matlab.classdef</code></td><td class="parity parity-divergence parity-divergent">0.82</td><td class="ms">2.78 ms</td><td class="ms">15.9 ms</td><td class="ms">82.6 ms</td><td class="ms">171.2 ms</td></tr>
</tbody>
<tbody class="lang-band lang-r"><tr class="lang-band-row" data-lang="r"><th colspan="6" scope="rowgroup"><span class="lang-band-dot"></span>R · external</th></tr>
<tr class="bk-row truth-source truth-source-qualitative"><td class="bk-name"><span class="truth-mark" title="Registry parity reference (r): enpls 6.1 — qualitative (rmse_rel ≤ 1e+00)">📐</span><code>ref.r_enpls</code></td><td class="parity parity-divergence parity-ref-source">0</td><td class="ms">75.0 ms</td><td class="ms">542.5 ms</td><td class="ms">5.9 s</td><td class="ms">1.9 s</td></tr>
</tbody>
</table>
</div>

:::

::::


---

_See also_: [benchmark overview](../benchmarks/overview.md) · [methods index](index.md) · [interactive dashboard](../landing/dashboard.md)