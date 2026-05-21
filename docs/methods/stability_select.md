# `stability_select` — MC-UVE (Monte-Carlo coefficient stability)

_Group_: **Variable selector** · _Registry tolerance_: `1.35`

## Description

Stability/MCUVE selection (§18 Phase 5c)

From the `pls4all.sklearn.StabilitySelector` docstring:

> MCUVE-style stability selector via Monte-Carlo subsampling.

> **Registry note** — R `plsVarSel::mcuve_pls` Monte-Carlo UVE stability ranking. Mask RMSE-rel ~0=perfect, ~1=half disagree, ~1.41=disjoint; tolerance widened to 1.35 to accept stochastic RNG divergence between pls4all splitmix64 and R sample().

### Parameters

| Name | Type | Default | Notes |
|------|------|---------|-------|
| `top_k` | `int` | `None` | Number of features to retain. |
| `n_components` | `int` | `2` | Number of latent components extracted (k). |
| `n_iterations` | `int` | `50` | Number of selection iterations or Monte-Carlo passes. |
| `seed` | `int` | `0` | Random seed for reproducible sampling/initialization. |
| `n_folds` | `int` | `3` | Number of cross-validation folds used inside the selector. |

## Explanations

### Bibliographic source

Cai, W., Li, Y. & Shao, X. (2008). *A variable selection method based on uninformative variable elimination for multivariate calibration of near-infrared spectra*. Chemometrics and Intelligent Laboratory Systems 90(2), 188–194.

### Mathematical principle

MC-UVE evaluates the **stability** of each feature's PLS coefficient across Monte-Carlo subsamples of the calibration set: $\mathrm{stab}_j = |\bar{b}_j| / s(b_j)$, where $\bar{b}_j$ and $s(b_j)$ are the mean and standard deviation of $b_j$ across $B$ bootstrap fits. Features with high stability ratio are reliably predictive; those with low ratio are noise-driven and discarded.

Conceptually a univariate analogue of stability selection (Meinshausen & Bühlmann 2010). The interaction with collinearity in the spectrum is benign: collinear features tend to share the contribution across bootstraps in a stable way, so their joint stability is high.

Typical Monte-Carlo budget: $B = 50$–$200$ subsamples, each at 80 % of the calibration size.

### Implementation

`p4a_stability_select`. Reference: R `plsVarSel`.

R roxygen note (`methods_extra.R::stability_select`):

> Stability selector (coefficient stability, MCUVE-style).
> @param n_components Integer. Number of latent components.
> @param top_k Method-specific parameter. See the underlying `*_fit()` function for the exact semantics.
> @param X Numeric matrix of predictors (rows = samples, cols = features).
> @param Y Numeric matrix or vector of responses, with one row per sample.
> @export

MATLAB header (`bindings/matlab/+pls4all/stability_select.m`):

```text
pls4all.stability_select  Coefficient-stability selector (MCUVE-style).
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
p4a_stability_select(ctx, cfg, &x_view, &y_view, /* hyperparams */, &res);
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
from pls4all._methods import stability_select
with pls4all.Context() as ctx, pls4all.Config() as cfg:
    res = stability_select(ctx, cfg, X, y, n_components=4)
# then: res.matrix("predictions"), res.matrix("coefficients"),
# res.vector("mask"), res.scalar("intercept"), …
```

:::

:::{tab-item} Python · pls4all.sklearn
:sync: python-sklearn
:class-label: lang-python

```python
from pls4all.sklearn import StabilitySelector
mdl = StabilitySelector(top_k, n_components=2, n_iterations=50, seed=0, n_folds=3)
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
res <- pls4all_method("stability_select", X, y,
                      n_components = 4L, params = list(top_k = 10L))
# res is a named list with MethodResult arrays/scalars.
# selected_indices / top_k_intervals are 1-based.
```

:::

:::{tab-item} R · pls4all (raw fn)
:sync: r-raw
:class-label: lang-r

```r
library(pls4all)
res  <- stability_select(X, Y, n_components, top_k = 10L)
yhat <- pls4all_predict(res, X_test)
```

:::

:::{tab-item} MATLAB · pls4all (MEX)
:sync: matlab-mex
:class-label: lang-matlab

```matlab
res = pls4all.stability_select(X, y, 4);
% see header of bindings/matlab/+pls4all/stability_select.m for full
% parameter surface:
%   res = stability_select(X, Y, n_components, top_k)
yhat = predict(res, Xtest);
```

:::

:::{tab-item} MATLAB · pls4all (classdef)
:sync: matlab-classdef
:class-label: lang-matlab

_No idiomatic classdef wrapper — invoke `pls4all.fit("stability_select", X, y, …)` directly from the unified MEX factory._

:::

::::


**Registry parity references** 📐

:::{card}
:class-card: external-refs

- 📐 **`ref.r_plsvarsel`** (R · r) — `plsVarSel` 0.10.0 · qualitative (rmse_rel ≤ 1e+00) — R `plsVarSel::mcuve_pls` Monte-Carlo UVE. Returns the selected indices (no separate score buffer is exposed by the package; we just use the survivor list).
:::

### Validation contract

Structural validation contract for `stability_select`, exported by the declarative validation framework (`benchmarks/validation/registry/`) from `benchmarks.parity_timing.registry`. Regenerated by `python -m benchmarks.validation.export_registry --write`.

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
| `n_components` | `4` |
| `n_features` | `40` |
| `n_samples` | `200` |
| `top_k` | `10` |

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
<tr class="bk-row"><td class="bk-name"><code>pls4all.cpp.blas</code></td><td class="parity parity-divergence parity-divergent">1.34</td><td class="ms">1.04 ms</td><td class="ms">8.52 ms</td><td class="ms">45.5 ms</td><td class="ms">88.8 ms</td></tr>
<tr class="bk-row"><td class="bk-name"><code>pls4all.cpp.blas+omp</code></td><td class="parity parity-divergence parity-divergent">1.34</td><td class="ms">1.10 ms</td><td class="ms ms-best">8.46 ms<span class="medal" title="fastest">🏆</span></td><td class="ms">46.1 ms</td><td class="ms">90.2 ms</td></tr>
<tr class="bk-row"><td class="bk-name"><code>pls4all.cpp.omp</code></td><td class="parity parity-divergence parity-divergent">1.34</td><td class="ms ms-best">1.04 ms<span class="medal" title="fastest">🏆</span></td><td class="ms">8.69 ms</td><td class="ms">46.1 ms</td><td class="ms">91.7 ms</td></tr>
<tr class="bk-row"><td class="bk-name"><code>pls4all.cpp.ref</code></td><td class="parity parity-divergence parity-divergent">1.34</td><td class="ms">1.06 ms</td><td class="ms">8.85 ms</td><td class="ms">46.4 ms</td><td class="ms ms-best">88.5 ms<span class="medal" title="fastest">🏆</span></td></tr>
</tbody>
<tbody class="lang-band lang-python"><tr class="lang-band-row" data-lang="python"><th colspan="6" scope="rowgroup"><span class="lang-band-dot"></span>Python · pls4all</th></tr>
<tr class="bk-row"><td class="bk-name"><code>pls4all.python</code></td><td class="parity parity-divergence parity-divergent">1.34</td><td class="ms">1.06 ms</td><td class="ms">8.51 ms</td><td class="ms">45.3 ms</td><td class="ms">90.2 ms</td></tr>
<tr class="bk-row"><td class="bk-name"><code>pls4all.sklearn</code></td><td class="parity parity-divergence parity-divergent">1.34</td><td class="ms">1.21 ms</td><td class="ms">8.70 ms</td><td class="ms ms-best">41.0 ms<span class="medal" title="fastest">🏆</span></td><td class="ms">91.1 ms</td></tr>
</tbody>
<tbody class="lang-band lang-r"><tr class="lang-band-row" data-lang="r"><th colspan="6" scope="rowgroup"><span class="lang-band-dot"></span>R · pls4all</th></tr>
<tr class="bk-row"><td class="bk-name"><code>pls4all.R</code></td><td class="parity parity-divergence parity-divergent">1.34</td><td class="ms">2.97 ms</td><td class="ms">45.3 ms</td><td class="ms">402.1 ms</td><td class="ms">552.8 ms</td></tr>
<tr class="bk-row"><td class="bk-name"><code>pls4all.R.formula</code></td><td class="parity parity-divergence parity-divergent">1.34</td><td class="ms">4.21 ms</td><td class="ms">65.4 ms</td><td class="ms">605.3 ms</td><td class="ms">742.8 ms</td></tr>
<tr class="bk-row"><td class="bk-name"><code>pls4all.R.mdatools</code></td><td class="parity parity-divergence parity-drift">—</td><td class="ms">4.42 ms</td><td class="ms">63.6 ms</td><td class="ms">597.3 ms</td><td class="ms">730.0 ms</td></tr>
<tr class="bk-row"><td class="bk-name"><code>pls4all.R.pls</code></td><td class="parity parity-divergence parity-drift">—</td><td class="ms">4.46 ms</td><td class="ms">66.4 ms</td><td class="ms">598.1 ms</td><td class="ms">649.9 ms</td></tr>
</tbody>
<tbody class="lang-band lang-matlab"><tr class="lang-band-row" data-lang="matlab"><th colspan="6" scope="rowgroup"><span class="lang-band-dot"></span>MATLAB · pls4all</th></tr>
<tr class="bk-row"><td class="bk-name"><code>pls4all.matlab</code></td><td class="parity parity-divergence parity-divergent">1.34</td><td class="ms">1.71 ms</td><td class="ms">13.5 ms</td><td class="ms">68.2 ms</td><td class="ms">135.5 ms</td></tr>
<tr class="bk-row"><td class="bk-name"><code>pls4all.matlab.classdef</code></td><td class="parity parity-divergence parity-divergent">1.34</td><td class="ms">2.06 ms</td><td class="ms">14.0 ms</td><td class="ms">69.1 ms</td><td class="ms">137.3 ms</td></tr>
</tbody>
<tbody class="lang-band lang-r"><tr class="lang-band-row" data-lang="r"><th colspan="6" scope="rowgroup"><span class="lang-band-dot"></span>R · external</th></tr>
<tr class="bk-row truth-source truth-source-qualitative"><td class="bk-name"><span class="truth-mark" title="Registry parity reference (r): plsVarSel 0.10.0 — qualitative (rmse_rel ≤ 1e+00)">📐</span><code>ref.r_plsvarsel</code></td><td class="parity parity-divergence parity-ref-source">0</td><td class="ms">83.1 ms</td><td class="ms">619.7 ms</td><td class="ms">3.8 s</td><td class="ms">62.3 s</td></tr>
</tbody>
</table>
</div>

:::

::::


---

_See also_: [benchmark overview](../benchmarks/overview.md) · [methods index](index.md) · [interactive dashboard](../landing/dashboard.md)