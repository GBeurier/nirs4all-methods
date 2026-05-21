# `variable_select_vip` — VIP (Variable Importance in Projection)

_Group_: **Variable selector** · _Registry tolerance_: `0.8`

## Description

VIP top-k variable selection (§18 Phase 5a, method=0)

From the `pls4all.sklearn.VIPSelector` docstring:

> Variable Importance in Projection top-k selector (Favilla 2013).

<details>
<summary>Full Python <code>sklearn</code>-wrapper docstring</summary>

```text
Variable Importance in Projection top-k selector (Favilla 2013).

Parameters
----------
top_k : int
    Number of features to keep.
n_components, solver, center_x, scale_x, tol, max_iter
    Underlying PLS hyperparameters used for VIP scoring.

Notes
-----
Exposes ``vip_scores_`` as an alias for the generic ``scores_``
attribute, for callers used to the chemometrics naming convention.
```

</details>

> **Registry note** — R `plsVarSel::VIP` top-k. pls4all's VIP scoring uses the same X-loading × y-weight formula. Mask RMSE-rel ~0=perfect overlap, ~1=half disagree, ~1.41=disjoint; tolerance 0.8 keeps the gate qualitative while still requiring substantial overlap with the R top-k.

### Parameters

| Name | Type | Default | Notes |
|------|------|---------|-------|
| `top_k` | `int` | `None` | Number of features to retain. |
| `n_components` | `int` | `2` | Number of latent components extracted (k). |
| `solver` | `str` | `'simpls'` | Inner algorithm: 'nipals', 'simpls', 'svd', 'kernel', 'orthogonal-scores', 'power', 'randomized-svd', 'wide-kernel'. |
| `center_x` | `bool` | `True` | Subtract the column mean of X before fitting. |
| `scale_x` | `bool` | `True` | Standardize X columns to unit variance before fitting. |
| `tol` | `float` | `1e-06` | Convergence tolerance for iterative solvers (NIPALS / power-iteration). |
| `max_iter` | `int` | `500` | Maximum iterations for iterative solvers. |

## Explanations

### Bibliographic source

Wold, S., Sjöström, M. & Eriksson, L. (2001). *PLS-regression: a basic tool of chemometrics*. Chemometrics and Intelligent Laboratory Systems 58(2), 109–130.

### Mathematical principle

VIP scores quantify each feature's contribution across all $k$ latent components of a PLS model, weighted by how much each component explains of $\mathbf{y}$: $\mathrm{VIP}_j = \sqrt{\frac{p}{\mathrm{SSY}} \sum_{a=1}^{k} w_{ja}^2 \, \mathrm{SSY}_a}$, where $w_{ja}$ is the loading weight of feature $j$ in component $a$ and $\mathrm{SSY}_a$ is the explained sum of squares of $\mathbf{y}$ in component $a$.

The normalisation guarantees $\sum_j \mathrm{VIP}_j^2 = p$, so the heuristic $\mathrm{VIP}_j > 1$ identifies features contributing more than their fair share. VIP is the workhorse of spectroscopic variable selection — simple, deterministic, fast, and well understood.

### Implementation

`p4a_variable_select_rank` with metric=VIP. Reference: R `plsVarSel 0.10.0`.

R roxygen note (`selectors.R::vip_select`):

> VIP (Variable Importance in Projection) ranker.

MATLAB header (`bindings/matlab/+pls4all/vip_select.m`):

```text
pls4all.vip_select  VIP-based feature ranking.

   res = pls4all.vip_select(X, Y, n_components, top_k)

 Fits an internal SIMPLS model (store_scores=1) and ranks features by
 their Variable Importance in Projection (VIP) scores.
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
p4a_variable_select_rank(ctx, cfg, &x_view, &y_view, /* hyperparams */, &res);
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
from pls4all._methods import variable_select_rank
with pls4all.Context() as ctx, pls4all.Config() as cfg:
    res = variable_select_rank(ctx, cfg, X, y, n_components=4)
# then: res.matrix("predictions"), res.matrix("coefficients"),
# res.vector("mask"), res.scalar("intercept"), …
```

:::

:::{tab-item} Python · pls4all.sklearn
:sync: python-sklearn
:class-label: lang-python

```python
from pls4all.sklearn import VIPSelector
mdl = VIPSelector(top_k, n_components=2, solver='simpls', center_x=True, scale_x=True, tol=1e-06, max_iter=500)
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
res <- pls4all_method("variable_select_vip", X, y,
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
res  <- vip_select(model, X, top_k)
yhat <- pls4all_predict(res, X_test)
```

:::

:::{tab-item} MATLAB · pls4all (MEX)
:sync: matlab-mex
:class-label: lang-matlab

```matlab
res = pls4all.vip_select(X, y, 4);
% see header of bindings/matlab/+pls4all/vip_select.m for full
% parameter surface:
%   res = vip_select(X, Y, n_components, top_k)
yhat = predict(res, Xtest);
```

:::

:::{tab-item} MATLAB · pls4all (classdef)
:sync: matlab-classdef
:class-label: lang-matlab

_No idiomatic classdef wrapper — invoke `pls4all.fit("variable_select_vip", X, y, …)` directly from the unified MEX factory._

:::

::::


**Registry parity references** 📐

:::{card}
:class-card: external-refs

- 📐 **`ref.r_plsvarsel`** (R · r) — `plsVarSel` 0.10.0 · qualitative (rmse_rel ≤ 8e-01) — R `plsVarSel::VIP` ranking on a fitted `pls::plsr` model. We take the top-k indices by VIP score.
:::

### Validation contract

Structural validation contract for `variable_select_vip`, exported by the declarative validation framework (`benchmarks/validation/registry/`) from `benchmarks.parity_timing.registry`. Regenerated by `python -m benchmarks.validation.export_registry --write`.

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
<tr class="bk-row"><td class="bk-name"><code>pls4all.cpp.blas</code></td><td class="parity parity-divergence parity-not_available">—</td><td class="ms">1.06 ms</td><td class="ms">8.40 ms</td><td class="ms ms-best">41.7 ms<span class="medal" title="fastest">🏆</span></td><td class="ms">81.4 ms</td></tr>
<tr class="bk-row"><td class="bk-name"><code>pls4all.cpp.blas+omp</code></td><td class="parity parity-divergence parity-divergent">0.77</td><td class="ms">1.11 ms</td><td class="ms">8.34 ms</td><td class="ms">41.7 ms</td><td class="ms">81.5 ms</td></tr>
<tr class="bk-row"><td class="bk-name"><code>pls4all.cpp.omp</code></td><td class="parity parity-divergence parity-not_available">—</td><td class="ms ms-best">1.04 ms<span class="medal" title="fastest">🏆</span></td><td class="ms">8.39 ms</td><td class="ms">42.4 ms</td><td class="ms">83.0 ms</td></tr>
<tr class="bk-row"><td class="bk-name"><code>pls4all.cpp.ref</code></td><td class="parity parity-divergence parity-not_available">—</td><td class="ms">1.10 ms</td><td class="ms">8.42 ms</td><td class="ms">42.2 ms</td><td class="ms">82.2 ms</td></tr>
</tbody>
<tbody class="lang-band lang-python"><tr class="lang-band-row" data-lang="python"><th colspan="6" scope="rowgroup"><span class="lang-band-dot"></span>Python · pls4all</th></tr>
<tr class="bk-row"><td class="bk-name"><code>pls4all.python</code></td><td class="parity parity-divergence parity-divergent">0.77</td><td class="ms">1.17 ms</td><td class="ms ms-best">8.23 ms<span class="medal" title="fastest">🏆</span></td><td class="ms">41.7 ms</td><td class="ms ms-best">80.0 ms<span class="medal" title="fastest">🏆</span></td></tr>
<tr class="bk-row"><td class="bk-name"><code>pls4all.sklearn</code></td><td class="parity parity-divergence parity-divergent">0.77</td><td class="ms">1.29 ms</td><td class="ms">8.66 ms</td><td class="ms">42.2 ms</td><td class="ms">80.4 ms</td></tr>
</tbody>
<tbody class="lang-band lang-r"><tr class="lang-band-row" data-lang="r"><th colspan="6" scope="rowgroup"><span class="lang-band-dot"></span>R · pls4all</th></tr>
<tr class="bk-row"><td class="bk-name"><code>pls4all.R</code></td><td class="parity parity-divergence parity-divergent">0.77</td><td class="ms">3.38 ms</td><td class="ms">41.4 ms</td><td class="ms">476.8 ms</td><td class="ms">526.1 ms</td></tr>
<tr class="bk-row"><td class="bk-name"><code>pls4all.R.formula</code></td><td class="parity parity-divergence parity-divergent">0.77</td><td class="ms">5.12 ms</td><td class="ms">74.1 ms</td><td class="ms">627.9 ms</td><td class="ms">493.6 ms</td></tr>
<tr class="bk-row"><td class="bk-name"><code>pls4all.R.mdatools</code></td><td class="parity parity-divergence parity-drift">—</td><td class="ms">5.31 ms</td><td class="ms">64.9 ms</td><td class="ms">613.9 ms</td><td class="ms">498.0 ms</td></tr>
<tr class="bk-row"><td class="bk-name"><code>pls4all.R.pls</code></td><td class="parity parity-divergence parity-drift">—</td><td class="ms">5.06 ms</td><td class="ms">63.2 ms</td><td class="ms">597.7 ms</td><td class="ms">481.0 ms</td></tr>
</tbody>
<tbody class="lang-band lang-matlab"><tr class="lang-band-row" data-lang="matlab"><th colspan="6" scope="rowgroup"><span class="lang-band-dot"></span>MATLAB · pls4all</th></tr>
<tr class="bk-row"><td class="bk-name"><code>pls4all.matlab</code></td><td class="parity parity-divergence parity-divergent">0.77</td><td class="ms">1.81 ms</td><td class="ms">13.1 ms</td><td class="ms">65.3 ms</td><td class="ms">130.5 ms</td></tr>
<tr class="bk-row"><td class="bk-name"><code>pls4all.matlab.classdef</code></td><td class="parity parity-divergence parity-divergent">0.77</td><td class="ms">2.32 ms</td><td class="ms">14.0 ms</td><td class="ms">66.9 ms</td><td class="ms">130.6 ms</td></tr>
</tbody>
<tbody class="lang-band lang-r"><tr class="lang-band-row" data-lang="r"><th colspan="6" scope="rowgroup"><span class="lang-band-dot"></span>R · external</th></tr>
<tr class="bk-row truth-source truth-source-qualitative"><td class="bk-name"><span class="truth-mark" title="Registry parity reference (r): plsVarSel 0.10.0 — qualitative (rmse_rel ≤ 8e-01)">📐</span><code>ref.r_plsvarsel</code></td><td class="parity parity-divergence parity-ref-source">0</td><td class="ms">11.6 ms</td><td class="ms">68.2 ms</td><td class="ms">609.5 ms</td><td class="ms">620.8 ms</td></tr>
</tbody>
</table>
</div>

:::

::::


---

_See also_: [benchmark overview](../benchmarks/overview.md) · [methods index](index.md) · [interactive dashboard](../landing/dashboard.md)