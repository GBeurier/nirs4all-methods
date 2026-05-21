# `bipls_select` — biPLS — Backward Interval PLS

_Group_: **Variable selector** · _Registry tolerance_: `0.7`

## Description

biPLS backward interval elimination (§18 Phase 5p)

From the `pls4all.sklearn.BiPLSSelector` docstring:

> biPLS — backward interval elimination (Nørgaard 2000).

> **Registry note** — R `mdatools::ipls(method='backward')`. Mask RMSE-rel ~0=perfect, ~1=half disagree, ~1.41=disjoint; tolerance 0.7 enforces ~50% overlap. Backward elimination is order-sensitive.

### Parameters

| Name | Type | Default | Notes |
|------|------|---------|-------|
| `n_components` | `int` | `2` | Number of latent components extracted (k). |
| `interval_width` | `int` | `10` | Width (in variables) of each contiguous spectral interval. |
| `min_intervals` | `int` | `2` | Minimum number of intervals retained by biPLS backward elimination. |
| `n_folds` | `int` | `3` | Number of cross-validation folds used inside the selector. |

## Explanations

### Bibliographic source

Leardi, R. & Nørgaard, L. (2004). *Sequential application of backward interval partial least squares and genetic algorithms for the selection of relevant spectral regions*. Journal of Chemometrics 18(11), 486–497.

### Mathematical principle

Start with the spectrum partitioned into $I$ equal intervals (typically 10–40). At each iteration, remove the interval whose removal **least** hurts CV-RMSE — i.e. the least informative interval. Iterate until removing any further interval materially worsens performance.

Returns a multi-band subset with each band aligned to the original equal-partition grid. The discrete structure makes biPLS robust to noise (no fine-grained fishing) and easy to interpret (each retained interval is a spectroscopic region of contiguous wavelengths).

Commonly chained with GA-PLS as a coarse-to-fine pipeline (Leardi & Nørgaard 2004): biPLS narrows the candidate intervals, GA-PLS does the within-interval feature selection.

### Implementation

`p4a_bipls_select`. Reference: R `plsVarSel`.

R roxygen note (`methods_extra.R::bipls_select`):

> biPLS — backward interval PLS.
> @param n_components Integer. Number of latent components.
> @param interval_width Method-specific parameter. See the underlying `*_fit()` function for the exact semantics.
> @param min_intervals Method-specific parameter. See the underlying `*_fit()` function for the exact semantics.
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
p4a_bipls_select(ctx, cfg, &x_view, &y_view, /* hyperparams */, &res);
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
from pls4all._methods import bipls_select
with pls4all.Context() as ctx, pls4all.Config() as cfg:
    res = bipls_select(ctx, cfg, X, y, n_components=4)
# then: res.matrix("predictions"), res.matrix("coefficients"),
# res.vector("mask"), res.scalar("intercept"), …
```

:::

:::{tab-item} Python · pls4all.sklearn
:sync: python-sklearn
:class-label: lang-python

```python
from pls4all.sklearn import BiPLSSelector
mdl = BiPLSSelector(n_components=2, interval_width=10, min_intervals=2, n_folds=3)
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
res <- pls4all_method("bipls_select", X, y,
                      n_components = 4L, params = list(interval_width = 5L, min_intervals = 2L))
# res is a named list with MethodResult arrays/scalars.
# selected_indices / top_k_intervals are 1-based.
```

:::

:::{tab-item} R · pls4all (raw fn)
:sync: r-raw
:class-label: lang-r

```r
library(pls4all)
res  <- bipls_select(X, Y, n_components,
              interval_width = 10L, min_intervals = 1L)
yhat <- pls4all_predict(res, X_test)
```

:::

:::{tab-item} MATLAB · pls4all (MEX)
:sync: matlab-mex
:class-label: lang-matlab

```matlab
res = pls4all.bipls_select(X, y, 4);
% see header of bindings/matlab/+pls4all/bipls_select.m for full
% parameter surface:
%   res = bipls_select(X, Y, n_components, interval_width, min_intervals)
yhat = predict(res, Xtest);
```

:::

:::{tab-item} MATLAB · pls4all (classdef)
:sync: matlab-classdef
:class-label: lang-matlab

_No idiomatic classdef wrapper — invoke `pls4all.fit("bipls_select", X, y, …)` directly from the unified MEX factory._

:::

::::


**Registry parity references** 📐

:::{card}
:class-card: external-refs

- 📐 **`ref.r_mdatools`** (R · r) — `mdatools` 0.15.0 · qualitative (rmse_rel ≤ 7e-01) — R `mdatools::ipls(method='backward')` — biPLS elimination. Returns variables from intervals that survive the backward sweep.
:::

### Validation contract

Structural validation contract for `bipls_select`, exported by the declarative validation framework (`benchmarks/validation/registry/`) from `benchmarks.parity_timing.registry`. Regenerated by `python -m benchmarks.validation.export_registry --write`.

| Field | Value |
|------|-------|
| Reference kind | `external` |
| Canonical reference role | `r` |
| Declared references | R |
| Comparator | `reference_parity` (external-reference validity gate) |
| Comparator metric | `rmse_rel` |
| Reference tolerance | `7e-01` (qualitative) |
| Binding tolerance | `1e-06` |
| Prediction key | `mask` |
| Required data flags | none |

**Benchmark cell parameters**

| Parameter | Value |
|-----------|-------|
| `interval_width` | `5` |
| `min_intervals` | `2` |
| `n_components` | `4` |
| `n_features` | `40` |
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

**Reference gate**: non-release diagnostic — shape/smoke comparison only (`rmse_rel ≤ 7e-01`). Treat it as evidence that both implementations ran, not as numerical agreement.

Rows tagged with **📐** are the canonical parity references for this method (declared in [`parity_timing.registry`](../benchmarks/methodology.md)). C++ and external rows show reference parity; pls4all language bindings show binding parity against the C++ backend. Hover the icon for role and tolerance band.

::::{tab-set}
:class: parity-tabs

:::{tab-item} 1 thread
:sync: threads-1

<div class="parity-table-wrap">
<table class="docutils parity-grouped">
<thead><tr><th scope="col">Backend</th><th scope="col">Divergence</th><th class="size-col" scope="col">100×50 (ms)</th><th class="size-col" scope="col">100×500 (ms)</th><th class="size-col" scope="col">100×2500 (ms)</th></tr></thead>
<tbody class="lang-band lang-cpp"><tr class="lang-band-row" data-lang="cpp"><th colspan="5" scope="rowgroup"><span class="lang-band-dot"></span>C++ native · libp4a</th></tr>
<tr class="bk-row"><td class="bk-name"><code>pls4all.cpp.blas</code></td><td class="parity parity-divergence parity-not_available">—</td><td class="ms ms-best">2.94 ms<span class="medal" title="fastest">🏆</span></td><td class="ms ms-best">1.3 s<span class="medal" title="fastest">🏆</span></td><td class="ms">341.5 s</td></tr>
<tr class="bk-row"><td class="bk-name"><code>pls4all.cpp.blas+omp</code></td><td class="parity parity-divergence parity-ref-qualitative">0</td><td class="ms">3.18 ms</td><td class="ms">1.4 s</td><td class="ms">345.7 s</td></tr>
<tr class="bk-row"><td class="bk-name"><code>pls4all.cpp.omp</code></td><td class="parity parity-divergence parity-not_available">—</td><td class="ms">3.42 ms</td><td class="ms">2.0 s</td><td class="ms">441.5 s</td></tr>
<tr class="bk-row"><td class="bk-name"><code>pls4all.cpp.ref</code></td><td class="parity parity-divergence parity-not_available">—</td><td class="ms">3.67 ms</td><td class="ms">2.1 s</td><td class="ms">444.8 s</td></tr>
</tbody>
<tbody class="lang-band lang-python"><tr class="lang-band-row" data-lang="python"><th colspan="5" scope="rowgroup"><span class="lang-band-dot"></span>Python · pls4all</th></tr>
<tr class="bk-row"><td class="bk-name"><code>pls4all.python</code></td><td class="parity parity-divergence parity-drift">0</td><td class="ms">3.21 ms</td><td class="ms">1.4 s</td><td class="ms">341.1 s</td></tr>
<tr class="bk-row"><td class="bk-name"><code>pls4all.sklearn</code></td><td class="parity parity-divergence parity-drift">0</td><td class="ms">3.29 ms</td><td class="ms">1.4 s</td><td class="ms ms-best">225.6 s<span class="medal" title="fastest">🏆</span></td></tr>
</tbody>
<tbody class="lang-band lang-r"><tr class="lang-band-row" data-lang="r"><th colspan="5" scope="rowgroup"><span class="lang-band-dot"></span>R · pls4all</th></tr>
<tr class="bk-row"><td class="bk-name"><code>pls4all.R</code></td><td class="parity parity-divergence parity-drift">0</td><td class="ms">5.96 ms</td><td class="ms">2.2 s</td><td class="ms">425.7 s</td></tr>
<tr class="bk-row"><td class="bk-name"><code>pls4all.R.formula</code></td><td class="parity parity-divergence parity-drift">0</td><td class="ms">7.64 ms</td><td class="ms">2.2 s</td><td class="ms">359.5 s</td></tr>
<tr class="bk-row"><td class="bk-name"><code>pls4all.R.mdatools</code></td><td class="parity parity-divergence parity-drift">—</td><td class="ms">7.41 ms</td><td class="ms">2.2 s</td><td class="ms">361.1 s</td></tr>
<tr class="bk-row"><td class="bk-name"><code>pls4all.R.pls</code></td><td class="parity parity-divergence parity-drift">—</td><td class="ms">7.52 ms</td><td class="ms">2.2 s</td><td class="ms">418.8 s</td></tr>
</tbody>
<tbody class="lang-band lang-matlab"><tr class="lang-band-row" data-lang="matlab"><th colspan="5" scope="rowgroup"><span class="lang-band-dot"></span>MATLAB · pls4all</th></tr>
<tr class="bk-row"><td class="bk-name"><code>pls4all.matlab</code></td><td class="parity parity-divergence parity-drift">0</td><td class="ms">4.36 ms</td><td class="ms">2.1 s</td><td class="ms">386.0 s</td></tr>
<tr class="bk-row"><td class="bk-name"><code>pls4all.matlab.classdef</code></td><td class="parity parity-divergence parity-drift">0</td><td class="ms">4.93 ms</td><td class="ms">2.1 s</td><td class="ms">441.6 s</td></tr>
</tbody>
<tbody class="lang-band lang-r"><tr class="lang-band-row" data-lang="r"><th colspan="5" scope="rowgroup"><span class="lang-band-dot"></span>R · external</th></tr>
<tr class="bk-row truth-source truth-source-qualitative"><td class="bk-name"><span class="truth-mark" title="Registry parity reference (r): mdatools 0.15.0 — qualitative (rmse_rel ≤ 7e-01)">📐</span><code>ref.r_mdatools</code></td><td class="parity parity-divergence parity-ref-source">0</td><td class="ms">415.7 ms</td><td class="ms">115.4 s</td><td class="ms ms-empty">—</td></tr>
</tbody>
</table>
</div>

:::

::::


---

_See also_: [benchmark overview](../benchmarks/overview.md) · [methods index](index.md) · [interactive dashboard](../landing/dashboard.md)