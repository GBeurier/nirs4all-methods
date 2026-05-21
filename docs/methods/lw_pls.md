# `lw_pls` — Locally-Weighted PLS (LW-PLS)

_Group_: **Nonlinear / local** · _Registry tolerance_: `0.001`

## Description

LW-PLS — Locally-weighted PLS (§17 Phase 4)

From the `pls4all.sklearn.LWPLSRegression` docstring:

> Locally-weighted PLS (Næs & Centner 1998).

> **Registry note** — In-tree `nirs4all.operators.models.sklearn.lwpls.LWPLS` is the sanctioned external reference. The C++ kernel uses the same Gaussian-distance weighting with `lambda=max(1, 0.5*n_neighbors)`.

### Parameters

| Name | Type | Default | Notes |
|------|------|---------|-------|
| `n_components` | `int` | `2` | Number of latent components extracted (k). |
| `n_neighbors` | `int` | `30` | Number of training neighbours used for each local prediction (LW-PLS). |

## Explanations

### Bibliographic source

Centner, V. & Massart, D. L. (1998). *Optimisation in locally weighted regression*. Analytical Chemistry 70(19), 4206–4211.

### Mathematical principle

Instead of fitting a single global PLS, LW-PLS refits a *per-prediction-point* local PLS using only the $k$-nearest calibration samples (in $\mathbf{X}$-space distance). This adapts the model to the local geometry around each query point and is effective on calibration sets that span heterogeneous regimes (e.g. a single instrument calibrated across several product classes).

The neighbourhood weight typically combines distance (Gaussian or tricube kernel on the Euclidean / Mahalanobis distance) with the inverse residual variance from a preliminary global fit. The local PLS uses few components (typically 2–4) because the neighbourhood is small.

Prediction cost is $O(n)$ for the neighbour search plus $O(k_{\mathrm{nn}} \cdot p \cdot k_{\mathrm{pls}})$ for the local fit, per query. KD-tree / ball-tree indices accelerate the neighbour search; pls4all uses an exhaustive scan because $p \gg n$ defeats most spatial indices for NIR data anyway.

### Implementation

`p4a_lw_pls_fit`. Reference: sanctioned git-pinned port `nirs4all.operators.models.sklearn.lwpls`.

R roxygen note (`methods_extra.R::lw_pls_fit`):

> Locally-weighted PLS (Næs & Centner 1998).
> @param n_components Integer. Number of latent components.
> @param n_neighbors Method-specific parameter. See the underlying `*_fit()` function for the exact semantics.
> @param X Numeric matrix of predictors (rows = samples, cols = features).
> @param Y Numeric matrix or vector of responses, with one row per sample.
> @export

MATLAB header (`bindings/matlab/+pls4all/lw_pls.m`):

```text
pls4all.lw_pls  Locally-weighted PLS (Næs & Centner 1998).
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
p4a_lw_pls_fit(ctx, cfg, &x_view, &y_view, /* hyperparams */, &res);
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
from pls4all._methods import lw_pls_fit
with pls4all.Context() as ctx, pls4all.Config() as cfg:
    res = lw_pls_fit(ctx, cfg, X, y, n_components=3)
# then: res.matrix("predictions"), res.matrix("coefficients"),
# res.vector("mask"), res.scalar("intercept"), …
```

:::

:::{tab-item} Python · pls4all.sklearn
:sync: python-sklearn
:class-label: lang-python

```python
from pls4all.sklearn import LWPLSRegression
mdl = LWPLSRegression(n_components=2, n_neighbors=30)
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
res <- pls4all_method("lw_pls", X, y,
                      n_components = 3L, params = list(n_neighbors = 30L))
# res is a named list with MethodResult arrays/scalars.
# selected_indices / top_k_intervals are 1-based.
```

:::

:::{tab-item} R · pls4all (raw fn)
:sync: r-raw
:class-label: lang-r

```r
library(pls4all)
res  <- lw_pls_fit(X, Y, n_components, n_neighbors = 30L)
yhat <- pls4all_predict(res, X_test)
```

:::

:::{tab-item} MATLAB · pls4all (MEX)
:sync: matlab-mex
:class-label: lang-matlab

```matlab
res = pls4all.lw_pls(X, y, 3);
% see header of bindings/matlab/+pls4all/lw_pls.m for full
% parameter surface:
%   res = lw_pls(X, Y, n_components, n_neighbors)
yhat = predict(res, Xtest);
```

:::

:::{tab-item} MATLAB · pls4all (classdef)
:sync: matlab-classdef
:class-label: lang-matlab

_No idiomatic classdef wrapper — invoke `pls4all.fit("lw_pls", X, y, …)` directly from the unified MEX factory._

:::

::::


**Registry parity references** 📐

:::{card}
:class-card: external-refs

- 📐 **`nirs4all`** (python · python) — `nirs4all` in-tree · relaxed (rmse_rel ≤ 1e-03) — In-tree Python LW-PLS (sanctioned external reference). Locally-weighted PLS (Naes 1990 / Centner 1998). The kernel-bandwidth (`lambda_in_similarity`) on the nirs4all side controls neighbour weighting differently from the k-NN cut-off used by pls4all — parity is qualitative.
:::

### Validation contract

Structural validation contract for `lw_pls`, exported by the declarative validation framework (`benchmarks/validation/registry/`) from `benchmarks.parity_timing.registry`. Regenerated by `python -m benchmarks.validation.export_registry --write`.

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
| `n_components` | `3` |
| `n_features` | `40` |
| `n_neighbors` | `30` |
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

**Reference gate**: non-release diagnostic — shape/smoke comparison only (`rmse_rel ≤ 5e+00`). Treat it as evidence that both implementations ran, not as numerical agreement.

Rows tagged with **📐** are the canonical parity references for this method (declared in [`parity_timing.registry`](../benchmarks/methodology.md)). C++ and external rows show reference parity; pls4all language bindings show binding parity against the C++ backend. Hover the icon for role and tolerance band.

::::{tab-set}
:class: parity-tabs

:::{tab-item} 1 thread
:sync: threads-1

<div class="parity-table-wrap">
<table class="docutils parity-grouped">
<thead><tr><th scope="col">Backend</th><th scope="col">Divergence</th><th class="size-col" scope="col">100×50 (ms)</th><th class="size-col" scope="col">100×500 (ms)</th><th class="size-col" scope="col">100×2500 (ms)</th><th class="size-col" scope="col">1000×500 (ms)</th></tr></thead>
<tbody class="lang-band lang-cpp"><tr class="lang-band-row" data-lang="cpp"><th colspan="6" scope="rowgroup"><span class="lang-band-dot"></span>C++ native · libp4a</th></tr>
<tr class="bk-row"><td class="bk-name"><code>pls4all.cpp.blas</code></td><td class="parity parity-divergence parity-divergent">6.7e-02</td><td class="ms ms-best">2.13 ms<span class="medal" title="fastest">🏆</span></td><td class="ms ms-best">18.0 ms<span class="medal" title="fastest">🏆</span></td><td class="ms ms-best">94.0 ms<span class="medal" title="fastest">🏆</span></td><td class="ms ms-best">394.0 ms<span class="medal" title="fastest">🏆</span></td></tr>
<tr class="bk-row"><td class="bk-name"><code>pls4all.cpp.blas+omp</code></td><td class="parity parity-divergence parity-divergent">6.7e-02</td><td class="ms">2.21 ms</td><td class="ms">18.3 ms</td><td class="ms">94.7 ms</td><td class="ms">422.4 ms</td></tr>
<tr class="bk-row"><td class="bk-name"><code>pls4all.cpp.omp</code></td><td class="parity parity-divergence parity-divergent">6.7e-02</td><td class="ms">2.39 ms</td><td class="ms">21.6 ms</td><td class="ms">112.5 ms</td><td class="ms">456.1 ms</td></tr>
<tr class="bk-row"><td class="bk-name"><code>pls4all.cpp.ref</code></td><td class="parity parity-divergence parity-divergent">6.7e-02</td><td class="ms">2.49 ms</td><td class="ms">21.7 ms</td><td class="ms">115.5 ms</td><td class="ms">441.5 ms</td></tr>
</tbody>
<tbody class="lang-band lang-python"><tr class="lang-band-row" data-lang="python"><th colspan="6" scope="rowgroup"><span class="lang-band-dot"></span>Python · pls4all</th></tr>
<tr class="bk-row"><td class="bk-name"><code>pls4all.python</code></td><td class="parity parity-divergence parity-divergent">6.7e-02</td><td class="ms">2.27 ms</td><td class="ms">18.1 ms</td><td class="ms">95.9 ms</td><td class="ms">406.9 ms</td></tr>
<tr class="bk-row"><td class="bk-name"><code>pls4all.sklearn</code></td><td class="parity parity-divergence parity-divergent">6.7e-02</td><td class="ms">2.50 ms</td><td class="ms">18.8 ms</td><td class="ms">95.4 ms</td><td class="ms">418.0 ms</td></tr>
</tbody>
<tbody class="lang-band lang-r"><tr class="lang-band-row" data-lang="r"><th colspan="6" scope="rowgroup"><span class="lang-band-dot"></span>R · pls4all</th></tr>
<tr class="bk-row"><td class="bk-name"><code>pls4all.R</code></td><td class="parity parity-divergence parity-divergent">6.7e-02</td><td class="ms">5.17 ms</td><td class="ms">55.2 ms</td><td class="ms">434.1 ms</td><td class="ms">831.5 ms</td></tr>
<tr class="bk-row"><td class="bk-name"><code>pls4all.R.formula</code></td><td class="parity parity-divergence parity-divergent">6.7e-02</td><td class="ms">6.61 ms</td><td class="ms">82.5 ms</td><td class="ms">623.5 ms</td><td class="ms">853.6 ms</td></tr>
<tr class="bk-row"><td class="bk-name"><code>pls4all.R.mdatools</code></td><td class="parity parity-divergence parity-divergent">1.5e-02</td><td class="ms">5.77 ms</td><td class="ms">78.3 ms</td><td class="ms">632.8 ms</td><td class="ms">882.2 ms</td></tr>
<tr class="bk-row"><td class="bk-name"><code>pls4all.R.pls</code></td><td class="parity parity-divergence parity-divergent">1.5e-02</td><td class="ms">6.62 ms</td><td class="ms">85.6 ms</td><td class="ms">621.2 ms</td><td class="ms">907.9 ms</td></tr>
</tbody>
<tbody class="lang-band lang-matlab"><tr class="lang-band-row" data-lang="matlab"><th colspan="6" scope="rowgroup"><span class="lang-band-dot"></span>MATLAB · pls4all</th></tr>
<tr class="bk-row"><td class="bk-name"><code>pls4all.matlab</code></td><td class="parity parity-divergence parity-divergent">6.7e-02</td><td class="ms">3.35 ms</td><td class="ms">26.9 ms</td><td class="ms">136.7 ms</td><td class="ms">482.6 ms</td></tr>
<tr class="bk-row"><td class="bk-name"><code>pls4all.matlab.classdef</code></td><td class="parity parity-divergence parity-divergent">6.7e-02</td><td class="ms">3.82 ms</td><td class="ms">27.3 ms</td><td class="ms">137.3 ms</td><td class="ms">517.4 ms</td></tr>
</tbody>
<tbody class="lang-band lang-python"><tr class="lang-band-row" data-lang="python"><th colspan="6" scope="rowgroup"><span class="lang-band-dot"></span>Python · external</th></tr>
<tr class="bk-row truth-source truth-source-relaxed"><td class="bk-name"><span class="truth-mark" title="Registry parity reference (python): nirs4all in-tree — relaxed (rmse_rel ≤ 1e-03)">📐</span><code>nirs4all</code></td><td class="parity parity-divergence parity-ref-source">0</td><td class="ms">11.2 ms</td><td class="ms">47.7 ms</td><td class="ms">410.9 ms</td><td class="ms">5.3 s</td></tr>
</tbody>
</table>
</div>

:::

::::


---

_See also_: [benchmark overview](../benchmarks/overview.md) · [methods index](index.md) · [interactive dashboard](../landing/dashboard.md)