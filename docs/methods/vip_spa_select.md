# `vip_spa_select` — VIP-seeded SPA

_Group_: **Variable selector** · _Registry tolerance_: `1.3`

## Description

VIP_SPA — VIP-mask then SPA greedy (Phase 53)

From the `pls4all.sklearn.VIPSPASelector` docstring:

> VIP_SPA — VIP-mask + SPA greedy (Phase 53).

> **Registry note** — Python `auswahl.VIP_SPA` (LSX-UniWue). Same VIP scoring and 0.3 threshold; auswahl enumerates every SPA start and picks the CV-best, pls4all uses argmax-VIP within mask. Mask metric ~0=perfect, ~1=half disagree, ~1.41=disjoint.

### Parameters

| Name | Type | Default | Notes |
|------|------|---------|-------|
| `top_k` | `int` | `None` | Number of features to retain. |
| `n_components` | `int` | `2` | Number of latent components extracted (k). |
| `vip_threshold` | `float` | `0.3` | Minimum VIP score required to enter the SPA candidate pool. |
| `seed` | `int` | `7` | registry benchmark cell value |

## Explanations

### Bibliographic source

Hybrid heuristic combining VIP ranking and the Successive Projections Algorithm. See registry notes; no single canonical paper.

### Mathematical principle

Use VIP scores to **seed** SPA's projection-orthogonal forward selection. SPA starts with the highest-VIP feature rather than the highest-coefficient one, then proceeds with the standard projection step. This biases SPA toward $y$-correlated seed features while preserving SPA's collinearity-minimising selection of subsequent features.

In practice this tends to outperform plain SPA on datasets where the first SPA seed is well-known to be noise-dominated (some real-world NIR datasets) but VIP correctly flags a different region as predictive.

### Implementation

`p4a_vip_spa_select`.

R roxygen note (`methods_extra.R::vip_spa_select`):

> VIP-SPA hybrid selector.
> @param n_components Integer. Number of latent components.
> @param vip_threshold Method-specific parameter. See the underlying `*_fit()` function for the exact semantics.
> @param top_k Method-specific parameter. See the underlying `*_fit()` function for the exact semantics.
> @param X Numeric matrix of predictors (rows = samples, cols = features).
> @param Y Numeric matrix or vector of responses, with one row per sample.
> @export

MATLAB header (`bindings/matlab/+pls4all/vip_spa_select.m`):

```text
pls4all.vip_spa_select  VIP-then-SPA hybrid (Phase 53).
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
p4a_vip_spa_select(ctx, cfg, &x_view, &y_view, /* hyperparams */, &res);
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
from pls4all._methods import vip_spa_select
with pls4all.Context() as ctx, pls4all.Config() as cfg:
    res = vip_spa_select(ctx, cfg, X, y, n_components=4)
# then: res.matrix("predictions"), res.matrix("coefficients"),
# res.vector("mask"), res.scalar("intercept"), …
```

:::

:::{tab-item} Python · pls4all.sklearn
:sync: python-sklearn
:class-label: lang-python

```python
from pls4all.sklearn import VIPSPASelector
mdl = VIPSPASelector(top_k, n_components=2, vip_threshold=0.3)
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
res <- pls4all_method("vip_spa_select", X, y,
                      n_components = 4L, params = list(vip_threshold = 0.3, top_k = 6L, seed = 7L))
# res is a named list with MethodResult arrays/scalars.
# selected_indices / top_k_intervals are 1-based.
```

:::

:::{tab-item} R · pls4all (raw fn)
:sync: r-raw
:class-label: lang-r

```r
library(pls4all)
res  <- vip_spa_select(X, Y, n_components,
                vip_threshold = 0.3, top_k = 10L)
yhat <- pls4all_predict(res, X_test)
```

:::

:::{tab-item} MATLAB · pls4all (MEX)
:sync: matlab-mex
:class-label: lang-matlab

```matlab
res = pls4all.vip_spa_select(X, y, 4);
% see header of bindings/matlab/+pls4all/vip_spa_select.m for full
% parameter surface:
%   res = vip_spa_select(X, Y, n_components, vip_threshold, top_k)
yhat = predict(res, Xtest);
```

:::

:::{tab-item} MATLAB · pls4all (classdef)
:sync: matlab-classdef
:class-label: lang-matlab

_No idiomatic classdef wrapper — invoke `pls4all.fit("vip_spa_select", X, y, …)` directly from the unified MEX factory._

:::

::::


**Registry parity references** 📐

:::{card}
:class-card: external-refs

- 📐 **`ref.python_auswahl`** (python · python) — `auswahl` 0.9.0 · qualitative (rmse_rel ≤ 1e+00) — Python `auswahl.VIP_SPA` from LSX-UniWue. Same VIP scoring and 0.3 threshold as pls4all; auswahl enumerates every candidate SPA start and picks the CV-best, pls4all takes argmax-VIP within the mask. Mask metric.
:::

### Validation contract

Structural validation contract for `vip_spa_select`, exported by the declarative validation framework (`benchmarks/validation/registry/`) from `benchmarks.parity_timing.registry`. Regenerated by `python -m benchmarks.validation.export_registry --write`.

| Field | Value |
|------|-------|
| Reference kind | `external` |
| Canonical reference role | `python` |
| Declared references | Python |
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
| `n_samples` | `80` |
| `seed` | `7` |
| `top_k` | `6` |
| `vip_threshold` | `0.3` |

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
<tr class="bk-row"><td class="bk-name"><code>pls4all.cpp.blas+omp</code></td><td class="parity parity-divergence parity-divergent">0.91</td><td class="ms ms-best">0.96 ms<span class="medal" title="fastest">🏆</span></td></tr>
</tbody>
<tbody class="lang-band lang-python"><tr class="lang-band-row" data-lang="python"><th colspan="3" scope="rowgroup"><span class="lang-band-dot"></span>Python · pls4all</th></tr>
<tr class="bk-row"><td class="bk-name"><code>pls4all.python</code></td><td class="parity parity-divergence parity-divergent">0.91</td><td class="ms">0.98 ms</td></tr>
<tr class="bk-row"><td class="bk-name"><code>pls4all.sklearn</code></td><td class="parity parity-divergence parity-divergent">0.91</td><td class="ms">1.14 ms</td></tr>
</tbody>
<tbody class="lang-band lang-r"><tr class="lang-band-row" data-lang="r"><th colspan="3" scope="rowgroup"><span class="lang-band-dot"></span>R · pls4all</th></tr>
<tr class="bk-row"><td class="bk-name"><code>pls4all.R</code></td><td class="parity parity-divergence parity-divergent">0.91</td><td class="ms">2.50 ms</td></tr>
<tr class="bk-row"><td class="bk-name"><code>pls4all.R.formula</code></td><td class="parity parity-divergence parity-divergent">0.91</td><td class="ms">4.16 ms</td></tr>
</tbody>
<tbody class="lang-band lang-matlab"><tr class="lang-band-row" data-lang="matlab"><th colspan="3" scope="rowgroup"><span class="lang-band-dot"></span>MATLAB · pls4all</th></tr>
<tr class="bk-row"><td class="bk-name"><code>pls4all.matlab</code></td><td class="parity parity-divergence parity-divergent">0.91</td><td class="ms">1.64 ms</td></tr>
<tr class="bk-row"><td class="bk-name"><code>pls4all.matlab.classdef</code></td><td class="parity parity-divergence parity-divergent">0.91</td><td class="ms">2.28 ms</td></tr>
</tbody>
<tbody class="lang-band lang-python"><tr class="lang-band-row" data-lang="python"><th colspan="3" scope="rowgroup"><span class="lang-band-dot"></span>Python · external</th></tr>
<tr class="bk-row truth-source truth-source-qualitative"><td class="bk-name"><span class="truth-mark" title="Registry parity reference (python): auswahl 0.9.0 — qualitative (rmse_rel ≤ 1e+00)">📐</span><code>ref.python_auswahl</code></td><td class="parity parity-divergence parity-ref-source">0</td><td class="ms">137.6 ms</td></tr>
</tbody>
</table>
</div>

:::

::::


---

_See also_: [benchmark overview](../benchmarks/overview.md) · [methods index](index.md) · [interactive dashboard](../landing/dashboard.md)