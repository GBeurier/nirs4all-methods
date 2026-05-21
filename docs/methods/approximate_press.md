# `approximate_press` — Approximate PRESS (leave-one-out by hat-matrix)

_Group_: **Diagnostic** · _Registry tolerance_: `10.0`

## Description

Approximate-PRESS component selection (§29)

> **Registry note** — R `pls::plsr(validation='LOO')$validation$PRESS` returns true LOO-PRESS; pls4all's approximate_press uses Eastment-Krzanowski leverage-inflated approximation. Same ordering, different scaling — widened tol flags external-ref presence.

### Parameters

| Name | Type | Default | Notes |
|------|------|---------|-------|
| `max_components` | `int` | `6` | registry benchmark cell value |

## Explanations

### Bibliographic source

Allen, D. M. (1974). *The relationship between variable selection and data augmentation and a method for prediction*. Technometrics 16(1), 125–127.

### Mathematical principle

Exact leave-one-out cross-validation requires $n$ refits and costs $O(n^2 p k)$. The approximate PRESS uses the hat-matrix shortcut: the leave-one-out residual for sample $i$ is approximately $r_i^{(-i)} \approx r_i / (1 - h_{ii})$, where $h_{ii}$ is the diagonal of the hat matrix $\mathbf{H} = \mathbf{T}(\mathbf{T}^{\top}\mathbf{T})^{-1}\mathbf{T}^{\top}$. Total PRESS is $\sum_i (r_i^{(-i)})^2$, computed in $O(n p k)$ from a single fit.

The approximation is exact for OLS and tight for PLS as long as no single $h_{ii}$ approaches 1 (a high-leverage outlier). Use exact LOO when the approximate PRESS diverges from the cross-validated RMSEP by more than a few percent.

Drives the one-SE rule for selecting the component count $k$.

### Implementation

`p4a_approximate_press_compute`. Returns a length-$(k_{\max}+1)$ vector indexed by component count.

R roxygen note (`diagnostics.R::approximate_press`):

> Approximate-PRESS component selection.

MATLAB header (`bindings/matlab/+pls4all/approximate_press.m`):

```text
pls4all.approximate_press  PRESS-curve component selection.

   res = pls4all.approximate_press(X, Y, max_components)

 For each k ∈ {1, …, max_components}, fits SIMPLS and approximates
 PRESS via leverage-inflated in-sample residuals. Returns:
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
p4a_approximate_press_compute(ctx, cfg, &x_view, &y_view, /* hyperparams */, &res);
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
from pls4all._methods import approximate_press_compute
with pls4all.Context() as ctx, pls4all.Config() as cfg:
    res = approximate_press_compute(ctx, cfg, X, y)
# then: res.matrix("predictions"), res.matrix("coefficients"),
# res.vector("mask"), res.scalar("intercept"), …
```

:::

:::{tab-item} Python · pls4all.sklearn
:sync: python-sklearn
:class-label: lang-python

```python
from pls4all.sklearn import approximate_press
result = approximate_press(X, y, n_components=2)
```

:::

:::{tab-item} R · pls4all_method()
:sync: r-dispatcher
:class-label: lang-r

```r
library(pls4all)
# Unified low-level dispatcher (May 2026 R cleanup):
res <- pls4all_method("approximate_press", X, y,
                      n_components = 2L, params = list(max_components = 6L))
# res is a named list with MethodResult arrays/scalars.
# selected_indices / top_k_intervals are 1-based.
```

:::

:::{tab-item} R · pls4all (raw fn)
:sync: r-raw
:class-label: lang-r

```r
library(pls4all)
res  <- approximate_press(X, Y, max_components)
yhat <- pls4all_predict(res, X_test)
```

:::

:::{tab-item} MATLAB · pls4all (MEX)
:sync: matlab-mex
:class-label: lang-matlab

```matlab
res = pls4all.approximate_press(X, y, 2);
% see header of bindings/matlab/+pls4all/approximate_press.m for full
% parameter surface:
%   res = approximate_press(X, Y, max_components)
yhat = predict(res, Xtest);
```

:::

:::{tab-item} MATLAB · pls4all (classdef)
:sync: matlab-classdef
:class-label: lang-matlab

_No idiomatic classdef wrapper — invoke `pls4all.fit("approximate_press", X, y, …)` directly from the unified MEX factory._

:::

::::


**Registry parity references** 📐

:::{card}
:class-card: external-refs

- 📐 **`ref.r_pls`** (R · r) — `pls` 2.8.5 · qualitative (rmse_rel ≤ 1e+01) — R `pls::plsr(validation='LOO')$validation$PRESS`. pls4all's approximate_press uses Eastment-Krzanowski leverage; R computes true LOO-PRESS. Same ordering, different scaling.
:::

### Validation contract

Structural validation contract for `approximate_press`, exported by the declarative validation framework (`benchmarks/validation/registry/`) from `benchmarks.parity_timing.registry`. Regenerated by `python -m benchmarks.validation.export_registry --write`.

| Field | Value |
|------|-------|
| Reference kind | `external` |
| Canonical reference role | `r` |
| Declared references | R |
| Comparator | `reference_parity` (external-reference validity gate) |
| Comparator metric | `rmse_rel` |
| Reference tolerance | `1e+01` (qualitative) |
| Binding tolerance | `1e-06` |
| Prediction key | `press_per_component` |
| Required data flags | none |

**Benchmark cell parameters**

| Parameter | Value |
|-----------|-------|
| `max_components` | `6` |
| `n_features` | `30` |
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

**Reference gate**: non-release diagnostic — shape/smoke comparison only (`rmse_rel ≤ 1e+01`). Treat it as evidence that both implementations ran, not as numerical agreement.

Rows tagged with **📐** are the canonical parity references for this method (declared in [`parity_timing.registry`](../benchmarks/methodology.md)). C++ and external rows show reference parity; pls4all language bindings show binding parity against the C++ backend. Hover the icon for role and tolerance band.

::::{tab-set}
:class: parity-tabs

:::{tab-item} 1 thread
:sync: threads-1

<div class="parity-table-wrap">
<table class="docutils parity-grouped">
<thead><tr><th scope="col">Backend</th><th scope="col">Divergence</th><th class="size-col" scope="col">100×50 (ms)</th><th class="size-col" scope="col">100×500 (ms)</th><th class="size-col" scope="col">100×2500 (ms)</th><th class="size-col" scope="col">1000×500 (ms)</th></tr></thead>
<tbody class="lang-band lang-cpp"><tr class="lang-band-row" data-lang="cpp"><th colspan="6" scope="rowgroup"><span class="lang-band-dot"></span>C++ native · libp4a</th></tr>
<tr class="bk-row"><td class="bk-name"><code>pls4all.cpp.blas</code></td><td class="parity parity-divergence parity-divergent">0.52</td><td class="ms">1.09 ms</td><td class="ms ms-best">9.05 ms<span class="medal" title="fastest">🏆</span></td><td class="ms ms-best">47.3 ms<span class="medal" title="fastest">🏆</span></td><td class="ms">97.6 ms</td></tr>
<tr class="bk-row"><td class="bk-name"><code>pls4all.cpp.blas+omp</code></td><td class="parity parity-divergence parity-divergent">0.52</td><td class="ms">1.19 ms</td><td class="ms">9.25 ms</td><td class="ms">48.1 ms</td><td class="ms">97.2 ms</td></tr>
<tr class="bk-row"><td class="bk-name"><code>pls4all.cpp.omp</code></td><td class="parity parity-divergence parity-divergent">0.52</td><td class="ms ms-best">1.06 ms<span class="medal" title="fastest">🏆</span></td><td class="ms">9.13 ms</td><td class="ms">47.4 ms</td><td class="ms ms-best">96.6 ms<span class="medal" title="fastest">🏆</span></td></tr>
<tr class="bk-row"><td class="bk-name"><code>pls4all.cpp.ref</code></td><td class="parity parity-divergence parity-divergent">0.52</td><td class="ms">1.08 ms</td><td class="ms">9.12 ms</td><td class="ms">47.7 ms</td><td class="ms">99.4 ms</td></tr>
</tbody>
<tbody class="lang-band lang-python"><tr class="lang-band-row" data-lang="python"><th colspan="6" scope="rowgroup"><span class="lang-band-dot"></span>Python · pls4all</th></tr>
<tr class="bk-row"><td class="bk-name"><code>pls4all.python</code></td><td class="parity parity-divergence parity-divergent">0.52</td><td class="ms">1.16 ms</td><td class="ms">9.45 ms</td><td class="ms">48.0 ms</td><td class="ms">100.4 ms</td></tr>
<tr class="bk-row"><td class="bk-name"><code>pls4all.sklearn</code></td><td class="parity parity-divergence parity-divergent">0.52</td><td class="ms">1.65 ms</td><td class="ms">9.27 ms</td><td class="ms">48.6 ms</td><td class="ms">97.9 ms</td></tr>
</tbody>
<tbody class="lang-band lang-r"><tr class="lang-band-row" data-lang="r"><th colspan="6" scope="rowgroup"><span class="lang-band-dot"></span>R · pls4all</th></tr>
<tr class="bk-row"><td class="bk-name"><code>pls4all.R</code></td><td class="parity parity-divergence parity-divergent">0.52</td><td class="ms">3.56 ms</td><td class="ms">46.0 ms</td><td class="ms">395.6 ms</td><td class="ms">466.9 ms</td></tr>
<tr class="bk-row"><td class="bk-name"><code>pls4all.R.formula</code></td><td class="parity parity-divergence parity-divergent">0.52</td><td class="ms">4.57 ms</td><td class="ms">57.9 ms</td><td class="ms">591.8 ms</td><td class="ms">666.7 ms</td></tr>
<tr class="bk-row"><td class="bk-name"><code>pls4all.R.mdatools</code></td><td class="parity parity-divergence parity-drift">—</td><td class="ms">4.76 ms</td><td class="ms">64.1 ms</td><td class="ms">577.9 ms</td><td class="ms">708.4 ms</td></tr>
<tr class="bk-row"><td class="bk-name"><code>pls4all.R.pls</code></td><td class="parity parity-divergence parity-drift">—</td><td class="ms">4.24 ms</td><td class="ms">58.1 ms</td><td class="ms">585.0 ms</td><td class="ms">648.8 ms</td></tr>
</tbody>
<tbody class="lang-band lang-matlab"><tr class="lang-band-row" data-lang="matlab"><th colspan="6" scope="rowgroup"><span class="lang-band-dot"></span>MATLAB · pls4all</th></tr>
<tr class="bk-row"><td class="bk-name"><code>pls4all.matlab</code></td><td class="parity parity-divergence parity-divergent">0.52</td><td class="ms">1.74 ms</td><td class="ms">14.1 ms</td><td class="ms">71.0 ms</td><td class="ms">152.2 ms</td></tr>
<tr class="bk-row"><td class="bk-name"><code>pls4all.matlab.classdef</code></td><td class="parity parity-divergence parity-divergent">0.52</td><td class="ms">2.32 ms</td><td class="ms">14.6 ms</td><td class="ms">71.3 ms</td><td class="ms">150.1 ms</td></tr>
</tbody>
<tbody class="lang-band lang-r"><tr class="lang-band-row" data-lang="r"><th colspan="6" scope="rowgroup"><span class="lang-band-dot"></span>R · external</th></tr>
<tr class="bk-row truth-source truth-source-qualitative"><td class="bk-name"><span class="truth-mark" title="Registry parity reference (r): pls 2.8.5 — qualitative (rmse_rel ≤ 1e+01)">📐</span><code>ref.r_pls</code></td><td class="parity parity-divergence parity-ref-source">0</td><td class="ms">35.9 ms</td><td class="ms">227.1 ms</td><td class="ms">1.4 s</td><td class="ms">17.0 s</td></tr>
</tbody>
</table>
</div>

:::

::::


---

_See also_: [benchmark overview](../benchmarks/overview.md) · [methods index](index.md) · [interactive dashboard](../landing/dashboard.md)