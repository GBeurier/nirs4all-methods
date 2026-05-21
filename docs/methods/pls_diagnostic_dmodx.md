# `pls_diagnostic_dmodx` — DModX (distance to the model in X)

_Group_: **Diagnostic** · _Registry tolerance_: `5.0`

## Description

PLS Distance-to-Model X (§9)

> **Registry note** — mdatools has no native DModX; the R script computes it from `$xdecomp$Q` and the same dof formula pls4all uses.

### Parameters

| Name | Type | Default | Notes |
|------|------|---------|-------|
| `n_components` | `int` | `4` | registry benchmark cell value |

## Explanations

### Bibliographic source

Eriksson, L., Byrne, T., Johansson, E., Trygg, J. & Vikström, C. (2013). *Multi- and Megavariate Data Analysis. Basic Principles and Applications*, 3rd ed., Umetrics Academy, §4.7.

### Mathematical principle

DModX is a **normalised** version of Q that accounts for the model's residual degrees of freedom: $\mathrm{DModX}_i = \sqrt{Q_i \,/\, (p - k - 1) \,\cdot\, (n - k - 1) / \bar{Q}_{\mathrm{cal}}}$ where $\bar{Q}_{\mathrm{cal}}$ is the average Q on the calibration set.

DModX is the diagnostic of choice in the SIMCA software (Umetrics) and is commonly used because the normalisation produces a unit-less quantity directly comparable across models with different $p$, $n$, $k$. Critical thresholds follow the F-distribution: DModX > 2 is the heuristic outlier cutoff in practice.

Conceptually equivalent to Q for monitoring purposes, but the normalisation is what makes it portable.

### Implementation

`p4a_pls_diagnostics_compute` with stat='dmodx'.

R roxygen note (`diagnostics.R::pls_diagnostics`):

> PLS diagnostics: T², Q, DModX from a fitted model.

MATLAB header (`bindings/matlab/+pls4all/pls_diagnostics.m`):

```text
pls4all.pls_diagnostics  Hotelling T2, Q residuals, DModX from a SIMPLS fit.

   res = pls4all.pls_diagnostics(X, Y, n_components)
   res = pls4all.pls_diagnostics(X, Y, n_components, X_reference)

 Fits an internal SIMPLS model (store_scores=1) and evaluates row-wise
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
p4a_pls_diagnostics_compute(ctx, cfg, &x_view, &y_view, /* hyperparams */, &res);
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
from pls4all._methods import pls_diagnostics_compute
with pls4all.Context() as ctx, pls4all.Config() as cfg:
    res = pls_diagnostics_compute(ctx, cfg, X, y, n_components=4)
# then: res.matrix("predictions"), res.matrix("coefficients"),
# res.vector("mask"), res.scalar("intercept"), …
```

:::

:::{tab-item} Python · pls4all.sklearn
:sync: python-sklearn
:class-label: lang-python

```python
from pls4all.sklearn import dmodx_score
result = dmodx_score(X, y, n_components=4)
```

:::

:::{tab-item} R · pls4all_method()
:sync: r-dispatcher
:class-label: lang-r

```r
library(pls4all)
# Unified low-level dispatcher (May 2026 R cleanup):
res <- pls4all_method("pls_diagnostic_dmodx", X, y,
                      n_components = 4L)
# res is a named list with MethodResult arrays/scalars.
# selected_indices / top_k_intervals are 1-based.
```

:::

:::{tab-item} R · pls4all (raw fn)
:sync: r-raw
:class-label: lang-r

```r
library(pls4all)
res  <- pls_diagnostics(model, X)
yhat <- pls4all_predict(res, X_test)
```

:::

:::{tab-item} MATLAB · pls4all (MEX)
:sync: matlab-mex
:class-label: lang-matlab

```matlab
res = pls4all.pls_diagnostics(X, y, 4);
% see header of bindings/matlab/+pls4all/pls_diagnostics.m for full
% parameter surface:
%   res = pls_diagnostics(X, Y, n_components, X_reference)
yhat = predict(res, Xtest);
```

:::

:::{tab-item} MATLAB · pls4all (classdef)
:sync: matlab-classdef
:class-label: lang-matlab

_No idiomatic classdef wrapper — invoke `pls4all.fit("pls_diagnostic_dmodx", X, y, …)` directly from the unified MEX factory._

:::

::::


**Registry parity references** 📐

:::{card}
:class-card: external-refs

- 📐 **`ref.r_mdatools`** (R · r) — `mdatools` 0.15.0 · qualitative (rmse_rel ≤ 5e+00) — R `mdatools::pls` with `predict()$xdecomp$T2 / $Q`. DModX is derived locally from $Q + DOF. mdatools uses different SIMPLS deflation / normalization conventions than pls4all, so cross-implementation parity is qualitative.
:::

### Validation contract

Structural validation contract for `pls_diagnostic_dmodx`, exported by the declarative validation framework (`benchmarks/validation/registry/`) from `benchmarks.parity_timing.registry`. Regenerated by `python -m benchmarks.validation.export_registry --write`.

| Field | Value |
|------|-------|
| Reference kind | `external` |
| Canonical reference role | `r` |
| Declared references | R |
| Comparator | `reference_parity` (external-reference validity gate) |
| Comparator metric | `rmse_rel` |
| Reference tolerance | `5e+00` (qualitative) |
| Binding tolerance | `1e-06` |
| Prediction key | `dmodx` |
| Required data flags | none |

**Benchmark cell parameters**

| Parameter | Value |
|-----------|-------|
| `n_components` | `4` |
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
<tr class="bk-row"><td class="bk-name"><code>pls4all.cpp.blas</code></td><td class="parity parity-divergence parity-ref-qualitative">2e-15</td><td class="ms">1.04 ms</td><td class="ms ms-best">8.52 ms<span class="medal" title="fastest">🏆</span></td><td class="ms">45.9 ms</td><td class="ms">86.6 ms</td></tr>
<tr class="bk-row"><td class="bk-name"><code>pls4all.cpp.blas+omp</code></td><td class="parity parity-divergence parity-ref-qualitative">2e-15</td><td class="ms">1.03 ms</td><td class="ms">8.65 ms</td><td class="ms">46.7 ms</td><td class="ms">87.1 ms</td></tr>
<tr class="bk-row"><td class="bk-name"><code>pls4all.cpp.omp</code></td><td class="parity parity-divergence parity-ref-qualitative">2e-15</td><td class="ms">1.03 ms</td><td class="ms">8.89 ms</td><td class="ms">46.0 ms</td><td class="ms">87.3 ms</td></tr>
<tr class="bk-row"><td class="bk-name"><code>pls4all.cpp.ref</code></td><td class="parity parity-divergence parity-ref-qualitative">2e-15</td><td class="ms ms-best">1.02 ms<span class="medal" title="fastest">🏆</span></td><td class="ms">8.69 ms</td><td class="ms">45.3 ms</td><td class="ms">87.6 ms</td></tr>
</tbody>
<tbody class="lang-band lang-python"><tr class="lang-band-row" data-lang="python"><th colspan="6" scope="rowgroup"><span class="lang-band-dot"></span>Python · pls4all</th></tr>
<tr class="bk-row"><td class="bk-name"><code>pls4all.python</code></td><td class="parity parity-divergence parity-drift">2e-15</td><td class="ms">1.03 ms</td><td class="ms">8.57 ms</td><td class="ms">46.1 ms</td><td class="ms">86.8 ms</td></tr>
<tr class="bk-row"><td class="bk-name"><code>pls4all.sklearn</code></td><td class="parity parity-divergence parity-drift">2e-15</td><td class="ms">1.25 ms</td><td class="ms">9.01 ms</td><td class="ms ms-best">43.6 ms<span class="medal" title="fastest">🏆</span></td><td class="ms ms-best">83.3 ms<span class="medal" title="fastest">🏆</span></td></tr>
</tbody>
<tbody class="lang-band lang-r"><tr class="lang-band-row" data-lang="r"><th colspan="6" scope="rowgroup"><span class="lang-band-dot"></span>R · pls4all</th></tr>
<tr class="bk-row"><td class="bk-name"><code>pls4all.R</code></td><td class="parity parity-divergence parity-drift">2e-15</td><td class="ms">2.96 ms</td><td class="ms">49.7 ms</td><td class="ms">544.9 ms</td><td class="ms">599.9 ms</td></tr>
<tr class="bk-row"><td class="bk-name"><code>pls4all.R.formula</code></td><td class="parity parity-divergence parity-drift">2e-15</td><td class="ms">4.99 ms</td><td class="ms">64.2 ms</td><td class="ms">670.3 ms</td><td class="ms">538.9 ms</td></tr>
<tr class="bk-row"><td class="bk-name"><code>pls4all.R.mdatools</code></td><td class="parity parity-divergence parity-drift">2e-15</td><td class="ms">4.89 ms</td><td class="ms">62.8 ms</td><td class="ms">634.4 ms</td><td class="ms">565.0 ms</td></tr>
<tr class="bk-row"><td class="bk-name"><code>pls4all.R.pls</code></td><td class="parity parity-divergence parity-drift">2e-15</td><td class="ms">4.60 ms</td><td class="ms">64.3 ms</td><td class="ms">639.7 ms</td><td class="ms">553.1 ms</td></tr>
</tbody>
<tbody class="lang-band lang-matlab"><tr class="lang-band-row" data-lang="matlab"><th colspan="6" scope="rowgroup"><span class="lang-band-dot"></span>MATLAB · pls4all</th></tr>
<tr class="bk-row"><td class="bk-name"><code>pls4all.matlab</code></td><td class="parity parity-divergence parity-drift">2e-15</td><td class="ms">1.74 ms</td><td class="ms">13.4 ms</td><td class="ms">70.2 ms</td><td class="ms">134.8 ms</td></tr>
<tr class="bk-row"><td class="bk-name"><code>pls4all.matlab.classdef</code></td><td class="parity parity-divergence parity-drift">2e-15</td><td class="ms">2.23 ms</td><td class="ms">14.7 ms</td><td class="ms">70.9 ms</td><td class="ms">136.8 ms</td></tr>
</tbody>
<tbody class="lang-band lang-r"><tr class="lang-band-row" data-lang="r"><th colspan="6" scope="rowgroup"><span class="lang-band-dot"></span>R · external</th></tr>
<tr class="bk-row truth-source truth-source-qualitative"><td class="bk-name"><span class="truth-mark" title="Registry parity reference (r): mdatools 0.15.0 — qualitative (rmse_rel ≤ 5e+00)">📐</span><code>ref.r_mdatools</code></td><td class="parity parity-divergence parity-ref-source">0</td><td class="ms">16.0 ms</td><td class="ms">114.3 ms</td><td class="ms">1.3 s</td><td class="ms">1.2 s</td></tr>
</tbody>
</table>
</div>

:::

::::


---

_See also_: [benchmark overview](../benchmarks/overview.md) · [methods index](index.md) · [interactive dashboard](../landing/dashboard.md)