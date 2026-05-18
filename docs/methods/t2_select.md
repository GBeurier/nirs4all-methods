# `t2_select` — Hotelling T² loading selection

_Group_: **Variable selector** · _Registry tolerance_: `1.2`

## Description

T²-PLS loading-weight selection (§18 Phase 5l)

From the `pls4all.sklearn.T2Selector` docstring:

> T²-PLS loading-weight selection (plsVarSel::T2_pls style).

> **Registry note** — R `plsVarSel::T2_pls` Hotelling T² loading selection with train=test to match pls4all's single-training-set selector. The remaining divergence is threshold semantics: plsVarSel chooses the `$mv$` min-error set across alpha levels, while pls4all thresholds training-score T² directly. Mask RMSE-rel ~0=perfect, ~1=half disagree, ~1.41=disjoint.

### Parameters

| Name | Type | Default | Notes |
|------|------|---------|-------|
| `alpha_thresholds` | `—` | `None` | Sequence of significance levels (α) defining the T² acceptance regions to sweep. |
| `n_components` | `int` | `2` | Number of latent components extracted (k). |
| `min_selected` | `int | None` | `None` | Lower bound on the surviving feature count after thresholding. |

## Explanations

### Bibliographic source

Mehmood, T. (2016). *Hotelling T² based variable selection in partial least squares regression*. Chemometrics and Intelligent Laboratory Systems 154, 23–28. https://doi.org/10.1016/j.chemolab.2016.03.020 — proposes T²-PLS, the loading-weights-level Hotelling T² selector. See also Wold, Sjöström & Eriksson (2001), Chemometrics and Intelligent Laboratory Systems 58(2), 109–130 §6.2 for the original T²-vs-VIP discussion in PLS.

### Mathematical principle

Apply Hotelling T² to the **loading weights** rather than the scores: features with loading vectors of large T² are deemed important. Threshold via the F-distribution upper control limit at a user-specified $\alpha$, with a top-$k$ fallback to avoid empty selections.

Distinct from sample-level T² monitoring (see `pls_diagnostic_t2`) — here T² acts as a multivariate feature ranker that respects correlation structure across components. Useful when the loadings have strong between-component structure and per-component VIP under-counts contributions spread across multiple components.

### Implementation

`p4a_t2_select`.

R roxygen note (`methods_extra.R::t2_select`):

> T²-PLS — sweep over α-thresholds.
> @param n_components Integer. Number of latent components.
> @param alpha_thresholds Method-specific parameter. See the underlying `*_fit()` function for the exact semantics.
> @param min_selected Method-specific parameter. See the underlying `*_fit()` function for the exact semantics.
> @param X Numeric matrix of predictors (rows = samples, cols = features).
> @param Y Numeric matrix or vector of responses, with one row per sample.
> @export

MATLAB header (`bindings/matlab/+pls4all/t2_select.m`):

```text
pls4all.t2_select  T²-based selector (sweep over alpha thresholds).
```

### Usage

All four pls4all bindings dispatch into the same C kernel; the external libraries on the right are the parity references registered in `benchmarks.parity_timing.registry`. Switch tabs to read the same fit in your language.

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
p4a_t2_select(ctx, cfg, &x_view, &y_view, /* hyperparams */, &res);
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
from pls4all._methods import t2_select
with pls4all.Context() as ctx, pls4all.Config() as cfg:
    res = t2_select(ctx, cfg, X, y)
# then: res.matrix("predictions"), res.matrix("coefficients"),
# res.vector("mask"), res.scalar("intercept"), …
```

:::

:::{tab-item} Python · pls4all.sklearn
:sync: python-sklearn
:class-label: lang-python

```python
from pls4all.sklearn import T2Selector
mdl = T2Selector(alpha_thresholds, n_components=2, min_selected=None)
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
res <- pls4all_method("t2_select", X, y,
                      n_components = 2L)
# res is a named list with MethodResult arrays/scalars.
# selected_indices / top_k_intervals are 1-based.
```

:::

:::{tab-item} R · pls4all (raw fn)
:sync: r-raw
:class-label: lang-r

```r
library(pls4all)
res  <- t2_select(X, Y, n_components, alpha_thresholds,
           min_selected = NULL)
yhat <- pls4all_predict(res, X_test)
```

:::

:::{tab-item} MATLAB · pls4all (MEX)
:sync: matlab-mex
:class-label: lang-matlab

```matlab
res = pls4all.t2_select(X, y, 2);
% see header of bindings/matlab/+pls4all/t2_select.m for full
% parameter surface:
%   res = t2_select(X, Y, n_components, alpha_thresholds, min_selected)
yhat = predict(res, Xtest);
```

:::

:::{tab-item} MATLAB · pls4all (classdef)
:sync: matlab-classdef
:class-label: lang-matlab

_No idiomatic classdef wrapper — invoke `pls4all.fit("t2_select", X, y, …)` directly from the unified MEX factory._

:::

::::


**Registry parity references** 📐

:::{card}
:class-card: external-refs

- 📐 **`ref.r_plsvarsel`** (R · r) — `plsVarSel` 0.10.0 · qualitative (rmse_rel ≤ 1e+00) — R `plsVarSel::T2_pls` — Hotelling T² loading-weight selection. Same idea as pls4all's T2_select.
:::

### Benchmarks

Median wall-clock per cell from [`benchmarks/cross_binding/results/full_matrix.csv`](../benchmarks/overview.md). Verdict legend: ✓ exact · ≈ drift · ✗ divergent · ⊘ not available in lib · — not run · ⚠ error. The fastest backend per column is marked with a 🏆 medal. Rows tagged with **📐** are *also* declared in [`benchmarks/parity_timing/registry.py`](../benchmarks/methodology.md) as the canonical parity references for this method (`python_reference` / `r_reference` / `extra_references`). Cell parity in this table is measured against the cross-binding reference (`pls4all.cpp` blas-omp, 1 thread); the 📐 icon points at the *library-of-record* the parity gate ultimately answers to. Hover the icon to see the role and tolerance band.

::::{tab-set}
:class: parity-tabs

:::{tab-item} 1 thread
:sync: threads-1

<div class="parity-table-wrap">
<table class="docutils parity-grouped">
<thead><tr><th scope="col">Backend</th><th scope="col">Parity</th><th class="size-col" scope="col">200×40 (ms)</th></tr></thead>
<tbody class="lang-band lang-python"><tr class="lang-band-row" data-lang="python"><th colspan="3" scope="rowgroup"><span class="lang-band-dot"></span>Python · pls4all</th></tr>
<tr class="bk-row"><td class="bk-name"><code>pls4all.registry</code></td><td class="parity parity-exact">✓ ref</td><td class="ms ms-best">171.0 ms<span class="medal" title="fastest">🏆</span></td></tr>
</tbody>
<tbody class="lang-band lang-r"><tr class="lang-band-row" data-lang="r"><th colspan="3" scope="rowgroup"><span class="lang-band-dot"></span>R · external</th></tr>
<tr class="bk-row truth-source truth-source-qualitative"><td class="bk-name"><span class="truth-mark" title="Registry parity reference (r): plsVarSel 0.10.0 — qualitative (rmse_rel ≤ 1e+00)">📐</span><code>ref.r_plsvarsel</code></td><td class="parity parity-divergent">✗ +1e+00</td><td class="ms">353.7 ms</td></tr>
</tbody>
</table>
</div>

:::

::::


---

_See also_: [benchmark overview](../benchmarks/overview.md) · [methods index](index.md) · [interactive dashboard](../landing/dashboard.md)