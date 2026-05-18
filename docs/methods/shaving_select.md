# `shaving_select` — Shaving (recursive elimination)

_Group_: **Variable selector** · _Registry tolerance_: `0.7`

## Description

Shaving iterative variable trimming

From the `pls4all.sklearn.ShavingSelector` docstring:

> Iterative SR-shaving variable elimination.

> **Registry note** — R `plsVarSel::shaving(method='SR')` iterative trimming. Mask RMSE-rel ~0=perfect, ~1=half disagree, ~1.41=disjoint; tolerance 0.7 enforces ~50% overlap.

### Parameters

| Name | Type | Default | Notes |
|------|------|---------|-------|
| `n_components` | `int` | `2` | Number of latent components extracted (k). |
| `n_steps` | `int` | `10` | Number of elimination passes performed. |
| `min_features` | `int | None` | `None` | Minimum number of variables the selector is allowed to keep (defaults to `n_components`). |
| `shave_fraction` | `float` | `0.2` | Fraction of the worst-ranked variables removed at each shaving step. |
| `n_folds` | `int` | `3` | Number of cross-validation folds used inside the selector. |

## Explanations

### Bibliographic source

Mehmood, T., Liland, K. H., Snipen, L. & Sæbø, S. (2012). *A review of variable selection methods in partial least squares regression*. Chemometrics and Intelligent Laboratory Systems 118, 62–69 (§3.2 Shaving).

### Mathematical principle

Shaving recursively eliminates a fraction $\rho \in (0, 1)$ of the lowest-scoring features at each step, refits PLS, and tracks the CV-RMSE of the shrinking subset. The subset with the lowest recorded CV-RMSE across the whole shaving trajectory is returned.

Compared to backward variable elimination (BVE — see next), shaving removes a **batch** of features per step instead of one, which is faster ($O(\log p)$ steps vs $O(p)$) but more aggressive — a single bad shave removes many useful features irrecoverably. Recommended $\rho \le 0.2$ to keep shave granularity reasonable.

### Implementation

`p4a_shaving_select`.

R roxygen note (`methods_extra.R::shaving_select`):

> Shaving selector.
> @param n_components Integer. Number of latent components.
> @param n_steps Method-specific parameter. See the underlying `*_fit()` function for the exact semantics.
> @param min_features Method-specific parameter. See the underlying `*_fit()` function for the exact semantics.
> @param shave_fraction Method-specific parameter. See the underlying `*_fit()` function for the exact semantics.
> @param X Numeric matrix of predictors (rows = samples, cols = features).
> @param Y Numeric matrix or vector of responses, with one row per sample.
> @export

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
p4a_shaving_select(ctx, cfg, &x_view, &y_view, /* hyperparams */, &res);
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
from pls4all._methods import shaving_select
with pls4all.Context() as ctx, pls4all.Config() as cfg:
    res = shaving_select(ctx, cfg, X, y, n_components=4)
# then: res.matrix("predictions"), res.matrix("coefficients"),
# res.vector("mask"), res.scalar("intercept"), …
```

:::

:::{tab-item} Python · pls4all.sklearn
:sync: python-sklearn
:class-label: lang-python

```python
from pls4all.sklearn import ShavingSelector
mdl = ShavingSelector(n_components=2, n_steps=10, min_features=None, shave_fraction=0.2, n_folds=3)
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
res <- pls4all_method("shaving_select", X, y,
                      n_components = 4L, params = list(n_steps = 4L, min_features = 5L, shave_fraction = 0.2))
# res is a named list with MethodResult arrays/scalars.
# selected_indices / top_k_intervals are 1-based.
```

:::

:::{tab-item} R · pls4all (raw fn)
:sync: r-raw
:class-label: lang-r

```r
library(pls4all)
res  <- shaving_select(X, Y, n_components,
                n_steps = 10L, min_features = 5L,
                shave_fraction = 0.1)
yhat <- pls4all_predict(res, X_test)
```

:::

:::{tab-item} MATLAB · pls4all (MEX)
:sync: matlab-mex
:class-label: lang-matlab

```matlab
res = pls4all.shaving_select(X, y, 4);
% see header of bindings/matlab/+pls4all/shaving_select.m for full
% parameter surface:
%   res = shaving_select(X, Y, n_components, n_steps, min_features, shave_fraction)
yhat = predict(res, Xtest);
```

:::

:::{tab-item} MATLAB · pls4all (classdef)
:sync: matlab-classdef
:class-label: lang-matlab

_No idiomatic classdef wrapper — invoke `pls4all.fit("shaving_select", X, y, …)` directly from the unified MEX factory._

:::

::::


**Registry parity references** 📐

:::{card}
:class-card: external-refs

- 📐 **`ref.r_plsvarsel`** (R · r) — `plsVarSel` 0.10.0 · qualitative (rmse_rel ≤ 7e-01) — R `plsVarSel::shaving(method='SR')` — iterative SR-shaving of low-importance features.
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
<tr class="bk-row"><td class="bk-name"><code>pls4all.registry</code></td><td class="parity parity-exact">✓ ref</td><td class="ms ms-best">173.4 ms<span class="medal" title="fastest">🏆</span></td></tr>
</tbody>
<tbody class="lang-band lang-r"><tr class="lang-band-row" data-lang="r"><th colspan="3" scope="rowgroup"><span class="lang-band-dot"></span>R · external</th></tr>
<tr class="bk-row truth-source truth-source-qualitative"><td class="bk-name"><span class="truth-mark" title="Registry parity reference (r): plsVarSel 0.10.0 — qualitative (rmse_rel ≤ 7e-01)">📐</span><code>ref.r_plsvarsel</code></td><td class="parity parity-divergent">✗ +1e+00</td><td class="ms">357.1 ms</td></tr>
</tbody>
</table>
</div>

:::

::::


---

_See also_: [benchmark overview](../benchmarks/overview.md) · [methods index](index.md) · [interactive dashboard](../landing/dashboard.md)