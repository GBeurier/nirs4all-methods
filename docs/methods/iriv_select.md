# `iriv_select` — IRIV — Iteratively Retaining Informative Variables

_Group_: **Variable selector** · _Registry tolerance_: `1.0`

## Description

Iteratively Retains Informative Variables (Phase 51)

From the `pls4all.sklearn.IRIVSelector` docstring:

> IRIV — Iteratively Retains Informative Variables (Yun 2014).

> **Registry note** — Octave-bridged libPLS 1.95 `iriv`. Mask metric; pls4all uses splitmix64 RNG vs MATLAB rand; libPLS's trailing BVE pass is omitted (expose BVE separately). Mask ~0=perfect, ~1=half disagree, ~1.41=disjoint.

### Parameters

| Name | Type | Default | Notes |
|------|------|---------|-------|
| `n_components` | `int` | `2` | Number of latent components extracted (k). |
| `max_rounds` | `int` | `5` | Maximum rounds of strongly/weakly informative reclassification. |
| `n_folds` | `int` | `5` | Number of cross-validation folds used inside the selector. |
| `seed` | `int` | `0` | Random seed for reproducible sampling/initialization. |
| `fold` | `int` | `3` | registry benchmark cell value |

## Explanations

### Bibliographic source

Yun, Y. H., Wang, W. T., Tan, M. L., Liang, Y. Z., Li, H. D., Cao, D. S., Lu, H. M. & Xu, Q. S. (2014). *A strategy that iteratively retains informative variables for selecting optimal variable subset in multivariate calibration*. Analytica Chimica Acta 807, 36–43.

### Mathematical principle

IRIV classifies each variable into four categories at each iteration: **strongly informative**, **weakly informative**, **uninformative**, **interfering**. The first two are retained, the last two eliminated. Iteration continues until no further interfering variables remain.

Categories are determined by Monte-Carlo subset analysis with a permutation-based reference distribution: each variable's CV-RMSE contribution distribution is compared against the distribution under random subset inclusion. This four-way classification is more nuanced than single-threshold methods and tends to handle correlated predictors well (correlated features can both be 'weakly informative').

### Implementation

`p4a_iriv_select`.

R roxygen note (`methods_extra.R::iriv_select`):

> IRIV — Iteratively Retains Informative Variables.
> @param n_components Integer. Number of latent components.
> @param max_rounds Method-specific parameter. See the underlying `*_fit()` function for the exact semantics.
> @param seed Integer. Random seed for reproducibility.
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
p4a_iriv_select(ctx, cfg, &x_view, &y_view, /* hyperparams */, &res);
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
from pls4all._methods import iriv_select
with pls4all.Context() as ctx, pls4all.Config() as cfg:
    res = iriv_select(ctx, cfg, X, y, n_components=4)
# then: res.matrix("predictions"), res.matrix("coefficients"),
# res.vector("mask"), res.scalar("intercept"), …
```

:::

:::{tab-item} Python · pls4all.sklearn
:sync: python-sklearn
:class-label: lang-python

```python
from pls4all.sklearn import IRIVSelector
mdl = IRIVSelector(n_components=2, max_rounds=5, n_folds=5, seed=0)
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
res <- pls4all_method("iriv_select", X, y,
                      n_components = 4L, params = list(max_rounds = 3L, fold = 3L, seed = 11L))
# res is a named list with MethodResult arrays/scalars.
# selected_indices / top_k_intervals are 1-based.
```

:::

:::{tab-item} R · pls4all (raw fn)
:sync: r-raw
:class-label: lang-r

```r
library(pls4all)
res  <- iriv_select(X, Y, n_components, max_rounds = 20L, seed = 0L)
yhat <- pls4all_predict(res, X_test)
```

:::

:::{tab-item} MATLAB · pls4all (MEX)
:sync: matlab-mex
:class-label: lang-matlab

```matlab
res = pls4all.iriv_select(X, y, 4);
% see header of bindings/matlab/+pls4all/iriv_select.m for full
% parameter surface:
%   res = iriv_select(X, Y, n_components, max_rounds, seed)
yhat = predict(res, Xtest);
```

:::

:::{tab-item} MATLAB · pls4all (classdef)
:sync: matlab-classdef
:class-label: lang-matlab

_No idiomatic classdef wrapper — invoke `pls4all.fit("iriv_select", X, y, …)` directly from the unified MEX factory._

:::

::::


**Registry parity references** 📐

:::{card}
:class-card: external-refs

- 📐 **`ref.matlab_libpls`** (matlab · python) — `libPLS` 1.95 · qualitative (rmse_rel ≤ 1e+00) — Octave-bridged libPLS 1.95 `iriv(X, y, A_max, fold, 'center')`. Requires Octave >= 11.1 + the `statistics` package for `ranksum`. RNG differs from pls4all; mask metric.
:::

### Benchmarks

Median wall-clock per cell from [`benchmarks/cross_binding/results/full_matrix.csv`](../benchmarks/overview.md). Verdict legend: ✓ exact · ≈ drift · ✗ divergent · ⊘ not available in lib · — not run · ⚠ error. The fastest backend per column is marked with a 🏆 medal. Rows tagged with **📐** are *also* declared in [`benchmarks/parity_timing/registry.py`](../benchmarks/methodology.md) as the canonical parity references for this method (`python_reference` / `r_reference` / `extra_references`). Cell parity in this table is measured against the cross-binding reference (`pls4all.cpp` blas-omp, 1 thread); the 📐 icon points at the *library-of-record* the parity gate ultimately answers to. Hover the icon to see the role and tolerance band.

::::{tab-set}
:class: parity-tabs

:::{tab-item} 1 thread
:sync: threads-1

<div class="parity-table-wrap">
<table class="docutils parity-grouped">
<thead><tr><th scope="col">Backend</th><th scope="col">Parity</th><th class="size-col" scope="col">80×25 (ms)</th></tr></thead>
<tbody class="lang-band lang-python"><tr class="lang-band-row" data-lang="python"><th colspan="3" scope="rowgroup"><span class="lang-band-dot"></span>Python · pls4all</th></tr>
<tr class="bk-row"><td class="bk-name"><code>pls4all.registry</code></td><td class="parity parity-exact">✓ ref</td><td class="ms ms-best">201.1 ms<span class="medal" title="fastest">🏆</span></td></tr>
</tbody>
</table>
</div>

:::

::::


---

_See also_: [benchmark overview](../benchmarks/overview.md) · [methods index](index.md) · [interactive dashboard](../landing/dashboard.md)