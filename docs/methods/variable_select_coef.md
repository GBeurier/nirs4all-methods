# `variable_select_coef` — Coefficient-magnitude selection

_Group_: **Variable selector** · _Registry tolerance_: `1.1`

## Description

|Coef| top-k selection (§18 Phase 5a, method=1)

From the `pls4all.sklearn.CoefficientSelector` docstring:

> |coef| top-k selector. Ranks features by the magnitude of their
PLS regression coefficient on Y.

> **Registry note** — R `pls::plsr(method='simpls')` |coef| ranking. The solver mismatch is fixed, but residual top-k drift remains because pls4all ranks its stored C-kernel coefficient vector while R reconstructs coefficients through `pls`'s SIMPLS convention. Mask RMSE-rel ~0=perfect, ~1=half disagree, ~1.41=disjoint; tolerance accepts this known coefficient-convention divergence.

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

Martens, H. & Næs, T. (1989). *Multivariate Calibration*, §5. — the simplest ranking baseline.

### Mathematical principle

Rank features by the absolute magnitude of their PLS regression coefficient $|b_j|$ in the original feature scale. Pick the top-$k$ as the selected subset.

This is the simplest possible PLS variable selector. It is biased — features with large variance get smaller coefficients for the same predictive effect — so it should usually be applied after autoscaling to remove the variance-induced bias. Once autoscaled, $|b_j|$ ranks features by their **standardised partial effect on $y$**, which is statistically meaningful.

Useful as a sanity-check baseline against more sophisticated selectors. If a complex method does not beat coefficient-magnitude selection, it is probably over-engineered.

### Implementation

`p4a_variable_select_rank` with metric=COEF.

R roxygen note (`selectors.R::coefficient_select`):

> Coefficient-magnitude ranker.
> @inheritParams vip_select
> @return A list with `scores` (|coef| sums) and `selected_indices`.
> @param model Method-specific parameter. See the underlying `*_fit()` function for the exact semantics.
> @param top_k Method-specific parameter. See the underlying `*_fit()` function for the exact semantics.
> @export

MATLAB header (`bindings/matlab/+pls4all/coefficient_select.m`):

```text
pls4all.coefficient_select  Coefficient-magnitude feature ranking.

   res = pls4all.coefficient_select(X, Y, n_components, top_k)

 Fits an internal SIMPLS model and ranks features by the magnitude of
 their regression coefficients.
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
from pls4all.sklearn import CoefficientSelector
mdl = CoefficientSelector(top_k, n_components=2, solver='simpls', center_x=True, scale_x=True, tol=1e-06, max_iter=500)
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
res <- pls4all_method("variable_select_coef", X, y,
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
res  <- coefficient_select(model, X, top_k)
yhat <- pls4all_predict(res, X_test)
```

:::

:::{tab-item} MATLAB · pls4all (MEX)
:sync: matlab-mex
:class-label: lang-matlab

```matlab
res = pls4all.coefficient_select(X, y, 4);
% see header of bindings/matlab/+pls4all/coefficient_select.m for full
% parameter surface:
%   res = coefficient_select(X, Y, n_components, top_k)
yhat = predict(res, Xtest);
```

:::

:::{tab-item} MATLAB · pls4all (classdef)
:sync: matlab-classdef
:class-label: lang-matlab

_No idiomatic classdef wrapper — invoke `pls4all.fit("variable_select_coef", X, y, …)` directly from the unified MEX factory._

:::

::::


**Registry parity references** 📐

:::{card}
:class-card: external-refs

- 📐 **`ref.r_pls`** (R · r) — `pls` 2.8.5 · qualitative (rmse_rel ≤ 1e+00) — R `pls::plsr` coefficient magnitudes — top-k indices ranked by |coef|. Mirrors method=1 of pls4all's ranker.
:::

### Benchmarks

Median wall-clock per cell from [`benchmarks/cross_binding/results/full_matrix.csv`](../benchmarks/overview.md). Verdict legend: ✓ exact · ≈ drift · ✗ divergent · ⊘ not available in lib · — not run · ⚠ error. The fastest backend per column is marked with a 🏆 medal. Rows tagged with **📐** are *also* declared in [`benchmarks/parity_timing/registry.py`](../benchmarks/methodology.md) as the canonical parity references for this method (`python_reference` / `r_reference` / `extra_references`). C++ and external rows show reference parity; pls4all language bindings show binding parity against the C++ backend. The 📐 icon points at the *library-of-record* the parity gate ultimately answers to. Hover the icon to see the role and tolerance band.

::::{tab-set}
:class: parity-tabs

:::{tab-item} 1 thread
:sync: threads-1

<div class="parity-table-wrap">
<table class="docutils parity-grouped">
<thead><tr><th scope="col">Backend</th><th scope="col">Parity</th><th class="size-col" scope="col">100×50 (ms)</th></tr></thead>
<tbody class="lang-band lang-cpp"><tr class="lang-band-row" data-lang="cpp"><th colspan="3" scope="rowgroup"><span class="lang-band-dot"></span>C++ native · libp4a</th></tr>
<tr class="bk-row"><td class="bk-name"><code>pls4all.cpp.blas</code></td><td class="parity parity-exact">✓ ref</td><td class="ms">1.05 ms</td></tr>
<tr class="bk-row"><td class="bk-name"><code>pls4all.cpp.blas+omp</code></td><td class="parity parity-exact">✓ ref</td><td class="ms">0.99 ms</td></tr>
<tr class="bk-row"><td class="bk-name"><code>pls4all.cpp.omp</code></td><td class="parity parity-exact">✓ ref</td><td class="ms">1.06 ms</td></tr>
<tr class="bk-row"><td class="bk-name"><code>pls4all.cpp.ref</code></td><td class="parity parity-exact">✓ ref</td><td class="ms">1.03 ms</td></tr>
</tbody>
<tbody class="lang-band lang-python"><tr class="lang-band-row" data-lang="python"><th colspan="3" scope="rowgroup"><span class="lang-band-dot"></span>Python · pls4all</th></tr>
<tr class="bk-row"><td class="bk-name"><code>pls4all.python</code></td><td class="parity parity-exact">✓ bind</td><td class="ms">1.08 ms</td></tr>
<tr class="bk-row"><td class="bk-name"><code>pls4all.registry</code></td><td class="parity parity-exact">✓ bind</td><td class="ms ms-best">0.95 ms<span class="medal" title="fastest">🏆</span></td></tr>
<tr class="bk-row"><td class="bk-name"><code>pls4all.sklearn</code></td><td class="parity parity-exact">✓ bind</td><td class="ms">1.14 ms</td></tr>
</tbody>
<tbody class="lang-band lang-r"><tr class="lang-band-row" data-lang="r"><th colspan="3" scope="rowgroup"><span class="lang-band-dot"></span>R · pls4all</th></tr>
<tr class="bk-row"><td class="bk-name"><code>pls4all.R</code></td><td class="parity parity-exact">✓ bind</td><td class="ms">2.72 ms</td></tr>
<tr class="bk-row"><td class="bk-name"><code>pls4all.R.formula</code></td><td class="parity parity-exact">✓ bind</td><td class="ms">3.69 ms</td></tr>
</tbody>
<tbody class="lang-band lang-matlab"><tr class="lang-band-row" data-lang="matlab"><th colspan="3" scope="rowgroup"><span class="lang-band-dot"></span>MATLAB · pls4all</th></tr>
<tr class="bk-row"><td class="bk-name"><code>pls4all.matlab</code></td><td class="parity parity-exact">✓ bind</td><td class="ms">1.72 ms</td></tr>
<tr class="bk-row"><td class="bk-name"><code>pls4all.matlab.classdef</code></td><td class="parity parity-exact">✓ bind</td><td class="ms">1.88 ms</td></tr>
</tbody>
<tbody class="lang-band lang-python"><tr class="lang-band-row" data-lang="python"><th colspan="3" scope="rowgroup"><span class="lang-band-dot"></span>Python · external</th></tr>
<tr class="bk-row"><td class="bk-name"><code>ikpls</code></td><td class="parity parity-not_available">⊘</td><td class="ms">—</td></tr>
<tr class="bk-row"><td class="bk-name"><code>sklearn</code></td><td class="parity parity-not_available">⊘</td><td class="ms">—</td></tr>
</tbody>
<tbody class="lang-band lang-r"><tr class="lang-band-row" data-lang="r"><th colspan="3" scope="rowgroup"><span class="lang-band-dot"></span>R · external</th></tr>
<tr class="bk-row"><td class="bk-name"><code>mixOmics</code></td><td class="parity parity-not_available">⊘</td><td class="ms">—</td></tr>
<tr class="bk-row"><td class="bk-name"><code>pls</code></td><td class="parity parity-not_available">⊘</td><td class="ms">—</td></tr>
<tr class="bk-row truth-source truth-source-qualitative"><td class="bk-name"><span class="truth-mark" title="Registry parity reference (r): pls 2.8.5 — qualitative (rmse_rel ≤ 1e+00)">📐</span><code>ref.r_pls</code></td><td class="parity parity-exact">✓ ref</td><td class="ms">284.0 ms</td></tr>
<tr class="bk-row"><td class="bk-name"><code>ropls</code></td><td class="parity parity-not_available">⊘</td><td class="ms">—</td></tr>
</tbody>
<tbody class="lang-band lang-matlab"><tr class="lang-band-row" data-lang="matlab"><th colspan="3" scope="rowgroup"><span class="lang-band-dot"></span>MATLAB · external</th></tr>
<tr class="bk-row"><td class="bk-name"><code>plsregress</code></td><td class="parity parity-not_run">—</td><td class="ms">—</td></tr>
</tbody>
<tbody class="lang-band lang-ext"><tr class="lang-band-row" data-lang="ext"><th colspan="3" scope="rowgroup"><span class="lang-band-dot"></span>Other</th></tr>
<tr class="bk-row"><td class="bk-name"><code>r_mdatools_compat</code></td><td class="parity parity-exact">✓ bind</td><td class="ms">3.49 ms</td></tr>
<tr class="bk-row"><td class="bk-name"><code>r_pls_compat</code></td><td class="parity parity-exact">✓ bind</td><td class="ms">3.20 ms</td></tr>
</tbody>
</table>
</div>

:::

::::


---

_See also_: [benchmark overview](../benchmarks/overview.md) · [methods index](index.md) · [interactive dashboard](../landing/dashboard.md)