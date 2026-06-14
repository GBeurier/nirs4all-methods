# `group_sparse_pls` — Group-sparse PLS (Liquet 2016)

_Group_: **Sparse** · _Registry tolerance_: `1e-08`

## Description

Group sparse PLS (§7)

From the `pls4all.sklearn.GroupSparsePLSRegression` docstring:

> Group-sparse PLS — L1 across pre-declared feature groups.

> **Registry note** — R `sgPLS::gPLS` (Liquet et al. 2016, regression mode, scale=TRUE). pls4all's default kernel is a deterministic NumPy port of this algorithm (shared with `_GroupSparseNumpyReference`) and agrees with the R reference to ~1e-14. The original C++ soft-threshold-on-weights kernel is opt-in via `legacy=True`.

### Parameters

| Name | Type | Default | Notes |
|------|------|---------|-------|
| `n_components` | `int` | `2` | Number of latent components extracted (k). |
| `group_assignment` | `—` | `None` | Integer array assigning each feature to a group (length n_features). |
| `group_lambda` | `float` | `0.05` | L1 penalty applied at the group level (group-sparse PLS). |

## Explanations

### Bibliographic source

Liquet, B., de Micheaux, P. L., Hejblum, B. P. & Thiébaut, R. (2016). *Group and sparse group partial least squares approaches applied in genomics context*. Bioinformatics 32(1), 35–42.

### Mathematical principle

When features partition into known groups — gene pathways, spectroscopic bands, biological assays — group-sparse PLS forces entire groups in or out together via a group-lasso penalty: $\mathcal{P}(\mathbf{w}) = \sum_g \sqrt{|g|}\,\|\mathbf{w}_g\|_2$, where $\mathbf{w}_g$ is the sub-vector of weights belonging to group $g$ and $|g|$ is its size. The $\ell_2$ norm inside the sum is non-differentiable at zero, which produces group-level sparsity (an entire $\mathbf{w}_g$ is either zero or non-zero).

Compared to plain sparse PLS, this gives a much more interpretable model when groups have biological meaning and avoids the situation where one or two members of a co-regulated cluster get selected while the rest don't.

Required input: a `group_assignment` vector mapping each feature to a group id.

### Implementation

`n4m_estimators_group_sparse_pls_fit`. Reference: CRAN `sgPLS 1.8.1`.

R roxygen note (`methods_extra.R::group_sparse_pls_fit`):

> Group-sparse PLS (group L1 across feature groups).
> @param n_components Integer. Number of latent components.
> @param group_assignment Method-specific parameter. See the underlying `*_fit()` function for the exact semantics.
> @param group_lambda Method-specific parameter. See the underlying `*_fit()` function for the exact semantics.
> @param X Numeric matrix of predictors (rows = samples, cols = features).
> @param Y Numeric matrix or vector of responses, with one row per sample.
> @export

MATLAB header (`bindings/matlab/+pls4all/group_sparse_pls.m`):

```text
pls4all.group_sparse_pls  Group-sparse PLS (group L1 over feature groups).
```

### Usage

Every pls4all binding tab dispatches into the same C kernel; the external libraries listed at the bottom of the page are the parity references registered in `benchmarks.parity_timing.registry`. Switch tabs to read the same fit in your language. The R package now ships drop-in-compatible facades for the CRAN `pls` package (`plsr`, `pcr`, `mvr`) and for the `mdatools::pls(x, y, ...)` matrix idiom — those tabs appear only on the methods that have a meaningful equivalence.

**pls4all bindings**

::::{tab-set}
:class: pls4all-bindings

:::{tab-item} C ABI · libn4m
:sync: c
:class-label: lang-c

```c
/* C ABI — libn4m */
n4m_context_t* ctx = n4m_context_create();
n4m_config_t*  cfg = n4m_config_create();
n4m_method_result_t* res = NULL;
n4m_estimators_group_sparse_pls_fit(ctx, cfg, &x_view, &y_view, /* hyperparams */, &res);
/* … read coefficients / mask / scores via */
/* n4m_method_result_get_double_matrix / vector / scalar … */
n4m_method_result_destroy(res);
n4m_config_destroy(cfg);
n4m_context_destroy(ctx);
```

:::

:::{tab-item} Python · pls4all (raw)
:sync: python-raw
:class-label: lang-python

```python
import pls4all
from pls4all._methods import group_sparse_pls_fit
with pls4all.Context() as ctx, pls4all.Config() as cfg:
    res = group_sparse_pls_fit(ctx, cfg, X, y, n_components=4, group_assignment=groups)
# then: res.matrix("predictions"), res.matrix("coefficients"),
# res.vector("mask"), res.scalar("intercept"), …
```

:::

:::{tab-item} Python · pls4all.sklearn
:sync: python-sklearn
:class-label: lang-python

```python
from pls4all.sklearn import GroupSparsePLSRegression
mdl = GroupSparsePLSRegression(n_components=2, group_assignment=None, group_lambda=0.05)
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
res <- pls4all_method("group_sparse_pls", X, y,
                      n_components = 4L, params = list(group_lambda = 0.1))
# res is a named list with MethodResult arrays/scalars.
# selected_indices / top_k_intervals are 1-based.
```

:::

:::{tab-item} R · pls4all (raw fn)
:sync: r-raw
:class-label: lang-r

```r
library(pls4all)
res  <- group_sparse_pls_fit(X, Y, n_components, group_assignment,
                      group_lambda = 0.05)
yhat <- pls4all_predict(res, X_test)
```

:::

:::{tab-item} MATLAB · pls4all (MEX)
:sync: matlab-mex
:class-label: lang-matlab

```matlab
res = pls4all.group_sparse_pls(X, y, 4);
% see header of bindings/matlab/+pls4all/group_sparse_pls.m for full
% parameter surface:
%   res = group_sparse_pls(X, Y, n_components, group_assignment, group_lambda)
yhat = predict(res, Xtest);
```

:::

:::{tab-item} MATLAB · pls4all (classdef)
:sync: matlab-classdef
:class-label: lang-matlab

_No idiomatic classdef wrapper — invoke `pls4all.fit("group_sparse_pls", X, y, …)` directly from the unified MEX factory._

:::

::::


**Registry parity references** 📐

:::{card}
:class-card: external-refs

- 📐 **`ref.python_numpy`** (python · python) — `numpy` in-tree · strict (rmse_rel ≤ 1e-08) — In-tree NumPy port of Liquet et al. 2016 group sparse PLS (R `sgPLS::gPLS`, regression mode, scale=TRUE). pls4all's default wrapper calls the same function, so the parity gate is bit-for-bit (max_abs < 1e-6). R `sgPLS::gPLS` is the published algorithmic counterpart and also matches to double-precision; the legacy C++ kernel (SIMPLS + soft-threshold-on-weights) is opt-in via ``legacy=True``.
- 📐 **`ref.r_sgpls`** (R · r) — `sgPLS` 1.8.1 · strict (rmse_rel ≤ 1e-08) — R `sgPLS::gPLS(X, Y, ncomp, ind.block.x, keepX=length(bnd))` (regression, scale=TRUE). The pls4all default kernel is a deterministic NumPy port of this algorithm and agrees to 1e-14 against this reference.
:::

### Benchmarks

Adaptive wall-clock per cell measured against [`full_matrix.csv`](../benchmarks/overview.md). Only backends that implement this method are listed; libraries without the method are omitted.

**Verdict** &nbsp;·&nbsp; ✓ ref / ≈ ref / ~ shape mark a reference-gate pass at strict / relaxed / qualitative tolerance &nbsp;·&nbsp; ✓ bind = pls4all binding agrees with the C++ baseline &nbsp;·&nbsp; ⇄ cross-check = documented by-design selector/RNG/model, noncanonical API/facade convention, or secondary oracle &nbsp;·&nbsp; ✗ divergent &nbsp;·&nbsp; ⚠ error &nbsp;·&nbsp; — not run. The fastest backend per column is marked 🏆.

**Reference gate**: strict — numeric equivalence (`rmse_rel_tol ≤ 1e-08`).

Rows tagged with **📐** are the canonical parity references for this method (declared in [`parity_timing.registry`](../benchmarks/methodology.md)). C++ and external rows show reference parity; pls4all language bindings show binding parity against the C++ backend. Hover the icon for role and tolerance band.

::::{tab-set}
:class: parity-tabs

:::{tab-item} 1 thread
:sync: threads-1

<div class="parity-table-wrap">
<table class="docutils parity-grouped">
<thead><tr><th scope="col">Backend</th><th scope="col">Parity</th><th class="size-col" scope="col">200×50 (ms)</th></tr></thead>
<tbody class="lang-band lang-cpp"><tr class="lang-band-row" data-lang="cpp"><th colspan="3" scope="rowgroup"><span class="lang-band-dot"></span>C++ native · libn4m</th></tr>
<tr class="bk-row"><td class="bk-name"><code>pls4all.cpp.blas+omp</code></td><td class="parity parity-ref-strict">✓ ref</td><td class="ms">8.36 ms</td></tr>
</tbody>
<tbody class="lang-band lang-python"><tr class="lang-band-row" data-lang="python"><th colspan="3" scope="rowgroup"><span class="lang-band-dot"></span>Python · pls4all</th></tr>
<tr class="bk-row"><td class="bk-name"><code>pls4all.python</code></td><td class="parity parity-exact">✓ bind</td><td class="ms">7.86 ms</td></tr>
<tr class="bk-row"><td class="bk-name"><code>pls4all.sklearn</code></td><td class="parity parity-cross_check">⇄ +1e-01</td><td class="ms">3.79 ms</td></tr>
</tbody>
<tbody class="lang-band lang-r"><tr class="lang-band-row" data-lang="r"><th colspan="3" scope="rowgroup"><span class="lang-band-dot"></span>R · pls4all</th></tr>
<tr class="bk-row"><td class="bk-name"><code>pls4all.R</code></td><td class="parity parity-cross_check">⇄ +1e-01</td><td class="ms">9.70 ms</td></tr>
<tr class="bk-row"><td class="bk-name"><code>pls4all.R.formula</code></td><td class="parity parity-cross_check">⇄ +1e-01</td><td class="ms">11.0 ms</td></tr>
<tr class="bk-row"><td class="bk-name"><code>pls4all.R.mdatools</code></td><td class="parity parity-cross_check">⇄ +1e-01</td><td class="ms">10.2 ms</td></tr>
<tr class="bk-row"><td class="bk-name"><code>pls4all.R.pls</code></td><td class="parity parity-cross_check">⇄ +1e-01</td><td class="ms">11.0 ms</td></tr>
</tbody>
<tbody class="lang-band lang-python"><tr class="lang-band-row" data-lang="python"><th colspan="3" scope="rowgroup"><span class="lang-band-dot"></span>Python · external</th></tr>
<tr class="bk-row truth-source truth-source-strict"><td class="bk-name"><span class="truth-mark" title="Registry parity reference (python): numpy in-tree — strict (rmse_rel ≤ 1e-08)">📐</span><code>ref.python_numpy</code></td><td class="parity parity-ref-source">source</td><td class="ms ms-best">2.98 ms<span class="medal" title="fastest">🏆</span></td></tr>
</tbody>
<tbody class="lang-band lang-r"><tr class="lang-band-row" data-lang="r"><th colspan="3" scope="rowgroup"><span class="lang-band-dot"></span>R · external</th></tr>
<tr class="bk-row truth-source truth-source-strict"><td class="bk-name"><span class="truth-mark" title="Registry parity reference (r): sgPLS 1.8.1 — strict (rmse_rel ≤ 1e-08)">📐</span><code>ref.r_sgpls</code></td><td class="parity parity-cross_check">⇄ +1e-15</td><td class="ms">33.7 ms</td></tr>
</tbody>
</table>
</div>

:::

:::{tab-item} 3 threads
:sync: threads-3

<div class="parity-table-wrap">
<table class="docutils parity-grouped">
<thead><tr><th scope="col">Backend</th><th scope="col">Parity</th><th class="size-col" scope="col">200×50 (ms)</th></tr></thead>
<tbody class="lang-band lang-cpp"><tr class="lang-band-row" data-lang="cpp"><th colspan="3" scope="rowgroup"><span class="lang-band-dot"></span>C++ native · libn4m</th></tr>
<tr class="bk-row"><td class="bk-name"><code>pls4all.cpp.blas+omp</code></td><td class="parity parity-ref-strict">✓ ref</td><td class="ms">2.62 ms</td></tr>
</tbody>
<tbody class="lang-band lang-python"><tr class="lang-band-row" data-lang="python"><th colspan="3" scope="rowgroup"><span class="lang-band-dot"></span>Python · pls4all</th></tr>
<tr class="bk-row"><td class="bk-name"><code>pls4all.python</code></td><td class="parity parity-exact">✓ bind</td><td class="ms">2.76 ms</td></tr>
<tr class="bk-row"><td class="bk-name"><code>pls4all.sklearn</code></td><td class="parity parity-cross_check">⇄ +1e-01</td><td class="ms ms-best">2.28 ms<span class="medal" title="fastest">🏆</span></td></tr>
</tbody>
<tbody class="lang-band lang-r"><tr class="lang-band-row" data-lang="r"><th colspan="3" scope="rowgroup"><span class="lang-band-dot"></span>R · pls4all</th></tr>
<tr class="bk-row"><td class="bk-name"><code>pls4all.R</code></td><td class="parity parity-cross_check">⇄ +1e-01</td><td class="ms">8.78 ms</td></tr>
<tr class="bk-row"><td class="bk-name"><code>pls4all.R.formula</code></td><td class="parity parity-cross_check">⇄ +1e-01</td><td class="ms">9.80 ms</td></tr>
<tr class="bk-row"><td class="bk-name"><code>pls4all.R.mdatools</code></td><td class="parity parity-cross_check">⇄ +1e-01</td><td class="ms">11.9 ms</td></tr>
<tr class="bk-row"><td class="bk-name"><code>pls4all.R.pls</code></td><td class="parity parity-cross_check">⇄ +1e-01</td><td class="ms">10.5 ms</td></tr>
</tbody>
<tbody class="lang-band lang-python"><tr class="lang-band-row" data-lang="python"><th colspan="3" scope="rowgroup"><span class="lang-band-dot"></span>Python · external</th></tr>
<tr class="bk-row truth-source truth-source-strict"><td class="bk-name"><span class="truth-mark" title="Registry parity reference (python): numpy in-tree — strict (rmse_rel ≤ 1e-08)">📐</span><code>ref.python_numpy</code></td><td class="parity parity-ref-source">source</td><td class="ms">3.11 ms</td></tr>
</tbody>
<tbody class="lang-band lang-r"><tr class="lang-band-row" data-lang="r"><th colspan="3" scope="rowgroup"><span class="lang-band-dot"></span>R · external</th></tr>
<tr class="bk-row truth-source truth-source-strict"><td class="bk-name"><span class="truth-mark" title="Registry parity reference (r): sgPLS 1.8.1 — strict (rmse_rel ≤ 1e-08)">📐</span><code>ref.r_sgpls</code></td><td class="parity parity-cross_check">⇄ +1e-15</td><td class="ms">31.2 ms</td></tr>
</tbody>
</table>
</div>

:::

:::{tab-item} 10 threads
:sync: threads-10

<div class="parity-table-wrap">
<table class="docutils parity-grouped">
<thead><tr><th scope="col">Backend</th><th scope="col">Parity</th><th class="size-col" scope="col">200×50 (ms)</th></tr></thead>
<tbody class="lang-band lang-cpp"><tr class="lang-band-row" data-lang="cpp"><th colspan="3" scope="rowgroup"><span class="lang-band-dot"></span>C++ native · libn4m</th></tr>
<tr class="bk-row"><td class="bk-name"><code>pls4all.cpp.blas+omp</code></td><td class="parity parity-ref-strict">✓ ref</td><td class="ms">2.91 ms</td></tr>
</tbody>
<tbody class="lang-band lang-python"><tr class="lang-band-row" data-lang="python"><th colspan="3" scope="rowgroup"><span class="lang-band-dot"></span>Python · pls4all</th></tr>
<tr class="bk-row"><td class="bk-name"><code>pls4all.python</code></td><td class="parity parity-exact">✓ bind</td><td class="ms">2.69 ms</td></tr>
<tr class="bk-row"><td class="bk-name"><code>pls4all.sklearn</code></td><td class="parity parity-cross_check">⇄ +1e-01</td><td class="ms ms-best">2.53 ms<span class="medal" title="fastest">🏆</span></td></tr>
</tbody>
<tbody class="lang-band lang-r"><tr class="lang-band-row" data-lang="r"><th colspan="3" scope="rowgroup"><span class="lang-band-dot"></span>R · pls4all</th></tr>
<tr class="bk-row"><td class="bk-name"><code>pls4all.R</code></td><td class="parity parity-cross_check">⇄ +1e-01</td><td class="ms">7.45 ms</td></tr>
<tr class="bk-row"><td class="bk-name"><code>pls4all.R.formula</code></td><td class="parity parity-cross_check">⇄ +1e-01</td><td class="ms">9.19 ms</td></tr>
<tr class="bk-row"><td class="bk-name"><code>pls4all.R.mdatools</code></td><td class="parity parity-cross_check">⇄ +1e-01</td><td class="ms">11.3 ms</td></tr>
<tr class="bk-row"><td class="bk-name"><code>pls4all.R.pls</code></td><td class="parity parity-cross_check">⇄ +1e-01</td><td class="ms">8.50 ms</td></tr>
</tbody>
<tbody class="lang-band lang-python"><tr class="lang-band-row" data-lang="python"><th colspan="3" scope="rowgroup"><span class="lang-band-dot"></span>Python · external</th></tr>
<tr class="bk-row truth-source truth-source-strict"><td class="bk-name"><span class="truth-mark" title="Registry parity reference (python): numpy in-tree — strict (rmse_rel ≤ 1e-08)">📐</span><code>ref.python_numpy</code></td><td class="parity parity-ref-source">source</td><td class="ms">2.66 ms</td></tr>
</tbody>
<tbody class="lang-band lang-r"><tr class="lang-band-row" data-lang="r"><th colspan="3" scope="rowgroup"><span class="lang-band-dot"></span>R · external</th></tr>
<tr class="bk-row truth-source truth-source-strict"><td class="bk-name"><span class="truth-mark" title="Registry parity reference (r): sgPLS 1.8.1 — strict (rmse_rel ≤ 1e-08)">📐</span><code>ref.r_sgpls</code></td><td class="parity parity-cross_check">⇄ +1e-15</td><td class="ms">26.7 ms</td></tr>
</tbody>
</table>
</div>

:::

::::


---

_See also_: [benchmark overview](../benchmarks/overview.md) · [methods index](index.md) · [interactive dashboard](../landing/dashboard.md)