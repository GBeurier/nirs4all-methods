# `group_sparse_pls` — Group-sparse PLS (Liquet 2016)

_Group_: **Sparse** · _Registry tolerance_: `10.0`

## Description

Group sparse PLS (§7)

From the `pls4all.sklearn.GroupSparsePLSRegression` docstring:

> Group-sparse PLS — L1 across pre-declared feature groups.

> **Registry note** — R `sgPLS::gPLS` (Liquet et al. 2016) — group-sparse PLS via group lasso penalty. pls4all's group_sparse_pls uses a different group-penalty formulation; widened tolerance to flag external-ref presence.

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

`p4a_group_sparse_pls_fit`. Reference: CRAN `sgPLS 1.8.1`.

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

:::{tab-item} C ABI · libp4a
:sync: c
:class-label: lang-c

```c
/* C ABI — libp4a */
p4a_context_t* ctx = p4a_context_create();
p4a_config_t*  cfg = p4a_config_create();
p4a_method_result_t* res = NULL;
p4a_group_sparse_pls_fit(ctx, cfg, &x_view, &y_view, /* hyperparams */, &res);
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

- 📐 **`ref.r_sgpls`** (R · r) — `sgPLS` 1.8.1 · qualitative (rmse_rel ≤ 1e+01) — R `sgPLS::gPLS(X, Y, ncomp, ind.block.x, keepX)` — group lasso penalized PLS. Different penalty formulation than pls4all's group_sparse_pls; qualitative parity, wide tol.
:::

### Validation contract

Structural validation contract for `group_sparse_pls`, exported by the declarative validation framework (`benchmarks/validation/registry/`) from `benchmarks.parity_timing.registry`. Regenerated by `python -m benchmarks.validation.export_registry --write`.

| Field | Value |
|------|-------|
| Reference kind | `external` |
| Canonical reference role | `r` |
| Declared references | R |
| Comparator | `reference_parity` (external-reference validity gate) |
| Comparator metric | `rmse_rel` |
| Reference tolerance | `1e+01` (qualitative) |
| Binding tolerance | `1e-06` |
| Prediction key | `predictions` |
| Required data flags | `group_assignment` |

**Benchmark cell parameters**

| Parameter | Value |
|-----------|-------|
| `group_lambda` | `0.1` |
| `n_components` | `4` |
| `n_features` | `50` |
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
<tr class="bk-row"><td class="bk-name"><code>pls4all.cpp.blas</code></td><td class="parity parity-divergence parity-divergent">5.6e-02</td><td class="ms">1.00 ms</td><td class="ms">8.44 ms</td><td class="ms">41.5 ms</td><td class="ms">86.0 ms</td></tr>
<tr class="bk-row"><td class="bk-name"><code>pls4all.cpp.blas+omp</code></td><td class="parity parity-divergence parity-divergent">5.6e-02</td><td class="ms">0.99 ms</td><td class="ms">8.25 ms</td><td class="ms">42.1 ms</td><td class="ms">85.3 ms</td></tr>
<tr class="bk-row"><td class="bk-name"><code>pls4all.cpp.omp</code></td><td class="parity parity-divergence parity-divergent">5.6e-02</td><td class="ms ms-best">0.98 ms<span class="medal" title="fastest">🏆</span></td><td class="ms">8.27 ms</td><td class="ms ms-best">41.2 ms<span class="medal" title="fastest">🏆</span></td><td class="ms">84.9 ms</td></tr>
<tr class="bk-row"><td class="bk-name"><code>pls4all.cpp.ref</code></td><td class="parity parity-divergence parity-divergent">5.6e-02</td><td class="ms">1.04 ms</td><td class="ms">8.48 ms</td><td class="ms">41.5 ms</td><td class="ms">83.7 ms</td></tr>
</tbody>
<tbody class="lang-band lang-python"><tr class="lang-band-row" data-lang="python"><th colspan="6" scope="rowgroup"><span class="lang-band-dot"></span>Python · pls4all</th></tr>
<tr class="bk-row"><td class="bk-name"><code>pls4all.python</code></td><td class="parity parity-divergence parity-divergent">5.6e-02</td><td class="ms">1.00 ms</td><td class="ms ms-best">8.22 ms<span class="medal" title="fastest">🏆</span></td><td class="ms">41.6 ms</td><td class="ms ms-best">83.3 ms<span class="medal" title="fastest">🏆</span></td></tr>
<tr class="bk-row"><td class="bk-name"><code>pls4all.sklearn</code></td><td class="parity parity-divergence parity-divergent">5.6e-02</td><td class="ms">1.19 ms</td><td class="ms">9.04 ms</td><td class="ms">42.5 ms</td><td class="ms">85.4 ms</td></tr>
</tbody>
<tbody class="lang-band lang-r"><tr class="lang-band-row" data-lang="r"><th colspan="6" scope="rowgroup"><span class="lang-band-dot"></span>R · pls4all</th></tr>
<tr class="bk-row"><td class="bk-name"><code>pls4all.R</code></td><td class="parity parity-divergence parity-divergent">5.6e-02</td><td class="ms">3.44 ms</td><td class="ms">55.0 ms</td><td class="ms">419.4 ms</td><td class="ms">489.8 ms</td></tr>
<tr class="bk-row"><td class="bk-name"><code>pls4all.R.formula</code></td><td class="parity parity-divergence parity-divergent">5.6e-02</td><td class="ms">5.07 ms</td><td class="ms">66.0 ms</td><td class="ms">577.0 ms</td><td class="ms">638.8 ms</td></tr>
<tr class="bk-row"><td class="bk-name"><code>pls4all.R.mdatools</code></td><td class="parity parity-divergence parity-drift">—</td><td class="ms">4.69 ms</td><td class="ms">65.9 ms</td><td class="ms">558.6 ms</td><td class="ms">629.2 ms</td></tr>
<tr class="bk-row"><td class="bk-name"><code>pls4all.R.pls</code></td><td class="parity parity-divergence parity-drift">—</td><td class="ms">5.39 ms</td><td class="ms">58.7 ms</td><td class="ms">550.2 ms</td><td class="ms">603.9 ms</td></tr>
</tbody>
<tbody class="lang-band lang-matlab"><tr class="lang-band-row" data-lang="matlab"><th colspan="6" scope="rowgroup"><span class="lang-band-dot"></span>MATLAB · pls4all</th></tr>
<tr class="bk-row"><td class="bk-name"><code>pls4all.matlab</code></td><td class="parity parity-divergence parity-divergent">5.6e-02</td><td class="ms">1.79 ms</td><td class="ms">13.3 ms</td><td class="ms">63.8 ms</td><td class="ms">132.6 ms</td></tr>
<tr class="bk-row"><td class="bk-name"><code>pls4all.matlab.classdef</code></td><td class="parity parity-divergence parity-divergent">5.6e-02</td><td class="ms">2.04 ms</td><td class="ms">13.9 ms</td><td class="ms">66.8 ms</td><td class="ms">135.4 ms</td></tr>
</tbody>
<tbody class="lang-band lang-r"><tr class="lang-band-row" data-lang="r"><th colspan="6" scope="rowgroup"><span class="lang-band-dot"></span>R · external</th></tr>
<tr class="bk-row truth-source truth-source-qualitative"><td class="bk-name"><span class="truth-mark" title="Registry parity reference (r): sgPLS 1.8.1 — qualitative (rmse_rel ≤ 1e+01)">📐</span><code>ref.r_sgpls</code></td><td class="parity parity-divergence parity-ref-source">0</td><td class="ms">17.6 ms</td><td class="ms">113.3 ms</td><td class="ms">1.1 s</td><td class="ms">1.1 s</td></tr>
</tbody>
</table>
</div>

:::

::::


---

_See also_: [benchmark overview](../benchmarks/overview.md) · [methods index](index.md) · [interactive dashboard](../landing/dashboard.md)