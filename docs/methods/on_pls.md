# `on_pls` — OnPLS (Orthogonal N-block PLS)

_Group_: **Multi-block / cross-modal** · _Registry tolerance_: `10.0`

## Description

OnPLS — Orthogonal multi-block decomposition (§18)

> **Registry note** — Python `OnPLS` (tomlof/OnPLS, vendored in bindings/python/vendor/OnPLS). Canonical Löfstedt & Trygg 2011. Sign/rotation freedom in joint loadings inflates divergence; widened tol.

_No tunable parameters declared at the binding level._

## Explanations

### Bibliographic source

Löfstedt, T. & Trygg, J. (2011). *OnPLS — a novel multiblock method for the modelling of predictive and orthogonal variation*. Journal of Chemometrics 25(8), 441–455.

### Mathematical principle

OnPLS generalises OPLS to multiple blocks: it decomposes the joint structure into a globally predictive component shared by all blocks plus block-unique orthogonal components per block. This separates 'integrated' biology / chemistry information from block-specific noise.

The procedure iteratively refines a joint component by alternating projections and orthogonalisations across blocks. Compared to SO-PLS — which is asymmetric in block order — OnPLS is **symmetric**: no causal directionality is implied between blocks. This is the right choice when blocks are observation modalities of the same underlying process (e.g. transcriptomics + metabolomics + proteomics on the same biological samples).

The CRAN `OnPLS` package was archived in 2024 so the Python implementation is vendored at `bindings/python/vendor/OnPLS/` to remove the dependency.

### Implementation

`p4a_on_pls_fit` — requires `n_joint`, `n_unique_per_block`, `block_sizes`. The CRAN OnPLS package is archived; pls4all carries an in-tree vendored port for the parity reference.

R roxygen note (`methods_extra.R::on_pls_fit`):

> OnPLS — Orthogonal multi-block PLS (joint + unique loadings).
> @param n_joint Method-specific parameter. See the underlying `*_fit()` function for the exact semantics.
> @param n_unique_per_block Method-specific parameter. See the underlying `*_fit()` function for the exact semantics.
> @param block_sizes Integer vector. Per-block feature counts for multi-block PLS.
> @param X Numeric matrix of predictors (rows = samples, cols = features).
> @param Y Numeric matrix or vector of responses, with one row per sample.
> @export

MATLAB header (`bindings/matlab/+pls4all/on_pls.m`):

```text
pls4all.on_pls  Orthogonal multi-block PLS (joint + unique loadings).
 n_components arg is unused by on_pls (it has its own block structure),
 but the dispatcher still requires it; we pass n_joint.
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
p4a_on_pls_fit(ctx, cfg, &x_view, &y_view, /* hyperparams */, &res);
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
from pls4all._methods import on_pls_fit
with pls4all.Context() as ctx, pls4all.Config() as cfg:
    res = on_pls_fit(ctx, cfg, X, y)
# then: res.matrix("predictions"), res.matrix("coefficients"),
# res.vector("mask"), res.scalar("intercept"), …
```

:::

:::{tab-item} Python · pls4all.sklearn
:sync: python-sklearn
:class-label: lang-python

```python
from pls4all.sklearn import on_pls
result = on_pls(X, y, n_components=2)
```

:::

:::{tab-item} R · pls4all_method()
:sync: r-dispatcher
:class-label: lang-r

```r
library(pls4all)
# Unified low-level dispatcher (May 2026 R cleanup):
res <- pls4all_method("on_pls", X, y,
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
res  <- on_pls_fit(X, Y, n_joint, n_unique_per_block, block_sizes)
yhat <- pls4all_predict(res, X_test)
```

:::

:::{tab-item} MATLAB · pls4all (MEX)
:sync: matlab-mex
:class-label: lang-matlab

```matlab
res = pls4all.on_pls(X, y, 2);
% see header of bindings/matlab/+pls4all/on_pls.m for full
% parameter surface:
%   res = on_pls(X, Y, n_joint, n_unique_per_block, block_sizes)
yhat = predict(res, Xtest);
```

:::

:::{tab-item} MATLAB · pls4all (classdef)
:sync: matlab-classdef
:class-label: lang-matlab

_No idiomatic classdef wrapper — invoke `pls4all.fit("on_pls", X, y, …)` directly from the unified MEX factory._

:::

::::


**Registry parity references** 📐

:::{card}
:class-card: external-refs

- 📐 **`ref.python_onpls`** (python · python) — `OnPLS` github tomlof/OnPLS · qualitative (rmse_rel ≤ 1e+01) — Python `OnPLS` (Löfstedt & Trygg 2011). Vendored from GitHub because R `multiblock 0.8.10` lacks `onpls`. Joint loadings have sign/rotation freedom; tolerance wide enough to flag external-ref presence.
:::

### Validation contract

Structural validation contract for `on_pls`, exported by the declarative validation framework (`benchmarks/validation/registry/`) from `benchmarks.parity_timing.registry`. Regenerated by `python -m benchmarks.validation.export_registry --write`.

| Field | Value |
|------|-------|
| Reference kind | `external` |
| Canonical reference role | `python` |
| Declared references | Python |
| Comparator | `reference_parity` (external-reference validity gate) |
| Comparator metric | `rmse_rel` |
| Reference tolerance | `1e+01` (qualitative) |
| Binding tolerance | `1e-06` |
| Prediction key | `joint_loadings_0` |
| Required data flags | none |

**Benchmark cell parameters**

| Parameter | Value |
|-----------|-------|
| `n_blocks` | `3` |
| `n_features` | `30` |
| `n_joint` | `2` |
| `n_samples` | `200` |
| `n_targets` | `2` |
| `n_unique_per_block` | `[1, 1, 1]` |

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
<tr class="bk-row"><td class="bk-name"><code>pls4all.cpp.blas</code></td><td class="parity parity-divergence parity-divergent">1.38</td><td class="ms">1.16 ms</td><td class="ms">18.4 ms</td><td class="ms ms-best">405.2 ms<span class="medal" title="fastest">🏆</span></td><td class="ms ms-best">170.5 ms<span class="medal" title="fastest">🏆</span></td></tr>
<tr class="bk-row"><td class="bk-name"><code>pls4all.cpp.blas+omp</code></td><td class="parity parity-divergence parity-divergent">1.38</td><td class="ms ms-best">1.11 ms<span class="medal" title="fastest">🏆</span></td><td class="ms">19.5 ms</td><td class="ms">442.6 ms</td><td class="ms">173.8 ms</td></tr>
<tr class="bk-row"><td class="bk-name"><code>pls4all.cpp.omp</code></td><td class="parity parity-divergence parity-divergent">1.38</td><td class="ms">1.12 ms</td><td class="ms ms-best">18.3 ms<span class="medal" title="fastest">🏆</span></td><td class="ms">406.3 ms</td><td class="ms">177.5 ms</td></tr>
<tr class="bk-row"><td class="bk-name"><code>pls4all.cpp.ref</code></td><td class="parity parity-divergence parity-divergent">1.38</td><td class="ms">1.12 ms</td><td class="ms">19.6 ms</td><td class="ms">446.6 ms</td><td class="ms">172.5 ms</td></tr>
</tbody>
<tbody class="lang-band lang-python"><tr class="lang-band-row" data-lang="python"><th colspan="6" scope="rowgroup"><span class="lang-band-dot"></span>Python · pls4all</th></tr>
<tr class="bk-row"><td class="bk-name"><code>pls4all.python</code></td><td class="parity parity-divergence parity-divergent">1.38</td><td class="ms">1.15 ms</td><td class="ms">19.6 ms</td><td class="ms">462.1 ms</td><td class="ms">175.2 ms</td></tr>
<tr class="bk-row"><td class="bk-name"><code>pls4all.sklearn</code></td><td class="parity parity-divergence parity-divergent">1.38</td><td class="ms">1.13 ms</td><td class="ms">19.5 ms</td><td class="ms">453.5 ms</td><td class="ms">172.4 ms</td></tr>
</tbody>
<tbody class="lang-band lang-r"><tr class="lang-band-row" data-lang="r"><th colspan="6" scope="rowgroup"><span class="lang-band-dot"></span>R · pls4all</th></tr>
<tr class="bk-row"><td class="bk-name"><code>pls4all.R</code></td><td class="parity parity-divergence parity-divergent">1.38</td><td class="ms">3.33 ms</td><td class="ms">56.8 ms</td><td class="ms">859.4 ms</td><td class="ms">657.4 ms</td></tr>
<tr class="bk-row"><td class="bk-name"><code>pls4all.R.formula</code></td><td class="parity parity-divergence parity-divergent">1.38</td><td class="ms">4.79 ms</td><td class="ms">84.5 ms</td><td class="ms">1.0 s</td><td class="ms">811.3 ms</td></tr>
<tr class="bk-row"><td class="bk-name"><code>pls4all.R.mdatools</code></td><td class="parity parity-divergence parity-divergent">1.20</td><td class="ms">4.86 ms</td><td class="ms">79.7 ms</td><td class="ms">960.7 ms</td><td class="ms">826.6 ms</td></tr>
<tr class="bk-row"><td class="bk-name"><code>pls4all.R.pls</code></td><td class="parity parity-divergence parity-divergent">1.20</td><td class="ms">6.95 ms</td><td class="ms">83.3 ms</td><td class="ms">1.0 s</td><td class="ms">788.6 ms</td></tr>
</tbody>
<tbody class="lang-band lang-matlab"><tr class="lang-band-row" data-lang="matlab"><th colspan="6" scope="rowgroup"><span class="lang-band-dot"></span>MATLAB · pls4all</th></tr>
<tr class="bk-row"><td class="bk-name"><code>pls4all.matlab</code></td><td class="parity parity-divergence parity-divergent">1.38</td><td class="ms">1.80 ms</td><td class="ms">24.4 ms</td><td class="ms">445.5 ms</td><td class="ms">219.8 ms</td></tr>
<tr class="bk-row"><td class="bk-name"><code>pls4all.matlab.classdef</code></td><td class="parity parity-divergence parity-divergent">1.38</td><td class="ms">2.25 ms</td><td class="ms">25.1 ms</td><td class="ms">457.1 ms</td><td class="ms">227.2 ms</td></tr>
</tbody>
<tbody class="lang-band lang-python"><tr class="lang-band-row" data-lang="python"><th colspan="6" scope="rowgroup"><span class="lang-band-dot"></span>Python · external</th></tr>
<tr class="bk-row truth-source truth-source-qualitative"><td class="bk-name"><span class="truth-mark" title="Registry parity reference (python): OnPLS github tomlof/OnPLS — qualitative (rmse_rel ≤ 1e+01)">📐</span><code>ref.python_onpls</code></td><td class="parity parity-divergence parity-ref-source">0</td><td class="ms">9.00 ms</td><td class="ms">50.8 ms</td><td class="ms">783.8 ms</td><td class="ms">800.3 ms</td></tr>
</tbody>
</table>
</div>

:::

::::


---

_See also_: [benchmark overview](../benchmarks/overview.md) · [methods index](index.md) · [interactive dashboard](../landing/dashboard.md)