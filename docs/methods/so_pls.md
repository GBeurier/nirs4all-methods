# `so_pls` — Sequential and Orthogonalised PLS (SO-PLS)

_Group_: **Multi-block / cross-modal** · _Registry tolerance_: `1.0`

## Description

SO-PLS — Sequential & Orthogonalized multi-block PLS (§17)

From the `pls4all.sklearn.SOPLSRegression` docstring:

> Sequential & Orthogonalised multi-block PLS (Næs et al. 2011).

> **Registry note** — R `multiblock::sopls 0.8.10` (Næs et al. 2011). Different deflation ordering than pls4all's SO-PLS; predictions diverge on synthetic block splits. Tolerance widened to flag external-ref presence.

### Parameters

| Name | Type | Default | Notes |
|------|------|---------|-------|
| `n_components_per_block` | `—` | `None` | Per-block latent-component budget (one int per block). |
| `block_sizes` | `—` | `None` | Sequence of contiguous block widths defining the X-block partition (columns of X). |

## Explanations

### Bibliographic source

Næs, T., Tomic, O., Mevik, B.-H. & Martens, H. (2011). *Path modelling by sequential PLS regression*. Journal of Chemometrics 25(1), 28–40.

### Mathematical principle

SO-PLS extends MB-PLS by sequentially processing blocks in a user-specified order, orthogonalising each subsequent block against the scores extracted from the previous ones. Concretely: fit PLS on block 1, extract scores $\mathbf{T}_1$; orthogonalise block 2 to $\mathbf{T}_1$, fit PLS on the residual; and so on.

This makes each block's contribution **additive and interpretable** — the unique variance in block $b$ that is predictive of $y$ given everything in blocks $1, \ldots, b-1$. Compared to MB-PLS (which fits all blocks simultaneously), SO-PLS encodes a domain hypothesis about which block is causally upstream of which.

Tuning: a per-block $k_b$ (number of components) plus the block order. The order matters; permuting blocks changes the attribution. Cross-validating the per-block $k_b$ is the standard procedure.

### Implementation

`p4a_so_pls_fit` — requires `n_components_per_block` and `block_sizes`. Reference: CRAN `multiblock 0.8.10`.

R roxygen note (`methods_extra.R::so_pls_fit`):

> Sequential & Orthogonalised multi-block PLS (Næs et al. 2011).
> `block_sizes` integer vector summing to ncol(X);
> `n_components_per_block` integer vector of same length.
> @param block_sizes Integer vector. Per-block feature counts for multi-block PLS.
> @param n_components_per_block Method-specific parameter. See the underlying `*_fit()` function for the exact semantics.
> @param X Numeric matrix of predictors (rows = samples, cols = features).
> @param Y Numeric matrix or vector of responses, with one row per sample.
> @export

MATLAB header (`bindings/matlab/+pls4all/so_pls.m`):

```text
pls4all.so_pls  Sequential & Orthogonalised multi-block PLS (Næs 2011).
 X: n × sum(block_sizes) concatenated multi-block matrix.
 n_components_per_block: int32 vector of length n_blocks.
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
p4a_so_pls_fit(ctx, cfg, &x_view, &y_view, /* hyperparams */, &res);
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
from pls4all._methods import so_pls_fit
with pls4all.Context() as ctx, pls4all.Config() as cfg:
    res = so_pls_fit(ctx, cfg, X, y)
# then: res.matrix("predictions"), res.matrix("coefficients"),
# res.vector("mask"), res.scalar("intercept"), …
```

:::

:::{tab-item} Python · pls4all.sklearn
:sync: python-sklearn
:class-label: lang-python

```python
from pls4all.sklearn import SOPLSRegression
mdl = SOPLSRegression(n_components_per_block=None, block_sizes=None)
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
res <- pls4all_method("so_pls", X, y,
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
res  <- so_pls_fit(X, Y, block_sizes, n_components_per_block)
yhat <- pls4all_predict(res, X_test)
```

:::

:::{tab-item} MATLAB · pls4all (MEX)
:sync: matlab-mex
:class-label: lang-matlab

```matlab
res = pls4all.so_pls(X, y, 2);
% see header of bindings/matlab/+pls4all/so_pls.m for full
% parameter surface:
%   res = so_pls(X, Y, n_components_per_block, block_sizes)
yhat = predict(res, Xtest);
```

:::

:::{tab-item} MATLAB · pls4all (classdef)
:sync: matlab-classdef
:class-label: lang-matlab

_No idiomatic classdef wrapper — invoke `pls4all.fit("so_pls", X, y, …)` directly from the unified MEX factory._

:::

::::


**Registry parity references** 📐

:::{card}
:class-card: external-refs

- 📐 **`ref.r_multiblock`** (R · r) — `multiblock` 0.8.10 · qualitative (rmse_rel ≤ 1e+00) — R `multiblock::sopls 0.8.10` (Næs et al. 2011). Different deflation ordering than pls4all's SO-PLS, so predictions diverge moderately on synthetic data — tolerance widened.
:::

### Validation contract

Structural validation contract for `so_pls`, exported by the declarative validation framework (`benchmarks/validation/registry/`) from `benchmarks.parity_timing.registry`. Regenerated by `python -m benchmarks.validation.export_registry --write`.

| Field | Value |
|------|-------|
| Reference kind | `external` |
| Canonical reference role | `r` |
| Declared references | R |
| Comparator | `reference_parity` (external-reference validity gate) |
| Comparator metric | `rmse_rel` |
| Reference tolerance | `1e+00` (qualitative) |
| Binding tolerance | `1e-06` |
| Prediction key | `predictions` |
| Required data flags | none |

**Benchmark cell parameters**

| Parameter | Value |
|-----------|-------|
| `n_blocks` | `3` |
| `n_components_per_block` | `[2, 2, 2]` |
| `n_features` | `30` |
| `n_samples` | `200` |
| `n_targets` | `2` |

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
<thead><tr><th scope="col">Backend</th><th scope="col">Divergence</th><th class="size-col" scope="col">100×50 (ms)</th><th class="size-col" scope="col">100×500 (ms)</th><th class="size-col" scope="col">100×2500 (ms)</th><th class="size-col" scope="col">1000×500 (ms)</th></tr></thead>
<tbody class="lang-band lang-cpp"><tr class="lang-band-row" data-lang="cpp"><th colspan="6" scope="rowgroup"><span class="lang-band-dot"></span>C++ native · libp4a</th></tr>
<tr class="bk-row"><td class="bk-name"><code>pls4all.cpp.blas</code></td><td class="parity parity-divergence parity-not_available">—</td><td class="ms">1.45 ms</td><td class="ms">12.0 ms</td><td class="ms ms-best">63.2 ms<span class="medal" title="fastest">🏆</span></td><td class="ms">140.2 ms</td></tr>
<tr class="bk-row"><td class="bk-name"><code>pls4all.cpp.blas+omp</code></td><td class="parity parity-divergence parity-divergent">1.37</td><td class="ms ms-best">1.40 ms<span class="medal" title="fastest">🏆</span></td><td class="ms">12.8 ms</td><td class="ms">65.5 ms</td><td class="ms">147.8 ms</td></tr>
<tr class="bk-row"><td class="bk-name"><code>pls4all.cpp.omp</code></td><td class="parity parity-divergence parity-not_available">—</td><td class="ms">1.70 ms</td><td class="ms ms-best">12.0 ms<span class="medal" title="fastest">🏆</span></td><td class="ms">63.9 ms</td><td class="ms ms-best">137.2 ms<span class="medal" title="fastest">🏆</span></td></tr>
<tr class="bk-row"><td class="bk-name"><code>pls4all.cpp.ref</code></td><td class="parity parity-divergence parity-not_available">—</td><td class="ms">1.70 ms</td><td class="ms">12.9 ms</td><td class="ms">66.1 ms</td><td class="ms">143.2 ms</td></tr>
</tbody>
<tbody class="lang-band lang-python"><tr class="lang-band-row" data-lang="python"><th colspan="6" scope="rowgroup"><span class="lang-band-dot"></span>Python · pls4all</th></tr>
<tr class="bk-row"><td class="bk-name"><code>pls4all.python</code></td><td class="parity parity-divergence parity-divergent">1.37</td><td class="ms">1.45 ms</td><td class="ms">13.0 ms</td><td class="ms">64.8 ms</td><td class="ms">145.9 ms</td></tr>
<tr class="bk-row"><td class="bk-name"><code>pls4all.sklearn</code></td><td class="parity parity-divergence parity-divergent">1.37</td><td class="ms">1.57 ms</td><td class="ms">13.8 ms</td><td class="ms">63.4 ms</td><td class="ms">139.6 ms</td></tr>
</tbody>
<tbody class="lang-band lang-r"><tr class="lang-band-row" data-lang="r"><th colspan="6" scope="rowgroup"><span class="lang-band-dot"></span>R · pls4all</th></tr>
<tr class="bk-row"><td class="bk-name"><code>pls4all.R</code></td><td class="parity parity-divergence parity-divergent">1.37</td><td class="ms">3.85 ms</td><td class="ms">49.9 ms</td><td class="ms">482.6 ms</td><td class="ms">578.2 ms</td></tr>
<tr class="bk-row"><td class="bk-name"><code>pls4all.R.formula</code></td><td class="parity parity-divergence parity-divergent">1.37</td><td class="ms">5.33 ms</td><td class="ms">74.4 ms</td><td class="ms">587.4 ms</td><td class="ms">800.6 ms</td></tr>
<tr class="bk-row"><td class="bk-name"><code>pls4all.R.mdatools</code></td><td class="parity parity-divergence parity-not_available">—</td><td class="ms">4.82 ms</td><td class="ms">78.4 ms</td><td class="ms">589.5 ms</td><td class="ms">748.7 ms</td></tr>
<tr class="bk-row"><td class="bk-name"><code>pls4all.R.pls</code></td><td class="parity parity-divergence parity-not_available">—</td><td class="ms">4.85 ms</td><td class="ms">77.2 ms</td><td class="ms">586.4 ms</td><td class="ms">764.8 ms</td></tr>
</tbody>
<tbody class="lang-band lang-matlab"><tr class="lang-band-row" data-lang="matlab"><th colspan="6" scope="rowgroup"><span class="lang-band-dot"></span>MATLAB · pls4all</th></tr>
<tr class="bk-row"><td class="bk-name"><code>pls4all.matlab</code></td><td class="parity parity-divergence parity-divergent">1.37</td><td class="ms">2.17 ms</td><td class="ms">18.1 ms</td><td class="ms">89.9 ms</td><td class="ms">188.8 ms</td></tr>
<tr class="bk-row"><td class="bk-name"><code>pls4all.matlab.classdef</code></td><td class="parity parity-divergence parity-divergent">1.37</td><td class="ms">2.45 ms</td><td class="ms">18.4 ms</td><td class="ms">89.4 ms</td><td class="ms">191.3 ms</td></tr>
</tbody>
<tbody class="lang-band lang-r"><tr class="lang-band-row" data-lang="r"><th colspan="6" scope="rowgroup"><span class="lang-band-dot"></span>R · external</th></tr>
<tr class="bk-row truth-source truth-source-qualitative"><td class="bk-name"><span class="truth-mark" title="Registry parity reference (r): multiblock 0.8.10 — qualitative (rmse_rel ≤ 1e+00)">📐</span><code>ref.r_multiblock</code></td><td class="parity parity-divergence parity-ref-source">0</td><td class="ms">1.0 s</td><td class="ms">1.2 s</td><td class="ms">1.5 s</td><td class="ms">3.2 s</td></tr>
</tbody>
</table>
</div>

:::

::::


---

_See also_: [benchmark overview](../benchmarks/overview.md) · [methods index](index.md) · [interactive dashboard](../landing/dashboard.md)