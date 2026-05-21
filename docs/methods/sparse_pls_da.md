# `sparse_pls_da` — Sparse PLS-DA (Lê Cao 2008)

_Group_: **Sparse** · _Registry tolerance_: `2.0`

## Description

Sparse PLS-DA (§7)

From the `pls4all.sklearn.SparsePLSDAClassifier` docstring:

> Sparse PLS-DA classifier.

> **Registry note** — R `spls::splsda` returns hard class labels; pls4all returns soft dummy-encoded scores. Tolerance widened to admit the soft-vs-hard scoring difference.

### Parameters

| Name | Type | Default | Notes |
|------|------|---------|-------|
| `n_components` | `int` | `2` | Number of latent components extracted (k). |
| `sparsity_lambda` | `float` | `0.05` | L1 soft-threshold magnitude applied to the PLS weight vectors. |
| `n_classes` | `int` | `3` | registry benchmark cell value |

## Explanations

### Bibliographic source

Lê Cao, K.-A., Rossouw, D., Robert-Granié, C. & Besse, P. (2008). *A sparse PLS for variable selection when integrating omics data*. Statistical Applications in Genetics and Molecular Biology 7(1).

### Mathematical principle

Discriminant variant of sparse PLS. Encode class labels $y \in \{0, 1, \ldots, C-1\}$ as a one-hot matrix $\mathbf{Y} \in \{0, 1\}^{n \times C}$, fit a sparse PLS regression on it, then assign new samples to the class with the largest predicted score. The L1 penalty selects a discriminative subset of features along each latent direction.

In high-dimensional biomarker discovery (microarray, MALDI-TOF, NIR food classification) sparse PLS-DA is a standard since it simultaneously builds the discriminant and shortlists the candidate markers in a single regularised fit. Class probabilities follow from a softmax over the predicted score columns.

### Implementation

`p4a_sparse_pls_da_fit`. Reference: Bioconductor `mixOmics::splsda`.

R roxygen note (`methods_extra.R::sparse_pls_da_fit`):

> Sparse PLS-DA classifier (`y_labels` is an integer vector of class IDs).
> @param y_labels Integer vector. Class labels.
> @param n_components Integer. Number of latent components.
> @param sparsity_lambda Method-specific parameter. See the underlying `*_fit()` function for the exact semantics.
> @param X Numeric matrix of predictors (rows = samples, cols = features).
> @export

MATLAB header (`bindings/matlab/+pls4all/sparse_pls_da.m`):

```text
pls4all.sparse_pls_da  Sparse PLS-DA classifier (Chun & Keles 2010 + DA).
 y_labels: integer class IDs in {0, …, n_classes-1}.
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
p4a_sparse_simpls_fit(ctx, cfg, &x_view, &y_view, /* hyperparams */, &res);
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
from pls4all._methods import sparse_simpls_fit
with pls4all.Context() as ctx, pls4all.Config() as cfg:
    res = sparse_simpls_fit(ctx, cfg, X, y, n_components=4, y_labels=y_labels)
# then: res.matrix("predictions"), res.matrix("coefficients"),
# res.vector("mask"), res.scalar("intercept"), …
```

:::

:::{tab-item} Python · pls4all.sklearn
:sync: python-sklearn
:class-label: lang-python

```python
from pls4all.sklearn import SparsePLSDAClassifier
mdl = SparsePLSDAClassifier(n_components=2, sparsity_lambda=0.05)
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
res <- pls4all_method("sparse_pls_da", X, y,
                      n_components = 4L, params = list(sparsity_lambda = 0.05, n_classes = 3L))
# res is a named list with MethodResult arrays/scalars.
# selected_indices / top_k_intervals are 1-based.
```

:::

:::{tab-item} R · pls4all (raw fn)
:sync: r-raw
:class-label: lang-r

```r
library(pls4all)
res  <- sparse_pls_da_fit(X, y_labels, n_components, sparsity_lambda = 0.05)
yhat <- pls4all_predict(res, X_test)
```

:::

:::{tab-item} MATLAB · pls4all (MEX)
:sync: matlab-mex
:class-label: lang-matlab

```matlab
res = pls4all.sparse_pls_da(X, y, 4);
% see header of bindings/matlab/+pls4all/sparse_pls_da.m for full
% parameter surface:
%   res = sparse_pls_da(X, y_labels, n_components, sparsity_lambda)
yhat = predict(res, Xtest);
```

:::

:::{tab-item} MATLAB · pls4all (classdef)
:sync: matlab-classdef
:class-label: lang-matlab

_No idiomatic classdef wrapper — invoke `pls4all.fit("sparse_pls_da", X, y, …)` directly from the unified MEX factory._

:::

::::


**Registry parity references** 📐

:::{card}
:class-card: external-refs

- 📐 **`ref.r_spls`** (R · r) — `spls` 2.3.2 · qualitative (rmse_rel ≤ 2e+00) — R `spls::splsda` (Chun & Keles). Predictions returned as hard class labels by the package; we one-hot encode them to match pls4all's soft-assignment prediction shape, so the parity check is on the classification *boundary* rather than continuous score values.
:::

### Validation contract

Structural validation contract for `sparse_pls_da`, exported by the declarative validation framework (`benchmarks/validation/registry/`) from `benchmarks.parity_timing.registry`. Regenerated by `python -m benchmarks.validation.export_registry --write`.

| Field | Value |
|------|-------|
| Reference kind | `external` |
| Canonical reference role | `r` |
| Declared references | R |
| Comparator | `reference_parity` (external-reference validity gate) |
| Comparator metric | `rmse_rel` |
| Reference tolerance | `2e+00` (qualitative) |
| Binding tolerance | `1e-06` |
| Prediction key | `predictions` |
| Required data flags | `y_labels` |

**Benchmark cell parameters**

| Parameter | Value |
|-----------|-------|
| `n_classes` | `3` |
| `n_components` | `4` |
| `n_features` | `50` |
| `n_samples` | `200` |
| `sparsity_lambda` | `0.05` |

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

**Reference gate**: non-release diagnostic — shape/smoke comparison only (`rmse_rel ≤ 2e+00`). Treat it as evidence that both implementations ran, not as numerical agreement.

Rows tagged with **📐** are the canonical parity references for this method (declared in [`parity_timing.registry`](../benchmarks/methodology.md)). C++ and external rows show reference parity; pls4all language bindings show binding parity against the C++ backend. Hover the icon for role and tolerance band.

::::{tab-set}
:class: parity-tabs

:::{tab-item} 1 thread
:sync: threads-1

<div class="parity-table-wrap">
<table class="docutils parity-grouped">
<thead><tr><th scope="col">Backend</th><th scope="col">Divergence</th><th class="size-col" scope="col">100×50 (ms)</th><th class="size-col" scope="col">100×500 (ms)</th><th class="size-col" scope="col">100×2500 (ms)</th><th class="size-col" scope="col">1000×500 (ms)</th></tr></thead>
<tbody class="lang-band lang-cpp"><tr class="lang-band-row" data-lang="cpp"><th colspan="6" scope="rowgroup"><span class="lang-band-dot"></span>C++ native · libp4a</th></tr>
<tr class="bk-row"><td class="bk-name"><code>pls4all.cpp.blas</code></td><td class="parity parity-divergence parity-divergent">1.20</td><td class="ms ms-best">0.96 ms<span class="medal" title="fastest">🏆</span></td><td class="ms ms-best">8.33 ms<span class="medal" title="fastest">🏆</span></td><td class="ms ms-best">43.0 ms<span class="medal" title="fastest">🏆</span></td><td class="ms ms-best">83.3 ms<span class="medal" title="fastest">🏆</span></td></tr>
<tr class="bk-row"><td class="bk-name"><code>pls4all.cpp.blas+omp</code></td><td class="parity parity-divergence parity-divergent">1.20</td><td class="ms">1.00 ms</td><td class="ms">8.35 ms</td><td class="ms">43.7 ms</td><td class="ms">83.7 ms</td></tr>
<tr class="bk-row"><td class="bk-name"><code>pls4all.cpp.omp</code></td><td class="parity parity-divergence parity-divergent">1.20</td><td class="ms">1.00 ms</td><td class="ms">8.38 ms</td><td class="ms">43.6 ms</td><td class="ms">83.8 ms</td></tr>
<tr class="bk-row"><td class="bk-name"><code>pls4all.cpp.ref</code></td><td class="parity parity-divergence parity-divergent">1.20</td><td class="ms">1.01 ms</td><td class="ms">8.33 ms</td><td class="ms">44.0 ms</td><td class="ms">84.7 ms</td></tr>
</tbody>
<tbody class="lang-band lang-python"><tr class="lang-band-row" data-lang="python"><th colspan="6" scope="rowgroup"><span class="lang-band-dot"></span>Python · pls4all</th></tr>
<tr class="bk-row"><td class="bk-name"><code>pls4all.python</code></td><td class="parity parity-divergence parity-divergent">1.20</td><td class="ms">0.99 ms</td><td class="ms">8.35 ms</td><td class="ms">44.2 ms</td><td class="ms">84.9 ms</td></tr>
<tr class="bk-row"><td class="bk-name"><code>pls4all.sklearn</code></td><td class="parity parity-divergence parity-divergent">1.20</td><td class="ms">1.30 ms</td><td class="ms">8.94 ms</td><td class="ms">45.7 ms</td><td class="ms">89.2 ms</td></tr>
</tbody>
<tbody class="lang-band lang-r"><tr class="lang-band-row" data-lang="r"><th colspan="6" scope="rowgroup"><span class="lang-band-dot"></span>R · pls4all</th></tr>
<tr class="bk-row"><td class="bk-name"><code>pls4all.R</code></td><td class="parity parity-divergence parity-divergent">1.20</td><td class="ms">3.60 ms</td><td class="ms">42.2 ms</td><td class="ms">450.9 ms</td><td class="ms">535.9 ms</td></tr>
<tr class="bk-row"><td class="bk-name"><code>pls4all.R.formula</code></td><td class="parity parity-divergence parity-divergent">1.20</td><td class="ms">4.82 ms</td><td class="ms">67.9 ms</td><td class="ms">668.2 ms</td><td class="ms">685.9 ms</td></tr>
<tr class="bk-row"><td class="bk-name"><code>pls4all.R.mdatools</code></td><td class="parity parity-divergence parity-not_available">—</td><td class="ms">5.08 ms</td><td class="ms">60.6 ms</td><td class="ms">642.9 ms</td><td class="ms">668.0 ms</td></tr>
<tr class="bk-row"><td class="bk-name"><code>pls4all.R.pls</code></td><td class="parity parity-divergence parity-not_available">—</td><td class="ms">4.81 ms</td><td class="ms">75.0 ms</td><td class="ms">631.7 ms</td><td class="ms">616.3 ms</td></tr>
</tbody>
<tbody class="lang-band lang-matlab"><tr class="lang-band-row" data-lang="matlab"><th colspan="6" scope="rowgroup"><span class="lang-band-dot"></span>MATLAB · pls4all</th></tr>
<tr class="bk-row"><td class="bk-name"><code>pls4all.matlab</code></td><td class="parity parity-divergence parity-divergent">1.20</td><td class="ms">1.84 ms</td><td class="ms">13.5 ms</td><td class="ms">68.4 ms</td><td class="ms">137.2 ms</td></tr>
<tr class="bk-row"><td class="bk-name"><code>pls4all.matlab.classdef</code></td><td class="parity parity-divergence parity-divergent">1.20</td><td class="ms">2.35 ms</td><td class="ms">14.0 ms</td><td class="ms">68.5 ms</td><td class="ms">130.5 ms</td></tr>
</tbody>
<tbody class="lang-band lang-r"><tr class="lang-band-row" data-lang="r"><th colspan="6" scope="rowgroup"><span class="lang-band-dot"></span>R · external</th></tr>
<tr class="bk-row"><td class="bk-name"><code>mixOmics</code></td><td class="parity parity-divergence parity-divergent">1.58</td><td class="ms">19.8 ms</td><td class="ms">73.1 ms</td><td class="ms">586.5 ms</td><td class="ms">994.6 ms</td></tr>
<tr class="bk-row truth-source truth-source-qualitative"><td class="bk-name"><span class="truth-mark" title="Registry parity reference (r): spls 2.3.2 — qualitative (rmse_rel ≤ 2e+00)">📐</span><code>ref.r_spls</code></td><td class="parity parity-divergence parity-ref-source">0</td><td class="ms">19.5 ms</td><td class="ms">160.5 ms</td><td class="ms">3.0 s</td><td class="ms">1.0 s</td></tr>
</tbody>
</table>
</div>

:::

::::


---

_See also_: [benchmark overview](../benchmarks/overview.md) · [methods index](index.md) · [interactive dashboard](../landing/dashboard.md)