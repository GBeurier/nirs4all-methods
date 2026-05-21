# `n_pls` — N-way PLS (Trilinear PLS, Bro 1996)

_Group_: **Multi-block / cross-modal** · _Registry tolerance_: `10.0`

## Description

N-PLS — 3-way tensor PLS (Bro 1996)

From the `pls4all.sklearn.NPLSRegression` docstring:

> N-PLS (3-way tensor) regression (Bro 1996).

> **Registry note** — Python `tensorly.parafac` + OLS as a qualitative reference for Bro 1996 multilinear PLS. Different algorithm (PARAFAC + OLS vs canonical N-PLS); widened tolerance flags external-ref presence.

### Parameters

| Name | Type | Default | Notes |
|------|------|---------|-------|
| `n_components` | `int` | `2` | Number of latent components extracted (k). |
| `mode_j` | `int` | `—` | Length of the second mode (J) of the 3-way input tensor. |
| `mode_k` | `int` | `—` | Length of the third mode (K) of the 3-way input tensor. |

## Explanations

### Bibliographic source

Bro, R. (1996). *Multiway calibration. Multilinear PLS*. Journal of Chemometrics 10(1), 47–61.

### Mathematical principle

When the predictor is naturally a tensor $\mathcal{X} \in \mathbb{R}^{n \times J \times K}$ — e.g. excitation × emission fluorescence spectra, or wavelength × time chromatographic data — N-way PLS decomposes $\mathcal{X}$ into a sum of rank-one tensor components: $\mathcal{X} \approx \sum_{a=1}^{k} \mathbf{t}_a \circ \mathbf{w}_a^{J} \circ \mathbf{w}_a^{K}$, where $\circ$ is the outer product. The score vectors $\mathbf{t}_a$ are computed to maximise covariance with $\mathbf{y}$, the loading vectors $\mathbf{w}_a^{J}, \mathbf{w}_a^{K}$ live in the respective tensor modes.

Compared to unfolding the tensor and running standard PLS, N-PLS respects the multilinear structure of the data and produces interpretable per-mode loadings. The computational cost is comparable to standard PLS — matrix-vector products in each mode rather than one large matrix-vector product.

Note that pls4all takes the tensor as a **flattened** matrix plus `mode_j` and `mode_k` shape parameters; the kernel reshapes internally.

### Implementation

`p4a_n_pls_fit`. Reference: Python `tensorly 0.9.0` (`tensorly.regression.tucker_regression`) and Bro's original MATLAB code.

R roxygen note (`methods_extra.R::n_pls_fit`):

> N-PLS (3-way tensor) regression. `X_flat` is the flattened (n, mode_j*mode_k) matrix.
> @param X_flat Numeric matrix. The flattened 3-way design tensor
>   (rows = samples, cols = mode_j * mode_k).
> @param Y Numeric matrix or vector of responses, with one row per sample.
> @param n_components Integer. Number of latent components.
> @param mode_j Integer. Size of the first non-sample tensor mode.
> @param mode_k Integer. Size of the second non-sample tensor mode.
> @export

MATLAB header (`bindings/matlab/+pls4all/NPlsRegression.m`):

```text
pls4all.NPlsRegression  N-PLS (3-way tensor) regression (Bro 1996).
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
p4a_n_pls_fit(ctx, cfg, &x_view, &y_view, /* hyperparams */, &res);
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
from pls4all._methods import n_pls_fit
with pls4all.Context() as ctx, pls4all.Config() as cfg:
    res = n_pls_fit(ctx, cfg, X, y, n_components=4)
# then: res.matrix("predictions"), res.matrix("coefficients"),
# res.vector("mask"), res.scalar("intercept"), …
```

:::

:::{tab-item} Python · pls4all.sklearn
:sync: python-sklearn
:class-label: lang-python

```python
from pls4all.sklearn import NPLSRegression
mdl = NPLSRegression(n_components=2, mode_j, mode_k)
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
res <- pls4all_method("n_pls", X, y,
                      n_components = 4L, params = list(mode_j = 8L, mode_k = 6L))
# res is a named list with MethodResult arrays/scalars.
# selected_indices / top_k_intervals are 1-based.
```

:::

:::{tab-item} R · pls4all (raw fn)
:sync: r-raw
:class-label: lang-r

```r
library(pls4all)
res  <- n_pls_fit(X_flat, Y, n_components, mode_j, mode_k)
yhat <- pls4all_predict(res, X_test)
```

:::

:::{tab-item} MATLAB · pls4all (MEX)
:sync: matlab-mex
:class-label: lang-matlab

```matlab
res = pls4all.n_pls(X, y, 4);
% see header of bindings/matlab/+pls4all/n_pls.m for full
% parameter surface:
%   res = n_pls(X_flat, Y, n_components, mode_j, mode_k)
yhat = predict(res, Xtest);
```

:::

:::{tab-item} MATLAB · pls4all (classdef)
:sync: matlab-classdef
:class-label: lang-matlab

```matlab
mdl  = pls4all.fit("n_pls", X, y, "NumComponents", 4);
yhat = predict(mdl, Xtest);
```

:::

::::


**Registry parity references** 📐

:::{card}
:class-card: external-refs

- 📐 **`ref.python_tensorly`** (python · python) — `tensorly` 0.9.0 · qualitative (rmse_rel ≤ 1e+01) — Python `tensorly.parafac` + OLS as a qualitative reference. pls4all's N-PLS uses Bro 1996 multilinear PLS; tensorly decomposes the tensor and we fit OLS on the mode-1 loadings. Predictions diverge by O(1) — flagged as external-ref presence, not bit-exact parity.
:::

### Validation contract

Structural validation contract for `n_pls`, exported by the declarative validation framework (`benchmarks/validation/registry/`) from `benchmarks.parity_timing.registry`. Regenerated by `python -m benchmarks.validation.export_registry --write`.

| Field | Value |
|------|-------|
| Reference kind | `external` |
| Canonical reference role | `python` |
| Declared references | Python |
| Comparator | `reference_parity` (external-reference validity gate) |
| Comparator metric | `rmse_rel` |
| Reference tolerance | `1e+01` (qualitative) |
| Binding tolerance | `1e-06` |
| Prediction key | `predictions` |
| Required data flags | none |

**Benchmark cell parameters**

| Parameter | Value |
|-----------|-------|
| `mode_j` | `8` |
| `mode_k` | `6` |
| `n_components` | `4` |
| `n_features` | `48` |
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
<tr class="bk-row"><td class="bk-name"><code>pls4all.cpp.blas</code></td><td class="parity parity-divergence parity-divergent">3.34</td><td class="ms ms-best">1.02 ms<span class="medal" title="fastest">🏆</span></td><td class="ms">8.43 ms</td><td class="ms">43.5 ms</td><td class="ms ms-best">84.4 ms<span class="medal" title="fastest">🏆</span></td></tr>
<tr class="bk-row"><td class="bk-name"><code>pls4all.cpp.blas+omp</code></td><td class="parity parity-divergence parity-divergent">3.34</td><td class="ms">1.04 ms</td><td class="ms">8.43 ms</td><td class="ms">43.7 ms</td><td class="ms">86.5 ms</td></tr>
<tr class="bk-row"><td class="bk-name"><code>pls4all.cpp.omp</code></td><td class="parity parity-divergence parity-divergent">3.34</td><td class="ms">1.06 ms</td><td class="ms">8.73 ms</td><td class="ms">44.0 ms</td><td class="ms">84.4 ms</td></tr>
<tr class="bk-row"><td class="bk-name"><code>pls4all.cpp.ref</code></td><td class="parity parity-divergence parity-divergent">3.34</td><td class="ms">1.04 ms</td><td class="ms">8.47 ms</td><td class="ms">43.4 ms</td><td class="ms">84.7 ms</td></tr>
</tbody>
<tbody class="lang-band lang-python"><tr class="lang-band-row" data-lang="python"><th colspan="6" scope="rowgroup"><span class="lang-band-dot"></span>Python · pls4all</th></tr>
<tr class="bk-row"><td class="bk-name"><code>pls4all.python</code></td><td class="parity parity-divergence parity-divergent">3.34</td><td class="ms">1.11 ms</td><td class="ms ms-best">8.42 ms<span class="medal" title="fastest">🏆</span></td><td class="ms ms-best">43.2 ms<span class="medal" title="fastest">🏆</span></td><td class="ms">84.9 ms</td></tr>
<tr class="bk-row"><td class="bk-name"><code>pls4all.sklearn</code></td><td class="parity parity-divergence parity-divergent">3.34</td><td class="ms">1.24 ms</td><td class="ms">8.93 ms</td><td class="ms">44.0 ms</td><td class="ms">85.1 ms</td></tr>
</tbody>
<tbody class="lang-band lang-r"><tr class="lang-band-row" data-lang="r"><th colspan="6" scope="rowgroup"><span class="lang-band-dot"></span>R · pls4all</th></tr>
<tr class="bk-row"><td class="bk-name"><code>pls4all.R</code></td><td class="parity parity-divergence parity-divergent">5.72</td><td class="ms">3.09 ms</td><td class="ms">40.7 ms</td><td class="ms">424.9 ms</td><td class="ms">453.6 ms</td></tr>
<tr class="bk-row"><td class="bk-name"><code>pls4all.R.formula</code></td><td class="parity parity-divergence parity-divergent">5.72</td><td class="ms">4.84 ms</td><td class="ms">58.6 ms</td><td class="ms">586.3 ms</td><td class="ms">638.7 ms</td></tr>
<tr class="bk-row"><td class="bk-name"><code>pls4all.R.mdatools</code></td><td class="parity parity-divergence parity-divergent">5.72</td><td class="ms">4.33 ms</td><td class="ms">54.0 ms</td><td class="ms">567.8 ms</td><td class="ms">594.7 ms</td></tr>
<tr class="bk-row"><td class="bk-name"><code>pls4all.R.pls</code></td><td class="parity parity-divergence parity-divergent">5.72</td><td class="ms">4.39 ms</td><td class="ms">59.2 ms</td><td class="ms">561.4 ms</td><td class="ms">609.1 ms</td></tr>
</tbody>
<tbody class="lang-band lang-matlab"><tr class="lang-band-row" data-lang="matlab"><th colspan="6" scope="rowgroup"><span class="lang-band-dot"></span>MATLAB · pls4all</th></tr>
<tr class="bk-row"><td class="bk-name"><code>pls4all.matlab</code></td><td class="parity parity-divergence parity-divergent">5.72</td><td class="ms">1.69 ms</td><td class="ms">13.3 ms</td><td class="ms">67.1 ms</td><td class="ms">132.3 ms</td></tr>
<tr class="bk-row"><td class="bk-name"><code>pls4all.matlab.classdef</code></td><td class="parity parity-divergence parity-divergent">5.72</td><td class="ms">2.09 ms</td><td class="ms">13.9 ms</td><td class="ms">68.5 ms</td><td class="ms">132.9 ms</td></tr>
</tbody>
<tbody class="lang-band lang-python"><tr class="lang-band-row" data-lang="python"><th colspan="6" scope="rowgroup"><span class="lang-band-dot"></span>Python · external</th></tr>
<tr class="bk-row truth-source truth-source-qualitative"><td class="bk-name"><span class="truth-mark" title="Registry parity reference (python): tensorly 0.9.0 — qualitative (rmse_rel ≤ 1e+01)">📐</span><code>ref.python_tensorly</code></td><td class="parity parity-divergence parity-ref-source">0</td><td class="ms">16.7 ms</td><td class="ms">36.6 ms</td><td class="ms">126.8 ms</td><td class="ms">271.8 ms</td></tr>
</tbody>
</table>
</div>

:::

::::


---

_See also_: [benchmark overview](../benchmarks/overview.md) · [methods index](index.md) · [interactive dashboard](../landing/dashboard.md)