# `gpr_pls` — Gaussian Process on PLS scores

_Group_: **Nonlinear / local** · _Registry tolerance_: `1e-08`

## Description

GPR-on-PLS — RBF Gaussian Process on PLS scores (§47)

From the `pls4all.sklearn.GPRPLSRegression` docstring:

> Gaussian-process head on SIMPLS training scores.

> **Registry note** — GP head parity (sklearn `GaussianProcessRegressor` with RBF+WhiteKernel, optimizer=None) on the same PLS scores. Architecturally separated to allow GPR-on-AOMPLS reuse of `fit_gp_on_scores`.

### Parameters

| Name | Type | Default | Notes |
|------|------|---------|-------|
| `n_components` | `int` | `2` | Number of latent components extracted (k). |
| `length_scale` | `float` | `1.0` |  |
| `noise_level` | `float` | `0.001` |  |
| `seed` | `int` | `0` | Random seed for reproducible sampling/initialization. |

## Explanations

### Bibliographic source

Bishop, C. M. (2006). *Pattern Recognition and Machine Learning*, §6.4 (Gaussian Processes). — combined with a preliminary PLS dimensionality reduction for spectroscopy.

### Mathematical principle

Spectroscopic data are too high-dimensional for a direct Gaussian Process: GP inference is $O(n^3)$ in samples but the *kernel quality* degrades rapidly when $p$ exceeds a few hundred — most pairwise distances become near-identical, the kernel matrix loses contrast and the GP under-fits.

GPR-PLS solves this by first projecting $\mathbf{X} \to \mathbf{T} = \mathbf{X}\mathbf{W}$ into a low-dimensional PLS latent space and **then** training a GP on $\mathbf{T}$. The latent space preserves the variance most relevant to $y$, the GP captures smooth non-linear residual structure, and the kernel matrix is well-conditioned because pairwise distances in $\mathbb{R}^k$ remain informative.

Default kernel: RBF with length scale $\ell$ and amplitude $\sigma_f^2$, plus an isotropic noise variance $\sigma_n^2$. Marginal-likelihood maximisation selects the three hyperparameters; pls4all uses a fixed-iteration L-BFGS pass to keep the cost bounded per cell.

### Implementation

`p4a_gpr_pls_fit`. Reference: sklearn `GaussianProcessRegressor` with an RBF kernel applied to the score matrix from a separate sklearn `PLSRegression`.

R roxygen note (`methods_extra.R::gpr_pls_fit`):

> Gaussian Process Regression on PLS scores (single-target Y).
> @param n_components Integer. Number of latent components.
> @param length_scale Method-specific parameter. See the underlying `*_fit()` function for the exact semantics.
> @param noise_level Method-specific parameter. See the underlying `*_fit()` function for the exact semantics.
> @param seed Integer. Random seed for reproducibility.
> @param X Numeric matrix of predictors (rows = samples, cols = features).
> @param Y Numeric matrix or vector of responses, with one row per sample.
> @export

MATLAB header (`bindings/matlab/+pls4all/gpr_pls.m`):

```text
pls4all.gpr_pls  GPR on PLS scores (RBF kernel, single-target Y).
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
p4a_gpr_pls_fit(ctx, cfg, &x_view, &y_view, /* hyperparams */, &res);
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
from pls4all._methods import gpr_pls_fit
with pls4all.Context() as ctx, pls4all.Config() as cfg:
    res = gpr_pls_fit(ctx, cfg, X, y, n_components=3)
# then: res.matrix("predictions"), res.matrix("coefficients"),
# res.vector("mask"), res.scalar("intercept"), …
```

:::

:::{tab-item} Python · pls4all.sklearn
:sync: python-sklearn
:class-label: lang-python

```python
from pls4all.sklearn import GPRPLSRegression
mdl = GPRPLSRegression(n_components=2, length_scale=1.0, noise_level=0.001, seed=0)
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
res <- pls4all_method("gpr_pls", X, y,
                      n_components = 3L, params = list(length_scale = 1.0, noise_level = 0.001, seed = 0L))
# res is a named list with MethodResult arrays/scalars.
# selected_indices / top_k_intervals are 1-based.
```

:::

:::{tab-item} R · pls4all (raw fn)
:sync: r-raw
:class-label: lang-r

```r
library(pls4all)
res  <- gpr_pls_fit(X, Y, n_components,
             length_scale = 1.0, noise_level = 1e-4, seed = 0L)
yhat <- pls4all_predict(res, X_test)
```

:::

:::{tab-item} MATLAB · pls4all (MEX)
:sync: matlab-mex
:class-label: lang-matlab

```matlab
res = pls4all.gpr_pls(X, y, 3);
% see header of bindings/matlab/+pls4all/gpr_pls.m for full
% parameter surface:
%   res = gpr_pls(X, Y, n_components, length_scale, noise_level, seed)
yhat = predict(res, Xtest);
```

:::

:::{tab-item} MATLAB · pls4all (classdef)
:sync: matlab-classdef
:class-label: lang-matlab

_No idiomatic classdef wrapper — invoke `pls4all.fit("gpr_pls", X, y, …)` directly from the unified MEX factory._

:::

::::


**Registry parity references** 📐

:::{card}
:class-card: external-refs

- 📐 **`ref.python_scikit_learn`** (python · python) — `scikit-learn` 1.4.2 · strict (rmse_rel ≤ 1e-08) — sklearn GP head on the same PLS training scores pls4all produces. PLS rotation conventions diverge per-component; comparing the GP head on shared T isolates the novel stage.
:::

### Validation contract

Structural validation contract for `gpr_pls`, exported by the declarative validation framework (`benchmarks/validation/registry/`) from `benchmarks.parity_timing.registry`. Regenerated by `python -m benchmarks.validation.export_registry --write`.

| Field | Value |
|------|-------|
| Reference kind | `external` |
| Canonical reference role | `python` |
| Declared references | Python |
| Comparator | `reference_parity` (external-reference validity gate) |
| Comparator metric | `rmse_rel` |
| Reference tolerance | `1e-08` (strict) |
| Binding tolerance | `1e-06` |
| Prediction key | `predictions` |
| Required data flags | none |

**Benchmark cell parameters**

| Parameter | Value |
|-----------|-------|
| `length_scale` | `1` |
| `n_components` | `3` |
| `n_features` | `25` |
| `n_samples` | `120` |
| `noise_level` | `0.001` |
| `seed` | `0` |

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

**Reference gate**: strict release — `rmse_rel ≤ 1e-08` is required; `rmse_rel ≤ 1e-06` is displayed as exact.

Rows tagged with **📐** are the canonical parity references for this method (declared in [`parity_timing.registry`](../benchmarks/methodology.md)). C++ and external rows show reference parity; pls4all language bindings show binding parity against the C++ backend. Hover the icon for role and tolerance band.

::::{tab-set}
:class: parity-tabs

:::{tab-item} 1 thread
:sync: threads-1

<div class="parity-table-wrap">
<table class="docutils parity-grouped">
<thead><tr><th scope="col">Backend</th><th scope="col">Divergence</th><th class="size-col" scope="col">100×50 (ms)</th><th class="size-col" scope="col">100×500 (ms)</th><th class="size-col" scope="col">100×2500 (ms)</th><th class="size-col" scope="col">1000×500 (ms)</th></tr></thead>
<tbody class="lang-band lang-cpp"><tr class="lang-band-row" data-lang="cpp"><th colspan="6" scope="rowgroup"><span class="lang-band-dot"></span>C++ native · libp4a</th></tr>
<tr class="bk-row"><td class="bk-name"><code>pls4all.cpp.blas</code></td><td class="parity parity-divergence parity-ref-strict">3e-10</td><td class="ms">1.30 ms</td><td class="ms">8.55 ms</td><td class="ms ms-best">41.4 ms<span class="medal" title="fastest">🏆</span></td><td class="ms">388.1 ms</td></tr>
<tr class="bk-row"><td class="bk-name"><code>pls4all.cpp.blas+omp</code></td><td class="parity parity-divergence parity-ref-strict">3e-10</td><td class="ms ms-best">1.29 ms<span class="medal" title="fastest">🏆</span></td><td class="ms">8.38 ms</td><td class="ms">41.7 ms</td><td class="ms">382.6 ms</td></tr>
<tr class="bk-row"><td class="bk-name"><code>pls4all.cpp.omp</code></td><td class="parity parity-divergence parity-ref-strict">3e-10</td><td class="ms">2.06 ms</td><td class="ms">8.68 ms</td><td class="ms">42.6 ms</td><td class="ms">386.8 ms</td></tr>
<tr class="bk-row"><td class="bk-name"><code>pls4all.cpp.ref</code></td><td class="parity parity-divergence parity-ref-strict">3e-10</td><td class="ms">1.32 ms</td><td class="ms">8.56 ms</td><td class="ms">42.9 ms</td><td class="ms">385.9 ms</td></tr>
</tbody>
<tbody class="lang-band lang-python"><tr class="lang-band-row" data-lang="python"><th colspan="6" scope="rowgroup"><span class="lang-band-dot"></span>Python · pls4all</th></tr>
<tr class="bk-row"><td class="bk-name"><code>pls4all.python</code></td><td class="parity parity-divergence parity-drift">3e-10</td><td class="ms">1.29 ms</td><td class="ms ms-best">8.35 ms<span class="medal" title="fastest">🏆</span></td><td class="ms">41.5 ms</td><td class="ms ms-best">374.9 ms<span class="medal" title="fastest">🏆</span></td></tr>
<tr class="bk-row"><td class="bk-name"><code>pls4all.sklearn</code></td><td class="parity parity-divergence parity-drift">3e-10</td><td class="ms">1.44 ms</td><td class="ms">8.65 ms</td><td class="ms">43.6 ms</td><td class="ms">380.8 ms</td></tr>
</tbody>
<tbody class="lang-band lang-r"><tr class="lang-band-row" data-lang="r"><th colspan="6" scope="rowgroup"><span class="lang-band-dot"></span>R · pls4all</th></tr>
<tr class="bk-row"><td class="bk-name"><code>pls4all.R</code></td><td class="parity parity-divergence parity-drift">3e-10</td><td class="ms">3.76 ms</td><td class="ms">46.5 ms</td><td class="ms">390.7 ms</td><td class="ms">891.4 ms</td></tr>
<tr class="bk-row"><td class="bk-name"><code>pls4all.R.formula</code></td><td class="parity parity-divergence parity-drift">3e-10</td><td class="ms">4.90 ms</td><td class="ms">73.8 ms</td><td class="ms">567.8 ms</td><td class="ms">997.9 ms</td></tr>
<tr class="bk-row"><td class="bk-name"><code>pls4all.R.mdatools</code></td><td class="parity parity-divergence parity-drift">2e-10</td><td class="ms">5.17 ms</td><td class="ms">64.3 ms</td><td class="ms">590.1 ms</td><td class="ms">912.0 ms</td></tr>
<tr class="bk-row"><td class="bk-name"><code>pls4all.R.pls</code></td><td class="parity parity-divergence parity-drift">2e-10</td><td class="ms">5.08 ms</td><td class="ms">76.0 ms</td><td class="ms">593.7 ms</td><td class="ms">867.5 ms</td></tr>
</tbody>
<tbody class="lang-band lang-matlab"><tr class="lang-band-row" data-lang="matlab"><th colspan="6" scope="rowgroup"><span class="lang-band-dot"></span>MATLAB · pls4all</th></tr>
<tr class="bk-row"><td class="bk-name"><code>pls4all.matlab</code></td><td class="parity parity-divergence parity-drift">3e-10</td><td class="ms">2.05 ms</td><td class="ms">13.6 ms</td><td class="ms">65.8 ms</td><td class="ms">418.1 ms</td></tr>
<tr class="bk-row"><td class="bk-name"><code>pls4all.matlab.classdef</code></td><td class="parity parity-divergence parity-drift">3e-10</td><td class="ms">2.40 ms</td><td class="ms">14.0 ms</td><td class="ms">67.5 ms</td><td class="ms">419.9 ms</td></tr>
</tbody>
<tbody class="lang-band lang-python"><tr class="lang-band-row" data-lang="python"><th colspan="6" scope="rowgroup"><span class="lang-band-dot"></span>Python · external</th></tr>
<tr class="bk-row truth-source truth-source-strict"><td class="bk-name"><span class="truth-mark" title="Registry parity reference (python): scikit-learn 1.4.2 — strict (rmse_rel ≤ 1e-08)">📐</span><code>ref.python_scikit_learn</code></td><td class="parity parity-divergence parity-ref-source">0</td><td class="ms">2.29 ms</td><td class="ms">10.2 ms</td><td class="ms">42.5 ms</td><td class="ms">414.3 ms</td></tr>
</tbody>
</table>
</div>

:::

::::


---

_See also_: [benchmark overview](../benchmarks/overview.md) · [methods index](index.md) · [interactive dashboard](../landing/dashboard.md)