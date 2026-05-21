# `random_frog_select` — Random Frog

_Group_: **Variable selector** · _Registry tolerance_: `1.35`

## Description

Random Frog selection (§18 Phase 5g)

From the `pls4all.sklearn.RandomFrogSelector` docstring:

> Random Frog feature selection (Li 2012).

> **Registry note** — Octave-bridged libPLS 1.95 `randomfrog_pls`. Canonical Li 2012 implementation. RNG diverges from pls4all splitmix64. Mask metric ~0=perfect, ~1.41=disjoint.

### Parameters

| Name | Type | Default | Notes |
|------|------|---------|-------|
| `top_k` | `int` | `None` | Number of features to retain. |
| `n_components` | `int` | `2` | Number of latent components extracted (k). |
| `n_iterations` | `int` | `100` | Number of selection iterations or Monte-Carlo passes. |
| `initial_size` | `int` | `20` | Starting subset size for the random-frog chain. |
| `min_size` | `int | None` | `None` | Minimum allowed subset size during the random-frog Markov chain. |
| `max_size` | `int | None` | `None` | Maximum allowed subset size during the random-frog Markov chain. |
| `n_folds` | `int` | `3` | Number of cross-validation folds used inside the selector. |
| `seed` | `int` | `0` | Random seed for reproducible sampling/initialization. |

## Explanations

### Bibliographic source

Li, H., Xu, Q. & Liang, Y. (2012). *Random frog: an efficient reversible jump Markov chain Monte Carlo-like approach for variable selection*. Analytica Chimica Acta 740, 20–26.

### Mathematical principle

Random Frog runs a random walk over feature subsets: at each step it proposes a transition to a neighbouring subset (add / remove / swap a feature) and accepts the transition with a Metropolis-style probability based on the improvement in CV-RMSE. Features that appear frequently in the visited subsets are deemed important.

Output: the **inclusion frequency** vector — fraction of iterations in which each feature was selected. Sort by frequency and take the top-$k$ for the final subset.

Random Frog is sample-efficient compared to GA-PLS (no population of full subsets to maintain) but slower to mix on very high-dimensional data. Recommended on spectra of moderate size (a few hundred wavelengths).

### Implementation

`p4a_random_frog_select`.

R roxygen note (`methods_extra.R::random_frog_select`):

> Random Frog (Phase 5g).
> @param n_components Integer. Number of latent components.
> @param n_iterations Integer >= 1. Number of outer-loop iterations.
> @param initial_size Method-specific parameter. See the underlying `*_fit()` function for the exact semantics.
> @param min_size Method-specific parameter. See the underlying `*_fit()` function for the exact semantics.
> @param max_size Method-specific parameter. See the underlying `*_fit()` function for the exact semantics.
> @param top_k Method-specific parameter. See the underlying `*_fit()` function for the exact semantics.
> @param seed Integer. Random seed for reproducibility.
> @param X Numeric matrix of predictors (rows = samples, cols = features).
> @param Y Numeric matrix or vector of responses, with one row per sample.
> @export

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
p4a_random_frog_select(ctx, cfg, &x_view, &y_view, /* hyperparams */, &res);
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
from pls4all._methods import random_frog_select
with pls4all.Context() as ctx, pls4all.Config() as cfg:
    res = random_frog_select(ctx, cfg, X, y, n_components=4)
# then: res.matrix("predictions"), res.matrix("coefficients"),
# res.vector("mask"), res.scalar("intercept"), …
```

:::

:::{tab-item} Python · pls4all.sklearn
:sync: python-sklearn
:class-label: lang-python

```python
from pls4all.sklearn import RandomFrogSelector
mdl = RandomFrogSelector(top_k, n_components=2, n_iterations=100, initial_size=20, min_size=None, max_size=None, n_folds=3, seed=0)
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
res <- pls4all_method("random_frog_select", X, y,
                      n_components = 4L, params = list(n_iterations = 10L, initial_size = 10L, min_size = 5L, max_size = 20L, top_k = 10L, seed = 11L))
# res is a named list with MethodResult arrays/scalars.
# selected_indices / top_k_intervals are 1-based.
```

:::

:::{tab-item} R · pls4all (raw fn)
:sync: r-raw
:class-label: lang-r

```r
library(pls4all)
res  <- random_frog_select(X, Y, n_components,
                    n_iterations = 100L,
                    initial_size = 30L, min_size = NULL,
                    max_size = NULL, top_k = 10L, seed = 0L)
yhat <- pls4all_predict(res, X_test)
```

:::

:::{tab-item} MATLAB · pls4all (MEX)
:sync: matlab-mex
:class-label: lang-matlab

```matlab
res  = pls4all.fit("random_frog_select", X, y, "NumComponents", 4);
yhat = predict(res, Xtest);
```

:::

:::{tab-item} MATLAB · pls4all (classdef)
:sync: matlab-classdef
:class-label: lang-matlab

_No idiomatic classdef wrapper — invoke `pls4all.fit("random_frog_select", X, y, …)` directly from the unified MEX factory._

:::

::::


**Registry parity references** 📐

:::{card}
:class-card: external-refs

- 📐 **`ref.matlab_libpls`** (matlab · python) — `libPLS` 1.95 · qualitative (rmse_rel ≤ 1e+00) — Octave-bridged libPLS 1.95 `randomfrog_pls(X, Y, A, 'center', N, Q, 'regcoef')`. RNG differs from pls4all splitmix64; mask metric. Top-10 ranked features mapped to a 1xp mask.
:::

### Validation contract

Structural validation contract for `random_frog_select`, exported by the declarative validation framework (`benchmarks/validation/registry/`) from `benchmarks.parity_timing.registry`. Regenerated by `python -m benchmarks.validation.export_registry --write`.

| Field | Value |
|------|-------|
| Reference kind | `external` |
| Canonical reference role | `python` |
| Declared references | Python |
| Comparator | `reference_parity` (external-reference validity gate) |
| Comparator metric | `rmse_rel` |
| Reference tolerance | `1e+00` (qualitative) |
| Binding tolerance | `1e-06` |
| Prediction key | `mask` |
| Required data flags | none |

**Benchmark cell parameters**

| Parameter | Value |
|-----------|-------|
| `initial_size` | `10` |
| `max_size` | `20` |
| `min_size` | `5` |
| `n_components` | `4` |
| `n_features` | `40` |
| `n_iterations` | `10` |
| `n_samples` | `200` |
| `seed` | `11` |
| `top_k` | `10` |

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
<tr class="bk-row"><td class="bk-name"><code>pls4all.cpp.blas</code></td><td class="parity parity-divergence parity-divergent">1.10</td><td class="ms">1.39 ms</td><td class="ms ms-best">8.81 ms<span class="medal" title="fastest">🏆</span></td><td class="ms">43.8 ms</td><td class="ms">87.4 ms</td></tr>
<tr class="bk-row"><td class="bk-name"><code>pls4all.cpp.blas+omp</code></td><td class="parity parity-divergence parity-divergent">1.18</td><td class="ms">1.41 ms</td><td class="ms">8.91 ms</td><td class="ms">43.5 ms</td><td class="ms ms-best">86.8 ms<span class="medal" title="fastest">🏆</span></td></tr>
<tr class="bk-row"><td class="bk-name"><code>pls4all.cpp.omp</code></td><td class="parity parity-divergence parity-divergent">1.10</td><td class="ms ms-best">1.38 ms<span class="medal" title="fastest">🏆</span></td><td class="ms">9.04 ms</td><td class="ms">44.5 ms</td><td class="ms">87.7 ms</td></tr>
<tr class="bk-row"><td class="bk-name"><code>pls4all.cpp.ref</code></td><td class="parity parity-divergence parity-divergent">1.10</td><td class="ms">1.44 ms</td><td class="ms">8.93 ms</td><td class="ms">44.8 ms</td><td class="ms">90.7 ms</td></tr>
</tbody>
<tbody class="lang-band lang-python"><tr class="lang-band-row" data-lang="python"><th colspan="6" scope="rowgroup"><span class="lang-band-dot"></span>Python · pls4all</th></tr>
<tr class="bk-row"><td class="bk-name"><code>pls4all.python</code></td><td class="parity parity-divergence parity-divergent">1.18</td><td class="ms">1.49 ms</td><td class="ms">8.99 ms</td><td class="ms">43.6 ms</td><td class="ms">89.6 ms</td></tr>
<tr class="bk-row"><td class="bk-name"><code>pls4all.sklearn</code></td><td class="parity parity-divergence parity-divergent">1.18</td><td class="ms">1.67 ms</td><td class="ms">9.22 ms</td><td class="ms ms-best">43.1 ms<span class="medal" title="fastest">🏆</span></td><td class="ms">89.4 ms</td></tr>
</tbody>
<tbody class="lang-band lang-r"><tr class="lang-band-row" data-lang="r"><th colspan="6" scope="rowgroup"><span class="lang-band-dot"></span>R · pls4all</th></tr>
<tr class="bk-row"><td class="bk-name"><code>pls4all.R</code></td><td class="parity parity-divergence parity-divergent">1.18</td><td class="ms">3.46 ms</td><td class="ms">48.9 ms</td><td class="ms">408.8 ms</td><td class="ms">565.6 ms</td></tr>
<tr class="bk-row"><td class="bk-name"><code>pls4all.R.formula</code></td><td class="parity parity-divergence parity-divergent">1.18</td><td class="ms">5.49 ms</td><td class="ms">63.1 ms</td><td class="ms">606.3 ms</td><td class="ms">754.1 ms</td></tr>
<tr class="bk-row"><td class="bk-name"><code>pls4all.R.mdatools</code></td><td class="parity parity-divergence parity-drift">—</td><td class="ms">5.83 ms</td><td class="ms">64.8 ms</td><td class="ms">608.2 ms</td><td class="ms">754.2 ms</td></tr>
<tr class="bk-row"><td class="bk-name"><code>pls4all.R.pls</code></td><td class="parity parity-divergence parity-drift">—</td><td class="ms">5.21 ms</td><td class="ms">67.5 ms</td><td class="ms">596.5 ms</td><td class="ms">709.7 ms</td></tr>
</tbody>
<tbody class="lang-band lang-matlab"><tr class="lang-band-row" data-lang="matlab"><th colspan="6" scope="rowgroup"><span class="lang-band-dot"></span>MATLAB · pls4all</th></tr>
<tr class="bk-row"><td class="bk-name"><code>pls4all.matlab</code></td><td class="parity parity-divergence parity-divergent">1.18</td><td class="ms">2.12 ms</td><td class="ms">14.1 ms</td><td class="ms">67.7 ms</td><td class="ms">139.6 ms</td></tr>
<tr class="bk-row"><td class="bk-name"><code>pls4all.matlab.classdef</code></td><td class="parity parity-divergence parity-divergent">1.18</td><td class="ms">2.49 ms</td><td class="ms">14.5 ms</td><td class="ms">68.7 ms</td><td class="ms">138.3 ms</td></tr>
</tbody>
<tbody class="lang-band lang-matlab"><tr class="lang-band-row" data-lang="matlab"><th colspan="6" scope="rowgroup"><span class="lang-band-dot"></span>MATLAB · external</th></tr>
<tr class="bk-row truth-source truth-source-qualitative"><td class="bk-name"><span class="truth-mark" title="Registry parity reference (python): libPLS 1.95 — qualitative (rmse_rel ≤ 1e+00)">📐</span><code>ref.matlab_libpls</code></td><td class="parity parity-divergence parity-ref-source">0</td><td class="ms">66.6 ms</td><td class="ms">77.0 ms</td><td class="ms">108.8 ms</td><td class="ms">164.3 ms</td></tr>
</tbody>
</table>
</div>

:::

::::


---

_See also_: [benchmark overview](../benchmarks/overview.md) · [methods index](index.md) · [interactive dashboard](../landing/dashboard.md)