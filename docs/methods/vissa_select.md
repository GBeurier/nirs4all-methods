# `vissa_select` — VISSA — Variable Iterative Space-Shrinkage

_Group_: **Variable selector** · _Registry tolerance_: `2.5`

## Description

VISSA-PLS — Variable Iterative Space Shrinkage (§49)

From the `pls4all.sklearn.VISSASelector` docstring:

> Variable Iterative Subspace Shrinkage Approach (Deng 2014).

> **Registry note** — Python `auswahl.VISSA 0.9.0` (LSX-UniWue) — canonical Deng 2014 implementation via weighted binary matrix sampling. RNG diverges from pls4all splitmix64; small selection set (~2/25) inflates mask divergence.

### Parameters

| Name | Type | Default | Notes |
|------|------|---------|-------|
| `n_components` | `int` | `2` | Number of latent components extracted (k). |
| `n_iterations` | `int` | `10` | Number of selection iterations or Monte-Carlo passes. |
| `n_submodels` | `int` | `60` | Number of bootstrap sub-models drawn per VISSA iteration. |
| `ratio_kept` | `float` | `0.1` | Fraction of top-scoring features retained at each VISSA shrinkage step. |
| `threshold` | `float` | `0.5` | Inclusion-probability cut-off below which features are dropped. |
| `floor_probability` | `float` | `0.05` | Lower bound applied to per-feature inclusion probabilities to avoid premature pruning. |
| `n_folds` | `int` | `3` | Number of cross-validation folds used inside the selector. |
| `seed` | `int` | `0` | Random seed for reproducible sampling/initialization. |

## Explanations

### Bibliographic source

Deng, B. C., Yun, Y. H., Liang, Y. Z. & Yi, L. Z. (2014). *A new strategy to prevent over-fitting in partial least squares models based on model population analysis*. Analytica Chimica Acta 880, 32–41.

### Mathematical principle

VISSA evaluates a **population of random subsets** of the same size, refines the population by selecting the best by CV-RMSE, and iteratively shrinks the search space toward features that survive in many high-performing subsets. Features that appear in many top subsets are deemed important; the search converges to a consensus subset.

Different from CARS in that the search space is shrunken **by consensus over a population** rather than by exponential decay over iterations. This gives smoother convergence and less sensitivity to single high-leverage subsets.

### Implementation

`p4a_vissa_select`.

R roxygen note (`methods_extra.R::vissa_select`):

> VISSA — Variable Iterative Space Shrinkage Approach.
> @param n_components Integer. Number of latent components.
> @param n_iterations Integer >= 1. Number of outer-loop iterations.
> @param n_submodels Integer >= 1. Number of inner sub-models per VISSA-style iteration.
> @param ratio_kept Numeric in (0, 1]. Fraction of features kept per iteration.
> @param threshold Numeric. Convergence / pruning threshold.
> @param floor_probability Numeric in [0, 0.5). Lower bound on per-feature retention probability.
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
p4a_vissa_select(ctx, cfg, &x_view, &y_view, /* hyperparams */, &res);
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
from pls4all._methods import vissa_select
with pls4all.Context() as ctx, pls4all.Config() as cfg:
    res = vissa_select(ctx, cfg, X, y, n_components=3)
# then: res.matrix("predictions"), res.matrix("coefficients"),
# res.vector("mask"), res.scalar("intercept"), …
```

:::

:::{tab-item} Python · pls4all.sklearn
:sync: python-sklearn
:class-label: lang-python

```python
from pls4all.sklearn import VISSASelector
mdl = VISSASelector(n_components=2, n_iterations=10, n_submodels=60, ratio_kept=0.1, threshold=0.5, floor_probability=0.05, n_folds=3, seed=0)
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
res <- pls4all_method("vissa_select", X, y,
                      n_components = 3L, params = list(n_iterations = 10L, n_submodels = 60L, ratio_kept = 0.1, threshold = 0.5, floor_probability = 0.05, seed = 42L))
# res is a named list with MethodResult arrays/scalars.
# selected_indices / top_k_intervals are 1-based.
```

:::

:::{tab-item} R · pls4all (raw fn)
:sync: r-raw
:class-label: lang-r

```r
library(pls4all)
res  <- vissa_select(X, Y, n_components,
              n_iterations = 20L, n_submodels = 100L,
              ratio_kept = 0.1, threshold = 0.5,
              floor_probability = 0.01, seed = 0L)
yhat <- pls4all_predict(res, X_test)
```

:::

:::{tab-item} MATLAB · pls4all (MEX)
:sync: matlab-mex
:class-label: lang-matlab

```matlab
res  = pls4all.fit("vissa_select", X, y, "NumComponents", 3);
yhat = predict(res, Xtest);
```

:::

:::{tab-item} MATLAB · pls4all (classdef)
:sync: matlab-classdef
:class-label: lang-matlab

_No idiomatic classdef wrapper — invoke `pls4all.fit("vissa_select", X, y, …)` directly from the unified MEX factory._

:::

::::


**Registry parity references** 📐

:::{card}
:class-card: external-refs

- 📐 **`ref.python_auswahl`** (python · python) — `auswahl` 0.9.0 · qualitative (rmse_rel ≤ 2e+00) — Python `auswahl.VISSA` from LSX-UniWue. Canonical Deng 2014 implementation. RNG diverges from pls4all splitmix64; mask metric.
:::

### Validation contract

Structural validation contract for `vissa_select`, exported by the declarative validation framework (`benchmarks/validation/registry/`) from `benchmarks.parity_timing.registry`. Regenerated by `python -m benchmarks.validation.export_registry --write`.

| Field | Value |
|------|-------|
| Reference kind | `external` |
| Canonical reference role | `python` |
| Declared references | Python |
| Comparator | `reference_parity` (external-reference validity gate) |
| Comparator metric | `rmse_rel` |
| Reference tolerance | `2e+00` (qualitative) |
| Binding tolerance | `1e-06` |
| Prediction key | `mask` |
| Required data flags | none |

**Benchmark cell parameters**

| Parameter | Value |
|-----------|-------|
| `floor_probability` | `0.05` |
| `n_components` | `3` |
| `n_features` | `25` |
| `n_iterations` | `10` |
| `n_samples` | `80` |
| `n_submodels` | `60` |
| `ratio_kept` | `0.1` |
| `seed` | `42` |
| `threshold` | `0.5` |

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
<tr class="bk-row"><td class="bk-name"><code>pls4all.cpp.blas</code></td><td class="parity parity-divergence parity-divergent">1.47</td><td class="ms ms-best">16.4 ms<span class="medal" title="fastest">🏆</span></td><td class="ms ms-best">133.9 ms<span class="medal" title="fastest">🏆</span></td><td class="ms ms-best">780.0 ms<span class="medal" title="fastest">🏆</span></td><td class="ms ms-best">1.4 s<span class="medal" title="fastest">🏆</span></td></tr>
<tr class="bk-row"><td class="bk-name"><code>pls4all.cpp.blas+omp</code></td><td class="parity parity-divergence parity-divergent">1.47</td><td class="ms">16.8 ms</td><td class="ms">137.3 ms</td><td class="ms">827.9 ms</td><td class="ms">1.5 s</td></tr>
<tr class="bk-row"><td class="bk-name"><code>pls4all.cpp.omp</code></td><td class="parity parity-divergence parity-divergent">1.47</td><td class="ms">18.4 ms</td><td class="ms">189.9 ms</td><td class="ms">1.2 s</td><td class="ms">1.9 s</td></tr>
<tr class="bk-row"><td class="bk-name"><code>pls4all.cpp.ref</code></td><td class="parity parity-divergence parity-divergent">1.47</td><td class="ms">19.9 ms</td><td class="ms">194.9 ms</td><td class="ms">1.2 s</td><td class="ms">1.9 s</td></tr>
</tbody>
<tbody class="lang-band lang-python"><tr class="lang-band-row" data-lang="python"><th colspan="6" scope="rowgroup"><span class="lang-band-dot"></span>Python · pls4all</th></tr>
<tr class="bk-row"><td class="bk-name"><code>pls4all.python</code></td><td class="parity parity-divergence parity-divergent">1.47</td><td class="ms">16.9 ms</td><td class="ms">137.3 ms</td><td class="ms">793.1 ms</td><td class="ms">1.6 s</td></tr>
<tr class="bk-row"><td class="bk-name"><code>pls4all.sklearn</code></td><td class="parity parity-divergence parity-divergent">1.47</td><td class="ms">16.9 ms</td><td class="ms">139.7 ms</td><td class="ms">819.7 ms</td><td class="ms">1.5 s</td></tr>
</tbody>
<tbody class="lang-band lang-r"><tr class="lang-band-row" data-lang="r"><th colspan="6" scope="rowgroup"><span class="lang-band-dot"></span>R · pls4all</th></tr>
<tr class="bk-row"><td class="bk-name"><code>pls4all.R</code></td><td class="parity parity-divergence parity-divergent">1.47</td><td class="ms">23.0 ms</td><td class="ms">263.5 ms</td><td class="ms">1.7 s</td><td class="ms">2.6 s</td></tr>
<tr class="bk-row"><td class="bk-name"><code>pls4all.R.formula</code></td><td class="parity parity-divergence parity-divergent">1.47</td><td class="ms">24.7 ms</td><td class="ms">264.8 ms</td><td class="ms">1.9 s</td><td class="ms">2.5 s</td></tr>
<tr class="bk-row"><td class="bk-name"><code>pls4all.R.mdatools</code></td><td class="parity parity-divergence parity-drift">—</td><td class="ms">24.2 ms</td><td class="ms">263.4 ms</td><td class="ms">1.8 s</td><td class="ms">2.9 s</td></tr>
<tr class="bk-row"><td class="bk-name"><code>pls4all.R.pls</code></td><td class="parity parity-divergence parity-drift">—</td><td class="ms">24.4 ms</td><td class="ms">252.6 ms</td><td class="ms">1.9 s</td><td class="ms">2.8 s</td></tr>
</tbody>
<tbody class="lang-band lang-matlab"><tr class="lang-band-row" data-lang="matlab"><th colspan="6" scope="rowgroup"><span class="lang-band-dot"></span>MATLAB · pls4all</th></tr>
<tr class="bk-row"><td class="bk-name"><code>pls4all.matlab</code></td><td class="parity parity-divergence parity-divergent">1.47</td><td class="ms">20.1 ms</td><td class="ms">202.2 ms</td><td class="ms">1.2 s</td><td class="ms">2.3 s</td></tr>
<tr class="bk-row"><td class="bk-name"><code>pls4all.matlab.classdef</code></td><td class="parity parity-divergence parity-divergent">1.47</td><td class="ms">20.9 ms</td><td class="ms">201.2 ms</td><td class="ms">1.2 s</td><td class="ms">2.3 s</td></tr>
</tbody>
<tbody class="lang-band lang-python"><tr class="lang-band-row" data-lang="python"><th colspan="6" scope="rowgroup"><span class="lang-band-dot"></span>Python · external</th></tr>
<tr class="bk-row truth-source truth-source-qualitative"><td class="bk-name"><span class="truth-mark" title="Registry parity reference (python): auswahl 0.9.0 — qualitative (rmse_rel ≤ 2e+00)">📐</span><code>ref.python_auswahl</code></td><td class="parity parity-divergence parity-ref-source">0</td><td class="ms">2.2 s</td><td class="ms">2.7 s</td><td class="ms">5.4 s</td><td class="ms">9.2 s</td></tr>
</tbody>
</table>
</div>

:::

::::


---

_See also_: [benchmark overview](../benchmarks/overview.md) · [methods index](index.md) · [interactive dashboard](../landing/dashboard.md)