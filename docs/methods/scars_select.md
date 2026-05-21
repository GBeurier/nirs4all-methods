# `scars_select` — SCARS — Stability-CARS

_Group_: **Variable selector** · _Registry tolerance_: `1.0`
 · _Parity reference_: **paper-only** (Zheng, K., Zhang, X., Tong, P., Yao, Y. & Du, Y. (2014). Pretreating near infrared spectra with fractional order Savitzky-Golay differentiation (FOSGD). Chemom. Intell. Lab. Syst. 132, 30-38. (SCARS hybrid; no widely installable port — smoke-tested.))

## Description

SCARS stability + CARS (§18 Phase 5h)

From the `pls4all.sklearn.SCARSSelector` docstring:

> Stability-CARS hybrid (Zheng 2014).

### Parameters

| Name | Type | Default | Notes |
|------|------|---------|-------|
| `n_components` | `int` | `2` | Number of latent components extracted (k). |
| `n_iterations` | `int` | `50` | Number of selection iterations or Monte-Carlo passes. |
| `min_features` | `int | None` | `None` | Minimum number of variables the selector is allowed to keep (defaults to `n_components`). |
| `sample_fraction` | `float` | `0.8` | Fraction of samples drawn per Monte-Carlo replicate. |
| `n_folds` | `int` | `3` | Number of cross-validation folds used inside the selector. |
| `seed` | `int` | `0` | Random seed for reproducible sampling/initialization. |

## Explanations

### Bibliographic source

Zheng, K., Li, Q., Wang, J., Geng, J., Cao, P., Sui, T., Wang, X. & Du, Y. (2012). *Stability competitive adaptive reweighted sampling (SCARS) and its applications to multivariate calibration of NIR spectra*. Chemometrics and Intelligent Laboratory Systems 112, 48–54.

### Mathematical principle

Replace CARS's coefficient-magnitude weights with **coefficient-stability** weights: $w_j = |\bar{b}_j| / s(b_j)$ from the bootstrap distribution. Stability-weighted retention is more robust to spurious high-magnitude coefficients caused by particular bootstrap subsamples.

Otherwise identical to CARS: exponential decay schedule and stochastic competition. SCARS typically improves CARS on datasets with strong baseline drift or where a few high-leverage samples dominate the coefficient estimates.

### Implementation

`p4a_scars_select`.

R roxygen note (`methods_extra.R::scars_select`):

> SCARS — Stability + CARS.
> @param n_components Integer. Number of latent components.
> @param n_iterations Integer >= 1. Number of outer-loop iterations.
> @param min_features Method-specific parameter. See the underlying `*_fit()` function for the exact semantics.
> @param sample_fraction Method-specific parameter. See the underlying `*_fit()` function for the exact semantics.
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
p4a_scars_select(ctx, cfg, &x_view, &y_view, /* hyperparams */, &res);
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
from pls4all._methods import scars_select
with pls4all.Context() as ctx, pls4all.Config() as cfg:
    res = scars_select(ctx, cfg, X, y, n_components=4)
# then: res.matrix("predictions"), res.matrix("coefficients"),
# res.vector("mask"), res.scalar("intercept"), …
```

:::

:::{tab-item} Python · pls4all.sklearn
:sync: python-sklearn
:class-label: lang-python

```python
from pls4all.sklearn import SCARSSelector
mdl = SCARSSelector(n_components=2, n_iterations=50, min_features=None, sample_fraction=0.8, n_folds=3, seed=0)
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
res <- pls4all_method("scars_select", X, y,
                      n_components = 4L, params = list(n_iterations = 8L, min_features = 5L, sample_fraction = 0.5, seed = 11L))
# res is a named list with MethodResult arrays/scalars.
# selected_indices / top_k_intervals are 1-based.
```

:::

:::{tab-item} R · pls4all (raw fn)
:sync: r-raw
:class-label: lang-r

```r
library(pls4all)
res  <- scars_select(X, Y, n_components,
              n_iterations = 50L, min_features = 5L,
              sample_fraction = 0.8, seed = 0L)
yhat <- pls4all_predict(res, X_test)
```

:::

:::{tab-item} MATLAB · pls4all (MEX)
:sync: matlab-mex
:class-label: lang-matlab

```matlab
res  = pls4all.fit("scars_select", X, y, "NumComponents", 4);
yhat = predict(res, Xtest);
```

:::

:::{tab-item} MATLAB · pls4all (classdef)
:sync: matlab-classdef
:class-label: lang-matlab

_No idiomatic classdef wrapper — invoke `pls4all.fit("scars_select", X, y, …)` directly from the unified MEX factory._

:::

::::


**Registry parity references** 📐

:::{card}
:class-card: external-refs

- 📜 **Paper-only** — no executable parity reference; the `pls4all` implementation is verified by a smoke fit only. Canonical citation: Zheng, K., Zhang, X., Tong, P., Yao, Y. & Du, Y. (2014). Pretreating near infrared spectra with fractional order Savitzky-Golay differentiation (FOSGD). Chemom. Intell. Lab. Syst. 132, 30-38. (SCARS hybrid; no widely installable port — smoke-tested.)
:::

### Validation contract

Structural validation contract for `scars_select`, exported by the declarative validation framework (`benchmarks/validation/registry/`) from `benchmarks.parity_timing.registry`. Regenerated by `python -m benchmarks.validation.export_registry --write`.

| Field | Value |
|------|-------|
| Reference kind | `paper_only` |
| Canonical reference role | — |
| Declared references | — |
| Comparator | `binding_parity` (binding-consistency gate) |
| Comparator metric | `max_abs_diff` |
| Reference tolerance | `1e+00` (qualitative) |
| Binding tolerance | `1e-06` |
| Prediction key | `mask` |
| Required data flags | none |
| Paper-only citation | Zheng, K., Zhang, X., Tong, P., Yao, Y. & Du, Y. (2014). Pretreating near infrared spectra with fractional order Savitzky-Golay differentiation (FOSGD). Chemom. Intell. Lab. Syst. 132, 30-38. (SCARS hybrid; no widely installable port — smoke-tested.) |

**Benchmark cell parameters**

| Parameter | Value |
|-----------|-------|
| `min_features` | `5` |
| `n_components` | `4` |
| `n_features` | `40` |
| `n_iterations` | `8` |
| `n_samples` | `200` |
| `sample_fraction` | `0.5` |
| `seed` | `11` |

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

**Reference gate**: strict release — `rmse_rel ≤ 1e-03` is required; `rmse_rel ≤ 1e-06` is displayed as exact.

::::{tab-set}
:class: parity-tabs

:::{tab-item} 1 thread
:sync: threads-1

<div class="parity-table-wrap">
<table class="docutils parity-grouped">
<thead><tr><th scope="col">Backend</th><th scope="col">Divergence</th><th class="size-col" scope="col">100×50 (ms)</th></tr></thead>
<tbody class="lang-band lang-cpp"><tr class="lang-band-row" data-lang="cpp"><th colspan="3" scope="rowgroup"><span class="lang-band-dot"></span>C++ native · libp4a</th></tr>
<tr class="bk-row"><td class="bk-name"><code>pls4all.cpp.blas+omp</code></td><td class="parity parity-divergence parity-not_available">—</td><td class="ms ms-best">1.39 ms<span class="medal" title="fastest">🏆</span></td></tr>
</tbody>
<tbody class="lang-band lang-python"><tr class="lang-band-row" data-lang="python"><th colspan="3" scope="rowgroup"><span class="lang-band-dot"></span>Python · pls4all</th></tr>
<tr class="bk-row"><td class="bk-name"><code>pls4all.python</code></td><td class="parity parity-divergence parity-not_available">—</td><td class="ms">1.50 ms</td></tr>
<tr class="bk-row"><td class="bk-name"><code>pls4all.sklearn</code></td><td class="parity parity-divergence parity-not_available">—</td><td class="ms">1.63 ms</td></tr>
</tbody>
<tbody class="lang-band lang-r"><tr class="lang-band-row" data-lang="r"><th colspan="3" scope="rowgroup"><span class="lang-band-dot"></span>R · pls4all</th></tr>
<tr class="bk-row"><td class="bk-name"><code>pls4all.R</code></td><td class="parity parity-divergence parity-not_available">—</td><td class="ms">3.10 ms</td></tr>
<tr class="bk-row"><td class="bk-name"><code>pls4all.R.formula</code></td><td class="parity parity-divergence parity-not_available">—</td><td class="ms">4.32 ms</td></tr>
</tbody>
<tbody class="lang-band lang-matlab"><tr class="lang-band-row" data-lang="matlab"><th colspan="3" scope="rowgroup"><span class="lang-band-dot"></span>MATLAB · pls4all</th></tr>
<tr class="bk-row"><td class="bk-name"><code>pls4all.matlab</code></td><td class="parity parity-divergence parity-not_available">—</td><td class="ms">2.11 ms</td></tr>
<tr class="bk-row"><td class="bk-name"><code>pls4all.matlab.classdef</code></td><td class="parity parity-divergence parity-not_available">—</td><td class="ms">2.60 ms</td></tr>
</tbody>
</table>
</div>

:::

::::


---

_See also_: [benchmark overview](../benchmarks/overview.md) · [methods index](index.md) · [interactive dashboard](../landing/dashboard.md)