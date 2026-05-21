# `random_subspace_pls` — Random-subspace PLS

_Group_: **Ensemble** · _Registry tolerance_: `2.0`

## Description

Random-subspace PLS (§20)

From the `pls4all.sklearn.RandomSubspacePLSRegression` docstring:

> Random-subspace PLS — Ho 1998.

> **Registry note** — sklearn `BaggingRegressor(PLSRegression(), max_features=k, bootstrap=False)`. Same full-row random subspace shape as pls4all; RNG/feature-subset order still differs, so parity is qualitative.

### Parameters

| Name | Type | Default | Notes |
|------|------|---------|-------|
| `n_components` | `int` | `2` | Number of latent components extracted (k). |
| `n_estimators` | `int` | `50` | Number of base PLS sub-models in the ensemble. |
| `features_per_subspace` | `int` | `10` | Number of features randomly drawn per random-subspace base learner. |
| `seed` | `int` | `0` | Random seed for reproducible sampling/initialization. |

## Explanations

### Bibliographic source

Ho, T. K. (1998). *The random subspace method for constructing decision forests*. IEEE TPAMI 20(8), 832–844. — adapted for PLS regressors.

### Mathematical principle

Each ensemble member fits PLS on a random subset of $m \ll p$ features. Compared to bagging (which randomises rows), random subspaces randomise **columns**, which is a much stronger variance-reduction mechanism for high-dimensional collinear data like NIR spectra: different subsets pick up different bands of the spectrum, and averaging across them smooths out band-specific noise.

Variance per member is higher than a full-feature PLS (less information per fit), but the ensemble average outperforms a single fit when the underlying truth is spread across many weakly-correlated features. Choosing $m \approx \sqrt{p}$ is a Breiman-style default; for spectra a more informed choice respects band widths.

Note that prediction on a new sample requires evaluating every member on **its own subset** of features, so the feature-index map must be stored per member.

### Implementation

`p4a_random_subspace_pls_fit`.

R roxygen note (`sklearn_extra.R::random_subspace_pls`):

> Random-subspace PLS — formula entry point.

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
p4a_random_subspace_pls_fit(ctx, cfg, &x_view, &y_view, /* hyperparams */, &res);
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
from pls4all._methods import random_subspace_pls_fit
with pls4all.Context() as ctx, pls4all.Config() as cfg:
    res = random_subspace_pls_fit(ctx, cfg, X, y, n_components=4)
# then: res.matrix("predictions"), res.matrix("coefficients"),
# res.vector("mask"), res.scalar("intercept"), …
```

:::

:::{tab-item} Python · pls4all.sklearn
:sync: python-sklearn
:class-label: lang-python

```python
from pls4all.sklearn import RandomSubspacePLSRegression
mdl = RandomSubspacePLSRegression(n_components=2, n_estimators=50, features_per_subspace=10, seed=0)
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
res <- pls4all_method("random_subspace_pls", X, y,
                      n_components = 4L, params = list(n_estimators = 10L, features_per_subspace = 20L, seed = 42L))
# res is a named list with MethodResult arrays/scalars.
# selected_indices / top_k_intervals are 1-based.
```

:::

:::{tab-item} R · pls4all (raw fn)
:sync: r-raw
:class-label: lang-r

```r
library(pls4all)
res  <- random_subspace_pls_fit(X, Y, n_components,
                         n_estimators = 50L,
                         features_per_subspace = 10L,
                         seed = 0L)
yhat <- pls4all_predict(res, X_test)
```

:::

:::{tab-item} R · pls4all (formula+S3)
:sync: r-formula
:class-label: lang-r

```r
library(pls4all)
fit  <- random_subspace_pls(y ~ ., data = train, ncomp = 4L)
yhat <- predict(fit, newdata = test)
summary(fit)
```

:::

:::{tab-item} MATLAB · pls4all (MEX)
:sync: matlab-mex
:class-label: lang-matlab

```matlab
res  = pls4all.fit("random_subspace_pls", X, y, "NumComponents", 4);
yhat = predict(res, Xtest);
```

:::

:::{tab-item} MATLAB · pls4all (classdef)
:sync: matlab-classdef
:class-label: lang-matlab

_No idiomatic classdef wrapper — invoke `pls4all.fit("random_subspace_pls", X, y, …)` directly from the unified MEX factory._

:::

::::


**Registry parity references** 📐

:::{card}
:class-card: external-refs

- 📐 **`ref.python_scikit_learn`** (python · python) — `scikit-learn` 1.8.0 · qualitative (rmse_rel ≤ 2e+00) — sklearn `BaggingRegressor(PLSRegression(), max_features=…, bootstrap=False)`. Random feature subspaces with full sample rows, matching pls4all's sampling shape. RNG differs from pls4all; qualitative parity.
:::

### Validation contract

Structural validation contract for `random_subspace_pls`, exported by the declarative validation framework (`benchmarks/validation/registry/`) from `benchmarks.parity_timing.registry`. Regenerated by `python -m benchmarks.validation.export_registry --write`.

| Field | Value |
|------|-------|
| Reference kind | `external` |
| Canonical reference role | `python` |
| Declared references | Python |
| Comparator | `reference_parity` (external-reference validity gate) |
| Comparator metric | `rmse_rel` |
| Reference tolerance | `2e+00` (qualitative) |
| Binding tolerance | `1e-06` |
| Prediction key | `predictions` |
| Required data flags | none |

**Benchmark cell parameters**

| Parameter | Value |
|-----------|-------|
| `features_per_subspace` | `20` |
| `n_components` | `4` |
| `n_estimators` | `10` |
| `n_features` | `30` |
| `n_samples` | `200` |
| `seed` | `42` |

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
<tr class="bk-row"><td class="bk-name"><code>pls4all.cpp.blas</code></td><td class="parity parity-divergence parity-divergent">1.25</td><td class="ms">1.71 ms</td><td class="ms">8.30 ms</td><td class="ms">39.2 ms</td><td class="ms">82.0 ms</td></tr>
<tr class="bk-row"><td class="bk-name"><code>pls4all.cpp.blas+omp</code></td><td class="parity parity-divergence parity-divergent">1.25</td><td class="ms">1.16 ms</td><td class="ms">8.44 ms</td><td class="ms">39.7 ms</td><td class="ms">83.3 ms</td></tr>
<tr class="bk-row"><td class="bk-name"><code>pls4all.cpp.omp</code></td><td class="parity parity-divergence parity-divergent">1.25</td><td class="ms ms-best">1.08 ms<span class="medal" title="fastest">🏆</span></td><td class="ms ms-best">8.20 ms<span class="medal" title="fastest">🏆</span></td><td class="ms">39.3 ms</td><td class="ms ms-best">81.8 ms<span class="medal" title="fastest">🏆</span></td></tr>
<tr class="bk-row"><td class="bk-name"><code>pls4all.cpp.ref</code></td><td class="parity parity-divergence parity-divergent">1.25</td><td class="ms">1.08 ms</td><td class="ms">8.24 ms</td><td class="ms">39.5 ms</td><td class="ms">81.9 ms</td></tr>
</tbody>
<tbody class="lang-band lang-python"><tr class="lang-band-row" data-lang="python"><th colspan="6" scope="rowgroup"><span class="lang-band-dot"></span>Python · pls4all</th></tr>
<tr class="bk-row"><td class="bk-name"><code>pls4all.python</code></td><td class="parity parity-divergence parity-divergent">0.66</td><td class="ms">1.15 ms</td><td class="ms">8.22 ms</td><td class="ms ms-best">38.7 ms<span class="medal" title="fastest">🏆</span></td><td class="ms">82.8 ms</td></tr>
<tr class="bk-row"><td class="bk-name"><code>pls4all.sklearn</code></td><td class="parity parity-divergence parity-divergent">1.25</td><td class="ms">1.33 ms</td><td class="ms">8.50 ms</td><td class="ms">40.5 ms</td><td class="ms">83.3 ms</td></tr>
</tbody>
<tbody class="lang-band lang-r"><tr class="lang-band-row" data-lang="r"><th colspan="6" scope="rowgroup"><span class="lang-band-dot"></span>R · pls4all</th></tr>
<tr class="bk-row"><td class="bk-name"><code>pls4all.R</code></td><td class="parity parity-divergence parity-divergent">0.66</td><td class="ms">3.89 ms</td><td class="ms">44.3 ms</td><td class="ms">400.5 ms</td><td class="ms">546.7 ms</td></tr>
<tr class="bk-row"><td class="bk-name"><code>pls4all.R.formula</code></td><td class="parity parity-divergence parity-divergent">0.66</td><td class="ms">5.10 ms</td><td class="ms">61.0 ms</td><td class="ms">618.6 ms</td><td class="ms">701.7 ms</td></tr>
<tr class="bk-row"><td class="bk-name"><code>pls4all.R.mdatools</code></td><td class="parity parity-divergence parity-divergent">0.66</td><td class="ms">5.42 ms</td><td class="ms">62.6 ms</td><td class="ms">588.5 ms</td><td class="ms">668.9 ms</td></tr>
<tr class="bk-row"><td class="bk-name"><code>pls4all.R.pls</code></td><td class="parity parity-divergence parity-divergent">0.66</td><td class="ms">5.22 ms</td><td class="ms">67.1 ms</td><td class="ms">583.5 ms</td><td class="ms">650.2 ms</td></tr>
</tbody>
<tbody class="lang-band lang-matlab"><tr class="lang-band-row" data-lang="matlab"><th colspan="6" scope="rowgroup"><span class="lang-band-dot"></span>MATLAB · pls4all</th></tr>
<tr class="bk-row"><td class="bk-name"><code>pls4all.matlab</code></td><td class="parity parity-divergence parity-divergent">0.66</td><td class="ms">1.85 ms</td><td class="ms">13.1 ms</td><td class="ms">63.8 ms</td><td class="ms">127.9 ms</td></tr>
<tr class="bk-row"><td class="bk-name"><code>pls4all.matlab.classdef</code></td><td class="parity parity-divergence parity-divergent">0.66</td><td class="ms">2.41 ms</td><td class="ms">13.6 ms</td><td class="ms">63.2 ms</td><td class="ms">128.5 ms</td></tr>
</tbody>
<tbody class="lang-band lang-python"><tr class="lang-band-row" data-lang="python"><th colspan="6" scope="rowgroup"><span class="lang-band-dot"></span>Python · external</th></tr>
<tr class="bk-row truth-source truth-source-qualitative"><td class="bk-name"><span class="truth-mark" title="Registry parity reference (python): scikit-learn 1.8.0 — qualitative (rmse_rel ≤ 2e+00)">📐</span><code>ref.python_scikit_learn</code></td><td class="parity parity-divergence parity-ref-source">0</td><td class="ms">10.4 ms</td><td class="ms">17.5 ms</td><td class="ms">50.8 ms</td><td class="ms">93.9 ms</td></tr>
</tbody>
</table>
</div>

:::

::::


---

_See also_: [benchmark overview](../benchmarks/overview.md) · [methods index](index.md) · [interactive dashboard](../landing/dashboard.md)