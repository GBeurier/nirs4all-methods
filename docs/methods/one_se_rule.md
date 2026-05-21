# `one_se_rule` — One-SE rule for component selection

_Group_: **Diagnostic** · _Registry tolerance_: `10.0`

## Description

One-SE component selection rule (§10)

> **Registry note** — R `pls::plsr` with k-fold CV + onesigma rule. pls4all's one_se_rule_compute consumes a synthetic fold-RMSE matrix while R reads training data — comparison is on the rule's RMSE-per-component vector, scale-divergent.

### Parameters

| Name | Type | Default | Notes |
|------|------|---------|-------|
| `max_components` | `int` | `8` | registry benchmark cell value |
| `n_folds` | `int` | `5` | registry benchmark cell value |

## Explanations

### Bibliographic source

Hastie, T., Tibshirani, R. & Friedman, J. (2009). *The Elements of Statistical Learning*, 2nd ed., Springer, §7.10.

### Mathematical principle

Cross-validated RMSE as a function of $k$ is typically U-shaped with a relatively flat minimum. Picking the absolute minimum $k^{\star}$ can over-fit because it exploits sampling noise. The one-SE rule instead picks the **smallest** $k$ whose CV-RMSE is within one standard error of $\mathrm{RMSE}(k^{\star})$.

This yields a more parsimonious model with negligible predictive cost — the smaller-$k$ alternative is indistinguishable from the optimum within the noise of the CV estimate. The rule is non-parametric (no assumption about the CV-RMSE distribution) and is the standard practice in regularised regression (`glmnet`, `pls::pls`).

Inputs: a fold × component RMSE matrix from cross-validation. Output: an integer component count.

### Implementation

`p4a_one_se_rule_compute`. Returns an integer.

R roxygen note (`methods_extra.R::one_se_rule`):

> One-SE rule from a (max_components × n_folds) fold RMSE matrix.
> @param fold_rmse_matrix Method-specific parameter. See the underlying `*_fit()` function for the exact semantics.
> @export

MATLAB header (`bindings/matlab/+pls4all/one_se_rule.m`):

```text
pls4all.one_se_rule  One-SE component selection from a fold RMSE matrix.
 fold_rmse_matrix: (max_components × n_folds) matrix of fold RMSE values.
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
p4a_one_se_rule_compute(ctx, cfg, &x_view, &y_view, /* hyperparams */, &res);
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
from pls4all._methods import one_se_rule_compute
with pls4all.Context() as ctx, pls4all.Config() as cfg:
    res = one_se_rule_compute(ctx, cfg, X, y)
# then: res.matrix("predictions"), res.matrix("coefficients"),
# res.vector("mask"), res.scalar("intercept"), …
```

:::

:::{tab-item} Python · pls4all.sklearn
:sync: python-sklearn
:class-label: lang-python

```python
from pls4all.sklearn import one_se_rule
result = one_se_rule(X, y, n_components=2)
```

:::

:::{tab-item} R · pls4all_method()
:sync: r-dispatcher
:class-label: lang-r

```r
library(pls4all)
# Unified low-level dispatcher (May 2026 R cleanup):
res <- pls4all_method("one_se_rule", X, y,
                      n_components = 2L, params = list(max_components = 8L, n_folds = 5L))
# res is a named list with MethodResult arrays/scalars.
# selected_indices / top_k_intervals are 1-based.
```

:::

:::{tab-item} R · pls4all (raw fn)
:sync: r-raw
:class-label: lang-r

```r
library(pls4all)
res  <- one_se_rule(fold_rmse_matrix)
yhat <- pls4all_predict(res, X_test)
```

:::

:::{tab-item} MATLAB · pls4all (MEX)
:sync: matlab-mex
:class-label: lang-matlab

```matlab
res = pls4all.one_se_rule(X, y, 2);
% see header of bindings/matlab/+pls4all/one_se_rule.m for full
% parameter surface:
%   res = one_se_rule(fold_rmse_matrix)
yhat = predict(res, Xtest);
```

:::

:::{tab-item} MATLAB · pls4all (classdef)
:sync: matlab-classdef
:class-label: lang-matlab

_No idiomatic classdef wrapper — invoke `pls4all.fit("one_se_rule", X, y, …)` directly from the unified MEX factory._

:::

::::


**Registry parity references** 📐

:::{card}
:class-card: external-refs

- 📐 **`ref.r_pls`** (R · r) — `pls` 2.8.5 · qualitative (rmse_rel ≤ 1e+01) — R `pls::plsr` + `pls::selectNcomp(method='onesigma')`. We compare the per-component CV-RMSE vectors; pls4all's one_se_rule_compute consumes a synthetic fold-RMSE matrix so the comparison is on rule output, not training data.
:::

### Validation contract

Structural validation contract for `one_se_rule`, exported by the declarative validation framework (`benchmarks/validation/registry/`) from `benchmarks.parity_timing.registry`. Regenerated by `python -m benchmarks.validation.export_registry --write`.

| Field | Value |
|------|-------|
| Reference kind | `external` |
| Canonical reference role | `r` |
| Declared references | R |
| Comparator | `reference_parity` (external-reference validity gate) |
| Comparator metric | `rmse_rel` |
| Reference tolerance | `1e+01` (qualitative) |
| Binding tolerance | `1e-06` |
| Prediction key | `mean_rmse_per_component` |
| Required data flags | none |

**Benchmark cell parameters**

| Parameter | Value |
|-----------|-------|
| `max_components` | `8` |
| `n_features` | `30` |
| `n_folds` | `5` |
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
<tr class="bk-row"><td class="bk-name"><code>pls4all.cpp.blas</code></td><td class="parity parity-divergence parity-divergent">0.67</td><td class="ms">0.98 ms</td><td class="ms">8.47 ms</td><td class="ms">39.6 ms</td><td class="ms">77.6 ms</td></tr>
<tr class="bk-row"><td class="bk-name"><code>pls4all.cpp.blas+omp</code></td><td class="parity parity-divergence parity-divergent">0.67</td><td class="ms">0.99 ms</td><td class="ms">7.99 ms</td><td class="ms ms-best">39.1 ms<span class="medal" title="fastest">🏆</span></td><td class="ms">78.0 ms</td></tr>
<tr class="bk-row"><td class="bk-name"><code>pls4all.cpp.omp</code></td><td class="parity parity-divergence parity-divergent">0.67</td><td class="ms">0.97 ms</td><td class="ms">8.06 ms</td><td class="ms">39.1 ms</td><td class="ms">79.2 ms</td></tr>
<tr class="bk-row"><td class="bk-name"><code>pls4all.cpp.ref</code></td><td class="parity parity-divergence parity-divergent">0.67</td><td class="ms">0.97 ms</td><td class="ms ms-best">7.81 ms<span class="medal" title="fastest">🏆</span></td><td class="ms">39.1 ms</td><td class="ms">78.2 ms</td></tr>
</tbody>
<tbody class="lang-band lang-python"><tr class="lang-band-row" data-lang="python"><th colspan="6" scope="rowgroup"><span class="lang-band-dot"></span>Python · pls4all</th></tr>
<tr class="bk-row"><td class="bk-name"><code>pls4all.python</code></td><td class="parity parity-divergence parity-divergent">0.67</td><td class="ms ms-best">0.95 ms<span class="medal" title="fastest">🏆</span></td><td class="ms">8.15 ms</td><td class="ms">39.8 ms</td><td class="ms ms-best">77.3 ms<span class="medal" title="fastest">🏆</span></td></tr>
<tr class="bk-row"><td class="bk-name"><code>pls4all.sklearn</code></td><td class="parity parity-divergence parity-divergent">0.67</td><td class="ms">1.00 ms</td><td class="ms">7.94 ms</td><td class="ms">39.7 ms</td><td class="ms">79.3 ms</td></tr>
</tbody>
<tbody class="lang-band lang-r"><tr class="lang-band-row" data-lang="r"><th colspan="6" scope="rowgroup"><span class="lang-band-dot"></span>R · pls4all</th></tr>
<tr class="bk-row"><td class="bk-name"><code>pls4all.R</code></td><td class="parity parity-divergence parity-divergent">0.67</td><td class="ms">3.05 ms</td><td class="ms">48.5 ms</td><td class="ms">449.9 ms</td><td class="ms">506.9 ms</td></tr>
<tr class="bk-row"><td class="bk-name"><code>pls4all.R.formula</code></td><td class="parity parity-divergence parity-divergent">0.67</td><td class="ms">4.37 ms</td><td class="ms">64.3 ms</td><td class="ms">574.1 ms</td><td class="ms">710.7 ms</td></tr>
<tr class="bk-row"><td class="bk-name"><code>pls4all.R.mdatools</code></td><td class="parity parity-divergence parity-divergent">0.67</td><td class="ms">4.72 ms</td><td class="ms">73.2 ms</td><td class="ms">584.7 ms</td><td class="ms">728.9 ms</td></tr>
<tr class="bk-row"><td class="bk-name"><code>pls4all.R.pls</code></td><td class="parity parity-divergence parity-divergent">0.67</td><td class="ms">4.46 ms</td><td class="ms">79.0 ms</td><td class="ms">570.5 ms</td><td class="ms">741.2 ms</td></tr>
</tbody>
<tbody class="lang-band lang-matlab"><tr class="lang-band-row" data-lang="matlab"><th colspan="6" scope="rowgroup"><span class="lang-band-dot"></span>MATLAB · pls4all</th></tr>
<tr class="bk-row"><td class="bk-name"><code>pls4all.matlab</code></td><td class="parity parity-divergence parity-divergent">0.67</td><td class="ms">1.70 ms</td><td class="ms">13.9 ms</td><td class="ms">63.0 ms</td><td class="ms">128.4 ms</td></tr>
<tr class="bk-row"><td class="bk-name"><code>pls4all.matlab.classdef</code></td><td class="parity parity-divergence parity-divergent">0.67</td><td class="ms">2.13 ms</td><td class="ms">17.4 ms</td><td class="ms">63.0 ms</td><td class="ms">128.7 ms</td></tr>
</tbody>
<tbody class="lang-band lang-r"><tr class="lang-band-row" data-lang="r"><th colspan="6" scope="rowgroup"><span class="lang-band-dot"></span>R · external</th></tr>
<tr class="bk-row truth-source truth-source-qualitative"><td class="bk-name"><span class="truth-mark" title="Registry parity reference (r): pls 2.8.5 — qualitative (rmse_rel ≤ 1e+01)">📐</span><code>ref.r_pls</code></td><td class="parity parity-divergence parity-ref-source">0</td><td class="ms">13.2 ms</td><td class="ms">78.1 ms</td><td class="ms">608.2 ms</td><td class="ms">708.6 ms</td></tr>
</tbody>
</table>
</div>

:::

::::


---

_See also_: [benchmark overview](../benchmarks/overview.md) · [methods index](index.md) · [interactive dashboard](../landing/dashboard.md)