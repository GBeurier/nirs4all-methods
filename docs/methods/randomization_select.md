# `randomization_select` — Randomisation test (Y-permutation)

_Group_: **Variable selector** · _Registry tolerance_: `1.3`

## Description

Randomization-test selector (§18 Phase 5o)

From the `pls4all.sklearn.RandomizationSelector` docstring:

> Randomization-test PLS selector (Y-permutation p-values).

> **Registry note** — Base R: SIMPLS coefs vs permuted-Y null distribution. Selects features with p < alpha. Different RNG than pls4all splitmix64; mask metric ~0=perfect, ~1=half disagree, ~1.41=disjoint.

### Parameters

| Name | Type | Default | Notes |
|------|------|---------|-------|
| `n_components` | `int` | `2` | Number of latent components extracted (k). |
| `n_permutations` | `int` | `200` | Number of Y-permutations used to build the null distribution. |
| `randomization_seed` | `int` | `0` | Seed for the permutation generator. |
| `alpha` | `float` | `0.05` | Significance level for the permutation-based variable retention test. |

## Explanations

### Bibliographic source

Westad, F. & Martens, H. (2000). *Variable selection in near infrared spectroscopy based on significance testing in partial least squares regression*. JNIRS 8(2), 117–124.

### Mathematical principle

Compute the observed PLS coefficient magnitudes $|b_j^{\mathrm{obs}}|$, then permute $\mathbf{y}$ $M$ times, refit PLS each time, and collect $|b_j^{(m)}|$. The empirical p-value of feature $j$ is $p_j = \frac{1 + \#\{m : |b_j^{(m)}| \ge |b_j^{\mathrm{obs}}|\}}{1 + M}$. Retain features with $p_j < \alpha$.

Y-permutation is the gold standard for **null-calibrated** significance testing in PLS — no distributional assumptions, no asymptotic approximations. Cost is $M$× a fit but trivially parallelisable.

Critically, Y-permutation tests the joint hypothesis 'feature $j$ contributes to $y$'; multiple-testing correction (Benjamini-Hochberg) is recommended for $p \gg 100$.

### Implementation

`p4a_randomization_select`.

R roxygen note (`methods_extra.R::randomization_select`):

> Randomization test selector.
> @param n_components Integer. Number of latent components.
> @param n_permutations Method-specific parameter. See the underlying `*_fit()` function for the exact semantics.
> @param randomization_seed Method-specific parameter. See the underlying `*_fit()` function for the exact semantics.
> @param alpha Numeric in [0, 1]. Elastic-net / penalty mixing parameter.
> @param X Numeric matrix of predictors (rows = samples, cols = features).
> @param Y Numeric matrix or vector of responses, with one row per sample.
> @export

### Usage

All four pls4all bindings dispatch into the same C kernel; the external libraries on the right are the parity references registered in `benchmarks.parity_timing.registry`. Switch tabs to read the same fit in your language.

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
p4a_randomization_select(ctx, cfg, &x_view, &y_view, /* hyperparams */, &res);
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
from pls4all._methods import randomization_select
with pls4all.Context() as ctx, pls4all.Config() as cfg:
    res = randomization_select(ctx, cfg, X, y, n_components=4)
# then: res.matrix("predictions"), res.matrix("coefficients"),
# res.vector("mask"), res.scalar("intercept"), …
```

:::

:::{tab-item} Python · pls4all.sklearn
:sync: python-sklearn
:class-label: lang-python

```python
from pls4all.sklearn import RandomizationSelector
mdl = RandomizationSelector(n_components=2, n_permutations=200, randomization_seed=0, alpha=0.05)
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
res <- pls4all_method("randomization_select", X, y,
                      n_components = 4L, params = list(n_permutations = 50L, alpha = 0.05, randomization_seed = 11L))
# res is a named list with MethodResult arrays/scalars.
# selected_indices / top_k_intervals are 1-based.
```

:::

:::{tab-item} R · pls4all (raw fn)
:sync: r-raw
:class-label: lang-r

```r
library(pls4all)
res  <- randomization_select(X, Y, n_components,
                      n_permutations = 100L,
                      randomization_seed = 0L, alpha = 0.05)
yhat <- pls4all_predict(res, X_test)
```

:::

:::{tab-item} MATLAB · pls4all (MEX)
:sync: matlab-mex
:class-label: lang-matlab

```matlab
res  = pls4all.fit("randomization_select", X, y, "NumComponents", 4);
yhat = predict(res, Xtest);
```

:::

:::{tab-item} MATLAB · pls4all (classdef)
:sync: matlab-classdef
:class-label: lang-matlab

_No idiomatic classdef wrapper — invoke `pls4all.fit("randomization_select", X, y, …)` directly from the unified MEX factory._

:::

::::


**Registry parity references** 📐

:::{card}
:class-card: external-refs

- 📐 **`ref.r_pls_stats`** (R · r) — `pls+stats` R 4.3.3 · qualitative (rmse_rel ≤ 1e+00) — Base R: SIMPLS coefficients vs permuted-Y null distribution. Selects features with empirical p-value < alpha. Same idea as pls4all's randomization_test selector.
:::

### Benchmarks

Median wall-clock per cell from [`benchmarks/cross_binding/results/full_matrix.csv`](../benchmarks/overview.md). Verdict legend: ✓ exact · ≈ drift · ✗ divergent · ⊘ not available in lib · — not run · ⚠ error. The fastest backend per column is marked with a 🏆 medal. Rows tagged with **📐** are *also* declared in [`benchmarks/parity_timing/registry.py`](../benchmarks/methodology.md) as the canonical parity references for this method (`python_reference` / `r_reference` / `extra_references`). Cell parity in this table is measured against the cross-binding reference (`pls4all.cpp` blas-omp, 1 thread); the 📐 icon points at the *library-of-record* the parity gate ultimately answers to. Hover the icon to see the role and tolerance band.

::::{tab-set}
:class: parity-tabs

:::{tab-item} 1 thread
:sync: threads-1

<div class="parity-table-wrap">
<table class="docutils parity-grouped">
<thead><tr><th scope="col">Backend</th><th scope="col">Parity</th><th class="size-col" scope="col">200×40 (ms)</th></tr></thead>
<tbody class="lang-band lang-python"><tr class="lang-band-row" data-lang="python"><th colspan="3" scope="rowgroup"><span class="lang-band-dot"></span>Python · pls4all</th></tr>
<tr class="bk-row"><td class="bk-name"><code>pls4all.registry</code></td><td class="parity parity-exact">✓ ref</td><td class="ms ms-best">171.7 ms<span class="medal" title="fastest">🏆</span></td></tr>
</tbody>
<tbody class="lang-band lang-r"><tr class="lang-band-row" data-lang="r"><th colspan="3" scope="rowgroup"><span class="lang-band-dot"></span>R · external</th></tr>
<tr class="bk-row truth-source truth-source-qualitative"><td class="bk-name"><span class="truth-mark" title="Registry parity reference (r): pls+stats R 4.3.3 — qualitative (rmse_rel ≤ 1e+00)">📐</span><code>ref.r_pls_stats</code></td><td class="parity parity-exact">✓ ref</td><td class="ms">180.1 ms</td></tr>
</tbody>
</table>
</div>

:::

::::


---

_See also_: [benchmark overview](../benchmarks/overview.md) · [methods index](index.md) · [interactive dashboard](../landing/dashboard.md)