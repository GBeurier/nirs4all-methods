# `random_subspace_pls` — Random-subspace PLS

_Group_: **Ensemble** · _Registry tolerance_: `1e-06`

## Description

Random-subspace PLS (§20)

From the `pls4all.sklearn.RandomSubspacePLSRegression` docstring:

> Random-subspace PLS — Ho 1998.

> **Registry note** — sklearn `BaggingRegressor(PLSRegression(scale=False), max_features=k, bootstrap=False, bootstrap_features=False, max_samples=1.0)`. pls4all's default now mirrors this convention exactly (same RNG, feature-subset order, and prediction averaging), so the gate is bit-for-bit. The legacy single-pass C++ kernel (splitmix feature shuffle + coefficient averaging) is opt-in via ``legacy=True``.

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

`n4m_random_subspace_pls_fit`.

### Usage

Every pls4all binding tab dispatches into the same C kernel; the external libraries listed at the bottom of the page are the parity references registered in `benchmarks.parity_timing.registry`. Switch tabs to read the same fit in your language. The R package now ships drop-in-compatible facades for the CRAN `pls` package (`plsr`, `pcr`, `mvr`) and for the `mdatools::pls(x, y, ...)` matrix idiom — those tabs appear only on the methods that have a meaningful equivalence.

**pls4all bindings**

::::{tab-set}
:class: pls4all-bindings

:::{tab-item} C ABI · libn4m
:sync: c
:class-label: lang-c

```c
/* C ABI — libn4m */
n4m_context_t* ctx = n4m_context_create();
n4m_config_t*  cfg = n4m_config_create();
n4m_method_result_t* res = NULL;
n4m_random_subspace_pls_fit(ctx, cfg, &x_view, &y_view, /* hyperparams */, &res);
/* … read coefficients / mask / scores via */
/* n4m_method_result_get_double_matrix / vector / scalar … */
n4m_method_result_destroy(res);
n4m_config_destroy(cfg);
n4m_context_destroy(ctx);
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

- 📐 **`ref.python_scikit_learn`** (python · python) — `scikit-learn` 1.8.0 · strict (rmse_rel ≤ 1e-06) — sklearn `BaggingRegressor(PLSRegression(), max_features=…, bootstrap=False)`. Random feature subspaces with full sample rows, matching pls4all's sampling shape. RNG differs from pls4all; qualitative parity.
:::

### Benchmarks

Adaptive wall-clock per cell measured against [`full_matrix.csv`](../benchmarks/overview.md). Only backends that implement this method are listed; libraries without the method are omitted.

**Verdict** &nbsp;·&nbsp; ✓ ref / ≈ ref / ~ shape mark a reference-gate pass at strict / relaxed / qualitative tolerance &nbsp;·&nbsp; ✓ bind = pls4all binding agrees with the C++ baseline &nbsp;·&nbsp; ✗ divergent &nbsp;·&nbsp; ⚠ error &nbsp;·&nbsp; — not run. The fastest backend per column is marked 🏆.

**Reference gate**: strict — numeric equivalence (`rmse_rel_tol ≤ 1e-06`).

Rows tagged with **📐** are the canonical parity references for this method (declared in [`parity_timing.registry`](../benchmarks/methodology.md)). C++ and external rows show reference parity; pls4all language bindings show binding parity against the C++ backend. Hover the icon for role and tolerance band.

::::{tab-set}
:class: parity-tabs

:::{tab-item} 1 thread
:sync: threads-1

<div class="parity-table-wrap">
<table class="docutils parity-grouped">
<thead><tr><th scope="col">Backend</th><th scope="col">Parity</th><th class="size-col" scope="col">50×250 (ms)</th><th class="size-col" scope="col">100×50 (ms)</th><th class="size-col" scope="col">100×500 (ms)</th><th class="size-col" scope="col">100×2500 (ms)</th><th class="size-col" scope="col">200×30 (ms)</th><th class="size-col" scope="col">250×50 (ms)</th><th class="size-col" scope="col">500×50 (ms)</th><th class="size-col" scope="col">500×500 (ms)</th><th class="size-col" scope="col">500×2500 (ms)</th><th class="size-col" scope="col">2500×50 (ms)</th><th class="size-col" scope="col">2500×500 (ms)</th><th class="size-col" scope="col">2500×2500 (ms)</th><th class="size-col" scope="col">10000×50 (ms)</th><th class="size-col" scope="col">10000×500 (ms)</th></tr></thead>
<tbody class="lang-band lang-cpp"><tr class="lang-band-row" data-lang="cpp"><th colspan="16" scope="rowgroup"><span class="lang-band-dot"></span>C++ native · libn4m</th></tr>
<tr class="bk-row"><td class="bk-name"><code>pls4all.cpp.blas</code></td><td class="parity parity-drift">≈</td><td class="ms ms-best">11.7 ms<span class="medal" title="fastest">🏆</span></td><td class="ms">14.8 ms</td><td class="ms">23.4 ms</td><td class="ms">70.4 ms</td><td class="ms">10.4 ms</td><td class="ms">14.8 ms</td><td class="ms">21.7 ms</td><td class="ms">71.4 ms</td><td class="ms">293.6 ms</td><td class="ms">62.5 ms</td><td class="ms">328.9 ms</td><td class="ms">1.6 s</td><td class="ms">223.8 ms</td><td class="ms ms-best">1.4 s<span class="medal" title="fastest">🏆</span></td></tr>
<tr class="bk-row"><td class="bk-name"><code>pls4all.cpp.blas+omp</code></td><td class="parity parity-drift">≈</td><td class="ms">14.7 ms</td><td class="ms ms-best">12.5 ms<span class="medal" title="fastest">🏆</span></td><td class="ms ms-best">22.5 ms<span class="medal" title="fastest">🏆</span></td><td class="ms ms-best">67.1 ms<span class="medal" title="fastest">🏆</span></td><td class="ms">12.6 ms</td><td class="ms">17.2 ms</td><td class="ms">21.7 ms</td><td class="ms">69.9 ms</td><td class="ms">293.3 ms</td><td class="ms">58.9 ms</td><td class="ms ms-best">320.2 ms<span class="medal" title="fastest">🏆</span></td><td class="ms ms-best">1.6 s<span class="medal" title="fastest">🏆</span></td><td class="ms ms-best">217.6 ms<span class="medal" title="fastest">🏆</span></td><td class="ms">1.5 s</td></tr>
<tr class="bk-row"><td class="bk-name"><code>pls4all.cpp.omp</code></td><td class="parity parity-drift">≈</td><td class="ms">19.3 ms</td><td class="ms">15.1 ms</td><td class="ms">24.2 ms</td><td class="ms">68.9 ms</td><td class="ms">10.6 ms</td><td class="ms">15.4 ms</td><td class="ms ms-best">19.6 ms<span class="medal" title="fastest">🏆</span></td><td class="ms ms-best">67.6 ms<span class="medal" title="fastest">🏆</span></td><td class="ms ms-best">284.1 ms<span class="medal" title="fastest">🏆</span></td><td class="ms">61.8 ms</td><td class="ms">328.0 ms</td><td class="ms">1.7 s</td><td class="ms">231.0 ms</td><td class="ms">1.5 s</td></tr>
<tr class="bk-row"><td class="bk-name"><code>pls4all.cpp.ref</code></td><td class="parity parity-drift">≈</td><td class="ms">12.0 ms</td><td class="ms">13.6 ms</td><td class="ms">23.7 ms</td><td class="ms">68.8 ms</td><td class="ms ms-best">9.83 ms<span class="medal" title="fastest">🏆</span></td><td class="ms ms-best">13.9 ms<span class="medal" title="fastest">🏆</span></td><td class="ms">21.3 ms</td><td class="ms">73.0 ms</td><td class="ms">291.2 ms</td><td class="ms ms-best">55.9 ms<span class="medal" title="fastest">🏆</span></td><td class="ms">324.1 ms</td><td class="ms">1.6 s</td><td class="ms">222.8 ms</td><td class="ms">1.5 s</td></tr>
</tbody>
<tbody class="lang-band lang-python"><tr class="lang-band-row" data-lang="python"><th colspan="16" scope="rowgroup"><span class="lang-band-dot"></span>Python · pls4all</th></tr>
<tr class="bk-row"><td class="bk-name"><code>pls4all.python</code></td><td class="parity parity-exact">✓ bind</td><td class="ms">13.8 ms</td><td class="ms ms-empty">—</td><td class="ms ms-empty">—</td><td class="ms ms-empty">—</td><td class="ms">10.6 ms</td><td class="ms">14.6 ms</td><td class="ms ms-empty">—</td><td class="ms ms-empty">—</td><td class="ms ms-empty">—</td><td class="ms ms-empty">—</td><td class="ms ms-empty">—</td><td class="ms ms-empty">—</td><td class="ms ms-empty">—</td><td class="ms ms-empty">—</td></tr>
<tr class="bk-row"><td class="bk-name"><code>pls4all.sklearn</code></td><td class="parity parity-divergent">✗ +1e+00</td><td class="ms">3.32 ms</td><td class="ms ms-empty">—</td><td class="ms ms-empty">—</td><td class="ms ms-empty">—</td><td class="ms">1.84 ms</td><td class="ms">3.87 ms</td><td class="ms ms-empty">—</td><td class="ms ms-empty">—</td><td class="ms ms-empty">—</td><td class="ms ms-empty">—</td><td class="ms ms-empty">—</td><td class="ms ms-empty">—</td><td class="ms ms-empty">—</td><td class="ms ms-empty">—</td></tr>
</tbody>
<tbody class="lang-band lang-r"><tr class="lang-band-row" data-lang="r"><th colspan="16" scope="rowgroup"><span class="lang-band-dot"></span>R · pls4all</th></tr>
<tr class="bk-row"><td class="bk-name"><code>pls4all.R</code></td><td class="parity parity-divergent">✗ +1e+00</td><td class="ms">13.2 ms</td><td class="ms ms-empty">—</td><td class="ms ms-empty">—</td><td class="ms ms-empty">—</td><td class="ms">5.10 ms</td><td class="ms">12.4 ms</td><td class="ms ms-empty">—</td><td class="ms ms-empty">—</td><td class="ms ms-empty">—</td><td class="ms ms-empty">—</td><td class="ms ms-empty">—</td><td class="ms ms-empty">—</td><td class="ms ms-empty">—</td><td class="ms ms-empty">—</td></tr>
<tr class="bk-row"><td class="bk-name"><code>pls4all.R.formula</code></td><td class="parity parity-divergent">✗ +1e+00</td><td class="ms">24.7 ms</td><td class="ms ms-empty">—</td><td class="ms ms-empty">—</td><td class="ms ms-empty">—</td><td class="ms">5.05 ms</td><td class="ms">9.66 ms</td><td class="ms ms-empty">—</td><td class="ms ms-empty">—</td><td class="ms ms-empty">—</td><td class="ms ms-empty">—</td><td class="ms ms-empty">—</td><td class="ms ms-empty">—</td><td class="ms ms-empty">—</td><td class="ms ms-empty">—</td></tr>
<tr class="bk-row"><td class="bk-name"><code>pls4all.R.mdatools</code></td><td class="parity parity-divergent">✗ +1e+00</td><td class="ms">21.9 ms</td><td class="ms ms-empty">—</td><td class="ms ms-empty">—</td><td class="ms ms-empty">—</td><td class="ms">5.99 ms</td><td class="ms">12.2 ms</td><td class="ms ms-empty">—</td><td class="ms ms-empty">—</td><td class="ms ms-empty">—</td><td class="ms ms-empty">—</td><td class="ms ms-empty">—</td><td class="ms ms-empty">—</td><td class="ms ms-empty">—</td><td class="ms ms-empty">—</td></tr>
<tr class="bk-row"><td class="bk-name"><code>pls4all.R.pls</code></td><td class="parity parity-divergent">✗ +1e+00</td><td class="ms">27.9 ms</td><td class="ms ms-empty">—</td><td class="ms ms-empty">—</td><td class="ms ms-empty">—</td><td class="ms">6.48 ms</td><td class="ms">11.6 ms</td><td class="ms ms-empty">—</td><td class="ms ms-empty">—</td><td class="ms ms-empty">—</td><td class="ms ms-empty">—</td><td class="ms ms-empty">—</td><td class="ms ms-empty">—</td><td class="ms ms-empty">—</td><td class="ms ms-empty">—</td></tr>
</tbody>
<tbody class="lang-band lang-matlab"><tr class="lang-band-row" data-lang="matlab"><th colspan="16" scope="rowgroup"><span class="lang-band-dot"></span>MATLAB · pls4all</th></tr>
<tr class="bk-row"><td class="bk-name"><code>pls4all.matlab</code></td><td class="parity parity-divergent">✗ +7e+00</td><td class="ms">4.39 ms</td><td class="ms ms-empty">—</td><td class="ms ms-empty">—</td><td class="ms ms-empty">—</td><td class="ms">2.38 ms</td><td class="ms">5.91 ms</td><td class="ms ms-empty">—</td><td class="ms ms-empty">—</td><td class="ms ms-empty">—</td><td class="ms ms-empty">—</td><td class="ms ms-empty">—</td><td class="ms ms-empty">—</td><td class="ms ms-empty">—</td><td class="ms ms-empty">—</td></tr>
<tr class="bk-row"><td class="bk-name"><code>pls4all.matlab.classdef</code></td><td class="parity parity-divergent">✗ +7e+00</td><td class="ms">5.04 ms</td><td class="ms ms-empty">—</td><td class="ms ms-empty">—</td><td class="ms ms-empty">—</td><td class="ms">2.91 ms</td><td class="ms">5.31 ms</td><td class="ms ms-empty">—</td><td class="ms ms-empty">—</td><td class="ms ms-empty">—</td><td class="ms ms-empty">—</td><td class="ms ms-empty">—</td><td class="ms ms-empty">—</td><td class="ms ms-empty">—</td><td class="ms ms-empty">—</td></tr>
</tbody>
<tbody class="lang-band lang-python"><tr class="lang-band-row" data-lang="python"><th colspan="16" scope="rowgroup"><span class="lang-band-dot"></span>Python · external</th></tr>
<tr class="bk-row truth-source truth-source-strict"><td class="bk-name"><span class="truth-mark" title="Registry parity reference (python): scikit-learn 1.8.0 — strict (rmse_rel ≤ 1e-06)">📐</span><code>ref.python_scikit_learn</code></td><td class="parity parity-ref-source">source</td><td class="ms">12.0 ms</td><td class="ms ms-empty">—</td><td class="ms ms-empty">—</td><td class="ms ms-empty">—</td><td class="ms">10.8 ms</td><td class="ms">18.5 ms</td><td class="ms ms-empty">—</td><td class="ms ms-empty">—</td><td class="ms ms-empty">—</td><td class="ms ms-empty">—</td><td class="ms ms-empty">—</td><td class="ms ms-empty">—</td><td class="ms ms-empty">—</td><td class="ms ms-empty">—</td></tr>
</tbody>
</table>
</div>

:::

:::{tab-item} 3 threads
:sync: threads-3

<div class="parity-table-wrap">
<table class="docutils parity-grouped">
<thead><tr><th scope="col">Backend</th><th scope="col">Parity</th><th class="size-col" scope="col">50×250 (ms)</th><th class="size-col" scope="col">100×50 (ms)</th><th class="size-col" scope="col">100×500 (ms)</th><th class="size-col" scope="col">100×2500 (ms)</th><th class="size-col" scope="col">200×30 (ms)</th><th class="size-col" scope="col">250×50 (ms)</th><th class="size-col" scope="col">500×50 (ms)</th><th class="size-col" scope="col">500×500 (ms)</th><th class="size-col" scope="col">500×2500 (ms)</th><th class="size-col" scope="col">2500×50 (ms)</th><th class="size-col" scope="col">2500×500 (ms)</th><th class="size-col" scope="col">2500×2500 (ms)</th><th class="size-col" scope="col">10000×50 (ms)</th><th class="size-col" scope="col">10000×500 (ms)</th></tr></thead>
<tbody class="lang-band lang-cpp"><tr class="lang-band-row" data-lang="cpp"><th colspan="16" scope="rowgroup"><span class="lang-band-dot"></span>C++ native · libn4m</th></tr>
<tr class="bk-row"><td class="bk-name"><code>pls4all.cpp.blas</code></td><td class="parity parity-ref-strict">✓ ref</td><td class="ms ms-empty">—</td><td class="ms ms-empty">—</td><td class="ms ms-empty">—</td><td class="ms ms-empty">—</td><td class="ms ms-best">9.42 ms<span class="medal" title="fastest">🏆</span></td><td class="ms ms-empty">—</td><td class="ms ms-empty">—</td><td class="ms ms-empty">—</td><td class="ms ms-empty">—</td><td class="ms ms-empty">—</td><td class="ms ms-empty">—</td><td class="ms ms-empty">—</td><td class="ms ms-empty">—</td><td class="ms ms-empty">—</td></tr>
<tr class="bk-row"><td class="bk-name"><code>pls4all.cpp.blas+omp</code></td><td class="parity parity-ref-strict">✓ ref</td><td class="ms ms-empty">—</td><td class="ms ms-empty">—</td><td class="ms ms-empty">—</td><td class="ms ms-empty">—</td><td class="ms">12.1 ms</td><td class="ms ms-empty">—</td><td class="ms ms-empty">—</td><td class="ms ms-empty">—</td><td class="ms ms-empty">—</td><td class="ms ms-empty">—</td><td class="ms ms-empty">—</td><td class="ms ms-empty">—</td><td class="ms ms-empty">—</td><td class="ms ms-empty">—</td></tr>
<tr class="bk-row"><td class="bk-name"><code>pls4all.cpp.omp</code></td><td class="parity parity-ref-strict">✓ ref</td><td class="ms ms-empty">—</td><td class="ms ms-empty">—</td><td class="ms ms-empty">—</td><td class="ms ms-empty">—</td><td class="ms">9.91 ms</td><td class="ms ms-empty">—</td><td class="ms ms-empty">—</td><td class="ms ms-empty">—</td><td class="ms ms-empty">—</td><td class="ms ms-empty">—</td><td class="ms ms-empty">—</td><td class="ms ms-empty">—</td><td class="ms ms-empty">—</td><td class="ms ms-empty">—</td></tr>
<tr class="bk-row"><td class="bk-name"><code>pls4all.cpp.ref</code></td><td class="parity parity-ref-strict">✓ ref</td><td class="ms ms-empty">—</td><td class="ms ms-empty">—</td><td class="ms ms-empty">—</td><td class="ms ms-empty">—</td><td class="ms">9.43 ms</td><td class="ms ms-empty">—</td><td class="ms ms-empty">—</td><td class="ms ms-empty">—</td><td class="ms ms-empty">—</td><td class="ms ms-empty">—</td><td class="ms ms-empty">—</td><td class="ms ms-empty">—</td><td class="ms ms-empty">—</td><td class="ms ms-empty">—</td></tr>
</tbody>
<tbody class="lang-band lang-python"><tr class="lang-band-row" data-lang="python"><th colspan="16" scope="rowgroup"><span class="lang-band-dot"></span>Python · pls4all</th></tr>
<tr class="bk-row"><td class="bk-name"><code>pls4all.python</code></td><td class="parity parity-exact">✓ bind</td><td class="ms ms-empty">—</td><td class="ms ms-empty">—</td><td class="ms ms-empty">—</td><td class="ms ms-empty">—</td><td class="ms">9.96 ms</td><td class="ms ms-empty">—</td><td class="ms ms-empty">—</td><td class="ms ms-empty">—</td><td class="ms ms-empty">—</td><td class="ms ms-empty">—</td><td class="ms ms-empty">—</td><td class="ms ms-empty">—</td><td class="ms ms-empty">—</td><td class="ms ms-empty">—</td></tr>
<tr class="bk-row"><td class="bk-name"><code>pls4all.sklearn</code></td><td class="parity parity-divergent">✗ +9e-01</td><td class="ms ms-empty">—</td><td class="ms ms-empty">—</td><td class="ms ms-empty">—</td><td class="ms ms-empty">—</td><td class="ms">1.73 ms</td><td class="ms ms-empty">—</td><td class="ms ms-empty">—</td><td class="ms ms-empty">—</td><td class="ms ms-empty">—</td><td class="ms ms-empty">—</td><td class="ms ms-empty">—</td><td class="ms ms-empty">—</td><td class="ms ms-empty">—</td><td class="ms ms-empty">—</td></tr>
</tbody>
<tbody class="lang-band lang-r"><tr class="lang-band-row" data-lang="r"><th colspan="16" scope="rowgroup"><span class="lang-band-dot"></span>R · pls4all</th></tr>
<tr class="bk-row"><td class="bk-name"><code>pls4all.R</code></td><td class="parity parity-divergent">✗ +1e+00</td><td class="ms ms-empty">—</td><td class="ms ms-empty">—</td><td class="ms ms-empty">—</td><td class="ms ms-empty">—</td><td class="ms">3.95 ms</td><td class="ms ms-empty">—</td><td class="ms ms-empty">—</td><td class="ms ms-empty">—</td><td class="ms ms-empty">—</td><td class="ms ms-empty">—</td><td class="ms ms-empty">—</td><td class="ms ms-empty">—</td><td class="ms ms-empty">—</td><td class="ms ms-empty">—</td></tr>
<tr class="bk-row"><td class="bk-name"><code>pls4all.R.formula</code></td><td class="parity parity-divergent">✗ +1e+00</td><td class="ms ms-empty">—</td><td class="ms ms-empty">—</td><td class="ms ms-empty">—</td><td class="ms ms-empty">—</td><td class="ms">5.02 ms</td><td class="ms ms-empty">—</td><td class="ms ms-empty">—</td><td class="ms ms-empty">—</td><td class="ms ms-empty">—</td><td class="ms ms-empty">—</td><td class="ms ms-empty">—</td><td class="ms ms-empty">—</td><td class="ms ms-empty">—</td><td class="ms ms-empty">—</td></tr>
<tr class="bk-row"><td class="bk-name"><code>pls4all.R.mdatools</code></td><td class="parity parity-divergent">✗ +1e+00</td><td class="ms ms-empty">—</td><td class="ms ms-empty">—</td><td class="ms ms-empty">—</td><td class="ms ms-empty">—</td><td class="ms">5.02 ms</td><td class="ms ms-empty">—</td><td class="ms ms-empty">—</td><td class="ms ms-empty">—</td><td class="ms ms-empty">—</td><td class="ms ms-empty">—</td><td class="ms ms-empty">—</td><td class="ms ms-empty">—</td><td class="ms ms-empty">—</td><td class="ms ms-empty">—</td></tr>
<tr class="bk-row"><td class="bk-name"><code>pls4all.R.pls</code></td><td class="parity parity-divergent">✗ +1e+00</td><td class="ms ms-empty">—</td><td class="ms ms-empty">—</td><td class="ms ms-empty">—</td><td class="ms ms-empty">—</td><td class="ms">5.01 ms</td><td class="ms ms-empty">—</td><td class="ms ms-empty">—</td><td class="ms ms-empty">—</td><td class="ms ms-empty">—</td><td class="ms ms-empty">—</td><td class="ms ms-empty">—</td><td class="ms ms-empty">—</td><td class="ms ms-empty">—</td><td class="ms ms-empty">—</td></tr>
</tbody>
<tbody class="lang-band lang-matlab"><tr class="lang-band-row" data-lang="matlab"><th colspan="16" scope="rowgroup"><span class="lang-band-dot"></span>MATLAB · pls4all</th></tr>
<tr class="bk-row"><td class="bk-name"><code>pls4all.matlab</code></td><td class="parity parity-divergent">✗ +7e+00</td><td class="ms ms-empty">—</td><td class="ms ms-empty">—</td><td class="ms ms-empty">—</td><td class="ms ms-empty">—</td><td class="ms">2.14 ms</td><td class="ms ms-empty">—</td><td class="ms ms-empty">—</td><td class="ms ms-empty">—</td><td class="ms ms-empty">—</td><td class="ms ms-empty">—</td><td class="ms ms-empty">—</td><td class="ms ms-empty">—</td><td class="ms ms-empty">—</td><td class="ms ms-empty">—</td></tr>
<tr class="bk-row"><td class="bk-name"><code>pls4all.matlab.classdef</code></td><td class="parity parity-divergent">✗ +7e+00</td><td class="ms ms-empty">—</td><td class="ms ms-empty">—</td><td class="ms ms-empty">—</td><td class="ms ms-empty">—</td><td class="ms">2.68 ms</td><td class="ms ms-empty">—</td><td class="ms ms-empty">—</td><td class="ms ms-empty">—</td><td class="ms ms-empty">—</td><td class="ms ms-empty">—</td><td class="ms ms-empty">—</td><td class="ms ms-empty">—</td><td class="ms ms-empty">—</td><td class="ms ms-empty">—</td></tr>
</tbody>
<tbody class="lang-band lang-python"><tr class="lang-band-row" data-lang="python"><th colspan="16" scope="rowgroup"><span class="lang-band-dot"></span>Python · external</th></tr>
<tr class="bk-row truth-source truth-source-strict"><td class="bk-name"><span class="truth-mark" title="Registry parity reference (python): scikit-learn 1.8.0 — strict (rmse_rel ≤ 1e-06)">📐</span><code>ref.python_scikit_learn</code></td><td class="parity parity-ref-source">source</td><td class="ms ms-empty">—</td><td class="ms ms-empty">—</td><td class="ms ms-empty">—</td><td class="ms ms-empty">—</td><td class="ms">10.9 ms</td><td class="ms ms-empty">—</td><td class="ms ms-empty">—</td><td class="ms ms-empty">—</td><td class="ms ms-empty">—</td><td class="ms ms-empty">—</td><td class="ms ms-empty">—</td><td class="ms ms-empty">—</td><td class="ms ms-empty">—</td><td class="ms ms-empty">—</td></tr>
</tbody>
</table>
</div>

:::

:::{tab-item} 10 threads
:sync: threads-10

<div class="parity-table-wrap">
<table class="docutils parity-grouped">
<thead><tr><th scope="col">Backend</th><th scope="col">Parity</th><th class="size-col" scope="col">50×250 (ms)</th><th class="size-col" scope="col">100×50 (ms)</th><th class="size-col" scope="col">100×500 (ms)</th><th class="size-col" scope="col">100×2500 (ms)</th><th class="size-col" scope="col">200×30 (ms)</th><th class="size-col" scope="col">250×50 (ms)</th><th class="size-col" scope="col">500×50 (ms)</th><th class="size-col" scope="col">500×500 (ms)</th><th class="size-col" scope="col">500×2500 (ms)</th><th class="size-col" scope="col">2500×50 (ms)</th><th class="size-col" scope="col">2500×500 (ms)</th><th class="size-col" scope="col">2500×2500 (ms)</th><th class="size-col" scope="col">10000×50 (ms)</th><th class="size-col" scope="col">10000×500 (ms)</th></tr></thead>
<tbody class="lang-band lang-cpp"><tr class="lang-band-row" data-lang="cpp"><th colspan="16" scope="rowgroup"><span class="lang-band-dot"></span>C++ native · libn4m</th></tr>
<tr class="bk-row"><td class="bk-name"><code>pls4all.cpp.blas</code></td><td class="parity parity-ref-strict">✓ ref</td><td class="ms ms-empty">—</td><td class="ms ms-empty">—</td><td class="ms ms-empty">—</td><td class="ms ms-empty">—</td><td class="ms">8.96 ms</td><td class="ms ms-empty">—</td><td class="ms ms-empty">—</td><td class="ms ms-empty">—</td><td class="ms ms-empty">—</td><td class="ms ms-empty">—</td><td class="ms ms-empty">—</td><td class="ms ms-empty">—</td><td class="ms ms-empty">—</td><td class="ms ms-empty">—</td></tr>
<tr class="bk-row"><td class="bk-name"><code>pls4all.cpp.blas+omp</code></td><td class="parity parity-ref-strict">✓ ref</td><td class="ms ms-empty">—</td><td class="ms ms-empty">—</td><td class="ms ms-empty">—</td><td class="ms ms-empty">—</td><td class="ms ms-best">8.80 ms<span class="medal" title="fastest">🏆</span></td><td class="ms ms-empty">—</td><td class="ms ms-empty">—</td><td class="ms ms-empty">—</td><td class="ms ms-empty">—</td><td class="ms ms-empty">—</td><td class="ms ms-empty">—</td><td class="ms ms-empty">—</td><td class="ms ms-empty">—</td><td class="ms ms-empty">—</td></tr>
<tr class="bk-row"><td class="bk-name"><code>pls4all.cpp.omp</code></td><td class="parity parity-ref-strict">✓ ref</td><td class="ms ms-empty">—</td><td class="ms ms-empty">—</td><td class="ms ms-empty">—</td><td class="ms ms-empty">—</td><td class="ms">8.85 ms</td><td class="ms ms-empty">—</td><td class="ms ms-empty">—</td><td class="ms ms-empty">—</td><td class="ms ms-empty">—</td><td class="ms ms-empty">—</td><td class="ms ms-empty">—</td><td class="ms ms-empty">—</td><td class="ms ms-empty">—</td><td class="ms ms-empty">—</td></tr>
<tr class="bk-row"><td class="bk-name"><code>pls4all.cpp.ref</code></td><td class="parity parity-ref-strict">✓ ref</td><td class="ms ms-empty">—</td><td class="ms ms-empty">—</td><td class="ms ms-empty">—</td><td class="ms ms-empty">—</td><td class="ms">8.94 ms</td><td class="ms ms-empty">—</td><td class="ms ms-empty">—</td><td class="ms ms-empty">—</td><td class="ms ms-empty">—</td><td class="ms ms-empty">—</td><td class="ms ms-empty">—</td><td class="ms ms-empty">—</td><td class="ms ms-empty">—</td><td class="ms ms-empty">—</td></tr>
</tbody>
<tbody class="lang-band lang-python"><tr class="lang-band-row" data-lang="python"><th colspan="16" scope="rowgroup"><span class="lang-band-dot"></span>Python · pls4all</th></tr>
<tr class="bk-row"><td class="bk-name"><code>pls4all.python</code></td><td class="parity parity-exact">✓ bind</td><td class="ms ms-empty">—</td><td class="ms ms-empty">—</td><td class="ms ms-empty">—</td><td class="ms ms-empty">—</td><td class="ms">8.96 ms</td><td class="ms ms-empty">—</td><td class="ms ms-empty">—</td><td class="ms ms-empty">—</td><td class="ms ms-empty">—</td><td class="ms ms-empty">—</td><td class="ms ms-empty">—</td><td class="ms ms-empty">—</td><td class="ms ms-empty">—</td><td class="ms ms-empty">—</td></tr>
<tr class="bk-row"><td class="bk-name"><code>pls4all.sklearn</code></td><td class="parity parity-divergent">✗ +9e-01</td><td class="ms ms-empty">—</td><td class="ms ms-empty">—</td><td class="ms ms-empty">—</td><td class="ms ms-empty">—</td><td class="ms">1.47 ms</td><td class="ms ms-empty">—</td><td class="ms ms-empty">—</td><td class="ms ms-empty">—</td><td class="ms ms-empty">—</td><td class="ms ms-empty">—</td><td class="ms ms-empty">—</td><td class="ms ms-empty">—</td><td class="ms ms-empty">—</td><td class="ms ms-empty">—</td></tr>
</tbody>
<tbody class="lang-band lang-r"><tr class="lang-band-row" data-lang="r"><th colspan="16" scope="rowgroup"><span class="lang-band-dot"></span>R · pls4all</th></tr>
<tr class="bk-row"><td class="bk-name"><code>pls4all.R</code></td><td class="parity parity-divergent">✗ +1e+00</td><td class="ms ms-empty">—</td><td class="ms ms-empty">—</td><td class="ms ms-empty">—</td><td class="ms ms-empty">—</td><td class="ms">3.41 ms</td><td class="ms ms-empty">—</td><td class="ms ms-empty">—</td><td class="ms ms-empty">—</td><td class="ms ms-empty">—</td><td class="ms ms-empty">—</td><td class="ms ms-empty">—</td><td class="ms ms-empty">—</td><td class="ms ms-empty">—</td><td class="ms ms-empty">—</td></tr>
<tr class="bk-row"><td class="bk-name"><code>pls4all.R.formula</code></td><td class="parity parity-divergent">✗ +1e+00</td><td class="ms ms-empty">—</td><td class="ms ms-empty">—</td><td class="ms ms-empty">—</td><td class="ms ms-empty">—</td><td class="ms">4.06 ms</td><td class="ms ms-empty">—</td><td class="ms ms-empty">—</td><td class="ms ms-empty">—</td><td class="ms ms-empty">—</td><td class="ms ms-empty">—</td><td class="ms ms-empty">—</td><td class="ms ms-empty">—</td><td class="ms ms-empty">—</td><td class="ms ms-empty">—</td></tr>
<tr class="bk-row"><td class="bk-name"><code>pls4all.R.mdatools</code></td><td class="parity parity-divergent">✗ +1e+00</td><td class="ms ms-empty">—</td><td class="ms ms-empty">—</td><td class="ms ms-empty">—</td><td class="ms ms-empty">—</td><td class="ms">3.82 ms</td><td class="ms ms-empty">—</td><td class="ms ms-empty">—</td><td class="ms ms-empty">—</td><td class="ms ms-empty">—</td><td class="ms ms-empty">—</td><td class="ms ms-empty">—</td><td class="ms ms-empty">—</td><td class="ms ms-empty">—</td><td class="ms ms-empty">—</td></tr>
<tr class="bk-row"><td class="bk-name"><code>pls4all.R.pls</code></td><td class="parity parity-divergent">✗ +1e+00</td><td class="ms ms-empty">—</td><td class="ms ms-empty">—</td><td class="ms ms-empty">—</td><td class="ms ms-empty">—</td><td class="ms">3.77 ms</td><td class="ms ms-empty">—</td><td class="ms ms-empty">—</td><td class="ms ms-empty">—</td><td class="ms ms-empty">—</td><td class="ms ms-empty">—</td><td class="ms ms-empty">—</td><td class="ms ms-empty">—</td><td class="ms ms-empty">—</td><td class="ms ms-empty">—</td></tr>
</tbody>
<tbody class="lang-band lang-matlab"><tr class="lang-band-row" data-lang="matlab"><th colspan="16" scope="rowgroup"><span class="lang-band-dot"></span>MATLAB · pls4all</th></tr>
<tr class="bk-row"><td class="bk-name"><code>pls4all.matlab</code></td><td class="parity parity-divergent">✗ +7e+00</td><td class="ms ms-empty">—</td><td class="ms ms-empty">—</td><td class="ms ms-empty">—</td><td class="ms ms-empty">—</td><td class="ms">2.01 ms</td><td class="ms ms-empty">—</td><td class="ms ms-empty">—</td><td class="ms ms-empty">—</td><td class="ms ms-empty">—</td><td class="ms ms-empty">—</td><td class="ms ms-empty">—</td><td class="ms ms-empty">—</td><td class="ms ms-empty">—</td><td class="ms ms-empty">—</td></tr>
<tr class="bk-row"><td class="bk-name"><code>pls4all.matlab.classdef</code></td><td class="parity parity-divergent">✗ +7e+00</td><td class="ms ms-empty">—</td><td class="ms ms-empty">—</td><td class="ms ms-empty">—</td><td class="ms ms-empty">—</td><td class="ms">2.39 ms</td><td class="ms ms-empty">—</td><td class="ms ms-empty">—</td><td class="ms ms-empty">—</td><td class="ms ms-empty">—</td><td class="ms ms-empty">—</td><td class="ms ms-empty">—</td><td class="ms ms-empty">—</td><td class="ms ms-empty">—</td><td class="ms ms-empty">—</td></tr>
</tbody>
<tbody class="lang-band lang-python"><tr class="lang-band-row" data-lang="python"><th colspan="16" scope="rowgroup"><span class="lang-band-dot"></span>Python · external</th></tr>
<tr class="bk-row truth-source truth-source-strict"><td class="bk-name"><span class="truth-mark" title="Registry parity reference (python): scikit-learn 1.8.0 — strict (rmse_rel ≤ 1e-06)">📐</span><code>ref.python_scikit_learn</code></td><td class="parity parity-ref-source">source</td><td class="ms ms-empty">—</td><td class="ms ms-empty">—</td><td class="ms ms-empty">—</td><td class="ms ms-empty">—</td><td class="ms">9.30 ms</td><td class="ms ms-empty">—</td><td class="ms ms-empty">—</td><td class="ms ms-empty">—</td><td class="ms ms-empty">—</td><td class="ms ms-empty">—</td><td class="ms ms-empty">—</td><td class="ms ms-empty">—</td><td class="ms ms-empty">—</td><td class="ms ms-empty">—</td></tr>
</tbody>
</table>
</div>

:::

::::


---

_See also_: [benchmark overview](../benchmarks/overview.md) · [methods index](index.md) · [interactive dashboard](../landing/dashboard.md)