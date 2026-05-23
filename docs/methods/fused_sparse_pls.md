# `fused_sparse_pls` — Fused-sparse PLS

_Group_: **Sparse** · _Registry tolerance_: `0.05`
 · _Parity reference_: **paper-only** (Yengo, L., Jacques, J., Biernacki, C. & Canouil, M. (2016). Variable clustering in high-dimensional linear regression: The R package `clere`. R Journal 8(1), 92-106. (Fused-sparse PLS variants have no widely installable R / Python port.))

## Description

Fused sparse PLS (§7)

From the `pls4all.sklearn.FusedSparsePLSRegression` docstring:

> Fused-sparse PLS — L1 + adjacent-coef smoothing.

### Parameters

| Name | Type | Default | Notes |
|------|------|---------|-------|
| `n_components` | `int` | `2` | Number of latent components extracted (k). |
| `l1_lambda` | `float` | `0.05` | L1 sparsity penalty for fused-sparse PLS. |
| `fusion_lambda` | `float` | `0.05` | Fused-lasso penalty enforcing smoothness between adjacent coefficients. |

## Explanations

### Bibliographic source

Tibshirani, R., Saunders, M., Rosset, S., Zhu, J. & Knight, K. (2005). *Sparsity and smoothness via the fused lasso*. JRSS B 67(1), 91–108. — generalised to PLS loadings.

### Mathematical principle

The fused lasso penalty combines L1 sparsity with an L1 penalty on first differences of neighbouring coefficients: $\mathcal{P}(\mathbf{w}) = \lambda_1 \|\mathbf{w}\|_1 + \lambda_2 \sum_{j=1}^{p-1}|w_{j+1} - w_j|$. For spectroscopic data this is uniquely useful: the second term *encourages neighbouring wavelengths to share a coefficient*, which produces piecewise-constant loadings reflecting the underlying band structure of the spectrum rather than the choppy sample-noise pattern that plain L1 alone produces.

Tuning two penalties simultaneously is non-trivial: $\lambda_1$ controls overall sparsity, $\lambda_2$ controls within-band smoothness. Plotting the selected wavelength bands as a function of $\lambda_2$ at fixed sparsity is a common diagnostic.

Computationally the fused-lasso step is a 1-D total variation denoising problem with a closed-form taut-string solution in $O(p)$; pls4all uses Condat's algorithm.

### Implementation

`n4m_fused_sparse_pls_fit`. No widely installable reference library — `sgPLS` covers group-sparse but not fused-sparse. Documented as `paper_only` in the registry.

R roxygen note (`methods_extra.R::fused_sparse_pls_fit`):

> Fused-sparse PLS (L1 + adjacent-coef smoothing).
> @param n_components Integer. Number of latent components.
> @param l1_lambda Method-specific parameter. See the underlying `*_fit()` function for the exact semantics.
> @param fusion_lambda Method-specific parameter. See the underlying `*_fit()` function for the exact semantics.
> @param X Numeric matrix of predictors (rows = samples, cols = features).
> @param Y Numeric matrix or vector of responses, with one row per sample.
> @export

MATLAB header (`bindings/matlab/+pls4all/fused_sparse_pls.m`):

```text
pls4all.fused_sparse_pls  Fused-sparse PLS (L1 + adjacent-coef smoothing).
```

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
n4m_fused_sparse_pls_fit(ctx, cfg, &x_view, &y_view, /* hyperparams */, &res);
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
from pls4all._methods import fused_sparse_pls_fit
with pls4all.Context() as ctx, pls4all.Config() as cfg:
    res = fused_sparse_pls_fit(ctx, cfg, X, y, n_components=4)
# then: res.matrix("predictions"), res.matrix("coefficients"),
# res.vector("mask"), res.scalar("intercept"), …
```

:::

:::{tab-item} Python · pls4all.sklearn
:sync: python-sklearn
:class-label: lang-python

```python
from pls4all.sklearn import FusedSparsePLSRegression
mdl = FusedSparsePLSRegression(n_components=2, l1_lambda=0.05, fusion_lambda=0.05)
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
res <- pls4all_method("fused_sparse_pls", X, y,
                      n_components = 4L, params = list(l1_lambda = 0.05, fusion_lambda = 0.1))
# res is a named list with MethodResult arrays/scalars.
# selected_indices / top_k_intervals are 1-based.
```

:::

:::{tab-item} R · pls4all (raw fn)
:sync: r-raw
:class-label: lang-r

```r
library(pls4all)
res  <- fused_sparse_pls_fit(X, Y, n_components,
                      l1_lambda = 0.05, fusion_lambda = 0.05)
yhat <- pls4all_predict(res, X_test)
```

:::

:::{tab-item} MATLAB · pls4all (MEX)
:sync: matlab-mex
:class-label: lang-matlab

```matlab
res = pls4all.fused_sparse_pls(X, y, 4);
% see header of bindings/matlab/+pls4all/fused_sparse_pls.m for full
% parameter surface:
%   res = fused_sparse_pls(X, Y, n_components, l1_lambda, fusion_lambda)
yhat = predict(res, Xtest);
```

:::

:::{tab-item} MATLAB · pls4all (classdef)
:sync: matlab-classdef
:class-label: lang-matlab

_No idiomatic classdef wrapper — invoke `pls4all.fit("fused_sparse_pls", X, y, …)` directly from the unified MEX factory._

:::

::::


**Registry parity references** 📐

:::{card}
:class-card: external-refs

- 📜 **Paper-only** — no executable parity reference; the `pls4all` implementation is verified by a smoke fit only. Canonical citation: Yengo, L., Jacques, J., Biernacki, C. & Canouil, M. (2016). Variable clustering in high-dimensional linear regression: The R package `clere`. R Journal 8(1), 92-106. (Fused-sparse PLS variants have no widely installable R / Python port.)
:::

### Benchmarks

Median wall-clock per cell from [`benchmarks/cross_binding/results/full_matrix.csv`](../benchmarks/overview.md). Verdict legend: ✓ exact · ≈ drift · ✗ divergent · ⊘ not available in lib · — not run · ⚠ error. The fastest backend per column is marked with a 🏆 medal.

::::{tab-set}
:class: parity-tabs

:::{tab-item} 1 thread
:sync: threads-1

<div class="parity-table-wrap">
<table class="docutils parity-grouped">
<thead><tr><th scope="col">Backend</th><th scope="col">Parity</th><th class="size-col" scope="col">100×50 (ms)</th></tr></thead>
<tbody class="lang-band lang-cpp"><tr class="lang-band-row" data-lang="cpp"><th colspan="3" scope="rowgroup"><span class="lang-band-dot"></span>C++ native · libn4m</th></tr>
<tr class="bk-row"><td class="bk-name"><code>pls4all.cpp.blas</code></td><td class="parity parity-not_available">⊘</td><td class="ms">0.99 ms</td></tr>
<tr class="bk-row"><td class="bk-name"><code>pls4all.cpp.blas+omp</code></td><td class="parity parity-not_available">⊘</td><td class="ms">1.00 ms</td></tr>
<tr class="bk-row"><td class="bk-name"><code>pls4all.cpp.omp</code></td><td class="parity parity-not_available">⊘</td><td class="ms">1.04 ms</td></tr>
<tr class="bk-row"><td class="bk-name"><code>pls4all.cpp.ref</code></td><td class="parity parity-not_available">⊘</td><td class="ms">1.07 ms</td></tr>
</tbody>
<tbody class="lang-band lang-python"><tr class="lang-band-row" data-lang="python"><th colspan="3" scope="rowgroup"><span class="lang-band-dot"></span>Python · pls4all</th></tr>
<tr class="bk-row"><td class="bk-name"><code>pls4all.python</code></td><td class="parity parity-exact">✓ bind</td><td class="ms">1.00 ms</td></tr>
<tr class="bk-row"><td class="bk-name"><code>pls4all.registry</code></td><td class="parity parity-exact">✓ bind</td><td class="ms ms-best">0.99 ms<span class="medal" title="fastest">🏆</span></td></tr>
<tr class="bk-row"><td class="bk-name"><code>pls4all.sklearn</code></td><td class="parity parity-exact">✓ bind</td><td class="ms">1.13 ms</td></tr>
</tbody>
<tbody class="lang-band lang-r"><tr class="lang-band-row" data-lang="r"><th colspan="3" scope="rowgroup"><span class="lang-band-dot"></span>R · pls4all</th></tr>
<tr class="bk-row"><td class="bk-name"><code>pls4all.R</code></td><td class="parity parity-exact">✓ bind</td><td class="ms">2.42 ms</td></tr>
<tr class="bk-row"><td class="bk-name"><code>pls4all.R.formula</code></td><td class="parity parity-exact">✓ bind</td><td class="ms">4.59 ms</td></tr>
</tbody>
<tbody class="lang-band lang-matlab"><tr class="lang-band-row" data-lang="matlab"><th colspan="3" scope="rowgroup"><span class="lang-band-dot"></span>MATLAB · pls4all</th></tr>
<tr class="bk-row"><td class="bk-name"><code>pls4all.matlab</code></td><td class="parity parity-exact">✓ bind</td><td class="ms">1.60 ms</td></tr>
<tr class="bk-row"><td class="bk-name"><code>pls4all.matlab.classdef</code></td><td class="parity parity-exact">✓ bind</td><td class="ms">1.90 ms</td></tr>
</tbody>
<tbody class="lang-band lang-python"><tr class="lang-band-row" data-lang="python"><th colspan="3" scope="rowgroup"><span class="lang-band-dot"></span>Python · external</th></tr>
<tr class="bk-row"><td class="bk-name"><code>ikpls</code></td><td class="parity parity-not_available">⊘</td><td class="ms">—</td></tr>
<tr class="bk-row"><td class="bk-name"><code>sklearn</code></td><td class="parity parity-not_available">⊘</td><td class="ms">—</td></tr>
</tbody>
<tbody class="lang-band lang-r"><tr class="lang-band-row" data-lang="r"><th colspan="3" scope="rowgroup"><span class="lang-band-dot"></span>R · external</th></tr>
<tr class="bk-row"><td class="bk-name"><code>mixOmics</code></td><td class="parity parity-not_available">⊘</td><td class="ms">—</td></tr>
<tr class="bk-row"><td class="bk-name"><code>pls</code></td><td class="parity parity-not_available">⊘</td><td class="ms">—</td></tr>
<tr class="bk-row"><td class="bk-name"><code>ropls</code></td><td class="parity parity-not_available">⊘</td><td class="ms">—</td></tr>
</tbody>
<tbody class="lang-band lang-matlab"><tr class="lang-band-row" data-lang="matlab"><th colspan="3" scope="rowgroup"><span class="lang-band-dot"></span>MATLAB · external</th></tr>
<tr class="bk-row"><td class="bk-name"><code>plsregress</code></td><td class="parity parity-not_run">—</td><td class="ms">—</td></tr>
</tbody>
<tbody class="lang-band lang-ext"><tr class="lang-band-row" data-lang="ext"><th colspan="3" scope="rowgroup"><span class="lang-band-dot"></span>Other</th></tr>
<tr class="bk-row"><td class="bk-name"><code>r_mdatools_compat</code></td><td class="parity parity-exact">✓ bind</td><td class="ms">3.51 ms</td></tr>
<tr class="bk-row"><td class="bk-name"><code>r_pls_compat</code></td><td class="parity parity-exact">✓ bind</td><td class="ms">3.46 ms</td></tr>
</tbody>
</table>
</div>

:::

::::


---

_See also_: [benchmark overview](../benchmarks/overview.md) · [methods index](index.md) · [interactive dashboard](../landing/dashboard.md)