# `di_pls` — Domain-Invariant PLS (di-PLS)

_Group_: **Calibration transfer** · _Registry tolerance_: `0.02`

## Description

Domain-invariant PLS

From the `pls4all.sklearn.DIPLSRegression` docstring:

> Domain-invariant PLS (Nikzad-Langerodi 2018).

> **Registry note** — Python `diPLSlib.models.DIPLS` (B-Analytics; original Nikzad-Langerodi 2018 authors). Same di-PLS penalty and Beer-Lambert geometry; rescale='Target' is the default config matching pls4all's source-centered fit.

### Parameters

| Name | Type | Default | Notes |
|------|------|---------|-------|
| `n_components` | `int` | `2` | Number of latent components extracted (k). |
| `di_lambda` | `float` | `1.0` | Domain-invariance penalty weight balancing covariance alignment vs response fit. |
| `X_target` | `matrix (required)` | `` | fit-time extra (not part of `__init__`) |

## Explanations

### Bibliographic source

Nikzad-Langerodi, R., Zellinger, W., Saminger-Platz, S. & Moser, B. A. (2018). *Domain-invariant partial-least-squares regression*. Analytical Chemistry 90(11), 6693–6701.

### Mathematical principle

Calibration transfer methods reconcile spectra acquired on different instruments or under different environmental conditions. di-PLS does this by augmenting the PLS objective with a domain-discrepancy penalty: $\mathcal{L}(\mathbf{w}) = -\operatorname{Cov}(\mathbf{X}_s\mathbf{w}, \mathbf{y}_s)^2 + \lambda \,\mathrm{MMD}^2(\mathbf{X}_s\mathbf{w}, \mathbf{X}_t\mathbf{w})$, where $(\mathbf{X}_s, \mathbf{y}_s)$ is a labelled source domain, $\mathbf{X}_t$ is an unlabelled target domain and MMD is the maximum mean discrepancy.

Minimising $\mathcal{L}$ produces latent directions $\mathbf{w}$ that simultaneously **predict $y$ in the source** and have **matched distributions across domains**. The model is therefore robust to drift between calibration and prediction sets without requiring labels on the target domain.

Computational cost is dominated by the MMD term, which is $O((n_s + n_t)^2)$ in a naive implementation; pls4all uses a linear-kernel MMD which reduces this to $O((n_s + n_t) p)$.

$\lambda$ controls the bias–transferability trade-off: $\lambda = 0$ recovers vanilla PLS on the source, large $\lambda$ shrinks toward a domain-aligned but potentially under-predictive model.

### Implementation

`n4m_di_pls_fit` — requires `X_target` at fit time. Reference: Python `diPLSlib.models.DIPLS` (Nikzad-Langerodi authors). The pls4all variant matches diPLSlib's `rescale='Target'` source-centred default.

R roxygen note (`sklearn_methods.R::di_pls`):

> Domain-invariant PLS -- formula entry point.
> @param X_target Numeric matrix for the target domain.
> @param di_lambda Numeric DI-PLS penalty.
> @inheritParams pls
> @export

MATLAB header (`bindings/matlab/+pls4all/DiPlsRegression.m`):

```text
pls4all.DiPlsRegression  Domain-Invariant PLS regression.
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
n4m_di_pls_fit(ctx, cfg, &x_view, &y_view, /* hyperparams */, &res);
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
from pls4all._methods import di_pls_fit
with pls4all.Context() as ctx, pls4all.Config() as cfg:
    res = di_pls_fit(ctx, cfg, X, y, n_components=4, X_target=X_target)
# then: res.matrix("predictions"), res.matrix("coefficients"),
# res.vector("mask"), res.scalar("intercept"), …
```

:::

:::{tab-item} Python · pls4all.sklearn
:sync: python-sklearn
:class-label: lang-python

```python
from pls4all.sklearn import DIPLSRegression
mdl = DIPLSRegression(n_components=2, di_lambda=1.0)
mdl.fit(X, y, X_target=X_target)
y_hat = mdl.predict(X_test)
```

:::

:::{tab-item} R · pls4all_method()
:sync: r-dispatcher
:class-label: lang-r

```r
library(pls4all)
# Unified low-level dispatcher (May 2026 R cleanup):
res <- pls4all_method("di_pls", X, y,
                      n_components = 4L, params = list(di_lambda = 1.0))
# res is a named list with MethodResult arrays/scalars.
# selected_indices / top_k_intervals are 1-based.
```

:::

:::{tab-item} R · pls4all (raw fn)
:sync: r-raw
:class-label: lang-r

```r
library(pls4all)
res  <- di_pls_fit(X_source, Y_source, n_components, X_target, di_lambda = 1.0)
yhat <- pls4all_predict(res, X_test)
```

:::

:::{tab-item} R · pls4all (formula+S3)
:sync: r-formula
:class-label: lang-r

```r
library(pls4all)
fit  <- di_pls(y ~ ., data = train, ncomp = 4L)
yhat <- predict(fit, newdata = test)
summary(fit)
```

:::

:::{tab-item} MATLAB · pls4all (MEX)
:sync: matlab-mex
:class-label: lang-matlab

```matlab
res = pls4all.di_pls(X, y, 4);
% see header of bindings/matlab/+pls4all/di_pls.m for full
% parameter surface:
%   res = di_pls(X_source, Y_source, n_components, X_target, di_lambda)
yhat = predict(res, Xtest);
```

:::

:::{tab-item} MATLAB · pls4all (classdef)
:sync: matlab-classdef
:class-label: lang-matlab

```matlab
mdl  = pls4all.fit("di_pls", X, y, "NumComponents", 4);
yhat = predict(mdl, Xtest);
```

:::

::::


**Registry parity references** 📐

:::{card}
:class-card: external-refs

- 📐 **`ref.python_diplslib`** (python · python) — `diPLSlib` 2.5.0 · relaxed (rmse_rel ≤ 2e-02) — Python `diPLSlib.models.DIPLS` (B-Analytics; Nikzad-Langerodi 2018 authors). Same di-PLS penalty applied during deflation; centering / target rescaling differ slightly, so tolerance is widened.
:::

### Benchmarks

Median wall-clock per cell from [`benchmarks/cross_binding/results/full_matrix.csv`](../benchmarks/overview.md). Verdict legend: ✓ exact · ≈ drift · ✗ divergent · ⊘ not available in lib · — not run · ⚠ error. The fastest backend per column is marked with a 🏆 medal. Rows tagged with **📐** are *also* declared in [`benchmarks/parity_timing/registry.py`](../benchmarks/methodology.md) as the canonical parity references for this method (`python_reference` / `r_reference` / `extra_references`). C++ and external rows show reference parity; pls4all language bindings show binding parity against the C++ backend. The 📐 icon points at the *library-of-record* the parity gate ultimately answers to. Hover the icon to see the role and tolerance band.

::::{tab-set}
:class: parity-tabs

:::{tab-item} 1 thread
:sync: threads-1

<div class="parity-table-wrap">
<table class="docutils parity-grouped">
<thead><tr><th scope="col">Backend</th><th scope="col">Parity</th><th class="size-col" scope="col">100×50 (ms)</th></tr></thead>
<tbody class="lang-band lang-cpp"><tr class="lang-band-row" data-lang="cpp"><th colspan="3" scope="rowgroup"><span class="lang-band-dot"></span>C++ native · libn4m</th></tr>
<tr class="bk-row"><td class="bk-name"><code>pls4all.cpp.blas</code></td><td class="parity parity-exact">✓ 9e-03</td><td class="ms">1.02 ms</td></tr>
<tr class="bk-row"><td class="bk-name"><code>pls4all.cpp.blas+omp</code></td><td class="parity parity-exact">✓ 2e-02</td><td class="ms">1.01 ms</td></tr>
<tr class="bk-row"><td class="bk-name"><code>pls4all.cpp.omp</code></td><td class="parity parity-exact">✓ 9e-03</td><td class="ms">1.00 ms</td></tr>
<tr class="bk-row"><td class="bk-name"><code>pls4all.cpp.ref</code></td><td class="parity parity-exact">✓ 9e-03</td><td class="ms">1.24 ms</td></tr>
</tbody>
<tbody class="lang-band lang-python"><tr class="lang-band-row" data-lang="python"><th colspan="3" scope="rowgroup"><span class="lang-band-dot"></span>Python · pls4all</th></tr>
<tr class="bk-row"><td class="bk-name"><code>pls4all.python</code></td><td class="parity parity-exact">✓ bind</td><td class="ms">1.04 ms</td></tr>
<tr class="bk-row"><td class="bk-name"><code>pls4all.registry</code></td><td class="parity parity-exact">✓ bind</td><td class="ms ms-best">0.97 ms<span class="medal" title="fastest">🏆</span></td></tr>
<tr class="bk-row"><td class="bk-name"><code>pls4all.sklearn</code></td><td class="parity parity-exact">✓ 4e-15</td><td class="ms">1.38 ms</td></tr>
</tbody>
<tbody class="lang-band lang-r"><tr class="lang-band-row" data-lang="r"><th colspan="3" scope="rowgroup"><span class="lang-band-dot"></span>R · pls4all</th></tr>
<tr class="bk-row"><td class="bk-name"><code>pls4all.R</code></td><td class="parity parity-exact">✓ 7e-15</td><td class="ms">4.79 ms</td></tr>
<tr class="bk-row"><td class="bk-name"><code>pls4all.R.formula</code></td><td class="parity parity-exact">✓ 7e-15</td><td class="ms">6.03 ms</td></tr>
</tbody>
<tbody class="lang-band lang-matlab"><tr class="lang-band-row" data-lang="matlab"><th colspan="3" scope="rowgroup"><span class="lang-band-dot"></span>MATLAB · pls4all</th></tr>
<tr class="bk-row"><td class="bk-name"><code>pls4all.matlab</code></td><td class="parity parity-exact">✓ 7e-15</td><td class="ms">3.13 ms</td></tr>
<tr class="bk-row"><td class="bk-name"><code>pls4all.matlab.classdef</code></td><td class="parity parity-exact">✓ 6e-15</td><td class="ms">3.66 ms</td></tr>
</tbody>
<tbody class="lang-band lang-python"><tr class="lang-band-row" data-lang="python"><th colspan="3" scope="rowgroup"><span class="lang-band-dot"></span>Python · external</th></tr>
<tr class="bk-row"><td class="bk-name"><code>ikpls</code></td><td class="parity parity-not_available">⊘</td><td class="ms">—</td></tr>
<tr class="bk-row truth-source truth-source-relaxed"><td class="bk-name"><span class="truth-mark" title="Registry parity reference (python): diPLSlib 2.5.0 — relaxed (rmse_rel ≤ 2e-02)">📐</span><code>ref.python_diplslib</code></td><td class="parity parity-exact">✓ ref</td><td class="ms">3.41 ms</td></tr>
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
<tr class="bk-row"><td class="bk-name"><code>r_mdatools_compat</code></td><td class="parity parity-exact">✓ bind</td><td class="ms">5.98 ms</td></tr>
<tr class="bk-row"><td class="bk-name"><code>r_pls_compat</code></td><td class="parity parity-exact">✓ bind</td><td class="ms">5.78 ms</td></tr>
</tbody>
</table>
</div>

:::

::::


---

_See also_: [benchmark overview](../benchmarks/overview.md) · [methods index](index.md) · [interactive dashboard](../landing/dashboard.md)