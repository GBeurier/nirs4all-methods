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

`p4a_di_pls_fit` — requires `X_target` at fit time. Reference: Python `diPLSlib.models.DIPLS` (Nikzad-Langerodi authors). The pls4all variant matches diPLSlib's `rescale='Target'` source-centred default.

R roxygen note (`methods_extra.R::di_pls_fit`):

> Domain-Invariant PLS (Nikzad-Langerodi 2018).
> @param X_source Method-specific parameter. See the underlying `*_fit()` function for the exact semantics.
> @param Y_source Method-specific parameter. See the underlying `*_fit()` function for the exact semantics.
> @param n_components Integer. Number of latent components.
> @param X_target Method-specific parameter. See the underlying `*_fit()` function for the exact semantics.
> @param di_lambda Method-specific parameter. See the underlying `*_fit()` function for the exact semantics.
> @export

MATLAB header (`bindings/matlab/+pls4all/di_pls.m`):

```text
pls4all.di_pls  Domain-Invariant PLS (Nikzad-Langerodi 2018).
```

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
p4a_di_pls_fit(ctx, cfg, &x_view, &y_view, /* hyperparams */, &res);
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

:::{tab-item} R · parsnip / mlr3
:sync: r-meta
:class-label: lang-r

```r
# parsnip / tidymodels
library(tidymodels)
pls4all::register_parsnip()
spec <- pls_pls4all_reg(num_comp = 4) %>%
    set_engine("pls4all", algorithm = "di_pls") %>%
    set_mode("regression")
wflow <- workflow() %>% add_model(spec) %>% add_recipe(rec)
fit <- fit(wflow, data = train)

# mlr3
library(mlr3)
pls4all::register_mlr3()
lrn <- lrn("regr.pls4all", algorithm = "di_pls", n_components = 4L)
lrn$train(task)
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

_No idiomatic classdef wrapper — invoke `pls4all.fit("di_pls", X, y, …)` directly from the unified MEX factory._

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
<tbody class="lang-band lang-cpp"><tr class="lang-band-row" data-lang="cpp"><th colspan="3" scope="rowgroup"><span class="lang-band-dot"></span>C++ native · libp4a</th></tr>
<tr class="bk-row"><td class="bk-name"><code>pls4all.cpp.blas</code></td><td class="parity parity-exact">✓ 2e-02</td><td class="ms">1.02 ms</td></tr>
<tr class="bk-row"><td class="bk-name"><code>pls4all.cpp.blas+omp</code></td><td class="parity parity-exact">✓ 2e-02</td><td class="ms">1.02 ms</td></tr>
<tr class="bk-row"><td class="bk-name"><code>pls4all.cpp.omp</code></td><td class="parity parity-exact">✓ 2e-02</td><td class="ms ms-best">1.00 ms<span class="medal" title="fastest">🏆</span></td></tr>
<tr class="bk-row"><td class="bk-name"><code>pls4all.cpp.ref</code></td><td class="parity parity-exact">✓ 2e-02</td><td class="ms">1.24 ms</td></tr>
</tbody>
<tbody class="lang-band lang-python"><tr class="lang-band-row" data-lang="python"><th colspan="3" scope="rowgroup"><span class="lang-band-dot"></span>Python · pls4all</th></tr>
<tr class="bk-row"><td class="bk-name"><code>pls4all.python</code></td><td class="parity parity-exact">✓ bind</td><td class="ms">1.11 ms</td></tr>
<tr class="bk-row"><td class="bk-name"><code>pls4all.registry</code></td><td class="parity parity-exact">✓ bind</td><td class="ms">1.12 ms</td></tr>
<tr class="bk-row"><td class="bk-name"><code>pls4all.sklearn</code></td><td class="parity parity-exact">✓ 4e-15</td><td class="ms">1.20 ms</td></tr>
</tbody>
<tbody class="lang-band lang-r"><tr class="lang-band-row" data-lang="r"><th colspan="3" scope="rowgroup"><span class="lang-band-dot"></span>R · pls4all</th></tr>
<tr class="bk-row"><td class="bk-name"><code>pls4all.R</code></td><td class="parity parity-not_run">—</td><td class="ms">—</td></tr>
<tr class="bk-row"><td class="bk-name"><code>pls4all.R.formula</code></td><td class="parity parity-not_run">—</td><td class="ms">—</td></tr>
</tbody>
<tbody class="lang-band lang-matlab"><tr class="lang-band-row" data-lang="matlab"><th colspan="3" scope="rowgroup"><span class="lang-band-dot"></span>MATLAB · pls4all</th></tr>
<tr class="bk-row"><td class="bk-name"><code>pls4all.matlab</code></td><td class="parity parity-not_run">—</td><td class="ms">—</td></tr>
<tr class="bk-row"><td class="bk-name"><code>pls4all.matlab.classdef</code></td><td class="parity parity-not_run">—</td><td class="ms">—</td></tr>
</tbody>
<tbody class="lang-band lang-python"><tr class="lang-band-row" data-lang="python"><th colspan="3" scope="rowgroup"><span class="lang-band-dot"></span>Python · external</th></tr>
<tr class="bk-row"><td class="bk-name"><code>ikpls</code></td><td class="parity parity-not_available">⊘</td><td class="ms">—</td></tr>
<tr class="bk-row truth-source truth-source-relaxed"><td class="bk-name"><span class="truth-mark" title="Registry parity reference (python): diPLSlib 2.5.0 — relaxed (rmse_rel ≤ 2e-02)">📐</span><code>ref.python_diplslib</code></td><td class="parity parity-exact">✓ ref</td><td class="ms">3.59 ms</td></tr>
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
</table>
</div>

:::

::::


---

_See also_: [benchmark overview](../benchmarks/overview.md) · [methods index](index.md) · [interactive dashboard](../landing/dashboard.md)