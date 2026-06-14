# `kernel_pls_rbf` — Kernel PLS (Rosipal & Trejo 2001)

_Group_: **Nonlinear / local** · _Registry tolerance_: `1e-08`

## Description

Non-linear kernel PLS (RBF kernel)

From the `pls4all.sklearn.KernelPLSRegression` docstring:

> Non-linear kernel PLS (Rosipal & Trejo 2001).

> **Registry note** — R `kernlab::kernelMatrix` (RBF/poly/sigmoid) + `pls::plsr` on the centered kernel matrix is the Rosipal-Trejo (2001) reference. pls4all's deflation ordering differs from the kernel-PLS-2 of Rosipal & Trejo so parity is qualitative.

### Parameters

| Name | Type | Default | Notes |
|------|------|---------|-------|
| `n_components` | `int` | `2` | Number of latent components extracted (k). |
| `kernel_type` | `int` | `1` | Kernel family: 0=linear, 1=RBF, 2=polynomial, 3=sigmoid. |
| `gamma` | `float` | `0.0` | RBF kernel bandwidth γ (with K(x, x') = exp(-γ‖x − x'‖²)). |
| `coef0` | `float` | `1.0` | Independent term in polynomial / sigmoid kernels. |
| `degree` | `int` | `3` | Degree of the polynomial kernel (ignored otherwise). |

## Explanations

### Bibliographic source

Rosipal, R. & Trejo, L. J. (2001). *Kernel partial least squares regression in reproducing kernel Hilbert space*. Journal of Machine Learning Research 2, 97–123.

### Mathematical principle

Kernel PLS runs the NIPALS PLS procedure entirely in the feature space of a Mercer kernel $k(\mathbf{x}, \mathbf{x}') = \langle \phi(\mathbf{x}), \phi(\mathbf{x}') \rangle$ without ever forming $\phi$ explicitly. The kernel matrix $\mathbf{K}_{ij} = k(\mathbf{x}_i, \mathbf{x}_j) \in \mathbb{R}^{n \times n}$ replaces $\mathbf{X}\mathbf{X}^{\top}$ in the NIPALS recursion and the score matrix is built directly from $\mathbf{K}$.

The RBF kernel $k(\mathbf{x}, \mathbf{x}') = \exp(-\gamma \|\mathbf{x} - \mathbf{x}'\|^2)$ is the standard choice for non-linear PLS: it captures smooth non-linear relationships between $\mathbf{X}$ and $y$ at the cost of a single bandwidth hyperparameter $\gamma$. Other kernels (polynomial, sigmoid) are available via the `kernel_type` enum.

Memory scales as $O(n^2)$ which is the binding constraint for kernel PLS on spectroscopy datasets; subsampling (Nyström) or random Fourier features are the standard scale-up strategies but are not currently exposed.

### Implementation

`n4m_estimators_kernel_pls_fit`. Predict-on-new-X is currently marked in-sample-only in the Python `sklearn` wrapper because the C ABI does not yet export the kernel-centring constants required to handle a fresh test point. The tier-1 entry point will refit on (X_train ∪ X_test) on demand.

R roxygen note (`methods_extra.R::kernel_pls_fit`):

> Non-linear kernel PLS (Rosipal & Trejo 2001).
> @param n_components Integer. Number of latent components.
> @param kernel_type Method-specific parameter. See the underlying `*_fit()` function for the exact semantics.
> @param gamma Numeric. CPPLS / kernel band parameter.
> @param coef0 Method-specific parameter. See the underlying `*_fit()` function for the exact semantics.
> @param degree Method-specific parameter. See the underlying `*_fit()` function for the exact semantics.
> @param X Numeric matrix of predictors (rows = samples, cols = features).
> @param Y Numeric matrix or vector of responses, with one row per sample.
> @export

MATLAB header (`bindings/matlab/+pls4all/KernelPlsRegression.m`):

```text
pls4all.KernelPlsRegression  Non-linear kernel PLS (Rosipal & Trejo 2001).

 **In-sample only**: the C ABI exports `alpha` and `y_mean` but NOT the
 kernel-centering state needed to compute K(X_new, X_train) at predict
 time. predict(X) returns the stored predictions when X matches the
 training matrix (content equality), otherwise raises an error.
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
n4m_estimators_kernel_pls_fit(ctx, cfg, &x_view, &y_view, /* hyperparams */, &res);
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
from pls4all._methods import kernel_pls_rbf_fit
with pls4all.Context() as ctx, pls4all.Config() as cfg:
    res = kernel_pls_rbf_fit(ctx, cfg, X, y, n_components=4)
# then: res.matrix("predictions"), res.matrix("coefficients"),
# res.vector("mask"), res.scalar("intercept"), …
```

:::

:::{tab-item} Python · pls4all.sklearn
:sync: python-sklearn
:class-label: lang-python

```python
from pls4all.sklearn import KernelPLSRegression
mdl = KernelPLSRegression(n_components=2, kernel_type=1, gamma=0.0, coef0=1.0, degree=3)
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
res <- pls4all_method("kernel_pls_rbf", X, y,
                      n_components = 4L, params = list(kernel_type = 1L, gamma = 0.02, coef0 = 1.0, degree = 3L))
# res is a named list with MethodResult arrays/scalars.
# selected_indices / top_k_intervals are 1-based.
```

:::

:::{tab-item} R · pls4all (raw fn)
:sync: r-raw
:class-label: lang-r

```r
library(pls4all)
res  <- kernel_pls_fit(X, Y, n_components,
                kernel_type = 1L, gamma = 0.0,
                coef0 = 1.0, degree = 3L)
yhat <- pls4all_predict(res, X_test)
```

:::

:::{tab-item} MATLAB · pls4all (MEX)
:sync: matlab-mex
:class-label: lang-matlab

```matlab
res = pls4all.kernel_pls(X, y, 4);
% see header of bindings/matlab/+pls4all/kernel_pls.m for full
% parameter surface:
%   res = kernel_pls(X, Y, n_components, kernel_type, gamma, coef0, degree)
yhat = predict(res, Xtest);
```

:::

:::{tab-item} MATLAB · pls4all (classdef)
:sync: matlab-classdef
:class-label: lang-matlab

```matlab
mdl  = pls4all.fit("kernel_pls_rbf", X, y, "NumComponents", 4);
yhat = predict(mdl, Xtest);
```

:::

::::


**Registry parity references** 📐

:::{card}
:class-card: external-refs

- 📐 **`ref.r_kernlab_pls`** (R · r) — `kernlab+pls` 0.9.33+2.8.5 · strict (rmse_rel ≤ 1e-08) — R `kernlab::kernelMatrix` (RBF/poly/sigmoid) + `pls::plsr` on the centered kernel matrix is a Rosipal-Trejo-style kernel PLS reference. pls4all uses a different deflation order so the parity is qualitative.
:::

### Benchmarks

Adaptive wall-clock per cell measured against [`full_matrix.csv`](../benchmarks/overview.md). Only backends that implement this method are listed; libraries without the method are omitted.

**Verdict** &nbsp;·&nbsp; ✓ ref / ≈ ref / ~ shape mark a reference-gate pass at strict / relaxed / qualitative tolerance &nbsp;·&nbsp; ✓ bind = pls4all binding agrees with the C++ baseline &nbsp;·&nbsp; ⇄ cross-check = documented by-design selector/RNG/model, noncanonical API/facade convention, or secondary oracle &nbsp;·&nbsp; ✗ divergent &nbsp;·&nbsp; ⚠ error &nbsp;·&nbsp; — not run. The fastest backend per column is marked 🏆.

**Reference gate**: strict — numeric equivalence (`rmse_rel_tol ≤ 1e-08`).

Rows tagged with **📐** are the canonical parity references for this method (declared in [`parity_timing.registry`](../benchmarks/methodology.md)). C++ and external rows show reference parity; pls4all language bindings show binding parity against the C++ backend. Hover the icon for role and tolerance band.

::::{tab-set}
:class: parity-tabs

:::{tab-item} 1 thread
:sync: threads-1

<div class="parity-table-wrap">
<table class="docutils parity-grouped">
<thead><tr><th scope="col">Backend</th><th scope="col">Parity</th><th class="size-col" scope="col">200×50 (ms)</th></tr></thead>
<tbody class="lang-band lang-cpp"><tr class="lang-band-row" data-lang="cpp"><th colspan="3" scope="rowgroup"><span class="lang-band-dot"></span>C++ native · libn4m</th></tr>
<tr class="bk-row"><td class="bk-name"><code>pls4all.cpp.blas+omp</code></td><td class="parity parity-ref-strict">✓ ref 5e-15</td><td class="ms ms-best">3.07 ms<span class="medal" title="fastest">🏆</span></td></tr>
</tbody>
<tbody class="lang-band lang-python"><tr class="lang-band-row" data-lang="python"><th colspan="3" scope="rowgroup"><span class="lang-band-dot"></span>Python · pls4all</th></tr>
<tr class="bk-row"><td class="bk-name"><code>pls4all.python</code></td><td class="parity parity-exact">✓ bind</td><td class="ms">3.23 ms</td></tr>
<tr class="bk-row"><td class="bk-name"><code>pls4all.sklearn</code></td><td class="parity parity-exact">✓ bind</td><td class="ms">3.23 ms</td></tr>
</tbody>
<tbody class="lang-band lang-r"><tr class="lang-band-row" data-lang="r"><th colspan="3" scope="rowgroup"><span class="lang-band-dot"></span>R · pls4all</th></tr>
<tr class="bk-row"><td class="bk-name"><code>pls4all.R</code></td><td class="parity parity-exact">✓ 1e-14</td><td class="ms">6.34 ms</td></tr>
<tr class="bk-row"><td class="bk-name"><code>pls4all.R.formula</code></td><td class="parity parity-exact">✓ 1e-14</td><td class="ms">7.05 ms</td></tr>
<tr class="bk-row"><td class="bk-name"><code>pls4all.R.mdatools</code></td><td class="parity parity-exact">✓ 1e-14</td><td class="ms">6.83 ms</td></tr>
<tr class="bk-row"><td class="bk-name"><code>pls4all.R.pls</code></td><td class="parity parity-exact">✓ 1e-14</td><td class="ms">7.21 ms</td></tr>
</tbody>
<tbody class="lang-band lang-r"><tr class="lang-band-row" data-lang="r"><th colspan="3" scope="rowgroup"><span class="lang-band-dot"></span>R · external</th></tr>
<tr class="bk-row truth-source truth-source-strict"><td class="bk-name"><span class="truth-mark" title="Registry parity reference (r): kernlab+pls 0.9.33+2.8.5 — strict (rmse_rel ≤ 1e-08)">📐</span><code>ref.r_kernlab_pls</code></td><td class="parity parity-ref-source">source</td><td class="ms">28.6 ms</td></tr>
</tbody>
</table>
</div>

:::

:::{tab-item} 3 threads
:sync: threads-3

<div class="parity-table-wrap">
<table class="docutils parity-grouped">
<thead><tr><th scope="col">Backend</th><th scope="col">Parity</th><th class="size-col" scope="col">200×50 (ms)</th></tr></thead>
<tbody class="lang-band lang-cpp"><tr class="lang-band-row" data-lang="cpp"><th colspan="3" scope="rowgroup"><span class="lang-band-dot"></span>C++ native · libn4m</th></tr>
<tr class="bk-row"><td class="bk-name"><code>pls4all.cpp.blas+omp</code></td><td class="parity parity-ref-strict">✓ ref 5e-15</td><td class="ms ms-best">3.12 ms<span class="medal" title="fastest">🏆</span></td></tr>
</tbody>
<tbody class="lang-band lang-python"><tr class="lang-band-row" data-lang="python"><th colspan="3" scope="rowgroup"><span class="lang-band-dot"></span>Python · pls4all</th></tr>
<tr class="bk-row"><td class="bk-name"><code>pls4all.python</code></td><td class="parity parity-exact">✓ bind</td><td class="ms">3.17 ms</td></tr>
<tr class="bk-row"><td class="bk-name"><code>pls4all.sklearn</code></td><td class="parity parity-exact">✓ bind</td><td class="ms">3.29 ms</td></tr>
</tbody>
<tbody class="lang-band lang-r"><tr class="lang-band-row" data-lang="r"><th colspan="3" scope="rowgroup"><span class="lang-band-dot"></span>R · pls4all</th></tr>
<tr class="bk-row"><td class="bk-name"><code>pls4all.R</code></td><td class="parity parity-exact">✓ 1e-14</td><td class="ms">5.98 ms</td></tr>
<tr class="bk-row"><td class="bk-name"><code>pls4all.R.formula</code></td><td class="parity parity-exact">✓ 1e-14</td><td class="ms">7.13 ms</td></tr>
<tr class="bk-row"><td class="bk-name"><code>pls4all.R.mdatools</code></td><td class="parity parity-exact">✓ 1e-14</td><td class="ms">7.16 ms</td></tr>
<tr class="bk-row"><td class="bk-name"><code>pls4all.R.pls</code></td><td class="parity parity-exact">✓ 1e-14</td><td class="ms">7.20 ms</td></tr>
</tbody>
<tbody class="lang-band lang-r"><tr class="lang-band-row" data-lang="r"><th colspan="3" scope="rowgroup"><span class="lang-band-dot"></span>R · external</th></tr>
<tr class="bk-row truth-source truth-source-strict"><td class="bk-name"><span class="truth-mark" title="Registry parity reference (r): kernlab+pls 0.9.33+2.8.5 — strict (rmse_rel ≤ 1e-08)">📐</span><code>ref.r_kernlab_pls</code></td><td class="parity parity-ref-source">source</td><td class="ms">27.3 ms</td></tr>
</tbody>
</table>
</div>

:::

:::{tab-item} 10 threads
:sync: threads-10

<div class="parity-table-wrap">
<table class="docutils parity-grouped">
<thead><tr><th scope="col">Backend</th><th scope="col">Parity</th><th class="size-col" scope="col">200×50 (ms)</th></tr></thead>
<tbody class="lang-band lang-cpp"><tr class="lang-band-row" data-lang="cpp"><th colspan="3" scope="rowgroup"><span class="lang-band-dot"></span>C++ native · libn4m</th></tr>
<tr class="bk-row"><td class="bk-name"><code>pls4all.cpp.blas+omp</code></td><td class="parity parity-ref-strict">✓ ref 5e-15</td><td class="ms">3.18 ms</td></tr>
</tbody>
<tbody class="lang-band lang-python"><tr class="lang-band-row" data-lang="python"><th colspan="3" scope="rowgroup"><span class="lang-band-dot"></span>Python · pls4all</th></tr>
<tr class="bk-row"><td class="bk-name"><code>pls4all.python</code></td><td class="parity parity-exact">✓ bind</td><td class="ms ms-best">3.14 ms<span class="medal" title="fastest">🏆</span></td></tr>
<tr class="bk-row"><td class="bk-name"><code>pls4all.sklearn</code></td><td class="parity parity-exact">✓ bind</td><td class="ms">3.22 ms</td></tr>
</tbody>
<tbody class="lang-band lang-r"><tr class="lang-band-row" data-lang="r"><th colspan="3" scope="rowgroup"><span class="lang-band-dot"></span>R · pls4all</th></tr>
<tr class="bk-row"><td class="bk-name"><code>pls4all.R</code></td><td class="parity parity-exact">✓ 1e-14</td><td class="ms">6.09 ms</td></tr>
<tr class="bk-row"><td class="bk-name"><code>pls4all.R.formula</code></td><td class="parity parity-exact">✓ 1e-14</td><td class="ms">7.28 ms</td></tr>
<tr class="bk-row"><td class="bk-name"><code>pls4all.R.mdatools</code></td><td class="parity parity-exact">✓ 1e-14</td><td class="ms">7.22 ms</td></tr>
<tr class="bk-row"><td class="bk-name"><code>pls4all.R.pls</code></td><td class="parity parity-exact">✓ 1e-14</td><td class="ms">7.41 ms</td></tr>
</tbody>
<tbody class="lang-band lang-r"><tr class="lang-band-row" data-lang="r"><th colspan="3" scope="rowgroup"><span class="lang-band-dot"></span>R · external</th></tr>
<tr class="bk-row truth-source truth-source-strict"><td class="bk-name"><span class="truth-mark" title="Registry parity reference (r): kernlab+pls 0.9.33+2.8.5 — strict (rmse_rel ≤ 1e-08)">📐</span><code>ref.r_kernlab_pls</code></td><td class="parity parity-ref-source">source</td><td class="ms">40.0 ms</td></tr>
</tbody>
</table>
</div>

:::

::::


---

_See also_: [benchmark overview](../benchmarks/overview.md) · [methods index](index.md) · [interactive dashboard](../landing/dashboard.md)