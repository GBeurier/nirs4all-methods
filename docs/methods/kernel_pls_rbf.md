# `kernel_pls_rbf` — Kernel PLS (Rosipal & Trejo 2001)

_Group_: **Nonlinear / local** · _Registry tolerance_: `2.0`

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

`p4a_kernel_pls_fit`. Predict-on-new-X is currently marked in-sample-only in the Python `sklearn` wrapper because the C ABI does not yet export the kernel-centring constants required to handle a fresh test point. The tier-1 entry point will refit on (X_train ∪ X_test) on demand.

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
p4a_kernel_pls_rbf_fit(ctx, cfg, &x_view, &y_view, /* hyperparams */, &res);
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

:::{tab-item} R · parsnip / mlr3
:sync: r-meta
:class-label: lang-r

```r
# parsnip / tidymodels
library(tidymodels)
pls4all::register_parsnip()
spec <- pls_pls4all_reg(num_comp = 4) %>%
    set_engine("pls4all", algorithm = "kernel_pls_rbf") %>%
    set_mode("regression")
wflow <- workflow() %>% add_model(spec) %>% add_recipe(rec)
fit <- fit(wflow, data = train)

# mlr3
library(mlr3)
pls4all::register_mlr3()
lrn <- lrn("regr.pls4all", algorithm = "kernel_pls_rbf", n_components = 4L)
lrn$train(task)
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

- 📐 **`ref.r_kernlab_pls`** (R · r) — `kernlab+pls` 0.9.33+2.8.5 · qualitative (rmse_rel ≤ 2e+00) — R `kernlab::kernelMatrix` (RBF/poly/sigmoid) + `pls::plsr` on the centered kernel matrix is a Rosipal-Trejo-style kernel PLS reference. pls4all uses a different deflation order so the parity is qualitative.
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
<tr class="bk-row"><td class="bk-name"><code>pls4all.cpp.blas</code></td><td class="parity parity-exact">✓ 1e-01</td><td class="ms">1.36 ms</td></tr>
<tr class="bk-row"><td class="bk-name"><code>pls4all.cpp.blas+omp</code></td><td class="parity parity-exact">✓ 1e-01</td><td class="ms">1.39 ms</td></tr>
<tr class="bk-row"><td class="bk-name"><code>pls4all.cpp.omp</code></td><td class="parity parity-exact">✓ 1e-01</td><td class="ms">1.41 ms</td></tr>
<tr class="bk-row"><td class="bk-name"><code>pls4all.cpp.ref</code></td><td class="parity parity-exact">✓ 1e-01</td><td class="ms ms-best">1.33 ms<span class="medal" title="fastest">🏆</span></td></tr>
</tbody>
<tbody class="lang-band lang-python"><tr class="lang-band-row" data-lang="python"><th colspan="3" scope="rowgroup"><span class="lang-band-dot"></span>Python · pls4all</th></tr>
<tr class="bk-row"><td class="bk-name"><code>pls4all.python</code></td><td class="parity parity-exact">✓ bind</td><td class="ms">1.38 ms</td></tr>
<tr class="bk-row"><td class="bk-name"><code>pls4all.registry</code></td><td class="parity parity-exact">✓ bind</td><td class="ms">1.39 ms</td></tr>
<tr class="bk-row"><td class="bk-name"><code>pls4all.sklearn</code></td><td class="parity parity-exact">✓ bind</td><td class="ms">1.48 ms</td></tr>
</tbody>
<tbody class="lang-band lang-r"><tr class="lang-band-row" data-lang="r"><th colspan="3" scope="rowgroup"><span class="lang-band-dot"></span>R · pls4all</th></tr>
<tr class="bk-row"><td class="bk-name"><code>pls4all.R</code></td><td class="parity parity-exact">✓ bind</td><td class="ms">3.50 ms</td></tr>
<tr class="bk-row"><td class="bk-name"><code>pls4all.R.formula</code></td><td class="parity parity-not_run">—</td><td class="ms">—</td></tr>
</tbody>
<tbody class="lang-band lang-matlab"><tr class="lang-band-row" data-lang="matlab"><th colspan="3" scope="rowgroup"><span class="lang-band-dot"></span>MATLAB · pls4all</th></tr>
<tr class="bk-row"><td class="bk-name"><code>pls4all.matlab</code></td><td class="parity parity-exact">✓ bind</td><td class="ms">1.91 ms</td></tr>
<tr class="bk-row"><td class="bk-name"><code>pls4all.matlab.classdef</code></td><td class="parity parity-not_run">—</td><td class="ms">—</td></tr>
</tbody>
<tbody class="lang-band lang-python"><tr class="lang-band-row" data-lang="python"><th colspan="3" scope="rowgroup"><span class="lang-band-dot"></span>Python · external</th></tr>
<tr class="bk-row"><td class="bk-name"><code>ikpls</code></td><td class="parity parity-not_available">⊘</td><td class="ms">—</td></tr>
<tr class="bk-row"><td class="bk-name"><code>sklearn</code></td><td class="parity parity-not_available">⊘</td><td class="ms">—</td></tr>
</tbody>
<tbody class="lang-band lang-r"><tr class="lang-band-row" data-lang="r"><th colspan="3" scope="rowgroup"><span class="lang-band-dot"></span>R · external</th></tr>
<tr class="bk-row"><td class="bk-name"><code>mixOmics</code></td><td class="parity parity-not_available">⊘</td><td class="ms">—</td></tr>
<tr class="bk-row"><td class="bk-name"><code>pls</code></td><td class="parity parity-not_available">⊘</td><td class="ms">—</td></tr>
<tr class="bk-row truth-source truth-source-qualitative"><td class="bk-name"><span class="truth-mark" title="Registry parity reference (r): kernlab+pls 0.9.33+2.8.5 — qualitative (rmse_rel ≤ 2e+00)">📐</span><code>ref.r_kernlab_pls</code></td><td class="parity parity-exact">✓ ref</td><td class="ms">299.6 ms</td></tr>
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