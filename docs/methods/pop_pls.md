# `pop_pls` — POP-PLS (per-component operator selection)

_Group_: **Adaptive** · _Registry tolerance_: `5.0`

## Description

POP-PLS — per-component adaptive operator selection

> **Registry note** — POPPLS/POP-PLS uses per-component operator selection over the same compact nirs4all bank. Reference is the in-tree nirs4all POPPLSRegressor; parity is qualitative.

### Parameters

| Name | Type | Default | Notes |
|------|------|---------|-------|
| `max_components` | `int` | `3` | registry benchmark cell value |
| `n_operators` | `int` | `9` | registry benchmark cell value |
| `cv` | `int` | `3` | registry benchmark cell value |

## Explanations

### Bibliographic source

Beurier, G., Reiter, R., Noûs, C., Rouan, L. & Cornet, D. (2026). *Reframing preprocessing selection as model-internal calibration in near-infrared spectroscopy: a large-scale benchmark of operator-adaptive PLS and Ridge models*. arXiv:2605.13587. https://arxiv.org/abs/2605.13587.

### Mathematical principle

POP-PLS (Per-Operator PLS) is the per-component ablation of AOM-PLS: each latent component may pick a *different* operator from the bank, rather than committing to one global operator. The setting is the same — centered $\mathbf{X} \in \mathbb{R}^{n\times p}$, response $\mathbf{Y}$, strict-linear bank $\{\mathbf{A}_b\}_{b=1}^{B}$, cross-covariance matrix $\mathbf{S} = \mathbf{X}^{\top}\mathbf{Y}$ — but the selection rule is local to each component.

**Per-component greedy selection.** Initialise $\mathbf{S}^{(0)} \leftarrow \mathbf{S}$. For $a = 1, \dots, K$:

1. **Score the bank** on the *current* deflated cross-covariance: for every $b$ evaluate the criterion $\mathcal{C}_a(b)$ of the SIMPLS-covariance step that would result from picking operator $b$ at component $a$ (covariance proxy $\lVert\mathbf{A}_b\mathbf{S}^{(a-1)}\rVert$, K-fold CV-RMSE on the resulting prefix, or approximate PRESS — same family of criteria as AOM-PLS).
2. **Pick the local minimiser** $b_a = \operatorname*{arg\,min}_b \mathcal{C}_a(b)$.
3. **Extract the component** $\mathbf{r}_a = \mathbf{u}_1\!\bigl(\mathbf{A}_{b_a}\mathbf{S}^{(a-1)}\bigr)$ in transformed space and lift it back through the *component-specific* adjoint:

$$\mathbf{z}_a \;=\; \mathbf{A}_{b_a}^{\top}\,\mathbf{r}_a, \qquad \mathbf{t}_a = \mathbf{X}\mathbf{z}_a.$$

4. **Deflate in the original space** so that the next component sees a residual cross-covariance free of $\mathbf{t}_a$:

$$\mathbf{S}^{(a)} \;=\; \bigl(\mathbf{I}_p - \mathbf{v}_a\mathbf{v}_a^{\top}\bigr)\mathbf{S}^{(a-1)}, \quad \mathbf{v}_a = \mathbf{p}_a / \lVert\mathbf{p}_a\rVert, \quad \mathbf{p}_a = \mathbf{X}^{\top}\mathbf{t}_a / \lVert\mathbf{t}_a\rVert^{2}.$$

**Closed-form coefficient.** With the selected sequence $(b_1, \dots, b_K)$ the model coefficients use exactly the same SIMPLS recovery formula as AOM-PLS, only with a *component-dependent* adjoint:

$$\mathbf{Z} = \bigl[\mathbf{A}_{b_1}^{\top}\mathbf{r}_1\;\cdots\;\mathbf{A}_{b_K}^{\top}\mathbf{r}_K\bigr], \qquad \mathbf{B} = \mathbf{Z}\bigl(\mathbf{P}^{\top}\mathbf{Z}\bigr)^{+}\mathbf{Q}^{\top}.$$

$\mathbf{B}$ lives in the original wavelength space, so — exactly as for AOM-PLS — predictions are a single dot product $\hat{\mathbf{Y}}(\mathbf{X}^{\star}) = \mathbf{X}^{\star}\mathbf{B}$, **with no preprocessing replay at predict time**. The relaxation buys wavelength-region adaptivity (the model can pick a smoother for one component and a derivative for the next), at the cost of $B$ extra cheap left actions per component.

### Implementation

`n4m_aom_per_component_select` via the Python/R/MATLAB dispatchers. Uses the same compact strict-linear bank as AOM-PLS; the per-component greedy is implemented in `select_per_component` (`aom_nirs/pls/selection.py`). Reference: git-pinned oracle `nirs4all.operators.models.sklearn.aom_pls.POPPLSRegressor` (sanctioned exception).

R roxygen note (`methods_extra.R::pop_pls`):

> POP-PLS with per-component operator selection.

MATLAB header (`bindings/matlab/+pls4all/pop_pls.m`):

```text
pls4all.pop_pls  POP-PLS per-component operator selection.
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
/* C ABI — libn4m AOM/POP selector path */
n4m_context_t* ctx = n4m_context_create();
n4m_config_t*  cfg = n4m_config_create();
n4m_operator_bank_t* bank = NULL;
n4m_validation_plan_t* plan = NULL;
n4m_aom_per_component_result_t* res = NULL;
n4m_operator_bank_create(&bank);
/* add compact nirs4all-style operators: identity, SG, detrend, FD */
n4m_validation_plan_create(&plan);
/* fill CV folds on plan */
n4m_aom_per_component_select(ctx, cfg, bank, &x_view, &y_view, plan,
              /* max_components */ 2, &res);
/* read predictions and selection diagnostics via result getters */
n4m_aom_per_component_result_destroy(res);
n4m_validation_plan_destroy(plan);
n4m_operator_bank_destroy(bank);
n4m_config_destroy(cfg);
n4m_context_destroy(ctx);
```

:::

:::{tab-item} Python · pls4all (raw)
:sync: python-raw
:class-label: lang-python

```python
import pls4all

with pls4all.Context() as ctx, pls4all.Config() as cfg:
    bank = pls4all.OperatorBank()
    plan = pls4all.ValidationPlan()
    # Add compact nirs4all-style operators and CV folds.
    res = pls4all.aom_per_component_select(
        ctx, cfg, bank, X.ravel(), y.ravel(), plan,
        max_components=2,
        x_rows=X.shape[0], x_cols=X.shape[1],
        y_rows=y.shape[0], y_cols=1,
    )
    values, rows, cols = res.predictions
```

:::

:::{tab-item} Python · pls4all.sklearn
:sync: python-sklearn
:class-label: lang-python

_No tier-2 sklearn-style class yet — exposed via the `pls4all.aom_global_select` / `pls4all.aom_per_component_select` low-level ABI._

:::

:::{tab-item} R · pls4all_method()
:sync: r-dispatcher
:class-label: lang-r

```r
library(pls4all)
# Unified low-level dispatcher (May 2026 R cleanup):
res <- pls4all_method("pop_pls", X, y,
                      n_components = 2L, params = list(max_components = 3L, n_operators = 9L, cv = 3L))
# res is a named list with MethodResult arrays/scalars.
# selected_indices / top_k_intervals are 1-based.
```

:::

:::{tab-item} R · pls4all (raw fn)
:sync: r-raw
:class-label: lang-r

```r
library(pls4all)
res  <- pop_pls(X, Y, max_components = 3L, n_operators = 9L, cv = 3L)
yhat <- pls4all_predict(res, X_test)
```

:::

:::{tab-item} MATLAB · pls4all (MEX)
:sync: matlab-mex
:class-label: lang-matlab

```matlab
res = pls4all.pop_pls(X, y, 2);
% see header of bindings/matlab/+pls4all/pop_pls.m for full
% parameter surface:
%   res = pop_pls(X, Y, max_components, n_operators, cv)
yhat = predict(res, Xtest);
```

:::

:::{tab-item} MATLAB · pls4all (classdef)
:sync: matlab-classdef
:class-label: lang-matlab

_No idiomatic classdef wrapper — invoke `pls4all.fit("pop_pls", X, y, …)` directly from the unified MEX factory._

:::

::::


**Registry parity references** 📐

:::{card}
:class-card: external-refs

- 📐 **`nirs4all`** (python · python) — `nirs4all` in-tree · qualitative (rmse_rel ≤ 5e+00) — In-tree nirs4all AOM/POP estimator stack (sanctioned reference). The pls4all ABI uses the same compact strict-linear bank and contiguous folds for cross-binding determinism; nirs4all remains the qualitative algorithmic reference.
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
<tr class="bk-row"><td class="bk-name"><code>pls4all.cpp.blas</code></td><td class="parity parity-exact">✓ 3e-15</td><td class="ms">2.80 ms</td></tr>
<tr class="bk-row"><td class="bk-name"><code>pls4all.cpp.blas+omp</code></td><td class="parity parity-exact">✓ 3e-15</td><td class="ms">3.05 ms</td></tr>
<tr class="bk-row"><td class="bk-name"><code>pls4all.cpp.omp</code></td><td class="parity parity-exact">✓ 3e-15</td><td class="ms ms-best">2.79 ms<span class="medal" title="fastest">🏆</span></td></tr>
<tr class="bk-row"><td class="bk-name"><code>pls4all.cpp.ref</code></td><td class="parity parity-exact">✓ 3e-15</td><td class="ms">3.03 ms</td></tr>
</tbody>
<tbody class="lang-band lang-python"><tr class="lang-band-row" data-lang="python"><th colspan="3" scope="rowgroup"><span class="lang-band-dot"></span>Python · pls4all</th></tr>
<tr class="bk-row"><td class="bk-name"><code>pls4all.python</code></td><td class="parity parity-exact">✓ bind</td><td class="ms">3.03 ms</td></tr>
<tr class="bk-row"><td class="bk-name"><code>pls4all.registry</code></td><td class="parity parity-exact">✓ bind</td><td class="ms">3.05 ms</td></tr>
<tr class="bk-row"><td class="bk-name"><code>pls4all.sklearn</code></td><td class="parity parity-exact">✓ bind</td><td class="ms">3.15 ms</td></tr>
</tbody>
<tbody class="lang-band lang-r"><tr class="lang-band-row" data-lang="r"><th colspan="3" scope="rowgroup"><span class="lang-band-dot"></span>R · pls4all</th></tr>
<tr class="bk-row"><td class="bk-name"><code>pls4all.R</code></td><td class="parity parity-exact">✓ bind</td><td class="ms">3.78 ms</td></tr>
<tr class="bk-row"><td class="bk-name"><code>pls4all.R.formula</code></td><td class="parity parity-exact">✓ bind</td><td class="ms">4.90 ms</td></tr>
</tbody>
<tbody class="lang-band lang-matlab"><tr class="lang-band-row" data-lang="matlab"><th colspan="3" scope="rowgroup"><span class="lang-band-dot"></span>MATLAB · pls4all</th></tr>
<tr class="bk-row"><td class="bk-name"><code>pls4all.matlab</code></td><td class="parity parity-exact">✓ bind</td><td class="ms">3.08 ms</td></tr>
<tr class="bk-row"><td class="bk-name"><code>pls4all.matlab.classdef</code></td><td class="parity parity-exact">✓ bind</td><td class="ms">3.37 ms</td></tr>
</tbody>
<tbody class="lang-band lang-python"><tr class="lang-band-row" data-lang="python"><th colspan="3" scope="rowgroup"><span class="lang-band-dot"></span>Python · external</th></tr>
<tr class="bk-row truth-source truth-source-qualitative"><td class="bk-name"><span class="truth-mark" title="Registry parity reference (python): nirs4all in-tree — qualitative (rmse_rel ≤ 5e+00)">📐</span><code>nirs4all</code></td><td class="parity parity-exact">✓ ref</td><td class="ms">26.1 ms</td></tr>
</tbody>
<tbody class="lang-band lang-ext"><tr class="lang-band-row" data-lang="ext"><th colspan="3" scope="rowgroup"><span class="lang-band-dot"></span>Other</th></tr>
<tr class="bk-row"><td class="bk-name"><code>r_mdatools_compat</code></td><td class="parity parity-exact">✓ bind</td><td class="ms">4.92 ms</td></tr>
<tr class="bk-row"><td class="bk-name"><code>r_pls_compat</code></td><td class="parity parity-exact">✓ bind</td><td class="ms">4.97 ms</td></tr>
</tbody>
</table>
</div>

:::

::::


---

_See also_: [benchmark overview](../benchmarks/overview.md) · [methods index](index.md) · [interactive dashboard](../landing/dashboard.md)