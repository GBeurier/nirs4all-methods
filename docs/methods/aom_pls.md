# `aom_pls` — AOM-PLS (global adaptive operator selection)

_Group_: **Adaptive** · _Registry tolerance_: `5.0`

## Description

AOM-PLS — global adaptive operator selection

> **Registry note** — Global AOMPLS/AOM-PLS selector with the compact strict-linear nirs4all bank: identity, Savitzky-Golay smooth/derivative, detrend and finite-difference operators. Reference is the in-tree nirs4all estimator stack; parity remains qualitative because selection tie-breaking and CV scoring details differ across implementations.

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

AOM-PLS treats spectroscopic preprocessing as a learnable step *inside* the PLS calibration. Let $\mathbf{X} \in \mathbb{R}^{n\times p}$ be the centered spectral matrix (rows = samples), $\mathbf{Y} \in \mathbb{R}^{n\times q}$ the centered response, and $\{\mathbf{A}_b\}_{b=1}^{B} \subset \mathbb{R}^{p\times p}$ a finite bank of **strict-linear** spectral operators. An operator is strict-linear when its action $\mathbf{X}_b = \mathbf{X}\mathbf{A}_b^{\top}$ depends only on the fixed wavelength grid (identity, Savitzky–Golay smoothing and derivatives, finite differences, polynomial detrending, Norris–Williams gap derivatives, Whittaker smoothing); SNV, MSC, EMSC, ASLS and OSC are *not* strict-linear and are handled as fold-local branches in `nirs4all`.

**Cross-covariance identity** (Eq. 2 of the paper). For centered $\mathbf{X}, \mathbf{Y}$ and any strict-linear $\mathbf{A}$,

$$\bigl(\mathbf{X}\mathbf{A}^{\top}\bigr)^{\top}\mathbf{Y} \;=\; \mathbf{A}\,\mathbf{X}^{\top}\mathbf{Y}.$$

Writing $\mathbf{S} = \mathbf{X}^{\top}\mathbf{Y}$, every operator can therefore be *screened* by the cheap left action $\mathbf{S}_b = \mathbf{A}_b\mathbf{S}$ ($O(p q)$ per candidate) instead of materializing $\mathbf{X}_b$ ($O(n p)$).

**Global selection (the AOM in AOM-PLS).** A single operator index $b^{\star}$ is chosen for *all* $K$ components by minimising a selection criterion $\mathcal{C}$ over $b$:

$$b^{\star} \;=\; \operatorname*{arg\,min}_{b\in\{1,\dots,B\}} \; \mathcal{C}\!\bigl(\text{SIMPLS}(\mathbf{X}_b, \mathbf{Y}; K)\bigr).$$

The default criterion is K-fold CV-RMSE (`criterion='cv'`); alternatives include the covariance proxy $-\lVert\mathbf{A}_b\mathbf{S}\rVert$ (fast prescreen), leverage-corrected approximate PRESS, and a hybrid covariance-then-CV refinement. The optimal prefix length $k \le K$ is selected jointly when `auto_prefix=True`.

**SIMPLS-covariance engine.** With $\mathbf{S}_b = \mathbf{A}_b\mathbf{S}$ precomputed, SIMPLS extracts component $a$ from the leading left singular vector $\mathbf{r}_a = \mathbf{u}_1(\mathbf{S}_b)$ and maps it back to the original wavelength grid through the operator's adjoint:

$$\mathbf{z}_a \;=\; \mathbf{A}_{b^{\star}}^{\top}\,\mathbf{r}_a, \qquad \mathbf{t}_a = \mathbf{X}\mathbf{z}_a.$$

Stacking $\mathbf{Z} = [\mathbf{z}_1\;\cdots\;\mathbf{z}_K]$, with original-space loadings $\mathbf{P} = \mathbf{X}^{\top}\mathbf{T}\,\operatorname{diag}(1/\lVert\mathbf{t}_a\rVert^{2})$ and $\mathbf{Q} = \mathbf{Y}^{\top}\mathbf{T}\,\operatorname{diag}(1/\lVert\mathbf{t}_a\rVert^{2})$, the recovered coefficient matrix is

$$\mathbf{B} \;=\; \mathbf{Z}\,\bigl(\mathbf{P}^{\top}\mathbf{Z}\bigr)^{+}\mathbf{Q}^{\top}, \qquad \hat{\mathbf{Y}}(\mathbf{X}^{\star}) = \mathbf{X}^{\star}\mathbf{B}.$$

Because $\mathbf{B}$ lives in the *original* feature space, **the fitted model is a single linear calibration on the raw wavelength grid: there is no preprocessing stage to replay at predict time** — the operator has been absorbed into the coefficients (paper §3.2). Computationally the bank exploration cost is roughly that of a single SIMPLS fit on $\mathbf{S}$ plus $B$ tiny left actions, which is the algorithmic gain that makes AOM-PLS comparable to vanilla PLS even with a $\sim$77-operator default bank.

### Implementation

`p4a_aom_global_select` via the Python/R/MATLAB dispatchers. The compact strict-linear bank shipped by pls4all (`compact_bank`: identity, Savitzky–Golay smooth/derivative, polynomial detrending, finite differences) mirrors the `default_bank` used in the paper experiments. Reference: git-pinned oracle `nirs4all.operators.models.sklearn.aom_pls.AOMPLSRegressor` (sanctioned exception).

R roxygen note (`methods_extra.R::aom_pls`):

> AOM-PLS with global operator selection.

MATLAB header (`bindings/matlab/+pls4all/aom_pls.m`):

```text
pls4all.aom_pls  AOM-PLS global operator selection.
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
/* C ABI — libp4a AOM/POP selector path */
p4a_context_t* ctx = p4a_context_create();
p4a_config_t*  cfg = p4a_config_create();
p4a_operator_bank_t* bank = NULL;
p4a_validation_plan_t* plan = NULL;
p4a_aom_global_result_t* res = NULL;
p4a_operator_bank_create(&bank);
/* add compact nirs4all-style operators: identity, SG, detrend, FD */
p4a_validation_plan_create(&plan);
/* fill CV folds on plan */
p4a_aom_global_select(ctx, cfg, bank, &x_view, &y_view, plan,
              /* max_components */ 2, &res);
/* read predictions and selection diagnostics via result getters */
p4a_aom_global_result_destroy(res);
p4a_validation_plan_destroy(plan);
p4a_operator_bank_destroy(bank);
p4a_config_destroy(cfg);
p4a_context_destroy(ctx);
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
    res = pls4all.aom_global_select(
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
res <- pls4all_method("aom_pls", X, y,
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
res  <- aom_pls(X, Y, max_components = 3L, n_operators = 9L, cv = 3L)
yhat <- pls4all_predict(res, X_test)
```

:::

:::{tab-item} MATLAB · pls4all (MEX)
:sync: matlab-mex
:class-label: lang-matlab

```matlab
res = pls4all.aom_pls(X, y, 2);
% see header of bindings/matlab/+pls4all/aom_pls.m for full
% parameter surface:
%   res = aom_pls(X, Y, max_components, n_operators, cv)
yhat = predict(res, Xtest);
```

:::

:::{tab-item} MATLAB · pls4all (classdef)
:sync: matlab-classdef
:class-label: lang-matlab

_No idiomatic classdef wrapper — invoke `pls4all.fit("aom_pls", X, y, …)` directly from the unified MEX factory._

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
<tbody class="lang-band lang-cpp"><tr class="lang-band-row" data-lang="cpp"><th colspan="3" scope="rowgroup"><span class="lang-band-dot"></span>C++ native · libp4a</th></tr>
<tr class="bk-row"><td class="bk-name"><code>pls4all.cpp.blas</code></td><td class="parity parity-exact">✓ 7e-16</td><td class="ms">2.60 ms</td></tr>
<tr class="bk-row"><td class="bk-name"><code>pls4all.cpp.blas+omp</code></td><td class="parity parity-exact">✓ 7e-16</td><td class="ms">2.64 ms</td></tr>
<tr class="bk-row"><td class="bk-name"><code>pls4all.cpp.omp</code></td><td class="parity parity-exact">✓ 7e-16</td><td class="ms">2.63 ms</td></tr>
<tr class="bk-row"><td class="bk-name"><code>pls4all.cpp.ref</code></td><td class="parity parity-exact">✓ 7e-16</td><td class="ms">2.64 ms</td></tr>
</tbody>
<tbody class="lang-band lang-python"><tr class="lang-band-row" data-lang="python"><th colspan="3" scope="rowgroup"><span class="lang-band-dot"></span>Python · pls4all</th></tr>
<tr class="bk-row"><td class="bk-name"><code>pls4all.python</code></td><td class="parity parity-exact">✓ bind</td><td class="ms">2.60 ms</td></tr>
<tr class="bk-row"><td class="bk-name"><code>pls4all.registry</code></td><td class="parity parity-exact">✓ bind</td><td class="ms">2.60 ms</td></tr>
<tr class="bk-row"><td class="bk-name"><code>pls4all.sklearn</code></td><td class="parity parity-exact">✓ bind</td><td class="ms ms-best">2.58 ms<span class="medal" title="fastest">🏆</span></td></tr>
</tbody>
<tbody class="lang-band lang-r"><tr class="lang-band-row" data-lang="r"><th colspan="3" scope="rowgroup"><span class="lang-band-dot"></span>R · pls4all</th></tr>
<tr class="bk-row"><td class="bk-name"><code>pls4all.R</code></td><td class="parity parity-exact">✓ 5e-15</td><td class="ms">3.66 ms</td></tr>
<tr class="bk-row"><td class="bk-name"><code>pls4all.R.formula</code></td><td class="parity parity-exact">✓ 5e-15</td><td class="ms">4.92 ms</td></tr>
</tbody>
<tbody class="lang-band lang-matlab"><tr class="lang-band-row" data-lang="matlab"><th colspan="3" scope="rowgroup"><span class="lang-band-dot"></span>MATLAB · pls4all</th></tr>
<tr class="bk-row"><td class="bk-name"><code>pls4all.matlab</code></td><td class="parity parity-exact">✓ 5e-15</td><td class="ms">2.78 ms</td></tr>
<tr class="bk-row"><td class="bk-name"><code>pls4all.matlab.classdef</code></td><td class="parity parity-exact">✓ 5e-15</td><td class="ms">3.39 ms</td></tr>
</tbody>
<tbody class="lang-band lang-python"><tr class="lang-band-row" data-lang="python"><th colspan="3" scope="rowgroup"><span class="lang-band-dot"></span>Python · external</th></tr>
<tr class="bk-row truth-source truth-source-qualitative"><td class="bk-name"><span class="truth-mark" title="Registry parity reference (python): nirs4all in-tree — qualitative (rmse_rel ≤ 5e+00)">📐</span><code>nirs4all</code></td><td class="parity parity-exact">✓ ref</td><td class="ms">10.8 ms</td></tr>
</tbody>
<tbody class="lang-band lang-ext"><tr class="lang-band-row" data-lang="ext"><th colspan="3" scope="rowgroup"><span class="lang-band-dot"></span>Other</th></tr>
<tr class="bk-row"><td class="bk-name"><code>r_mdatools_compat</code></td><td class="parity parity-exact">✓ 5e-15</td><td class="ms">4.86 ms</td></tr>
<tr class="bk-row"><td class="bk-name"><code>r_pls_compat</code></td><td class="parity parity-exact">✓ 5e-15</td><td class="ms">5.13 ms</td></tr>
</tbody>
</table>
</div>

:::

::::


---

_See also_: [benchmark overview](../benchmarks/overview.md) · [methods index](index.md) · [interactive dashboard](../landing/dashboard.md)