# `aom_pls` — AOM-PLS (global adaptive operator selection)

_Group_: **Adaptive** · _Registry tolerance_: `0.001`

## Description

AOM-PLS — global adaptive operator selection

> **Registry note** — Global AOMPLS/AOM-PLS selector with the compact strict-linear nirs4all bank: identity, Savitzky-Golay smooth/derivative, detrend and finite-difference operators. Reference is the in-tree nirs4all estimator stack with the same deterministic contiguous CV folds as the ABI wrappers.

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

- 📐 **`nirs4all`** (python · python) — `nirs4all` in-tree · relaxed (rmse_rel ≤ 1e-03) — In-tree nirs4all AOM/POP estimator stack (sanctioned reference). The pls4all ABI uses the same compact strict-linear bank and contiguous folds for cross-binding determinism; nirs4all remains the qualitative algorithmic reference.
:::

### Validation contract

Structural validation contract for `aom_pls`, exported by the declarative validation framework (`benchmarks/validation/registry/`) from `benchmarks.parity_timing.registry`. Regenerated by `python -m benchmarks.validation.export_registry --write`.

| Field | Value |
|------|-------|
| Reference kind | `external` |
| Canonical reference role | `python` |
| Declared references | Python |
| Comparator | `reference_parity` (external-reference validity gate) |
| Comparator metric | `rmse_rel` |
| Reference tolerance | `1e-03` (relaxed) |
| Binding tolerance | `1e-06` |
| Prediction key | `predictions` |
| Required data flags | none |

**Benchmark cell parameters**

| Parameter | Value |
|-----------|-------|
| `cv` | `3` |
| `max_components` | `3` |
| `n_features` | `40` |
| `n_operators` | `9` |
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

**Reference gate**: non-release diagnostic — shape/smoke comparison only (`rmse_rel ≤ 5e+00`). Treat it as evidence that both implementations ran, not as numerical agreement.

Rows tagged with **📐** are the canonical parity references for this method (declared in [`parity_timing.registry`](../benchmarks/methodology.md)). C++ and external rows show reference parity; pls4all language bindings show binding parity against the C++ backend. Hover the icon for role and tolerance band.

::::{tab-set}
:class: parity-tabs

:::{tab-item} 1 thread
:sync: threads-1

<div class="parity-table-wrap">
<table class="docutils parity-grouped">
<thead><tr><th scope="col">Backend</th><th scope="col">Divergence</th><th class="size-col" scope="col">100×50 (ms)</th><th class="size-col" scope="col">100×500 (ms)</th><th class="size-col" scope="col">100×2500 (ms)</th><th class="size-col" scope="col">1000×500 (ms)</th></tr></thead>
<tbody class="lang-band lang-cpp"><tr class="lang-band-row" data-lang="cpp"><th colspan="6" scope="rowgroup"><span class="lang-band-dot"></span>C++ native · libp4a</th></tr>
<tr class="bk-row"><td class="bk-name"><code>pls4all.cpp.blas</code></td><td class="parity parity-divergence parity-ref-qualitative">2e-15</td><td class="ms">2.71 ms</td><td class="ms">22.5 ms</td><td class="ms">162.9 ms</td><td class="ms">295.7 ms</td></tr>
<tr class="bk-row"><td class="bk-name"><code>pls4all.cpp.blas+omp</code></td><td class="parity parity-divergence parity-ref-qualitative">2e-15</td><td class="ms">2.68 ms</td><td class="ms">22.7 ms</td><td class="ms">159.7 ms</td><td class="ms">301.3 ms</td></tr>
<tr class="bk-row"><td class="bk-name"><code>pls4all.cpp.omp</code></td><td class="parity parity-divergence parity-ref-qualitative">2e-15</td><td class="ms">2.92 ms</td><td class="ms">24.3 ms</td><td class="ms">171.0 ms</td><td class="ms">309.0 ms</td></tr>
<tr class="bk-row"><td class="bk-name"><code>pls4all.cpp.ref</code></td><td class="parity parity-divergence parity-ref-qualitative">2e-15</td><td class="ms">2.85 ms</td><td class="ms">23.5 ms</td><td class="ms">169.8 ms</td><td class="ms">307.6 ms</td></tr>
</tbody>
<tbody class="lang-band lang-python"><tr class="lang-band-row" data-lang="python"><th colspan="6" scope="rowgroup"><span class="lang-band-dot"></span>Python · pls4all</th></tr>
<tr class="bk-row"><td class="bk-name"><code>pls4all.python</code></td><td class="parity parity-divergence parity-drift">2e-15</td><td class="ms ms-best">2.56 ms<span class="medal" title="fastest">🏆</span></td><td class="ms">22.8 ms</td><td class="ms ms-best">157.4 ms<span class="medal" title="fastest">🏆</span></td><td class="ms">288.0 ms</td></tr>
<tr class="bk-row"><td class="bk-name"><code>pls4all.sklearn</code></td><td class="parity parity-divergence parity-drift">2e-15</td><td class="ms">2.77 ms</td><td class="ms ms-best">22.4 ms<span class="medal" title="fastest">🏆</span></td><td class="ms">158.7 ms</td><td class="ms">296.7 ms</td></tr>
</tbody>
<tbody class="lang-band lang-r"><tr class="lang-band-row" data-lang="r"><th colspan="6" scope="rowgroup"><span class="lang-band-dot"></span>R · pls4all</th></tr>
<tr class="bk-row"><td class="bk-name"><code>pls4all.R</code></td><td class="parity parity-divergence parity-drift">1e-15</td><td class="ms">5.20 ms</td><td class="ms">61.6 ms</td><td class="ms">476.1 ms</td><td class="ms">721.8 ms</td></tr>
<tr class="bk-row"><td class="bk-name"><code>pls4all.R.formula</code></td><td class="parity parity-divergence parity-drift">1e-15</td><td class="ms">6.60 ms</td><td class="ms">75.0 ms</td><td class="ms">652.8 ms</td><td class="ms">854.7 ms</td></tr>
<tr class="bk-row"><td class="bk-name"><code>pls4all.R.mdatools</code></td><td class="parity parity-divergence parity-drift">1e-15</td><td class="ms">6.30 ms</td><td class="ms">76.7 ms</td><td class="ms">658.2 ms</td><td class="ms">924.7 ms</td></tr>
<tr class="bk-row"><td class="bk-name"><code>pls4all.R.pls</code></td><td class="parity parity-divergence parity-drift">1e-15</td><td class="ms">6.69 ms</td><td class="ms">75.4 ms</td><td class="ms">655.3 ms</td><td class="ms">858.2 ms</td></tr>
</tbody>
<tbody class="lang-band lang-matlab"><tr class="lang-band-row" data-lang="matlab"><th colspan="6" scope="rowgroup"><span class="lang-band-dot"></span>MATLAB · pls4all</th></tr>
<tr class="bk-row"><td class="bk-name"><code>pls4all.matlab</code></td><td class="parity parity-divergence parity-drift">2e-15</td><td class="ms">3.16 ms</td><td class="ms">24.8 ms</td><td class="ms">176.9 ms</td><td class="ms">352.0 ms</td></tr>
<tr class="bk-row"><td class="bk-name"><code>pls4all.matlab.classdef</code></td><td class="parity parity-divergence parity-drift">2e-15</td><td class="ms">3.46 ms</td><td class="ms">25.9 ms</td><td class="ms">180.6 ms</td><td class="ms">382.3 ms</td></tr>
</tbody>
<tbody class="lang-band lang-python"><tr class="lang-band-row" data-lang="python"><th colspan="6" scope="rowgroup"><span class="lang-band-dot"></span>Python · external</th></tr>
<tr class="bk-row truth-source truth-source-relaxed"><td class="bk-name"><span class="truth-mark" title="Registry parity reference (python): nirs4all in-tree — relaxed (rmse_rel ≤ 1e-03)">📐</span><code>nirs4all</code></td><td class="parity parity-divergence parity-ref-source">0</td><td class="ms">12.0 ms</td><td class="ms">59.5 ms</td><td class="ms">440.1 ms</td><td class="ms ms-best">263.2 ms<span class="medal" title="fastest">🏆</span></td></tr>
</tbody>
</table>
</div>

:::

::::


---

_See also_: [benchmark overview](../benchmarks/overview.md) · [methods index](index.md) · [interactive dashboard](../landing/dashboard.md)