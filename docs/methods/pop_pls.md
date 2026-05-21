# `pop_pls` — POP-PLS (per-component operator selection)

_Group_: **Adaptive** · _Registry tolerance_: `0.001`

## Description

POP-PLS — per-component adaptive operator selection

> **Registry note** — POPPLS/POP-PLS uses per-component operator selection over the same compact nirs4all bank. Reference is the in-tree nirs4all POPPLSRegressor with deterministic contiguous CV folds.

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

`p4a_aom_per_component_select` via the Python/R/MATLAB dispatchers. Uses the same compact strict-linear bank as AOM-PLS; the per-component greedy is implemented in `select_per_component` (`aom_nirs/pls/selection.py`). Reference: git-pinned oracle `nirs4all.operators.models.sklearn.aom_pls.POPPLSRegressor` (sanctioned exception).

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

:::{tab-item} C ABI · libp4a
:sync: c
:class-label: lang-c

```c
/* C ABI — libp4a AOM/POP selector path */
p4a_context_t* ctx = p4a_context_create();
p4a_config_t*  cfg = p4a_config_create();
p4a_operator_bank_t* bank = NULL;
p4a_validation_plan_t* plan = NULL;
p4a_aom_per_component_result_t* res = NULL;
p4a_operator_bank_create(&bank);
/* add compact nirs4all-style operators: identity, SG, detrend, FD */
p4a_validation_plan_create(&plan);
/* fill CV folds on plan */
p4a_aom_per_component_select(ctx, cfg, bank, &x_view, &y_view, plan,
              /* max_components */ 2, &res);
/* read predictions and selection diagnostics via result getters */
p4a_aom_per_component_result_destroy(res);
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

- 📐 **`nirs4all`** (python · python) — `nirs4all` in-tree · relaxed (rmse_rel ≤ 1e-03) — In-tree nirs4all AOM/POP estimator stack (sanctioned reference). The pls4all ABI uses the same compact strict-linear bank and contiguous folds for cross-binding determinism; nirs4all remains the qualitative algorithmic reference.
:::

### Validation contract

Structural validation contract for `pop_pls`, exported by the declarative validation framework (`benchmarks/validation/registry/`) from `benchmarks.parity_timing.registry`. Regenerated by `python -m benchmarks.validation.export_registry --write`.

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

_No benchmark rows recorded for this method._

---

_See also_: [benchmark overview](../benchmarks/overview.md) · [methods index](index.md) · [interactive dashboard](../landing/dashboard.md)