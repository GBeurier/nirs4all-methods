# `aom_preprocess` — AOM (Adaptive Operator Mixture) preprocessing bank

_Group_: **Diagnostic** · _Registry tolerance_: `5.0`

## Description

AOM preprocessing pipeline (§17 Phase 4)

> **Registry note** — In-tree `nirs4all.operators.models.sklearn.aom_pls` is the sanctioned provider. pls4all currently exposes the preprocessing primitive, while nirs4all exposes the full AOM/POP estimator stack; parity is qualitative.

### Parameters

| Name | Type | Default | Notes |
|------|------|---------|-------|
| `n_operators` | `int` | `3` | registry benchmark cell value |
| `gating_mode` | `int` | `0` | registry benchmark cell value |

## Explanations

### Bibliographic source

Beurier, G., Reiter, R., Noûs, C., Rouan, L. & Cornet, D. (2026). *Reframing preprocessing selection as model-internal calibration in near-infrared spectroscopy: a large-scale benchmark of operator-adaptive PLS and Ridge models*. arXiv:2605.13587. https://arxiv.org/abs/2605.13587 — introduces operator-adaptive PLS (AOM-PLS / POP-PLS) and the bench against 50+ NIRS datasets that the git-pinned oracle `nirs4all.operators.models.sklearn.aom_pls` is calibrated against.

### Mathematical principle

`aom_preprocess` is the **operator-bank primitive** that AOM-PLS and POP-PLS build on. Given the centered spectral matrix $\mathbf{X} \in \mathbb{R}^{n\times p}$ and a finite bank of strict-linear operators $\{\mathbf{A}_b\}_{b=1}^{M} \subset \mathbb{R}^{p\times p}$ — matrices fully determined by the wavelength grid (identity, Savitzky–Golay smooth/derivative, finite difference, polynomial detrending, Norris–Williams, Whittaker; SNV / MSC / EMSC / ASLS / OSC are excluded because they depend on $\mathbf{y}$ or on a reference spectrum) — `aom_preprocess` materializes the $M$ preprocessed views $\mathbf{X}_b = \mathbf{X}\mathbf{A}_b^{\top}$ and gates them.

Two gating modes are supported:

* **soft** ($\texttt{gating\_mode}=1$): equal-weight average

$$\mathbf{X}_{\text{AOM}}^{\text{soft}} \;=\; \frac{1}{M}\sum_{b=1}^{M}\mathbf{X}\mathbf{A}_b^{\top}.$$

* **hard** ($\texttt{gating\_mode}=0$): deterministic first-operator selection,

$$\mathbf{X}_{\text{AOM}}^{\text{hard}} \;=\; \mathbf{X}\mathbf{A}_{1}^{\top}.$$

Both modes preserve the **cross-covariance identity** exploited by the AOM/POP selectors: with $\mathbf{S} = \mathbf{X}^{\top}\mathbf{Y}$ and any $\mathbf{A}_b$ in the bank,

$$\bigl(\mathbf{X}\mathbf{A}_b^{\top}\bigr)^{\top}\mathbf{Y} \;=\; \mathbf{A}_b\,\mathbf{S},$$

so a downstream PLS step can score the whole bank by $M$ cheap $O(pq)$ left actions instead of $M$ full $O(np)$ matrix products. The motivation is that **no single preprocessing is best on all calibrations** — different wavelength regions favour different transforms — and the AOM-PLS / POP-PLS selectors exploit that by picking, respectively, a global operator (one $b^{\star}$ for the whole model) or a per-component operator (one $b_a$ for each latent direction). Predictions on new spectra reuse the absorbed operator(s) through the recovered original-space coefficients — **no preprocessing replay at predict time**.

### Implementation

`p4a_aom_preprocess_fit`. Reference: git-pinned oracle `nirs4all.operators.models.sklearn.aom_pls` (sanctioned exception).

R roxygen note (`methods_extra.R::aom_preprocess`):

> Adaptive Operator-Mixture preprocessing fit/transform.

MATLAB header (`bindings/matlab/+pls4all/aom_preprocess.m`):

```text
pls4all.aom_preprocess  AOM preprocessing fit/transform.
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
/* C ABI — libp4a */
p4a_context_t* ctx = p4a_context_create();
p4a_config_t*  cfg = p4a_config_create();
p4a_method_result_t* res = NULL;
p4a_aom_preprocess_fit(ctx, cfg, &x_view, &y_view, /* hyperparams */, &res);
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
from pls4all._methods import aom_preprocess_fit
with pls4all.Context() as ctx, pls4all.Config() as cfg:
    res = aom_preprocess_fit(ctx, cfg, X, y)
# then: res.matrix("predictions"), res.matrix("coefficients"),
# res.vector("mask"), res.scalar("intercept"), …
```

:::

:::{tab-item} Python · pls4all.sklearn
:sync: python-sklearn
:class-label: lang-python

```python
from pls4all.sklearn import aom_preprocess
result = aom_preprocess(X, y, n_components=2)
```

:::

:::{tab-item} R · pls4all_method()
:sync: r-dispatcher
:class-label: lang-r

```r
library(pls4all)
# Unified low-level dispatcher (May 2026 R cleanup):
res <- pls4all_method("aom_preprocess", X, y,
                      n_components = 2L, params = list(n_operators = 3L, gating_mode = 0L))
# res is a named list with MethodResult arrays/scalars.
# selected_indices / top_k_intervals are 1-based.
```

:::

:::{tab-item} R · pls4all (raw fn)
:sync: r-raw
:class-label: lang-r

```r
library(pls4all)
res  <- aom_preprocess(X, Y = NULL, n_operators = 3L, gating_mode = 0L)
yhat <- pls4all_predict(res, X_test)
```

:::

:::{tab-item} MATLAB · pls4all (MEX)
:sync: matlab-mex
:class-label: lang-matlab

```matlab
res = pls4all.aom_preprocess(X, y, 2);
% see header of bindings/matlab/+pls4all/aom_preprocess.m for full
% parameter surface:
%   res = aom_preprocess(X, Y, n_operators, gating_mode)
yhat = predict(res, Xtest);
```

:::

:::{tab-item} MATLAB · pls4all (classdef)
:sync: matlab-classdef
:class-label: lang-matlab

_No idiomatic classdef wrapper — invoke `pls4all.fit("aom_preprocess", X, y, …)` directly from the unified MEX factory._

:::

::::


**Registry parity references** 📐

:::{card}
:class-card: external-refs

- 📐 **`nirs4all`** (python · python) — `nirs4all` in-tree · qualitative (rmse_rel ≤ 5e+00) — In-tree nirs4all AOM provider (sanctioned external reference). pls4all's current primitive exposes a small operator-bank preprocessing kernel, while nirs4all exposes the full AOM/POP estimator stack; the parity remains qualitative.
:::

### Validation contract

Structural validation contract for `aom_preprocess`, exported by the declarative validation framework (`benchmarks/validation/registry/`) from `benchmarks.parity_timing.registry`. Regenerated by `python -m benchmarks.validation.export_registry --write`.

| Field | Value |
|------|-------|
| Reference kind | `external` |
| Canonical reference role | `python` |
| Declared references | Python |
| Comparator | `reference_parity` (external-reference validity gate) |
| Comparator metric | `rmse_rel` |
| Reference tolerance | `5e+00` (qualitative) |
| Binding tolerance | `1e-06` |
| Prediction key | `transformed` |
| Required data flags | none |

**Benchmark cell parameters**

| Parameter | Value |
|-----------|-------|
| `gating_mode` | `0` |
| `n_features` | `40` |
| `n_operators` | `3` |
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
<tr class="bk-row"><td class="bk-name"><code>pls4all.cpp.blas</code></td><td class="parity parity-divergence parity-ref-qualitative">0</td><td class="ms ms-best">1.01 ms<span class="medal" title="fastest">🏆</span></td><td class="ms">8.73 ms</td><td class="ms">46.3 ms</td><td class="ms">87.3 ms</td></tr>
<tr class="bk-row"><td class="bk-name"><code>pls4all.cpp.blas+omp</code></td><td class="parity parity-divergence parity-ref-qualitative">0</td><td class="ms">1.13 ms</td><td class="ms ms-best">8.70 ms<span class="medal" title="fastest">🏆</span></td><td class="ms">47.1 ms</td><td class="ms">87.5 ms</td></tr>
<tr class="bk-row"><td class="bk-name"><code>pls4all.cpp.omp</code></td><td class="parity parity-divergence parity-ref-qualitative">0</td><td class="ms">1.07 ms</td><td class="ms">8.91 ms</td><td class="ms">46.7 ms</td><td class="ms">86.7 ms</td></tr>
<tr class="bk-row"><td class="bk-name"><code>pls4all.cpp.ref</code></td><td class="parity parity-divergence parity-ref-qualitative">0</td><td class="ms">1.01 ms</td><td class="ms">8.88 ms</td><td class="ms">47.3 ms</td><td class="ms ms-best">85.5 ms<span class="medal" title="fastest">🏆</span></td></tr>
</tbody>
<tbody class="lang-band lang-python"><tr class="lang-band-row" data-lang="python"><th colspan="6" scope="rowgroup"><span class="lang-band-dot"></span>Python · pls4all</th></tr>
<tr class="bk-row"><td class="bk-name"><code>pls4all.python</code></td><td class="parity parity-divergence parity-ref-qualitative">0</td><td class="ms">1.09 ms</td><td class="ms">8.76 ms</td><td class="ms">46.6 ms</td><td class="ms">86.7 ms</td></tr>
<tr class="bk-row"><td class="bk-name"><code>pls4all.sklearn</code></td><td class="parity parity-divergence parity-ref-qualitative">0</td><td class="ms">1.02 ms</td><td class="ms">8.95 ms</td><td class="ms ms-best">45.4 ms<span class="medal" title="fastest">🏆</span></td><td class="ms">93.1 ms</td></tr>
</tbody>
<tbody class="lang-band lang-r"><tr class="lang-band-row" data-lang="r"><th colspan="6" scope="rowgroup"><span class="lang-band-dot"></span>R · pls4all</th></tr>
<tr class="bk-row"><td class="bk-name"><code>pls4all.R</code></td><td class="parity parity-divergence parity-ref-qualitative">0</td><td class="ms">3.67 ms</td><td class="ms">48.1 ms</td><td class="ms">534.5 ms</td><td class="ms">606.2 ms</td></tr>
<tr class="bk-row"><td class="bk-name"><code>pls4all.R.formula</code></td><td class="parity parity-divergence parity-ref-qualitative">0</td><td class="ms">5.20 ms</td><td class="ms">63.8 ms</td><td class="ms">658.7 ms</td><td class="ms">601.9 ms</td></tr>
<tr class="bk-row"><td class="bk-name"><code>pls4all.R.mdatools</code></td><td class="parity parity-divergence parity-ref-qualitative">0</td><td class="ms">5.38 ms</td><td class="ms">74.8 ms</td><td class="ms">640.0 ms</td><td class="ms">693.6 ms</td></tr>
<tr class="bk-row"><td class="bk-name"><code>pls4all.R.pls</code></td><td class="parity parity-divergence parity-ref-qualitative">0</td><td class="ms">5.34 ms</td><td class="ms">71.3 ms</td><td class="ms">653.3 ms</td><td class="ms">743.8 ms</td></tr>
</tbody>
<tbody class="lang-band lang-matlab"><tr class="lang-band-row" data-lang="matlab"><th colspan="6" scope="rowgroup"><span class="lang-band-dot"></span>MATLAB · pls4all</th></tr>
<tr class="bk-row"><td class="bk-name"><code>pls4all.matlab</code></td><td class="parity parity-divergence parity-ref-qualitative">0</td><td class="ms">1.77 ms</td><td class="ms">13.9 ms</td><td class="ms">69.8 ms</td><td class="ms">139.7 ms</td></tr>
<tr class="bk-row"><td class="bk-name"><code>pls4all.matlab.classdef</code></td><td class="parity parity-divergence parity-ref-qualitative">0</td><td class="ms">2.36 ms</td><td class="ms">15.3 ms</td><td class="ms">68.5 ms</td><td class="ms">141.0 ms</td></tr>
</tbody>
<tbody class="lang-band lang-python"><tr class="lang-band-row" data-lang="python"><th colspan="6" scope="rowgroup"><span class="lang-band-dot"></span>Python · external</th></tr>
<tr class="bk-row truth-source truth-source-qualitative"><td class="bk-name"><span class="truth-mark" title="Registry parity reference (python): nirs4all in-tree — qualitative (rmse_rel ≤ 5e+00)">📐</span><code>nirs4all</code></td><td class="parity parity-divergence parity-ref-source">0</td><td class="ms">1.34 ms</td><td class="ms">9.45 ms</td><td class="ms">47.5 ms</td><td class="ms">90.3 ms</td></tr>
</tbody>
</table>
</div>

:::

::::


---

_See also_: [benchmark overview](../benchmarks/overview.md) · [methods index](index.md) · [interactive dashboard](../landing/dashboard.md)