# `pop_pls` — POP-PLS (per-component operator selection)

_Group_: **Adaptive** · _Registry tolerance_: `1e-08`

## Description

POP-PLS — per-component adaptive operator selection

> **Registry note** — POPPLS/POP-PLS uses per-component operator selection over the same compact nirs4all bank. Reference is the in-tree nirs4all POPPLSRegressor; parity is qualitative.

### Parameters

| Name | Type | Default | Notes |
|------|------|---------|-------|
| `X` | `—` | `required` | current public binding signature |
| `y` | `—` | `required` | current public binding signature |
| `max_components` | `int` | `3` | current public binding signature |
| `operators` | `—` | `None` | current public binding signature |
| `cv` | `int` | `3` | current public binding signature |
| `fold_ids` | `—` | `None` | current public binding signature |
| `center_x` | `bool \| None` | `None` | current public binding signature |
| `scale_x` | `bool \| None` | `None` | current public binding signature |
| `center_y` | `bool \| None` | `None` | current public binding signature |
| `scale_y` | `bool \| None` | `None` | current public binding signature |
| `n_operators` | `int` | `9` | registry benchmark cell value |

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

### Appropriate uses

Population-based search over PLS model configurations when exhaustive enumeration is impractical.

### Limits and validation

Stochastic search has no guarantee of the global optimum and must evaluate candidates on leakage-free validation data.

### Implementation

`n4m_model_selection_pop_pls_select` via the Python/R/MATLAB dispatchers. Uses the same compact strict-linear bank as AOM-PLS; the per-component greedy is implemented in `select_per_component` (`aom_nirs/pls/selection.py`). Reference: git-pinned oracle `nirs4all.operators.models.sklearn.aom_pls.POPPLSRegressor` (sanctioned exception).

### Sources and provenance

Current implementation: [cpp/src/core/extra_pls.cpp](https://github.com/GBeurier/nirs4all-methods/blob/main/cpp/src/core/extra_pls.cpp).


### API and bindings

**C ABI (ABI 2):** [`n4m_model_selection_pop_pls_result_destroy`](https://github.com/GBeurier/nirs4all-methods/blob/main/cpp/include/n4m/model_selection.h#L194) · [`n4m_model_selection_pop_pls_result_get_best_score`](https://github.com/GBeurier/nirs4all-methods/blob/main/cpp/include/n4m/model_selection.h#L203) · [`n4m_model_selection_pop_pls_result_get_component_scores`](https://github.com/GBeurier/nirs4all-methods/blob/main/cpp/include/n4m/model_selection.h#L212) · [`n4m_model_selection_pop_pls_result_get_coefficients`](https://github.com/GBeurier/nirs4all-methods/blob/main/cpp/include/n4m/model_selection.h#L221) · [`n4m_model_selection_pop_pls_result_get_input_coefficients`](https://github.com/GBeurier/nirs4all-methods/blob/main/cpp/include/n4m/model_selection.h#L224) · [`n4m_model_selection_pop_pls_result_get_intercept`](https://github.com/GBeurier/nirs4all-methods/blob/main/cpp/include/n4m/model_selection.h#L227) · [`n4m_model_selection_pop_pls_result_get_max_components`](https://github.com/GBeurier/nirs4all-methods/blob/main/cpp/include/n4m/model_selection.h#L199) · [`n4m_model_selection_pop_pls_result_get_n_operators`](https://github.com/GBeurier/nirs4all-methods/blob/main/cpp/include/n4m/model_selection.h#L197) · [`n4m_model_selection_pop_pls_result_get_operator_kinds`](https://github.com/GBeurier/nirs4all-methods/blob/main/cpp/include/n4m/model_selection.h#L206) · [`n4m_model_selection_pop_pls_result_get_predictions`](https://github.com/GBeurier/nirs4all-methods/blob/main/cpp/include/n4m/model_selection.h#L218) · [`n4m_model_selection_pop_pls_result_get_prefix_scores`](https://github.com/GBeurier/nirs4all-methods/blob/main/cpp/include/n4m/model_selection.h#L215) · [`n4m_model_selection_pop_pls_result_get_selected_n_components`](https://github.com/GBeurier/nirs4all-methods/blob/main/cpp/include/n4m/model_selection.h#L201) · [`n4m_model_selection_pop_pls_result_get_selected_operator_indices`](https://github.com/GBeurier/nirs4all-methods/blob/main/cpp/include/n4m/model_selection.h#L209) · [`n4m_model_selection_pop_pls_select`](https://github.com/GBeurier/nirs4all-methods/blob/main/cpp/include/n4m/model_selection.h#L184). Use the linked public header for the exact signature, configuration, and result handles.

**Python (verified public re-export):**

```python
from n4m.model_selection.aom_search import aom_per_component_select
result = aom_per_component_select(X, y)
```

Source signature: `aom_per_component_select(X, y, *, max_components: int = 3, operators = None, cv: int = 3, fold_ids = None, center_x: bool | None = None, scale_x: bool | None = None, center_y: bool | None = None, scale_y: bool | None = None)` ([`n4m/_impl/native.py`](https://github.com/GBeurier/nirs4all-methods/blob/main/bindings/python/src/n4m/_impl/native.py#L6121)).

**R (source-verified):** [`pop_pls(X, Y, max_components = 3L, n_operators = 9L, cv = 3L)`](https://github.com/GBeurier/nirs4all-methods/blob/main/bindings/r/n4m/R/methods_extra.R).

```r
library(n4m)
result <- pop_pls(X, Y)
```

**MATLAB / Octave (source-verified):** [`pop_pls(X, Y, max_components, n_operators, cv)`](https://github.com/GBeurier/nirs4all-methods/blob/main/bindings/matlab/+n4m/pop_pls.m).

The source signature has additional required inputs, so no example call is fabricated.

**Registry parity references** 📐

:::{card}
:class-card: external-refs

- 📐 **`nirs4all`** (python · python) — `nirs4all` in-tree · strict (rmse_rel ≤ 1e-08) — In-tree nirs4all AOM/POP estimator stack (sanctioned reference). The pls4all ABI uses the same compact strict-linear bank and contiguous folds for cross-binding determinism; nirs4all remains the qualitative algorithmic reference.

:::

### Benchmarks

**Archived measurement identity.** Backend labels in this table are the raw IDs recorded when the benchmark ran (including historical `pls4all.*` IDs). They preserve measurement provenance and do not describe a current public Python, R, or MATLAB binding; use the source-verified **API and bindings** section above for current entry points.

Adaptive wall-clock per cell measured against [`full_matrix.csv`](../benchmarks/overview.md). Only backends that implement this method are listed; libraries without the method are omitted.

**Verdict** &nbsp;·&nbsp; ✓ ref / ≈ ref / ~ shape mark a reference-gate pass at strict / relaxed / qualitative tolerance &nbsp;·&nbsp; ✓ bind = archived binding-harness result agrees with the C++ baseline &nbsp;·&nbsp; ⇄ cross-check = documented by-design selector/RNG/model, noncanonical API/facade convention, or secondary oracle &nbsp;·&nbsp; ✗ divergent &nbsp;·&nbsp; ⚠ error &nbsp;·&nbsp; — not run. The fastest backend per column is marked 🏆.

**Reference gate**: strict — numeric equivalence (`rmse_rel_tol ≤ 1e-08`).

Rows tagged with **📐** are the canonical parity references for this method (declared in [`parity_timing.registry`](../benchmarks/methodology.md)). C++ and external rows show reference parity; archived language-harness rows show binding parity against the C++ backend. Hover the icon for role and tolerance band.

::::{tab-set}
:class: parity-tabs

:::{tab-item} 1 thread
:sync: threads-1

<div class="parity-table-wrap">
<table class="docutils parity-grouped">
<thead><tr><th scope="col">Backend</th><th scope="col">Parity</th><th class="size-col" scope="col">200×40 (ms)</th></tr></thead>
<tbody class="lang-band lang-cpp"><tr class="lang-band-row" data-lang="cpp"><th colspan="3" scope="rowgroup"><span class="lang-band-dot"></span>C++ native · libn4m</th></tr>
<tr class="bk-row"><td class="bk-name"><code>pls4all.cpp.blas+omp</code></td><td class="parity parity-ref-strict">✓ ref 5e-15</td><td class="ms ms-best">6.24 ms<span class="medal" title="fastest">🏆</span></td></tr>
</tbody>
<tbody class="lang-band lang-python"><tr class="lang-band-row" data-lang="python"><th colspan="3" scope="rowgroup"><span class="lang-band-dot"></span>Python · archived pls4all benchmark</th></tr>
<tr class="bk-row"><td class="bk-name"><code>pls4all.python</code></td><td class="parity parity-exact">✓ bind</td><td class="ms">7.23 ms</td></tr>
<tr class="bk-row"><td class="bk-name"><code>pls4all.sklearn</code></td><td class="parity parity-exact">✓ bind</td><td class="ms">6.51 ms</td></tr>
</tbody>
<tbody class="lang-band lang-r"><tr class="lang-band-row" data-lang="r"><th colspan="3" scope="rowgroup"><span class="lang-band-dot"></span>R · archived pls4all benchmark</th></tr>
<tr class="bk-row"><td class="bk-name"><code>pls4all.R</code></td><td class="parity parity-exact">✓ bind</td><td class="ms">12.0 ms</td></tr>
<tr class="bk-row"><td class="bk-name"><code>pls4all.R.formula</code></td><td class="parity parity-exact">✓ bind</td><td class="ms">12.5 ms</td></tr>
<tr class="bk-row"><td class="bk-name"><code>pls4all.R.mdatools</code></td><td class="parity parity-exact">✓ bind</td><td class="ms">24.7 ms</td></tr>
<tr class="bk-row"><td class="bk-name"><code>pls4all.R.pls</code></td><td class="parity parity-exact">✓ bind</td><td class="ms">13.2 ms</td></tr>
</tbody>
<tbody class="lang-band lang-python"><tr class="lang-band-row" data-lang="python"><th colspan="3" scope="rowgroup"><span class="lang-band-dot"></span>Python · external</th></tr>
<tr class="bk-row truth-source truth-source-strict"><td class="bk-name"><span class="truth-mark" title="Registry parity reference (python): nirs4all in-tree — strict (rmse_rel ≤ 1e-08)">📐</span><code>nirs4all</code></td><td class="parity parity-ref-source">source</td><td class="ms">47.8 ms</td></tr>
</tbody>
</table>
</div>

:::

:::{tab-item} 3 threads
:sync: threads-3

<div class="parity-table-wrap">
<table class="docutils parity-grouped">
<thead><tr><th scope="col">Backend</th><th scope="col">Parity</th><th class="size-col" scope="col">200×40 (ms)</th></tr></thead>
<tbody class="lang-band lang-cpp"><tr class="lang-band-row" data-lang="cpp"><th colspan="3" scope="rowgroup"><span class="lang-band-dot"></span>C++ native · libn4m</th></tr>
<tr class="bk-row"><td class="bk-name"><code>pls4all.cpp.blas+omp</code></td><td class="parity parity-ref-strict">✓ ref 5e-15</td><td class="ms">7.34 ms</td></tr>
</tbody>
<tbody class="lang-band lang-python"><tr class="lang-band-row" data-lang="python"><th colspan="3" scope="rowgroup"><span class="lang-band-dot"></span>Python · archived pls4all benchmark</th></tr>
<tr class="bk-row"><td class="bk-name"><code>pls4all.python</code></td><td class="parity parity-exact">✓ bind</td><td class="ms">7.46 ms</td></tr>
<tr class="bk-row"><td class="bk-name"><code>pls4all.sklearn</code></td><td class="parity parity-exact">✓ bind</td><td class="ms ms-best">6.77 ms<span class="medal" title="fastest">🏆</span></td></tr>
</tbody>
<tbody class="lang-band lang-r"><tr class="lang-band-row" data-lang="r"><th colspan="3" scope="rowgroup"><span class="lang-band-dot"></span>R · archived pls4all benchmark</th></tr>
<tr class="bk-row"><td class="bk-name"><code>pls4all.R</code></td><td class="parity parity-exact">✓ bind</td><td class="ms">13.2 ms</td></tr>
<tr class="bk-row"><td class="bk-name"><code>pls4all.R.formula</code></td><td class="parity parity-exact">✓ bind</td><td class="ms">12.9 ms</td></tr>
<tr class="bk-row"><td class="bk-name"><code>pls4all.R.mdatools</code></td><td class="parity parity-exact">✓ bind</td><td class="ms">10.0 ms</td></tr>
<tr class="bk-row"><td class="bk-name"><code>pls4all.R.pls</code></td><td class="parity parity-exact">✓ bind</td><td class="ms">12.4 ms</td></tr>
</tbody>
<tbody class="lang-band lang-python"><tr class="lang-band-row" data-lang="python"><th colspan="3" scope="rowgroup"><span class="lang-band-dot"></span>Python · external</th></tr>
<tr class="bk-row truth-source truth-source-strict"><td class="bk-name"><span class="truth-mark" title="Registry parity reference (python): nirs4all in-tree — strict (rmse_rel ≤ 1e-08)">📐</span><code>nirs4all</code></td><td class="parity parity-ref-source">source</td><td class="ms">36.4 ms</td></tr>
</tbody>
</table>
</div>

:::

:::{tab-item} 10 threads
:sync: threads-10

<div class="parity-table-wrap">
<table class="docutils parity-grouped">
<thead><tr><th scope="col">Backend</th><th scope="col">Parity</th><th class="size-col" scope="col">200×40 (ms)</th></tr></thead>
<tbody class="lang-band lang-cpp"><tr class="lang-band-row" data-lang="cpp"><th colspan="3" scope="rowgroup"><span class="lang-band-dot"></span>C++ native · libn4m</th></tr>
<tr class="bk-row"><td class="bk-name"><code>pls4all.cpp.blas+omp</code></td><td class="parity parity-ref-strict">✓ ref 5e-15</td><td class="ms">15.3 ms</td></tr>
</tbody>
<tbody class="lang-band lang-python"><tr class="lang-band-row" data-lang="python"><th colspan="3" scope="rowgroup"><span class="lang-band-dot"></span>Python · archived pls4all benchmark</th></tr>
<tr class="bk-row"><td class="bk-name"><code>pls4all.python</code></td><td class="parity parity-exact">✓ bind</td><td class="ms">11.1 ms</td></tr>
<tr class="bk-row"><td class="bk-name"><code>pls4all.sklearn</code></td><td class="parity parity-exact">✓ bind</td><td class="ms ms-best">7.06 ms<span class="medal" title="fastest">🏆</span></td></tr>
</tbody>
<tbody class="lang-band lang-r"><tr class="lang-band-row" data-lang="r"><th colspan="3" scope="rowgroup"><span class="lang-band-dot"></span>R · archived pls4all benchmark</th></tr>
<tr class="bk-row"><td class="bk-name"><code>pls4all.R</code></td><td class="parity parity-exact">✓ bind</td><td class="ms">34.2 ms</td></tr>
<tr class="bk-row"><td class="bk-name"><code>pls4all.R.formula</code></td><td class="parity parity-exact">✓ bind</td><td class="ms">22.8 ms</td></tr>
<tr class="bk-row"><td class="bk-name"><code>pls4all.R.mdatools</code></td><td class="parity parity-exact">✓ bind</td><td class="ms">13.5 ms</td></tr>
<tr class="bk-row"><td class="bk-name"><code>pls4all.R.pls</code></td><td class="parity parity-exact">✓ bind</td><td class="ms">12.5 ms</td></tr>
</tbody>
<tbody class="lang-band lang-python"><tr class="lang-band-row" data-lang="python"><th colspan="3" scope="rowgroup"><span class="lang-band-dot"></span>Python · external</th></tr>
<tr class="bk-row truth-source truth-source-strict"><td class="bk-name"><span class="truth-mark" title="Registry parity reference (python): nirs4all in-tree — strict (rmse_rel ≤ 1e-08)">📐</span><code>nirs4all</code></td><td class="parity parity-ref-source">source</td><td class="ms">38.6 ms</td></tr>
</tbody>
</table>
</div>

:::

::::


---

_See also_: [benchmark overview](../benchmarks/overview.md) · [methods index](index.md) · [interactive dashboard](../landing/dashboard.md)