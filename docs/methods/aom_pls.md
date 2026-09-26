# `aom_pls` — AOM-PLS (global adaptive operator selection)

_Group_: **Adaptive** · _Registry tolerance_: `1e-08`

## Description

AOM-PLS — global adaptive operator selection

> **Registry note** — Global AOMPLS/AOM-PLS selector with the compact strict-linear nirs4all bank: identity, Savitzky-Golay smooth/derivative, detrend and finite-difference operators. Reference is the in-tree nirs4all estimator stack; parity remains qualitative because selection tie-breaking and CV scoring details differ across implementations.

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

### Appropriate uses

Joint automated choice of preprocessing and PLS complexity for a defined calibration task.

### Limits and validation

The selected pipeline is conditional on the candidate grid and validation split and needs an untouched external assessment.

### Implementation

`n4m_model_selection_aom_pls_select` via the Python/R/MATLAB dispatchers. The compact strict-linear bank shipped by pls4all (`compact_bank`: identity, Savitzky–Golay smooth/derivative, polynomial detrending, finite differences) mirrors the `default_bank` used in the paper experiments. Reference: git-pinned oracle `nirs4all.operators.models.sklearn.aom_pls.AOMPLSRegressor` (sanctioned exception).

### Sources and provenance

Current implementation: [cpp/src/core/aom_preprocessing.cpp](https://github.com/GBeurier/nirs4all-methods/blob/main/cpp/src/core/aom_preprocessing.cpp).


### API and bindings

**C ABI (ABI 2):** [`n4m_model_selection_aom_pls_result_destroy`](https://github.com/GBeurier/nirs4all-methods/blob/main/cpp/include/n4m/model_selection.h#L149) · [`n4m_model_selection_aom_pls_result_get_best_score`](https://github.com/GBeurier/nirs4all-methods/blob/main/cpp/include/n4m/model_selection.h#L159) · [`n4m_model_selection_aom_pls_result_get_coefficients`](https://github.com/GBeurier/nirs4all-methods/blob/main/cpp/include/n4m/model_selection.h#L174) · [`n4m_model_selection_aom_pls_result_get_input_coefficients`](https://github.com/GBeurier/nirs4all-methods/blob/main/cpp/include/n4m/model_selection.h#L177) · [`n4m_model_selection_aom_pls_result_get_intercept`](https://github.com/GBeurier/nirs4all-methods/blob/main/cpp/include/n4m/model_selection.h#L180) · [`n4m_model_selection_aom_pls_result_get_max_components`](https://github.com/GBeurier/nirs4all-methods/blob/main/cpp/include/n4m/model_selection.h#L153) · [`n4m_model_selection_aom_pls_result_get_n_operators`](https://github.com/GBeurier/nirs4all-methods/blob/main/cpp/include/n4m/model_selection.h#L151) · [`n4m_model_selection_aom_pls_result_get_operator_kinds`](https://github.com/GBeurier/nirs4all-methods/blob/main/cpp/include/n4m/model_selection.h#L162) · [`n4m_model_selection_aom_pls_result_get_operator_scores`](https://github.com/GBeurier/nirs4all-methods/blob/main/cpp/include/n4m/model_selection.h#L165) · [`n4m_model_selection_aom_pls_result_get_predictions`](https://github.com/GBeurier/nirs4all-methods/blob/main/cpp/include/n4m/model_selection.h#L171) · [`n4m_model_selection_aom_pls_result_get_rmse_curves`](https://github.com/GBeurier/nirs4all-methods/blob/main/cpp/include/n4m/model_selection.h#L168) · [`n4m_model_selection_aom_pls_result_get_selected_n_components`](https://github.com/GBeurier/nirs4all-methods/blob/main/cpp/include/n4m/model_selection.h#L157) · [`n4m_model_selection_aom_pls_result_get_selected_operator_index`](https://github.com/GBeurier/nirs4all-methods/blob/main/cpp/include/n4m/model_selection.h#L155) · [`n4m_model_selection_aom_pls_select`](https://github.com/GBeurier/nirs4all-methods/blob/main/cpp/include/n4m/model_selection.h#L139). Use the linked public header for the exact signature, configuration, and result handles.

**Python (verified public re-export):**

```python
from n4m.model_selection.aom_search import aom_global_select
result = aom_global_select(X, y)
```

Source signature: `aom_global_select(X, y, *, max_components: int = 3, operators = None, cv: int = 3, fold_ids = None, center_x: bool | None = None, scale_x: bool | None = None, center_y: bool | None = None, scale_y: bool | None = None)` ([`n4m/_impl/native.py`](https://github.com/GBeurier/nirs4all-methods/blob/main/bindings/python/src/n4m/_impl/native.py#L6126)).

**R (source-verified):** [`aom_pls(X, Y, max_components = 3L, n_operators = 9L, cv = 3L)`](https://github.com/GBeurier/nirs4all-methods/blob/main/bindings/r/n4m/R/methods_extra.R).

```r
library(n4m)
result <- aom_pls(X, Y)
```

**MATLAB / Octave (source-verified):** [`aom_pls(X, Y, max_components, n_operators, cv)`](https://github.com/GBeurier/nirs4all-methods/blob/main/bindings/matlab/+n4m/aom_pls.m).

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
<tr class="bk-row"><td class="bk-name"><code>pls4all.cpp.blas+omp</code></td><td class="parity parity-ref-strict">✓ ref 6e-16</td><td class="ms">4.20 ms</td></tr>
</tbody>
<tbody class="lang-band lang-python"><tr class="lang-band-row" data-lang="python"><th colspan="3" scope="rowgroup"><span class="lang-band-dot"></span>Python · archived pls4all benchmark</th></tr>
<tr class="bk-row"><td class="bk-name"><code>pls4all.python</code></td><td class="parity parity-exact">✓ bind</td><td class="ms">4.16 ms</td></tr>
<tr class="bk-row"><td class="bk-name"><code>pls4all.sklearn</code></td><td class="parity parity-exact">✓ bind</td><td class="ms ms-best">4.15 ms<span class="medal" title="fastest">🏆</span></td></tr>
</tbody>
<tbody class="lang-band lang-r"><tr class="lang-band-row" data-lang="r"><th colspan="3" scope="rowgroup"><span class="lang-band-dot"></span>R · archived pls4all benchmark</th></tr>
<tr class="bk-row"><td class="bk-name"><code>pls4all.R</code></td><td class="parity parity-exact">✓ 6e-15</td><td class="ms">7.84 ms</td></tr>
<tr class="bk-row"><td class="bk-name"><code>pls4all.R.formula</code></td><td class="parity parity-exact">✓ 6e-15</td><td class="ms">9.43 ms</td></tr>
<tr class="bk-row"><td class="bk-name"><code>pls4all.R.mdatools</code></td><td class="parity parity-exact">✓ 6e-15</td><td class="ms">17.6 ms</td></tr>
<tr class="bk-row"><td class="bk-name"><code>pls4all.R.pls</code></td><td class="parity parity-exact">✓ 6e-15</td><td class="ms">9.87 ms</td></tr>
</tbody>
<tbody class="lang-band lang-python"><tr class="lang-band-row" data-lang="python"><th colspan="3" scope="rowgroup"><span class="lang-band-dot"></span>Python · external</th></tr>
<tr class="bk-row truth-source truth-source-strict"><td class="bk-name"><span class="truth-mark" title="Registry parity reference (python): nirs4all in-tree — strict (rmse_rel ≤ 1e-08)">📐</span><code>nirs4all</code></td><td class="parity parity-ref-source">source</td><td class="ms">41.9 ms</td></tr>
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
<tr class="bk-row"><td class="bk-name"><code>pls4all.cpp.blas+omp</code></td><td class="parity parity-ref-strict">✓ ref 6e-16</td><td class="ms">14.6 ms</td></tr>
</tbody>
<tbody class="lang-band lang-python"><tr class="lang-band-row" data-lang="python"><th colspan="3" scope="rowgroup"><span class="lang-band-dot"></span>Python · archived pls4all benchmark</th></tr>
<tr class="bk-row"><td class="bk-name"><code>pls4all.python</code></td><td class="parity parity-exact">✓ bind</td><td class="ms ms-best">8.48 ms<span class="medal" title="fastest">🏆</span></td></tr>
<tr class="bk-row"><td class="bk-name"><code>pls4all.sklearn</code></td><td class="parity parity-exact">✓ bind</td><td class="ms">14.7 ms</td></tr>
</tbody>
<tbody class="lang-band lang-r"><tr class="lang-band-row" data-lang="r"><th colspan="3" scope="rowgroup"><span class="lang-band-dot"></span>R · archived pls4all benchmark</th></tr>
<tr class="bk-row"><td class="bk-name"><code>pls4all.R</code></td><td class="parity parity-exact">✓ 6e-15</td><td class="ms">27.0 ms</td></tr>
<tr class="bk-row"><td class="bk-name"><code>pls4all.R.formula</code></td><td class="parity parity-exact">✓ 6e-15</td><td class="ms">40.4 ms</td></tr>
<tr class="bk-row"><td class="bk-name"><code>pls4all.R.mdatools</code></td><td class="parity parity-exact">✓ 6e-15</td><td class="ms">50.0 ms</td></tr>
<tr class="bk-row"><td class="bk-name"><code>pls4all.R.pls</code></td><td class="parity parity-exact">✓ 6e-15</td><td class="ms">47.9 ms</td></tr>
</tbody>
<tbody class="lang-band lang-python"><tr class="lang-band-row" data-lang="python"><th colspan="3" scope="rowgroup"><span class="lang-band-dot"></span>Python · external</th></tr>
<tr class="bk-row truth-source truth-source-strict"><td class="bk-name"><span class="truth-mark" title="Registry parity reference (python): nirs4all in-tree — strict (rmse_rel ≤ 1e-08)">📐</span><code>nirs4all</code></td><td class="parity parity-ref-source">source</td><td class="ms">59.7 ms</td></tr>
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
<tr class="bk-row"><td class="bk-name"><code>pls4all.cpp.blas+omp</code></td><td class="parity parity-ref-strict">✓ ref 6e-16</td><td class="ms">23.7 ms</td></tr>
</tbody>
<tbody class="lang-band lang-python"><tr class="lang-band-row" data-lang="python"><th colspan="3" scope="rowgroup"><span class="lang-band-dot"></span>Python · archived pls4all benchmark</th></tr>
<tr class="bk-row"><td class="bk-name"><code>pls4all.python</code></td><td class="parity parity-exact">✓ bind</td><td class="ms">23.7 ms</td></tr>
<tr class="bk-row"><td class="bk-name"><code>pls4all.sklearn</code></td><td class="parity parity-exact">✓ bind</td><td class="ms">15.5 ms</td></tr>
</tbody>
<tbody class="lang-band lang-r"><tr class="lang-band-row" data-lang="r"><th colspan="3" scope="rowgroup"><span class="lang-band-dot"></span>R · archived pls4all benchmark</th></tr>
<tr class="bk-row"><td class="bk-name"><code>pls4all.R</code></td><td class="parity parity-exact">✓ 6e-15</td><td class="ms">13.4 ms</td></tr>
<tr class="bk-row"><td class="bk-name"><code>pls4all.R.formula</code></td><td class="parity parity-exact">✓ 6e-15</td><td class="ms">16.4 ms</td></tr>
<tr class="bk-row"><td class="bk-name"><code>pls4all.R.mdatools</code></td><td class="parity parity-exact">✓ 6e-15</td><td class="ms">12.9 ms</td></tr>
<tr class="bk-row"><td class="bk-name"><code>pls4all.R.pls</code></td><td class="parity parity-exact">✓ 6e-15</td><td class="ms ms-best">12.4 ms<span class="medal" title="fastest">🏆</span></td></tr>
</tbody>
<tbody class="lang-band lang-python"><tr class="lang-band-row" data-lang="python"><th colspan="3" scope="rowgroup"><span class="lang-band-dot"></span>Python · external</th></tr>
<tr class="bk-row truth-source truth-source-strict"><td class="bk-name"><span class="truth-mark" title="Registry parity reference (python): nirs4all in-tree — strict (rmse_rel ≤ 1e-08)">📐</span><code>nirs4all</code></td><td class="parity parity-ref-source">source</td><td class="ms">16.6 ms</td></tr>
</tbody>
</table>
</div>

:::

::::


---

_See also_: [benchmark overview](../benchmarks/overview.md) · [methods index](index.md) · [interactive dashboard](../landing/dashboard.md)