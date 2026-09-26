# `aom_preprocess` — AOM (Adaptive Operator Mixture) preprocessing bank

_Group_: **Diagnostic** · _Registry tolerance_: `1e-08`

## Description

AOM preprocessing pipeline (§17 Phase 4)

> **Registry note** — In-tree `nirs4all.operators.models.sklearn.aom_pls` is the sanctioned provider. pls4all currently exposes the preprocessing primitive, while nirs4all exposes the full AOM/POP estimator stack; parity is qualitative.

### Parameters

| Name | Type | Default | Notes |
|------|------|---------|-------|
| `X` | `—` | `required` | current public binding signature |
| `y` | `—` | `None` | current public binding signature |
| `operators` | `—` | `None` | current public binding signature |
| `gating_mode` | `str \| int` | `'soft'` | current public binding signature |
| `n_operators` | `int` | `3` | registry benchmark cell value |

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

### Appropriate uses

Automated comparison of candidate spectral preprocessing chains under an explicit validation protocol.

### Limits and validation

Searching many chains increases selection bias; every fitted transform and chain choice must be confined to training folds.

### Implementation

`n4m_model_selection_aom_preprocessing_fit`. Reference: git-pinned oracle `nirs4all.operators.models.sklearn.aom_pls` (sanctioned exception).

### Sources and provenance

Current implementation: [cpp/src/core/aom_preprocessing.cpp](https://github.com/GBeurier/nirs4all-methods/blob/main/cpp/src/core/aom_preprocessing.cpp).


### API and bindings

**C ABI (ABI 2):** [`n4m_model_selection_aom_preprocessing_fit`](https://github.com/GBeurier/nirs4all-methods/blob/main/cpp/include/n4m/model_selection.h#L453). Use the linked public header for the exact signature, configuration, and result handles.

**Python (verified public re-export):**

```python
from n4m.model_selection.aom_search import aom_preprocess
result = aom_preprocess(X)
```

Source signature: `aom_preprocess(X, y = None, *, operators = None, gating_mode: str | int = 'soft')` ([`n4m/_impl/native.py`](https://github.com/GBeurier/nirs4all-methods/blob/main/bindings/python/src/n4m/_impl/native.py#L5786)).

**R:** no current source-verified entry point was found for this catalog method.

**MATLAB / Octave:** no current source-verified entry point was found for this catalog method.

**Registry parity references** 📐

:::{card}
:class-card: external-refs

- 📐 **`nirs4all`** (python · python) — `nirs4all` in-tree · strict (rmse_rel ≤ 1e-08) — In-tree nirs4all AOM provider (sanctioned external reference). pls4all's current primitive exposes a small operator-bank preprocessing kernel, while nirs4all exposes the full AOM/POP estimator stack; the parity remains qualitative.

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
<tr class="bk-row"><td class="bk-name"><code>pls4all.cpp.blas+omp</code></td><td class="parity parity-ref-strict">✓ ref</td><td class="ms">1.85 ms</td></tr>
</tbody>
<tbody class="lang-band lang-python"><tr class="lang-band-row" data-lang="python"><th colspan="3" scope="rowgroup"><span class="lang-band-dot"></span>Python · archived pls4all benchmark</th></tr>
<tr class="bk-row"><td class="bk-name"><code>pls4all.python</code></td><td class="parity parity-exact">✓ bind</td><td class="ms">1.76 ms</td></tr>
<tr class="bk-row"><td class="bk-name"><code>pls4all.sklearn</code></td><td class="parity parity-exact">✓ bind</td><td class="ms ms-best">1.75 ms<span class="medal" title="fastest">🏆</span></td></tr>
</tbody>
<tbody class="lang-band lang-r"><tr class="lang-band-row" data-lang="r"><th colspan="3" scope="rowgroup"><span class="lang-band-dot"></span>R · archived pls4all benchmark</th></tr>
<tr class="bk-row"><td class="bk-name"><code>pls4all.R</code></td><td class="parity parity-exact">✓ bind</td><td class="ms">6.07 ms</td></tr>
<tr class="bk-row"><td class="bk-name"><code>pls4all.R.formula</code></td><td class="parity parity-exact">✓ bind</td><td class="ms">7.96 ms</td></tr>
<tr class="bk-row"><td class="bk-name"><code>pls4all.R.mdatools</code></td><td class="parity parity-exact">✓ bind</td><td class="ms">6.56 ms</td></tr>
<tr class="bk-row"><td class="bk-name"><code>pls4all.R.pls</code></td><td class="parity parity-exact">✓ bind</td><td class="ms">7.33 ms</td></tr>
</tbody>
<tbody class="lang-band lang-python"><tr class="lang-band-row" data-lang="python"><th colspan="3" scope="rowgroup"><span class="lang-band-dot"></span>Python · external</th></tr>
<tr class="bk-row truth-source truth-source-strict"><td class="bk-name"><span class="truth-mark" title="Registry parity reference (python): nirs4all in-tree — strict (rmse_rel ≤ 1e-08)">📐</span><code>nirs4all</code></td><td class="parity parity-ref-source">source</td><td class="ms">2.13 ms</td></tr>
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
<tr class="bk-row"><td class="bk-name"><code>pls4all.cpp.blas+omp</code></td><td class="parity parity-ref-strict">✓ ref</td><td class="ms">2.95 ms</td></tr>
</tbody>
<tbody class="lang-band lang-python"><tr class="lang-band-row" data-lang="python"><th colspan="3" scope="rowgroup"><span class="lang-band-dot"></span>Python · archived pls4all benchmark</th></tr>
<tr class="bk-row"><td class="bk-name"><code>pls4all.python</code></td><td class="parity parity-exact">✓ bind</td><td class="ms ms-best">1.74 ms<span class="medal" title="fastest">🏆</span></td></tr>
<tr class="bk-row"><td class="bk-name"><code>pls4all.sklearn</code></td><td class="parity parity-exact">✓ bind</td><td class="ms">3.37 ms</td></tr>
</tbody>
<tbody class="lang-band lang-r"><tr class="lang-band-row" data-lang="r"><th colspan="3" scope="rowgroup"><span class="lang-band-dot"></span>R · archived pls4all benchmark</th></tr>
<tr class="bk-row"><td class="bk-name"><code>pls4all.R</code></td><td class="parity parity-exact">✓ bind</td><td class="ms">10.4 ms</td></tr>
<tr class="bk-row"><td class="bk-name"><code>pls4all.R.formula</code></td><td class="parity parity-exact">✓ bind</td><td class="ms">35.7 ms</td></tr>
<tr class="bk-row"><td class="bk-name"><code>pls4all.R.mdatools</code></td><td class="parity parity-exact">✓ bind</td><td class="ms">29.5 ms</td></tr>
<tr class="bk-row"><td class="bk-name"><code>pls4all.R.pls</code></td><td class="parity parity-exact">✓ bind</td><td class="ms">32.9 ms</td></tr>
</tbody>
<tbody class="lang-band lang-python"><tr class="lang-band-row" data-lang="python"><th colspan="3" scope="rowgroup"><span class="lang-band-dot"></span>Python · external</th></tr>
<tr class="bk-row truth-source truth-source-strict"><td class="bk-name"><span class="truth-mark" title="Registry parity reference (python): nirs4all in-tree — strict (rmse_rel ≤ 1e-08)">📐</span><code>nirs4all</code></td><td class="parity parity-ref-source">source</td><td class="ms">11.5 ms</td></tr>
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
<tr class="bk-row"><td class="bk-name"><code>pls4all.cpp.blas+omp</code></td><td class="parity parity-ref-strict">✓ ref</td><td class="ms">3.32 ms</td></tr>
</tbody>
<tbody class="lang-band lang-python"><tr class="lang-band-row" data-lang="python"><th colspan="3" scope="rowgroup"><span class="lang-band-dot"></span>Python · archived pls4all benchmark</th></tr>
<tr class="bk-row"><td class="bk-name"><code>pls4all.python</code></td><td class="parity parity-exact">✓ bind</td><td class="ms">7.59 ms</td></tr>
<tr class="bk-row"><td class="bk-name"><code>pls4all.sklearn</code></td><td class="parity parity-exact">✓ bind</td><td class="ms">4.10 ms</td></tr>
</tbody>
<tbody class="lang-band lang-r"><tr class="lang-band-row" data-lang="r"><th colspan="3" scope="rowgroup"><span class="lang-band-dot"></span>R · archived pls4all benchmark</th></tr>
<tr class="bk-row"><td class="bk-name"><code>pls4all.R</code></td><td class="parity parity-exact">✓ bind</td><td class="ms">9.94 ms</td></tr>
<tr class="bk-row"><td class="bk-name"><code>pls4all.R.formula</code></td><td class="parity parity-exact">✓ bind</td><td class="ms">9.00 ms</td></tr>
<tr class="bk-row"><td class="bk-name"><code>pls4all.R.mdatools</code></td><td class="parity parity-exact">✓ bind</td><td class="ms">8.97 ms</td></tr>
<tr class="bk-row"><td class="bk-name"><code>pls4all.R.pls</code></td><td class="parity parity-exact">✓ bind</td><td class="ms">8.32 ms</td></tr>
</tbody>
<tbody class="lang-band lang-python"><tr class="lang-band-row" data-lang="python"><th colspan="3" scope="rowgroup"><span class="lang-band-dot"></span>Python · external</th></tr>
<tr class="bk-row truth-source truth-source-strict"><td class="bk-name"><span class="truth-mark" title="Registry parity reference (python): nirs4all in-tree — strict (rmse_rel ≤ 1e-08)">📐</span><code>nirs4all</code></td><td class="parity parity-ref-source">source</td><td class="ms ms-best">1.87 ms<span class="medal" title="fastest">🏆</span></td></tr>
</tbody>
</table>
</div>

:::

::::


---

_See also_: [benchmark overview](../benchmarks/overview.md) · [methods index](index.md) · [interactive dashboard](../landing/dashboard.md)