# `pls_lda` — PLS-LDA

_Group_: **Classification & GLM** · _Registry tolerance_: `1e-08`

## Description

PLS-LDA — LDA on PLS scores (§17 Phase 4)

> **Registry note** — sklearn `PLSRegression(scale=False) -> LinearDiscriminantAnalysis` is the canonical reference. The in-kernel pooled-covariance LDA head reproduces sklearn's multiclass `decision_function` at ~1e-15. `legacy=True` opts into the historical SIMPLS+scaled variant (not parity-equivalent).

### Parameters

| Name | Type | Default | Notes |
|------|------|---------|-------|
| `n_components` | `int` | `3` | registry benchmark cell value |
| `n_classes` | `int` | `3` | registry benchmark cell value |

## Explanations

### Bibliographic source

Barker, M. & Rayens, W. (2003). *Partial least squares for discrimination*. Journal of Chemometrics 17(3), 166--173. DOI [10.1002/cem.785](https://doi.org/10.1002/cem.785).

### Mathematical principle

PLS-LDA is a two-stage classifier: first project $\mathbf{X}$ into the PLS latent space using one-hot encoded class labels as $\mathbf{Y}$, then fit Linear Discriminant Analysis on the resulting scores $\mathbf{T} = \mathbf{X}\mathbf{W}$.

LDA in the latent space is well-conditioned (the score matrix has $k \ll p$ columns by construction), and the PLS projection has already aligned the latent axes with the class separation direction. This is more robust than applying LDA directly to high-dimensional $\mathbf{X}$, where the within-class covariance is singular.

The decision boundary is **linear in the latent space** (and therefore also linear in the original feature space via $\mathbf{W}$). For non-linear class boundaries use PLS-QDA or PLS-logistic.

### Appropriate uses

Multiclass classification using supervised PLS scores followed by a shared-covariance Gaussian discriminant.

### Limits and validation

LDA covariance assumptions apply in score space; score extraction and class imbalance must be handled inside validation.

### Implementation

`n4m_estimators_pls_lda_fit`. The reference is composite (sklearn `PLSRegression` + sklearn `LinearDiscriminantAnalysis`); no library exposes a single PLS-LDA call.

### Sources and provenance

Current implementation: [cpp/src/core/pls_lda.cpp](https://github.com/GBeurier/nirs4all-methods/blob/main/cpp/src/core/pls_lda.cpp).


### API and bindings

**C ABI (ABI 2):** [`n4m_estimators_pls_lda_fit`](https://github.com/GBeurier/nirs4all-methods/blob/main/cpp/include/n4m/estimators/classification.h#L54). Use the linked public header for the exact signature, configuration, and result handles.

**Python:** no current AST-verified public `n4m` re-export was found for this method. The linked C ABI above is the documented surface in this checkout.

**R (source-verified):** [`pls_lda_fit(X, y_labels, n_components, n_classes = NULL)`](https://github.com/GBeurier/nirs4all-methods/blob/main/bindings/r/n4m/R/methods_extra.R).

The source signature has additional required inputs, so no example call is fabricated.

**MATLAB / Octave (source-verified):** [`pls_lda(X, y_labels, n_components, n_classes)`](https://github.com/GBeurier/nirs4all-methods/blob/main/bindings/matlab/+n4m/pls_lda.m).

The source signature has additional required inputs, so no example call is fabricated.

**Registry parity references** 📐

:::{card}
:class-card: external-refs

- 📐 **`ref.python_scikit_learn`** (python · python) — `scikit-learn` 1.8.0 · strict (rmse_rel ≤ 1e-08) — sklearn `PLSRegression -> LinearDiscriminantAnalysis` pipeline. pls4all's PLS-LDA uses a single SIMPLS pass with an internal LDA head; sklearn fits PLS on dummy-encoded targets and feeds the scores into LDA — both are LDA on PLS scores but the latent bases diverge. We compare class boundaries via one-hot decision scores.

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
<tr class="bk-row"><td class="bk-name"><code>pls4all.cpp.blas+omp</code></td><td class="parity parity-ref-strict">✓ ref 3e-16</td><td class="ms ms-best">3.14 ms<span class="medal" title="fastest">🏆</span></td></tr>
</tbody>
<tbody class="lang-band lang-python"><tr class="lang-band-row" data-lang="python"><th colspan="3" scope="rowgroup"><span class="lang-band-dot"></span>Python · archived pls4all benchmark</th></tr>
<tr class="bk-row"><td class="bk-name"><code>pls4all.python</code></td><td class="parity parity-exact">✓ bind</td><td class="ms">7.48 ms</td></tr>
</tbody>
<tbody class="lang-band lang-python"><tr class="lang-band-row" data-lang="python"><th colspan="3" scope="rowgroup"><span class="lang-band-dot"></span>Python · external</th></tr>
<tr class="bk-row truth-source truth-source-strict"><td class="bk-name"><span class="truth-mark" title="Registry parity reference (python): scikit-learn 1.8.0 — strict (rmse_rel ≤ 1e-08)">📐</span><code>ref.python_scikit_learn</code></td><td class="parity parity-ref-source">source</td><td class="ms">7.32 ms</td></tr>
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
<tr class="bk-row"><td class="bk-name"><code>pls4all.cpp.blas+omp</code></td><td class="parity parity-ref-strict">✓ ref 3e-16</td><td class="ms">6.86 ms</td></tr>
</tbody>
<tbody class="lang-band lang-python"><tr class="lang-band-row" data-lang="python"><th colspan="3" scope="rowgroup"><span class="lang-band-dot"></span>Python · archived pls4all benchmark</th></tr>
<tr class="bk-row"><td class="bk-name"><code>pls4all.python</code></td><td class="parity parity-exact">✓ bind</td><td class="ms ms-best">4.05 ms<span class="medal" title="fastest">🏆</span></td></tr>
</tbody>
<tbody class="lang-band lang-python"><tr class="lang-band-row" data-lang="python"><th colspan="3" scope="rowgroup"><span class="lang-band-dot"></span>Python · external</th></tr>
<tr class="bk-row truth-source truth-source-strict"><td class="bk-name"><span class="truth-mark" title="Registry parity reference (python): scikit-learn 1.8.0 — strict (rmse_rel ≤ 1e-08)">📐</span><code>ref.python_scikit_learn</code></td><td class="parity parity-ref-source">source</td><td class="ms">4.54 ms</td></tr>
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
<tr class="bk-row"><td class="bk-name"><code>pls4all.cpp.blas+omp</code></td><td class="parity parity-ref-strict">✓ ref 3e-16</td><td class="ms ms-best">1.95 ms<span class="medal" title="fastest">🏆</span></td></tr>
</tbody>
<tbody class="lang-band lang-python"><tr class="lang-band-row" data-lang="python"><th colspan="3" scope="rowgroup"><span class="lang-band-dot"></span>Python · archived pls4all benchmark</th></tr>
<tr class="bk-row"><td class="bk-name"><code>pls4all.python</code></td><td class="parity parity-exact">✓ bind</td><td class="ms">2.67 ms</td></tr>
</tbody>
<tbody class="lang-band lang-python"><tr class="lang-band-row" data-lang="python"><th colspan="3" scope="rowgroup"><span class="lang-band-dot"></span>Python · external</th></tr>
<tr class="bk-row truth-source truth-source-strict"><td class="bk-name"><span class="truth-mark" title="Registry parity reference (python): scikit-learn 1.8.0 — strict (rmse_rel ≤ 1e-08)">📐</span><code>ref.python_scikit_learn</code></td><td class="parity parity-ref-source">source</td><td class="ms">3.59 ms</td></tr>
</tbody>
</table>
</div>

:::

::::


---

_See also_: [benchmark overview](../benchmarks/overview.md) · [methods index](index.md) · [interactive dashboard](../landing/dashboard.md)