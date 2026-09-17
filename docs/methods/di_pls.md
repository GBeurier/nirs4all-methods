# `di_pls` — Domain-Invariant PLS (di-PLS)

_Group_: **Calibration transfer** · _Registry tolerance_: `1e-06`

## Description

Domain-invariant PLS

> **Registry note** — Python `diPLSlib.models.DIPLS` (B-Analytics; Nikzad-Langerodi 2018 authors). pls4all `di_pls_fit` defaults to the diPLSlib algorithm (centered NIPALS, convex-relaxation penalty, target-mean rescale) — bit-for-bit parity with `DIPLS(centering=True, rescale='Target')`. Set `cfg.di_pls_legacy = 1` to fall back to the pre-0.97.4 SIMPLS direction projection.

### Parameters

| Name | Type | Default | Notes |
|------|------|---------|-------|
| `n_components` | `int` | `4` | registry benchmark cell value |
| `di_lambda` | `float` | `1.0` | registry benchmark cell value |

## Explanations

### Bibliographic source

Nikzad-Langerodi, R., Zellinger, W., Saminger-Platz, S. & Moser, B. A. (2018). *Domain-invariant partial-least-squares regression*. Analytical Chemistry 90(11), 6693–6701. Verified primary link: [https://doi.org/10.1021/acs.analchem.8b00498](https://doi.org/10.1021/acs.analchem.8b00498).

### Mathematical principle

Calibration transfer methods reconcile spectra acquired on different instruments or under different environmental conditions. di-PLS does this by augmenting the PLS objective with a domain-discrepancy penalty: $\mathcal{L}(\mathbf{w}) = -\operatorname{Cov}(\mathbf{X}_s\mathbf{w}, \mathbf{y}_s)^2 + \lambda \,\mathrm{MMD}^2(\mathbf{X}_s\mathbf{w}, \mathbf{X}_t\mathbf{w})$, where $(\mathbf{X}_s, \mathbf{y}_s)$ is a labelled source domain, $\mathbf{X}_t$ is an unlabelled target domain and MMD is the maximum mean discrepancy.

Minimising $\mathcal{L}$ produces latent directions $\mathbf{w}$ that simultaneously **predict $y$ in the source** and have **matched distributions across domains**. The model is therefore robust to drift between calibration and prediction sets without requiring labels on the target domain.

Computational cost is dominated by the MMD term, which is $O((n_s + n_t)^2)$ in a naive implementation; pls4all uses a linear-kernel MMD which reduces this to $O((n_s + n_t) p)$.

$\lambda$ controls the bias–transferability trade-off: $\lambda = 0$ recovers vanilla PLS on the source, large $\lambda$ shrinks toward a domain-aligned but potentially under-predictive model.

### Appropriate uses

Adapting a linear calibration across labelled source and target domains with different predictor distributions.

### Limits and validation

Domain-alignment penalties can remove predictive variation; target information and tuning must remain inside validation folds.

### Implementation

`n4m_domain_adaptation_di_pls_fit` — requires `X_target` at fit time. Reference: Python `diPLSlib.models.DIPLS` (Nikzad-Langerodi authors). The pls4all variant matches diPLSlib's `rescale='Target'` source-centred default.

### Sources and provenance

Current implementation: [cpp/src/core/extra_pls.cpp](https://github.com/GBeurier/nirs4all-methods/blob/main/cpp/src/core/extra_pls.cpp).


### API and bindings

**C ABI (ABI 2):** [`n4m_domain_adaptation_di_pls_fit`](https://github.com/GBeurier/nirs4all-methods/blob/main/cpp/include/n4m/domain_adaptation.h#L130). Use the linked public header for the exact signature, configuration, and result handles.

**Python:** no current AST-verified public `n4m` re-export was found for this method. The linked C ABI above is the documented surface in this checkout.

**R (source-verified):** [`di_pls(formula, data, ncomp = 2L, X_target, di_lambda = 1.0, na.action = stats::na.omit)`](https://github.com/GBeurier/nirs4all-methods/blob/main/bindings/r/n4m/R/sklearn_methods.R).

The source signature has additional required inputs, so no example call is fabricated.

**MATLAB / Octave (source-verified):** [`di_pls(X_source, Y_source, n_components, X_target, di_lambda)`](https://github.com/GBeurier/nirs4all-methods/blob/main/bindings/matlab/+n4m/di_pls.m).

The source signature has additional required inputs, so no example call is fabricated.

**Registry parity references** 📐

:::{card}
:class-card: external-refs

- 📐 **`ref.python_diplslib`** (python · python) — `diPLSlib` 2.5.0 · strict (rmse_rel ≤ 1e-06) — Python `diPLSlib.models.DIPLS` (B-Analytics; Nikzad-Langerodi 2018 authors). Same di-PLS penalty applied during deflation; centering / target rescaling differ slightly, so tolerance is widened.

:::

### Benchmarks

**Archived measurement identity.** Backend labels in this table are the raw IDs recorded when the benchmark ran (including historical `pls4all.*` IDs). They preserve measurement provenance and do not describe a current public Python, R, or MATLAB binding; use the source-verified **API and bindings** section above for current entry points.

Adaptive wall-clock per cell measured against [`full_matrix.csv`](../benchmarks/overview.md). Only backends that implement this method are listed; libraries without the method are omitted.

**Verdict** &nbsp;·&nbsp; ✓ ref / ≈ ref / ~ shape mark a reference-gate pass at strict / relaxed / qualitative tolerance &nbsp;·&nbsp; ✓ bind = archived binding-harness result agrees with the C++ baseline &nbsp;·&nbsp; ⇄ cross-check = documented by-design selector/RNG/model, noncanonical API/facade convention, or secondary oracle &nbsp;·&nbsp; ✗ divergent &nbsp;·&nbsp; ⚠ error &nbsp;·&nbsp; — not run. The fastest backend per column is marked 🏆.

**Reference gate**: strict — numeric equivalence (`rmse_rel_tol ≤ 1e-06`).

Rows tagged with **📐** are the canonical parity references for this method (declared in [`parity_timing.registry`](../benchmarks/methodology.md)). C++ and external rows show reference parity; archived language-harness rows show binding parity against the C++ backend. Hover the icon for role and tolerance band.

::::{tab-set}
:class: parity-tabs

:::{tab-item} 1 thread
:sync: threads-1

<div class="parity-table-wrap">
<table class="docutils parity-grouped">
<thead><tr><th scope="col">Backend</th><th scope="col">Parity</th><th class="size-col" scope="col">200×50 (ms)</th></tr></thead>
<tbody class="lang-band lang-cpp"><tr class="lang-band-row" data-lang="cpp"><th colspan="3" scope="rowgroup"><span class="lang-band-dot"></span>C++ native · libn4m</th></tr>
<tr class="bk-row"><td class="bk-name"><code>pls4all.cpp.blas+omp</code></td><td class="parity parity-ref-strict">✓ ref 8e-15</td><td class="ms ms-best">2.58 ms<span class="medal" title="fastest">🏆</span></td></tr>
</tbody>
<tbody class="lang-band lang-python"><tr class="lang-band-row" data-lang="python"><th colspan="3" scope="rowgroup"><span class="lang-band-dot"></span>Python · archived pls4all benchmark</th></tr>
<tr class="bk-row"><td class="bk-name"><code>pls4all.python</code></td><td class="parity parity-exact">✓ bind</td><td class="ms">2.74 ms</td></tr>
<tr class="bk-row"><td class="bk-name"><code>pls4all.sklearn</code></td><td class="parity parity-exact">✓ 4e-15</td><td class="ms">2.82 ms</td></tr>
</tbody>
<tbody class="lang-band lang-r"><tr class="lang-band-row" data-lang="r"><th colspan="3" scope="rowgroup"><span class="lang-band-dot"></span>R · archived pls4all benchmark</th></tr>
<tr class="bk-row"><td class="bk-name"><code>pls4all.R</code></td><td class="parity parity-exact">✓ 6e-15</td><td class="ms">9.69 ms</td></tr>
<tr class="bk-row"><td class="bk-name"><code>pls4all.R.formula</code></td><td class="parity parity-exact">✓ 6e-15</td><td class="ms">12.4 ms</td></tr>
<tr class="bk-row"><td class="bk-name"><code>pls4all.R.mdatools</code></td><td class="parity parity-exact">✓ 6e-15</td><td class="ms">12.8 ms</td></tr>
<tr class="bk-row"><td class="bk-name"><code>pls4all.R.pls</code></td><td class="parity parity-exact">✓ 6e-15</td><td class="ms">12.4 ms</td></tr>
</tbody>
<tbody class="lang-band lang-python"><tr class="lang-band-row" data-lang="python"><th colspan="3" scope="rowgroup"><span class="lang-band-dot"></span>Python · external</th></tr>
<tr class="bk-row truth-source truth-source-strict"><td class="bk-name"><span class="truth-mark" title="Registry parity reference (python): diPLSlib 2.5.0 — strict (rmse_rel ≤ 1e-06)">📐</span><code>ref.python_diplslib</code></td><td class="parity parity-ref-source">source</td><td class="ms">3.78 ms</td></tr>
</tbody>
</table>
</div>

:::

:::{tab-item} 3 threads
:sync: threads-3

<div class="parity-table-wrap">
<table class="docutils parity-grouped">
<thead><tr><th scope="col">Backend</th><th scope="col">Parity</th><th class="size-col" scope="col">200×50 (ms)</th></tr></thead>
<tbody class="lang-band lang-cpp"><tr class="lang-band-row" data-lang="cpp"><th colspan="3" scope="rowgroup"><span class="lang-band-dot"></span>C++ native · libn4m</th></tr>
<tr class="bk-row"><td class="bk-name"><code>pls4all.cpp.blas+omp</code></td><td class="parity parity-ref-strict">✓ ref 8e-15</td><td class="ms">2.80 ms</td></tr>
</tbody>
<tbody class="lang-band lang-python"><tr class="lang-band-row" data-lang="python"><th colspan="3" scope="rowgroup"><span class="lang-band-dot"></span>Python · archived pls4all benchmark</th></tr>
<tr class="bk-row"><td class="bk-name"><code>pls4all.python</code></td><td class="parity parity-exact">✓ bind</td><td class="ms ms-best">2.70 ms<span class="medal" title="fastest">🏆</span></td></tr>
<tr class="bk-row"><td class="bk-name"><code>pls4all.sklearn</code></td><td class="parity parity-exact">✓ 4e-15</td><td class="ms">2.98 ms</td></tr>
</tbody>
<tbody class="lang-band lang-r"><tr class="lang-band-row" data-lang="r"><th colspan="3" scope="rowgroup"><span class="lang-band-dot"></span>R · archived pls4all benchmark</th></tr>
<tr class="bk-row"><td class="bk-name"><code>pls4all.R</code></td><td class="parity parity-exact">✓ 6e-15</td><td class="ms">9.91 ms</td></tr>
<tr class="bk-row"><td class="bk-name"><code>pls4all.R.formula</code></td><td class="parity parity-exact">✓ 6e-15</td><td class="ms">12.2 ms</td></tr>
<tr class="bk-row"><td class="bk-name"><code>pls4all.R.mdatools</code></td><td class="parity parity-exact">✓ 6e-15</td><td class="ms">12.9 ms</td></tr>
<tr class="bk-row"><td class="bk-name"><code>pls4all.R.pls</code></td><td class="parity parity-exact">✓ 6e-15</td><td class="ms">11.2 ms</td></tr>
</tbody>
<tbody class="lang-band lang-python"><tr class="lang-band-row" data-lang="python"><th colspan="3" scope="rowgroup"><span class="lang-band-dot"></span>Python · external</th></tr>
<tr class="bk-row truth-source truth-source-strict"><td class="bk-name"><span class="truth-mark" title="Registry parity reference (python): diPLSlib 2.5.0 — strict (rmse_rel ≤ 1e-06)">📐</span><code>ref.python_diplslib</code></td><td class="parity parity-ref-source">source</td><td class="ms">3.79 ms</td></tr>
</tbody>
</table>
</div>

:::

:::{tab-item} 10 threads
:sync: threads-10

<div class="parity-table-wrap">
<table class="docutils parity-grouped">
<thead><tr><th scope="col">Backend</th><th scope="col">Parity</th><th class="size-col" scope="col">200×50 (ms)</th></tr></thead>
<tbody class="lang-band lang-cpp"><tr class="lang-band-row" data-lang="cpp"><th colspan="3" scope="rowgroup"><span class="lang-band-dot"></span>C++ native · libn4m</th></tr>
<tr class="bk-row"><td class="bk-name"><code>pls4all.cpp.blas+omp</code></td><td class="parity parity-ref-strict">✓ ref 8e-15</td><td class="ms ms-best">2.74 ms<span class="medal" title="fastest">🏆</span></td></tr>
</tbody>
<tbody class="lang-band lang-python"><tr class="lang-band-row" data-lang="python"><th colspan="3" scope="rowgroup"><span class="lang-band-dot"></span>Python · archived pls4all benchmark</th></tr>
<tr class="bk-row"><td class="bk-name"><code>pls4all.python</code></td><td class="parity parity-exact">✓ bind</td><td class="ms">2.82 ms</td></tr>
<tr class="bk-row"><td class="bk-name"><code>pls4all.sklearn</code></td><td class="parity parity-exact">✓ 4e-15</td><td class="ms">2.95 ms</td></tr>
</tbody>
<tbody class="lang-band lang-r"><tr class="lang-band-row" data-lang="r"><th colspan="3" scope="rowgroup"><span class="lang-band-dot"></span>R · archived pls4all benchmark</th></tr>
<tr class="bk-row"><td class="bk-name"><code>pls4all.R</code></td><td class="parity parity-exact">✓ 6e-15</td><td class="ms">11.1 ms</td></tr>
<tr class="bk-row"><td class="bk-name"><code>pls4all.R.formula</code></td><td class="parity parity-exact">✓ 6e-15</td><td class="ms">14.0 ms</td></tr>
<tr class="bk-row"><td class="bk-name"><code>pls4all.R.mdatools</code></td><td class="parity parity-exact">✓ 6e-15</td><td class="ms">13.0 ms</td></tr>
<tr class="bk-row"><td class="bk-name"><code>pls4all.R.pls</code></td><td class="parity parity-exact">✓ 6e-15</td><td class="ms">12.4 ms</td></tr>
</tbody>
<tbody class="lang-band lang-python"><tr class="lang-band-row" data-lang="python"><th colspan="3" scope="rowgroup"><span class="lang-band-dot"></span>Python · external</th></tr>
<tr class="bk-row truth-source truth-source-strict"><td class="bk-name"><span class="truth-mark" title="Registry parity reference (python): diPLSlib 2.5.0 — strict (rmse_rel ≤ 1e-06)">📐</span><code>ref.python_diplslib</code></td><td class="parity parity-ref-source">source</td><td class="ms">3.92 ms</td></tr>
</tbody>
</table>
</div>

:::

::::


---

_See also_: [benchmark overview](../benchmarks/overview.md) · [methods index](index.md) · [interactive dashboard](../landing/dashboard.md)