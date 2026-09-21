# `o2pls` — O2-PLS (two-way orthogonal)

_Group_: **Multi-block / cross-modal** · _Registry tolerance_: `1e-10`

## Description

O2PLS — bi-directional OPLS (Trygg & Wold 2003)

> **Registry note** — R `OmicsPLS::o2m` 2.1.0 (Bouhaddani 2018) joint-SVD O2PLS. pls4all defaults to the OmicsPLS::o2m algorithm and matches R bit-for-bit (~1e-13 max_abs). The pre-0.97 peel-then-PLS path (Trygg & Wold 2003 §3.2 power-iteration recipe) is reachable via the `legacy=True` adapter kwarg / `cfg.solver = NIPALS`.

### Parameters

| Name | Type | Default | Notes |
|------|------|---------|-------|
| `n_targets` | `int` | `4` | registry benchmark cell value |
| `n_predictive` | `int` | `2` | registry benchmark cell value |
| `n_x_orthogonal` | `int` | `1` | registry benchmark cell value |
| `n_y_orthogonal` | `int` | `1` | registry benchmark cell value |

## Explanations

### Bibliographic source

Trygg, J. & Wold, S. (2003). *O2-PLS, a two-block (X–Y) latent variable regression method with an integral OSC filter*. Journal of Chemometrics 17(1), 53–64. Verified primary link: [https://doi.org/10.1002/cem.775](https://doi.org/10.1002/cem.775).

### Mathematical principle

O2-PLS extends OPLS symmetrically to both $\mathbf{X}$ and $\mathbf{Y}$: it decomposes each block into a joint predictive component plus block-orthogonal components. Unlike OPLS, which is asymmetric ($\mathbf{Y}$ drives the decomposition of $\mathbf{X}$), O2-PLS treats both matrices as observation blocks of equal status.

Required hyperparameters: $n_{\mathrm{pred}}$ (joint components), $n_{\mathrm{X,ortho}}$ (X-unique orthogonal), $n_{\mathrm{Y,ortho}}$ (Y-unique orthogonal). Choosing all three by cross-validation is a 3-D grid which can be expensive; a common compromise fixes the orthogonal counts at 1 and tunes only $n_{\mathrm{pred}}$.

O2-PLS is dominant in metabolomics ↔ transcriptomics integration where the analyst wants to disentangle platform-specific orthogonal variation from biology that is consistent across platforms.

### Appropriate uses

Two-block integration that separates joint predictive covariance from block-specific structured variation.

### Limits and validation

Joint and orthogonal ranks are not identified automatically and may be unstable in small samples.

### Implementation

`n4m_estimators_o2pls_fit`. Reference: CRAN `OmicsPLS 2.1.0`.

### Sources and provenance

Current implementation: [cpp/src/core/extra_pls.cpp](https://github.com/GBeurier/nirs4all-methods/blob/main/cpp/src/core/extra_pls.cpp).


### API and bindings

**C ABI (ABI 2):** [`n4m_estimators_o2pls_fit`](https://github.com/GBeurier/nirs4all-methods/blob/main/cpp/include/n4m/estimators/multiblock.h#L24). Use the linked public header for the exact signature, configuration, and result handles.

**Python:** no current AST-verified public `n4m` re-export was found for this method. The linked C ABI above is the documented surface in this checkout.

**R (source-verified):** [`o2pls(formula, data, n_predictive = 2L, n_x_orthogonal = 1L, n_y_orthogonal = 1L, na.action = stats::na.omit)`](https://github.com/GBeurier/nirs4all-methods/blob/main/bindings/r/n4m/R/sklearn_extra.R).

The source signature has additional required inputs, so no example call is fabricated.

**MATLAB / Octave (source-verified):** [`o2pls(X, Y, n_predictive, n_x_orthogonal, n_y_orthogonal)`](https://github.com/GBeurier/nirs4all-methods/blob/main/bindings/matlab/+n4m/o2pls.m).

The source signature has additional required inputs, so no example call is fabricated.

**Registry parity references** 📐

:::{card}
:class-card: external-refs

- 📐 **`ref.r_omicspls`** (R · r) — `OmicsPLS` 2.1.0 · strict (rmse_rel ≤ 1e-10) — R `OmicsPLS::o2m` (Bouhaddani 2018), joint-SVD O2PLS. pls4all's default O2PLS path now matches this algorithm bit-for-bit (max_abs ~1e-13 on the parity sizes); the legacy peel-then-PLS implementation is opt-in.

:::

### Benchmarks

**Archived measurement identity.** Backend labels in this table are the raw IDs recorded when the benchmark ran (including historical `pls4all.*` IDs). They preserve measurement provenance and do not describe a current public Python, R, or MATLAB binding; use the source-verified **API and bindings** section above for current entry points.

Adaptive wall-clock per cell measured against [`full_matrix.csv`](../benchmarks/overview.md). Only backends that implement this method are listed; libraries without the method are omitted.

**Verdict** &nbsp;·&nbsp; ✓ ref / ≈ ref / ~ shape mark a reference-gate pass at strict / relaxed / qualitative tolerance &nbsp;·&nbsp; ✓ bind = archived binding-harness result agrees with the C++ baseline &nbsp;·&nbsp; ⇄ cross-check = documented by-design selector/RNG/model, noncanonical API/facade convention, or secondary oracle &nbsp;·&nbsp; ✗ divergent &nbsp;·&nbsp; ⚠ error &nbsp;·&nbsp; — not run. The fastest backend per column is marked 🏆.

**Reference gate**: strict — numeric equivalence (`rmse_rel_tol ≤ 1e-10`).

Rows tagged with **📐** are the canonical parity references for this method (declared in [`parity_timing.registry`](../benchmarks/methodology.md)). C++ and external rows show reference parity; archived language-harness rows show binding parity against the C++ backend. Hover the icon for role and tolerance band.

::::{tab-set}
:class: parity-tabs

:::{tab-item} 1 thread
:sync: threads-1

<div class="parity-table-wrap">
<table class="docutils parity-grouped">
<thead><tr><th scope="col">Backend</th><th scope="col">Parity</th><th class="size-col" scope="col">200×30 (ms)</th></tr></thead>
<tbody class="lang-band lang-cpp"><tr class="lang-band-row" data-lang="cpp"><th colspan="3" scope="rowgroup"><span class="lang-band-dot"></span>C++ native · libn4m</th></tr>
<tr class="bk-row"><td class="bk-name"><code>pls4all.cpp.blas+omp</code></td><td class="parity parity-ref-strict">✓ ref 4e-10</td><td class="ms">1.21 ms</td></tr>
</tbody>
<tbody class="lang-band lang-python"><tr class="lang-band-row" data-lang="python"><th colspan="3" scope="rowgroup"><span class="lang-band-dot"></span>Python · archived pls4all benchmark</th></tr>
<tr class="bk-row"><td class="bk-name"><code>pls4all.python</code></td><td class="parity parity-exact">✓ bind</td><td class="ms ms-best">1.19 ms<span class="medal" title="fastest">🏆</span></td></tr>
<tr class="bk-row"><td class="bk-name"><code>pls4all.sklearn</code></td><td class="parity parity-exact">✓ 9e-16</td><td class="ms">1.37 ms</td></tr>
</tbody>
<tbody class="lang-band lang-r"><tr class="lang-band-row" data-lang="r"><th colspan="3" scope="rowgroup"><span class="lang-band-dot"></span>R · archived pls4all benchmark</th></tr>
<tr class="bk-row"><td class="bk-name"><code>pls4all.R</code></td><td class="parity parity-exact">✓ bind</td><td class="ms">2.69 ms</td></tr>
<tr class="bk-row"><td class="bk-name"><code>pls4all.R.formula</code></td><td class="parity parity-exact">✓ bind</td><td class="ms">3.44 ms</td></tr>
<tr class="bk-row"><td class="bk-name"><code>pls4all.R.mdatools</code></td><td class="parity parity-exact">✓ bind</td><td class="ms">3.42 ms</td></tr>
<tr class="bk-row"><td class="bk-name"><code>pls4all.R.pls</code></td><td class="parity parity-exact">✓ bind</td><td class="ms">3.58 ms</td></tr>
</tbody>
<tbody class="lang-band lang-r"><tr class="lang-band-row" data-lang="r"><th colspan="3" scope="rowgroup"><span class="lang-band-dot"></span>R · external</th></tr>
<tr class="bk-row truth-source truth-source-strict"><td class="bk-name"><span class="truth-mark" title="Registry parity reference (r): OmicsPLS 2.1.0 — strict (rmse_rel ≤ 1e-10)">📐</span><code>ref.r_omicspls</code></td><td class="parity parity-ref-source">source</td><td class="ms">7.21 ms</td></tr>
</tbody>
</table>
</div>

:::

:::{tab-item} 3 threads
:sync: threads-3

<div class="parity-table-wrap">
<table class="docutils parity-grouped">
<thead><tr><th scope="col">Backend</th><th scope="col">Parity</th><th class="size-col" scope="col">200×30 (ms)</th></tr></thead>
<tbody class="lang-band lang-cpp"><tr class="lang-band-row" data-lang="cpp"><th colspan="3" scope="rowgroup"><span class="lang-band-dot"></span>C++ native · libn4m</th></tr>
<tr class="bk-row"><td class="bk-name"><code>pls4all.cpp.blas+omp</code></td><td class="parity parity-ref-strict">✓ ref 4e-10</td><td class="ms ms-best">1.22 ms<span class="medal" title="fastest">🏆</span></td></tr>
</tbody>
<tbody class="lang-band lang-python"><tr class="lang-band-row" data-lang="python"><th colspan="3" scope="rowgroup"><span class="lang-band-dot"></span>Python · archived pls4all benchmark</th></tr>
<tr class="bk-row"><td class="bk-name"><code>pls4all.python</code></td><td class="parity parity-exact">✓ bind</td><td class="ms">1.24 ms</td></tr>
<tr class="bk-row"><td class="bk-name"><code>pls4all.sklearn</code></td><td class="parity parity-exact">✓ 9e-16</td><td class="ms">1.44 ms</td></tr>
</tbody>
<tbody class="lang-band lang-r"><tr class="lang-band-row" data-lang="r"><th colspan="3" scope="rowgroup"><span class="lang-band-dot"></span>R · archived pls4all benchmark</th></tr>
<tr class="bk-row"><td class="bk-name"><code>pls4all.R</code></td><td class="parity parity-exact">✓ bind</td><td class="ms">2.78 ms</td></tr>
<tr class="bk-row"><td class="bk-name"><code>pls4all.R.formula</code></td><td class="parity parity-exact">✓ bind</td><td class="ms">3.52 ms</td></tr>
<tr class="bk-row"><td class="bk-name"><code>pls4all.R.mdatools</code></td><td class="parity parity-exact">✓ bind</td><td class="ms">3.53 ms</td></tr>
<tr class="bk-row"><td class="bk-name"><code>pls4all.R.pls</code></td><td class="parity parity-exact">✓ bind</td><td class="ms">3.68 ms</td></tr>
</tbody>
<tbody class="lang-band lang-r"><tr class="lang-band-row" data-lang="r"><th colspan="3" scope="rowgroup"><span class="lang-band-dot"></span>R · external</th></tr>
<tr class="bk-row truth-source truth-source-strict"><td class="bk-name"><span class="truth-mark" title="Registry parity reference (r): OmicsPLS 2.1.0 — strict (rmse_rel ≤ 1e-10)">📐</span><code>ref.r_omicspls</code></td><td class="parity parity-ref-source">source</td><td class="ms">7.41 ms</td></tr>
</tbody>
</table>
</div>

:::

:::{tab-item} 10 threads
:sync: threads-10

<div class="parity-table-wrap">
<table class="docutils parity-grouped">
<thead><tr><th scope="col">Backend</th><th scope="col">Parity</th><th class="size-col" scope="col">200×30 (ms)</th></tr></thead>
<tbody class="lang-band lang-cpp"><tr class="lang-band-row" data-lang="cpp"><th colspan="3" scope="rowgroup"><span class="lang-band-dot"></span>C++ native · libn4m</th></tr>
<tr class="bk-row"><td class="bk-name"><code>pls4all.cpp.blas+omp</code></td><td class="parity parity-ref-strict">✓ ref 4e-10</td><td class="ms ms-best">1.24 ms<span class="medal" title="fastest">🏆</span></td></tr>
</tbody>
<tbody class="lang-band lang-python"><tr class="lang-band-row" data-lang="python"><th colspan="3" scope="rowgroup"><span class="lang-band-dot"></span>Python · archived pls4all benchmark</th></tr>
<tr class="bk-row"><td class="bk-name"><code>pls4all.python</code></td><td class="parity parity-exact">✓ bind</td><td class="ms">2.09 ms</td></tr>
<tr class="bk-row"><td class="bk-name"><code>pls4all.sklearn</code></td><td class="parity parity-exact">✓ 9e-16</td><td class="ms">1.45 ms</td></tr>
</tbody>
<tbody class="lang-band lang-r"><tr class="lang-band-row" data-lang="r"><th colspan="3" scope="rowgroup"><span class="lang-band-dot"></span>R · archived pls4all benchmark</th></tr>
<tr class="bk-row"><td class="bk-name"><code>pls4all.R</code></td><td class="parity parity-exact">✓ bind</td><td class="ms">3.09 ms</td></tr>
<tr class="bk-row"><td class="bk-name"><code>pls4all.R.formula</code></td><td class="parity parity-exact">✓ bind</td><td class="ms">3.52 ms</td></tr>
<tr class="bk-row"><td class="bk-name"><code>pls4all.R.mdatools</code></td><td class="parity parity-exact">✓ bind</td><td class="ms">3.49 ms</td></tr>
<tr class="bk-row"><td class="bk-name"><code>pls4all.R.pls</code></td><td class="parity parity-exact">✓ bind</td><td class="ms">3.25 ms</td></tr>
</tbody>
<tbody class="lang-band lang-r"><tr class="lang-band-row" data-lang="r"><th colspan="3" scope="rowgroup"><span class="lang-band-dot"></span>R · external</th></tr>
<tr class="bk-row truth-source truth-source-strict"><td class="bk-name"><span class="truth-mark" title="Registry parity reference (r): OmicsPLS 2.1.0 — strict (rmse_rel ≤ 1e-10)">📐</span><code>ref.r_omicspls</code></td><td class="parity parity-ref-source">source</td><td class="ms">7.31 ms</td></tr>
</tbody>
</table>
</div>

:::

::::


---

_See also_: [benchmark overview](../benchmarks/overview.md) · [methods index](index.md) · [interactive dashboard](../landing/dashboard.md)