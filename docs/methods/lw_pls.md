# `lw_pls` — Locally-Weighted PLS (LW-PLS)

_Group_: **Nonlinear / local** · _Registry tolerance_: `1e-08`

## Description

LW-PLS — Locally-weighted PLS (§17 Phase 4)

> **Registry note** — In-tree `nirs4all.operators.models.sklearn.lwpls.LWPLS` is the sanctioned external reference. pls4all defaults to the Gaussian-weighted local PLS that matches nirs4all bit-for-bit (max_abs < 1e-13); the legacy k-NN cutoff variant is opt-in via cfg.solver = SIMPLS.

### Parameters

| Name | Type | Default | Notes |
|------|------|---------|-------|
| `n_components` | `int` | `3` | registry benchmark cell value |
| `n_neighbors` | `int` | `30` | registry benchmark cell value |

## Explanations

### Bibliographic source

Centner, V. & Massart, D. L. (1998). *Optimization in locally weighted regression*. Analytical Chemistry 70(19), 4206--4211. DOI [10.1021/ac980208r](https://doi.org/10.1021/ac980208r).

### Mathematical principle

Instead of fitting a single global PLS, LW-PLS refits a *per-prediction-point* local PLS using only the $k$-nearest calibration samples (in $\mathbf{X}$-space distance). This adapts the model to the local geometry around each query point and is effective on calibration sets that span heterogeneous regimes (e.g. a single instrument calibrated across several product classes).

The neighbourhood weight typically combines distance (Gaussian or tricube kernel on the Euclidean / Mahalanobis distance) with the inverse residual variance from a preliminary global fit. The local PLS uses few components (typically 2–4) because the neighbourhood is small.

Prediction cost is $O(n)$ for the neighbour search plus $O(k_{\mathrm{nn}} \cdot p \cdot k_{\mathrm{pls}})$ for the local fit, per query. KD-tree / ball-tree indices accelerate the neighbour search; pls4all uses an exhaustive scan because $p \gg n$ defeats most spatial indices for NIR data anyway.

### Appropriate uses

Local nonlinear calibration where each query is well represented by nearby calibration samples.

### Limits and validation

A separate weighted model is fitted per query; distance scaling, bandwidth, and poor neighbourhood coverage can dominate predictions.

### Implementation

`n4m_estimators_lw_pls_fit`. Reference: sanctioned git-pinned port `nirs4all.operators.models.sklearn.lwpls`.

### Sources and provenance

Current implementation: [cpp/src/core/extra_pls.cpp](https://github.com/GBeurier/nirs4all-methods/blob/main/cpp/src/core/extra_pls.cpp).


### API and bindings

**C ABI (ABI 2):** [`n4m_estimators_lw_pls_fit`](https://github.com/GBeurier/nirs4all-methods/blob/main/cpp/include/n4m/estimators/regression.h#L340). Use the linked public header for the exact signature, configuration, and result handles.

**Python:** no current AST-verified public `n4m` re-export was found for this method. The linked C ABI above is the documented surface in this checkout.

**R (source-verified):** [`lw_pls_fit(X, Y, n_components, n_neighbors = 30L)`](https://github.com/GBeurier/nirs4all-methods/blob/main/bindings/r/n4m/R/methods_extra.R).

The source signature has additional required inputs, so no example call is fabricated.

**MATLAB / Octave (source-verified):** [`lw_pls(X, Y, n_components, n_neighbors)`](https://github.com/GBeurier/nirs4all-methods/blob/main/bindings/matlab/+n4m/lw_pls.m).

The source signature has additional required inputs, so no example call is fabricated.

**Registry parity references** 📐

:::{card}
:class-card: external-refs

- 📐 **`nirs4all`** (python · python) — `nirs4all` in-tree · strict (rmse_rel ≤ 1e-08) — In-tree Python LW-PLS (sanctioned external reference). Locally-weighted PLS (Naes 1990 / Centner 1998). pls4all's default solver (NIPALS) implements the same Gaussian-weighted local PLS as the nirs4all reference, deriving the kernel bandwidth `lambda = max(1.0, 0.5 * n_neighbors)`. The legacy k-NN cutoff variant remains available via cfg.solver = SIMPLS.

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
<tr class="bk-row"><td class="bk-name"><code>pls4all.cpp.blas+omp</code></td><td class="parity parity-ref-strict">✓ ref 7e-16</td><td class="ms">25.6 ms</td></tr>
</tbody>
<tbody class="lang-band lang-python"><tr class="lang-band-row" data-lang="python"><th colspan="3" scope="rowgroup"><span class="lang-band-dot"></span>Python · archived pls4all benchmark</th></tr>
<tr class="bk-row"><td class="bk-name"><code>pls4all.python</code></td><td class="parity parity-exact">✓ bind</td><td class="ms">37.2 ms</td></tr>
<tr class="bk-row"><td class="bk-name"><code>pls4all.sklearn</code></td><td class="parity parity-cross_check">⇄ +1e+00</td><td class="ms ms-best">9.19 ms<span class="medal" title="fastest">🏆</span></td></tr>
</tbody>
<tbody class="lang-band lang-r"><tr class="lang-band-row" data-lang="r"><th colspan="3" scope="rowgroup"><span class="lang-band-dot"></span>R · archived pls4all benchmark</th></tr>
<tr class="bk-row"><td class="bk-name"><code>pls4all.R</code></td><td class="parity parity-cross_check">⇄ +1e+00</td><td class="ms">20.3 ms</td></tr>
<tr class="bk-row"><td class="bk-name"><code>pls4all.R.formula</code></td><td class="parity parity-cross_check">⇄ +1e+00</td><td class="ms">23.2 ms</td></tr>
<tr class="bk-row"><td class="bk-name"><code>pls4all.R.mdatools</code></td><td class="parity parity-cross_check">⇄ +1e+00</td><td class="ms">24.6 ms</td></tr>
<tr class="bk-row"><td class="bk-name"><code>pls4all.R.pls</code></td><td class="parity parity-cross_check">⇄ +1e+00</td><td class="ms">23.2 ms</td></tr>
</tbody>
<tbody class="lang-band lang-python"><tr class="lang-band-row" data-lang="python"><th colspan="3" scope="rowgroup"><span class="lang-band-dot"></span>Python · external</th></tr>
<tr class="bk-row truth-source truth-source-strict"><td class="bk-name"><span class="truth-mark" title="Registry parity reference (python): nirs4all in-tree — strict (rmse_rel ≤ 1e-08)">📐</span><code>nirs4all</code></td><td class="parity parity-ref-source">source</td><td class="ms">23.9 ms</td></tr>
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
<tr class="bk-row"><td class="bk-name"><code>pls4all.cpp.blas+omp</code></td><td class="parity parity-ref-strict">✓ ref 7e-16</td><td class="ms">10.8 ms</td></tr>
</tbody>
<tbody class="lang-band lang-python"><tr class="lang-band-row" data-lang="python"><th colspan="3" scope="rowgroup"><span class="lang-band-dot"></span>Python · archived pls4all benchmark</th></tr>
<tr class="bk-row"><td class="bk-name"><code>pls4all.python</code></td><td class="parity parity-exact">✓ bind</td><td class="ms">10.3 ms</td></tr>
<tr class="bk-row"><td class="bk-name"><code>pls4all.sklearn</code></td><td class="parity parity-cross_check">⇄ +1e+00</td><td class="ms ms-best">4.44 ms<span class="medal" title="fastest">🏆</span></td></tr>
</tbody>
<tbody class="lang-band lang-r"><tr class="lang-band-row" data-lang="r"><th colspan="3" scope="rowgroup"><span class="lang-band-dot"></span>R · archived pls4all benchmark</th></tr>
<tr class="bk-row"><td class="bk-name"><code>pls4all.R</code></td><td class="parity parity-cross_check">⇄ +1e+00</td><td class="ms">9.02 ms</td></tr>
<tr class="bk-row"><td class="bk-name"><code>pls4all.R.formula</code></td><td class="parity parity-cross_check">⇄ +1e+00</td><td class="ms">10.0 ms</td></tr>
<tr class="bk-row"><td class="bk-name"><code>pls4all.R.mdatools</code></td><td class="parity parity-cross_check">⇄ +1e+00</td><td class="ms">10.8 ms</td></tr>
<tr class="bk-row"><td class="bk-name"><code>pls4all.R.pls</code></td><td class="parity parity-cross_check">⇄ +1e+00</td><td class="ms">11.1 ms</td></tr>
</tbody>
<tbody class="lang-band lang-python"><tr class="lang-band-row" data-lang="python"><th colspan="3" scope="rowgroup"><span class="lang-band-dot"></span>Python · external</th></tr>
<tr class="bk-row truth-source truth-source-strict"><td class="bk-name"><span class="truth-mark" title="Registry parity reference (python): nirs4all in-tree — strict (rmse_rel ≤ 1e-08)">📐</span><code>nirs4all</code></td><td class="parity parity-ref-source">source</td><td class="ms">23.7 ms</td></tr>
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
<tr class="bk-row"><td class="bk-name"><code>pls4all.cpp.blas+omp</code></td><td class="parity parity-ref-strict">✓ ref 7e-16</td><td class="ms">11.8 ms</td></tr>
</tbody>
<tbody class="lang-band lang-python"><tr class="lang-band-row" data-lang="python"><th colspan="3" scope="rowgroup"><span class="lang-band-dot"></span>Python · archived pls4all benchmark</th></tr>
<tr class="bk-row"><td class="bk-name"><code>pls4all.python</code></td><td class="parity parity-exact">✓ bind</td><td class="ms">12.9 ms</td></tr>
<tr class="bk-row"><td class="bk-name"><code>pls4all.sklearn</code></td><td class="parity parity-cross_check">⇄ +1e+00</td><td class="ms ms-best">5.14 ms<span class="medal" title="fastest">🏆</span></td></tr>
</tbody>
<tbody class="lang-band lang-r"><tr class="lang-band-row" data-lang="r"><th colspan="3" scope="rowgroup"><span class="lang-band-dot"></span>R · archived pls4all benchmark</th></tr>
<tr class="bk-row"><td class="bk-name"><code>pls4all.R</code></td><td class="parity parity-cross_check">⇄ +1e+00</td><td class="ms">9.85 ms</td></tr>
<tr class="bk-row"><td class="bk-name"><code>pls4all.R.formula</code></td><td class="parity parity-cross_check">⇄ +1e+00</td><td class="ms">9.03 ms</td></tr>
<tr class="bk-row"><td class="bk-name"><code>pls4all.R.mdatools</code></td><td class="parity parity-cross_check">⇄ +1e+00</td><td class="ms">11.2 ms</td></tr>
<tr class="bk-row"><td class="bk-name"><code>pls4all.R.pls</code></td><td class="parity parity-cross_check">⇄ +1e+00</td><td class="ms">12.1 ms</td></tr>
</tbody>
<tbody class="lang-band lang-python"><tr class="lang-band-row" data-lang="python"><th colspan="3" scope="rowgroup"><span class="lang-band-dot"></span>Python · external</th></tr>
<tr class="bk-row truth-source truth-source-strict"><td class="bk-name"><span class="truth-mark" title="Registry parity reference (python): nirs4all in-tree — strict (rmse_rel ≤ 1e-08)">📐</span><code>nirs4all</code></td><td class="parity parity-ref-source">source</td><td class="ms">29.0 ms</td></tr>
</tbody>
</table>
</div>

:::

::::


---

_See also_: [benchmark overview](../benchmarks/overview.md) · [methods index](index.md) · [interactive dashboard](../landing/dashboard.md)