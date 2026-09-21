# `n_pls` — N-way PLS (Trilinear PLS, Bro 1996)

_Group_: **Multi-block / cross-modal** · _Registry tolerance_: `1e-06`

## Description

N-PLS — 3-way tensor PLS (PARAFAC + OLS by default; Bro 1996 opt-in)

> **Registry note** — Python `tensorly.parafac` + OLS reference (rank = n_components, init='random', random_state=0). pls4all default now matches this convention bit-for-bit. The Bro 1996 multilinear PLS C++ kernel is still available as an opt-in via `legacy=True`.

### Parameters

| Name | Type | Default | Notes |
|------|------|---------|-------|
| `n_components` | `int` | `4` | registry benchmark cell value |
| `mode_j` | `int` | `8` | registry benchmark cell value |
| `mode_k` | `int` | `6` | registry benchmark cell value |

## Explanations

### Bibliographic source

Bro, R. (1996). *Multiway calibration. Multilinear PLS*. Journal of Chemometrics 10(1), 47–61. Verified primary link: [https://doi.org/10.1002/(SICI)1099-128X(199601)10:1%3C47::AID-CEM400%3E3.0.CO;2-C](https://doi.org/10.1002/(SICI)1099-128X(199601)10:1%3C47::AID-CEM400%3E3.0.CO;2-C).

### Mathematical principle

When the predictor is naturally a tensor $\mathcal{X} \in \mathbb{R}^{n \times J \times K}$ — e.g. excitation × emission fluorescence spectra, or wavelength × time chromatographic data — N-way PLS decomposes $\mathcal{X}$ into a sum of rank-one tensor components: $\mathcal{X} \approx \sum_{a=1}^{k} \mathbf{t}_a \circ \mathbf{w}_a^{J} \circ \mathbf{w}_a^{K}$, where $\circ$ is the outer product. The score vectors $\mathbf{t}_a$ are computed to maximise covariance with $\mathbf{y}$, the loading vectors $\mathbf{w}_a^{J}, \mathbf{w}_a^{K}$ live in the respective tensor modes.

Compared to unfolding the tensor and running standard PLS, N-PLS respects the multilinear structure of the data and produces interpretable per-mode loadings. The computational cost is comparable to standard PLS — matrix-vector products in each mode rather than one large matrix-vector product.

Note that pls4all takes the tensor as a **flattened** matrix plus `mode_j` and `mode_k` shape parameters; the kernel reshapes internally.

### Appropriate uses

Regression of multiway arrays while preserving tensor modes instead of unfolding all variables into one matrix.

### Limits and validation

Mode ranks and scaling must be chosen carefully; separable tensor components can underfit nonseparable structure.

### Implementation

`n4m_estimators_n_pls_fit`. Reference: Python `tensorly 0.9.0` (`tensorly.regression.tucker_regression`) and Bro's original MATLAB code.

### Sources and provenance

Current implementation: [cpp/src/core/extra_pls.cpp](https://github.com/GBeurier/nirs4all-methods/blob/main/cpp/src/core/extra_pls.cpp).


### API and bindings

**C ABI (ABI 2):** [`n4m_estimators_n_pls_fit`](https://github.com/GBeurier/nirs4all-methods/blob/main/cpp/include/n4m/estimators/regression.h#L175). Use the linked public header for the exact signature, configuration, and result handles.

**Python:** no current AST-verified public `n4m` re-export was found for this method. The linked C ABI above is the documented surface in this checkout.

**R (source-verified):** [`n_pls_fit(X_flat, Y, n_components, mode_j, mode_k)`](https://github.com/GBeurier/nirs4all-methods/blob/main/bindings/r/n4m/R/methods_extra.R).

The source signature has additional required inputs, so no example call is fabricated.

**MATLAB / Octave (source-verified):** [`n_pls(X_flat, Y, n_components, mode_j, mode_k)`](https://github.com/GBeurier/nirs4all-methods/blob/main/bindings/matlab/+n4m/n_pls.m).

The source signature has additional required inputs, so no example call is fabricated.

**Registry parity references** 📐

:::{card}
:class-card: external-refs

- 📐 **`ref.python_tensorly`** (python · python) — `tensorly` 0.9.0 · strict (rmse_rel ≤ 1e-06) — Python `tensorly.parafac` + OLS on mode-1 loadings. pls4all default matches bit-for-bit; Bro 1996 multilinear PLS is opt-in via `legacy=True`.

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
<thead><tr><th scope="col">Backend</th><th scope="col">Parity</th><th class="size-col" scope="col">200×48 (ms)</th></tr></thead>
<tbody class="lang-band lang-cpp"><tr class="lang-band-row" data-lang="cpp"><th colspan="3" scope="rowgroup"><span class="lang-band-dot"></span>C++ native · libn4m</th></tr>
<tr class="bk-row"><td class="bk-name"><code>pls4all.cpp.blas+omp</code></td><td class="parity parity-ref-strict">✓ ref</td><td class="ms">19.5 ms</td></tr>
</tbody>
<tbody class="lang-band lang-python"><tr class="lang-band-row" data-lang="python"><th colspan="3" scope="rowgroup"><span class="lang-band-dot"></span>Python · archived pls4all benchmark</th></tr>
<tr class="bk-row"><td class="bk-name"><code>pls4all.python</code></td><td class="parity parity-exact">✓ bind</td><td class="ms">19.0 ms</td></tr>
<tr class="bk-row"><td class="bk-name"><code>pls4all.sklearn</code></td><td class="parity parity-cross_check">⇄ +5e+00</td><td class="ms ms-best">2.03 ms<span class="medal" title="fastest">🏆</span></td></tr>
</tbody>
<tbody class="lang-band lang-r"><tr class="lang-band-row" data-lang="r"><th colspan="3" scope="rowgroup"><span class="lang-band-dot"></span>R · archived pls4all benchmark</th></tr>
<tr class="bk-row"><td class="bk-name"><code>pls4all.R</code></td><td class="parity parity-cross_check">⇄ +5e+00</td><td class="ms">4.51 ms</td></tr>
<tr class="bk-row"><td class="bk-name"><code>pls4all.R.formula</code></td><td class="parity parity-cross_check">⇄ +5e+00</td><td class="ms">5.67 ms</td></tr>
<tr class="bk-row"><td class="bk-name"><code>pls4all.R.mdatools</code></td><td class="parity parity-cross_check">⇄ +5e+00</td><td class="ms">5.37 ms</td></tr>
<tr class="bk-row"><td class="bk-name"><code>pls4all.R.pls</code></td><td class="parity parity-cross_check">⇄ +5e+00</td><td class="ms">5.68 ms</td></tr>
</tbody>
<tbody class="lang-band lang-python"><tr class="lang-band-row" data-lang="python"><th colspan="3" scope="rowgroup"><span class="lang-band-dot"></span>Python · external</th></tr>
<tr class="bk-row truth-source truth-source-strict"><td class="bk-name"><span class="truth-mark" title="Registry parity reference (python): tensorly 0.9.0 — strict (rmse_rel ≤ 1e-06)">📐</span><code>ref.python_tensorly</code></td><td class="parity parity-ref-source">source</td><td class="ms">19.3 ms</td></tr>
</tbody>
</table>
</div>

:::

:::{tab-item} 3 threads
:sync: threads-3

<div class="parity-table-wrap">
<table class="docutils parity-grouped">
<thead><tr><th scope="col">Backend</th><th scope="col">Parity</th><th class="size-col" scope="col">200×48 (ms)</th></tr></thead>
<tbody class="lang-band lang-cpp"><tr class="lang-band-row" data-lang="cpp"><th colspan="3" scope="rowgroup"><span class="lang-band-dot"></span>C++ native · libn4m</th></tr>
<tr class="bk-row"><td class="bk-name"><code>pls4all.cpp.blas+omp</code></td><td class="parity parity-ref-strict">✓ ref</td><td class="ms">19.1 ms</td></tr>
</tbody>
<tbody class="lang-band lang-python"><tr class="lang-band-row" data-lang="python"><th colspan="3" scope="rowgroup"><span class="lang-band-dot"></span>Python · archived pls4all benchmark</th></tr>
<tr class="bk-row"><td class="bk-name"><code>pls4all.python</code></td><td class="parity parity-exact">✓ bind</td><td class="ms">20.3 ms</td></tr>
<tr class="bk-row"><td class="bk-name"><code>pls4all.sklearn</code></td><td class="parity parity-cross_check">⇄ +5e+00</td><td class="ms ms-best">1.97 ms<span class="medal" title="fastest">🏆</span></td></tr>
</tbody>
<tbody class="lang-band lang-r"><tr class="lang-band-row" data-lang="r"><th colspan="3" scope="rowgroup"><span class="lang-band-dot"></span>R · archived pls4all benchmark</th></tr>
<tr class="bk-row"><td class="bk-name"><code>pls4all.R</code></td><td class="parity parity-cross_check">⇄ +5e+00</td><td class="ms">6.01 ms</td></tr>
<tr class="bk-row"><td class="bk-name"><code>pls4all.R.formula</code></td><td class="parity parity-cross_check">⇄ +5e+00</td><td class="ms">6.25 ms</td></tr>
<tr class="bk-row"><td class="bk-name"><code>pls4all.R.mdatools</code></td><td class="parity parity-cross_check">⇄ +5e+00</td><td class="ms">5.93 ms</td></tr>
<tr class="bk-row"><td class="bk-name"><code>pls4all.R.pls</code></td><td class="parity parity-cross_check">⇄ +5e+00</td><td class="ms">6.14 ms</td></tr>
</tbody>
<tbody class="lang-band lang-python"><tr class="lang-band-row" data-lang="python"><th colspan="3" scope="rowgroup"><span class="lang-band-dot"></span>Python · external</th></tr>
<tr class="bk-row truth-source truth-source-strict"><td class="bk-name"><span class="truth-mark" title="Registry parity reference (python): tensorly 0.9.0 — strict (rmse_rel ≤ 1e-06)">📐</span><code>ref.python_tensorly</code></td><td class="parity parity-ref-source">source</td><td class="ms">19.3 ms</td></tr>
</tbody>
</table>
</div>

:::

:::{tab-item} 10 threads
:sync: threads-10

<div class="parity-table-wrap">
<table class="docutils parity-grouped">
<thead><tr><th scope="col">Backend</th><th scope="col">Parity</th><th class="size-col" scope="col">200×48 (ms)</th></tr></thead>
<tbody class="lang-band lang-cpp"><tr class="lang-band-row" data-lang="cpp"><th colspan="3" scope="rowgroup"><span class="lang-band-dot"></span>C++ native · libn4m</th></tr>
<tr class="bk-row"><td class="bk-name"><code>pls4all.cpp.blas+omp</code></td><td class="parity parity-ref-strict">✓ ref</td><td class="ms">18.9 ms</td></tr>
</tbody>
<tbody class="lang-band lang-python"><tr class="lang-band-row" data-lang="python"><th colspan="3" scope="rowgroup"><span class="lang-band-dot"></span>Python · archived pls4all benchmark</th></tr>
<tr class="bk-row"><td class="bk-name"><code>pls4all.python</code></td><td class="parity parity-exact">✓ bind</td><td class="ms">19.4 ms</td></tr>
<tr class="bk-row"><td class="bk-name"><code>pls4all.sklearn</code></td><td class="parity parity-cross_check">⇄ +5e+00</td><td class="ms ms-best">1.87 ms<span class="medal" title="fastest">🏆</span></td></tr>
</tbody>
<tbody class="lang-band lang-r"><tr class="lang-band-row" data-lang="r"><th colspan="3" scope="rowgroup"><span class="lang-band-dot"></span>R · archived pls4all benchmark</th></tr>
<tr class="bk-row"><td class="bk-name"><code>pls4all.R</code></td><td class="parity parity-cross_check">⇄ +5e+00</td><td class="ms">4.84 ms</td></tr>
<tr class="bk-row"><td class="bk-name"><code>pls4all.R.formula</code></td><td class="parity parity-cross_check">⇄ +5e+00</td><td class="ms">5.52 ms</td></tr>
<tr class="bk-row"><td class="bk-name"><code>pls4all.R.mdatools</code></td><td class="parity parity-cross_check">⇄ +5e+00</td><td class="ms">5.17 ms</td></tr>
<tr class="bk-row"><td class="bk-name"><code>pls4all.R.pls</code></td><td class="parity parity-cross_check">⇄ +5e+00</td><td class="ms">5.46 ms</td></tr>
</tbody>
<tbody class="lang-band lang-python"><tr class="lang-band-row" data-lang="python"><th colspan="3" scope="rowgroup"><span class="lang-band-dot"></span>Python · external</th></tr>
<tr class="bk-row truth-source truth-source-strict"><td class="bk-name"><span class="truth-mark" title="Registry parity reference (python): tensorly 0.9.0 — strict (rmse_rel ≤ 1e-06)">📐</span><code>ref.python_tensorly</code></td><td class="parity parity-ref-source">source</td><td class="ms">18.8 ms</td></tr>
</tbody>
</table>
</div>

:::

::::


---

_See also_: [benchmark overview](../benchmarks/overview.md) · [methods index](index.md) · [interactive dashboard](../landing/dashboard.md)