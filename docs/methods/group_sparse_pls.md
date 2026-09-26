# `group_sparse_pls` — Group-sparse PLS (Liquet 2016)

_Group_: **Sparse** · _Registry tolerance_: `1e-08`

## Description

Group sparse PLS (§7)

> **Registry note** — R `sgPLS::gPLS` (Liquet et al. 2016, regression mode, scale=TRUE). The archived benchmark's default Python reference is a deterministic NumPy port of that algorithm. Its `legacy=True` route invokes the native C ABI, which uses post-SIMPLS coefficient shrinkage; these are distinct estimators, not numerical equivalents.

### Parameters

| Name | Type | Default | Notes |
|------|------|---------|-------|
| `n_components` | `int` | `4` | registry benchmark cell value |
| `group_lambda` | `float` | `0.1` | registry benchmark cell value |

## Explanations

### Bibliographic source

Liquet, B., de Micheaux, P. L., Hejblum, B. P. & Thiébaut, R. (2016). *Group and sparse group partial least squares approaches applied in genomics context*. Bioinformatics 32(1), 35–42. Verified primary link: [https://doi.org/10.1093/bioinformatics/btv535](https://doi.org/10.1093/bioinformatics/btv535).

### Mathematical principle

The routine fits ordinary SIMPLS, then applies the proximal map `B_g <- max(0, 1 - group_lambda / ||B_g||_F) B_g` to each group of predictive coefficient rows. Entire groups can be zeroed, but latent directions are not refitted.

### Appropriate uses

Exploratory group selection with predefined wavelength groups when post-fit coefficient shrinkage is acceptable.

### Limits and validation

The penalty shrinks predictive coefficient groups and affects predictions, but does not refit latent directions or implement sgPLS::gPLS. Its units depend on raw-X and target scaling; select it using held-out data.

### Implementation

`group_lambda` applies group-lasso proximal shrinkage to the returned SIMPLS predictive coefficients. This changes predictions but is not a latent-direction refit or a numerical implementation of sgPLS::gPLS.

### Sources and provenance

Current implementation: [cpp/src/core/extra_pls.cpp](https://github.com/GBeurier/nirs4all-methods/blob/main/cpp/src/core/extra_pls.cpp).


### API and bindings

**C ABI (ABI 2):** [`n4m_estimators_group_sparse_pls_fit`](https://github.com/GBeurier/nirs4all-methods/blob/main/cpp/include/n4m/estimators/regression.h#L209). Use the linked public header for the exact signature, configuration, and result handles.

**Python:** no current AST-verified public `n4m` re-export was found for this method. The linked C ABI above is the documented surface in this checkout.

**R (source-verified):** [`group_sparse_pls_fit(X, Y, n_components, group_assignment, group_lambda = 0.05)`](https://github.com/GBeurier/nirs4all-methods/blob/main/bindings/r/n4m/R/methods_extra.R).

The source signature has additional required inputs, so no example call is fabricated.

**MATLAB / Octave (source-verified):** [`group_sparse_pls(X, Y, n_components, group_assignment, group_lambda)`](https://github.com/GBeurier/nirs4all-methods/blob/main/bindings/matlab/+n4m/group_sparse_pls.m).

The source signature has additional required inputs, so no example call is fabricated.

**Registry parity references** 📐

:::{card}
:class-card: external-refs

- 📐 **`ref.python_numpy`** (python · python) — `numpy` in-tree · strict (rmse_rel ≤ 1e-08) — In-tree NumPy port of Liquet et al. 2016 group sparse PLS (R `sgPLS::gPLS`, regression mode, scale=TRUE). The archived default Python reference uses that port. The opt-in native route uses post-SIMPLS coefficient shrinkage and is not numerically equivalent to sgPLS::gPLS.

- 📐 **`ref.r_sgpls`** (R · r) — `sgPLS` 1.8.1 · strict (rmse_rel ≤ 1e-08) — R `sgPLS::gPLS(X, Y, ncomp, ind.block.x, keepX=length(bnd))` (regression, scale=TRUE). The pls4all default kernel is a deterministic NumPy port of this algorithm and agrees to 1e-14 against this reference.

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
<thead><tr><th scope="col">Backend</th><th scope="col">Parity</th><th class="size-col" scope="col">200×50 (ms)</th></tr></thead>
<tbody class="lang-band lang-cpp"><tr class="lang-band-row" data-lang="cpp"><th colspan="3" scope="rowgroup"><span class="lang-band-dot"></span>C++ native · libn4m</th></tr>
<tr class="bk-row"><td class="bk-name"><code>pls4all.cpp.blas+omp</code></td><td class="parity parity-ref-strict">✓ ref</td><td class="ms">8.36 ms</td></tr>
</tbody>
<tbody class="lang-band lang-python"><tr class="lang-band-row" data-lang="python"><th colspan="3" scope="rowgroup"><span class="lang-band-dot"></span>Python · archived pls4all benchmark</th></tr>
<tr class="bk-row"><td class="bk-name"><code>pls4all.python</code></td><td class="parity parity-exact">✓ bind</td><td class="ms">7.86 ms</td></tr>
<tr class="bk-row"><td class="bk-name"><code>pls4all.sklearn</code></td><td class="parity parity-cross_check">⇄ +1e-01</td><td class="ms">3.79 ms</td></tr>
</tbody>
<tbody class="lang-band lang-r"><tr class="lang-band-row" data-lang="r"><th colspan="3" scope="rowgroup"><span class="lang-band-dot"></span>R · archived pls4all benchmark</th></tr>
<tr class="bk-row"><td class="bk-name"><code>pls4all.R</code></td><td class="parity parity-cross_check">⇄ +1e-01</td><td class="ms">9.70 ms</td></tr>
<tr class="bk-row"><td class="bk-name"><code>pls4all.R.formula</code></td><td class="parity parity-cross_check">⇄ +1e-01</td><td class="ms">11.0 ms</td></tr>
<tr class="bk-row"><td class="bk-name"><code>pls4all.R.mdatools</code></td><td class="parity parity-cross_check">⇄ +1e-01</td><td class="ms">10.2 ms</td></tr>
<tr class="bk-row"><td class="bk-name"><code>pls4all.R.pls</code></td><td class="parity parity-cross_check">⇄ +1e-01</td><td class="ms">11.0 ms</td></tr>
</tbody>
<tbody class="lang-band lang-python"><tr class="lang-band-row" data-lang="python"><th colspan="3" scope="rowgroup"><span class="lang-band-dot"></span>Python · external</th></tr>
<tr class="bk-row truth-source truth-source-strict"><td class="bk-name"><span class="truth-mark" title="Registry parity reference (python): numpy in-tree — strict (rmse_rel ≤ 1e-08)">📐</span><code>ref.python_numpy</code></td><td class="parity parity-ref-source">source</td><td class="ms ms-best">2.98 ms<span class="medal" title="fastest">🏆</span></td></tr>
</tbody>
<tbody class="lang-band lang-r"><tr class="lang-band-row" data-lang="r"><th colspan="3" scope="rowgroup"><span class="lang-band-dot"></span>R · external</th></tr>
<tr class="bk-row truth-source truth-source-strict"><td class="bk-name"><span class="truth-mark" title="Registry parity reference (r): sgPLS 1.8.1 — strict (rmse_rel ≤ 1e-08)">📐</span><code>ref.r_sgpls</code></td><td class="parity parity-cross_check">⇄ +1e-15</td><td class="ms">33.7 ms</td></tr>
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
<tr class="bk-row"><td class="bk-name"><code>pls4all.cpp.blas+omp</code></td><td class="parity parity-ref-strict">✓ ref</td><td class="ms">2.62 ms</td></tr>
</tbody>
<tbody class="lang-band lang-python"><tr class="lang-band-row" data-lang="python"><th colspan="3" scope="rowgroup"><span class="lang-band-dot"></span>Python · archived pls4all benchmark</th></tr>
<tr class="bk-row"><td class="bk-name"><code>pls4all.python</code></td><td class="parity parity-exact">✓ bind</td><td class="ms">2.76 ms</td></tr>
<tr class="bk-row"><td class="bk-name"><code>pls4all.sklearn</code></td><td class="parity parity-cross_check">⇄ +1e-01</td><td class="ms ms-best">2.28 ms<span class="medal" title="fastest">🏆</span></td></tr>
</tbody>
<tbody class="lang-band lang-r"><tr class="lang-band-row" data-lang="r"><th colspan="3" scope="rowgroup"><span class="lang-band-dot"></span>R · archived pls4all benchmark</th></tr>
<tr class="bk-row"><td class="bk-name"><code>pls4all.R</code></td><td class="parity parity-cross_check">⇄ +1e-01</td><td class="ms">8.78 ms</td></tr>
<tr class="bk-row"><td class="bk-name"><code>pls4all.R.formula</code></td><td class="parity parity-cross_check">⇄ +1e-01</td><td class="ms">9.80 ms</td></tr>
<tr class="bk-row"><td class="bk-name"><code>pls4all.R.mdatools</code></td><td class="parity parity-cross_check">⇄ +1e-01</td><td class="ms">11.9 ms</td></tr>
<tr class="bk-row"><td class="bk-name"><code>pls4all.R.pls</code></td><td class="parity parity-cross_check">⇄ +1e-01</td><td class="ms">10.5 ms</td></tr>
</tbody>
<tbody class="lang-band lang-python"><tr class="lang-band-row" data-lang="python"><th colspan="3" scope="rowgroup"><span class="lang-band-dot"></span>Python · external</th></tr>
<tr class="bk-row truth-source truth-source-strict"><td class="bk-name"><span class="truth-mark" title="Registry parity reference (python): numpy in-tree — strict (rmse_rel ≤ 1e-08)">📐</span><code>ref.python_numpy</code></td><td class="parity parity-ref-source">source</td><td class="ms">3.11 ms</td></tr>
</tbody>
<tbody class="lang-band lang-r"><tr class="lang-band-row" data-lang="r"><th colspan="3" scope="rowgroup"><span class="lang-band-dot"></span>R · external</th></tr>
<tr class="bk-row truth-source truth-source-strict"><td class="bk-name"><span class="truth-mark" title="Registry parity reference (r): sgPLS 1.8.1 — strict (rmse_rel ≤ 1e-08)">📐</span><code>ref.r_sgpls</code></td><td class="parity parity-cross_check">⇄ +1e-15</td><td class="ms">31.2 ms</td></tr>
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
<tr class="bk-row"><td class="bk-name"><code>pls4all.cpp.blas+omp</code></td><td class="parity parity-ref-strict">✓ ref</td><td class="ms">2.91 ms</td></tr>
</tbody>
<tbody class="lang-band lang-python"><tr class="lang-band-row" data-lang="python"><th colspan="3" scope="rowgroup"><span class="lang-band-dot"></span>Python · archived pls4all benchmark</th></tr>
<tr class="bk-row"><td class="bk-name"><code>pls4all.python</code></td><td class="parity parity-exact">✓ bind</td><td class="ms">2.69 ms</td></tr>
<tr class="bk-row"><td class="bk-name"><code>pls4all.sklearn</code></td><td class="parity parity-cross_check">⇄ +1e-01</td><td class="ms ms-best">2.53 ms<span class="medal" title="fastest">🏆</span></td></tr>
</tbody>
<tbody class="lang-band lang-r"><tr class="lang-band-row" data-lang="r"><th colspan="3" scope="rowgroup"><span class="lang-band-dot"></span>R · archived pls4all benchmark</th></tr>
<tr class="bk-row"><td class="bk-name"><code>pls4all.R</code></td><td class="parity parity-cross_check">⇄ +1e-01</td><td class="ms">7.45 ms</td></tr>
<tr class="bk-row"><td class="bk-name"><code>pls4all.R.formula</code></td><td class="parity parity-cross_check">⇄ +1e-01</td><td class="ms">9.19 ms</td></tr>
<tr class="bk-row"><td class="bk-name"><code>pls4all.R.mdatools</code></td><td class="parity parity-cross_check">⇄ +1e-01</td><td class="ms">11.3 ms</td></tr>
<tr class="bk-row"><td class="bk-name"><code>pls4all.R.pls</code></td><td class="parity parity-cross_check">⇄ +1e-01</td><td class="ms">8.50 ms</td></tr>
</tbody>
<tbody class="lang-band lang-python"><tr class="lang-band-row" data-lang="python"><th colspan="3" scope="rowgroup"><span class="lang-band-dot"></span>Python · external</th></tr>
<tr class="bk-row truth-source truth-source-strict"><td class="bk-name"><span class="truth-mark" title="Registry parity reference (python): numpy in-tree — strict (rmse_rel ≤ 1e-08)">📐</span><code>ref.python_numpy</code></td><td class="parity parity-ref-source">source</td><td class="ms">2.66 ms</td></tr>
</tbody>
<tbody class="lang-band lang-r"><tr class="lang-band-row" data-lang="r"><th colspan="3" scope="rowgroup"><span class="lang-band-dot"></span>R · external</th></tr>
<tr class="bk-row truth-source truth-source-strict"><td class="bk-name"><span class="truth-mark" title="Registry parity reference (r): sgPLS 1.8.1 — strict (rmse_rel ≤ 1e-08)">📐</span><code>ref.r_sgpls</code></td><td class="parity parity-cross_check">⇄ +1e-15</td><td class="ms">26.7 ms</td></tr>
</tbody>
</table>
</div>

:::

::::


---

_See also_: [benchmark overview](../benchmarks/overview.md) · [methods index](index.md) · [interactive dashboard](../landing/dashboard.md)