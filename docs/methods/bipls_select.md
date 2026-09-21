# `bipls_select` — biPLS — Backward Interval PLS

_Group_: **Variable selector** · _Registry tolerance_: `0.7`

## Description

biPLS backward interval elimination (§18 Phase 5p)

> **Registry note** — R `mdatools::ipls(method='backward')`. Mask RMSE-rel ~0=perfect, ~1=half disagree, ~1.41=disjoint; tolerance 0.7 enforces ~50% overlap. Backward elimination is order-sensitive.

### Parameters

| Name | Type | Default | Notes |
|------|------|---------|-------|
| `n_components` | `int` | `2` | current public binding signature |
| `interval_width` | `int` | `10` | current public binding signature |
| `min_intervals` | `int` | `2` | current public binding signature |
| `n_folds` | `int` | `3` | current public binding signature |

## Explanations

### Bibliographic source

Leardi, R. & Nørgaard, L. (2004). *Sequential application of backward interval partial least squares and genetic algorithms for the selection of relevant spectral regions*. Journal of Chemometrics 18(11), 486–497. Verified primary link: [https://doi.org/10.1002/cem.893](https://doi.org/10.1002/cem.893).

### Mathematical principle

Start with the spectrum partitioned into $I$ equal intervals (typically 10–40). At each iteration, remove the interval whose removal **least** hurts CV-RMSE — i.e. the least informative interval. Iterate until removing any further interval materially worsens performance.

Returns a multi-band subset with each band aligned to the original equal-partition grid. The discrete structure makes biPLS robust to noise (no fine-grained fishing) and easy to interpret (each retained interval is a spectroscopic region of contiguous wavelengths).

Commonly chained with GA-PLS as a coarse-to-fine pipeline (Leardi & Nørgaard 2004): biPLS narrows the candidate intervals, GA-PLS does the within-interval feature selection.

### Appropriate uses

Backward elimination of predefined spectral intervals using PLS validation error.

### Limits and validation

Greedy deletion cannot recover removed intervals and repeated CV comparison can produce optimistic subset estimates.

### Implementation

`n4m_feature_selection_bipls_select`. Reference: R `plsVarSel`.

### Sources and provenance

Current implementation: [cpp/src/core/bipls_selection.cpp](https://github.com/GBeurier/nirs4all-methods/blob/main/cpp/src/core/bipls_selection.cpp).


### API and bindings

**C ABI (ABI 2):** [`n4m_feature_selection_bipls_select`](https://github.com/GBeurier/nirs4all-methods/blob/main/cpp/include/n4m/feature_selection.h#L313). Use the linked public header for the exact signature, configuration, and result handles.

**Python (verified public re-export):**

```python
from n4m.feature_selection.wrapper import BiPLS
```

Source signature: `BiPLS(*, n_components: int = 2, interval_width: int = 10, min_intervals: int = 2, n_folds: int = 3)` ([`n4m/_impl/selection.py`](https://github.com/GBeurier/nirs4all-methods/blob/main/bindings/python/src/n4m/_impl/selection.py#L541)).

**R (source-verified):** [`bipls_select(X, Y, n_components, interval_width = 10L, min_intervals = 1L)`](https://github.com/GBeurier/nirs4all-methods/blob/main/bindings/r/n4m/R/methods_extra.R).

The source signature has additional required inputs, so no example call is fabricated.

**MATLAB / Octave (source-verified):** [`bipls_select(X, Y, n_components, interval_width, min_intervals)`](https://github.com/GBeurier/nirs4all-methods/blob/main/bindings/matlab/+n4m/bipls_select.m).

The source signature has additional required inputs, so no example call is fabricated.

**Registry parity references** 📐

:::{card}
:class-card: external-refs

- 📐 **`ref.r_mdatools`** (R · r) — `mdatools` 0.15.0 · qualitative (rmse_rel ≤ 7e-01) — R `mdatools::ipls(method='backward')` — biPLS elimination. Returns variables from intervals that survive the backward sweep.

:::

### Benchmarks

**Archived measurement identity.** Backend labels in this table are the raw IDs recorded when the benchmark ran (including historical `pls4all.*` IDs). They preserve measurement provenance and do not describe a current public Python, R, or MATLAB binding; use the source-verified **API and bindings** section above for current entry points.

Adaptive wall-clock per cell measured against [`full_matrix.csv`](../benchmarks/overview.md). Only backends that implement this method are listed; libraries without the method are omitted.

**Verdict** &nbsp;·&nbsp; ✓ ref / ≈ ref / ~ shape mark a reference-gate pass at strict / relaxed / qualitative tolerance &nbsp;·&nbsp; ✓ bind = archived binding-harness result agrees with the C++ baseline &nbsp;·&nbsp; ⇄ cross-check = documented by-design selector/RNG/model, noncanonical API/facade convention, or secondary oracle &nbsp;·&nbsp; ✗ divergent &nbsp;·&nbsp; ⚠ error &nbsp;·&nbsp; — not run. The fastest backend per column is marked 🏆.

**Reference gate**: qualitative — shape/smoke comparison only. The external library and archived benchmark harness do not produce numerically equivalent output for this method (see the MethodSpec notes); the `rmse_rel_tol ≤ 7e-01` budget is set wide on purpose. Treat ~ shape as *“we ran both, both finished”*, not as numerical agreement.

Rows tagged with **📐** are the canonical parity references for this method (declared in [`parity_timing.registry`](../benchmarks/methodology.md)). C++ and external rows show reference parity; archived language-harness rows show binding parity against the C++ backend. Hover the icon for role and tolerance band.

::::{tab-set}
:class: parity-tabs

:::{tab-item} 1 thread
:sync: threads-1

<div class="parity-table-wrap">
<table class="docutils parity-grouped">
<thead><tr><th scope="col">Backend</th><th scope="col">Parity</th><th class="size-col" scope="col">200×40 (ms)</th></tr></thead>
<tbody class="lang-band lang-cpp"><tr class="lang-band-row" data-lang="cpp"><th colspan="3" scope="rowgroup"><span class="lang-band-dot"></span>C++ native · libn4m</th></tr>
<tr class="bk-row"><td class="bk-name"><code>pls4all.cpp.blas+omp</code></td><td class="parity parity-exact">✓ J 1.00</td><td class="ms">3.83 ms</td></tr>
</tbody>
<tbody class="lang-band lang-python"><tr class="lang-band-row" data-lang="python"><th colspan="3" scope="rowgroup"><span class="lang-band-dot"></span>Python · archived pls4all benchmark</th></tr>
<tr class="bk-row"><td class="bk-name"><code>pls4all.python</code></td><td class="parity parity-exact">✓ J 1.00</td><td class="ms ms-best">3.60 ms<span class="medal" title="fastest">🏆</span></td></tr>
<tr class="bk-row"><td class="bk-name"><code>pls4all.sklearn</code></td><td class="parity parity-exact">✓ J 1.00</td><td class="ms">3.69 ms</td></tr>
</tbody>
<tbody class="lang-band lang-r"><tr class="lang-band-row" data-lang="r"><th colspan="3" scope="rowgroup"><span class="lang-band-dot"></span>R · archived pls4all benchmark</th></tr>
<tr class="bk-row"><td class="bk-name"><code>pls4all.R</code></td><td class="parity parity-exact">✓ J 1.00</td><td class="ms">8.26 ms</td></tr>
<tr class="bk-row"><td class="bk-name"><code>pls4all.R.formula</code></td><td class="parity parity-exact">✓ J 1.00</td><td class="ms">11.7 ms</td></tr>
<tr class="bk-row"><td class="bk-name"><code>pls4all.R.mdatools</code></td><td class="parity parity-exact">✓ J 1.00</td><td class="ms">9.90 ms</td></tr>
<tr class="bk-row"><td class="bk-name"><code>pls4all.R.pls</code></td><td class="parity parity-exact">✓ J 1.00</td><td class="ms">8.83 ms</td></tr>
</tbody>
<tbody class="lang-band lang-r"><tr class="lang-band-row" data-lang="r"><th colspan="3" scope="rowgroup"><span class="lang-band-dot"></span>R · external</th></tr>
<tr class="bk-row truth-source truth-source-qualitative"><td class="bk-name"><span class="truth-mark" title="Registry parity reference (r): mdatools 0.15.0 — qualitative (rmse_rel ≤ 7e-01)">📐</span><code>ref.r_mdatools</code></td><td class="parity parity-ref-source">source</td><td class="ms">365.6 ms</td></tr>
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
<tr class="bk-row"><td class="bk-name"><code>pls4all.cpp.blas+omp</code></td><td class="parity parity-exact">✓ J 1.00</td><td class="ms ms-best">3.49 ms<span class="medal" title="fastest">🏆</span></td></tr>
</tbody>
<tbody class="lang-band lang-python"><tr class="lang-band-row" data-lang="python"><th colspan="3" scope="rowgroup"><span class="lang-band-dot"></span>Python · archived pls4all benchmark</th></tr>
<tr class="bk-row"><td class="bk-name"><code>pls4all.python</code></td><td class="parity parity-exact">✓ J 1.00</td><td class="ms">3.58 ms</td></tr>
<tr class="bk-row"><td class="bk-name"><code>pls4all.sklearn</code></td><td class="parity parity-exact">✓ J 1.00</td><td class="ms">3.74 ms</td></tr>
</tbody>
<tbody class="lang-band lang-r"><tr class="lang-band-row" data-lang="r"><th colspan="3" scope="rowgroup"><span class="lang-band-dot"></span>R · archived pls4all benchmark</th></tr>
<tr class="bk-row"><td class="bk-name"><code>pls4all.R</code></td><td class="parity parity-exact">✓ J 1.00</td><td class="ms">9.93 ms</td></tr>
<tr class="bk-row"><td class="bk-name"><code>pls4all.R.formula</code></td><td class="parity parity-exact">✓ J 1.00</td><td class="ms">25.5 ms</td></tr>
<tr class="bk-row"><td class="bk-name"><code>pls4all.R.mdatools</code></td><td class="parity parity-exact">✓ J 1.00</td><td class="ms">15.1 ms</td></tr>
<tr class="bk-row"><td class="bk-name"><code>pls4all.R.pls</code></td><td class="parity parity-exact">✓ J 1.00</td><td class="ms">24.8 ms</td></tr>
</tbody>
<tbody class="lang-band lang-r"><tr class="lang-band-row" data-lang="r"><th colspan="3" scope="rowgroup"><span class="lang-band-dot"></span>R · external</th></tr>
<tr class="bk-row truth-source truth-source-qualitative"><td class="bk-name"><span class="truth-mark" title="Registry parity reference (r): mdatools 0.15.0 — qualitative (rmse_rel ≤ 7e-01)">📐</span><code>ref.r_mdatools</code></td><td class="parity parity-ref-source">source</td><td class="ms">1.0 s</td></tr>
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
<tr class="bk-row"><td class="bk-name"><code>pls4all.cpp.blas+omp</code></td><td class="parity parity-exact">✓ J 1.00</td><td class="ms">14.3 ms</td></tr>
</tbody>
<tbody class="lang-band lang-python"><tr class="lang-band-row" data-lang="python"><th colspan="3" scope="rowgroup"><span class="lang-band-dot"></span>Python · archived pls4all benchmark</th></tr>
<tr class="bk-row"><td class="bk-name"><code>pls4all.python</code></td><td class="parity parity-exact">✓ J 1.00</td><td class="ms">13.4 ms</td></tr>
<tr class="bk-row"><td class="bk-name"><code>pls4all.sklearn</code></td><td class="parity parity-exact">✓ J 1.00</td><td class="ms ms-best">5.97 ms<span class="medal" title="fastest">🏆</span></td></tr>
</tbody>
<tbody class="lang-band lang-r"><tr class="lang-band-row" data-lang="r"><th colspan="3" scope="rowgroup"><span class="lang-band-dot"></span>R · archived pls4all benchmark</th></tr>
<tr class="bk-row"><td class="bk-name"><code>pls4all.R</code></td><td class="parity parity-exact">✓ J 1.00</td><td class="ms">23.9 ms</td></tr>
<tr class="bk-row"><td class="bk-name"><code>pls4all.R.formula</code></td><td class="parity parity-exact">✓ J 1.00</td><td class="ms">35.4 ms</td></tr>
<tr class="bk-row"><td class="bk-name"><code>pls4all.R.mdatools</code></td><td class="parity parity-exact">✓ J 1.00</td><td class="ms">16.3 ms</td></tr>
<tr class="bk-row"><td class="bk-name"><code>pls4all.R.pls</code></td><td class="parity parity-exact">✓ J 1.00</td><td class="ms">21.7 ms</td></tr>
</tbody>
<tbody class="lang-band lang-r"><tr class="lang-band-row" data-lang="r"><th colspan="3" scope="rowgroup"><span class="lang-band-dot"></span>R · external</th></tr>
<tr class="bk-row truth-source truth-source-qualitative"><td class="bk-name"><span class="truth-mark" title="Registry parity reference (r): mdatools 0.15.0 — qualitative (rmse_rel ≤ 7e-01)">📐</span><code>ref.r_mdatools</code></td><td class="parity parity-ref-source">source</td><td class="ms">333.9 ms</td></tr>
</tbody>
</table>
</div>

:::

::::


---

_See also_: [benchmark overview](../benchmarks/overview.md) · [methods index](index.md) · [interactive dashboard](../landing/dashboard.md)