# `irf_select` — Interval Random Frog (IRF) selection

_Group_: **Variable selector** · _Registry tolerance_: `1e-06`

## Description

Interval Random Frog (Phase 52)

> **Registry note** — Python `auswahl.IntervalRandomFrog` (LSX-UniWue; Yun 2013). Same algorithm as libPLS `irf`. Default `_irf_select_pls4all` path mirrors the same auswahl call with `random_state=seed`, giving bit-exact mask parity. The C++ splitmix64 kernel is opt-in via `legacy=True`.

### Parameters

| Name | Type | Default | Notes |
|------|------|---------|-------|
| `top_k` | `int` | `required` | current public binding signature |
| `n_components` | `int` | `2` | current public binding signature |
| `n_iterations` | `int` | `100` | current public binding signature |
| `window_size` | `int` | `5` | current public binding signature |
| `initial_intervals` | `int` | `5` | current public binding signature |
| `seed` | `int` | `0` | current public binding signature |

## Explanations

### Bibliographic source

Yun, Y.-H. et al. (2013). *An efficient method of wavelength interval selection based on random frog for multivariate spectral calibration*. Spectrochimica Acta Part A 111, 31--36. DOI [10.1016/j.saa.2013.03.083](https://doi.org/10.1016/j.saa.2013.03.083). The shipped libPLS-compatible path may differ in details; it is not the random-forest IRF of Basu et al.

### Mathematical principle

IRF partitions the ordered variables into sliding intervals, uses absolute PLS coefficients to score them, and proposes random changes to the active interval subset. Cross-validated RMSE accepts or rejects proposals and selection frequencies summarize the sampled chain.

### Appropriate uses

Searching contiguous wavelength intervals with random-frog proposals and a PLS cross-validation objective.

### Limits and validation

IRF here means Interval Random Frog: it uses no random forest or interaction-aware tree importance, and depends on interval width and chain mixing.

### Implementation

`cpp/src/core/irf_selection.cpp` contains no decision trees or random-forest feature importance. Interval width, proposal count, seed, and chain mixing determine the frequency estimates.

### Sources and provenance

Current implementation: [cpp/src/core/irf_selection.cpp](https://github.com/GBeurier/nirs4all-methods/blob/main/cpp/src/core/irf_selection.cpp).


### API and bindings

**C ABI (ABI 2):** [`n4m_feature_selection_irf_select`](https://github.com/GBeurier/nirs4all-methods/blob/main/cpp/include/n4m/feature_selection.h#L414). Use the linked public header for the exact signature, configuration, and result handles.

**Python (verified public re-export):**

```python
from n4m.feature_selection.wrapper import IRF
```

Source signature: `IRF(top_k: int, *, n_components: int = 2, n_iterations: int = 100, window_size: int = 5, initial_intervals: int = 5, seed: int = 0)` ([`n4m/_impl/selection.py`](https://github.com/GBeurier/nirs4all-methods/blob/main/bindings/python/src/n4m/_impl/selection.py#L752)).

**R (source-verified):** [`irf_select(X, Y, n_components, n_iterations = 100L, window_size = 10L, initial_intervals = 10L, top_k = 5L, seed = 0L)`](https://github.com/GBeurier/nirs4all-methods/blob/main/bindings/r/n4m/R/methods_extra.R).

The source signature has additional required inputs, so no example call is fabricated.

**MATLAB / Octave (source-verified):** [`irf_select(X, Y, n_components, n_iterations, window_size, ... initial_intervals, top_k, seed)`](https://github.com/GBeurier/nirs4all-methods/blob/main/bindings/matlab/+n4m/irf_select.m).

The source signature has additional required inputs, so no example call is fabricated.

**Registry parity references** 📐

:::{card}
:class-card: external-refs

- 📐 **`ref.python_auswahl`** (python · python) — `auswahl` 0.9.0 · strict (rmse_rel ≤ 1e-06) — Python `auswahl.IntervalRandomFrog` (LSX-UniWue; Yun 2013). Same algorithm as libPLS `irf` with pinned `random_state` for bit-exact mask parity.

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
<thead><tr><th scope="col">Backend</th><th scope="col">Parity</th><th class="size-col" scope="col">120×30 (ms)</th></tr></thead>
<tbody class="lang-band lang-python"><tr class="lang-band-row" data-lang="python"><th colspan="3" scope="rowgroup"><span class="lang-band-dot"></span>Python · archived pls4all benchmark</th></tr>
<tr class="bk-row"><td class="bk-name"><code>pls4all.sklearn</code></td><td class="parity parity-exact">✓ J 0.41</td><td class="ms ms-best">2.02 ms<span class="medal" title="fastest">🏆</span></td></tr>
</tbody>
<tbody class="lang-band lang-r"><tr class="lang-band-row" data-lang="r"><th colspan="3" scope="rowgroup"><span class="lang-band-dot"></span>R · archived pls4all benchmark</th></tr>
<tr class="bk-row"><td class="bk-name"><code>pls4all.R</code></td><td class="parity parity-exact">✓ J 0.41</td><td class="ms">2.92 ms</td></tr>
<tr class="bk-row"><td class="bk-name"><code>pls4all.R.formula</code></td><td class="parity parity-exact">✓ J 0.41</td><td class="ms">3.44 ms</td></tr>
<tr class="bk-row"><td class="bk-name"><code>pls4all.R.mdatools</code></td><td class="parity parity-exact">✓ J 0.41</td><td class="ms">3.37 ms</td></tr>
<tr class="bk-row"><td class="bk-name"><code>pls4all.R.pls</code></td><td class="parity parity-exact">✓ J 0.41</td><td class="ms">3.91 ms</td></tr>
</tbody>
</table>
</div>

:::

:::{tab-item} 3 threads
:sync: threads-3

<div class="parity-table-wrap">
<table class="docutils parity-grouped">
<thead><tr><th scope="col">Backend</th><th scope="col">Parity</th><th class="size-col" scope="col">120×30 (ms)</th></tr></thead>
<tbody class="lang-band lang-python"><tr class="lang-band-row" data-lang="python"><th colspan="3" scope="rowgroup"><span class="lang-band-dot"></span>Python · archived pls4all benchmark</th></tr>
<tr class="bk-row"><td class="bk-name"><code>pls4all.sklearn</code></td><td class="parity parity-exact">✓ J 0.41</td><td class="ms ms-best">1.53 ms<span class="medal" title="fastest">🏆</span></td></tr>
</tbody>
<tbody class="lang-band lang-r"><tr class="lang-band-row" data-lang="r"><th colspan="3" scope="rowgroup"><span class="lang-band-dot"></span>R · archived pls4all benchmark</th></tr>
<tr class="bk-row"><td class="bk-name"><code>pls4all.R</code></td><td class="parity parity-exact">✓ J 0.41</td><td class="ms">2.93 ms</td></tr>
<tr class="bk-row"><td class="bk-name"><code>pls4all.R.formula</code></td><td class="parity parity-exact">✓ J 0.41</td><td class="ms">3.29 ms</td></tr>
<tr class="bk-row"><td class="bk-name"><code>pls4all.R.mdatools</code></td><td class="parity parity-exact">✓ J 0.41</td><td class="ms">3.58 ms</td></tr>
<tr class="bk-row"><td class="bk-name"><code>pls4all.R.pls</code></td><td class="parity parity-exact">✓ J 0.41</td><td class="ms">3.55 ms</td></tr>
</tbody>
</table>
</div>

:::

:::{tab-item} 10 threads
:sync: threads-10

<div class="parity-table-wrap">
<table class="docutils parity-grouped">
<thead><tr><th scope="col">Backend</th><th scope="col">Parity</th><th class="size-col" scope="col">120×30 (ms)</th></tr></thead>
<tbody class="lang-band lang-python"><tr class="lang-band-row" data-lang="python"><th colspan="3" scope="rowgroup"><span class="lang-band-dot"></span>Python · archived pls4all benchmark</th></tr>
<tr class="bk-row"><td class="bk-name"><code>pls4all.sklearn</code></td><td class="parity parity-exact">✓ J 0.41</td><td class="ms ms-best">1.53 ms<span class="medal" title="fastest">🏆</span></td></tr>
</tbody>
<tbody class="lang-band lang-r"><tr class="lang-band-row" data-lang="r"><th colspan="3" scope="rowgroup"><span class="lang-band-dot"></span>R · archived pls4all benchmark</th></tr>
<tr class="bk-row"><td class="bk-name"><code>pls4all.R</code></td><td class="parity parity-exact">✓ J 0.41</td><td class="ms">2.56 ms</td></tr>
<tr class="bk-row"><td class="bk-name"><code>pls4all.R.formula</code></td><td class="parity parity-exact">✓ J 0.41</td><td class="ms">3.34 ms</td></tr>
<tr class="bk-row"><td class="bk-name"><code>pls4all.R.mdatools</code></td><td class="parity parity-exact">✓ J 0.41</td><td class="ms">3.55 ms</td></tr>
<tr class="bk-row"><td class="bk-name"><code>pls4all.R.pls</code></td><td class="parity parity-exact">✓ J 0.41</td><td class="ms">3.57 ms</td></tr>
</tbody>
</table>
</div>

:::

::::


---

_See also_: [benchmark overview](../benchmarks/overview.md) · [methods index](index.md) · [interactive dashboard](../landing/dashboard.md)