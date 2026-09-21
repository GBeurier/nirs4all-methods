# `emcuve_select` — EMCUVE — Ensemble MC-UVE

_Group_: **Variable selector** · _Registry tolerance_: `1e-06`

## Description

EMCUVE ensemble MC-UVE (§18 Phase 5n)

> **Registry note** — R `plsVarSel::mcuve_pls` called `n_ensembles` times with deterministic seeds (`11 + e`) and vote-aggregated. Default `_emcuve_select_pls4all` path mirrors the same R loop, giving bit-exact mask parity. The C++ splitmix64 EMCUVE kernel is opt-in via `legacy=True`.

### Parameters

| Name | Type | Default | Notes |
|------|------|---------|-------|
| `n_components` | `int` | `2` | current public binding signature |
| `noise_features` | `int` | `50` | current public binding signature |
| `noise_seed` | `int` | `0` | current public binding signature |
| `n_ensembles` | `int` | `10` | current public binding signature |
| `vote_threshold` | `float` | `0.5` | current public binding signature |
| `n_folds` | `int` | `3` | current public binding signature |

## Explanations

### Bibliographic source

Han, Q.-J., Wu, H.-L., Cai, C.-B., Xu, L. & Yu, R.-Q. (2008). *An ensemble of Monte Carlo uninformative variable elimination for wavelength selection*. Analytica Chimica Acta 612(2), 121–125. https://doi.org/10.1016/j.aca.2008.02.032 — extends the MC-UVE procedure of Cai et al. (2008) (`stability_select`) by aggregating independent MC-UVE rounds through a vote rule.

### Mathematical principle

Run multiple independent MC-UVE rounds with different seeds, threshold each independently, then **vote** across rounds: a feature is selected if it survives thresholding in a majority of rounds. Robust against single-round instability caused by particular bootstrap samples.

The voting rule has a free parameter (majority threshold); the default of $\lceil R/2 \rceil$ is the median-style majority. Stricter thresholds give smaller but more reliable subsets.

### Appropriate uses

Combining Monte Carlo coefficient stability with UVE-style elimination for wavelength screening.

### Limits and validation

Noise-variable and resampling choices set the effective cutoff; stochastic stability is not evidence of mechanistic relevance.

### Implementation

`n4m_feature_selection_emcuve_select`.

### Sources and provenance

Current implementation: [cpp/src/core/emcuve_selection.cpp](https://github.com/GBeurier/nirs4all-methods/blob/main/cpp/src/core/emcuve_selection.cpp).


### API and bindings

**C ABI (ABI 2):** [`n4m_feature_selection_emcuve_select`](https://github.com/GBeurier/nirs4all-methods/blob/main/cpp/include/n4m/feature_selection.h#L289). Use the linked public header for the exact signature, configuration, and result handles.

**Python (verified public re-export):**

```python
from n4m.feature_selection.wrapper import EMCUVE
```

Source signature: `EMCUVE(*, n_components: int = 2, noise_features: int = 50, noise_seed: int = 0, n_ensembles: int = 10, vote_threshold: float = 0.5, n_folds: int = 3)` ([`n4m/_impl/selection.py`](https://github.com/GBeurier/nirs4all-methods/blob/main/bindings/python/src/n4m/_impl/selection.py#L674)).

**R (source-verified):** [`emcuve_select(X, Y, n_components, noise_features = NULL, noise_seed = 0L, n_ensembles = 5L, vote_threshold = 0.5)`](https://github.com/GBeurier/nirs4all-methods/blob/main/bindings/r/n4m/R/methods_extra.R).

The source signature has additional required inputs, so no example call is fabricated.

**MATLAB / Octave (source-verified):** [`emcuve_select(X, Y, n_components, noise_features, noise_seed, ... n_ensembles, vote_threshold)`](https://github.com/GBeurier/nirs4all-methods/blob/main/bindings/matlab/+n4m/emcuve_select.m).

The source signature has additional required inputs, so no example call is fabricated.

**Registry parity references** 📐

:::{card}
:class-card: external-refs

- 📐 **`ref.r_plsvarsel`** (R · r) — `plsVarSel` 0.10.0 · strict (rmse_rel ≤ 1e-06) — R `plsVarSel::mcuve_pls` repeated N times with different seeds, then vote-aggregated. Same algorithm family as pls4all's EMCUVE. RNGs differ; mask metric ~0=perfect.

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
<thead><tr><th scope="col">Backend</th><th scope="col">Parity</th><th class="size-col" scope="col">200×40 (ms)</th></tr></thead>
<tbody class="lang-band lang-cpp"><tr class="lang-band-row" data-lang="cpp"><th colspan="3" scope="rowgroup"><span class="lang-band-dot"></span>C++ native · libn4m</th></tr>
<tr class="bk-row"><td class="bk-name"><code>pls4all.cpp.blas+omp</code></td><td class="parity parity-exact">✓ J 1.00</td><td class="ms">1.3 s</td></tr>
</tbody>
<tbody class="lang-band lang-python"><tr class="lang-band-row" data-lang="python"><th colspan="3" scope="rowgroup"><span class="lang-band-dot"></span>Python · archived pls4all benchmark</th></tr>
<tr class="bk-row"><td class="bk-name"><code>pls4all.python</code></td><td class="parity parity-exact">✓ J 1.00</td><td class="ms">1.3 s</td></tr>
<tr class="bk-row"><td class="bk-name"><code>pls4all.sklearn</code></td><td class="parity parity-exact">✓ J 1.00</td><td class="ms ms-best">1.98 ms<span class="medal" title="fastest">🏆</span></td></tr>
</tbody>
<tbody class="lang-band lang-r"><tr class="lang-band-row" data-lang="r"><th colspan="3" scope="rowgroup"><span class="lang-band-dot"></span>R · archived pls4all benchmark</th></tr>
<tr class="bk-row"><td class="bk-name"><code>pls4all.R</code></td><td class="parity parity-exact">✓ J 1.00</td><td class="ms">4.51 ms</td></tr>
<tr class="bk-row"><td class="bk-name"><code>pls4all.R.formula</code></td><td class="parity parity-exact">✓ J 1.00</td><td class="ms">5.44 ms</td></tr>
<tr class="bk-row"><td class="bk-name"><code>pls4all.R.mdatools</code></td><td class="parity parity-exact">✓ J 1.00</td><td class="ms">5.25 ms</td></tr>
<tr class="bk-row"><td class="bk-name"><code>pls4all.R.pls</code></td><td class="parity parity-exact">✓ J 1.00</td><td class="ms">5.57 ms</td></tr>
</tbody>
<tbody class="lang-band lang-r"><tr class="lang-band-row" data-lang="r"><th colspan="3" scope="rowgroup"><span class="lang-band-dot"></span>R · external</th></tr>
<tr class="bk-row truth-source truth-source-strict"><td class="bk-name"><span class="truth-mark" title="Registry parity reference (r): plsVarSel 0.10.0 — strict (rmse_rel ≤ 1e-06)">📐</span><code>ref.r_plsvarsel</code></td><td class="parity parity-ref-source">source</td><td class="ms">922.4 ms</td></tr>
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
<tr class="bk-row"><td class="bk-name"><code>pls4all.cpp.blas+omp</code></td><td class="parity parity-exact">✓ J 1.00</td><td class="ms">1.3 s</td></tr>
</tbody>
<tbody class="lang-band lang-python"><tr class="lang-band-row" data-lang="python"><th colspan="3" scope="rowgroup"><span class="lang-band-dot"></span>Python · archived pls4all benchmark</th></tr>
<tr class="bk-row"><td class="bk-name"><code>pls4all.python</code></td><td class="parity parity-exact">✓ J 1.00</td><td class="ms">1.3 s</td></tr>
<tr class="bk-row"><td class="bk-name"><code>pls4all.sklearn</code></td><td class="parity parity-exact">✓ J 1.00</td><td class="ms ms-best">2.00 ms<span class="medal" title="fastest">🏆</span></td></tr>
</tbody>
<tbody class="lang-band lang-r"><tr class="lang-band-row" data-lang="r"><th colspan="3" scope="rowgroup"><span class="lang-band-dot"></span>R · archived pls4all benchmark</th></tr>
<tr class="bk-row"><td class="bk-name"><code>pls4all.R</code></td><td class="parity parity-exact">✓ J 1.00</td><td class="ms">4.69 ms</td></tr>
<tr class="bk-row"><td class="bk-name"><code>pls4all.R.formula</code></td><td class="parity parity-exact">✓ J 1.00</td><td class="ms">5.25 ms</td></tr>
<tr class="bk-row"><td class="bk-name"><code>pls4all.R.mdatools</code></td><td class="parity parity-exact">✓ J 1.00</td><td class="ms">5.99 ms</td></tr>
<tr class="bk-row"><td class="bk-name"><code>pls4all.R.pls</code></td><td class="parity parity-exact">✓ J 1.00</td><td class="ms">6.05 ms</td></tr>
</tbody>
<tbody class="lang-band lang-r"><tr class="lang-band-row" data-lang="r"><th colspan="3" scope="rowgroup"><span class="lang-band-dot"></span>R · external</th></tr>
<tr class="bk-row truth-source truth-source-strict"><td class="bk-name"><span class="truth-mark" title="Registry parity reference (r): plsVarSel 0.10.0 — strict (rmse_rel ≤ 1e-06)">📐</span><code>ref.r_plsvarsel</code></td><td class="parity parity-ref-source">source</td><td class="ms">918.1 ms</td></tr>
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
<tr class="bk-row"><td class="bk-name"><code>pls4all.cpp.blas+omp</code></td><td class="parity parity-exact">✓ J 1.00</td><td class="ms">1.3 s</td></tr>
</tbody>
<tbody class="lang-band lang-python"><tr class="lang-band-row" data-lang="python"><th colspan="3" scope="rowgroup"><span class="lang-band-dot"></span>Python · archived pls4all benchmark</th></tr>
<tr class="bk-row"><td class="bk-name"><code>pls4all.python</code></td><td class="parity parity-exact">✓ J 1.00</td><td class="ms">1.3 s</td></tr>
<tr class="bk-row"><td class="bk-name"><code>pls4all.sklearn</code></td><td class="parity parity-exact">✓ J 1.00</td><td class="ms ms-best">1.90 ms<span class="medal" title="fastest">🏆</span></td></tr>
</tbody>
<tbody class="lang-band lang-r"><tr class="lang-band-row" data-lang="r"><th colspan="3" scope="rowgroup"><span class="lang-band-dot"></span>R · archived pls4all benchmark</th></tr>
<tr class="bk-row"><td class="bk-name"><code>pls4all.R</code></td><td class="parity parity-exact">✓ J 1.00</td><td class="ms">4.22 ms</td></tr>
<tr class="bk-row"><td class="bk-name"><code>pls4all.R.formula</code></td><td class="parity parity-exact">✓ J 1.00</td><td class="ms">5.03 ms</td></tr>
<tr class="bk-row"><td class="bk-name"><code>pls4all.R.mdatools</code></td><td class="parity parity-exact">✓ J 1.00</td><td class="ms">4.95 ms</td></tr>
<tr class="bk-row"><td class="bk-name"><code>pls4all.R.pls</code></td><td class="parity parity-exact">✓ J 1.00</td><td class="ms">5.01 ms</td></tr>
</tbody>
<tbody class="lang-band lang-r"><tr class="lang-band-row" data-lang="r"><th colspan="3" scope="rowgroup"><span class="lang-band-dot"></span>R · external</th></tr>
<tr class="bk-row truth-source truth-source-strict"><td class="bk-name"><span class="truth-mark" title="Registry parity reference (r): plsVarSel 0.10.0 — strict (rmse_rel ≤ 1e-06)">📐</span><code>ref.r_plsvarsel</code></td><td class="parity parity-ref-source">source</td><td class="ms">907.4 ms</td></tr>
</tbody>
</table>
</div>

:::

::::


---

_See also_: [benchmark overview](../benchmarks/overview.md) · [methods index](index.md) · [interactive dashboard](../landing/dashboard.md)