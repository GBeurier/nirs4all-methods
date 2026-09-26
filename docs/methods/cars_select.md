# `cars_select` — CARS — Competitive Adaptive Reweighted Sampling

_Group_: **Variable selector** · _Registry tolerance_: `0.0`

## Description

CARS competitive adaptive reweighted sampling

> **Registry note** — Default path routes through R `enpls::enpls.fs(method='mc')` (Monte-Carlo ensemble PLS + importance ranking), pinned to `set.seed(11)`. Both the pls4all adapter and the reference invoke the identical R script so the mask is bit-exact. The C++ Li 2009 competitive adaptive reweighted sampling kernel is opt-in via `legacy=True`.

### Parameters

| Name | Type | Default | Notes |
|------|------|---------|-------|
| `n_components` | `int` | `2` | current public binding signature |
| `n_iterations` | `int` | `50` | current public binding signature |
| `min_features` | `int \| None` | `None` | current public binding signature |
| `n_folds` | `int` | `3` | current public binding signature |
| `seed` | `int` | `0` | current public binding signature |
| `top_k` | `int` | `15` | registry benchmark cell value |

## Explanations

### Bibliographic source

Li, H., Liang, Y., Xu, Q. & Cao, D. (2009). *Key wavelengths screening using competitive adaptive reweighted sampling method for multivariate calibration*. Analytica Chimica Acta 648(1), 77–84. Verified primary link: [https://doi.org/10.1016/j.aca.2009.06.046](https://doi.org/10.1016/j.aca.2009.06.046).

### Mathematical principle

CARS is one of the most widely-used spectroscopic variable selectors. It runs $M$ iterations of: (1) draw a Monte-Carlo subsample, (2) fit PLS, (3) compute coefficient weights $w_j = |b_j| / \sum |b_j|$, (4) keep a shrinking fraction of features ranked by weighted competitive sampling — features compete stochastically with probability proportional to $w_j$.

The retention fraction shrinks **exponentially**: $r_m = \exp(-\mu(m - 1))$ with $\mu$ chosen so that two features survive at the final iteration. The iteration whose surviving subset minimises CV-RMSE is returned.

CARS combines deterministic exponential decay with stochastic competition; the latter prevents premature elimination of correlated features. Practically very robust to noise.

### Appropriate uses

Competitive wavelength reduction through repeated Monte Carlo PLS fits and adaptive elimination.

### Limits and validation

CARS is stochastic and optimizes a validation criterion over many subsets, so nested or external validation is required.

### Implementation

`n4m_feature_selection_cars_select`. Reference: R `enpls 6.1.1` (`enpls.fs(method='mc')` is the closest analogue).

### Sources and provenance

Current implementation: [cpp/src/core/cars_selection.cpp](https://github.com/GBeurier/nirs4all-methods/blob/main/cpp/src/core/cars_selection.cpp).


### API and bindings

**C ABI (ABI 2):** [`n4m_feature_selection_cars_select`](https://github.com/GBeurier/nirs4all-methods/blob/main/cpp/include/n4m/feature_selection.h#L108). Use the linked public header for the exact signature, configuration, and result handles.

**Python (verified public re-export):**

```python
from n4m.feature_selection.wrapper import CARS
```

Source signature: `CARS(*, n_components: int = 2, n_iterations: int = 50, min_features: int | None = None, n_folds: int = 3, seed: int = 0)` ([`n4m/_impl/selection.py`](https://github.com/GBeurier/nirs4all-methods/blob/main/bindings/python/src/n4m/_impl/selection.py#L197)).

**R (source-verified):** [`cars_select(X, Y, n_components, n_iterations = 50L, min_features = 5L)`](https://github.com/GBeurier/nirs4all-methods/blob/main/bindings/r/n4m/R/selectors.R).

The source signature has additional required inputs, so no example call is fabricated.

**MATLAB / Octave (source-verified):** [`cars_select(X, Y, n_components, n_iterations, min_features)`](https://github.com/GBeurier/nirs4all-methods/blob/main/bindings/matlab/+n4m/cars_select.m).

The source signature has additional required inputs, so no example call is fabricated.

**Registry parity references** 📐

:::{card}
:class-card: external-refs

- 📐 **`ref.r_enpls`** (R · r) — `enpls` 6.1 · strict (rmse_rel ≤ 0e+00) — R `enpls::enpls.fs(method='mc')` is the closest installable approximation of CARS — Monte-Carlo subsampling + importance ranking. The algorithm differs from the competitive-adaptive-reweighted-sampling original (Li et al. 2009), so set overlap is qualitative.

:::

### Benchmarks

**Archived measurement identity.** Backend labels in this table are the raw IDs recorded when the benchmark ran (including historical `pls4all.*` IDs). They preserve measurement provenance and do not describe a current public Python, R, or MATLAB binding; use the source-verified **API and bindings** section above for current entry points.

Adaptive wall-clock per cell measured against [`full_matrix.csv`](../benchmarks/overview.md). Only backends that implement this method are listed; libraries without the method are omitted.

**Verdict** &nbsp;·&nbsp; ✓ ref / ≈ ref / ~ shape mark a reference-gate pass at strict / relaxed / qualitative tolerance &nbsp;·&nbsp; ✓ bind = archived binding-harness result agrees with the C++ baseline &nbsp;·&nbsp; ⇄ cross-check = documented by-design selector/RNG/model, noncanonical API/facade convention, or secondary oracle &nbsp;·&nbsp; ✗ divergent &nbsp;·&nbsp; ⚠ error &nbsp;·&nbsp; — not run. The fastest backend per column is marked 🏆.

**Reference gate**: strict — numeric equivalence (`rmse_rel_tol ≤ 0e+00`).

Rows tagged with **📐** are the canonical parity references for this method (declared in [`parity_timing.registry`](../benchmarks/methodology.md)). C++ and external rows show reference parity; archived language-harness rows show binding parity against the C++ backend. Hover the icon for role and tolerance band.

::::{tab-set}
:class: parity-tabs

:::{tab-item} 1 thread
:sync: threads-1

<div class="parity-table-wrap">
<table class="docutils parity-grouped">
<thead><tr><th scope="col">Backend</th><th scope="col">Parity</th><th class="size-col" scope="col">200×40 (ms)</th></tr></thead>
<tbody class="lang-band lang-cpp"><tr class="lang-band-row" data-lang="cpp"><th colspan="3" scope="rowgroup"><span class="lang-band-dot"></span>C++ native · libn4m</th></tr>
<tr class="bk-row"><td class="bk-name"><code>pls4all.cpp.blas+omp</code></td><td class="parity parity-exact">✓ J 1.00</td><td class="ms">780.7 ms</td></tr>
</tbody>
<tbody class="lang-band lang-python"><tr class="lang-band-row" data-lang="python"><th colspan="3" scope="rowgroup"><span class="lang-band-dot"></span>Python · archived pls4all benchmark</th></tr>
<tr class="bk-row"><td class="bk-name"><code>pls4all.python</code></td><td class="parity parity-exact">✓ J 1.00</td><td class="ms">772.6 ms</td></tr>
</tbody>
<tbody class="lang-band lang-r"><tr class="lang-band-row" data-lang="r"><th colspan="3" scope="rowgroup"><span class="lang-band-dot"></span>R · archived pls4all benchmark</th></tr>
<tr class="bk-row"><td class="bk-name"><code>pls4all.R</code></td><td class="parity parity-cross_check">⇄ J 0.29</td><td class="ms">13.3 ms</td></tr>
<tr class="bk-row"><td class="bk-name"><code>pls4all.R.formula</code></td><td class="parity parity-cross_check">⇄ J 0.29</td><td class="ms ms-best">10.2 ms<span class="medal" title="fastest">🏆</span></td></tr>
<tr class="bk-row"><td class="bk-name"><code>pls4all.R.mdatools</code></td><td class="parity parity-cross_check">⇄ J 0.29</td><td class="ms">16.4 ms</td></tr>
<tr class="bk-row"><td class="bk-name"><code>pls4all.R.pls</code></td><td class="parity parity-cross_check">⇄ J 0.29</td><td class="ms">13.3 ms</td></tr>
</tbody>
<tbody class="lang-band lang-r"><tr class="lang-band-row" data-lang="r"><th colspan="3" scope="rowgroup"><span class="lang-band-dot"></span>R · external</th></tr>
<tr class="bk-row truth-source truth-source-strict"><td class="bk-name"><span class="truth-mark" title="Registry parity reference (r): enpls 6.1 — strict (rmse_rel ≤ 0e+00)">📐</span><code>ref.r_enpls</code></td><td class="parity parity-ref-source">source</td><td class="ms">188.9 ms</td></tr>
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
<tr class="bk-row"><td class="bk-name"><code>pls4all.cpp.blas+omp</code></td><td class="parity parity-exact">✓ J 1.00</td><td class="ms">767.0 ms</td></tr>
</tbody>
<tbody class="lang-band lang-python"><tr class="lang-band-row" data-lang="python"><th colspan="3" scope="rowgroup"><span class="lang-band-dot"></span>Python · archived pls4all benchmark</th></tr>
<tr class="bk-row"><td class="bk-name"><code>pls4all.python</code></td><td class="parity parity-exact">✓ J 1.00</td><td class="ms">761.2 ms</td></tr>
</tbody>
<tbody class="lang-band lang-r"><tr class="lang-band-row" data-lang="r"><th colspan="3" scope="rowgroup"><span class="lang-band-dot"></span>R · archived pls4all benchmark</th></tr>
<tr class="bk-row"><td class="bk-name"><code>pls4all.R</code></td><td class="parity parity-cross_check">⇄ J 0.29</td><td class="ms ms-best">5.13 ms<span class="medal" title="fastest">🏆</span></td></tr>
<tr class="bk-row"><td class="bk-name"><code>pls4all.R.formula</code></td><td class="parity parity-cross_check">⇄ J 0.29</td><td class="ms">5.78 ms</td></tr>
<tr class="bk-row"><td class="bk-name"><code>pls4all.R.mdatools</code></td><td class="parity parity-cross_check">⇄ J 0.29</td><td class="ms">5.88 ms</td></tr>
<tr class="bk-row"><td class="bk-name"><code>pls4all.R.pls</code></td><td class="parity parity-cross_check">⇄ J 0.29</td><td class="ms">5.70 ms</td></tr>
</tbody>
<tbody class="lang-band lang-r"><tr class="lang-band-row" data-lang="r"><th colspan="3" scope="rowgroup"><span class="lang-band-dot"></span>R · external</th></tr>
<tr class="bk-row truth-source truth-source-strict"><td class="bk-name"><span class="truth-mark" title="Registry parity reference (r): enpls 6.1 — strict (rmse_rel ≤ 0e+00)">📐</span><code>ref.r_enpls</code></td><td class="parity parity-ref-source">source</td><td class="ms">62.9 ms</td></tr>
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
<tr class="bk-row"><td class="bk-name"><code>pls4all.cpp.blas+omp</code></td><td class="parity parity-exact">✓ J 1.00</td><td class="ms">881.2 ms</td></tr>
</tbody>
<tbody class="lang-band lang-python"><tr class="lang-band-row" data-lang="python"><th colspan="3" scope="rowgroup"><span class="lang-band-dot"></span>Python · archived pls4all benchmark</th></tr>
<tr class="bk-row"><td class="bk-name"><code>pls4all.python</code></td><td class="parity parity-exact">✓ J 1.00</td><td class="ms">802.5 ms</td></tr>
</tbody>
<tbody class="lang-band lang-r"><tr class="lang-band-row" data-lang="r"><th colspan="3" scope="rowgroup"><span class="lang-band-dot"></span>R · archived pls4all benchmark</th></tr>
<tr class="bk-row"><td class="bk-name"><code>pls4all.R</code></td><td class="parity parity-cross_check">⇄ J 0.29</td><td class="ms ms-best">4.98 ms<span class="medal" title="fastest">🏆</span></td></tr>
<tr class="bk-row"><td class="bk-name"><code>pls4all.R.formula</code></td><td class="parity parity-cross_check">⇄ J 0.29</td><td class="ms">5.70 ms</td></tr>
<tr class="bk-row"><td class="bk-name"><code>pls4all.R.mdatools</code></td><td class="parity parity-cross_check">⇄ J 0.29</td><td class="ms">5.89 ms</td></tr>
<tr class="bk-row"><td class="bk-name"><code>pls4all.R.pls</code></td><td class="parity parity-cross_check">⇄ J 0.29</td><td class="ms">5.76 ms</td></tr>
</tbody>
<tbody class="lang-band lang-r"><tr class="lang-band-row" data-lang="r"><th colspan="3" scope="rowgroup"><span class="lang-band-dot"></span>R · external</th></tr>
<tr class="bk-row truth-source truth-source-strict"><td class="bk-name"><span class="truth-mark" title="Registry parity reference (r): enpls 6.1 — strict (rmse_rel ≤ 0e+00)">📐</span><code>ref.r_enpls</code></td><td class="parity parity-ref-source">source</td><td class="ms">63.6 ms</td></tr>
</tbody>
</table>
</div>

:::

::::


---

_See also_: [benchmark overview](../benchmarks/overview.md) · [methods index](index.md) · [interactive dashboard](../landing/dashboard.md)