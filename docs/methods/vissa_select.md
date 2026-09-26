# `vissa_select` — VISSA — Variable Iterative Space-Shrinkage

_Group_: **Variable selector** · _Registry tolerance_: `1e-06`

## Description

VISSA-PLS — Variable Iterative Space Shrinkage (§49)

> **Registry note** — Python `auswahl.VISSA 0.9.0` (LSX-UniWue) — canonical Deng 2014 implementation via weighted binary matrix sampling. Default `_vissa_select_pls4all` path mirrors the same auswahl call with seed=11, giving bit-exact mask parity. The C++ splitmix64 VISSA kernel is opt-in via `legacy=True`.

### Parameters

| Name | Type | Default | Notes |
|------|------|---------|-------|
| `n_components` | `int` | `2` | current public binding signature |
| `n_iterations` | `int` | `10` | current public binding signature |
| `n_submodels` | `int` | `60` | current public binding signature |
| `ratio_kept` | `float` | `0.1` | current public binding signature |
| `threshold` | `float` | `0.5` | current public binding signature |
| `floor_probability` | `float` | `0.05` | current public binding signature |
| `n_folds` | `int` | `3` | current public binding signature |
| `seed` | `int` | `0` | current public binding signature |

## Explanations

### Bibliographic source

Deng, B. C., Yun, Y. H., Liang, Y. Z. & Yi, L. Z. (2015). *A new strategy to prevent over-fitting in partial least squares models based on model population analysis*. Analytica Chimica Acta 880, 32--41. DOI [10.1016/j.aca.2015.04.045](https://doi.org/10.1016/j.aca.2015.04.045).

### Mathematical principle

VISSA evaluates a **population of random subsets** of the same size, refines the population by selecting the best by CV-RMSE, and iteratively shrinks the search space toward features that survive in many high-performing subsets. Features that appear in many top subsets are deemed important; the search converges to a consensus subset.

Different from CARS in that the search space is shrunken **by consensus over a population** rather than by exponential decay over iterations. This gives smoother convergence and less sensitivity to single high-leverage subsets.

### Appropriate uses

Iteratively concentrating weighted sampling on wavelengths associated with better PLS subsets.

### Limits and validation

Sampling probabilities are path-dependent and stochastic; reported subsets need independent assessment.

### Implementation

`n4m_feature_selection_vissa_select`.

### Sources and provenance

Current implementation: [cpp/src/core/vissa_selection.cpp](https://github.com/GBeurier/nirs4all-methods/blob/main/cpp/src/core/vissa_selection.cpp).


### API and bindings

**C ABI (ABI 2):** [`n4m_feature_selection_vissa_select`](https://github.com/GBeurier/nirs4all-methods/blob/main/cpp/include/n4m/feature_selection.h#L216). Use the linked public header for the exact signature, configuration, and result handles.

**Python (verified public re-export):**

```python
from n4m.feature_selection.wrapper import VISSA
```

Source signature: `VISSA(*, n_components: int = 2, n_iterations: int = 10, n_submodels: int = 60, ratio_kept: float = 0.1, threshold: float = 0.5, floor_probability: float = 0.05, n_folds: int = 3, seed: int = 0)` ([`n4m/_impl/selection.py`](https://github.com/GBeurier/nirs4all-methods/blob/main/bindings/python/src/n4m/_impl/selection.py#L365)).

**R (source-verified):** [`vissa_select(X, Y, n_components, n_iterations = 20L, n_submodels = 100L, ratio_kept = 0.1, threshold = 0.5, floor_probability = 0.01, seed = 0L)`](https://github.com/GBeurier/nirs4all-methods/blob/main/bindings/r/n4m/R/methods_extra.R).

The source signature has additional required inputs, so no example call is fabricated.

**MATLAB / Octave (source-verified):** [`vissa_select(X, Y, n_components, n_iterations, n_submodels, ... ratio_kept, threshold, floor_probability, seed)`](https://github.com/GBeurier/nirs4all-methods/blob/main/bindings/matlab/+n4m/vissa_select.m).

The source signature has additional required inputs, so no example call is fabricated.

**Registry parity references** 📐

:::{card}
:class-card: external-refs

- 📐 **`ref.python_auswahl`** (python · python) — `auswahl` 0.9.0 · strict (rmse_rel ≤ 1e-06) — Python `auswahl.VISSA` from LSX-UniWue with deterministic seed=11; the pls4all default path calls the same helper with the same seed, so masks coincide bit-for-bit. The C++ splitmix64 VISSA kernel is opt-in via legacy=True.

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
<thead><tr><th scope="col">Backend</th><th scope="col">Parity</th><th class="size-col" scope="col">80×25 (ms)</th></tr></thead>
<tbody class="lang-band lang-python"><tr class="lang-band-row" data-lang="python"><th colspan="3" scope="rowgroup"><span class="lang-band-dot"></span>Python · archived pls4all benchmark</th></tr>
<tr class="bk-row"><td class="bk-name"><code>pls4all.sklearn</code></td><td class="parity parity-exact">✓ J 0.60</td><td class="ms ms-best">10.8 ms<span class="medal" title="fastest">🏆</span></td></tr>
</tbody>
<tbody class="lang-band lang-r"><tr class="lang-band-row" data-lang="r"><th colspan="3" scope="rowgroup"><span class="lang-band-dot"></span>R · archived pls4all benchmark</th></tr>
<tr class="bk-row"><td class="bk-name"><code>pls4all.R</code></td><td class="parity parity-exact">✓ J 0.60</td><td class="ms">11.2 ms</td></tr>
<tr class="bk-row"><td class="bk-name"><code>pls4all.R.formula</code></td><td class="parity parity-exact">✓ J 0.60</td><td class="ms">11.1 ms</td></tr>
<tr class="bk-row"><td class="bk-name"><code>pls4all.R.mdatools</code></td><td class="parity parity-exact">✓ J 0.60</td><td class="ms">49.1 ms</td></tr>
<tr class="bk-row"><td class="bk-name"><code>pls4all.R.pls</code></td><td class="parity parity-exact">✓ J 0.60</td><td class="ms">21.4 ms</td></tr>
</tbody>
</table>
</div>

:::

:::{tab-item} 3 threads
:sync: threads-3

<div class="parity-table-wrap">
<table class="docutils parity-grouped">
<thead><tr><th scope="col">Backend</th><th scope="col">Parity</th><th class="size-col" scope="col">80×25 (ms)</th></tr></thead>
<tbody class="lang-band lang-python"><tr class="lang-band-row" data-lang="python"><th colspan="3" scope="rowgroup"><span class="lang-band-dot"></span>Python · archived pls4all benchmark</th></tr>
<tr class="bk-row"><td class="bk-name"><code>pls4all.sklearn</code></td><td class="parity parity-exact">✓ J 0.60</td><td class="ms">13.0 ms</td></tr>
</tbody>
<tbody class="lang-band lang-r"><tr class="lang-band-row" data-lang="r"><th colspan="3" scope="rowgroup"><span class="lang-band-dot"></span>R · archived pls4all benchmark</th></tr>
<tr class="bk-row"><td class="bk-name"><code>pls4all.R</code></td><td class="parity parity-exact">✓ J 0.60</td><td class="ms">13.9 ms</td></tr>
<tr class="bk-row"><td class="bk-name"><code>pls4all.R.formula</code></td><td class="parity parity-exact">✓ J 0.60</td><td class="ms">14.6 ms</td></tr>
<tr class="bk-row"><td class="bk-name"><code>pls4all.R.mdatools</code></td><td class="parity parity-exact">✓ J 0.60</td><td class="ms ms-best">11.4 ms<span class="medal" title="fastest">🏆</span></td></tr>
<tr class="bk-row"><td class="bk-name"><code>pls4all.R.pls</code></td><td class="parity parity-exact">✓ J 0.60</td><td class="ms">11.4 ms</td></tr>
</tbody>
</table>
</div>

:::

:::{tab-item} 10 threads
:sync: threads-10

<div class="parity-table-wrap">
<table class="docutils parity-grouped">
<thead><tr><th scope="col">Backend</th><th scope="col">Parity</th><th class="size-col" scope="col">80×25 (ms)</th></tr></thead>
<tbody class="lang-band lang-python"><tr class="lang-band-row" data-lang="python"><th colspan="3" scope="rowgroup"><span class="lang-band-dot"></span>Python · archived pls4all benchmark</th></tr>
<tr class="bk-row"><td class="bk-name"><code>pls4all.sklearn</code></td><td class="parity parity-exact">✓ J 0.60</td><td class="ms ms-best">8.85 ms<span class="medal" title="fastest">🏆</span></td></tr>
</tbody>
<tbody class="lang-band lang-r"><tr class="lang-band-row" data-lang="r"><th colspan="3" scope="rowgroup"><span class="lang-band-dot"></span>R · archived pls4all benchmark</th></tr>
<tr class="bk-row"><td class="bk-name"><code>pls4all.R</code></td><td class="parity parity-exact">✓ J 0.60</td><td class="ms">11.8 ms</td></tr>
<tr class="bk-row"><td class="bk-name"><code>pls4all.R.formula</code></td><td class="parity parity-exact">✓ J 0.60</td><td class="ms">12.5 ms</td></tr>
<tr class="bk-row"><td class="bk-name"><code>pls4all.R.mdatools</code></td><td class="parity parity-exact">✓ J 0.60</td><td class="ms">13.1 ms</td></tr>
<tr class="bk-row"><td class="bk-name"><code>pls4all.R.pls</code></td><td class="parity parity-exact">✓ J 0.60</td><td class="ms">11.2 ms</td></tr>
</tbody>
</table>
</div>

:::

::::


---

_See also_: [benchmark overview](../benchmarks/overview.md) · [methods index](index.md) · [interactive dashboard](../landing/dashboard.md)