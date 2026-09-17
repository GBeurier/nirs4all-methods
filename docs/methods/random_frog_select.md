# `random_frog_select` — Random Frog

_Group_: **Variable selector** · _Registry tolerance_: `1e-06`

## Description

Random Frog selection (§18 Phase 5g)

> **Registry note** — Python `auswahl.RandomFrog` (LSX-UniWue; Li 2012). Same algorithm as libPLS `randomfrog_pls`. Default `_random_frog_select_pls4all` path mirrors the same auswahl call with `random_state=seed`, giving bit-exact mask parity. The C++ splitmix64 kernel is opt-in via `legacy=True`.

### Parameters

| Name | Type | Default | Notes |
|------|------|---------|-------|
| `top_k` | `int` | `required` | current public binding signature |
| `n_components` | `int` | `2` | current public binding signature |
| `n_iterations` | `int` | `100` | current public binding signature |
| `initial_size` | `int` | `20` | current public binding signature |
| `min_size` | `int \| None` | `None` | current public binding signature |
| `max_size` | `int \| None` | `None` | current public binding signature |
| `n_folds` | `int` | `3` | current public binding signature |
| `seed` | `int` | `0` | current public binding signature |

## Explanations

### Bibliographic source

Li, H., Xu, Q. & Liang, Y. (2012). *Random frog: an efficient reversible jump Markov chain Monte Carlo-like approach for variable selection*. Analytica Chimica Acta 740, 20–26. Verified primary link: [https://doi.org/10.1016/j.aca.2012.06.031](https://doi.org/10.1016/j.aca.2012.06.031).

### Mathematical principle

Random Frog runs a random walk over feature subsets: at each step it proposes a transition to a neighbouring subset (add / remove / swap a feature) and accepts the transition with a Metropolis-style probability based on the improvement in CV-RMSE. Features that appear frequently in the visited subsets are deemed important.

Output: the **inclusion frequency** vector — fraction of iterations in which each feature was selected. Sort by frequency and take the top-$k$ for the final subset.

Random Frog is sample-efficient compared to GA-PLS (no population of full subsets to maintain) but slower to mix on very high-dimensional data. Recommended on spectra of moderate size (a few hundred wavelengths).

### Appropriate uses

Sampling variable subsets to estimate inclusion probabilities under a PLS validation objective.

### Limits and validation

Finite-chain inclusion frequencies depend on initialization, proposal settings, and seed; they are not Bayesian posterior probabilities.

### Implementation

`n4m_feature_selection_random_frog_select`.

### Sources and provenance

Current implementation: [cpp/src/core/random_frog_selection.cpp](https://github.com/GBeurier/nirs4all-methods/blob/main/cpp/src/core/random_frog_selection.cpp).


### API and bindings

**C ABI (ABI 2):** [`n4m_feature_selection_random_frog_select`](https://github.com/GBeurier/nirs4all-methods/blob/main/cpp/include/n4m/feature_selection.h#L119). Use the linked public header for the exact signature, configuration, and result handles.

**Python (verified public re-export):**

```python
from n4m.feature_selection.wrapper import RandomFrog
```

Source signature: `RandomFrog(top_k: int, *, n_components: int = 2, n_iterations: int = 100, initial_size: int = 20, min_size: int | None = None, max_size: int | None = None, n_folds: int = 3, seed: int = 0)` ([`n4m/_impl/selection.py`](https://github.com/GBeurier/nirs4all-methods/blob/main/bindings/python/src/n4m/_impl/selection.py#L223)).

**R (source-verified):** [`random_frog_select(X, Y, n_components, n_iterations = 100L, initial_size = 30L, min_size = NULL, max_size = NULL, top_k = 10L, seed = 0L)`](https://github.com/GBeurier/nirs4all-methods/blob/main/bindings/r/n4m/R/methods_extra.R).

The source signature has additional required inputs, so no example call is fabricated.

**MATLAB / Octave (source-verified):** [`random_frog_select(X, Y, n_components, n_iterations, ... initial_size, min_size, max_size, top_k, seed)`](https://github.com/GBeurier/nirs4all-methods/blob/main/bindings/matlab/+n4m/random_frog_select.m).

The source signature has additional required inputs, so no example call is fabricated.

**Registry parity references** 📐

:::{card}
:class-card: external-refs

- 📐 **`ref.python_auswahl`** (python · python) — `auswahl` 0.9.0 · strict (rmse_rel ≤ 1e-06) — Python `auswahl.RandomFrog` (LSX-UniWue; Li 2012). Same algorithm as libPLS `randomfrog_pls` with pinned `random_state` for bit-exact mask parity.

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
<tbody class="lang-band lang-python"><tr class="lang-band-row" data-lang="python"><th colspan="3" scope="rowgroup"><span class="lang-band-dot"></span>Python · archived pls4all benchmark</th></tr>
<tr class="bk-row"><td class="bk-name"><code>pls4all.sklearn</code></td><td class="parity parity-exact">✓ J 0.18</td><td class="ms ms-best">2.19 ms<span class="medal" title="fastest">🏆</span></td></tr>
</tbody>
<tbody class="lang-band lang-r"><tr class="lang-band-row" data-lang="r"><th colspan="3" scope="rowgroup"><span class="lang-band-dot"></span>R · archived pls4all benchmark</th></tr>
<tr class="bk-row"><td class="bk-name"><code>pls4all.R</code></td><td class="parity parity-exact">✓ J 0.18</td><td class="ms">4.98 ms</td></tr>
<tr class="bk-row"><td class="bk-name"><code>pls4all.R.formula</code></td><td class="parity parity-exact">✓ J 0.18</td><td class="ms">5.67 ms</td></tr>
<tr class="bk-row"><td class="bk-name"><code>pls4all.R.mdatools</code></td><td class="parity parity-exact">✓ J 0.18</td><td class="ms">5.96 ms</td></tr>
<tr class="bk-row"><td class="bk-name"><code>pls4all.R.pls</code></td><td class="parity parity-exact">✓ J 0.18</td><td class="ms">5.94 ms</td></tr>
</tbody>
</table>
</div>

:::

:::{tab-item} 3 threads
:sync: threads-3

<div class="parity-table-wrap">
<table class="docutils parity-grouped">
<thead><tr><th scope="col">Backend</th><th scope="col">Parity</th><th class="size-col" scope="col">200×40 (ms)</th></tr></thead>
<tbody class="lang-band lang-python"><tr class="lang-band-row" data-lang="python"><th colspan="3" scope="rowgroup"><span class="lang-band-dot"></span>Python · archived pls4all benchmark</th></tr>
<tr class="bk-row"><td class="bk-name"><code>pls4all.sklearn</code></td><td class="parity parity-exact">✓ J 0.18</td><td class="ms ms-best">2.19 ms<span class="medal" title="fastest">🏆</span></td></tr>
</tbody>
<tbody class="lang-band lang-r"><tr class="lang-band-row" data-lang="r"><th colspan="3" scope="rowgroup"><span class="lang-band-dot"></span>R · archived pls4all benchmark</th></tr>
<tr class="bk-row"><td class="bk-name"><code>pls4all.R</code></td><td class="parity parity-exact">✓ J 0.18</td><td class="ms">4.83 ms</td></tr>
<tr class="bk-row"><td class="bk-name"><code>pls4all.R.formula</code></td><td class="parity parity-exact">✓ J 0.18</td><td class="ms">5.64 ms</td></tr>
<tr class="bk-row"><td class="bk-name"><code>pls4all.R.mdatools</code></td><td class="parity parity-exact">✓ J 0.18</td><td class="ms">5.77 ms</td></tr>
<tr class="bk-row"><td class="bk-name"><code>pls4all.R.pls</code></td><td class="parity parity-exact">✓ J 0.18</td><td class="ms">5.73 ms</td></tr>
</tbody>
</table>
</div>

:::

:::{tab-item} 10 threads
:sync: threads-10

<div class="parity-table-wrap">
<table class="docutils parity-grouped">
<thead><tr><th scope="col">Backend</th><th scope="col">Parity</th><th class="size-col" scope="col">200×40 (ms)</th></tr></thead>
<tbody class="lang-band lang-python"><tr class="lang-band-row" data-lang="python"><th colspan="3" scope="rowgroup"><span class="lang-band-dot"></span>Python · archived pls4all benchmark</th></tr>
<tr class="bk-row"><td class="bk-name"><code>pls4all.sklearn</code></td><td class="parity parity-exact">✓ J 0.18</td><td class="ms ms-best">2.19 ms<span class="medal" title="fastest">🏆</span></td></tr>
</tbody>
<tbody class="lang-band lang-r"><tr class="lang-band-row" data-lang="r"><th colspan="3" scope="rowgroup"><span class="lang-band-dot"></span>R · archived pls4all benchmark</th></tr>
<tr class="bk-row"><td class="bk-name"><code>pls4all.R</code></td><td class="parity parity-exact">✓ J 0.18</td><td class="ms">5.15 ms</td></tr>
<tr class="bk-row"><td class="bk-name"><code>pls4all.R.formula</code></td><td class="parity parity-exact">✓ J 0.18</td><td class="ms">5.73 ms</td></tr>
<tr class="bk-row"><td class="bk-name"><code>pls4all.R.mdatools</code></td><td class="parity parity-exact">✓ J 0.18</td><td class="ms">5.76 ms</td></tr>
<tr class="bk-row"><td class="bk-name"><code>pls4all.R.pls</code></td><td class="parity parity-exact">✓ J 0.18</td><td class="ms">5.71 ms</td></tr>
</tbody>
</table>
</div>

:::

::::


---

_See also_: [benchmark overview](../benchmarks/overview.md) · [methods index](index.md) · [interactive dashboard](../landing/dashboard.md)