# `iriv_select` — IRIV — Iteratively Retaining Informative Variables

_Group_: **Variable selector** · _Registry tolerance_: `1e-06`

## Description

Iteratively Retains Informative Variables (Phase 51)

> **Registry note** — NumPy port of libPLS `iriv` (Yun 2014). Mann-Whitney U test via `scipy.stats.mannwhitneyu`. Default `_iriv_select_pls4all` path invokes the same NumPy function with `np.random.default_rng(seed)`, giving bit-exact mask parity. The C++ splitmix64 kernel is opt-in via `legacy=True`.

### Parameters

| Name | Type | Default | Notes |
|------|------|---------|-------|
| `n_components` | `int` | `2` | current public binding signature |
| `max_rounds` | `int` | `5` | current public binding signature |
| `n_folds` | `int` | `5` | current public binding signature |
| `seed` | `int` | `0` | current public binding signature |
| `fold` | `int` | `3` | registry benchmark cell value |

## Explanations

### Bibliographic source

Yun, Y. H., Wang, W. T., Tan, M. L., Liang, Y. Z., Li, H. D., Cao, D. S., Lu, H. M. & Xu, Q. S. (2014). *A strategy that iteratively retains informative variables for selecting optimal variable subset in multivariate calibration*. Analytica Chimica Acta 807, 36–43. Verified primary link: [https://doi.org/10.1016/j.aca.2013.11.032](https://doi.org/10.1016/j.aca.2013.11.032).

### Mathematical principle

IRIV classifies each variable into four categories at each iteration: **strongly informative**, **weakly informative**, **uninformative**, **interfering**. The first two are retained, the last two eliminated. Iteration continues until no further interfering variables remain.

Categories are determined by Monte-Carlo subset analysis with a permutation-based reference distribution: each variable's CV-RMSE contribution distribution is compared against the distribution under random subset inclusion. This four-way classification is more nuanced than single-threshold methods and tends to handle correlated predictors well (correlated features can both be 'weakly informative').

### Appropriate uses

Iteratively comparing variable subsets to classify informative and interfering wavelengths.

### Limits and validation

Repeated random subset tests are stochastic and computationally expensive; significance labels depend on the sampled combinations.

### Implementation

`n4m_feature_selection_iriv_select`.

### Sources and provenance

Current implementation: [cpp/src/core/iriv_selection.cpp](https://github.com/GBeurier/nirs4all-methods/blob/main/cpp/src/core/iriv_selection.cpp).


### API and bindings

**C ABI (ABI 2):** [`n4m_feature_selection_iriv_select`](https://github.com/GBeurier/nirs4all-methods/blob/main/cpp/include/n4m/feature_selection.h#L386). Use the linked public header for the exact signature, configuration, and result handles.

**Python (verified public re-export):**

```python
from n4m.feature_selection.wrapper import IRIV
```

Source signature: `IRIV(*, n_components: int = 2, max_rounds: int = 5, n_folds: int = 5, seed: int = 0)` ([`n4m/_impl/selection.py`](https://github.com/GBeurier/nirs4all-methods/blob/main/bindings/python/src/n4m/_impl/selection.py#L728)).

**R (source-verified):** [`iriv_select(X, Y, n_components, max_rounds = 20L, seed = 0L)`](https://github.com/GBeurier/nirs4all-methods/blob/main/bindings/r/n4m/R/methods_extra.R).

The source signature has additional required inputs, so no example call is fabricated.

**MATLAB / Octave (source-verified):** [`iriv_select(X, Y, n_components, max_rounds, seed)`](https://github.com/GBeurier/nirs4all-methods/blob/main/bindings/matlab/+n4m/iriv_select.m).

The source signature has additional required inputs, so no example call is fabricated.

**Registry parity references** 📐

:::{card}
:class-card: external-refs

- 📐 **`ref.python_iriv_numpy_port`** (python · python) — `iriv_numpy_port` 1.0.0 · strict (rmse_rel ≤ 1e-06) — NumPy port of libPLS `iriv` (Yun 2014). Mann-Whitney U test via `scipy.stats.mannwhitneyu`; binary-matrix sampler keyed to `np.random.default_rng(seed)` for bit-exact reproducibility.

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
<tbody class="lang-band lang-cpp"><tr class="lang-band-row" data-lang="cpp"><th colspan="3" scope="rowgroup"><span class="lang-band-dot"></span>C++ native · libn4m</th></tr>
<tr class="bk-row"><td class="bk-name"><code>pls4all.cpp.blas+omp</code></td><td class="parity parity-exact">✓ J 1.00</td><td class="ms">268.0 ms</td></tr>
</tbody>
<tbody class="lang-band lang-python"><tr class="lang-band-row" data-lang="python"><th colspan="3" scope="rowgroup"><span class="lang-band-dot"></span>Python · archived pls4all benchmark</th></tr>
<tr class="bk-row"><td class="bk-name"><code>pls4all.python</code></td><td class="parity parity-exact">✓ J 1.00</td><td class="ms">270.2 ms</td></tr>
<tr class="bk-row"><td class="bk-name"><code>pls4all.sklearn</code></td><td class="parity parity-cross_check">⇄ J 0.60</td><td class="ms ms-best">25.3 ms<span class="medal" title="fastest">🏆</span></td></tr>
</tbody>
<tbody class="lang-band lang-r"><tr class="lang-band-row" data-lang="r"><th colspan="3" scope="rowgroup"><span class="lang-band-dot"></span>R · archived pls4all benchmark</th></tr>
<tr class="bk-row"><td class="bk-name"><code>pls4all.R</code></td><td class="parity parity-cross_check">⇄ J 0.60</td><td class="ms">29.4 ms</td></tr>
<tr class="bk-row"><td class="bk-name"><code>pls4all.R.formula</code></td><td class="parity parity-cross_check">⇄ J 0.60</td><td class="ms">30.0 ms</td></tr>
<tr class="bk-row"><td class="bk-name"><code>pls4all.R.mdatools</code></td><td class="parity parity-cross_check">⇄ J 0.60</td><td class="ms">30.8 ms</td></tr>
<tr class="bk-row"><td class="bk-name"><code>pls4all.R.pls</code></td><td class="parity parity-cross_check">⇄ J 0.60</td><td class="ms">30.5 ms</td></tr>
</tbody>
<tbody class="lang-band lang-python"><tr class="lang-band-row" data-lang="python"><th colspan="3" scope="rowgroup"><span class="lang-band-dot"></span>Python · external</th></tr>
<tr class="bk-row truth-source truth-source-strict"><td class="bk-name"><span class="truth-mark" title="Registry parity reference (python): iriv_numpy_port 1.0.0 — strict (rmse_rel ≤ 1e-06)">📐</span><code>ref.python_iriv_numpy_port</code></td><td class="parity parity-ref-source">source</td><td class="ms">274.8 ms</td></tr>
</tbody>
</table>
</div>

:::

:::{tab-item} 3 threads
:sync: threads-3

<div class="parity-table-wrap">
<table class="docutils parity-grouped">
<thead><tr><th scope="col">Backend</th><th scope="col">Parity</th><th class="size-col" scope="col">80×25 (ms)</th></tr></thead>
<tbody class="lang-band lang-cpp"><tr class="lang-band-row" data-lang="cpp"><th colspan="3" scope="rowgroup"><span class="lang-band-dot"></span>C++ native · libn4m</th></tr>
<tr class="bk-row"><td class="bk-name"><code>pls4all.cpp.blas+omp</code></td><td class="parity parity-exact">✓ J 1.00</td><td class="ms">273.2 ms</td></tr>
</tbody>
<tbody class="lang-band lang-python"><tr class="lang-band-row" data-lang="python"><th colspan="3" scope="rowgroup"><span class="lang-band-dot"></span>Python · archived pls4all benchmark</th></tr>
<tr class="bk-row"><td class="bk-name"><code>pls4all.python</code></td><td class="parity parity-exact">✓ J 1.00</td><td class="ms">267.2 ms</td></tr>
<tr class="bk-row"><td class="bk-name"><code>pls4all.sklearn</code></td><td class="parity parity-cross_check">⇄ J 0.60</td><td class="ms ms-best">25.1 ms<span class="medal" title="fastest">🏆</span></td></tr>
</tbody>
<tbody class="lang-band lang-r"><tr class="lang-band-row" data-lang="r"><th colspan="3" scope="rowgroup"><span class="lang-band-dot"></span>R · archived pls4all benchmark</th></tr>
<tr class="bk-row"><td class="bk-name"><code>pls4all.R</code></td><td class="parity parity-cross_check">⇄ J 0.60</td><td class="ms">29.8 ms</td></tr>
<tr class="bk-row"><td class="bk-name"><code>pls4all.R.formula</code></td><td class="parity parity-cross_check">⇄ J 0.60</td><td class="ms">30.8 ms</td></tr>
<tr class="bk-row"><td class="bk-name"><code>pls4all.R.mdatools</code></td><td class="parity parity-cross_check">⇄ J 0.60</td><td class="ms">30.6 ms</td></tr>
<tr class="bk-row"><td class="bk-name"><code>pls4all.R.pls</code></td><td class="parity parity-cross_check">⇄ J 0.60</td><td class="ms">30.3 ms</td></tr>
</tbody>
<tbody class="lang-band lang-python"><tr class="lang-band-row" data-lang="python"><th colspan="3" scope="rowgroup"><span class="lang-band-dot"></span>Python · external</th></tr>
<tr class="bk-row truth-source truth-source-strict"><td class="bk-name"><span class="truth-mark" title="Registry parity reference (python): iriv_numpy_port 1.0.0 — strict (rmse_rel ≤ 1e-06)">📐</span><code>ref.python_iriv_numpy_port</code></td><td class="parity parity-ref-source">source</td><td class="ms">275.1 ms</td></tr>
</tbody>
</table>
</div>

:::

:::{tab-item} 10 threads
:sync: threads-10

<div class="parity-table-wrap">
<table class="docutils parity-grouped">
<thead><tr><th scope="col">Backend</th><th scope="col">Parity</th><th class="size-col" scope="col">80×25 (ms)</th></tr></thead>
<tbody class="lang-band lang-cpp"><tr class="lang-band-row" data-lang="cpp"><th colspan="3" scope="rowgroup"><span class="lang-band-dot"></span>C++ native · libn4m</th></tr>
<tr class="bk-row"><td class="bk-name"><code>pls4all.cpp.blas+omp</code></td><td class="parity parity-exact">✓ J 1.00</td><td class="ms">274.2 ms</td></tr>
</tbody>
<tbody class="lang-band lang-python"><tr class="lang-band-row" data-lang="python"><th colspan="3" scope="rowgroup"><span class="lang-band-dot"></span>Python · archived pls4all benchmark</th></tr>
<tr class="bk-row"><td class="bk-name"><code>pls4all.python</code></td><td class="parity parity-exact">✓ J 1.00</td><td class="ms">275.4 ms</td></tr>
<tr class="bk-row"><td class="bk-name"><code>pls4all.sklearn</code></td><td class="parity parity-cross_check">⇄ J 0.60</td><td class="ms ms-best">25.4 ms<span class="medal" title="fastest">🏆</span></td></tr>
</tbody>
<tbody class="lang-band lang-r"><tr class="lang-band-row" data-lang="r"><th colspan="3" scope="rowgroup"><span class="lang-band-dot"></span>R · archived pls4all benchmark</th></tr>
<tr class="bk-row"><td class="bk-name"><code>pls4all.R</code></td><td class="parity parity-cross_check">⇄ J 0.60</td><td class="ms">29.9 ms</td></tr>
<tr class="bk-row"><td class="bk-name"><code>pls4all.R.formula</code></td><td class="parity parity-cross_check">⇄ J 0.60</td><td class="ms">30.7 ms</td></tr>
<tr class="bk-row"><td class="bk-name"><code>pls4all.R.mdatools</code></td><td class="parity parity-cross_check">⇄ J 0.60</td><td class="ms">31.0 ms</td></tr>
<tr class="bk-row"><td class="bk-name"><code>pls4all.R.pls</code></td><td class="parity parity-cross_check">⇄ J 0.60</td><td class="ms">30.7 ms</td></tr>
</tbody>
<tbody class="lang-band lang-python"><tr class="lang-band-row" data-lang="python"><th colspan="3" scope="rowgroup"><span class="lang-band-dot"></span>Python · external</th></tr>
<tr class="bk-row truth-source truth-source-strict"><td class="bk-name"><span class="truth-mark" title="Registry parity reference (python): iriv_numpy_port 1.0.0 — strict (rmse_rel ≤ 1e-06)">📐</span><code>ref.python_iriv_numpy_port</code></td><td class="parity parity-ref-source">source</td><td class="ms">270.7 ms</td></tr>
</tbody>
</table>
</div>

:::

::::


---

_See also_: [benchmark overview](../benchmarks/overview.md) · [methods index](index.md) · [interactive dashboard](../landing/dashboard.md)