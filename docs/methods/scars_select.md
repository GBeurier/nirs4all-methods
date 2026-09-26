# `scars_select` — SCARS — Stability-CARS

_Group_: **Variable selector** · _Registry tolerance_: `1e-06`

## Description

SCARS stability + CARS (§18 Phase 5h)

> **Registry note** — NumPy port of Stability CARS (Zheng 2014) — Monte-Carlo subsampling + stability scoring + CARS exponential shrinkage. Default `_scars_select_pls4all` path invokes the same NumPy function with `np.random.default_rng(seed)`, giving bit-exact mask parity. The C++ splitmix64 kernel is opt-in via `legacy=True`.

### Parameters

| Name | Type | Default | Notes |
|------|------|---------|-------|
| `n_components` | `int` | `2` | current public binding signature |
| `n_iterations` | `int` | `50` | current public binding signature |
| `min_features` | `int \| None` | `None` | current public binding signature |
| `sample_fraction` | `float` | `0.8` | current public binding signature |
| `n_folds` | `int` | `3` | current public binding signature |
| `seed` | `int` | `0` | current public binding signature |

## Explanations

### Bibliographic source

Zheng, K., Li, Q., Wang, J., Geng, J., Cao, P., Sui, T., Wang, X. & Du, Y. (2012). *Stability competitive adaptive reweighted sampling (SCARS) and its applications to multivariate calibration of NIR spectra*. Chemometrics and Intelligent Laboratory Systems 112, 48–54. Verified primary link: [https://doi.org/10.1016/j.chemolab.2012.01.002](https://doi.org/10.1016/j.chemolab.2012.01.002).

### Mathematical principle

Replace CARS's coefficient-magnitude weights with **coefficient-stability** weights: $w_j = |\bar{b}_j| / s(b_j)$ from the bootstrap distribution. Stability-weighted retention is more robust to spurious high-magnitude coefficients caused by particular bootstrap subsamples.

Otherwise identical to CARS: exponential decay schedule and stochastic competition. SCARS typically improves CARS on datasets with strong baseline drift or where a few high-leverage samples dominate the coefficient estimates.

### Appropriate uses

Sequential competitive elimination of weak wavelengths across repeated PLS evaluations.

### Limits and validation

Aggressive elimination is irreversible within a run and can discard correlated alternatives; validation reuse creates optimism.

### Implementation

`n4m_feature_selection_scars_select`.

### Sources and provenance

Current implementation: [cpp/src/core/scars_selection.cpp](https://github.com/GBeurier/nirs4all-methods/blob/main/cpp/src/core/scars_selection.cpp).


### API and bindings

**C ABI (ABI 2):** [`n4m_feature_selection_scars_select`](https://github.com/GBeurier/nirs4all-methods/blob/main/cpp/include/n4m/feature_selection.h#L134). Use the linked public header for the exact signature, configuration, and result handles.

**Python (verified public re-export):**

```python
from n4m.feature_selection.wrapper import SCARS
```

Source signature: `SCARS(*, n_components: int = 2, n_iterations: int = 50, min_features: int | None = None, sample_fraction: float = 0.8, n_folds: int = 3, seed: int = 0)` ([`n4m/_impl/selection.py`](https://github.com/GBeurier/nirs4all-methods/blob/main/bindings/python/src/n4m/_impl/selection.py#L259)).

**R (source-verified):** [`scars_select(X, Y, n_components, n_iterations = 50L, min_features = 5L, sample_fraction = 0.8, seed = 0L)`](https://github.com/GBeurier/nirs4all-methods/blob/main/bindings/r/n4m/R/methods_extra.R).

The source signature has additional required inputs, so no example call is fabricated.

**MATLAB / Octave (source-verified):** [`scars_select(X, Y, n_components, n_iterations, min_features, ... sample_fraction, seed)`](https://github.com/GBeurier/nirs4all-methods/blob/main/bindings/matlab/+n4m/scars_select.m).

The source signature has additional required inputs, so no example call is fabricated.

**Registry parity references** 📐

:::{card}
:class-card: external-refs

- 📐 **`ref.python_scars_numpy_port`** (python · python) — `scars_numpy_port` 1.0.0 · strict (rmse_rel ≤ 1e-06) — NumPy port of Stability CARS (Zheng 2014) — Monte-Carlo subsampling + stability scoring + CARS exponential shrinkage. Pinned `np.random.default_rng(seed)` for bit-exact reproducibility.

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
<tr class="bk-row"><td class="bk-name"><code>pls4all.cpp.blas+omp</code></td><td class="parity parity-exact">✓ J 1.00</td><td class="ms">7.65 ms</td></tr>
</tbody>
<tbody class="lang-band lang-python"><tr class="lang-band-row" data-lang="python"><th colspan="3" scope="rowgroup"><span class="lang-band-dot"></span>Python · archived pls4all benchmark</th></tr>
<tr class="bk-row"><td class="bk-name"><code>pls4all.python</code></td><td class="parity parity-exact">✓ J 1.00</td><td class="ms">10.2 ms</td></tr>
<tr class="bk-row"><td class="bk-name"><code>pls4all.sklearn</code></td><td class="parity parity-cross_check">⇄ J 0.33</td><td class="ms">4.14 ms</td></tr>
</tbody>
<tbody class="lang-band lang-r"><tr class="lang-band-row" data-lang="r"><th colspan="3" scope="rowgroup"><span class="lang-band-dot"></span>R · archived pls4all benchmark</th></tr>
<tr class="bk-row"><td class="bk-name"><code>pls4all.R</code></td><td class="parity parity-cross_check">⇄ J 0.33</td><td class="ms">4.85 ms</td></tr>
<tr class="bk-row"><td class="bk-name"><code>pls4all.R.formula</code></td><td class="parity parity-cross_check">⇄ J 0.33</td><td class="ms">5.55 ms</td></tr>
<tr class="bk-row"><td class="bk-name"><code>pls4all.R.mdatools</code></td><td class="parity parity-cross_check">⇄ J 0.33</td><td class="ms">5.92 ms</td></tr>
<tr class="bk-row"><td class="bk-name"><code>pls4all.R.pls</code></td><td class="parity parity-cross_check">⇄ J 0.33</td><td class="ms">5.72 ms</td></tr>
</tbody>
<tbody class="lang-band lang-python"><tr class="lang-band-row" data-lang="python"><th colspan="3" scope="rowgroup"><span class="lang-band-dot"></span>Python · external</th></tr>
<tr class="bk-row truth-source truth-source-strict"><td class="bk-name"><span class="truth-mark" title="Registry parity reference (python): scars_numpy_port 1.0.0 — strict (rmse_rel ≤ 1e-06)">📐</span><code>ref.python_scars_numpy_port</code></td><td class="parity parity-ref-source">source</td><td class="ms ms-best">3.61 ms<span class="medal" title="fastest">🏆</span></td></tr>
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
<tr class="bk-row"><td class="bk-name"><code>pls4all.cpp.blas+omp</code></td><td class="parity parity-exact">✓ J 1.00</td><td class="ms">3.71 ms</td></tr>
</tbody>
<tbody class="lang-band lang-python"><tr class="lang-band-row" data-lang="python"><th colspan="3" scope="rowgroup"><span class="lang-band-dot"></span>Python · archived pls4all benchmark</th></tr>
<tr class="bk-row"><td class="bk-name"><code>pls4all.python</code></td><td class="parity parity-exact">✓ J 1.00</td><td class="ms">3.88 ms</td></tr>
<tr class="bk-row"><td class="bk-name"><code>pls4all.sklearn</code></td><td class="parity parity-cross_check">⇄ J 0.33</td><td class="ms ms-best">2.29 ms<span class="medal" title="fastest">🏆</span></td></tr>
</tbody>
<tbody class="lang-band lang-r"><tr class="lang-band-row" data-lang="r"><th colspan="3" scope="rowgroup"><span class="lang-band-dot"></span>R · archived pls4all benchmark</th></tr>
<tr class="bk-row"><td class="bk-name"><code>pls4all.R</code></td><td class="parity parity-cross_check">⇄ J 0.33</td><td class="ms">5.06 ms</td></tr>
<tr class="bk-row"><td class="bk-name"><code>pls4all.R.formula</code></td><td class="parity parity-cross_check">⇄ J 0.33</td><td class="ms">7.16 ms</td></tr>
<tr class="bk-row"><td class="bk-name"><code>pls4all.R.mdatools</code></td><td class="parity parity-cross_check">⇄ J 0.33</td><td class="ms">6.22 ms</td></tr>
<tr class="bk-row"><td class="bk-name"><code>pls4all.R.pls</code></td><td class="parity parity-cross_check">⇄ J 0.33</td><td class="ms">6.17 ms</td></tr>
</tbody>
<tbody class="lang-band lang-python"><tr class="lang-band-row" data-lang="python"><th colspan="3" scope="rowgroup"><span class="lang-band-dot"></span>Python · external</th></tr>
<tr class="bk-row truth-source truth-source-strict"><td class="bk-name"><span class="truth-mark" title="Registry parity reference (python): scars_numpy_port 1.0.0 — strict (rmse_rel ≤ 1e-06)">📐</span><code>ref.python_scars_numpy_port</code></td><td class="parity parity-ref-source">source</td><td class="ms">3.67 ms</td></tr>
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
<tr class="bk-row"><td class="bk-name"><code>pls4all.cpp.blas+omp</code></td><td class="parity parity-exact">✓ J 1.00</td><td class="ms">3.66 ms</td></tr>
</tbody>
<tbody class="lang-band lang-python"><tr class="lang-band-row" data-lang="python"><th colspan="3" scope="rowgroup"><span class="lang-band-dot"></span>Python · archived pls4all benchmark</th></tr>
<tr class="bk-row"><td class="bk-name"><code>pls4all.python</code></td><td class="parity parity-exact">✓ J 1.00</td><td class="ms">3.91 ms</td></tr>
<tr class="bk-row"><td class="bk-name"><code>pls4all.sklearn</code></td><td class="parity parity-cross_check">⇄ J 0.33</td><td class="ms ms-best">2.22 ms<span class="medal" title="fastest">🏆</span></td></tr>
</tbody>
<tbody class="lang-band lang-r"><tr class="lang-band-row" data-lang="r"><th colspan="3" scope="rowgroup"><span class="lang-band-dot"></span>R · archived pls4all benchmark</th></tr>
<tr class="bk-row"><td class="bk-name"><code>pls4all.R</code></td><td class="parity parity-cross_check">⇄ J 0.33</td><td class="ms">4.89 ms</td></tr>
<tr class="bk-row"><td class="bk-name"><code>pls4all.R.formula</code></td><td class="parity parity-cross_check">⇄ J 0.33</td><td class="ms">14.1 ms</td></tr>
<tr class="bk-row"><td class="bk-name"><code>pls4all.R.mdatools</code></td><td class="parity parity-cross_check">⇄ J 0.33</td><td class="ms">17.6 ms</td></tr>
<tr class="bk-row"><td class="bk-name"><code>pls4all.R.pls</code></td><td class="parity parity-cross_check">⇄ J 0.33</td><td class="ms">12.6 ms</td></tr>
</tbody>
<tbody class="lang-band lang-python"><tr class="lang-band-row" data-lang="python"><th colspan="3" scope="rowgroup"><span class="lang-band-dot"></span>Python · external</th></tr>
<tr class="bk-row truth-source truth-source-strict"><td class="bk-name"><span class="truth-mark" title="Registry parity reference (python): scars_numpy_port 1.0.0 — strict (rmse_rel ≤ 1e-06)">📐</span><code>ref.python_scars_numpy_port</code></td><td class="parity parity-ref-source">source</td><td class="ms">9.38 ms</td></tr>
</tbody>
</table>
</div>

:::

::::


---

_See also_: [benchmark overview](../benchmarks/overview.md) · [methods index](index.md) · [interactive dashboard](../landing/dashboard.md)