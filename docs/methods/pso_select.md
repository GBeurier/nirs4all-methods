# `pso_select` — PSO-PLS — Particle Swarm Optimisation

_Group_: **Variable selector** · _Registry tolerance_: `1e-06`

## Description

PSO-PLS — Binary Particle Swarm variable selection (§48)

> **Registry note** — Python `pyswarms 1.3.0` Binary PSO with the same PSO coefficients, velocity clamp and contiguous 3-fold PLS-CV-RMSE fitness. Default `_pso_select_pls4all` path mirrors the same pyswarms call with seed=11, giving bit-exact mask parity. The C++ splitmix64 PSO kernel is opt-in via `legacy=True`.

### Parameters

| Name | Type | Default | Notes |
|------|------|---------|-------|
| `n_components` | `int` | `3` | registry benchmark cell value |
| `n_swarm` | `int` | `10` | registry benchmark cell value |
| `n_iterations` | `int` | `12` | registry benchmark cell value |
| `w` | `float` | `0.729` | registry benchmark cell value |
| `c1` | `float` | `1.494` | registry benchmark cell value |
| `c2` | `float` | `1.494` | registry benchmark cell value |
| `v_max` | `float` | `4.0` | registry benchmark cell value |
| `seed` | `int` | `42` | registry benchmark cell value |

## Explanations

### Bibliographic source

Kennedy, J. & Eberhart, R. (1995). *Particle swarm optimization*. IEEE ICNN 1995, vol. 4, 1942–1948. — binary PSO variant used for variable selection. Verified primary link: [https://doi.org/10.1109/ICNN.1995.488968](https://doi.org/10.1109/ICNN.1995.488968).

### Mathematical principle

Binary PSO maintains a swarm of particles where each particle's position is a $p$-bit feature mask. Velocity updates blend the particle's personal best, the swarm's global best, and an inertia term; positions are stochastically rounded to 0/1 via a sigmoid.

Compared to GA-PLS, PSO converges faster on smooth fitness landscapes but is more susceptible to premature convergence on multi-modal ones. The two methods are complementary: PSO for quick reconnaissance, GA for final polish.

Recommended swarm size: 20–50. The fitness is again PLS CV-RMSE on the masked subset.

### Appropriate uses

Population-based exploration of binary wavelength masks under a predictive fitness function.

### Limits and validation

Binary PSO is a heuristic without global-optimum guarantees and can overfit a repeatedly queried validation set.

### Implementation

`n4m_feature_selection_pso_select`. Reference: Python `pyswarms` for the PSO core, wrapped against PLS CV-RMSE.

### Sources and provenance

Current implementation: [cpp/src/core/pso_selection.cpp](https://github.com/GBeurier/nirs4all-methods/blob/main/cpp/src/core/pso_selection.cpp).


### API and bindings

**C ABI (ABI 2):** [`n4m_feature_selection_pso_select`](https://github.com/GBeurier/nirs4all-methods/blob/main/cpp/include/n4m/feature_selection.h#L178). Use the linked public header for the exact signature, configuration, and result handles.

**Python:** no current AST-verified public `n4m` re-export was found for this method. The linked C ABI above is the documented surface in this checkout.

**R (source-verified):** [`pso_select(X, Y, n_components, n_swarm = 30L, n_iterations = 50L, w = 0.729, c1 = 1.494, c2 = 1.494, v_max = 4.0, seed = 0L)`](https://github.com/GBeurier/nirs4all-methods/blob/main/bindings/r/n4m/R/methods_extra.R).

The source signature has additional required inputs, so no example call is fabricated.

**MATLAB / Octave (source-verified):** [`pso_select(X, Y, n_components, n_swarm, n_iterations, ... w, c1, c2, v_max, seed)`](https://github.com/GBeurier/nirs4all-methods/blob/main/bindings/matlab/+n4m/pso_select.m).

The source signature has additional required inputs, so no example call is fabricated.

**Registry parity references** 📐

:::{card}
:class-card: external-refs

- 📐 **`ref.python_pyswarms`** (python · python) — `pyswarms` 1.3.0 · strict (rmse_rel ≤ 1e-06) — Python `pyswarms.discrete.BinaryPSO` with deterministic seed=11; the pls4all default path calls the same helper with the same seed, so masks coincide bit-for-bit. The C++ splitmix64 PSO kernel is opt-in via legacy=True.

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
<tr class="bk-row"><td class="bk-name"><code>pls4all.cpp.blas+omp</code></td><td class="parity parity-exact">✓ J 1.00</td><td class="ms">219.8 ms</td></tr>
</tbody>
<tbody class="lang-band lang-python"><tr class="lang-band-row" data-lang="python"><th colspan="3" scope="rowgroup"><span class="lang-band-dot"></span>Python · archived pls4all benchmark</th></tr>
<tr class="bk-row"><td class="bk-name"><code>pls4all.python</code></td><td class="parity parity-exact">✓ J 1.00</td><td class="ms">219.3 ms</td></tr>
<tr class="bk-row"><td class="bk-name"><code>pls4all.sklearn</code></td><td class="parity parity-cross_check">⇄ J 0.46</td><td class="ms ms-best">4.05 ms<span class="medal" title="fastest">🏆</span></td></tr>
</tbody>
<tbody class="lang-band lang-r"><tr class="lang-band-row" data-lang="r"><th colspan="3" scope="rowgroup"><span class="lang-band-dot"></span>R · archived pls4all benchmark</th></tr>
<tr class="bk-row"><td class="bk-name"><code>pls4all.R</code></td><td class="parity parity-cross_check">⇄ J 0.46</td><td class="ms">6.11 ms</td></tr>
<tr class="bk-row"><td class="bk-name"><code>pls4all.R.formula</code></td><td class="parity parity-cross_check">⇄ J 0.46</td><td class="ms">6.29 ms</td></tr>
<tr class="bk-row"><td class="bk-name"><code>pls4all.R.mdatools</code></td><td class="parity parity-cross_check">⇄ J 0.46</td><td class="ms">7.33 ms</td></tr>
<tr class="bk-row"><td class="bk-name"><code>pls4all.R.pls</code></td><td class="parity parity-cross_check">⇄ J 0.46</td><td class="ms">7.12 ms</td></tr>
</tbody>
<tbody class="lang-band lang-python"><tr class="lang-band-row" data-lang="python"><th colspan="3" scope="rowgroup"><span class="lang-band-dot"></span>Python · external</th></tr>
<tr class="bk-row truth-source truth-source-strict"><td class="bk-name"><span class="truth-mark" title="Registry parity reference (python): pyswarms 1.3.0 — strict (rmse_rel ≤ 1e-06)">📐</span><code>ref.python_pyswarms</code></td><td class="parity parity-ref-source">source</td><td class="ms">163.8 ms</td></tr>
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
<tr class="bk-row"><td class="bk-name"><code>pls4all.cpp.blas+omp</code></td><td class="parity parity-exact">✓ J 1.00</td><td class="ms">169.0 ms</td></tr>
</tbody>
<tbody class="lang-band lang-python"><tr class="lang-band-row" data-lang="python"><th colspan="3" scope="rowgroup"><span class="lang-band-dot"></span>Python · archived pls4all benchmark</th></tr>
<tr class="bk-row"><td class="bk-name"><code>pls4all.python</code></td><td class="parity parity-exact">✓ J 1.00</td><td class="ms">298.6 ms</td></tr>
<tr class="bk-row"><td class="bk-name"><code>pls4all.sklearn</code></td><td class="parity parity-cross_check">⇄ J 0.46</td><td class="ms ms-best">3.17 ms<span class="medal" title="fastest">🏆</span></td></tr>
</tbody>
<tbody class="lang-band lang-r"><tr class="lang-band-row" data-lang="r"><th colspan="3" scope="rowgroup"><span class="lang-band-dot"></span>R · archived pls4all benchmark</th></tr>
<tr class="bk-row"><td class="bk-name"><code>pls4all.R</code></td><td class="parity parity-cross_check">⇄ J 0.46</td><td class="ms">20.9 ms</td></tr>
<tr class="bk-row"><td class="bk-name"><code>pls4all.R.formula</code></td><td class="parity parity-cross_check">⇄ J 0.46</td><td class="ms">20.3 ms</td></tr>
<tr class="bk-row"><td class="bk-name"><code>pls4all.R.mdatools</code></td><td class="parity parity-cross_check">⇄ J 0.46</td><td class="ms">16.0 ms</td></tr>
<tr class="bk-row"><td class="bk-name"><code>pls4all.R.pls</code></td><td class="parity parity-cross_check">⇄ J 0.46</td><td class="ms">24.6 ms</td></tr>
</tbody>
<tbody class="lang-band lang-python"><tr class="lang-band-row" data-lang="python"><th colspan="3" scope="rowgroup"><span class="lang-band-dot"></span>Python · external</th></tr>
<tr class="bk-row truth-source truth-source-strict"><td class="bk-name"><span class="truth-mark" title="Registry parity reference (python): pyswarms 1.3.0 — strict (rmse_rel ≤ 1e-06)">📐</span><code>ref.python_pyswarms</code></td><td class="parity parity-ref-source">source</td><td class="ms">217.8 ms</td></tr>
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
<tr class="bk-row"><td class="bk-name"><code>pls4all.cpp.blas+omp</code></td><td class="parity parity-exact">✓ J 1.00</td><td class="ms">165.6 ms</td></tr>
</tbody>
<tbody class="lang-band lang-python"><tr class="lang-band-row" data-lang="python"><th colspan="3" scope="rowgroup"><span class="lang-band-dot"></span>Python · archived pls4all benchmark</th></tr>
<tr class="bk-row"><td class="bk-name"><code>pls4all.python</code></td><td class="parity parity-exact">✓ J 1.00</td><td class="ms">181.1 ms</td></tr>
<tr class="bk-row"><td class="bk-name"><code>pls4all.sklearn</code></td><td class="parity parity-cross_check">⇄ J 0.46</td><td class="ms ms-best">2.80 ms<span class="medal" title="fastest">🏆</span></td></tr>
</tbody>
<tbody class="lang-band lang-r"><tr class="lang-band-row" data-lang="r"><th colspan="3" scope="rowgroup"><span class="lang-band-dot"></span>R · archived pls4all benchmark</th></tr>
<tr class="bk-row"><td class="bk-name"><code>pls4all.R</code></td><td class="parity parity-cross_check">⇄ J 0.46</td><td class="ms">4.19 ms</td></tr>
<tr class="bk-row"><td class="bk-name"><code>pls4all.R.formula</code></td><td class="parity parity-cross_check">⇄ J 0.46</td><td class="ms">4.78 ms</td></tr>
<tr class="bk-row"><td class="bk-name"><code>pls4all.R.mdatools</code></td><td class="parity parity-cross_check">⇄ J 0.46</td><td class="ms">4.76 ms</td></tr>
<tr class="bk-row"><td class="bk-name"><code>pls4all.R.pls</code></td><td class="parity parity-cross_check">⇄ J 0.46</td><td class="ms">4.74 ms</td></tr>
</tbody>
<tbody class="lang-band lang-python"><tr class="lang-band-row" data-lang="python"><th colspan="3" scope="rowgroup"><span class="lang-band-dot"></span>Python · external</th></tr>
<tr class="bk-row truth-source truth-source-strict"><td class="bk-name"><span class="truth-mark" title="Registry parity reference (python): pyswarms 1.3.0 — strict (rmse_rel ≤ 1e-06)">📐</span><code>ref.python_pyswarms</code></td><td class="parity parity-ref-source">source</td><td class="ms">152.3 ms</td></tr>
</tbody>
</table>
</div>

:::

::::


---

_See also_: [benchmark overview](../benchmarks/overview.md) · [methods index](index.md) · [interactive dashboard](../landing/dashboard.md)