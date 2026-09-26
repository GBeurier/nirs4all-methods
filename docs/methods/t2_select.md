# `t2_select` — Hotelling T² loading selection

_Group_: **Variable selector** · _Registry tolerance_: `2.0`

## Description

T²-PLS loading-weight selection (§18 Phase 5l)

> **Registry note** — R `plsVarSel::T2_pls` (Mehmood 2016) Hotelling-T² loading-weight selection, called with train=test so it matches pls4all's single-training-set selector. VERIFIED IDENTICAL where it matters: pls4all's per-feature T², the UCL = ((p-1)^2/p)*qbeta(1-alpha, a/2, (p-a-1)/2), the sample covariance over features (/(p-1)), the t2>ucl threshold, the top-k floor fallback, and the min-CV-error alpha pick are all bit-identical to plsVarSel t2_calc / T2_pls. The residual mask divergence is an UPSTREAM PLS loading-weight convention difference: plsVarSel computes T² from R `pls::plsr` `loading.weights` (== sklearn `x_weights_`), while pls4all computes T² from the fitted model's `weights_w`; on borderline features near the UCL the two select slightly different sets (Jaccard ~0.7–1.0; both capture the true signal). Both are valid T²-PLS selectors. Gated on Jaccard via the orchestrator SELECTION_DIVERGENCE_ALLOWLIST.

### Parameters

| Name | Type | Default | Notes |
|------|------|---------|-------|
| `alpha_thresholds` | `—` | `required` | current public binding signature |
| `n_components` | `int` | `2` | current public binding signature |
| `min_selected` | `int \| None` | `None` | current public binding signature |

## Explanations

### Bibliographic source

Mehmood, T. (2016). *Hotelling T² based variable selection in partial least squares regression*. Chemometrics and Intelligent Laboratory Systems 154, 23–28. https://doi.org/10.1016/j.chemolab.2016.03.020 — proposes T²-PLS, the loading-weights-level Hotelling T² selector. See also Wold, Sjöström & Eriksson (2001), Chemometrics and Intelligent Laboratory Systems 58(2), 109–130 §6.2 for the original T²-vs-VIP discussion in PLS.

### Mathematical principle

The routine derives a variable statistic from PLS weight contributions and compares it with a cutoff based on a Beta quantile $B_{1-\alpha}(a/2,(p-a-1)/2)$ and factor $(p-1)^2/p$. Variables above the cutoff are selected; the highest statistics fill any `min_selected` shortfall. It repeats this process for each supplied alpha, evaluates a PLS cross-validation fit for every resulting subset, and reports both `selected_indices_best_error` (smallest CV RMSE) and `selected_indices_min_set` (fewest variables, then CV-RMSE tie-break).

### Appropriate uses

Screening variables by their contribution to leverage in the implemented PLS weight space.

### Limits and validation

The cutoff uses a Beta quantile with an implementation-specific weight-based statistic, not the archived F-limit/top-k algorithm; min_selected may add variables.

### Implementation

This is not the F control-limit plus top-k procedure formerly documented. Beta-quantile parameter validity and the fallback selected count depend on p and a; alpha selection and its CV must be performed inside an outer validation loop to avoid selection leakage.

### Sources and provenance

Current implementation: [cpp/src/core/t2_selection.cpp](https://github.com/GBeurier/nirs4all-methods/blob/main/cpp/src/core/t2_selection.cpp).


### API and bindings

**C ABI (ABI 2):** [`n4m_feature_selection_t2_select`](https://github.com/GBeurier/nirs4all-methods/blob/main/cpp/include/n4m/feature_selection.h#L255). Use the linked public header for the exact signature, configuration, and result handles.

**Python (verified public re-export):**

```python
from n4m.feature_selection.wrapper import T2
```

Source signature: `T2(alpha_thresholds, *, n_components: int = 2, min_selected: int | None = None)` ([`n4m/_impl/selection.py`](https://github.com/GBeurier/nirs4all-methods/blob/main/bindings/python/src/n4m/_impl/selection.py#L593)).

**R (source-verified):** [`t2_select(X, Y, n_components, alpha_thresholds, min_selected = NULL)`](https://github.com/GBeurier/nirs4all-methods/blob/main/bindings/r/n4m/R/methods_extra.R).

The source signature has additional required inputs, so no example call is fabricated.

**MATLAB / Octave (source-verified):** [`t2_select(X, Y, n_components, alpha_thresholds, min_selected)`](https://github.com/GBeurier/nirs4all-methods/blob/main/bindings/matlab/+n4m/t2_select.m).

The source signature has additional required inputs, so no example call is fabricated.

**Registry parity references** 📐

:::{card}
:class-card: external-refs

- 📐 **`ref.r_plsvarsel`** (R · r) — `plsVarSel` 0.10.0 · qualitative (rmse_rel ≤ 2e+00) — R `plsVarSel::T2_pls` — Hotelling T² loading-weight selection. Same idea as pls4all's T2_select.

:::

### Benchmarks

**Archived measurement identity.** Backend labels in this table are the raw IDs recorded when the benchmark ran (including historical `pls4all.*` IDs). They preserve measurement provenance and do not describe a current public Python, R, or MATLAB binding; use the source-verified **API and bindings** section above for current entry points.

Adaptive wall-clock per cell measured against [`full_matrix.csv`](../benchmarks/overview.md). Only backends that implement this method are listed; libraries without the method are omitted.

**Verdict** &nbsp;·&nbsp; ✓ ref / ≈ ref / ~ shape mark a reference-gate pass at strict / relaxed / qualitative tolerance &nbsp;·&nbsp; ✓ bind = archived binding-harness result agrees with the C++ baseline &nbsp;·&nbsp; ⇄ cross-check = documented by-design selector/RNG/model, noncanonical API/facade convention, or secondary oracle &nbsp;·&nbsp; ✗ divergent &nbsp;·&nbsp; ⚠ error &nbsp;·&nbsp; — not run. The fastest backend per column is marked 🏆.

**Reference gate**: qualitative — shape/smoke comparison only. The external library and archived benchmark harness do not produce numerically equivalent output for this method (see the MethodSpec notes); the `rmse_rel_tol ≤ 2e+00` budget is set wide on purpose. Treat ~ shape as *“we ran both, both finished”*, not as numerical agreement.

Rows tagged with **📐** are the canonical parity references for this method (declared in [`parity_timing.registry`](../benchmarks/methodology.md)). C++ and external rows show reference parity; archived language-harness rows show binding parity against the C++ backend. Hover the icon for role and tolerance band.

::::{tab-set}
:class: parity-tabs

:::{tab-item} 1 thread
:sync: threads-1

<div class="parity-table-wrap">
<table class="docutils parity-grouped">
<thead><tr><th scope="col">Backend</th><th scope="col">Parity</th><th class="size-col" scope="col">200×40 (ms)</th></tr></thead>
<tbody class="lang-band lang-cpp"><tr class="lang-band-row" data-lang="cpp"><th colspan="3" scope="rowgroup"><span class="lang-band-dot"></span>C++ native · libn4m</th></tr>
<tr class="bk-row"><td class="bk-name"><code>pls4all.cpp.blas+omp</code></td><td class="parity parity-cross_check">⇄ J 0.80</td><td class="ms ms-best">1.67 ms<span class="medal" title="fastest">🏆</span></td></tr>
</tbody>
<tbody class="lang-band lang-python"><tr class="lang-band-row" data-lang="python"><th colspan="3" scope="rowgroup"><span class="lang-band-dot"></span>Python · archived pls4all benchmark</th></tr>
<tr class="bk-row"><td class="bk-name"><code>pls4all.python</code></td><td class="parity parity-exact">✓ J 0.80</td><td class="ms">1.69 ms</td></tr>
<tr class="bk-row"><td class="bk-name"><code>pls4all.sklearn</code></td><td class="parity parity-exact">✓ J 0.80</td><td class="ms">1.83 ms</td></tr>
</tbody>
<tbody class="lang-band lang-r"><tr class="lang-band-row" data-lang="r"><th colspan="3" scope="rowgroup"><span class="lang-band-dot"></span>R · archived pls4all benchmark</th></tr>
<tr class="bk-row"><td class="bk-name"><code>pls4all.R</code></td><td class="parity parity-exact">✓ J 0.80</td><td class="ms">4.21 ms</td></tr>
<tr class="bk-row"><td class="bk-name"><code>pls4all.R.formula</code></td><td class="parity parity-exact">✓ J 0.80</td><td class="ms">6.34 ms</td></tr>
<tr class="bk-row"><td class="bk-name"><code>pls4all.R.mdatools</code></td><td class="parity parity-exact">✓ J 0.80</td><td class="ms">6.75 ms</td></tr>
<tr class="bk-row"><td class="bk-name"><code>pls4all.R.pls</code></td><td class="parity parity-exact">✓ J 0.80</td><td class="ms">5.80 ms</td></tr>
</tbody>
<tbody class="lang-band lang-r"><tr class="lang-band-row" data-lang="r"><th colspan="3" scope="rowgroup"><span class="lang-band-dot"></span>R · external</th></tr>
<tr class="bk-row truth-source truth-source-qualitative"><td class="bk-name"><span class="truth-mark" title="Registry parity reference (r): plsVarSel 0.10.0 — qualitative (rmse_rel ≤ 2e+00)">📐</span><code>ref.r_plsvarsel</code></td><td class="parity parity-ref-source">source</td><td class="ms">37.0 ms</td></tr>
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
<tr class="bk-row"><td class="bk-name"><code>pls4all.cpp.blas+omp</code></td><td class="parity parity-cross_check">⇄ J 0.80</td><td class="ms ms-best">1.67 ms<span class="medal" title="fastest">🏆</span></td></tr>
</tbody>
<tbody class="lang-band lang-python"><tr class="lang-band-row" data-lang="python"><th colspan="3" scope="rowgroup"><span class="lang-band-dot"></span>Python · archived pls4all benchmark</th></tr>
<tr class="bk-row"><td class="bk-name"><code>pls4all.python</code></td><td class="parity parity-exact">✓ J 0.80</td><td class="ms">1.69 ms</td></tr>
<tr class="bk-row"><td class="bk-name"><code>pls4all.sklearn</code></td><td class="parity parity-exact">✓ J 0.80</td><td class="ms">1.89 ms</td></tr>
</tbody>
<tbody class="lang-band lang-r"><tr class="lang-band-row" data-lang="r"><th colspan="3" scope="rowgroup"><span class="lang-band-dot"></span>R · archived pls4all benchmark</th></tr>
<tr class="bk-row"><td class="bk-name"><code>pls4all.R</code></td><td class="parity parity-exact">✓ J 0.80</td><td class="ms">5.03 ms</td></tr>
<tr class="bk-row"><td class="bk-name"><code>pls4all.R.formula</code></td><td class="parity parity-exact">✓ J 0.80</td><td class="ms">5.50 ms</td></tr>
<tr class="bk-row"><td class="bk-name"><code>pls4all.R.mdatools</code></td><td class="parity parity-exact">✓ J 0.80</td><td class="ms">5.35 ms</td></tr>
<tr class="bk-row"><td class="bk-name"><code>pls4all.R.pls</code></td><td class="parity parity-exact">✓ J 0.80</td><td class="ms">5.34 ms</td></tr>
</tbody>
<tbody class="lang-band lang-r"><tr class="lang-band-row" data-lang="r"><th colspan="3" scope="rowgroup"><span class="lang-band-dot"></span>R · external</th></tr>
<tr class="bk-row truth-source truth-source-qualitative"><td class="bk-name"><span class="truth-mark" title="Registry parity reference (r): plsVarSel 0.10.0 — qualitative (rmse_rel ≤ 2e+00)">📐</span><code>ref.r_plsvarsel</code></td><td class="parity parity-ref-source">source</td><td class="ms">35.9 ms</td></tr>
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
<tr class="bk-row"><td class="bk-name"><code>pls4all.cpp.blas+omp</code></td><td class="parity parity-cross_check">⇄ J 0.80</td><td class="ms ms-best">1.66 ms<span class="medal" title="fastest">🏆</span></td></tr>
</tbody>
<tbody class="lang-band lang-python"><tr class="lang-band-row" data-lang="python"><th colspan="3" scope="rowgroup"><span class="lang-band-dot"></span>Python · archived pls4all benchmark</th></tr>
<tr class="bk-row"><td class="bk-name"><code>pls4all.python</code></td><td class="parity parity-exact">✓ J 0.80</td><td class="ms">1.76 ms</td></tr>
<tr class="bk-row"><td class="bk-name"><code>pls4all.sklearn</code></td><td class="parity parity-exact">✓ J 0.80</td><td class="ms">1.83 ms</td></tr>
</tbody>
<tbody class="lang-band lang-r"><tr class="lang-band-row" data-lang="r"><th colspan="3" scope="rowgroup"><span class="lang-band-dot"></span>R · archived pls4all benchmark</th></tr>
<tr class="bk-row"><td class="bk-name"><code>pls4all.R</code></td><td class="parity parity-exact">✓ J 0.80</td><td class="ms">4.63 ms</td></tr>
<tr class="bk-row"><td class="bk-name"><code>pls4all.R.formula</code></td><td class="parity parity-exact">✓ J 0.80</td><td class="ms">5.24 ms</td></tr>
<tr class="bk-row"><td class="bk-name"><code>pls4all.R.mdatools</code></td><td class="parity parity-exact">✓ J 0.80</td><td class="ms">5.78 ms</td></tr>
<tr class="bk-row"><td class="bk-name"><code>pls4all.R.pls</code></td><td class="parity parity-exact">✓ J 0.80</td><td class="ms">5.58 ms</td></tr>
</tbody>
<tbody class="lang-band lang-r"><tr class="lang-band-row" data-lang="r"><th colspan="3" scope="rowgroup"><span class="lang-band-dot"></span>R · external</th></tr>
<tr class="bk-row truth-source truth-source-qualitative"><td class="bk-name"><span class="truth-mark" title="Registry parity reference (r): plsVarSel 0.10.0 — qualitative (rmse_rel ≤ 2e+00)">📐</span><code>ref.r_plsvarsel</code></td><td class="parity parity-ref-source">source</td><td class="ms">35.4 ms</td></tr>
</tbody>
</table>
</div>

:::

::::


---

_See also_: [benchmark overview](../benchmarks/overview.md) · [methods index](index.md) · [interactive dashboard](../landing/dashboard.md)