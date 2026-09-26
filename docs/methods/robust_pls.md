# `robust_pls` — Robust PLS (Partial Robust M-regression)

_Group_: **Robust / weighted** · _Registry tolerance_: `1e-08`

## Description

Robust PLS (Partial Robust M-regression, Serneels 2005)

> **Registry note** — R `chemometrics::prm` (Serneels et al. 2005) — Partial Robust M-regression. pls4all defaults to PRM matching the R algorithm bit-for-bit (median centering, Fair weights on leverage + residual, univariate SIMPLS inner kernel, intercept = median(y - X@b)). The legacy Huber-IRLS over weighted SIMPLS path is reachable via ``cfg.robust_pls_legacy = 1``.

### Parameters

| Name | Type | Default | Notes |
|------|------|---------|-------|
| `X` | `—` | `required` | current public binding signature |
| `y` | `—` | `required` | current public binding signature |
| `huber_k` | `float` | `1.345` | current public binding signature |
| `max_irls_iter` | `int` | `5` | current public binding signature |
| `n_components` | `int` | `2` | current public binding signature |
| `center_x` | `bool \| None` | `None` | current public binding signature |
| `scale_x` | `bool \| None` | `None` | current public binding signature |
| `center_y` | `bool \| None` | `None` | current public binding signature |
| `scale_y` | `bool \| None` | `None` | current public binding signature |

## Explanations

### Bibliographic source

Serneels, S., Croux, C., Filzmoser, P. & Van Espen, P. J. (2005). *Partial robust M-regression*. Chemometrics and Intelligent Laboratory Systems 79(1--2), 55--64. DOI [10.1016/j.chemolab.2005.04.007](https://doi.org/10.1016/j.chemolab.2005.04.007).

### Mathematical principle

The default robust path follows a partial robust M-regression style iteration: residual and leverage information update Fair robustness weights before weighted latent fits. Fair weights decay smoothly rather than clipping residual influence at the Huber threshold.

### Appropriate uses

Calibration containing moderate response or leverage contamination where robust residual weighting is justified.

### Limits and validation

The default PRM-style path uses Fair weights and reinterprets huber_k as its tuning constant; Huber IRLS is available only in legacy mode.

### Implementation

In the default path the public `huber_k` value is interpreted as the Fair tuning constant for parity with the referenced implementation. The separately gated legacy branch performs the Huber IRLS algorithm described by older documentation.

### Sources and provenance

Current implementation: [cpp/src/core/extra_pls.cpp](https://github.com/GBeurier/nirs4all-methods/blob/main/cpp/src/core/extra_pls.cpp).


### API and bindings

**C ABI (ABI 2):** [`n4m_estimators_robust_pls_fit`](https://github.com/GBeurier/nirs4all-methods/blob/main/cpp/include/n4m/estimators/regression.h#L97). Use the linked public header for the exact signature, configuration, and result handles.

**Python (verified public re-export):**

```python
from n4m.estimators.regression.robust import robust_pls
result = robust_pls(X, y)
```

Source signature: `robust_pls(X, y, *, huber_k: float = 1.345, max_irls_iter: int = 5, n_components: int = 2, center_x: bool | None = None, scale_x: bool | None = None, center_y: bool | None = None, scale_y: bool | None = None)` ([`n4m/_impl/native.py`](https://github.com/GBeurier/nirs4all-methods/blob/main/bindings/python/src/n4m/_impl/native.py#L8677)).

**R (source-verified):** [`robust_pls(formula, data, ncomp = 2L, huber_k = 1.345, max_irls_iter = 20L, na.action = stats::na.omit)`](https://github.com/GBeurier/nirs4all-methods/blob/main/bindings/r/n4m/R/sklearn_extra.R).

The source signature has additional required inputs, so no example call is fabricated.

**MATLAB / Octave (source-verified):** [`robust_pls(X, Y, n_components, huber_k, max_irls_iter)`](https://github.com/GBeurier/nirs4all-methods/blob/main/bindings/matlab/+n4m/robust_pls.m).

The source signature has additional required inputs, so no example call is fabricated.

**Registry parity references** 📐

:::{card}
:class-card: external-refs

- 📐 **`ref.r_chemometrics`** (R · r) — `chemometrics` 0.7.x · strict (rmse_rel ≤ 1e-08) — R `chemometrics::prm` (Partial Robust M-regression). pls4all uses Huber IRLS over weighted SIMPLS; this is an M-estimator variant from the same family. Predictions diverge by O(0.5).

:::

### Benchmarks

**Archived measurement identity.** Backend labels in this table are the raw IDs recorded when the benchmark ran (including historical `pls4all.*` IDs). They preserve measurement provenance and do not describe a current public Python, R, or MATLAB binding; use the source-verified **API and bindings** section above for current entry points.

Adaptive wall-clock per cell measured against [`full_matrix.csv`](../benchmarks/overview.md). Only backends that implement this method are listed; libraries without the method are omitted.

**Verdict** &nbsp;·&nbsp; ✓ ref / ≈ ref / ~ shape mark a reference-gate pass at strict / relaxed / qualitative tolerance &nbsp;·&nbsp; ✓ bind = archived binding-harness result agrees with the C++ baseline &nbsp;·&nbsp; ⇄ cross-check = documented by-design selector/RNG/model, noncanonical API/facade convention, or secondary oracle &nbsp;·&nbsp; ✗ divergent &nbsp;·&nbsp; ⚠ error &nbsp;·&nbsp; — not run. The fastest backend per column is marked 🏆.

**Reference gate**: strict — numeric equivalence (`rmse_rel_tol ≤ 1e-08`).

Rows tagged with **📐** are the canonical parity references for this method (declared in [`parity_timing.registry`](../benchmarks/methodology.md)). C++ and external rows show reference parity; archived language-harness rows show binding parity against the C++ backend. Hover the icon for role and tolerance band.

::::{tab-set}
:class: parity-tabs

:::{tab-item} 1 thread
:sync: threads-1

<div class="parity-table-wrap">
<table class="docutils parity-grouped">
<thead><tr><th scope="col">Backend</th><th scope="col">Parity</th><th class="size-col" scope="col">200×50 (ms)</th></tr></thead>
<tbody class="lang-band lang-cpp"><tr class="lang-band-row" data-lang="cpp"><th colspan="3" scope="rowgroup"><span class="lang-band-dot"></span>C++ native · libn4m</th></tr>
<tr class="bk-row"><td class="bk-name"><code>pls4all.cpp.blas+omp</code></td><td class="parity parity-ref-strict">✓ ref 1e-14</td><td class="ms">2.07 ms</td></tr>
</tbody>
<tbody class="lang-band lang-python"><tr class="lang-band-row" data-lang="python"><th colspan="3" scope="rowgroup"><span class="lang-band-dot"></span>Python · archived pls4all benchmark</th></tr>
<tr class="bk-row"><td class="bk-name"><code>pls4all.python</code></td><td class="parity parity-exact">✓ bind</td><td class="ms ms-best">1.95 ms<span class="medal" title="fastest">🏆</span></td></tr>
<tr class="bk-row"><td class="bk-name"><code>pls4all.sklearn</code></td><td class="parity parity-exact">✓ bind</td><td class="ms">2.15 ms</td></tr>
</tbody>
<tbody class="lang-band lang-r"><tr class="lang-band-row" data-lang="r"><th colspan="3" scope="rowgroup"><span class="lang-band-dot"></span>R · archived pls4all benchmark</th></tr>
<tr class="bk-row"><td class="bk-name"><code>pls4all.R</code></td><td class="parity parity-exact">✓ bind</td><td class="ms">4.38 ms</td></tr>
<tr class="bk-row"><td class="bk-name"><code>pls4all.R.formula</code></td><td class="parity parity-exact">✓ bind</td><td class="ms">5.55 ms</td></tr>
<tr class="bk-row"><td class="bk-name"><code>pls4all.R.mdatools</code></td><td class="parity parity-exact">✓ bind</td><td class="ms">5.77 ms</td></tr>
<tr class="bk-row"><td class="bk-name"><code>pls4all.R.pls</code></td><td class="parity parity-exact">✓ bind</td><td class="ms">6.62 ms</td></tr>
</tbody>
<tbody class="lang-band lang-r"><tr class="lang-band-row" data-lang="r"><th colspan="3" scope="rowgroup"><span class="lang-band-dot"></span>R · external</th></tr>
<tr class="bk-row truth-source truth-source-strict"><td class="bk-name"><span class="truth-mark" title="Registry parity reference (r): chemometrics 0.7.x — strict (rmse_rel ≤ 1e-08)">📐</span><code>ref.r_chemometrics</code></td><td class="parity parity-ref-source">source</td><td class="ms">17.9 ms</td></tr>
</tbody>
</table>
</div>

:::

:::{tab-item} 3 threads
:sync: threads-3

<div class="parity-table-wrap">
<table class="docutils parity-grouped">
<thead><tr><th scope="col">Backend</th><th scope="col">Parity</th><th class="size-col" scope="col">200×50 (ms)</th></tr></thead>
<tbody class="lang-band lang-cpp"><tr class="lang-band-row" data-lang="cpp"><th colspan="3" scope="rowgroup"><span class="lang-band-dot"></span>C++ native · libn4m</th></tr>
<tr class="bk-row"><td class="bk-name"><code>pls4all.cpp.blas+omp</code></td><td class="parity parity-ref-strict">✓ ref 1e-14</td><td class="ms">2.02 ms</td></tr>
</tbody>
<tbody class="lang-band lang-python"><tr class="lang-band-row" data-lang="python"><th colspan="3" scope="rowgroup"><span class="lang-band-dot"></span>Python · archived pls4all benchmark</th></tr>
<tr class="bk-row"><td class="bk-name"><code>pls4all.python</code></td><td class="parity parity-exact">✓ bind</td><td class="ms ms-best">2.00 ms<span class="medal" title="fastest">🏆</span></td></tr>
<tr class="bk-row"><td class="bk-name"><code>pls4all.sklearn</code></td><td class="parity parity-exact">✓ bind</td><td class="ms">2.10 ms</td></tr>
</tbody>
<tbody class="lang-band lang-r"><tr class="lang-band-row" data-lang="r"><th colspan="3" scope="rowgroup"><span class="lang-band-dot"></span>R · archived pls4all benchmark</th></tr>
<tr class="bk-row"><td class="bk-name"><code>pls4all.R</code></td><td class="parity parity-exact">✓ bind</td><td class="ms">5.43 ms</td></tr>
<tr class="bk-row"><td class="bk-name"><code>pls4all.R.formula</code></td><td class="parity parity-exact">✓ bind</td><td class="ms">6.63 ms</td></tr>
<tr class="bk-row"><td class="bk-name"><code>pls4all.R.mdatools</code></td><td class="parity parity-exact">✓ bind</td><td class="ms">5.84 ms</td></tr>
<tr class="bk-row"><td class="bk-name"><code>pls4all.R.pls</code></td><td class="parity parity-exact">✓ bind</td><td class="ms">5.99 ms</td></tr>
</tbody>
<tbody class="lang-band lang-r"><tr class="lang-band-row" data-lang="r"><th colspan="3" scope="rowgroup"><span class="lang-band-dot"></span>R · external</th></tr>
<tr class="bk-row truth-source truth-source-strict"><td class="bk-name"><span class="truth-mark" title="Registry parity reference (r): chemometrics 0.7.x — strict (rmse_rel ≤ 1e-08)">📐</span><code>ref.r_chemometrics</code></td><td class="parity parity-ref-source">source</td><td class="ms">17.0 ms</td></tr>
</tbody>
</table>
</div>

:::

:::{tab-item} 10 threads
:sync: threads-10

<div class="parity-table-wrap">
<table class="docutils parity-grouped">
<thead><tr><th scope="col">Backend</th><th scope="col">Parity</th><th class="size-col" scope="col">200×50 (ms)</th></tr></thead>
<tbody class="lang-band lang-cpp"><tr class="lang-band-row" data-lang="cpp"><th colspan="3" scope="rowgroup"><span class="lang-band-dot"></span>C++ native · libn4m</th></tr>
<tr class="bk-row"><td class="bk-name"><code>pls4all.cpp.blas+omp</code></td><td class="parity parity-ref-strict">✓ ref 1e-14</td><td class="ms">2.05 ms</td></tr>
</tbody>
<tbody class="lang-band lang-python"><tr class="lang-band-row" data-lang="python"><th colspan="3" scope="rowgroup"><span class="lang-band-dot"></span>Python · archived pls4all benchmark</th></tr>
<tr class="bk-row"><td class="bk-name"><code>pls4all.python</code></td><td class="parity parity-exact">✓ bind</td><td class="ms ms-best">1.98 ms<span class="medal" title="fastest">🏆</span></td></tr>
<tr class="bk-row"><td class="bk-name"><code>pls4all.sklearn</code></td><td class="parity parity-exact">✓ bind</td><td class="ms">2.25 ms</td></tr>
</tbody>
<tbody class="lang-band lang-r"><tr class="lang-band-row" data-lang="r"><th colspan="3" scope="rowgroup"><span class="lang-band-dot"></span>R · archived pls4all benchmark</th></tr>
<tr class="bk-row"><td class="bk-name"><code>pls4all.R</code></td><td class="parity parity-exact">✓ bind</td><td class="ms">4.94 ms</td></tr>
<tr class="bk-row"><td class="bk-name"><code>pls4all.R.formula</code></td><td class="parity parity-exact">✓ bind</td><td class="ms">7.07 ms</td></tr>
<tr class="bk-row"><td class="bk-name"><code>pls4all.R.mdatools</code></td><td class="parity parity-exact">✓ bind</td><td class="ms">5.59 ms</td></tr>
<tr class="bk-row"><td class="bk-name"><code>pls4all.R.pls</code></td><td class="parity parity-exact">✓ bind</td><td class="ms">5.68 ms</td></tr>
</tbody>
<tbody class="lang-band lang-r"><tr class="lang-band-row" data-lang="r"><th colspan="3" scope="rowgroup"><span class="lang-band-dot"></span>R · external</th></tr>
<tr class="bk-row truth-source truth-source-strict"><td class="bk-name"><span class="truth-mark" title="Registry parity reference (r): chemometrics 0.7.x — strict (rmse_rel ≤ 1e-08)">📐</span><code>ref.r_chemometrics</code></td><td class="parity parity-ref-source">source</td><td class="ms">17.5 ms</td></tr>
</tbody>
</table>
</div>

:::

::::


---

_See also_: [benchmark overview](../benchmarks/overview.md) · [methods index](index.md) · [interactive dashboard](../landing/dashboard.md)