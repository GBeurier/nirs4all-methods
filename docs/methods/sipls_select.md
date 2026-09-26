# `sipls_select` — siPLS — Synergy Interval PLS

_Group_: **Variable selector** · _Registry tolerance_: `0.7`
 · _Parity reference_: **paper-only** (Nørgaard, L., Saudland, A., Wagner, J., Nielsen, J. P., Munck, L. & Engelsen, S. B. (2000). Interval partial least-squares regression (iPLS). Appl. Spectrosc. 54(3), 413-419. (No widely installable R / Python port of siPLS' synergistic combinations; smoke-tested only.))

## Description

siPLS synergistic interval selection (§18 Phase 5q)

### Parameters

| Name | Type | Default | Notes |
|------|------|---------|-------|
| `n_components` | `int` | `2` | current public binding signature |
| `interval_width` | `int` | `10` | current public binding signature |
| `combination_size` | `int` | `2` | current public binding signature |
| `n_folds` | `int` | `3` | current public binding signature |

## Explanations

### Bibliographic source

Nørgaard, L., Saudland, A., Wagner, J., Nielsen, J. P., Munck, L. & Engelsen, S. B. (2000). *Interval partial least-squares regression (iPLS): a comparative chemometric study with an example from near-infrared spectroscopy*. Applied Spectroscopy 54(3), 413–419 — same paper as `interval_select`; siPLS is the synergy-combinations extension proposed in §3. Verified primary link: [https://doi.org/10.1366/0003702001949500](https://doi.org/10.1366/0003702001949500).

### Mathematical principle

Exhaustively score every combination of $m$ fixed-size intervals (out of $I$ total) by CV-RMSE; return the best combination. This captures **synergy** — pairs (or triples) of intervals that work well together even if neither alone is the best single interval.

Combinatorial cost is $\binom{I}{m}$ which is manageable only for small $m$ (typically $m \le 4$ with $I \le 20$, giving up to ~5000 combinations). For larger search spaces use biPLS or interval-GA.

siPLS is the natural extension of iPLS when single-interval performance is unsatisfactory: it explicitly looks for complementary bands.

### Appropriate uses

Evaluating combinations of a small number of spectral intervals for interpretable regional models.

### Limits and validation

Combination count grows rapidly and results depend on the initial interval grid and validation split.

### Implementation

`n4m_feature_selection_sipls_select`. Reference: R `plsVarSel`.

### Sources and provenance

Current implementation: [cpp/src/core/sipls_selection.cpp](https://github.com/GBeurier/nirs4all-methods/blob/main/cpp/src/core/sipls_selection.cpp).


### API and bindings

**C ABI (ABI 2):** [`n4m_feature_selection_sipls_select`](https://github.com/GBeurier/nirs4all-methods/blob/main/cpp/include/n4m/feature_selection.h#L324). Use the linked public header for the exact signature, configuration, and result handles.

**Python (verified public re-export):**

```python
from n4m.feature_selection.wrapper import SiPLS
```

Source signature: `SiPLS(*, n_components: int = 2, interval_width: int = 10, combination_size: int = 2, n_folds: int = 3)` ([`n4m/_impl/selection.py`](https://github.com/GBeurier/nirs4all-methods/blob/main/bindings/python/src/n4m/_impl/selection.py#L565)).

**R (source-verified):** [`sipls_select(X, Y, n_components, interval_width = 10L, combination_size = 2L)`](https://github.com/GBeurier/nirs4all-methods/blob/main/bindings/r/n4m/R/methods_extra.R).

The source signature has additional required inputs, so no example call is fabricated.

**MATLAB / Octave (source-verified):** [`sipls_select(X, Y, n_components, interval_width, combination_size)`](https://github.com/GBeurier/nirs4all-methods/blob/main/bindings/matlab/+n4m/sipls_select.m).

The source signature has additional required inputs, so no example call is fabricated.

**Registry parity references** 📐

:::{card}
:class-card: external-refs

- 📜 **Paper-only** — no executable parity reference; the paper citation is recorded in *Bibliographic source*.

:::

### Benchmarks

**Archived measurement identity.** Backend labels in this table are the raw IDs recorded when the benchmark ran (including historical `pls4all.*` IDs). They preserve measurement provenance and do not describe a current public Python, R, or MATLAB binding; use the source-verified **API and bindings** section above for current entry points.

Adaptive wall-clock per cell measured against [`full_matrix.csv`](../benchmarks/overview.md). Only backends that implement this method are listed; libraries without the method are omitted.

**Verdict** &nbsp;·&nbsp; ✓ ref / ≈ ref / ~ shape mark a reference-gate pass at strict / relaxed / qualitative tolerance &nbsp;·&nbsp; ✓ bind = archived binding-harness result agrees with the C++ baseline &nbsp;·&nbsp; ⇄ cross-check = documented by-design selector/RNG/model, noncanonical API/facade convention, or secondary oracle &nbsp;·&nbsp; ✗ divergent &nbsp;·&nbsp; ⚠ error &nbsp;·&nbsp; — not run. The fastest backend per column is marked 🏆.

::::{tab-set}
:class: parity-tabs

:::{tab-item} 1 thread
:sync: threads-1

<div class="parity-table-wrap">
<table class="docutils parity-grouped">
<thead><tr><th scope="col">Backend</th><th scope="col">Parity</th><th class="size-col" scope="col">200×40 (ms)</th></tr></thead>
<tbody class="lang-band lang-python"><tr class="lang-band-row" data-lang="python"><th colspan="3" scope="rowgroup"><span class="lang-band-dot"></span>Python · archived pls4all benchmark</th></tr>
<tr class="bk-row"><td class="bk-name"><code>pls4all.python</code></td><td class="parity parity-exact">✓ bind</td><td class="ms">3.18 ms</td></tr>
<tr class="bk-row"><td class="bk-name"><code>pls4all.sklearn</code></td><td class="parity parity-exact">✓ bind</td><td class="ms ms-best">2.82 ms<span class="medal" title="fastest">🏆</span></td></tr>
</tbody>
<tbody class="lang-band lang-r"><tr class="lang-band-row" data-lang="r"><th colspan="3" scope="rowgroup"><span class="lang-band-dot"></span>R · archived pls4all benchmark</th></tr>
<tr class="bk-row"><td class="bk-name"><code>pls4all.R</code></td><td class="parity parity-exact">✓ bind</td><td class="ms">7.86 ms</td></tr>
<tr class="bk-row"><td class="bk-name"><code>pls4all.R.formula</code></td><td class="parity parity-exact">✓ bind</td><td class="ms">8.15 ms</td></tr>
<tr class="bk-row"><td class="bk-name"><code>pls4all.R.mdatools</code></td><td class="parity parity-exact">✓ bind</td><td class="ms">8.80 ms</td></tr>
<tr class="bk-row"><td class="bk-name"><code>pls4all.R.pls</code></td><td class="parity parity-exact">✓ bind</td><td class="ms">8.12 ms</td></tr>
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
<tr class="bk-row"><td class="bk-name"><code>pls4all.python</code></td><td class="parity parity-exact">✓ bind</td><td class="ms ms-best">2.97 ms<span class="medal" title="fastest">🏆</span></td></tr>
<tr class="bk-row"><td class="bk-name"><code>pls4all.sklearn</code></td><td class="parity parity-exact">✓ bind</td><td class="ms">3.11 ms</td></tr>
</tbody>
<tbody class="lang-band lang-r"><tr class="lang-band-row" data-lang="r"><th colspan="3" scope="rowgroup"><span class="lang-band-dot"></span>R · archived pls4all benchmark</th></tr>
<tr class="bk-row"><td class="bk-name"><code>pls4all.R</code></td><td class="parity parity-exact">✓ bind</td><td class="ms">6.18 ms</td></tr>
<tr class="bk-row"><td class="bk-name"><code>pls4all.R.formula</code></td><td class="parity parity-exact">✓ bind</td><td class="ms">7.15 ms</td></tr>
<tr class="bk-row"><td class="bk-name"><code>pls4all.R.mdatools</code></td><td class="parity parity-exact">✓ bind</td><td class="ms">7.20 ms</td></tr>
<tr class="bk-row"><td class="bk-name"><code>pls4all.R.pls</code></td><td class="parity parity-exact">✓ bind</td><td class="ms">7.56 ms</td></tr>
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
<tr class="bk-row"><td class="bk-name"><code>pls4all.python</code></td><td class="parity parity-exact">✓ bind</td><td class="ms ms-best">2.57 ms<span class="medal" title="fastest">🏆</span></td></tr>
<tr class="bk-row"><td class="bk-name"><code>pls4all.sklearn</code></td><td class="parity parity-exact">✓ bind</td><td class="ms">2.73 ms</td></tr>
</tbody>
<tbody class="lang-band lang-r"><tr class="lang-band-row" data-lang="r"><th colspan="3" scope="rowgroup"><span class="lang-band-dot"></span>R · archived pls4all benchmark</th></tr>
<tr class="bk-row"><td class="bk-name"><code>pls4all.R</code></td><td class="parity parity-exact">✓ bind</td><td class="ms">6.76 ms</td></tr>
<tr class="bk-row"><td class="bk-name"><code>pls4all.R.formula</code></td><td class="parity parity-exact">✓ bind</td><td class="ms">7.43 ms</td></tr>
<tr class="bk-row"><td class="bk-name"><code>pls4all.R.mdatools</code></td><td class="parity parity-exact">✓ bind</td><td class="ms">18.1 ms</td></tr>
<tr class="bk-row"><td class="bk-name"><code>pls4all.R.pls</code></td><td class="parity parity-exact">✓ bind</td><td class="ms">13.1 ms</td></tr>
</tbody>
</table>
</div>

:::

::::


---

_See also_: [benchmark overview](../benchmarks/overview.md) · [methods index](index.md) · [interactive dashboard](../landing/dashboard.md)