# `weighted_snv` — Weighted SNV

_Group_: **Preprocessing** · _Registry tolerance_: `rtol=1e-5`, `atol=1e-8`

## Description

`weighted_snv` is a chemometrics4all preprocessing method exposed through the C ABI and the generated language bindings.

### Parameters

No public constructor parameters are required for the documented default call.

## Explanations

### Bibliographic source

- `ref.statsmodels` — statsmodels.stats.weightstats.DescrStatsW (Python).

### Mathematical principle

`weighted_snv` follows the standard preprocessing operator contract: an input matrix $\mathbf{X}\in\mathbb{R}^{n\times p}$ is transformed, scored, split, or filtered by the method-specific kernel while preserving the parity contract recorded by the benchmark snapshots. The precise numerical convention is the one exposed by the C ABI signatures and the registered external references below.

### Implementation

Benchmark comparator backends are registered in the matrix and stored as reproducible snapshots when they define the canonical contract.

### Implementations

| Layer | Entry point | Language | Contract |
|-------|-------------|----------|----------|
| C ABI | — | C/C++ | Stable libc4a entry point family. |
| Python | `chemometrics4all.python.weighted_snv` | Python | ABI-close function backed by ctypes. |
| R | `weighted_snv(X, weights = NULL, ddof = 0L, eps = 1e-12)` | R | Public package wrapper around the C ABI. |
| ref.statsmodels | `statsmodels.stats.weightstats.DescrStatsW` | Python | canonical/comparator |

### Usage

Every chemometrics4all binding dispatches into the same C kernel. Registered comparator/source rows are listed in the benchmark card below.

::::{tab-set}
:class: chemometrics4all-bindings


:::{tab-item} C ABI · libc4a
:sync: c
:class-label: lang-c

```c
/* C ABI prefix */
c4a_*
```

:::

:::{tab-item} Python ABI · chemometrics4all.python
:sync: python-abi
:class-label: lang-python

```python
from chemometrics4all import python as c4a

Xt = c4a.weighted_snv(X, weights=weights)
```

:::

:::{tab-item} R · chemometrics4all
:sync: r
:class-label: lang-r

```r
library(chemometrics4all)
res <- weighted_snv(X, weights = seq(1, 2, length.out = ncol(X)), ddof = 0L, eps = 1e-12)
```

:::

::::


**Benchmark Comparators And Sources** ◆

:::{card}
:class-card: external-refs

- ◆ **`ref.statsmodels`** (Python · canonical) — `statsmodels.stats.weightstats.DescrStatsW` · statsmodels 0.14.6
:::

### Validation contract

- Operation: `fit_transform` · comparator: `default_allclose` · tolerance: `rtol=1e-05`, `atol=1e-08` · quality: **strict**
- Default validation dataset: `64×128` · seed `1234567904`
- Suites: smoke `3` cells; benchmark `11` cells · Default C/Python/reference parity comparator.
- Metrics: `finite_output`, `max_abs_diff`, `shape_equal`
- Truth sources:
  - `vsn_2020` — VSN: variable sorting for normalization (doi:10.1002/cem.3164)

| Backend | Library | Gate | Comparator | Note |
|---------|---------|------|------------|------|
| `ref.statsmodels` | `statsmodels.stats.weightstats.DescrStatsW` | Python / parity | `default_allclose` |  |

### Benchmarks
Median wall-clock per cell from [`docs/_static/bench-data.json`](../benchmarks/overview.md). Divergence is the worst finite value over the visible sizes for each backend, preferring reference max-abs difference and falling back to binding max-abs difference when no reference comparison is recorded. Rows without a recorded comparison show `—`; the fastest backend per column is marked 🏆.
::::{tab-set}
:class: parity-tabs

:::{tab-item} 1 thread
:sync: threads-1

<div class="parity-table-wrap">
<table class="docutils parity-grouped">
<thead><tr><th>Backend</th><th>Divergence</th><th>100×50</th><th>100×500</th><th>100×2500</th></tr></thead>
<tbody class="lang-band lang-cpp"><tr class="lang-band-row" data-lang="cpp"><th colspan="5" scope="rowgroup"><span class="lang-band-dot"></span>C++ native · libc4a</th></tr>
<tr class="bk-row"><td class="bk-name"><code>C4A.cpp</code></td><td class="parity parity-divergence parity-exact" title="worst reference max abs diff over visible sizes">2.5e-14</td><td class="ms">0.018 ms</td><td class="ms">0.080 ms</td><td class="ms">0.362 ms</td></tr>
</tbody>
<tbody class="lang-band lang-python"><tr class="lang-band-row" data-lang="python"><th colspan="5" scope="rowgroup"><span class="lang-band-dot"></span>Python · chemometrics4all</th></tr>
<tr class="bk-row"><td class="bk-name"><code>C4A.python</code></td><td class="parity parity-divergence parity-exact" title="worst reference max abs diff over visible sizes">2.5e-14</td><td class="ms ms-best">🏆 0.014 ms</td><td class="ms ms-best">🏆 0.073 ms</td><td class="ms">0.357 ms</td></tr>
<tr class="bk-row"><td class="bk-name"><code>C4A.sklearn</code></td><td class="parity parity-divergence parity-exact" title="worst reference max abs diff over visible sizes">2.5e-14</td><td class="ms">0.017 ms</td><td class="ms">0.083 ms</td><td class="ms ms-best">🏆 0.350 ms</td></tr>
</tbody>
<tbody class="lang-band lang-r"><tr class="lang-band-row" data-lang="r"><th colspan="5" scope="rowgroup"><span class="lang-band-dot"></span>R · chemometrics4all</th></tr>
<tr class="bk-row"><td class="bk-name"><code>C4A.R</code></td><td class="parity parity-divergence parity-exact" title="worst reference max abs diff over visible sizes">3.0e-14</td><td class="ms">0.035 ms</td><td class="ms">0.363 ms</td><td class="ms">1.969 ms</td></tr>
</tbody>
<tbody class="lang-band lang-python"><tr class="lang-band-row" data-lang="python"><th colspan="5" scope="rowgroup"><span class="lang-band-dot"></span>Python · external</th></tr>
<tr class="bk-row truth-source-strict"><td class="bk-name"><span class="truth-mark" title="Registry parity reference (Python): statsmodels.stats.weightstats.DescrStatsW · statsmodels 0.14.6 — canonical">◆</span><code>ref.statsmodels</code></td><td class="parity parity-divergence parity-exact" title="worst reference max abs diff over visible sizes">0</td><td class="ms">0.439 ms</td><td class="ms">0.510 ms</td><td class="ms">0.925 ms</td></tr>
</tbody>
</table>
</div>

:::
::::

---

_See also_: [benchmark overview](../benchmarks/overview.md) · [methods index](index.md) · [interactive dashboard](../landing/dashboard.md)
