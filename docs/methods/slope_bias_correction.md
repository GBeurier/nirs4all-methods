# `slope_bias_correction` — Slope/bias correction

_Group_: **Calibration Transfer** · _Registry tolerance_: `rtol=1e-5`, `atol=1e-8`

## Description

`slope_bias_correction` is a chemometrics4all calibration transfer method exposed through the C ABI and the generated language bindings.

### Parameters

No public constructor parameters are required for the documented default call.

## Explanations

### Bibliographic source

- `ref.sklearn` — sklearn.linear_model.LinearRegression (Python).
- `ref.pycaltransfer` — pycaltransfer.slope_bias_correction (Python).

### Mathematical principle

`slope_bias_correction` follows the standard calibration transfer operator contract: an input matrix $\mathbf{X}\in\mathbb{R}^{n\times p}$ is transformed, scored, split, or filtered by the method-specific kernel while preserving the parity contract recorded by the benchmark snapshots. The precise numerical convention is the one exposed by the C ABI signatures and the registered external references below.

### Implementation

Benchmark comparator backends are registered in the matrix and stored as reproducible snapshots when they define the canonical contract.

### Implementations

| Layer | Entry point | Language | Contract |
|-------|-------------|----------|----------|
| C ABI | — | C/C++ | Stable libc4a entry point family. |
| Python | `chemometrics4all.python.slope_bias_correction` | Python | ABI-close function backed by ctypes. |
| R | `{target <- 2.0 * y + 1.0; slope_bias_correction(y, target, y)}` | R | Public package wrapper around the C ABI. |
| ref.sklearn | `sklearn.linear_model.LinearRegression` | Python | canonical/comparator |
| ref.pycaltransfer | `pycaltransfer.slope_bias_correction` | Python | canonical/comparator |

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

y_corr = c4a.slope_bias_correction(y_source, y_target, y)
```

:::

:::{tab-item} R · chemometrics4all
:sync: r
:class-label: lang-r

```r
library(chemometrics4all)
res <- {target <- 2.0 * y + 1.0; slope_bias_correction(y, target, y)}
```

:::

::::


**Benchmark Comparators And Sources** ◆

:::{card}
:class-card: external-refs

- ◆ **`ref.sklearn`** (Python · canonical) — `sklearn.linear_model.LinearRegression` · scikit-learn 1.8.0
- ◆ **`ref.pycaltransfer`** (Python · comparator) — `pycaltransfer.slope_bias_correction` · pycaltransfer 0.1.8
:::

### Validation contract

- Operation: `fit_transform_vector` · comparator: `default_allclose` · tolerance: `rtol=1e-05`, `atol=1e-08` · quality: **strict**
- Default validation dataset: `80×1` · seed `1234567902`
- Suites: smoke `3` cells; benchmark `11` cells · Default C/Python/reference parity comparator.
- Metrics: `finite_output`, `max_abs_diff`, `shape_equal`
- Truth sources:
  - `workman_2018_calibration_transfer` — A Review of Calibration Transfer Practices and Instrument Differences in Spectroscopy (doi:10.1177/0003702817736064)

| Backend | Library | Gate | Comparator | Note |
|---------|---------|------|------------|------|
| `ref.sklearn` | `sklearn.linear_model.LinearRegression` | Python / parity | `default_allclose` |  |
| `ref.pycaltransfer` | `pycaltransfer.slope_bias_correction` | Python / parity | `default_allclose` |  |

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
<tr class="bk-row"><td class="bk-name"><code>C4A.cpp</code></td><td class="parity parity-divergence parity-exact" title="worst reference max abs diff over visible sizes">2.6e-14</td><td class="ms">0.010 ms</td><td class="ms">0.010 ms</td><td class="ms">0.010 ms</td></tr>
</tbody>
<tbody class="lang-band lang-python"><tr class="lang-band-row" data-lang="python"><th colspan="5" scope="rowgroup"><span class="lang-band-dot"></span>Python · chemometrics4all</th></tr>
<tr class="bk-row"><td class="bk-name"><code>C4A.python</code></td><td class="parity parity-divergence parity-exact" title="worst reference max abs diff over visible sizes">2.6e-14</td><td class="ms">0.009 ms</td><td class="ms">0.010 ms</td><td class="ms">0.010 ms</td></tr>
<tr class="bk-row"><td class="bk-name"><code>C4A.sklearn</code></td><td class="parity parity-divergence parity-exact" title="worst reference max abs diff over visible sizes">2.6e-14</td><td class="ms">0.011 ms</td><td class="ms">0.011 ms</td><td class="ms">0.011 ms</td></tr>
</tbody>
<tbody class="lang-band lang-r"><tr class="lang-band-row" data-lang="r"><th colspan="5" scope="rowgroup"><span class="lang-band-dot"></span>R · chemometrics4all</th></tr>
<tr class="bk-row"><td class="bk-name"><code>C4A.R</code></td><td class="parity parity-divergence parity-exact" title="worst reference max abs diff over visible sizes">2.3e-14</td><td class="ms ms-best">🏆 0.002 ms</td><td class="ms ms-best">🏆 0.002 ms</td><td class="ms ms-best">🏆 0.002 ms</td></tr>
</tbody>
<tbody class="lang-band lang-python"><tr class="lang-band-row" data-lang="python"><th colspan="5" scope="rowgroup"><span class="lang-band-dot"></span>Python · external</th></tr>
<tr class="bk-row truth-source-relaxed"><td class="bk-name"><span class="truth-mark" title="Registry parity reference (Python): pycaltransfer.slope_bias_correction · pycaltransfer 0.1.8 — comparator">◆</span><code>ref.pycaltransfer</code></td><td class="parity parity-divergence parity-exact" title="worst reference max abs diff over visible sizes">0</td><td class="ms">0.235 ms</td><td class="ms">0.197 ms</td><td class="ms">0.197 ms</td></tr>
<tr class="bk-row truth-source-strict"><td class="bk-name"><span class="truth-mark" title="Registry parity reference (Python): sklearn.linear_model.LinearRegression · scikit-learn 1.8.0 — canonical">◆</span><code>ref.sklearn</code></td><td class="parity parity-divergence parity-exact" title="worst reference max abs diff over visible sizes">0</td><td class="ms">0.235 ms</td><td class="ms">0.235 ms</td><td class="ms">0.238 ms</td></tr>
</tbody>
</table>
</div>

:::
::::

---

_See also_: [benchmark overview](../benchmarks/overview.md) · [methods index](index.md) · [interactive dashboard](../landing/dashboard.md)
