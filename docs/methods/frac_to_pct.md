# `frac_to_pct` — Fraction to percent

_Group_: **Preprocessing** · _Registry tolerance_: `rtol=1e-5`, `atol=1e-8` · _Source_: [`docs/algorithms/fraction_to_percent.md`](../algorithms/fraction_to_percent.md)

## Description

From the `n4m.FractionToPercent` Python wrapper docstring:

> Convert fractional reflectance/transmittance to percent.

### Parameters

None — the transform is fully data-driven.

## Explanations

### Bibliographic source

Implementation matches `nirs4all.operators.transforms.signal_conversion.FractionToPercent` (0.8.11).

### Mathematical principle

Element-wise multiplication by 100:

$$
x'_{i,j} = 100 \cdot x_{i,j}
$$

### Implementation

* The multiply uses `* 100.0`. `100.0` is exactly representable in IEEE 754 binary64, so the multiply is bit-exact — equivalent to a divide by `0.01` only in exact arithmetic; in floating point the two forms differ in the last bit. We use the literal multiply to match nirs4all's `X * 100.0`.
* Parity tolerance vs nirs4all reference: `1e-12 abs / 1e-13 rel`.

C ABI entry points used by the language bindings:

```c
n4m_status_t n4m_pp_frac_to_pct_create( n4m_pp_frac_to_pct_handle_t** out);
void n4m_pp_frac_to_pct_destroy( n4m_pp_frac_to_pct_handle_t* handle);
n4m_status_t n4m_pp_frac_to_pct_transform( const n4m_pp_frac_to_pct_handle_t* handle, n4m_matrix_view_t X, n4m_matrix_view_t out);
```

Benchmark comparator backends are registered in the matrix and stored as reproducible snapshots when they define the canonical contract.

### Implementations

| Layer | Entry point | Language | Contract |
|-------|-------------|----------|----------|
| C ABI | `n4m_pp_frac_to_pct` | C/C++ | Stable libn4m entry point family. |
| Python | `n4m.python.frac_to_pct` | Python | ABI-close function backed by ctypes. |
| Python sklearn | `n4m.sklearn.FractionToPercent` | Python | scikit-learn-compatible estimator backed by ctypes. |
| R | `fraction_to_percent(X)` | R | Public package wrapper around the C ABI. |
| ref.nirs4all | `nirs4all.FractionToPercent` | Python | canonical/comparator |

### Usage

Every nirs4all-methods binding dispatches into the same C kernel. Registered comparator/source rows are listed in the benchmark card below.

::::{tab-set}
:class: nirs4all-methods-bindings


:::{tab-item} C ABI · libn4m
:sync: c
:class-label: lang-c

```c
n4m_status_t n4m_pp_frac_to_pct_create( n4m_pp_frac_to_pct_handle_t** out);
void n4m_pp_frac_to_pct_destroy( n4m_pp_frac_to_pct_handle_t* handle);
n4m_status_t n4m_pp_frac_to_pct_transform( const n4m_pp_frac_to_pct_handle_t* handle, n4m_matrix_view_t X, n4m_matrix_view_t out);
```

:::

:::{tab-item} Python ABI · n4m.python
:sync: python-abi
:class-label: lang-python

```python
from n4m import python as n4m

Xt = n4m.frac_to_pct(X)
```

:::

:::{tab-item} Python sklearn · n4m.sklearn
:sync: python-sklearn
:class-label: lang-python

```python
from n4m.sklearn import FractionToPercent

op = FractionToPercent()
Xt = op.fit_transform(X)
```

:::

:::{tab-item} R · nirs4all-methods
:sync: r
:class-label: lang-r

```r
library(n4m)
res <- fraction_to_percent(X)
```

:::

::::


**Benchmark Comparators And Sources** ◆

:::{card}
:class-card: external-refs

- ◆ **`ref.nirs4all`** (Python · canonical) — `nirs4all.FractionToPercent` · nirs4all@cd731a23+dirty
:::

### Validation contract

- Operation: `cross_binding_callable` · comparator: `default_allclose` · tolerance: `rtol=1e-05`, `atol=1e-08` · quality: **strict**
- Default validation dataset: `100×50` · seed `20260556`
- Suites: smoke `3` cells; benchmark `11` cells · Default C/Python/reference parity comparator.
- Metrics: `max_abs_diff`, `rel_l2_diff`, `rms_diff`, `shape_equal`
- Truth sources: cross-binding references declared directly in `benchmarks/cross_binding/orchestrator.py`.

| Backend | Library | Gate | Comparator | Note |
|---------|---------|------|------------|------|
| `ref.nirs4all` | `nirs4all.FractionToPercent` | Python / parity | `default_allclose` |  |

### Benchmarks
Median wall-clock per cell from [`docs/_static/bench-data.json`](../benchmarks/overview.md). Divergence is the worst finite value over the visible sizes for each backend, preferring reference max-abs difference and falling back to binding max-abs difference when no reference comparison is recorded. Rows without a recorded comparison show `—`; the fastest backend per column is marked 🏆.
::::{tab-set}
:class: parity-tabs

:::{tab-item} 1 thread
:sync: threads-1

<div class="parity-table-wrap">
<table class="docutils parity-grouped">
<thead><tr><th>Backend</th><th>Divergence</th><th>100×50</th><th>100×500</th><th>100×2500</th></tr></thead>
<tbody class="lang-band lang-cpp"><tr class="lang-band-row" data-lang="cpp"><th colspan="5" scope="rowgroup"><span class="lang-band-dot"></span>C++ native · libn4m</th></tr>
<tr class="bk-row"><td class="bk-name"><code>N4M.cpp</code></td><td class="parity parity-divergence parity-exact" title="worst reference max abs diff over visible sizes">0</td><td class="ms ms-best">🏆 0.001 ms</td><td class="ms ms-best">🏆 0.007 ms</td><td class="ms ms-best">🏆 0.071 ms</td></tr>
</tbody>
<tbody class="lang-band lang-python"><tr class="lang-band-row" data-lang="python"><th colspan="5" scope="rowgroup"><span class="lang-band-dot"></span>Python · nirs4all-methods</th></tr>
<tr class="bk-row"><td class="bk-name"><code>N4M.python</code></td><td class="parity parity-divergence parity-exact" title="worst reference max abs diff over visible sizes">0</td><td class="ms">0.006 ms</td><td class="ms">0.016 ms</td><td class="ms">0.078 ms</td></tr>
<tr class="bk-row"><td class="bk-name"><code>N4M.sklearn</code></td><td class="parity parity-divergence parity-exact" title="worst reference max abs diff over visible sizes">0</td><td class="ms">0.008 ms</td><td class="ms">0.018 ms</td><td class="ms">0.084 ms</td></tr>
</tbody>
<tbody class="lang-band lang-r"><tr class="lang-band-row" data-lang="r"><th colspan="5" scope="rowgroup"><span class="lang-band-dot"></span>R · nirs4all-methods</th></tr>
<tr class="bk-row"><td class="bk-name"><code>N4M.R</code></td><td class="parity parity-divergence parity-exact" title="worst reference max abs diff over visible sizes">5.7e-14</td><td class="ms">0.027 ms</td><td class="ms">0.270 ms</td><td class="ms">1.656 ms</td></tr>
</tbody>
<tbody class="lang-band lang-python"><tr class="lang-band-row" data-lang="python"><th colspan="5" scope="rowgroup"><span class="lang-band-dot"></span>Python · external</th></tr>
<tr class="bk-row truth-source-strict"><td class="bk-name"><span class="truth-mark" title="Registry parity reference (Python): nirs4all.FractionToPercent · nirs4all@cd731a23+dirty — canonical">◆</span><code>ref.nirs4all</code></td><td class="parity parity-divergence parity-exact" title="worst reference max abs diff over visible sizes">0</td><td class="ms">0.005 ms</td><td class="ms">0.026 ms</td><td class="ms">0.186 ms</td></tr>
</tbody>
</table>
</div>

:::
::::

---

_See also_: [benchmark overview](../benchmarks/overview.md) · [methods index](index.md) · [interactive dashboard](../landing/dashboard.md)
