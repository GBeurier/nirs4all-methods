# `pct_to_frac` — Percent to fraction

_Group_: **Preprocessing** · _Registry tolerance_: `rtol=1e-5`, `atol=1e-8` · _Source_: [`docs/algorithms/percent_to_fraction.md`](../algorithms/percent_to_fraction.md)

## Description

From the `chemometrics4all.PercentToFraction` Python wrapper docstring:

> Convert percent reflectance/transmittance to fraction.

### Parameters

None — the transform is fully data-driven.

## Explanations

### Bibliographic source

Implementation matches `nirs4all.operators.transforms.signal_conversion.PercentToFraction` (0.8.11).

### Mathematical principle

Element-wise division by 100:

$$
x'_{i,j} = \frac{x_{i,j}}{100}
$$

### Implementation

* Division is performed inline as `X[i] / 100.0` — not replaced with multiplication by `0.01`, because `0.01` has no exact binary representation. The two forms diverge in the last bit; the inline divide matches numpy's element-wise true-divide bit-for-bit.
* Parity tolerance vs nirs4all reference: `1e-12 abs / 1e-13 rel`.

C ABI entry points used by the language bindings:

```c
c4a_status_t c4a_pp_pct_to_frac_create( c4a_pp_pct_to_frac_handle_t** out);
void c4a_pp_pct_to_frac_destroy( c4a_pp_pct_to_frac_handle_t* handle);
c4a_status_t c4a_pp_pct_to_frac_transform( const c4a_pp_pct_to_frac_handle_t* handle, c4a_matrix_view_t X, c4a_matrix_view_t out);
```

Benchmark comparator backends are registered in the matrix and stored as reproducible snapshots when they define the canonical contract.

### Implementations

| Layer | Entry point | Language | Contract |
|-------|-------------|----------|----------|
| C ABI | `c4a_pp_pct_to_frac` | C/C++ | Stable libc4a entry point family. |
| Python | `chemometrics4all.python.pct_to_frac` | Python | ABI-close function backed by ctypes. |
| Python sklearn | `chemometrics4all.sklearn.PercentToFraction` | Python | scikit-learn-compatible estimator backed by ctypes. |
| R | `percent_to_fraction(X)` | R | Public package wrapper around the C ABI. |
| ref.nirs4all | `nirs4all.PercentToFraction` | Python | canonical/comparator |

### Usage

Every chemometrics4all binding dispatches into the same C kernel. Registered comparator/source rows are listed in the benchmark card below.

::::{tab-set}
:class: chemometrics4all-bindings


:::{tab-item} C ABI · libc4a
:sync: c
:class-label: lang-c

```c
c4a_status_t c4a_pp_pct_to_frac_create( c4a_pp_pct_to_frac_handle_t** out);
void c4a_pp_pct_to_frac_destroy( c4a_pp_pct_to_frac_handle_t* handle);
c4a_status_t c4a_pp_pct_to_frac_transform( const c4a_pp_pct_to_frac_handle_t* handle, c4a_matrix_view_t X, c4a_matrix_view_t out);
```

:::

:::{tab-item} Python ABI · chemometrics4all.python
:sync: python-abi
:class-label: lang-python

```python
from chemometrics4all import python as c4a

Xt = c4a.pct_to_frac(X)
```

:::

:::{tab-item} Python sklearn · chemometrics4all.sklearn
:sync: python-sklearn
:class-label: lang-python

```python
from chemometrics4all.sklearn import PercentToFraction

op = PercentToFraction()
Xt = op.fit_transform(X)
```

:::

:::{tab-item} R · chemometrics4all
:sync: r
:class-label: lang-r

```r
library(chemometrics4all)
res <- percent_to_fraction(X)
```

:::

::::


**Benchmark Comparators And Sources** ◆

:::{card}
:class-card: external-refs

- ◆ **`ref.nirs4all`** (Python · canonical) — `nirs4all.PercentToFraction` · nirs4all@cd731a23+dirty
:::

### Validation contract

- Operation: `cross_binding_callable` · comparator: `default_allclose` · tolerance: `rtol=1e-05`, `atol=1e-08` · quality: **strict**
- Default validation dataset: `100×50` · seed `20260556`
- Suites: smoke `3` cells; benchmark `11` cells · Default C/Python/reference parity comparator.
- Metrics: `max_abs_diff`, `rel_l2_diff`, `rms_diff`, `shape_equal`
- Truth sources: cross-binding references declared directly in `benchmarks/cross_binding/orchestrator.py`.

| Backend | Library | Gate | Comparator | Note |
|---------|---------|------|------------|------|
| `ref.nirs4all` | `nirs4all.PercentToFraction` | Python / parity | `default_allclose` |  |

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
<tr class="bk-row"><td class="bk-name"><code>C4A.cpp</code></td><td class="parity parity-divergence parity-exact" title="worst reference max abs diff over visible sizes">0</td><td class="ms ms-best">🏆 0.002 ms</td><td class="ms ms-best">🏆 0.022 ms</td><td class="ms ms-best">🏆 0.112 ms</td></tr>
</tbody>
<tbody class="lang-band lang-python"><tr class="lang-band-row" data-lang="python"><th colspan="5" scope="rowgroup"><span class="lang-band-dot"></span>Python · chemometrics4all</th></tr>
<tr class="bk-row"><td class="bk-name"><code>C4A.python</code></td><td class="parity parity-divergence parity-exact" title="worst reference max abs diff over visible sizes">0</td><td class="ms">0.007 ms</td><td class="ms">0.027 ms</td><td class="ms">0.119 ms</td></tr>
<tr class="bk-row"><td class="bk-name"><code>C4A.sklearn</code></td><td class="parity parity-divergence parity-exact" title="worst reference max abs diff over visible sizes">0</td><td class="ms">0.009 ms</td><td class="ms">0.030 ms</td><td class="ms">0.125 ms</td></tr>
</tbody>
<tbody class="lang-band lang-r"><tr class="lang-band-row" data-lang="r"><th colspan="5" scope="rowgroup"><span class="lang-band-dot"></span>R · chemometrics4all</th></tr>
<tr class="bk-row"><td class="bk-name"><code>C4A.R</code></td><td class="parity parity-divergence parity-exact" title="worst reference max abs diff over visible sizes">5.2e-18</td><td class="ms">0.028 ms</td><td class="ms">0.272 ms</td><td class="ms">1.570 ms</td></tr>
</tbody>
<tbody class="lang-band lang-python"><tr class="lang-band-row" data-lang="python"><th colspan="5" scope="rowgroup"><span class="lang-band-dot"></span>Python · external</th></tr>
<tr class="bk-row truth-source-strict"><td class="bk-name"><span class="truth-mark" title="Registry parity reference (Python): nirs4all.PercentToFraction · nirs4all@cd731a23+dirty — canonical">◆</span><code>ref.nirs4all</code></td><td class="parity parity-divergence parity-exact" title="worst reference max abs diff over visible sizes">0</td><td class="ms">0.006 ms</td><td class="ms">0.036 ms</td><td class="ms">0.196 ms</td></tr>
</tbody>
</table>
</div>

:::
::::

---

_See also_: [benchmark overview](../benchmarks/overview.md) · [methods index](index.md) · [interactive dashboard](../landing/dashboard.md)
