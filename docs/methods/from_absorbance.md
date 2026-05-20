# `from_absorbance` — From absorbance

_Group_: **Preprocessing** · _Registry tolerance_: `rtol=1e-5`, `atol=1e-8` · _Source_: [`docs/algorithms/from_absorbance.md`](../algorithms/from_absorbance.md)

## Description

From the `n4m.FromAbsorbance` Python wrapper docstring:

> R = 10**(-A), optionally returned as percent.

### Parameters

| Name | Type | Default | Notes |
|------|------|---------|-------|
| `is_percent` | `bool` | `False` |  |

## Explanations

### Bibliographic source

Implementation matches `nirs4all.operators.transforms.signal_conversion.FromAbsorbance` (0.8.11).

### Mathematical principle

Element-wise inverse log-10: convert absorbance back to reflectance or transmittance.

$$
y_{i,j} = 10^{-x_{i,j}}, \qquad
\text{out}_{i,j} = \begin{cases} 100 \cdot y_{i,j} & \text{if `is\_percent`} \\ y_{i,j} & \text{otherwise} \end{cases}
$$

### Implementation

* The power is computed as `pow(10.0, -x)` — element-wise `np.power(10.0, -X)` semantics. The alternative `exp(-x · ln10)` introduces one extra multiply rounding step that would diverge in the last bit.
* The percent multiply uses `* 100.0` as a separate inline operation (not folded into the exponent), mirroring `result = result * 100.0` in nirs4all.
* Parity tolerance vs nirs4all reference: `1e-12 abs / 1e-13 rel`.

C ABI entry points used by the language bindings:

```c
n4m_status_t n4m_pp_from_absorbance_create( n4m_pp_from_absorbance_handle_t** out, int is_percent);
void n4m_pp_from_absorbance_destroy( n4m_pp_from_absorbance_handle_t* handle);
n4m_status_t n4m_pp_from_absorbance_transform( const n4m_pp_from_absorbance_handle_t* handle, n4m_matrix_view_t X, n4m_matrix_view_t out);
```

Benchmark comparator backends are registered in the matrix and stored as reproducible snapshots when they define the canonical contract.

### Implementations

| Layer | Entry point | Language | Contract |
|-------|-------------|----------|----------|
| C ABI | `n4m_pp_from_absorbance` | C/C++ | Stable libn4m entry point family. |
| Python | `n4m.python.from_absorbance` | Python | ABI-close function backed by ctypes. |
| Python sklearn | `n4m.sklearn.FromAbsorbance` | Python | scikit-learn-compatible estimator backed by ctypes. |
| R | `from_absorbance(X, is_percent = FALSE)` | R | Public package wrapper around the C ABI. |
| ref.nirs4all | `nirs4all.FromAbsorbance` | Python | canonical/comparator |

### Usage

Every nirs4all-methods binding dispatches into the same C kernel. Registered comparator/source rows are listed in the benchmark card below.

::::{tab-set}
:class: nirs4all-methods-bindings


:::{tab-item} C ABI · libn4m
:sync: c
:class-label: lang-c

```c
n4m_status_t n4m_pp_from_absorbance_create( n4m_pp_from_absorbance_handle_t** out, int is_percent);
void n4m_pp_from_absorbance_destroy( n4m_pp_from_absorbance_handle_t* handle);
n4m_status_t n4m_pp_from_absorbance_transform( const n4m_pp_from_absorbance_handle_t* handle, n4m_matrix_view_t X, n4m_matrix_view_t out);
```

:::

:::{tab-item} Python ABI · n4m.python
:sync: python-abi
:class-label: lang-python

```python
from n4m import python as n4m

Xt = n4m.from_absorbance(X)
```

:::

:::{tab-item} Python sklearn · n4m.sklearn
:sync: python-sklearn
:class-label: lang-python

```python
from n4m.sklearn import FromAbsorbance

op = FromAbsorbance(is_percent=False)
Xt = op.fit_transform(X)
```

:::

:::{tab-item} R · nirs4all-methods
:sync: r
:class-label: lang-r

```r
library(n4m)
res <- from_absorbance(X)
```

:::

::::


**Benchmark Comparators And Sources** ◆

:::{card}
:class-card: external-refs

- ◆ **`ref.nirs4all`** (Python · canonical) — `nirs4all.FromAbsorbance` · nirs4all@cd731a23+dirty
:::

### Validation contract

- Operation: `cross_binding_callable` · comparator: `default_allclose` · tolerance: `rtol=1e-05`, `atol=1e-08` · quality: **strict**
- Default validation dataset: `100×50` · seed `20260556`
- Suites: smoke `3` cells; benchmark `11` cells · Default C/Python/reference parity comparator.
- Metrics: `max_abs_diff`, `rel_l2_diff`, `rms_diff`, `shape_equal`
- Truth sources: cross-binding references declared directly in `benchmarks/cross_binding/orchestrator.py`.

| Backend | Library | Gate | Comparator | Note |
|---------|---------|------|------------|------|
| `ref.nirs4all` | `nirs4all.FromAbsorbance` | Python / parity | `default_allclose` |  |

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
<tr class="bk-row"><td class="bk-name"><code>N4M.cpp</code></td><td class="parity parity-divergence parity-exact" title="worst reference max abs diff over visible sizes">1.1e-16</td><td class="ms ms-best">🏆 0.017 ms</td><td class="ms ms-best">🏆 0.164 ms</td><td class="ms">0.828 ms</td></tr>
</tbody>
<tbody class="lang-band lang-python"><tr class="lang-band-row" data-lang="python"><th colspan="5" scope="rowgroup"><span class="lang-band-dot"></span>Python · nirs4all-methods</th></tr>
<tr class="bk-row"><td class="bk-name"><code>N4M.python</code></td><td class="parity parity-divergence parity-exact" title="worst reference max abs diff over visible sizes">1.1e-16</td><td class="ms">0.022 ms</td><td class="ms">0.173 ms</td><td class="ms">0.821 ms</td></tr>
<tr class="bk-row"><td class="bk-name"><code>N4M.sklearn</code></td><td class="parity parity-divergence parity-exact" title="worst reference max abs diff over visible sizes">1.1e-16</td><td class="ms">0.025 ms</td><td class="ms">0.173 ms</td><td class="ms ms-best">🏆 0.806 ms</td></tr>
</tbody>
<tbody class="lang-band lang-r"><tr class="lang-band-row" data-lang="r"><th colspan="5" scope="rowgroup"><span class="lang-band-dot"></span>R · nirs4all-methods</th></tr>
<tr class="bk-row"><td class="bk-name"><code>N4M.R</code></td><td class="parity parity-divergence parity-exact" title="worst reference max abs diff over visible sizes">6.1e-16</td><td class="ms">0.039 ms</td><td class="ms">0.473 ms</td><td class="ms">2.453 ms</td></tr>
</tbody>
<tbody class="lang-band lang-python"><tr class="lang-band-row" data-lang="python"><th colspan="5" scope="rowgroup"><span class="lang-band-dot"></span>Python · external</th></tr>
<tr class="bk-row truth-source-strict"><td class="bk-name"><span class="truth-mark" title="Registry parity reference (Python): nirs4all.FromAbsorbance · nirs4all@cd731a23+dirty — canonical">◆</span><code>ref.nirs4all</code></td><td class="parity parity-divergence parity-exact" title="worst reference max abs diff over visible sizes">0</td><td class="ms">0.048 ms</td><td class="ms">0.440 ms</td><td class="ms">2.401 ms</td></tr>
</tbody>
</table>
</div>

:::
::::

---

_See also_: [benchmark overview](../benchmarks/overview.md) · [methods index](index.md) · [interactive dashboard](../landing/dashboard.md)
