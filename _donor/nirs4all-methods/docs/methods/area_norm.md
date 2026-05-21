# `area_norm` — Area normalization

_Group_: **Preprocessing** · _Registry tolerance_: `rtol=1e-5`, `atol=1e-8` · _Source_: [`docs/algorithms/area_normalization.md`](../algorithms/area_normalization.md)

## Description

From the `n4m.AreaNormalization` Python wrapper docstring:

> Per-row area normalisation.

### Parameters

| Name | Type | Default | Notes |
|------|------|---------|-------|
| `method` | `str` | `'sum'` | Algorithm variant selected by the wrapper. |

## Explanations

### Bibliographic source

Implementation matches `nirs4all.operators.transforms.nirs.AreaNormalization`. The `trapz` integrand follows the implementation used by `numpy.trapz` (which is bit-identical to `scipy.integrate.trapezoid` for unit spacing).

### Mathematical principle

Per-spectrum normalisation by a row-area metric. For each row $x_i \in \mathbb{R}^m$:

$$
A_i = \text{area}(x_i), \qquad
x'_{i,j} = \frac{x_{i,j}}{A_i}
$$

where the area metric is one of:

| Method | Formula | Notes |
| --- | --- | --- |
| `sum` | $A_i = \sum_j x_{i,j}$ | Signed total intensity. |
| `abs_sum` | $A_i = \sum_j |x_{i,j}|$ | Absolute intensity (robust to mean-centered data). |
| `trapz` | $A_i = \frac{1}{2} \sum_{j=0}^{m-2} (x_{i,j} + x_{i,j+1})$ | Trapezoidal integration with unit spacing. |

If $|A_i| < 10^{-10}$, the divisor is replaced by $1.0$ to avoid amplifying noise on near-zero rows.

### Implementation

* `sum` / `abs_sum` accumulate left-to-right with the same rounding order as `numpy.add.reduce`.
* `trapz` matches `numpy.trapz` / `scipy.integrate.trapezoid` with default $\Delta x = 1$, computed as $\frac{1}{2} \sum (x[:-1] + x[1:])$ — the pair-sum form is preserved bit-for-bit.
* Final scaling uses `x / area`, not `x * (1/area)`.

C ABI entry points used by the language bindings:

```c
n4m_status_t n4m_pp_area_create(n4m_pp_area_handle_t** out, int32_t method);
void n4m_pp_area_destroy(n4m_pp_area_handle_t* handle);
n4m_status_t n4m_pp_area_transform(const n4m_pp_area_handle_t* handle, n4m_matrix_view_t X, n4m_matrix_view_t out);
```

Benchmark comparator backends are registered in the matrix and stored as reproducible snapshots when they define the canonical contract.

### Implementations

| Layer | Entry point | Language | Contract |
|-------|-------------|----------|----------|
| C ABI | `n4m_pp_area` | C/C++ | Stable libn4m entry point family. |
| Python | `n4m.python.area_norm` | Python | ABI-close function backed by ctypes. |
| Python sklearn | `n4m.sklearn.AreaNormalization` | Python | scikit-learn-compatible estimator backed by ctypes. |
| R | `area_normalization(X, method = "sum")` | R | Public package wrapper around the C ABI. |
| ref.nirs4all | `nirs4all.AreaNormalization` | Python | canonical/comparator |

### Usage

Every nirs4all-methods binding dispatches into the same C kernel. Registered comparator/source rows are listed in the benchmark card below.

::::{tab-set}
:class: nirs4all-methods-bindings


:::{tab-item} C ABI · libn4m
:sync: c
:class-label: lang-c

```c
n4m_status_t n4m_pp_area_create(n4m_pp_area_handle_t** out, int32_t method);
void n4m_pp_area_destroy(n4m_pp_area_handle_t* handle);
n4m_status_t n4m_pp_area_transform(const n4m_pp_area_handle_t* handle, n4m_matrix_view_t X, n4m_matrix_view_t out);
```

:::

:::{tab-item} Python ABI · n4m.python
:sync: python-abi
:class-label: lang-python

```python
from n4m import python as n4m

Xt = n4m.area_norm(X)
```

:::

:::{tab-item} Python sklearn · n4m.sklearn
:sync: python-sklearn
:class-label: lang-python

```python
from n4m.sklearn import AreaNormalization

op = AreaNormalization(method='sum')
Xt = op.fit_transform(X)
```

:::

:::{tab-item} R · nirs4all-methods
:sync: r
:class-label: lang-r

```r
library(n4m)
res <- area_normalization(X, method = 'sum')
```

:::

::::


**Benchmark Comparators And Sources** ◆

:::{card}
:class-card: external-refs

- ◆ **`ref.nirs4all`** (Python · canonical) — `nirs4all.AreaNormalization` · nirs4all@cd731a23+dirty
:::

### Validation contract

- Operation: `cross_binding_callable` · comparator: `default_allclose` · tolerance: `rtol=1e-05`, `atol=1e-08` · quality: **strict**
- Default validation dataset: `100×50` · seed `20260556`
- Suites: smoke `3` cells; benchmark `11` cells · Default C/Python/reference parity comparator.
- Metrics: `max_abs_diff`, `rel_l2_diff`, `rms_diff`, `shape_equal`
- Truth sources: cross-binding references declared directly in `benchmarks/cross_binding/orchestrator.py`.

| Backend | Library | Gate | Comparator | Note |
|---------|---------|------|------------|------|
| `ref.nirs4all` | `nirs4all.AreaNormalization` | Python / parity | `default_allclose` |  |

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
<tr class="bk-row"><td class="bk-name"><code>N4M.cpp</code></td><td class="parity parity-divergence parity-exact" title="worst reference max abs diff over visible sizes">1.0e-17</td><td class="ms ms-best">🏆 0.001 ms</td><td class="ms ms-best">🏆 0.012 ms</td><td class="ms ms-best">🏆 0.077 ms</td></tr>
</tbody>
<tbody class="lang-band lang-python"><tr class="lang-band-row" data-lang="python"><th colspan="5" scope="rowgroup"><span class="lang-band-dot"></span>Python · nirs4all-methods</th></tr>
<tr class="bk-row"><td class="bk-name"><code>N4M.python</code></td><td class="parity parity-divergence parity-exact" title="worst reference max abs diff over visible sizes">1.0e-17</td><td class="ms">0.006 ms</td><td class="ms">0.018 ms</td><td class="ms">0.094 ms</td></tr>
<tr class="bk-row"><td class="bk-name"><code>N4M.sklearn</code></td><td class="parity parity-divergence parity-exact" title="worst reference max abs diff over visible sizes">1.0e-17</td><td class="ms">0.008 ms</td><td class="ms">0.021 ms</td><td class="ms">0.090 ms</td></tr>
</tbody>
<tbody class="lang-band lang-r"><tr class="lang-band-row" data-lang="r"><th colspan="5" scope="rowgroup"><span class="lang-band-dot"></span>R · nirs4all-methods</th></tr>
<tr class="bk-row"><td class="bk-name"><code>N4M.R</code></td><td class="parity parity-divergence parity-exact" title="worst reference max abs diff over visible sizes">5.6e-17</td><td class="ms">0.027 ms</td><td class="ms">0.252 ms</td><td class="ms">1.945 ms</td></tr>
</tbody>
<tbody class="lang-band lang-python"><tr class="lang-band-row" data-lang="python"><th colspan="5" scope="rowgroup"><span class="lang-band-dot"></span>Python · external</th></tr>
<tr class="bk-row truth-source-strict"><td class="bk-name"><span class="truth-mark" title="Registry parity reference (Python): nirs4all.AreaNormalization · nirs4all@cd731a23+dirty — canonical">◆</span><code>ref.nirs4all</code></td><td class="parity parity-divergence parity-exact" title="worst reference max abs diff over visible sizes">0</td><td class="ms">0.230 ms</td><td class="ms">0.285 ms</td><td class="ms">0.701 ms</td></tr>
</tbody>
</table>
</div>

:::
::::

---

_See also_: [benchmark overview](../benchmarks/overview.md) · [methods index](index.md) · [interactive dashboard](../landing/dashboard.md)
