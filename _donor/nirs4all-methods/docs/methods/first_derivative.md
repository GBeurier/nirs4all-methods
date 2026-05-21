# `first_derivative` — First derivative

_Group_: **Preprocessing** · _Registry tolerance_: `rtol=1e-5`, `atol=1e-8` · _Source_: [`docs/algorithms/first_derivative.md`](../algorithms/first_derivative.md)

## Description

From the `n4m.FirstDerivative` Python wrapper docstring:

> ``np.gradient(X, delta, axis=1, edge_order=...)`` (shape-preserving).

### Parameters

| Name | Type | Default | Notes |
|------|------|---------|-------|
| `delta` | `float` | `1.0` | Sample spacing along the wavelength axis. |
| `edge_order` | `int` | `2` |  |

## Explanations

### Bibliographic source

Documentation: [`numpy.gradient`](https://numpy.org/doc/stable/reference/generated/numpy.gradient.html).

### Mathematical principle

Numerical first derivative along the wavelength axis with second-order
accuracy at the edges, mirroring `np.gradient(X, delta, axis=1,
edge_order)`:

$$
\hat{X}_{i,j} = \begin{cases}
\dfrac{X_{i, j+1} - X_{i, j-1}}{2 \delta} & 1 \leq j \leq p - 2 \\[4pt]
\dfrac{-3 X_{i, 0} + 4 X_{i, 1} - X_{i, 2}}{2 \delta} & j = 0,\;\text{edge\_order} = 2 \\[4pt]
\dfrac{X_{i, 1} - X_{i, 0}}{\delta} & j = 0,\;\text{edge\_order} = 1 \\
\end{cases}
$$

(and symmetrically at the right edge).  Output shape matches input shape
(shape-preserving).

### Implementation

- `_create` returns `N4M_ERR_INVALID_ARGUMENT` for $\delta = 0$ or an
  edge order outside $\{1, 2\}$.
- `_transform` requires `cols >= 3` for `edge_order = 2` and `cols >= 2`
  for `edge_order = 1`.
- Tolerance against `np.gradient`: $10^{-12}$ absolute / $10^{-13}$
  relative — pure arithmetic over the input samples.

C ABI entry points used by the language bindings:

```c
n4m_status_t n4m_pp_first_derivative_create( n4m_pp_first_derivative_handle_t** out, double delta, int32_t edge_order);
void n4m_pp_first_derivative_destroy( n4m_pp_first_derivative_handle_t* handle);
n4m_status_t n4m_pp_first_derivative_transform( const n4m_pp_first_derivative_handle_t* handle, n4m_matrix_view_t X, n4m_matrix_view_t out);
```

Benchmark comparator backends are registered in the matrix and stored as reproducible snapshots when they define the canonical contract.

### Implementations

| Layer | Entry point | Language | Contract |
|-------|-------------|----------|----------|
| C ABI | `n4m_pp_first_derivative` | C/C++ | Stable libn4m entry point family. |
| Python | `n4m.python.first_derivative` | Python | ABI-close function backed by ctypes. |
| Python sklearn | `n4m.sklearn.FirstDerivative` | Python | scikit-learn-compatible estimator backed by ctypes. |
| R | `first_derivative(X, delta = 1.0, edge_order = 2L)` | R | Public package wrapper around the C ABI. |
| ref.nirs4all | `nirs4all.FirstDerivative` | Python | canonical/comparator |

### Usage

Every nirs4all-methods binding dispatches into the same C kernel. Registered comparator/source rows are listed in the benchmark card below.

::::{tab-set}
:class: nirs4all-methods-bindings


:::{tab-item} C ABI · libn4m
:sync: c
:class-label: lang-c

```c
n4m_status_t n4m_pp_first_derivative_create( n4m_pp_first_derivative_handle_t** out, double delta, int32_t edge_order);
void n4m_pp_first_derivative_destroy( n4m_pp_first_derivative_handle_t* handle);
n4m_status_t n4m_pp_first_derivative_transform( const n4m_pp_first_derivative_handle_t* handle, n4m_matrix_view_t X, n4m_matrix_view_t out);
```

:::

:::{tab-item} Python ABI · n4m.python
:sync: python-abi
:class-label: lang-python

```python
from n4m import python as n4m

Xt = n4m.first_derivative(X)
```

:::

:::{tab-item} Python sklearn · n4m.sklearn
:sync: python-sklearn
:class-label: lang-python

```python
from n4m.sklearn import FirstDerivative

op = FirstDerivative(delta=1.0, edge_order=2)
Xt = op.fit_transform(X)
```

:::

:::{tab-item} R · nirs4all-methods
:sync: r
:class-label: lang-r

```r
library(n4m)
res <- first_derivative(X)
```

:::

::::


**Benchmark Comparators And Sources** ◆

:::{card}
:class-card: external-refs

- ◆ **`ref.nirs4all`** (Python · canonical) — `nirs4all.FirstDerivative` · nirs4all@cd731a23+dirty
:::

### Validation contract

- Operation: `cross_binding_callable` · comparator: `default_allclose` · tolerance: `rtol=1e-05`, `atol=1e-08` · quality: **strict**
- Default validation dataset: `100×50` · seed `20260556`
- Suites: smoke `3` cells; benchmark `11` cells · Default C/Python/reference parity comparator.
- Metrics: `max_abs_diff`, `rel_l2_diff`, `rms_diff`, `shape_equal`
- Truth sources: cross-binding references declared directly in `benchmarks/cross_binding/orchestrator.py`.

| Backend | Library | Gate | Comparator | Note |
|---------|---------|------|------------|------|
| `ref.nirs4all` | `nirs4all.FirstDerivative` | Python / parity | `default_allclose` |  |

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
<tr class="bk-row"><td class="bk-name"><code>N4M.cpp</code></td><td class="parity parity-divergence parity-exact" title="worst reference max abs diff over visible sizes">5.6e-17</td><td class="ms ms-best">🏆 0.003 ms</td><td class="ms ms-best">🏆 0.022 ms</td><td class="ms ms-best">🏆 0.141 ms</td></tr>
</tbody>
<tbody class="lang-band lang-python"><tr class="lang-band-row" data-lang="python"><th colspan="5" scope="rowgroup"><span class="lang-band-dot"></span>Python · nirs4all-methods</th></tr>
<tr class="bk-row"><td class="bk-name"><code>N4M.python</code></td><td class="parity parity-divergence parity-exact" title="worst reference max abs diff over visible sizes">5.6e-17</td><td class="ms">0.009 ms</td><td class="ms">0.029 ms</td><td class="ms">0.147 ms</td></tr>
<tr class="bk-row"><td class="bk-name"><code>N4M.sklearn</code></td><td class="parity parity-divergence parity-exact" title="worst reference max abs diff over visible sizes">5.6e-17</td><td class="ms">0.010 ms</td><td class="ms">0.035 ms</td><td class="ms">0.148 ms</td></tr>
</tbody>
<tbody class="lang-band lang-r"><tr class="lang-band-row" data-lang="r"><th colspan="5" scope="rowgroup"><span class="lang-band-dot"></span>R · nirs4all-methods</th></tr>
<tr class="bk-row"><td class="bk-name"><code>N4M.R</code></td><td class="parity parity-divergence parity-exact" title="worst reference max abs diff over visible sizes">1.0e-16</td><td class="ms">0.031 ms</td><td class="ms">0.272 ms</td><td class="ms">1.758 ms</td></tr>
</tbody>
<tbody class="lang-band lang-python"><tr class="lang-band-row" data-lang="python"><th colspan="5" scope="rowgroup"><span class="lang-band-dot"></span>Python · external</th></tr>
<tr class="bk-row truth-source-strict"><td class="bk-name"><span class="truth-mark" title="Registry parity reference (Python): nirs4all.FirstDerivative · nirs4all@cd731a23+dirty — canonical">◆</span><code>ref.nirs4all</code></td><td class="parity parity-divergence parity-exact" title="worst reference max abs diff over visible sizes">0</td><td class="ms">0.022 ms</td><td class="ms">0.105 ms</td><td class="ms">0.551 ms</td></tr>
</tbody>
</table>
</div>

:::
::::

---

_See also_: [benchmark overview](../benchmarks/overview.md) · [methods index](index.md) · [interactive dashboard](../landing/dashboard.md)
