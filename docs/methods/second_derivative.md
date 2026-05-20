# `second_derivative` — Second derivative

_Group_: **Preprocessing** · _Registry tolerance_: `rtol=1e-5`, `atol=1e-8` · _Source_: [`docs/algorithms/second_derivative.md`](../algorithms/second_derivative.md)

## Description

From the `chemometrics4all.SecondDerivative` Python wrapper docstring:

> Two passes of ``np.gradient`` (shape-preserving).

### Parameters

| Name | Type | Default | Notes |
|------|------|---------|-------|
| `delta` | `float` | `1.0` | Sample spacing along the wavelength axis. |
| `edge_order` | `int` | `2` |  |

## Explanations

### Bibliographic source

Documentation: [`numpy.gradient`](https://numpy.org/doc/stable/reference/generated/numpy.gradient.html).

### Mathematical principle

Numerical second derivative along the wavelength axis as the composition
of two `np.gradient` passes:

$$
\hat{X}^{(2)} = \nabla\big(\nabla(X)\big) \quad \text{along axis=1, with } \delta, \text{edge\_order}.
$$

This is the operator nirs4all uses by default for the SecondDerivative
preprocessing.  Shape-preserving; the second pass reuses the FirstDerivative
edge formula on the already-differentiated row.

### Implementation

- `_transform` requires `cols >= 3` for `edge_order = 2` (since two
  passes still need a 3-element interior).
- Tolerance: $10^{-12}$ absolute / $10^{-13}$ relative.

C ABI entry points used by the language bindings:

```c
c4a_status_t c4a_pp_second_derivative_create( c4a_pp_second_derivative_handle_t** out, double delta, int32_t edge_order);
void c4a_pp_second_derivative_destroy( c4a_pp_second_derivative_handle_t* handle);
c4a_status_t c4a_pp_second_derivative_transform( const c4a_pp_second_derivative_handle_t* handle, c4a_matrix_view_t X, c4a_matrix_view_t out);
```

Benchmark comparator backends are registered in the matrix and stored as reproducible snapshots when they define the canonical contract.

### Implementations

| Layer | Entry point | Language | Contract |
|-------|-------------|----------|----------|
| C ABI | `c4a_pp_second_derivative` | C/C++ | Stable libc4a entry point family. |
| Python | `chemometrics4all.python.second_derivative` | Python | ABI-close function backed by ctypes. |
| Python sklearn | `chemometrics4all.sklearn.SecondDerivative` | Python | scikit-learn-compatible estimator backed by ctypes. |
| R | `second_derivative(X, delta = 1.0, edge_order = 2L)` | R | Public package wrapper around the C ABI. |
| ref.nirs4all | `nirs4all.SecondDerivative` | Python | canonical/comparator |

### Usage

Every chemometrics4all binding dispatches into the same C kernel. Registered comparator/source rows are listed in the benchmark card below.

::::{tab-set}
:class: chemometrics4all-bindings


:::{tab-item} C ABI · libc4a
:sync: c
:class-label: lang-c

```c
c4a_status_t c4a_pp_second_derivative_create( c4a_pp_second_derivative_handle_t** out, double delta, int32_t edge_order);
void c4a_pp_second_derivative_destroy( c4a_pp_second_derivative_handle_t* handle);
c4a_status_t c4a_pp_second_derivative_transform( const c4a_pp_second_derivative_handle_t* handle, c4a_matrix_view_t X, c4a_matrix_view_t out);
```

:::

:::{tab-item} Python ABI · chemometrics4all.python
:sync: python-abi
:class-label: lang-python

```python
from chemometrics4all import python as c4a

Xt = c4a.second_derivative(X)
```

:::

:::{tab-item} Python sklearn · chemometrics4all.sklearn
:sync: python-sklearn
:class-label: lang-python

```python
from chemometrics4all.sklearn import SecondDerivative

op = SecondDerivative(delta=1.0, edge_order=2)
Xt = op.fit_transform(X)
```

:::

:::{tab-item} R · chemometrics4all
:sync: r
:class-label: lang-r

```r
library(chemometrics4all)
res <- second_derivative(X)
```

:::

::::


**Benchmark Comparators And Sources** ◆

:::{card}
:class-card: external-refs

- ◆ **`ref.nirs4all`** (Python · canonical) — `nirs4all.SecondDerivative` · nirs4all@cd731a23+dirty
:::

### Validation contract

- Operation: `cross_binding_callable` · comparator: `default_allclose` · tolerance: `rtol=1e-05`, `atol=1e-08` · quality: **strict**
- Default validation dataset: `100×50` · seed `20260556`
- Suites: smoke `3` cells; benchmark `11` cells · Default C/Python/reference parity comparator.
- Metrics: `max_abs_diff`, `rel_l2_diff`, `rms_diff`, `shape_equal`
- Truth sources: cross-binding references declared directly in `benchmarks/cross_binding/orchestrator.py`.

| Backend | Library | Gate | Comparator | Note |
|---------|---------|------|------------|------|
| `ref.nirs4all` | `nirs4all.SecondDerivative` | Python / parity | `default_allclose` |  |

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
<tr class="bk-row"><td class="bk-name"><code>C4A.cpp</code></td><td class="parity parity-divergence parity-exact" title="worst reference max abs diff over visible sizes">8.3e-17</td><td class="ms ms-best">🏆 0.005 ms</td><td class="ms ms-best">🏆 0.043 ms</td><td class="ms ms-best">🏆 0.272 ms</td></tr>
</tbody>
<tbody class="lang-band lang-python"><tr class="lang-band-row" data-lang="python"><th colspan="5" scope="rowgroup"><span class="lang-band-dot"></span>Python · chemometrics4all</th></tr>
<tr class="bk-row"><td class="bk-name"><code>C4A.python</code></td><td class="parity parity-divergence parity-exact" title="worst reference max abs diff over visible sizes">8.3e-17</td><td class="ms">0.010 ms</td><td class="ms">0.051 ms</td><td class="ms">0.314 ms</td></tr>
<tr class="bk-row"><td class="bk-name"><code>C4A.sklearn</code></td><td class="parity parity-divergence parity-exact" title="worst reference max abs diff over visible sizes">8.3e-17</td><td class="ms">0.012 ms</td><td class="ms">0.055 ms</td><td class="ms">0.327 ms</td></tr>
</tbody>
<tbody class="lang-band lang-r"><tr class="lang-band-row" data-lang="r"><th colspan="5" scope="rowgroup"><span class="lang-band-dot"></span>R · chemometrics4all</th></tr>
<tr class="bk-row"><td class="bk-name"><code>C4A.R</code></td><td class="parity parity-divergence parity-exact" title="worst reference max abs diff over visible sizes">1.3e-16</td><td class="ms">0.031 ms</td><td class="ms">0.307 ms</td><td class="ms">1.875 ms</td></tr>
</tbody>
<tbody class="lang-band lang-python"><tr class="lang-band-row" data-lang="python"><th colspan="5" scope="rowgroup"><span class="lang-band-dot"></span>Python · external</th></tr>
<tr class="bk-row truth-source-strict"><td class="bk-name"><span class="truth-mark" title="Registry parity reference (Python): nirs4all.SecondDerivative · nirs4all@cd731a23+dirty — canonical">◆</span><code>ref.nirs4all</code></td><td class="parity parity-divergence parity-exact" title="worst reference max abs diff over visible sizes">0</td><td class="ms">0.038 ms</td><td class="ms">0.223 ms</td><td class="ms">1.075 ms</td></tr>
</tbody>
</table>
</div>

:::
::::

---

_See also_: [benchmark overview](../benchmarks/overview.md) · [methods index](index.md) · [interactive dashboard](../landing/dashboard.md)
