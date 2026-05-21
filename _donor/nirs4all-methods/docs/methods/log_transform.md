# `log_transform` — Log transform

_Group_: **Preprocessing** · _Registry tolerance_: `rtol=1e-5`, `atol=1e-8` · _Source_: [`docs/algorithms/log_transform.md`](../algorithms/log_transform.md)

## Description

From the `n4m.LogTransform` Python wrapper docstring:

> Element-wise logarithm with optional fit-time auto-offset.

### Parameters

| Name | Type | Default | Notes |
|------|------|---------|-------|
| `base` | `float` | `0.0` |  |
| `offset` | `float` | `0.0` |  |
| `auto_offset` | `bool` | `True` |  |
| `min_value` | `float` | `1e-08` |  |

## Explanations

### Bibliographic source

Implementation matches `nirs4all.operators.transforms.nirs.LogTransform`.

### Mathematical principle

Element-wise logarithm with optional fit-time safety offset:

1. Compute the **fitted offset**:

   $$
   \tilde{x}^{(\text{temp})}_{i,j} = x_{i,j} + \text{offset}, \qquad
   m^{(\text{temp})} = \min_{i, j} \tilde{x}^{(\text{temp})}_{i,j}
   $$

   $$
   \delta_{\text{auto}} =
   \begin{cases}
     \text{min\_value} - m^{(\text{temp})} & \text{if auto\_offset } \wedge \, m^{(\text{temp})} \le 0 \\
     0 & \text{otherwise}
   \end{cases}, \qquad
   \delta = \text{offset} + \delta_{\text{auto}}
   $$

2. Apply the fitted offset to the **raw** $x$ (not $\tilde{x}^{(\text{temp})}$):

   $$
   y_{i,j} = x_{i,j} + \delta, \qquad
   y_{i,j} \leftarrow \begin{cases}
     \text{min\_value} & \text{if } y_{i,j} \le 0 \\
     y_{i,j} & \text{otherwise}
   \end{cases}
   $$

3. Take the logarithm:

   $$
   x'_{i,j} =
   \begin{cases}
     \log y_{i,j} & \text{if natural log} \\
     \dfrac{\log y_{i,j}}{\log B} & \text{if base } B \neq e
   \end{cases}
   $$

Note: the fitted offset is computed against $x + \text{offset}$ but applied to the raw $x$. The two differ in floating-point rounding only when `offset != 0`, but the difference must be reproduced exactly to match nirs4all.

### Implementation

* Natural-log fast path skips the `log(base)` division.
* For arbitrary bases, the change-of-base formula is `log(y) / log(base)` — a true element-wise division, matching `np.log(X) / np.log(base)`.
* `min(X + offset)` is computed by accumulating the per-element offset addition before the min, preserving nirs4all's rounding tree.

C ABI entry points used by the language bindings:

```c
n4m_status_t n4m_pp_log_create(n4m_pp_log_handle_t** out, double base, double offset, int auto_offset, double min_value);
void n4m_pp_log_destroy(n4m_pp_log_handle_t* handle);
n4m_status_t n4m_pp_log_fit(n4m_pp_log_handle_t* handle, n4m_matrix_view_t X);
n4m_status_t n4m_pp_log_is_fitted(const n4m_pp_log_handle_t* handle, int* out_fitted);
n4m_status_t n4m_pp_log_transform(const n4m_pp_log_handle_t* handle, n4m_matrix_view_t X, n4m_matrix_view_t out);
```

Benchmark comparator backends are registered in the matrix and stored as reproducible snapshots when they define the canonical contract.

### Implementations

| Layer | Entry point | Language | Contract |
|-------|-------------|----------|----------|
| C ABI | `n4m_pp_log` | C/C++ | Stable libn4m entry point family. |
| Python | `n4m.python.log_transform` | Python | ABI-close function backed by ctypes. |
| Python sklearn | `n4m.sklearn.LogTransform` | Python | scikit-learn-compatible estimator backed by ctypes. |
| ref.numpy | `numpy.log(auto-offset)` | Python | canonical/comparator |
| ref.nirs4all | `nirs4all.LogTransform` | Python | canonical/comparator |

### Usage

Every nirs4all-methods binding dispatches into the same C kernel. Registered comparator/source rows are listed in the benchmark card below.

::::{tab-set}
:class: nirs4all-methods-bindings


:::{tab-item} C ABI · libn4m
:sync: c
:class-label: lang-c

```c
n4m_status_t n4m_pp_log_create(n4m_pp_log_handle_t** out, double base, double offset, int auto_offset, double min_value);
void n4m_pp_log_destroy(n4m_pp_log_handle_t* handle);
n4m_status_t n4m_pp_log_fit(n4m_pp_log_handle_t* handle, n4m_matrix_view_t X);
n4m_status_t n4m_pp_log_is_fitted(const n4m_pp_log_handle_t* handle, int* out_fitted);
n4m_status_t n4m_pp_log_transform(const n4m_pp_log_handle_t* handle, n4m_matrix_view_t X, n4m_matrix_view_t out);
```

:::

:::{tab-item} Python ABI · n4m.python
:sync: python-abi
:class-label: lang-python

```python
from n4m import python as n4m

Xt = n4m.log_transform(X)
```

:::

:::{tab-item} Python sklearn · n4m.sklearn
:sync: python-sklearn
:class-label: lang-python

```python
from n4m.sklearn import LogTransform

op = LogTransform(base=0.0, offset=0.0, auto_offset=True, min_value=1e-08)
Xt = op.fit_transform(X)
```

:::

::::


**Benchmark Comparators And Sources** ◆

:::{card}
:class-card: external-refs

- ◆ **`ref.numpy`** (Python · canonical) — `numpy.log(auto-offset)` · numpy 2.3.5
- ◆ **`ref.nirs4all`** (Python · comparator) — `nirs4all.LogTransform` · nirs4all@cd731a23+dirty
:::

### Validation contract

- Operation: `cross_binding_callable` · comparator: `default_allclose` · tolerance: `rtol=1e-05`, `atol=1e-08` · quality: **strict**
- Default validation dataset: `100×50` · seed `20260556`
- Suites: smoke `3` cells; benchmark `11` cells · Default C/Python/reference parity comparator.
- Metrics: `max_abs_diff`, `rel_l2_diff`, `rms_diff`, `shape_equal`
- Truth sources: cross-binding references declared directly in `benchmarks/cross_binding/orchestrator.py`.

| Backend | Library | Gate | Comparator | Note |
|---------|---------|------|------------|------|
| `ref.numpy` | `numpy.log(auto-offset)` | Python / parity | `default_allclose` |  |
| `ref.nirs4all` | `nirs4all.LogTransform` | Python / parity | `default_allclose` |  |

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
<tr class="bk-row"><td class="bk-name"><code>N4M.cpp</code></td><td class="parity parity-divergence parity-exact" title="worst reference max abs diff over visible sizes">0</td><td class="ms">0.025 ms</td><td class="ms">0.189 ms</td><td class="ms">0.900 ms</td></tr>
</tbody>
<tbody class="lang-band lang-python"><tr class="lang-band-row" data-lang="python"><th colspan="5" scope="rowgroup"><span class="lang-band-dot"></span>Python · nirs4all-methods</th></tr>
<tr class="bk-row"><td class="bk-name"><code>N4M.python</code></td><td class="parity parity-divergence parity-exact" title="worst reference max abs diff over visible sizes">0</td><td class="ms">0.026 ms</td><td class="ms">0.192 ms</td><td class="ms">0.944 ms</td></tr>
<tr class="bk-row"><td class="bk-name"><code>N4M.sklearn</code></td><td class="parity parity-divergence parity-exact" title="worst reference max abs diff over visible sizes">0</td><td class="ms">0.027 ms</td><td class="ms">0.196 ms</td><td class="ms">0.967 ms</td></tr>
</tbody>
<tbody class="lang-band lang-python"><tr class="lang-band-row" data-lang="python"><th colspan="5" scope="rowgroup"><span class="lang-band-dot"></span>Python · external</th></tr>
<tr class="bk-row truth-source-relaxed"><td class="bk-name"><span class="truth-mark" title="Registry parity reference (Python): nirs4all.LogTransform · nirs4all@cd731a23+dirty — comparator">◆</span><code>ref.nirs4all</code></td><td class="parity parity-divergence parity-exact" title="worst reference max abs diff over visible sizes">0</td><td class="ms">0.025 ms</td><td class="ms">0.211 ms</td><td class="ms">1.065 ms</td></tr>
<tr class="bk-row truth-source-strict"><td class="bk-name"><span class="truth-mark" title="Registry parity reference (Python): numpy.log(auto-offset) · numpy 2.3.5 — canonical">◆</span><code>ref.numpy</code></td><td class="parity parity-divergence parity-exact" title="worst reference max abs diff over visible sizes">0</td><td class="ms ms-best">🏆 0.018 ms</td><td class="ms ms-best">🏆 0.150 ms</td><td class="ms ms-best">🏆 0.809 ms</td></tr>
</tbody>
</table>
</div>

:::
::::

---

_See also_: [benchmark overview](../benchmarks/overview.md) · [methods index](index.md) · [interactive dashboard](../landing/dashboard.md)
