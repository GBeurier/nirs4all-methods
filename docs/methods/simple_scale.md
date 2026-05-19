# `simple_scale` — Simple scale

_Group_: **Preprocessing** · _Registry tolerance_: `rtol=1e-5`, `atol=1e-8` · _Source_: [`docs/algorithms/simple_scale.md`](../algorithms/simple_scale.md)

## Description

From the `chemometrics4all.SimpleScale` Python wrapper docstring:

> Column-wise min-max scaling to ``[0, 1]``.

### Parameters

None — the transform is fully data-driven.

## Explanations

### Bibliographic source

Implementation matches `nirs4all.operators.transforms.scalers.SimpleScale`.

### Mathematical principle

Column-wise (axis=0) min-max scaling to the $[0, 1]$ range. For each column $j$:

$$
\mu_j = \min_i x_{i,j}, \qquad
M_j = \max_i x_{i,j}, \qquad
x'_{i,j} = \frac{x_{i,j} - \mu_j}{M_j - \mu_j}
$$

This is the special case of [Normalize](normalize.md) with `feature_range = (0, 1)` written as a dedicated operator for performance and API symmetry.

### Implementation

* Per-column min/max found via a single left-to-right scan.
* Final scaling uses `(x - min) / (max - min)` — a single subtraction and a single division per element, matching nirs4all.

C ABI entry points used by the language bindings:

```c
c4a_status_t c4a_pp_simple_scale_create( c4a_pp_simple_scale_handle_t** out);
void c4a_pp_simple_scale_destroy( c4a_pp_simple_scale_handle_t* handle);
c4a_status_t c4a_pp_simple_scale_transform( const c4a_pp_simple_scale_handle_t* handle, c4a_matrix_view_t X, c4a_matrix_view_t out);
```

Reference backends are registered in the benchmark matrix and stored as reproducible snapshots when they define the canonical contract.

### Implementations

| Layer | Entry point | Language | Contract |
|-------|-------------|----------|----------|
| C ABI | `c4a_pp_simple_scale` | C/C++ | Stable libc4a entry point family. |
| Python | `chemometrics4all.SimpleScale` | Python | sklearn-style wrapper backed by ctypes. |
| R | `simple_scale(X)` | R | Public package wrapper around the C ABI. |
| ref.sklearn | `sklearn.preprocessing.MinMaxScaler` | Python | canonical/comparator |
| ref.numpy | `numpy` | Python | canonical/comparator |
| ref.r.base | `R base` | R | canonical/comparator |

### Usage

Every chemometrics4all binding dispatches into the same C kernel. The registry references are listed in the parity card below.

::::{tab-set}
:class: chemometrics4all-bindings


:::{tab-item} C ABI · libc4a
:sync: c
:class-label: lang-c

```c
c4a_status_t c4a_pp_simple_scale_create( c4a_pp_simple_scale_handle_t** out);
void c4a_pp_simple_scale_destroy( c4a_pp_simple_scale_handle_t* handle);
c4a_status_t c4a_pp_simple_scale_transform( const c4a_pp_simple_scale_handle_t* handle, c4a_matrix_view_t X, c4a_matrix_view_t out);
```

:::

:::{tab-item} Python · chemometrics4all
:sync: python
:class-label: lang-python

```python
from chemometrics4all import SimpleScale

op = SimpleScale()
Xt = op.fit_transform(X)
```

:::

:::{tab-item} R · chemometrics4all
:sync: r
:class-label: lang-r

```r
library(chemometrics4all)
res <- simple_scale(X)
```

:::

::::


**Registry parity references** 📐

:::{card}
:class-card: external-refs

- 📐 **`ref.sklearn`** (Python · canonical) — `sklearn.preprocessing.MinMaxScaler` · sklearn 1.8.0
- 📐 **`ref.numpy`** (Python · comparator) — `numpy` · numpy 2.3.5
- 📐 **`ref.r.base`** (R · comparator) — `R base`
:::

### Benchmarks
Median wall-clock per cell from [`docs/_static/bench-data.json`](../benchmarks/overview.md). Verdict legend: ✓ exact · ≈ context/drift · ✗ divergent · ⊘ not available · — not run · ⚠ error.
::::{tab-set}
:class: parity-tabs

:::{tab-item} 1 thread
:sync: threads-1

<div class="parity-table-wrap">
<table class="docutils parity-grouped">
<thead><tr><th>Backend</th><th>Parity</th><th>100×50</th><th>100×500</th><th>100×2500</th></tr></thead>
<tbody class="lang-band lang-cpp"><tr class="lang-band-row" data-lang="cpp"><th colspan="5" scope="rowgroup"><span class="lang-band-dot"></span>C++ native · libc4a</th></tr>
<tr class="bk-row"><td class="bk-name"><code>C++</code></td><td class="parity parity-exact">✓ exact</td><td class="ms ms-best">🏆 0.006 ms</td><td class="ms ms-best">🏆 0.045 ms</td><td class="ms ms-best">🏆 0.223 ms</td></tr>
</tbody>
<tbody class="lang-band lang-python"><tr class="lang-band-row" data-lang="python"><th colspan="5" scope="rowgroup"><span class="lang-band-dot"></span>Python · chemometrics4all</th></tr>
<tr class="bk-row"><td class="bk-name"><code>Py</code></td><td class="parity parity-exact">✓ bind</td><td class="ms">0.011 ms</td><td class="ms">0.051 ms</td><td class="ms">0.243 ms</td></tr>
</tbody>
<tbody class="lang-band lang-r"><tr class="lang-band-row" data-lang="r"><th colspan="5" scope="rowgroup"><span class="lang-band-dot"></span>R · chemometrics4all</th></tr>
<tr class="bk-row"><td class="bk-name"><code>R</code></td><td class="parity parity-exact">✓ bind</td><td class="ms">0.026 ms</td><td class="ms">0.291 ms</td><td class="ms">1.648 ms</td></tr>
</tbody>
<tbody class="lang-band lang-python"><tr class="lang-band-row" data-lang="python"><th colspan="5" scope="rowgroup"><span class="lang-band-dot"></span>Python · external</th></tr>
<tr class="bk-row truth-source-relaxed"><td class="bk-name"><span class="truth-mark" title="Registry parity reference (Python): numpy · numpy 2.3.5 — comparator">📐</span><code>ref.numpy</code></td><td class="parity parity-exact">✓ ref</td><td class="ms">0.013 ms</td><td class="ms">0.072 ms</td><td class="ms">0.370 ms</td></tr>
<tr class="bk-row truth-source-strict"><td class="bk-name"><span class="truth-mark" title="Registry parity reference (Python): sklearn.preprocessing.MinMaxScaler · sklearn 1.8.0 — canonical">📐</span><code>ref.sklearn</code></td><td class="parity parity-exact">✓ ref</td><td class="ms">0.127 ms</td><td class="ms">0.213 ms</td><td class="ms">0.766 ms</td></tr>
</tbody>
<tbody class="lang-band lang-r"><tr class="lang-band-row" data-lang="r"><th colspan="5" scope="rowgroup"><span class="lang-band-dot"></span>R · external</th></tr>
<tr class="bk-row truth-source-relaxed"><td class="bk-name"><span class="truth-mark" title="Registry parity reference (R): R base · R base — comparator">📐</span><code>ref.r.base</code></td><td class="parity parity-exact">✓ ref</td><td class="ms">0.219 ms</td><td class="ms">1.656 ms</td><td class="ms">9.438 ms</td></tr>
</tbody>
</table>
</div>

:::
::::

---

_See also_: [benchmark overview](../benchmarks/overview.md) · [methods index](index.md) · [interactive dashboard](../landing/dashboard.md)
