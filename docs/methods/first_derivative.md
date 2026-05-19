# `first_derivative` — First derivative

_Group_: **Preprocessing** · _Registry tolerance_: `rtol=1e-5`, `atol=1e-8` · _Source_: [`docs/algorithms/first_derivative.md`](../algorithms/first_derivative.md)

## Description

From the `chemometrics4all.FirstDerivative` Python wrapper docstring:

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

- `_create` returns `C4A_ERR_INVALID_ARGUMENT` for $\delta = 0$ or an
  edge order outside $\{1, 2\}$.
- `_transform` requires `cols >= 3` for `edge_order = 2` and `cols >= 2`
  for `edge_order = 1`.
- Tolerance against `np.gradient`: $10^{-12}$ absolute / $10^{-13}$
  relative — pure arithmetic over the input samples.

C ABI entry points used by the language bindings:

```c
c4a_status_t c4a_pp_first_derivative_create( c4a_pp_first_derivative_handle_t** out, double delta, int32_t edge_order);
void c4a_pp_first_derivative_destroy( c4a_pp_first_derivative_handle_t* handle);
c4a_status_t c4a_pp_first_derivative_transform( const c4a_pp_first_derivative_handle_t* handle, c4a_matrix_view_t X, c4a_matrix_view_t out);
```

Reference backends are registered in the benchmark matrix and stored as reproducible snapshots when they define the canonical contract.

### Implementations

| Layer | Entry point | Language | Contract |
|-------|-------------|----------|----------|
| C ABI | `c4a_pp_first_derivative` | C/C++ | Stable libc4a entry point family. |
| Python | `chemometrics4all.FirstDerivative` | Python | sklearn-style wrapper backed by ctypes. |
| R | `first_derivative(X, delta = 1.0, edge_order = 2L)` | R | Public package wrapper around the C ABI. |
| ref.nirs4all | `nirs4all.FirstDerivative` | Python | canonical/comparator |
| ref.numpy | `numpy.gradient` | Python | canonical/comparator |

### Usage

Every chemometrics4all binding dispatches into the same C kernel. The registry references are listed in the parity card below.

::::{tab-set}
:class: chemometrics4all-bindings


:::{tab-item} C ABI · libc4a
:sync: c
:class-label: lang-c

```c
c4a_status_t c4a_pp_first_derivative_create( c4a_pp_first_derivative_handle_t** out, double delta, int32_t edge_order);
void c4a_pp_first_derivative_destroy( c4a_pp_first_derivative_handle_t* handle);
c4a_status_t c4a_pp_first_derivative_transform( const c4a_pp_first_derivative_handle_t* handle, c4a_matrix_view_t X, c4a_matrix_view_t out);
```

:::

:::{tab-item} Python · chemometrics4all
:sync: python
:class-label: lang-python

```python
from chemometrics4all import FirstDerivative

op = FirstDerivative(delta=1.0, edge_order=2)
Xt = op.fit_transform(X)
```

:::

:::{tab-item} R · chemometrics4all
:sync: r
:class-label: lang-r

```r
library(chemometrics4all)
res <- first_derivative(X)
```

:::

::::


**Registry parity references** 📐

:::{card}
:class-card: external-refs

- 📐 **`ref.nirs4all`** (Python · canonical) — `nirs4all.FirstDerivative` · nirs4all@cd731a23+dirty
- 📐 **`ref.numpy`** (Python · comparator) — `numpy.gradient` · numpy 2.3.5
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
<tr class="bk-row"><td class="bk-name"><code>C++</code></td><td class="parity parity-exact">✓ exact</td><td class="ms ms-best">🏆 0.003 ms</td><td class="ms ms-best">🏆 0.022 ms</td><td class="ms ms-best">🏆 0.144 ms</td></tr>
</tbody>
<tbody class="lang-band lang-python"><tr class="lang-band-row" data-lang="python"><th colspan="5" scope="rowgroup"><span class="lang-band-dot"></span>Python · chemometrics4all</th></tr>
<tr class="bk-row"><td class="bk-name"><code>Py</code></td><td class="parity parity-exact">✓ bind</td><td class="ms">0.008 ms</td><td class="ms">0.029 ms</td><td class="ms">0.152 ms</td></tr>
</tbody>
<tbody class="lang-band lang-r"><tr class="lang-band-row" data-lang="r"><th colspan="5" scope="rowgroup"><span class="lang-band-dot"></span>R · chemometrics4all</th></tr>
<tr class="bk-row"><td class="bk-name"><code>R</code></td><td class="parity parity-exact">✓ bind</td><td class="ms">0.026 ms</td><td class="ms">0.258 ms</td><td class="ms">1.594 ms</td></tr>
</tbody>
<tbody class="lang-band lang-python"><tr class="lang-band-row" data-lang="python"><th colspan="5" scope="rowgroup"><span class="lang-band-dot"></span>Python · external</th></tr>
<tr class="bk-row truth-source-strict"><td class="bk-name"><span class="truth-mark" title="Registry parity reference (Python): nirs4all.FirstDerivative · nirs4all@cd731a23+dirty — canonical">📐</span><code>ref.nirs4all</code></td><td class="parity parity-exact">✓ ref</td><td class="ms">0.020 ms</td><td class="ms">0.111 ms</td><td class="ms">0.592 ms</td></tr>
<tr class="bk-row truth-source-relaxed"><td class="bk-name"><span class="truth-mark" title="Registry parity reference (Python): numpy.gradient · numpy 2.3.5 — comparator">📐</span><code>ref.numpy</code></td><td class="parity parity-exact">✓ ref</td><td class="ms">0.027 ms</td><td class="ms">0.082 ms</td><td class="ms">0.426 ms</td></tr>
</tbody>
</table>
</div>

:::
::::

---

_See also_: [benchmark overview](../benchmarks/overview.md) · [methods index](index.md) · [interactive dashboard](../landing/dashboard.md)
