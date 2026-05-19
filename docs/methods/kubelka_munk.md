# `kubelka_munk` — Kubelka-Munk

_Group_: **Preprocessing** · _Registry tolerance_: `rtol=1e-5`, `atol=1e-8` · _Source_: [`docs/algorithms/kubelka_munk.md`](../algorithms/kubelka_munk.md)

## Description

From the `chemometrics4all.KubelkaMunk` Python wrapper docstring:

> KM = (1 - R)^2 / (2 R), with R guarded by epsilon.

### Parameters

| Name | Type | Default | Notes |
|------|------|---------|-------|
| `is_percent` | `bool` | `False` |  |
| `epsilon` | `float` | `1e-10` |  |

## Explanations

### Bibliographic source

Implementation matches `nirs4all.operators.transforms.signal_conversion.KubelkaMunk` (0.8.11).

For background see Kubelka, P. & Munk, F. (1931), "An article on optics of paint layers", Z. Tech. Phys., 12, 593-601.

### Mathematical principle

Diffuse-reflectance transformation. For each element of the input matrix:

$$
R_{i,j} = \min\!\bigl(\max(\tilde{x}_{i,j},\, \varepsilon),\, 1 - \varepsilon\bigr), \qquad
F(R_{i,j}) = \frac{(1 - R_{i,j})^2}{2 R_{i,j}}, \qquad
\tilde{x}_{i,j} = \begin{cases} x_{i,j} / 100 & \text{if `is\_percent`} \\ x_{i,j} & \text{otherwise} \end{cases}
$$

The clamp keeps $R \in [\varepsilon,\, 1 - \varepsilon]$ to avoid division by zero (when $R \to 0$) and to keep the function well-defined for reflectance-like inputs that may slightly overshoot 1.

This is theoretically more appropriate for scattering media (powders) than simple $-\log_{10}(R)$, though in NIR the benefit is dataset-dependent.

### Implementation

* `is_percent` divides by `100.0` element-wise.
* The clamp is two-sided symmetric: `R = clip(x, epsilon, 1 - epsilon)`. `1 - epsilon` is computed once (not per element) and compared via the `if v > upper, v = upper` branch.
* The squared term is `(1 - R) * (1 - R)` — a single multiply of the subtraction result, matching numpy's `np.square` ufunc.
* The division `(1 - R)^2 / (2 R)` is performed as a true element-wise divide; the denominator `2.0 * R` is computed inline to mirror nirs4all's operator-precedence evaluation of the literal `(1.0 - R)**2 / (2.0 * R)` expression.
* Parity tolerance vs nirs4all reference: `1e-12 abs / 1e-13 rel`.

C ABI entry points used by the language bindings:

```c
c4a_status_t c4a_pp_kubelka_munk_create( c4a_pp_kubelka_munk_handle_t** out, int is_percent, double epsilon);
void c4a_pp_kubelka_munk_destroy( c4a_pp_kubelka_munk_handle_t* handle);
c4a_status_t c4a_pp_kubelka_munk_transform( const c4a_pp_kubelka_munk_handle_t* handle, c4a_matrix_view_t X, c4a_matrix_view_t out);
```

Reference backends are registered in the benchmark matrix and stored as reproducible snapshots when they define the canonical contract.

### Implementations

| Layer | Entry point | Language | Contract |
|-------|-------------|----------|----------|
| C ABI | `c4a_pp_kubelka_munk` | C/C++ | Stable libc4a entry point family. |
| Python | `chemometrics4all.KubelkaMunk` | Python | sklearn-style wrapper backed by ctypes. |
| R | `kubelka_munk(X, is_percent = FALSE, epsilon = 1e-10)` | R | Public package wrapper around the C ABI. |
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
c4a_status_t c4a_pp_kubelka_munk_create( c4a_pp_kubelka_munk_handle_t** out, int is_percent, double epsilon);
void c4a_pp_kubelka_munk_destroy( c4a_pp_kubelka_munk_handle_t* handle);
c4a_status_t c4a_pp_kubelka_munk_transform( const c4a_pp_kubelka_munk_handle_t* handle, c4a_matrix_view_t X, c4a_matrix_view_t out);
```

:::

:::{tab-item} Python · chemometrics4all
:sync: python
:class-label: lang-python

```python
from chemometrics4all import KubelkaMunk

op = KubelkaMunk(is_percent=False, epsilon=1e-10)
Xt = op.fit_transform(X)
```

:::

:::{tab-item} R · chemometrics4all
:sync: r
:class-label: lang-r

```r
library(chemometrics4all)
res <- kubelka_munk(X)
```

:::

::::


**Registry parity references** 📐

:::{card}
:class-card: external-refs

- 📐 **`ref.numpy`** (Python · canonical) — `numpy` · numpy 2.3.5
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
<tr class="bk-row"><td class="bk-name"><code>C++</code></td><td class="parity parity-exact">✓ exact</td><td class="ms ms-best">🏆 0.002 ms</td><td class="ms ms-best">🏆 0.022 ms</td><td class="ms ms-best">🏆 0.107 ms</td></tr>
</tbody>
<tbody class="lang-band lang-python"><tr class="lang-band-row" data-lang="python"><th colspan="5" scope="rowgroup"><span class="lang-band-dot"></span>Python · chemometrics4all</th></tr>
<tr class="bk-row"><td class="bk-name"><code>Py</code></td><td class="parity parity-exact">✓ bind</td><td class="ms">0.008 ms</td><td class="ms">0.028 ms</td><td class="ms">0.120 ms</td></tr>
</tbody>
<tbody class="lang-band lang-r"><tr class="lang-band-row" data-lang="r"><th colspan="5" scope="rowgroup"><span class="lang-band-dot"></span>R · chemometrics4all</th></tr>
<tr class="bk-row"><td class="bk-name"><code>R</code></td><td class="parity parity-exact">✓ bind</td><td class="ms">0.024 ms</td><td class="ms">0.236 ms</td><td class="ms">1.617 ms</td></tr>
</tbody>
<tbody class="lang-band lang-python"><tr class="lang-band-row" data-lang="python"><th colspan="5" scope="rowgroup"><span class="lang-band-dot"></span>Python · external</th></tr>
<tr class="bk-row truth-source-strict"><td class="bk-name"><span class="truth-mark" title="Registry parity reference (Python): numpy · numpy 2.3.5 — canonical">📐</span><code>ref.numpy</code></td><td class="parity parity-exact">✓ ref</td><td class="ms">0.009 ms</td><td class="ms">0.077 ms</td><td class="ms">0.544 ms</td></tr>
</tbody>
<tbody class="lang-band lang-r"><tr class="lang-band-row" data-lang="r"><th colspan="5" scope="rowgroup"><span class="lang-band-dot"></span>R · external</th></tr>
<tr class="bk-row truth-source-relaxed"><td class="bk-name"><span class="truth-mark" title="Registry parity reference (R): R base · R base — comparator">📐</span><code>ref.r.base</code></td><td class="parity parity-exact">✓ ref</td><td class="ms">0.036 ms</td><td class="ms">0.293 ms</td><td class="ms">1.859 ms</td></tr>
</tbody>
</table>
</div>

:::
::::

---

_See also_: [benchmark overview](../benchmarks/overview.md) · [methods index](index.md) · [interactive dashboard](../landing/dashboard.md)
