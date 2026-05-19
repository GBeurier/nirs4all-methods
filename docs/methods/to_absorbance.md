# `to_absorbance` — To absorbance

_Group_: **Preprocessing** · _Registry tolerance_: `rtol=1e-5`, `atol=1e-8` · _Source_: [`docs/algorithms/to_absorbance.md`](../algorithms/to_absorbance.md)

## Description

From the `chemometrics4all.ToAbsorbance` Python wrapper docstring:

> A = -log10(max(R, epsilon)). Optional %-scaling.

### Parameters

| Name | Type | Default | Notes |
|------|------|---------|-------|
| `is_percent` | `bool` | `False` |  |
| `epsilon` | `float` | `1e-10` |  |
| `clip_negative` | `bool` | `True` |  |

## Explanations

### Bibliographic source

Implementation matches `nirs4all.operators.transforms.signal_conversion.ToAbsorbance` (0.8.11).

### Mathematical principle

Element-wise log-10 conversion from reflectance (or transmittance) to absorbance:

$$
A_{i,j} = -\log_{10}\!\bigl(\max(\tilde{x}_{i,j},\, \varepsilon)\bigr), \qquad
\tilde{x}_{i,j} = \begin{cases} x_{i,j} / 100 & \text{if `is\_percent`} \\ x_{i,j} & \text{otherwise} \end{cases}
$$

The clamp `max(·, epsilon)` keeps the log domain strictly positive — values at or below `epsilon` are clipped before the log.

### Implementation

* `is_percent` divides by `100.0` element-wise (not multiplication by `0.01`, which has no exact binary representation).
* The clamp is one-sided: `if x < epsilon, x = epsilon`. The two nirs4all paths (`np.clip(X, epsilon, None)` and `np.maximum(X, epsilon)`) produce identical output, so the C engine takes a single branch regardless of `clip_negative`.
* `-log10(x)` is the literal IEEE 754 `-log10(·)` call to match numpy's vectorised `-np.log10(X)` rounding.
* Parity tolerance vs nirs4all reference: `1e-12 abs / 1e-13 rel`.

C ABI entry points used by the language bindings:

```c
c4a_status_t c4a_pp_to_absorbance_create( c4a_pp_to_absorbance_handle_t** out, int is_percent, double epsilon, int clip_negative);
void c4a_pp_to_absorbance_destroy( c4a_pp_to_absorbance_handle_t* handle);
c4a_status_t c4a_pp_to_absorbance_transform( const c4a_pp_to_absorbance_handle_t* handle, c4a_matrix_view_t X, c4a_matrix_view_t out);
```

Reference backends are registered in the benchmark matrix and stored as reproducible snapshots when they define the canonical contract.

### Implementations

| Layer | Entry point | Language | Contract |
|-------|-------------|----------|----------|
| C ABI | `c4a_pp_to_absorbance` | C/C++ | Stable libc4a entry point family. |
| Python | `chemometrics4all.ToAbsorbance` | Python | sklearn-style wrapper backed by ctypes. |
| R | `to_absorbance(X, is_percent = FALSE, epsilon = 1e-10, clip_negative = TRUE)` | R | Public package wrapper around the C ABI. |
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
c4a_status_t c4a_pp_to_absorbance_create( c4a_pp_to_absorbance_handle_t** out, int is_percent, double epsilon, int clip_negative);
void c4a_pp_to_absorbance_destroy( c4a_pp_to_absorbance_handle_t* handle);
c4a_status_t c4a_pp_to_absorbance_transform( const c4a_pp_to_absorbance_handle_t* handle, c4a_matrix_view_t X, c4a_matrix_view_t out);
```

:::

:::{tab-item} Python · chemometrics4all
:sync: python
:class-label: lang-python

```python
from chemometrics4all import ToAbsorbance

op = ToAbsorbance(is_percent=False, epsilon=1e-10, clip_negative=True)
Xt = op.fit_transform(X)
```

:::

:::{tab-item} R · chemometrics4all
:sync: r
:class-label: lang-r

```r
library(chemometrics4all)
res <- to_absorbance(X)
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
<tr class="bk-row"><td class="bk-name"><code>C4A.cpp</code></td><td class="parity parity-exact">✓ exact</td><td class="ms ms-best">🏆 0.015 ms</td><td class="ms">0.160 ms</td><td class="ms">0.758 ms</td></tr>
</tbody>
<tbody class="lang-band lang-python"><tr class="lang-band-row" data-lang="python"><th colspan="5" scope="rowgroup"><span class="lang-band-dot"></span>Python · chemometrics4all</th></tr>
<tr class="bk-row"><td class="bk-name"><code>C4A.sklearn</code></td><td class="parity parity-exact">✓ bind</td><td class="ms">0.021 ms</td><td class="ms ms-best">🏆 0.156 ms</td><td class="ms ms-best">🏆 0.751 ms</td></tr>
</tbody>
<tbody class="lang-band lang-r"><tr class="lang-band-row" data-lang="r"><th colspan="5" scope="rowgroup"><span class="lang-band-dot"></span>R · chemometrics4all</th></tr>
<tr class="bk-row"><td class="bk-name"><code>C4A.R</code></td><td class="parity parity-exact">✓ bind</td><td class="ms">0.037 ms</td><td class="ms">0.332 ms</td><td class="ms">1.859 ms</td></tr>
</tbody>
<tbody class="lang-band lang-python"><tr class="lang-band-row" data-lang="python"><th colspan="5" scope="rowgroup"><span class="lang-band-dot"></span>Python · external</th></tr>
<tr class="bk-row truth-source-strict"><td class="bk-name"><span class="truth-mark" title="Registry parity reference (Python): numpy · numpy 2.3.5 — canonical">📐</span><code>ref.numpy</code></td><td class="parity parity-exact">✓ ref</td><td class="ms">0.027 ms</td><td class="ms">0.248 ms</td><td class="ms">1.362 ms</td></tr>
</tbody>
<tbody class="lang-band lang-r"><tr class="lang-band-row" data-lang="r"><th colspan="5" scope="rowgroup"><span class="lang-band-dot"></span>R · external</th></tr>
<tr class="bk-row truth-source-relaxed"><td class="bk-name"><span class="truth-mark" title="Registry parity reference (R): R base · R base — comparator">📐</span><code>ref.r.base</code></td><td class="parity parity-exact">✓ ref</td><td class="ms">0.084 ms</td><td class="ms">0.805 ms</td><td class="ms">4.094 ms</td></tr>
</tbody>
</table>
</div>

:::
::::

---

_See also_: [benchmark overview](../benchmarks/overview.md) · [methods index](index.md) · [interactive dashboard](../landing/dashboard.md)
