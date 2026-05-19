# `from_absorbance` — From absorbance

_Group_: **Preprocessing** · _Registry tolerance_: `rtol=1e-5`, `atol=1e-8` · _Source_: [`docs/algorithms/from_absorbance.md`](../algorithms/from_absorbance.md)

## Description

From the `chemometrics4all.FromAbsorbance` Python wrapper docstring:

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
c4a_status_t c4a_pp_from_absorbance_create( c4a_pp_from_absorbance_handle_t** out, int is_percent);
void c4a_pp_from_absorbance_destroy( c4a_pp_from_absorbance_handle_t* handle);
c4a_status_t c4a_pp_from_absorbance_transform( const c4a_pp_from_absorbance_handle_t* handle, c4a_matrix_view_t X, c4a_matrix_view_t out);
```

Reference backends are registered in the benchmark matrix and stored as reproducible snapshots when they define the canonical contract.

### Implementations

| Layer | Entry point | Language | Contract |
|-------|-------------|----------|----------|
| C ABI | `c4a_pp_from_absorbance` | C/C++ | Stable libc4a entry point family. |
| Python | `chemometrics4all.FromAbsorbance` | Python | sklearn-style wrapper backed by ctypes. |
| R | `from_absorbance(X, is_percent = FALSE)` | R | Public package wrapper around the C ABI. |
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
c4a_status_t c4a_pp_from_absorbance_create( c4a_pp_from_absorbance_handle_t** out, int is_percent);
void c4a_pp_from_absorbance_destroy( c4a_pp_from_absorbance_handle_t* handle);
c4a_status_t c4a_pp_from_absorbance_transform( const c4a_pp_from_absorbance_handle_t* handle, c4a_matrix_view_t X, c4a_matrix_view_t out);
```

:::

:::{tab-item} Python · chemometrics4all
:sync: python
:class-label: lang-python

```python
from chemometrics4all import FromAbsorbance

op = FromAbsorbance(is_percent=False)
Xt = op.fit_transform(X)
```

:::

:::{tab-item} R · chemometrics4all
:sync: r
:class-label: lang-r

```r
library(chemometrics4all)
res <- from_absorbance(X)
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
<tr class="bk-row"><td class="bk-name"><code>C++</code></td><td class="parity parity-exact">✓ exact</td><td class="ms ms-best">🏆 0.017 ms</td><td class="ms ms-best">🏆 0.161 ms</td><td class="ms ms-best">🏆 0.834 ms</td></tr>
</tbody>
<tbody class="lang-band lang-python"><tr class="lang-band-row" data-lang="python"><th colspan="5" scope="rowgroup"><span class="lang-band-dot"></span>Python · chemometrics4all</th></tr>
<tr class="bk-row"><td class="bk-name"><code>Py</code></td><td class="parity parity-exact">✓ bind</td><td class="ms">0.023 ms</td><td class="ms">0.175 ms</td><td class="ms">0.837 ms</td></tr>
</tbody>
<tbody class="lang-band lang-r"><tr class="lang-band-row" data-lang="r"><th colspan="5" scope="rowgroup"><span class="lang-band-dot"></span>R · chemometrics4all</th></tr>
<tr class="bk-row"><td class="bk-name"><code>R</code></td><td class="parity parity-exact">✓ bind</td><td class="ms">0.035 ms</td><td class="ms">0.352 ms</td><td class="ms">1.969 ms</td></tr>
</tbody>
<tbody class="lang-band lang-python"><tr class="lang-band-row" data-lang="python"><th colspan="5" scope="rowgroup"><span class="lang-band-dot"></span>Python · external</th></tr>
<tr class="bk-row truth-source-strict"><td class="bk-name"><span class="truth-mark" title="Registry parity reference (Python): numpy · numpy 2.3.5 — canonical">📐</span><code>ref.numpy</code></td><td class="parity parity-exact">✓ ref</td><td class="ms">0.041 ms</td><td class="ms">0.407 ms</td><td class="ms">2.094 ms</td></tr>
</tbody>
<tbody class="lang-band lang-r"><tr class="lang-band-row" data-lang="r"><th colspan="5" scope="rowgroup"><span class="lang-band-dot"></span>R · external</th></tr>
<tr class="bk-row truth-source-relaxed"><td class="bk-name"><span class="truth-mark" title="Registry parity reference (R): R base · R base — comparator">📐</span><code>ref.r.base</code></td><td class="parity parity-exact">✓ ref</td><td class="ms">0.064 ms</td><td class="ms">0.637 ms</td><td class="ms">3.375 ms</td></tr>
</tbody>
</table>
</div>

:::
::::

---

_See also_: [benchmark overview](../benchmarks/overview.md) · [methods index](index.md) · [interactive dashboard](../landing/dashboard.md)
