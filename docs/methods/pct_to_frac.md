# `pct_to_frac` — Percent to fraction

_Group_: **Preprocessing** · _Registry tolerance_: `rtol=1e-5`, `atol=1e-8` · _Source_: [`docs/algorithms/percent_to_fraction.md`](../algorithms/percent_to_fraction.md)

## Description

From the `chemometrics4all.PercentToFraction` Python wrapper docstring:

> Convert percent reflectance/transmittance to fraction.

### Parameters

None — the transform is fully data-driven.

## Explanations

### Bibliographic source

Implementation matches `nirs4all.operators.transforms.signal_conversion.PercentToFraction` (0.8.11).

### Mathematical principle

Element-wise division by 100:

$$
x'_{i,j} = \frac{x_{i,j}}{100}
$$

### Implementation

* Division is performed inline as `X[i] / 100.0` — not replaced with multiplication by `0.01`, because `0.01` has no exact binary representation. The two forms diverge in the last bit; the inline divide matches numpy's element-wise true-divide bit-for-bit.
* Parity tolerance vs nirs4all reference: `1e-12 abs / 1e-13 rel`.

C ABI entry points used by the language bindings:

```c
c4a_status_t c4a_pp_pct_to_frac_create( c4a_pp_pct_to_frac_handle_t** out);
void c4a_pp_pct_to_frac_destroy( c4a_pp_pct_to_frac_handle_t* handle);
c4a_status_t c4a_pp_pct_to_frac_transform( const c4a_pp_pct_to_frac_handle_t* handle, c4a_matrix_view_t X, c4a_matrix_view_t out);
```

Reference backends are registered in the benchmark matrix and stored as reproducible snapshots when they define the canonical contract.

### Implementations

| Layer | Entry point | Language | Contract |
|-------|-------------|----------|----------|
| C ABI | `c4a_pp_pct_to_frac` | C/C++ | Stable libc4a entry point family. |
| Python | `chemometrics4all.PercentToFraction` | Python | sklearn-style wrapper backed by ctypes. |
| R | `percent_to_fraction(X)` | R | Public package wrapper around the C ABI. |
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
c4a_status_t c4a_pp_pct_to_frac_create( c4a_pp_pct_to_frac_handle_t** out);
void c4a_pp_pct_to_frac_destroy( c4a_pp_pct_to_frac_handle_t* handle);
c4a_status_t c4a_pp_pct_to_frac_transform( const c4a_pp_pct_to_frac_handle_t* handle, c4a_matrix_view_t X, c4a_matrix_view_t out);
```

:::

:::{tab-item} Python · chemometrics4all
:sync: python
:class-label: lang-python

```python
from chemometrics4all import PercentToFraction

op = PercentToFraction()
Xt = op.fit_transform(X)
```

:::

:::{tab-item} R · chemometrics4all
:sync: r
:class-label: lang-r

```r
library(chemometrics4all)
res <- percent_to_fraction(X)
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
<tr class="bk-row"><td class="bk-name"><code>C++</code></td><td class="parity parity-exact">✓ exact</td><td class="ms ms-best">🏆 0.002 ms</td><td class="ms ms-best">🏆 0.021 ms</td><td class="ms">0.113 ms</td></tr>
</tbody>
<tbody class="lang-band lang-python"><tr class="lang-band-row" data-lang="python"><th colspan="5" scope="rowgroup"><span class="lang-band-dot"></span>Python · chemometrics4all</th></tr>
<tr class="bk-row"><td class="bk-name"><code>Py</code></td><td class="parity parity-exact">✓ bind</td><td class="ms">0.008 ms</td><td class="ms">0.027 ms</td><td class="ms">0.118 ms</td></tr>
</tbody>
<tbody class="lang-band lang-r"><tr class="lang-band-row" data-lang="r"><th colspan="5" scope="rowgroup"><span class="lang-band-dot"></span>R · chemometrics4all</th></tr>
<tr class="bk-row"><td class="bk-name"><code>R</code></td><td class="parity parity-exact">✓ bind</td><td class="ms">0.025 ms</td><td class="ms">0.242 ms</td><td class="ms">1.430 ms</td></tr>
</tbody>
<tbody class="lang-band lang-python"><tr class="lang-band-row" data-lang="python"><th colspan="5" scope="rowgroup"><span class="lang-band-dot"></span>Python · external</th></tr>
<tr class="bk-row truth-source-strict"><td class="bk-name"><span class="truth-mark" title="Registry parity reference (Python): numpy · numpy 2.3.5 — canonical">📐</span><code>ref.numpy</code></td><td class="parity parity-exact">✓ ref</td><td class="ms">0.003 ms</td><td class="ms">0.022 ms</td><td class="ms ms-best">🏆 0.110 ms</td></tr>
</tbody>
<tbody class="lang-band lang-r"><tr class="lang-band-row" data-lang="r"><th colspan="5" scope="rowgroup"><span class="lang-band-dot"></span>R · external</th></tr>
<tr class="bk-row truth-source-relaxed"><td class="bk-name"><span class="truth-mark" title="Registry parity reference (R): R base · R base — comparator">📐</span><code>ref.r.base</code></td><td class="parity parity-exact">✓ ref</td><td class="ms">0.005 ms</td><td class="ms">0.038 ms</td><td class="ms">0.246 ms</td></tr>
</tbody>
</table>
</div>

:::
::::

---

_See also_: [benchmark overview](../benchmarks/overview.md) · [methods index](index.md) · [interactive dashboard](../landing/dashboard.md)
