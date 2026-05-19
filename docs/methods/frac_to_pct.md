# `frac_to_pct` — Fraction to percent

_Group_: **Preprocessing** · _Registry tolerance_: `rtol=1e-5`, `atol=1e-8` · _Source_: [`docs/algorithms/fraction_to_percent.md`](../algorithms/fraction_to_percent.md)

## Description

From the `chemometrics4all.FractionToPercent` Python wrapper docstring:

> Convert fractional reflectance/transmittance to percent.

### Parameters

None — the transform is fully data-driven.

## Explanations

### Bibliographic source

Implementation matches `nirs4all.operators.transforms.signal_conversion.FractionToPercent` (0.8.11).

### Mathematical principle

Element-wise multiplication by 100:

$$
x'_{i,j} = 100 \cdot x_{i,j}
$$

### Implementation

* The multiply uses `* 100.0`. `100.0` is exactly representable in IEEE 754 binary64, so the multiply is bit-exact — equivalent to a divide by `0.01` only in exact arithmetic; in floating point the two forms differ in the last bit. We use the literal multiply to match nirs4all's `X * 100.0`.
* Parity tolerance vs nirs4all reference: `1e-12 abs / 1e-13 rel`.

C ABI entry points used by the language bindings:

```c
c4a_status_t c4a_pp_frac_to_pct_create( c4a_pp_frac_to_pct_handle_t** out);
void c4a_pp_frac_to_pct_destroy( c4a_pp_frac_to_pct_handle_t* handle);
c4a_status_t c4a_pp_frac_to_pct_transform( const c4a_pp_frac_to_pct_handle_t* handle, c4a_matrix_view_t X, c4a_matrix_view_t out);
```

Reference backends are registered in the benchmark matrix and stored as reproducible snapshots when they define the canonical contract.

### Implementations

| Layer | Entry point | Language | Contract |
|-------|-------------|----------|----------|
| C ABI | `c4a_pp_frac_to_pct` | C/C++ | Stable libc4a entry point family. |
| Python | `chemometrics4all.FractionToPercent` | Python | sklearn-style wrapper backed by ctypes. |
| R | `fraction_to_percent(X)` | R | Public package wrapper around the C ABI. |
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
c4a_status_t c4a_pp_frac_to_pct_create( c4a_pp_frac_to_pct_handle_t** out);
void c4a_pp_frac_to_pct_destroy( c4a_pp_frac_to_pct_handle_t* handle);
c4a_status_t c4a_pp_frac_to_pct_transform( const c4a_pp_frac_to_pct_handle_t* handle, c4a_matrix_view_t X, c4a_matrix_view_t out);
```

:::

:::{tab-item} Python · chemometrics4all
:sync: python
:class-label: lang-python

```python
from chemometrics4all import FractionToPercent

op = FractionToPercent()
Xt = op.fit_transform(X)
```

:::

:::{tab-item} R · chemometrics4all
:sync: r
:class-label: lang-r

```r
library(chemometrics4all)
res <- fraction_to_percent(X)
```

:::

::::


**Registry parity references** ◆

:::{card}
:class-card: external-refs

- ◆ **`ref.numpy`** (Python · canonical) — `numpy` · numpy 2.3.5
- ◆ **`ref.r.base`** (R · comparator) — `R base`
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
<tr class="bk-row"><td class="bk-name"><code>C4A.cpp</code></td><td class="parity parity-exact">✓ exact</td><td class="ms ms-best">🏆 0.001 ms</td><td class="ms ms-best">🏆 0.007 ms</td><td class="ms ms-best">🏆 0.075 ms</td></tr>
</tbody>
<tbody class="lang-band lang-python"><tr class="lang-band-row" data-lang="python"><th colspan="5" scope="rowgroup"><span class="lang-band-dot"></span>Python · chemometrics4all</th></tr>
<tr class="bk-row"><td class="bk-name"><code>C4A.sklearn</code></td><td class="parity parity-exact">✓ bind</td><td class="ms">0.006 ms</td><td class="ms">0.016 ms</td><td class="ms">0.085 ms</td></tr>
</tbody>
<tbody class="lang-band lang-r"><tr class="lang-band-row" data-lang="r"><th colspan="5" scope="rowgroup"><span class="lang-band-dot"></span>R · chemometrics4all</th></tr>
<tr class="bk-row"><td class="bk-name"><code>C4A.R</code></td><td class="parity parity-exact">✓ bind</td><td class="ms">0.023 ms</td><td class="ms">0.272 ms</td><td class="ms">1.617 ms</td></tr>
</tbody>
<tbody class="lang-band lang-python"><tr class="lang-band-row" data-lang="python"><th colspan="5" scope="rowgroup"><span class="lang-band-dot"></span>Python · external</th></tr>
<tr class="bk-row truth-source-strict"><td class="bk-name"><span class="truth-mark" title="Registry parity reference (Python): numpy · numpy 2.3.5 — canonical">◆</span><code>ref.numpy</code></td><td class="parity parity-exact">✓ ref</td><td class="ms">0.001 ms</td><td class="ms">0.009 ms</td><td class="ms">0.076 ms</td></tr>
</tbody>
<tbody class="lang-band lang-r"><tr class="lang-band-row" data-lang="r"><th colspan="5" scope="rowgroup"><span class="lang-band-dot"></span>R · external</th></tr>
<tr class="bk-row truth-source-relaxed"><td class="bk-name"><span class="truth-mark" title="Registry parity reference (R): R base · R base — comparator">◆</span><code>ref.r.base</code></td><td class="parity parity-exact">✓ ref</td><td class="ms">0.004 ms</td><td class="ms">0.041 ms</td><td class="ms">0.250 ms</td></tr>
</tbody>
</table>
</div>

:::
::::

---

_See also_: [benchmark overview](../benchmarks/overview.md) · [methods index](index.md) · [interactive dashboard](../landing/dashboard.md)
