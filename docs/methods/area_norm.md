# `area_norm` — Area normalization

_Group_: **Preprocessing** · _Registry tolerance_: `rtol=1e-5`, `atol=1e-8` · _Source_: [`docs/algorithms/area_normalization.md`](../algorithms/area_normalization.md)

## Description

From the `chemometrics4all.AreaNormalization` Python wrapper docstring:

> Per-row area normalisation.

### Parameters

| Name | Type | Default | Notes |
|------|------|---------|-------|
| `method` | `str` | `'sum'` | Algorithm variant selected by the wrapper. |

## Explanations

### Bibliographic source

Implementation matches `nirs4all.operators.transforms.nirs.AreaNormalization`. The `trapz` integrand follows the implementation used by `numpy.trapz` (which is bit-identical to `scipy.integrate.trapezoid` for unit spacing).

### Mathematical principle

Per-spectrum normalisation by a row-area metric. For each row $x_i \in \mathbb{R}^m$:

$$
A_i = \text{area}(x_i), \qquad
x'_{i,j} = \frac{x_{i,j}}{A_i}
$$

where the area metric is one of:

| Method | Formula | Notes |
| --- | --- | --- |
| `sum` | $A_i = \sum_j x_{i,j}$ | Signed total intensity. |
| `abs_sum` | $A_i = \sum_j |x_{i,j}|$ | Absolute intensity (robust to mean-centered data). |
| `trapz` | $A_i = \frac{1}{2} \sum_{j=0}^{m-2} (x_{i,j} + x_{i,j+1})$ | Trapezoidal integration with unit spacing. |

If $|A_i| < 10^{-10}$, the divisor is replaced by $1.0$ to avoid amplifying noise on near-zero rows.

### Implementation

* `sum` / `abs_sum` accumulate left-to-right with the same rounding order as `numpy.add.reduce`.
* `trapz` matches `numpy.trapz` / `scipy.integrate.trapezoid` with default $\Delta x = 1$, computed as $\frac{1}{2} \sum (x[:-1] + x[1:])$ — the pair-sum form is preserved bit-for-bit.
* Final scaling uses `x / area`, not `x * (1/area)`.

C ABI entry points used by the language bindings:

```c
c4a_status_t c4a_pp_area_create(c4a_pp_area_handle_t** out, int32_t method);
void c4a_pp_area_destroy(c4a_pp_area_handle_t* handle);
c4a_status_t c4a_pp_area_transform(const c4a_pp_area_handle_t* handle, c4a_matrix_view_t X, c4a_matrix_view_t out);
```

Reference backends are registered in the benchmark matrix and stored as reproducible snapshots when they define the canonical contract.

### Implementations

| Layer | Entry point | Language | Contract |
|-------|-------------|----------|----------|
| C ABI | `c4a_pp_area` | C/C++ | Stable libc4a entry point family. |
| Python | `chemometrics4all.AreaNormalization` | Python | sklearn-style wrapper backed by ctypes. |
| R | `area_normalization(X, method = "sum")` | R | Public package wrapper around the C ABI. |
| ref.nirs4all | `nirs4all.AreaNormalization` | Python | canonical/comparator |
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
c4a_status_t c4a_pp_area_create(c4a_pp_area_handle_t** out, int32_t method);
void c4a_pp_area_destroy(c4a_pp_area_handle_t* handle);
c4a_status_t c4a_pp_area_transform(const c4a_pp_area_handle_t* handle, c4a_matrix_view_t X, c4a_matrix_view_t out);
```

:::

:::{tab-item} Python · chemometrics4all
:sync: python
:class-label: lang-python

```python
from chemometrics4all import AreaNormalization

op = AreaNormalization(method='sum')
Xt = op.fit_transform(X)
```

:::

:::{tab-item} R · chemometrics4all
:sync: r
:class-label: lang-r

```r
library(chemometrics4all)
res <- area_normalization(X, method = 'sum')
```

:::

::::


**Registry parity references** ◆

:::{card}
:class-card: external-refs

- ◆ **`ref.nirs4all`** (Python · canonical) — `nirs4all.AreaNormalization` · nirs4all@cd731a23+dirty
- ◆ **`ref.numpy`** (Python · comparator) — `numpy` · numpy 2.3.5
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
<tr class="bk-row"><td class="bk-name"><code>C4A.cpp</code></td><td class="parity parity-exact">✓ exact</td><td class="ms ms-best">🏆 0.001 ms</td><td class="ms ms-best">🏆 0.012 ms</td><td class="ms ms-best">🏆 0.076 ms</td></tr>
</tbody>
<tbody class="lang-band lang-python"><tr class="lang-band-row" data-lang="python"><th colspan="5" scope="rowgroup"><span class="lang-band-dot"></span>Python · chemometrics4all</th></tr>
<tr class="bk-row"><td class="bk-name"><code>C4A.sklearn</code></td><td class="parity parity-exact">✓ bind</td><td class="ms">0.007 ms</td><td class="ms">0.020 ms</td><td class="ms">0.090 ms</td></tr>
</tbody>
<tbody class="lang-band lang-r"><tr class="lang-band-row" data-lang="r"><th colspan="5" scope="rowgroup"><span class="lang-band-dot"></span>R · chemometrics4all</th></tr>
<tr class="bk-row"><td class="bk-name"><code>C4A.R</code></td><td class="parity parity-exact">✓ bind</td><td class="ms">0.025 ms</td><td class="ms">0.285 ms</td><td class="ms">1.609 ms</td></tr>
</tbody>
<tbody class="lang-band lang-python"><tr class="lang-band-row" data-lang="python"><th colspan="5" scope="rowgroup"><span class="lang-band-dot"></span>Python · external</th></tr>
<tr class="bk-row truth-source-strict"><td class="bk-name"><span class="truth-mark" title="Registry parity reference (Python): nirs4all.AreaNormalization · nirs4all@cd731a23+dirty — canonical">◆</span><code>ref.nirs4all</code></td><td class="parity parity-exact">✓ ref</td><td class="ms">0.227 ms</td><td class="ms">0.282 ms</td><td class="ms">0.738 ms</td></tr>
<tr class="bk-row truth-source-relaxed"><td class="bk-name"><span class="truth-mark" title="Registry parity reference (Python): numpy · numpy 2.3.5 — comparator">◆</span><code>ref.numpy</code></td><td class="parity parity-exact">✓ ref</td><td class="ms">0.008 ms</td><td class="ms">0.043 ms</td><td class="ms">0.197 ms</td></tr>
</tbody>
<tbody class="lang-band lang-r"><tr class="lang-band-row" data-lang="r"><th colspan="5" scope="rowgroup"><span class="lang-band-dot"></span>R · external</th></tr>
<tr class="bk-row truth-source-relaxed"><td class="bk-name"><span class="truth-mark" title="Registry parity reference (R): R base · R base — comparator">◆</span><code>ref.r.base</code></td><td class="parity parity-exact">✓ ref</td><td class="ms">0.040 ms</td><td class="ms">0.272 ms</td><td class="ms">1.484 ms</td></tr>
</tbody>
</table>
</div>

:::
::::

---

_See also_: [benchmark overview](../benchmarks/overview.md) · [methods index](index.md) · [interactive dashboard](../landing/dashboard.md)
