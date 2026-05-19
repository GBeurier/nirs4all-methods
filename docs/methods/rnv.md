# `rnv` — RNV

_Group_: **Preprocessing** · _Registry tolerance_: `rtol=1e-5`, `atol=1e-8` · _Source_: [`docs/algorithms/rnv.md`](../algorithms/rnv.md)

## Description

From the `chemometrics4all.RNV` Python wrapper docstring:

> Robust SNV using median + k * MAD.

### Parameters

| Name | Type | Default | Notes |
|------|------|---------|-------|
| `with_center` | `bool` | `True` |  |
| `with_scale` | `bool` | `True` |  |
| `k` | `float` | `1.4826` |  |

## Explanations

### Bibliographic source

Implementation matches `nirs4all.operators.transforms.scalers.RobustStandardNormalVariate`. The MAD consistency constant $k = 1/\Phi^{-1}(3/4) \approx 1.4826$ comes from Hampel (1974), "The Influence Curve and its Role in Robust Estimation."

### Mathematical principle

A median-based robust variant of SNV. For each spectrum $x_i$:

$$
\mathrm{med}_i = \operatorname{median}_j x_{i,j}, \qquad
\mathrm{MAD}_i = \operatorname{median}_j |x_{i,j} - \mathrm{med}_i|
$$

$$
x'_{i,j} = \frac{x_{i,j} - \mathrm{med}_i}{k \cdot \mathrm{MAD}_i}
$$

where $k = 1.4826$ is the consistency constant that makes $k \cdot \text{MAD}$ an unbiased estimator of the standard deviation under Gaussian assumptions.

When `with_center=False`, the MAD is computed on $|x_{i,j}|$ rather than on centered residuals — this matches nirs4all's `if/if` semantics (each block reads from the current state of `X`). If $k \cdot \text{MAD}_i = 0$, the divisor is replaced by $1.0$.

### Implementation

* Median computed via in-place median-of-three quickselect with Lomuto partitioning. For even sample counts the two middle order statistics are averaged via $(a + b) / 2$, matching `numpy.median`.
* The final scaling is evaluated as $(x - \text{med}) / (k \cdot \text{MAD})$ — a single division per element, matching NumPy's broadcast.

C ABI entry points used by the language bindings:

```c
c4a_status_t c4a_pp_rnv_create(c4a_pp_rnv_handle_t** out, int with_center, int with_scale, double k);
void c4a_pp_rnv_destroy(c4a_pp_rnv_handle_t* handle);
c4a_status_t c4a_pp_rnv_transform(const c4a_pp_rnv_handle_t* handle, c4a_matrix_view_t X, c4a_matrix_view_t out);
```

Reference backends are registered in the benchmark matrix and stored as reproducible snapshots when they define the canonical contract.

### Implementations

| Layer | Entry point | Language | Contract |
|-------|-------------|----------|----------|
| C ABI | `c4a_pp_rnv` | C/C++ | Stable libc4a entry point family. |
| Python | `chemometrics4all.RNV` | Python | sklearn-style wrapper backed by ctypes. |
| R | `rnv(X, with_center = TRUE, with_scale = TRUE, k = 1.4826)` | R | Public package wrapper around the C ABI. |
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
c4a_status_t c4a_pp_rnv_create(c4a_pp_rnv_handle_t** out, int with_center, int with_scale, double k);
void c4a_pp_rnv_destroy(c4a_pp_rnv_handle_t* handle);
c4a_status_t c4a_pp_rnv_transform(const c4a_pp_rnv_handle_t* handle, c4a_matrix_view_t X, c4a_matrix_view_t out);
```

:::

:::{tab-item} Python · chemometrics4all
:sync: python
:class-label: lang-python

```python
from chemometrics4all import RNV

op = RNV(with_center=True, with_scale=True, k=1.4826)
Xt = op.fit_transform(X)
```

:::

:::{tab-item} R · chemometrics4all
:sync: r
:class-label: lang-r

```r
library(chemometrics4all)
res <- rnv(X)
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
<tr class="bk-row"><td class="bk-name"><code>C4A.cpp</code></td><td class="parity parity-exact">✓ exact</td><td class="ms ms-best">🏆 0.031 ms</td><td class="ms ms-best">🏆 0.405 ms</td><td class="ms ms-best">🏆 2.085 ms</td></tr>
</tbody>
<tbody class="lang-band lang-python"><tr class="lang-band-row" data-lang="python"><th colspan="5" scope="rowgroup"><span class="lang-band-dot"></span>Python · chemometrics4all</th></tr>
<tr class="bk-row"><td class="bk-name"><code>C4A.sklearn</code></td><td class="parity parity-exact">✓ bind</td><td class="ms">0.036 ms</td><td class="ms">0.421 ms</td><td class="ms">2.099 ms</td></tr>
</tbody>
<tbody class="lang-band lang-r"><tr class="lang-band-row" data-lang="r"><th colspan="5" scope="rowgroup"><span class="lang-band-dot"></span>R · chemometrics4all</th></tr>
<tr class="bk-row"><td class="bk-name"><code>C4A.R</code></td><td class="parity parity-exact">✓ bind</td><td class="ms">0.048 ms</td><td class="ms">0.602 ms</td><td class="ms">3.438 ms</td></tr>
</tbody>
<tbody class="lang-band lang-python"><tr class="lang-band-row" data-lang="python"><th colspan="5" scope="rowgroup"><span class="lang-band-dot"></span>Python · external</th></tr>
<tr class="bk-row truth-source-strict"><td class="bk-name"><span class="truth-mark" title="Registry parity reference (Python): numpy · numpy 2.3.5 — canonical">◆</span><code>ref.numpy</code></td><td class="parity parity-exact">✓ ref</td><td class="ms">0.085 ms</td><td class="ms">0.656 ms</td><td class="ms">3.272 ms</td></tr>
</tbody>
<tbody class="lang-band lang-r"><tr class="lang-band-row" data-lang="r"><th colspan="5" scope="rowgroup"><span class="lang-band-dot"></span>R · external</th></tr>
<tr class="bk-row truth-source-relaxed"><td class="bk-name"><span class="truth-mark" title="Registry parity reference (R): R base · R base — comparator">◆</span><code>ref.r.base</code></td><td class="parity parity-exact">✓ ref</td><td class="ms">2.938 ms</td><td class="ms">5.344 ms</td><td class="ms">16.250 ms</td></tr>
</tbody>
</table>
</div>

:::
::::

---

_See also_: [benchmark overview](../benchmarks/overview.md) · [methods index](index.md) · [interactive dashboard](../landing/dashboard.md)
