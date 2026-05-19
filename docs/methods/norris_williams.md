# `norris_williams` — Norris-Williams

_Group_: **Preprocessing** · _Registry tolerance_: `rtol=1e-5`, `atol=1e-8` · _Source_: [`docs/algorithms/norris_williams.md`](../algorithms/norris_williams.md)

## Description

From the `chemometrics4all.NorrisWilliams` Python wrapper docstring:

> Segment smoothing followed by gap finite differences.

### Parameters

| Name | Type | Default | Notes |
|------|------|---------|-------|
| `segment` | `int` | `5` |  |
| `gap` | `int` | `5` |  |
| `derivative_order` | `int` | `1` |  |
| `delta` | `float` | `1.0` | Sample spacing along the wavelength axis. |

## Explanations

### Bibliographic source

Norris, K. H. & Williams, P. C. (1984).  "Optimization of mathematical
treatments of raw near-infrared signal in the measurement of protein
in hard red spring wheat."  *Cereal Chemistry* 61, 158-165.

### Mathematical principle

Two-pass derivative: a centred moving-average smoother followed by a gap
finite difference, applied `derivative_order` times.  Mirrors
`nirs4all.operators.transforms.norris_williams.norris_williams`.

Single pass on a row $x$:

$$
\tilde{x}_j = \frac{1}{\text{segment}} \sum_{k = -\lfloor \text{segment}/2 \rfloor}^{\lfloor \text{segment}/2 \rfloor} \overline{x}_{j+k},
\qquad
\hat{x}_j = \frac{\tilde{x}_{j + \text{gap}} - \tilde{x}_{j - \text{gap}}}{2 \cdot \text{gap} \cdot \delta}.
$$

where $\overline{x}$ extends $x$ to out-of-range indices by replicating
the boundary samples (edge padding).  When `derivative_order` is 2 the
(smooth, gap-diff) pair is applied a second time on the already-derived
row.

`segment = 1` disables the smoothing pass (the moving average over one
sample is the identity).  `gap` controls the spacing of the finite
difference and is independent of `segment`.

### Implementation

- `_create` returns `C4A_ERR_INVALID_ARGUMENT` for even or non-positive
  segments, $\text{gap} < 1$, $\text{derivative\_order} \notin \{1, 2\}$,
  or $\delta = 0$.
- Tolerance vs nirs4all (`np.pad` + arithmetic): $10^{-12}$ absolute /
  $10^{-13}$ relative.

C ABI entry points used by the language bindings:

```c
c4a_status_t c4a_pp_norris_williams_create( c4a_pp_norris_williams_handle_t** out, int32_t segment, int32_t gap, int32_t derivative_order, double delta);
void c4a_pp_norris_williams_destroy( c4a_pp_norris_williams_handle_t* handle);
c4a_status_t c4a_pp_norris_williams_transform( const c4a_pp_norris_williams_handle_t* handle, c4a_matrix_view_t X, c4a_matrix_view_t out);
```

Reference backends are registered in the benchmark matrix and stored as reproducible snapshots when they define the canonical contract.

### Implementations

| Layer | Entry point | Language | Contract |
|-------|-------------|----------|----------|
| C ABI | `c4a_pp_norris_williams` | C/C++ | Stable libc4a entry point family. |
| Python | `chemometrics4all.NorrisWilliams` | Python | sklearn-style wrapper backed by ctypes. |
| R | `norris_williams(X, segment = 5L, gap = 5L, derivative_order = 1L, delta = 1.0)` | R | Public package wrapper around the C ABI. |
| ref.nirs4all | `nirs4all.NorrisWilliams` | Python | canonical/comparator |

### Usage

Every chemometrics4all binding dispatches into the same C kernel. The registry references are listed in the parity card below.

::::{tab-set}
:class: chemometrics4all-bindings


:::{tab-item} C ABI · libc4a
:sync: c
:class-label: lang-c

```c
c4a_status_t c4a_pp_norris_williams_create( c4a_pp_norris_williams_handle_t** out, int32_t segment, int32_t gap, int32_t derivative_order, double delta);
void c4a_pp_norris_williams_destroy( c4a_pp_norris_williams_handle_t* handle);
c4a_status_t c4a_pp_norris_williams_transform( const c4a_pp_norris_williams_handle_t* handle, c4a_matrix_view_t X, c4a_matrix_view_t out);
```

:::

:::{tab-item} Python · chemometrics4all
:sync: python
:class-label: lang-python

```python
from chemometrics4all import NorrisWilliams

op = NorrisWilliams(segment=5, gap=5, derivative_order=1, delta=1.0)
Xt = op.fit_transform(X)
```

:::

:::{tab-item} R · chemometrics4all
:sync: r
:class-label: lang-r

```r
library(chemometrics4all)
res <- norris_williams(X, segment = 5L, gap = 5L)
```

:::

::::


**Registry parity references** 📐

:::{card}
:class-card: external-refs

- 📐 **`ref.nirs4all`** (Python · canonical) — `nirs4all.NorrisWilliams` · nirs4all@cd731a23+dirty
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
<tr class="bk-row"><td class="bk-name"><code>C4A.cpp</code></td><td class="parity parity-exact">✓ exact</td><td class="ms ms-best">🏆 0.016 ms</td><td class="ms">0.179 ms</td><td class="ms ms-best">🏆 0.769 ms</td></tr>
</tbody>
<tbody class="lang-band lang-python"><tr class="lang-band-row" data-lang="python"><th colspan="5" scope="rowgroup"><span class="lang-band-dot"></span>Python · chemometrics4all</th></tr>
<tr class="bk-row"><td class="bk-name"><code>C4A.sklearn</code></td><td class="parity parity-exact">✓ bind</td><td class="ms">0.022 ms</td><td class="ms ms-best">🏆 0.162 ms</td><td class="ms">0.827 ms</td></tr>
</tbody>
<tbody class="lang-band lang-r"><tr class="lang-band-row" data-lang="r"><th colspan="5" scope="rowgroup"><span class="lang-band-dot"></span>R · chemometrics4all</th></tr>
<tr class="bk-row"><td class="bk-name"><code>C4A.R</code></td><td class="parity parity-exact">✓ bind</td><td class="ms">0.039 ms</td><td class="ms">0.398 ms</td><td class="ms">2.234 ms</td></tr>
</tbody>
<tbody class="lang-band lang-python"><tr class="lang-band-row" data-lang="python"><th colspan="5" scope="rowgroup"><span class="lang-band-dot"></span>Python · external</th></tr>
<tr class="bk-row truth-source-strict"><td class="bk-name"><span class="truth-mark" title="Registry parity reference (Python): nirs4all.NorrisWilliams · nirs4all@cd731a23+dirty — canonical">📐</span><code>ref.nirs4all</code></td><td class="parity parity-exact">✓ ref</td><td class="ms">0.244 ms</td><td class="ms">2.404 ms</td><td class="ms">10.447 ms</td></tr>
</tbody>
</table>
</div>

:::
::::

---

_See also_: [benchmark overview](../benchmarks/overview.md) · [methods index](index.md) · [interactive dashboard](../landing/dashboard.md)
