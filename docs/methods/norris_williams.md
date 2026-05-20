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

Benchmark comparator backends are registered in the matrix and stored as reproducible snapshots when they define the canonical contract.

### Implementations

| Layer | Entry point | Language | Contract |
|-------|-------------|----------|----------|
| C ABI | `c4a_pp_norris_williams` | C/C++ | Stable libc4a entry point family. |
| Python | `chemometrics4all.python.norris_williams` | Python | ABI-close function backed by ctypes. |
| Python sklearn | `chemometrics4all.sklearn.NorrisWilliams` | Python | scikit-learn-compatible estimator backed by ctypes. |
| R | `norris_williams(X, segment = 5L, gap = 5L, derivative_order = 1L, delta = 1.0)` | R | Public package wrapper around the C ABI. |
| ref.nirs4all | `nirs4all.NorrisWilliams` | Python | canonical/comparator |

### Usage

Every chemometrics4all binding dispatches into the same C kernel. Registered comparator/source rows are listed in the benchmark card below.

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

:::{tab-item} Python ABI · chemometrics4all.python
:sync: python-abi
:class-label: lang-python

```python
from chemometrics4all import python as c4a

Xt = c4a.norris_williams(X)
```

:::

:::{tab-item} Python sklearn · chemometrics4all.sklearn
:sync: python-sklearn
:class-label: lang-python

```python
from chemometrics4all.sklearn import NorrisWilliams

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


**Benchmark Comparators And Sources** ◆

:::{card}
:class-card: external-refs

- ◆ **`ref.nirs4all`** (Python · canonical) — `nirs4all.NorrisWilliams` · nirs4all@cd731a23+dirty
:::

### Validation contract

- Operation: `cross_binding_callable` · comparator: `default_allclose` · tolerance: `rtol=1e-05`, `atol=1e-08` · quality: **strict**
- Default validation dataset: `100×50` · seed `20260556`
- Suites: smoke `3` cells; benchmark `11` cells · Default C/Python/reference parity comparator.
- Metrics: `max_abs_diff`, `rel_l2_diff`, `rms_diff`, `shape_equal`
- Truth sources: cross-binding references declared directly in `benchmarks/cross_binding/orchestrator.py`.

| Backend | Library | Gate | Comparator | Note |
|---------|---------|------|------------|------|
| `ref.nirs4all` | `nirs4all.NorrisWilliams` | Python / parity | `default_allclose` |  |

### Benchmarks
Median wall-clock per cell from [`docs/_static/bench-data.json`](../benchmarks/overview.md). Divergence is the worst finite value over the visible sizes for each backend, preferring reference max-abs difference and falling back to binding max-abs difference when no reference comparison is recorded. Rows without a recorded comparison show `—`; the fastest backend per column is marked 🏆.
::::{tab-set}
:class: parity-tabs

:::{tab-item} 1 thread
:sync: threads-1

<div class="parity-table-wrap">
<table class="docutils parity-grouped">
<thead><tr><th>Backend</th><th>Divergence</th><th>100×50</th><th>100×500</th><th>100×2500</th></tr></thead>
<tbody class="lang-band lang-cpp"><tr class="lang-band-row" data-lang="cpp"><th colspan="5" scope="rowgroup"><span class="lang-band-dot"></span>C++ native · libc4a</th></tr>
<tr class="bk-row"><td class="bk-name"><code>C4A.cpp</code></td><td class="parity parity-divergence parity-exact" title="worst reference max abs diff over visible sizes">0</td><td class="ms ms-best">🏆 0.017 ms</td><td class="ms ms-best">🏆 0.147 ms</td><td class="ms ms-best">🏆 0.772 ms</td></tr>
</tbody>
<tbody class="lang-band lang-python"><tr class="lang-band-row" data-lang="python"><th colspan="5" scope="rowgroup"><span class="lang-band-dot"></span>Python · chemometrics4all</th></tr>
<tr class="bk-row"><td class="bk-name"><code>C4A.python</code></td><td class="parity parity-divergence parity-exact" title="worst reference max abs diff over visible sizes">0</td><td class="ms">0.022 ms</td><td class="ms">0.165 ms</td><td class="ms">0.847 ms</td></tr>
<tr class="bk-row"><td class="bk-name"><code>C4A.sklearn</code></td><td class="parity parity-divergence parity-exact" title="worst reference max abs diff over visible sizes">0</td><td class="ms">0.026 ms</td><td class="ms">0.163 ms</td><td class="ms">0.848 ms</td></tr>
</tbody>
<tbody class="lang-band lang-r"><tr class="lang-band-row" data-lang="r"><th colspan="5" scope="rowgroup"><span class="lang-band-dot"></span>R · chemometrics4all</th></tr>
<tr class="bk-row"><td class="bk-name"><code>C4A.R</code></td><td class="parity parity-divergence parity-exact" title="worst reference max abs diff over visible sizes">5.0e-17</td><td class="ms">0.043 ms</td><td class="ms">0.359 ms</td><td class="ms">2.312 ms</td></tr>
</tbody>
<tbody class="lang-band lang-python"><tr class="lang-band-row" data-lang="python"><th colspan="5" scope="rowgroup"><span class="lang-band-dot"></span>Python · external</th></tr>
<tr class="bk-row truth-source-strict"><td class="bk-name"><span class="truth-mark" title="Registry parity reference (Python): nirs4all.NorrisWilliams · nirs4all@cd731a23+dirty — canonical">◆</span><code>ref.nirs4all</code></td><td class="parity parity-divergence parity-exact" title="worst reference max abs diff over visible sizes">0</td><td class="ms">0.289 ms</td><td class="ms">1.933 ms</td><td class="ms">11.403 ms</td></tr>
</tbody>
</table>
</div>

:::
::::

---

_See also_: [benchmark overview](../benchmarks/overview.md) · [methods index](index.md) · [interactive dashboard](../landing/dashboard.md)
