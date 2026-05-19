# `normalize` — Normalize

_Group_: **Preprocessing** · _Registry tolerance_: `rtol=1e-5`, `atol=1e-8` · _Source_: [`docs/algorithms/normalize.md`](../algorithms/normalize.md)

## Description

From the `chemometrics4all.Normalize` Python wrapper docstring:

> Column-wise normalisation.

### Parameters

| Name | Type | Default | Notes |
|------|------|---------|-------|
| `feature_min` | `float` | `-1.0` |  |
| `feature_max` | `float` | `1.0` |  |

## Explanations

### Bibliographic source

Implementation matches `nirs4all.operators.transforms.scalers.Normalize`. Note this is **column-wise** — distinct from `sklearn.preprocessing.normalize`, which is row-wise.

### Mathematical principle

Column-wise (axis=0) normalisation with two modes selected by the desired output range $(\ell, u)$:

* **linalg-norm mode** (`(ell, u) == (-1, 1)`, the default in nirs4all): each column is divided by its L2 norm.
  $$
  n_j = \sqrt{\sum_{i=1}^{n} x_{i,j}^2}, \qquad
  x'_{i,j} = \frac{x_{i,j}}{n_j}
  $$

* **user-defined-range mode** (any other $(\ell, u)$): each column is min-max scaled into $[\ell, u]$.
  $$
  \mu_j = \min_i x_{i,j}, \;\; M_j = \max_i x_{i,j}, \;\;
  f_j = \frac{u - \ell}{M_j - \mu_j}, \;\;
  x'_{i,j} = \ell + f_j (x_{i,j} - \mu_j)
  $$

### Implementation

* Column sum-of-squares accumulates left-to-right with the same rounding sequence as `np.linalg.norm(X, axis=0)`.
* All divisions are evaluated per element (no precomputed reciprocal-multiplication), matching `X / linalg_norm_` in nirs4all.
* The user-defined-range arithmetic exactly mirrors `imin + f * (X - min_)` from nirs4all (single multiplication, single addition).

C ABI entry points used by the language bindings:

```c
c4a_status_t c4a_pp_normalize_create(c4a_pp_normalize_handle_t** out, double feature_min, double feature_max);
void c4a_pp_normalize_destroy(c4a_pp_normalize_handle_t* handle);
c4a_status_t c4a_pp_normalize_transform( const c4a_pp_normalize_handle_t* handle, c4a_matrix_view_t X, c4a_matrix_view_t out);
```

Reference backends are registered in the benchmark matrix and stored as reproducible snapshots when they define the canonical contract.

### Implementations

| Layer | Entry point | Language | Contract |
|-------|-------------|----------|----------|
| C ABI | `c4a_pp_normalize` | C/C++ | Stable libc4a entry point family. |
| Python | `chemometrics4all.Normalize` | Python | sklearn-style wrapper backed by ctypes. |
| R | `normalize(X, feature_min = -1.0, feature_max = 1.0)` | R | Public package wrapper around the C ABI. |
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
c4a_status_t c4a_pp_normalize_create(c4a_pp_normalize_handle_t** out, double feature_min, double feature_max);
void c4a_pp_normalize_destroy(c4a_pp_normalize_handle_t* handle);
c4a_status_t c4a_pp_normalize_transform( const c4a_pp_normalize_handle_t* handle, c4a_matrix_view_t X, c4a_matrix_view_t out);
```

:::

:::{tab-item} Python · chemometrics4all
:sync: python
:class-label: lang-python

```python
from chemometrics4all import Normalize

op = Normalize(feature_min=-1.0, feature_max=1.0)
Xt = op.fit_transform(X)
```

:::

:::{tab-item} R · chemometrics4all
:sync: r
:class-label: lang-r

```r
library(chemometrics4all)
res <- normalize(X, feature_min = -1, feature_max = 1)
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
<tr class="bk-row"><td class="bk-name"><code>C4A.cpp</code></td><td class="parity parity-exact">✓ exact</td><td class="ms ms-best">🏆 0.002 ms</td><td class="ms ms-best">🏆 0.016 ms</td><td class="ms ms-best">🏆 0.119 ms</td></tr>
</tbody>
<tbody class="lang-band lang-python"><tr class="lang-band-row" data-lang="python"><th colspan="5" scope="rowgroup"><span class="lang-band-dot"></span>Python · chemometrics4all</th></tr>
<tr class="bk-row"><td class="bk-name"><code>C4A.sklearn</code></td><td class="parity parity-exact">✓ bind</td><td class="ms">0.008 ms</td><td class="ms">0.023 ms</td><td class="ms">0.132 ms</td></tr>
</tbody>
<tbody class="lang-band lang-r"><tr class="lang-band-row" data-lang="r"><th colspan="5" scope="rowgroup"><span class="lang-band-dot"></span>R · chemometrics4all</th></tr>
<tr class="bk-row"><td class="bk-name"><code>C4A.R</code></td><td class="parity parity-exact">✓ bind</td><td class="ms">0.026 ms</td><td class="ms">0.260 ms</td><td class="ms">1.648 ms</td></tr>
</tbody>
<tbody class="lang-band lang-python"><tr class="lang-band-row" data-lang="python"><th colspan="5" scope="rowgroup"><span class="lang-band-dot"></span>Python · external</th></tr>
<tr class="bk-row truth-source-strict"><td class="bk-name"><span class="truth-mark" title="Registry parity reference (Python): numpy · numpy 2.3.5 — canonical">📐</span><code>ref.numpy</code></td><td class="parity parity-exact">✓ ref</td><td class="ms">0.009 ms</td><td class="ms">0.056 ms</td><td class="ms">0.272 ms</td></tr>
</tbody>
<tbody class="lang-band lang-r"><tr class="lang-band-row" data-lang="r"><th colspan="5" scope="rowgroup"><span class="lang-band-dot"></span>R · external</th></tr>
<tr class="bk-row truth-source-relaxed"><td class="bk-name"><span class="truth-mark" title="Registry parity reference (R): R base · R base — comparator">📐</span><code>ref.r.base</code></td><td class="parity parity-exact">✓ ref</td><td class="ms">0.039 ms</td><td class="ms">0.242 ms</td><td class="ms">1.859 ms</td></tr>
</tbody>
</table>
</div>

:::
::::

---

_See also_: [benchmark overview](../benchmarks/overview.md) · [methods index](index.md) · [interactive dashboard](../landing/dashboard.md)
