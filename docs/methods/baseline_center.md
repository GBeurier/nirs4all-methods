# `baseline_center` — Baseline center

_Group_: **Preprocessing** · _Registry tolerance_: `rtol=1e-5`, `atol=1e-8` · _Source_: [`docs/algorithms/baseline_center.md`](../algorithms/baseline_center.md)

## Description

From the `chemometrics4all.BaselineCenter` Python wrapper docstring:

> Column-mean baseline centering.

### Parameters

This operator takes no constructor parameters. Use `_create()`, then
`_fit(X)`, then `_transform(X)` / `_inverse_transform(X)`.

## Explanations

### Bibliographic source

Implementation matches `nirs4all.operators.transforms.signal.Baseline`.
The arithmetic is the same as scikit-learn's
`sklearn.preprocessing.StandardScaler(with_std=False)`.

### Mathematical principle

Per-column mean centering. Let $X \in \mathbb{R}^{n \times p}$ be the
training matrix.

**Fit** computes per-column means:

$$
\mu_j = \overline{X_{:, j}} = \frac{1}{n} \sum_{i=1}^{n} X_{i,j}.
$$

**Transform**:

$$
X'_{i,j} = X_{i,j} - \mu_j.
$$

**Inverse transform**:

$$
X_{i,j} = X'_{i,j} + \mu_j.
$$

### Implementation

* `_fit` requires `rows >= 1` and `cols >= 1`.
* `_transform` and `_inverse_transform` return `C4A_ERR_NOT_FITTED` before
  fit, and `C4A_ERR_SHAPE_MISMATCH` on column-count drift.
* Tolerance against nirs4all + numpy: 1e-12 absolute / 1e-13 relative
  (closed-form arithmetic — no least-squares involvement).

C ABI entry points used by the language bindings:

```c
c4a_status_t c4a_pp_baseline_create(c4a_pp_baseline_handle_t** out);
void c4a_pp_baseline_destroy(c4a_pp_baseline_handle_t* handle);
c4a_status_t c4a_pp_baseline_fit(c4a_pp_baseline_handle_t* handle, c4a_matrix_view_t X);
c4a_status_t c4a_pp_baseline_inverse_transform( const c4a_pp_baseline_handle_t* handle, c4a_matrix_view_t X, c4a_matrix_view_t out);
c4a_status_t c4a_pp_baseline_is_fitted( const c4a_pp_baseline_handle_t* handle, int* out_fitted);
c4a_status_t c4a_pp_baseline_transform( const c4a_pp_baseline_handle_t* handle, c4a_matrix_view_t X, c4a_matrix_view_t out);
```

Reference backends are registered in the benchmark matrix and stored as reproducible snapshots when they define the canonical contract.

### Implementations

| Layer | Entry point | Language | Contract |
|-------|-------------|----------|----------|
| C ABI | `c4a_pp_baseline` | C/C++ | Stable libc4a entry point family. |
| Python | `chemometrics4all.BaselineCenter` | Python | sklearn-style wrapper backed by ctypes. |
| R | `baseline_center(X, X_fit = X)` | R | Public package wrapper around the C ABI. |
| ref.nirs4all | `nirs4all.Baseline` | Python | canonical/comparator |

### Usage

Every chemometrics4all binding dispatches into the same C kernel. The registry references are listed in the parity card below.

::::{tab-set}
:class: chemometrics4all-bindings


:::{tab-item} C ABI · libc4a
:sync: c
:class-label: lang-c

```c
c4a_status_t c4a_pp_baseline_create(c4a_pp_baseline_handle_t** out);
void c4a_pp_baseline_destroy(c4a_pp_baseline_handle_t* handle);
c4a_status_t c4a_pp_baseline_fit(c4a_pp_baseline_handle_t* handle, c4a_matrix_view_t X);
c4a_status_t c4a_pp_baseline_inverse_transform( const c4a_pp_baseline_handle_t* handle, c4a_matrix_view_t X, c4a_matrix_view_t out);
c4a_status_t c4a_pp_baseline_is_fitted( const c4a_pp_baseline_handle_t* handle, int* out_fitted);
c4a_status_t c4a_pp_baseline_transform( const c4a_pp_baseline_handle_t* handle, c4a_matrix_view_t X, c4a_matrix_view_t out);
```

:::

:::{tab-item} Python · chemometrics4all
:sync: python
:class-label: lang-python

```python
from chemometrics4all import BaselineCenter

op = BaselineCenter()
Xt = op.fit_transform(X)
```

:::

:::{tab-item} R · chemometrics4all
:sync: r
:class-label: lang-r

```r
library(chemometrics4all)
res <- baseline_center(X)
```

:::

::::


**Registry parity references** 📐

:::{card}
:class-card: external-refs

- 📐 **`ref.nirs4all`** (Python · canonical) — `nirs4all.Baseline` · nirs4all@cd731a23+dirty
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
<tr class="bk-row"><td class="bk-name"><code>C4A.cpp</code></td><td class="parity parity-exact">✓ exact</td><td class="ms ms-best">🏆 0.009 ms</td><td class="ms">0.031 ms</td><td class="ms">0.165 ms</td></tr>
</tbody>
<tbody class="lang-band lang-python"><tr class="lang-band-row" data-lang="python"><th colspan="5" scope="rowgroup"><span class="lang-band-dot"></span>Python · chemometrics4all</th></tr>
<tr class="bk-row"><td class="bk-name"><code>C4A.sklearn</code></td><td class="parity parity-exact">✓ bind</td><td class="ms">0.009 ms</td><td class="ms ms-best">🏆 0.030 ms</td><td class="ms ms-best">🏆 0.157 ms</td></tr>
</tbody>
<tbody class="lang-band lang-r"><tr class="lang-band-row" data-lang="r"><th colspan="5" scope="rowgroup"><span class="lang-band-dot"></span>R · chemometrics4all</th></tr>
<tr class="bk-row"><td class="bk-name"><code>C4A.R</code></td><td class="parity parity-exact">✓ bind</td><td class="ms">0.087 ms</td><td class="ms">0.400 ms</td><td class="ms">2.750 ms</td></tr>
</tbody>
<tbody class="lang-band lang-python"><tr class="lang-band-row" data-lang="python"><th colspan="5" scope="rowgroup"><span class="lang-band-dot"></span>Python · external</th></tr>
<tr class="bk-row truth-source-strict"><td class="bk-name"><span class="truth-mark" title="Registry parity reference (Python): nirs4all.Baseline · nirs4all@cd731a23+dirty — canonical">📐</span><code>ref.nirs4all</code></td><td class="parity parity-exact">✓ ref</td><td class="ms">0.013 ms</td><td class="ms">0.057 ms</td><td class="ms">0.367 ms</td></tr>
</tbody>
</table>
</div>

:::
::::

---

_See also_: [benchmark overview](../benchmarks/overview.md) · [methods index](index.md) · [interactive dashboard](../landing/dashboard.md)
