# `osc` — OSC

_Group_: **Feature extraction** · _Registry tolerance_: `rtol=1e-5`, `atol=1e-8` · _Source_: [`docs/algorithms/osc.md`](../algorithms/osc.md)

## Description

From the `chemometrics4all.OSC` Python wrapper docstring:

> Orthogonal Signal Correction.

### Parameters

| Name | Type | Default | Notes |
|------|------|---------|-------|
| `n_components` | `int` | `1` | Number of latent components or projected columns. |
| `scale` | `bool` | `True` | Whether to standardize internal variables before fitting. |

## Explanations

### Bibliographic source

1. Wold, S., Antti, H., Lindgren, F., Öhman, J. (1998). Orthogonal
   signal correction of near-infrared spectra. *Chemometrics and
   Intelligent Laboratory Systems*, 44(1-2), 175-185.
2. Westerhuis, J. A., de Jong, S., Smilde, A. K. (2001). Direct
   orthogonal signal correction. *Chemometrics and Intelligent
   Laboratory Systems*, 56(1), 13-25.
3. Fearn, T. (2000). On orthogonal signal correction. *Chemometrics
   and Intelligent Laboratory Systems*, 50(1), 47-52.

### Mathematical principle

This is the **Direct OSC (DOSC)** variant (Westerhuis 2001), more
numerically stable than the original Wold 1998 iteration.

For each row pair `(X, y)` (univariate target):

1. Center + scale `X` and `y` to their per-column / scalar mean and
   sample standard deviation (ddof=1).
2. Compute the Y-predictive direction
   `w_pls = X^T y / ||X^T y||`
   (this is the SVD-fallback path documented below).
3. For each orthogonal component `i = 1..n_components`:
   - `w = X_res^T y / ||X_res^T y||`
   - `t = X_res w`,  `p = X_res^T t / ||t||²`
   - Subtract the `w_pls` projection from the loading: `w_ortho = p − (p · w_pls) w_pls`, normalise.
   - `t_ortho = X_res w_ortho`,  `p_ortho = X_res^T t_ortho / ||t_ortho||²`
   - Deflate: `X_res ← X_res − t_ortho p_ortho^T`
   - Store `W_ortho[:, i] = w_ortho`, `P_ortho[:, i] = p_ortho`

At transform time the same deflation is replayed against the centered +
scaled input, then the result is unscaled back to the original units.

### Implementation

C ABI entry points used by the language bindings:

```c
c4a_status_t c4a_pp_osc_create(c4a_pp_osc_handle_t** out, int32_t n_components, int scale);
void c4a_pp_osc_destroy(c4a_pp_osc_handle_t* handle);
c4a_status_t c4a_pp_osc_fit(c4a_pp_osc_handle_t* handle, c4a_matrix_view_t X, const double* y, int64_t y_len);
c4a_status_t c4a_pp_osc_inverse_transform( const c4a_pp_osc_handle_t* handle, c4a_matrix_view_t X, c4a_matrix_view_t out);
c4a_status_t c4a_pp_osc_is_fitted(const c4a_pp_osc_handle_t* handle, int* out_fitted);
c4a_status_t c4a_pp_osc_transform(const c4a_pp_osc_handle_t* handle, c4a_matrix_view_t X, c4a_matrix_view_t out);
```

Reference backends are registered in the benchmark matrix and stored as reproducible snapshots when they define the canonical contract.

### Implementations

| Layer | Entry point | Language | Contract |
|-------|-------------|----------|----------|
| C ABI | `c4a_pp_osc` | C/C++ | Stable libc4a entry point family. |
| Python | `chemometrics4all.OSC` | Python | sklearn-style wrapper backed by ctypes. |
| R | `osc(X, y, n_components = 1L, scale = TRUE)` | R | Public package wrapper around the C ABI. |
| ref.nirs4all | `nirs4all.OSC` | Python | canonical/comparator |

### Usage

Every chemometrics4all binding dispatches into the same C kernel. The registry references are listed in the parity card below.

::::{tab-set}
:class: chemometrics4all-bindings


:::{tab-item} C ABI · libc4a
:sync: c
:class-label: lang-c

```c
c4a_status_t c4a_pp_osc_create(c4a_pp_osc_handle_t** out, int32_t n_components, int scale);
void c4a_pp_osc_destroy(c4a_pp_osc_handle_t* handle);
c4a_status_t c4a_pp_osc_fit(c4a_pp_osc_handle_t* handle, c4a_matrix_view_t X, const double* y, int64_t y_len);
c4a_status_t c4a_pp_osc_inverse_transform( const c4a_pp_osc_handle_t* handle, c4a_matrix_view_t X, c4a_matrix_view_t out);
c4a_status_t c4a_pp_osc_is_fitted(const c4a_pp_osc_handle_t* handle, int* out_fitted);
c4a_status_t c4a_pp_osc_transform(const c4a_pp_osc_handle_t* handle, c4a_matrix_view_t X, c4a_matrix_view_t out);
```

:::

:::{tab-item} Python · chemometrics4all
:sync: python
:class-label: lang-python

```python
from chemometrics4all import OSC

op = OSC(n_components=1, scale=True)
Xt = op.fit_transform(X, y)
```

:::

:::{tab-item} R · chemometrics4all
:sync: r
:class-label: lang-r

```r
library(chemometrics4all)
res <- osc(X, y, n_components = 1L, scale = TRUE)
```

:::

::::


**Registry parity references** ◆

:::{card}
:class-card: external-refs

- ◆ **`ref.nirs4all`** (Python · canonical) — `nirs4all.OSC` · nirs4all@cd731a23+dirty
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
<tr class="bk-row"><td class="bk-name"><code>C4A.cpp</code></td><td class="parity parity-exact">✓ exact</td><td class="ms ms-best">🏆 0.031 ms</td><td class="ms">0.241 ms</td><td class="ms ms-best">🏆 1.238 ms</td></tr>
</tbody>
<tbody class="lang-band lang-python"><tr class="lang-band-row" data-lang="python"><th colspan="5" scope="rowgroup"><span class="lang-band-dot"></span>Python · chemometrics4all</th></tr>
<tr class="bk-row"><td class="bk-name"><code>C4A.sklearn</code></td><td class="parity parity-exact">✓ bind</td><td class="ms">0.032 ms</td><td class="ms ms-best">🏆 0.239 ms</td><td class="ms">1.322 ms</td></tr>
</tbody>
<tbody class="lang-band lang-r"><tr class="lang-band-row" data-lang="r"><th colspan="5" scope="rowgroup"><span class="lang-band-dot"></span>R · chemometrics4all</th></tr>
<tr class="bk-row"><td class="bk-name"><code>C4A.R</code></td><td class="parity parity-exact">✓ bind</td><td class="ms">0.043 ms</td><td class="ms">0.484 ms</td><td class="ms">3.094 ms</td></tr>
</tbody>
<tbody class="lang-band lang-python"><tr class="lang-band-row" data-lang="python"><th colspan="5" scope="rowgroup"><span class="lang-band-dot"></span>Python · external</th></tr>
<tr class="bk-row truth-source-strict"><td class="bk-name"><span class="truth-mark" title="Registry parity reference (Python): nirs4all.OSC · nirs4all@cd731a23+dirty — canonical">◆</span><code>ref.nirs4all</code></td><td class="parity parity-exact">✓ ref</td><td class="ms">0.099 ms</td><td class="ms">0.435 ms</td><td class="ms">2.637 ms</td></tr>
</tbody>
</table>
</div>

:::
::::

---

_See also_: [benchmark overview](../benchmarks/overview.md) · [methods index](index.md) · [interactive dashboard](../landing/dashboard.md)
