# `epo` — EPO

_Group_: **Feature extraction** · _Registry tolerance_: `rtol=1e-5`, `atol=1e-8` · _Source_: [`docs/algorithms/epo.md`](../algorithms/epo.md)

## Description

From the `chemometrics4all.EPO` Python wrapper docstring:

> External Parameter Orthogonalisation.

### Parameters

| Name | Type | Default | Notes |
|------|------|---------|-------|
| `scale` | `bool` | `True` | Whether to standardize internal variables before fitting. |

## Explanations

### Bibliographic source

1. Roger, J. M., Chauchard, F., Bellon-Maurel, V. (2003). EPO–PLS:
   external parameter orthogonalisation of PLS. Application to
   temperature-independent measurement of sugar content of intact
   fruits. *Chemometrics and Intelligent Laboratory Systems*, 66(2),
   191-204.

### Mathematical principle

For a univariate external parameter `d` (length `n_samples`):

1. Compute calibration means (when `scale != 0`):
   `X_mean = mean(X, axis=0)`,  `d_mean = mean(d)`.
   When `scale == 0` both means are taken as zero.
2. Solve the per-feature scalar regression
   `B[j] = (d − d_mean)^T (X[:, j] − X_mean[j]) / (d − d_mean)^T (d − d_mean)`,
   for `j = 0..cols-1`.

At fit time the c4a engine stores `X_mean`, `d_mean`, and `B`. The
training-time output `X - outer(d - d_mean, B)` is what nirs4all's
`fit_transform(X, d)` returns.

### Implementation

C ABI entry points used by the language bindings:

```c
c4a_status_t c4a_pp_epo_create(c4a_pp_epo_handle_t** out, int scale);
void c4a_pp_epo_destroy(c4a_pp_epo_handle_t* handle);
c4a_status_t c4a_pp_epo_fit(c4a_pp_epo_handle_t* handle, c4a_matrix_view_t X, const double* d, int64_t d_len);
c4a_status_t c4a_pp_epo_inverse_transform( const c4a_pp_epo_handle_t* handle, c4a_matrix_view_t X, c4a_matrix_view_t out);
c4a_status_t c4a_pp_epo_is_fitted(const c4a_pp_epo_handle_t* handle, int* out_fitted);
c4a_status_t c4a_pp_epo_transform(const c4a_pp_epo_handle_t* handle, c4a_matrix_view_t X, c4a_matrix_view_t out);
c4a_status_t c4a_pp_epo_transform_with_d( const c4a_pp_epo_handle_t* handle, c4a_matrix_view_t X, const double* d, int64_t d_len, c4a_matrix_view_t out);
```

Reference backends are registered in the benchmark matrix and stored as reproducible snapshots when they define the canonical contract.

### Implementations

| Layer | Entry point | Language | Contract |
|-------|-------------|----------|----------|
| C ABI | `c4a_pp_epo` | C/C++ | Stable libc4a entry point family. |
| Python | `chemometrics4all.EPO` | Python | sklearn-style wrapper backed by ctypes. |
| R | `epo(X, d, scale = TRUE)` | R | Public package wrapper around the C ABI. |
| ref.nirs4all | `nirs4all.EPO` | Python | canonical/comparator |

### Usage

Every chemometrics4all binding dispatches into the same C kernel. The registry references are listed in the parity card below.

::::{tab-set}
:class: chemometrics4all-bindings


:::{tab-item} C ABI · libc4a
:sync: c
:class-label: lang-c

```c
c4a_status_t c4a_pp_epo_create(c4a_pp_epo_handle_t** out, int scale);
void c4a_pp_epo_destroy(c4a_pp_epo_handle_t* handle);
c4a_status_t c4a_pp_epo_fit(c4a_pp_epo_handle_t* handle, c4a_matrix_view_t X, const double* d, int64_t d_len);
c4a_status_t c4a_pp_epo_inverse_transform( const c4a_pp_epo_handle_t* handle, c4a_matrix_view_t X, c4a_matrix_view_t out);
c4a_status_t c4a_pp_epo_is_fitted(const c4a_pp_epo_handle_t* handle, int* out_fitted);
c4a_status_t c4a_pp_epo_transform(const c4a_pp_epo_handle_t* handle, c4a_matrix_view_t X, c4a_matrix_view_t out);
c4a_status_t c4a_pp_epo_transform_with_d( const c4a_pp_epo_handle_t* handle, c4a_matrix_view_t X, const double* d, int64_t d_len, c4a_matrix_view_t out);
```

:::

:::{tab-item} Python · chemometrics4all
:sync: python
:class-label: lang-python

```python
from chemometrics4all import EPO

op = EPO(scale=True)
Xt = op.fit_transform(X, d)
```

:::

:::{tab-item} R · chemometrics4all
:sync: r
:class-label: lang-r

```r
library(chemometrics4all)
res <- epo(X, d, scale = TRUE)
```

:::

::::


**Registry parity references** ◆

:::{card}
:class-card: external-refs

- ◆ **`ref.nirs4all`** (Python · canonical) — `nirs4all.EPO` · nirs4all@cd731a23+dirty
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
<tr class="bk-row"><td class="bk-name"><code>C4A.cpp</code></td><td class="parity parity-exact">✓ exact</td><td class="ms ms-best">🏆 0.014 ms</td><td class="ms ms-best">🏆 0.054 ms</td><td class="ms">0.289 ms</td></tr>
</tbody>
<tbody class="lang-band lang-python"><tr class="lang-band-row" data-lang="python"><th colspan="5" scope="rowgroup"><span class="lang-band-dot"></span>Python · chemometrics4all</th></tr>
<tr class="bk-row"><td class="bk-name"><code>C4A.sklearn</code></td><td class="parity parity-exact">✓ bind</td><td class="ms">0.017 ms</td><td class="ms">0.055 ms</td><td class="ms ms-best">🏆 0.273 ms</td></tr>
</tbody>
<tbody class="lang-band lang-r"><tr class="lang-band-row" data-lang="r"><th colspan="5" scope="rowgroup"><span class="lang-band-dot"></span>R · chemometrics4all</th></tr>
<tr class="bk-row"><td class="bk-name"><code>C4A.R</code></td><td class="parity parity-exact">✓ bind</td><td class="ms">0.029 ms</td><td class="ms">0.283 ms</td><td class="ms">1.953 ms</td></tr>
</tbody>
<tbody class="lang-band lang-python"><tr class="lang-band-row" data-lang="python"><th colspan="5" scope="rowgroup"><span class="lang-band-dot"></span>Python · external</th></tr>
<tr class="bk-row truth-source-strict"><td class="bk-name"><span class="truth-mark" title="Registry parity reference (Python): nirs4all.EPO · nirs4all@cd731a23+dirty — canonical">◆</span><code>ref.nirs4all</code></td><td class="parity parity-exact">✓ ref</td><td class="ms">0.056 ms</td><td class="ms">0.333 ms</td><td class="ms">2.002 ms</td></tr>
</tbody>
</table>
</div>

:::
::::

---

_See also_: [benchmark overview](../benchmarks/overview.md) · [methods index](index.md) · [interactive dashboard](../landing/dashboard.md)
