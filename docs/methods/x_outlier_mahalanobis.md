# `x_outlier_mahalanobis` — X outlier Mahalanobis

_Group_: **Filter** · _Registry tolerance_: `rtol=1e-5`, `atol=1e-8` · _Source_: [`docs/algorithms/x_outlier_filter.md`](../algorithms/x_outlier_filter.md)

## Description

From the `chemometrics4all.XOutlierFilter` Python wrapper docstring:

> Multivariate outlier filter on the design matrix ``X``.

<details><summary>Full wrapper docstring</summary>

```text
Multivariate outlier filter on the design matrix ``X``.

``method`` selects one of six scoring strategies; see
:c:type:`c4a_filter_x_outlier_method_t`.
```

</details>

### Parameters

| Name | Type | Default | Notes |
|------|------|---------|-------|
| `method` | `str` | `'mahalanobis'` | Algorithm variant selected by the wrapper. |
| `use_threshold` | `bool` | `False` |  |
| `threshold` | `float` | `0.0` | Outlier or quality threshold. |
| `n_components` | `int` | `0` | Number of latent components or projected columns. |
| `contamination` | `float` | `0.1` | Expected outlier fraction. |
| `seed` | `int` | `0` | Random seed for deterministic splitting or filtering. |
| `n_estimators` | `int` | `100` |  |
| `max_samples` | `int` | `256` |  |

## Explanations

### Bibliographic source

The frozen Python reference lives at
`parity/python_generator/src/c4a_parity_filters_ref/x_outlier.py`. It
imports sklearn's `EmpiricalCovariance`, `MinCovDet`, `PCA`,
`IsolationForest`, and `LocalOutlierFactor` directly and is validated
once against nirs4all 0.8.x — subsequent nirs4all releases can drift
without breaking the parity gate.

### Mathematical principle

`x_outlier_mahalanobis` follows the standard filter operator contract: an input matrix $\mathbf{X}\in\mathbb{R}^{n\times p}$ is transformed, scored, split, or filtered by the method-specific kernel while preserving the parity contract recorded by the benchmark snapshots. The precise numerical convention is the one exposed by the C ABI signatures and the registered external references below.

### Implementation

C ABI entry points used by the language bindings:

```c
c4a_status_t c4a_filter_x_outlier_apply( const c4a_filter_x_outlier_handle_t* handle, c4a_matrix_view_t X, uint8_t* mask_out, c4a_filter_stats_t* stats_out);
c4a_status_t c4a_filter_x_outlier_create( c4a_filter_x_outlier_handle_t** out, int32_t method, int use_threshold, double threshold, int32_t n_components, double contamination, uint64_t seed, int32_t n_estimators, int64_t max_samples);
void c4a_filter_x_outlier_destroy(c4a_filter_x_outlier_handle_t* handle);
c4a_status_t c4a_filter_x_outlier_fit( c4a_filter_x_outlier_handle_t* handle, c4a_matrix_view_t X);
c4a_status_t c4a_filter_x_outlier_is_fitted( const c4a_filter_x_outlier_handle_t* handle, int* out_fitted);
```

Reference backends are registered in the benchmark matrix and stored as reproducible snapshots when they define the canonical contract.

### Implementations

| Layer | Entry point | Language | Contract |
|-------|-------------|----------|----------|
| C ABI | `c4a_filter_x_outlier` | C/C++ | Stable libc4a entry point family. |
| Python | `chemometrics4all.XOutlierFilter` | Python | sklearn-style wrapper backed by ctypes. |
| R | `x_outlier_filter(X, method = "mahalanobis", threshold = NULL, n_components = min(5L, ncol(X)), contamination = 0.1, seed = 0, n_estimators = 100L, max_samples = 256)` | R | Public package wrapper around the C ABI. |
| ref.sklearn | `sklearn.PCA(full)+EmpiricalCovariance+chi2` | Python | canonical/comparator |
| ref.nirs4all | `nirs4all.XOutlierFilter(PCA auto)` | Python | context only; nirs4all leaves sklearn PCA on auto solver, which switches solver on wide matrices; c4a gates the full-SVD PCA contract |

### Usage

Every chemometrics4all binding dispatches into the same C kernel. The registry references are listed in the parity card below.

::::{tab-set}
:class: chemometrics4all-bindings


:::{tab-item} C ABI · libc4a
:sync: c
:class-label: lang-c

```c
c4a_status_t c4a_filter_x_outlier_apply( const c4a_filter_x_outlier_handle_t* handle, c4a_matrix_view_t X, uint8_t* mask_out, c4a_filter_stats_t* stats_out);
c4a_status_t c4a_filter_x_outlier_create( c4a_filter_x_outlier_handle_t** out, int32_t method, int use_threshold, double threshold, int32_t n_components, double contamination, uint64_t seed, int32_t n_estimators, int64_t max_samples);
void c4a_filter_x_outlier_destroy(c4a_filter_x_outlier_handle_t* handle);
c4a_status_t c4a_filter_x_outlier_fit( c4a_filter_x_outlier_handle_t* handle, c4a_matrix_view_t X);
c4a_status_t c4a_filter_x_outlier_is_fitted( const c4a_filter_x_outlier_handle_t* handle, int* out_fitted);
```

:::

:::{tab-item} Python · chemometrics4all
:sync: python
:class-label: lang-python

```python
from chemometrics4all import XOutlierFilter

flt = XOutlierFilter(method='mahalanobis', use_threshold=False, threshold=0.0, n_components=0, contamination=0.1, seed=0)
mask, stats = flt.fit_apply(X)
```

:::

:::{tab-item} R · chemometrics4all
:sync: r
:class-label: lang-r

```r
library(chemometrics4all)
res <- x_outlier_filter(X, method = 'mahalanobis', n_components = min(5L, ncol(X)))$mask
```

:::

::::


**Registry parity references** 📐

:::{card}
:class-card: external-refs

- 📐 **`ref.sklearn`** (Python · canonical) — `sklearn.PCA(full)+EmpiricalCovariance+chi2` · sklearn 1.8.0
- ℹ **`ref.nirs4all`** (Python · context) — `nirs4all.XOutlierFilter(PCA auto)` · nirs4all@cd731a23+dirty — nirs4all leaves sklearn PCA on auto solver, which switches solver on wide matrices; c4a gates the full-SVD PCA contract
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
<tr class="bk-row"><td class="bk-name"><code>C++</code></td><td class="parity parity-exact">✓ exact</td><td class="ms">0.223 ms</td><td class="ms">1.977 ms</td><td class="ms ms-best">🏆 4.846 ms</td></tr>
</tbody>
<tbody class="lang-band lang-python"><tr class="lang-band-row" data-lang="python"><th colspan="5" scope="rowgroup"><span class="lang-band-dot"></span>Python · chemometrics4all</th></tr>
<tr class="bk-row"><td class="bk-name"><code>Py</code></td><td class="parity parity-exact">✓ bind</td><td class="ms">0.218 ms</td><td class="ms">2.021 ms</td><td class="ms">4.920 ms</td></tr>
</tbody>
<tbody class="lang-band lang-r"><tr class="lang-band-row" data-lang="r"><th colspan="5" scope="rowgroup"><span class="lang-band-dot"></span>R · chemometrics4all</th></tr>
<tr class="bk-row"><td class="bk-name"><code>R</code></td><td class="parity parity-exact">✓ bind</td><td class="ms ms-best">🏆 0.209 ms</td><td class="ms ms-best">🏆 1.953 ms</td><td class="ms">5.375 ms</td></tr>
</tbody>
<tbody class="lang-band lang-python"><tr class="lang-band-row" data-lang="python"><th colspan="5" scope="rowgroup"><span class="lang-band-dot"></span>Python · external</th></tr>
<tr class="bk-row truth-source-relaxed"><td class="bk-name"><span class="truth-mark" title="Registry parity reference (Python): nirs4all.XOutlierFilter(PCA auto) · nirs4all@cd731a23+dirty — context">📐</span><code>ref.nirs4all</code></td><td class="parity parity-context">✓ ref</td><td class="ms">1.199 ms</td><td class="ms">3.554 ms</td><td class="ms">8.209 ms</td></tr>
<tr class="bk-row truth-source-strict"><td class="bk-name"><span class="truth-mark" title="Registry parity reference (Python): sklearn.PCA(full)+EmpiricalCovariance+chi2 · sklearn 1.8.0 — canonical">📐</span><code>ref.sklearn</code></td><td class="parity parity-exact">✓ ref</td><td class="ms">1.088 ms</td><td class="ms">3.905 ms</td><td class="ms">17.293 ms</td></tr>
</tbody>
</table>
</div>

:::
::::

---

_See also_: [benchmark overview](../benchmarks/overview.md) · [methods index](index.md) · [interactive dashboard](../landing/dashboard.md)
