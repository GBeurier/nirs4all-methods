# `flexible_pca` — Flexible PCA

_Group_: **Feature extraction** · _Registry tolerance_: sign-invariant `rtol=1e-5`, `atol=1e-8` · _Source_: [`docs/algorithms/flexible_pca.md`](../algorithms/flexible_pca.md)

## Description

From the `n4m.FlexiblePCA` Python wrapper docstring:

> PCA with integer or explained-variance component selection.

### Parameters

| Parameter      | Default | Meaning                                    |
| -------------- | ------- | ------------------------------------------ |
| `n_components` | —       | Flexible count or variance-ratio specifier |

The constructor rejects values `<= 0` or `NaN` with
`N4M_ERR_INVALID_ARGUMENT`.

## Explanations

### Bibliographic source

nirs4all 0.8.x ``operators/transforms/feature_selection.py``, class
``FlexiblePCA``, with ``whiten=False``.

### Mathematical principle

PCA (Principal Component Analysis) with a flexible component
specification. The number of retained components can be supplied either
as an explicit count or as a cumulative-variance-ratio threshold.

Let $X \in \mathbb{R}^{m \times n}$ be the training matrix.

**Fit** centres the data and computes a compact SVD:

$$
\mu = \frac{1}{m} \sum_{i=1}^{m} X_{i, :}, \qquad
X_c = X - \mathbf{1}_m \mu^\top, \qquad
X_c = U \, \mathrm{diag}(S) \, V^\top
$$

with $U \in \mathbb{R}^{m \times k}$, $S \in \mathbb{R}^{k}$,
$V^\top \in \mathbb{R}^{k \times n}$, $k = \min(m, n)$.

Signs are canonicalised so the largest-absolute-value element of each
$U_{:, j}$ is positive (with ties broken by the smallest index — i.e.
``np.argmax(|U|, axis=0)``). The flip is propagated to $V^\top$.

Explained variance and the cumulative-variance ratio are:

$$
\mathrm{evar}_j = \frac{S_j^2}{m - 1}, \qquad
\rho_j = \frac{\mathrm{evar}_j}{\sum_{i=1}^{k} \mathrm{evar}_i}.
$$

The number of components $k'$ retained is:

- **count mode** (when ``n_components >= 1``):
  $k' = \min(\lfloor n_{components} \rfloor, k)$.
- **variance-ratio mode** (when ``0 < n_components < 1``):
  smallest $k'$ such that $\sum_{j=1}^{k'} \rho_j \geq n_{components}$.

The kept components are stored as ``components`` $= V^\top_{[:k', :]}$.

**Transform** projects a fresh matrix onto the learned components:

$$
\text{out} = (X_\text{new} - \mathbf{1}\mu^\top)\,\text{components}^\top.
$$

The n4m engine uses a one-sided Jacobi SVD (``cpp/src/core/common/svd.c``)
which agrees with ``np.linalg.svd`` to within a few ULPs on the singular
values; the resulting transform parity tolerance is ``1e-10`` absolute /
``1e-11`` relative against the internal parity fixture.

### Implementation

Stateful: `_create / _fit / _transform / _destroy` with companion
`_is_fitted` and `_output_cols` helpers. `_output_cols` reads the
learned $k'$ off the fitted state. Calling `_transform` or
`_output_cols` before `_fit` returns `N4M_ERR_NOT_FITTED`. Calling
`_fit` again replaces the prior fit (sklearn-style refit semantics).

C ABI entry points used by the language bindings:

```c
n4m_status_t n4m_pp_flex_pca_create(n4m_pp_flex_pca_handle_t** out, double n_components);
void n4m_pp_flex_pca_destroy(n4m_pp_flex_pca_handle_t* handle);
n4m_status_t n4m_pp_flex_pca_fit(n4m_pp_flex_pca_handle_t* handle, n4m_matrix_view_t X);
n4m_status_t n4m_pp_flex_pca_is_fitted( const n4m_pp_flex_pca_handle_t* handle, int* out_fitted);
n4m_status_t n4m_pp_flex_pca_output_cols( const n4m_pp_flex_pca_handle_t* handle, int64_t* out_cols);
n4m_status_t n4m_pp_flex_pca_transform( const n4m_pp_flex_pca_handle_t* handle, n4m_matrix_view_t X, n4m_matrix_view_t out);
```

Benchmark comparator backends are registered in the matrix and stored as reproducible snapshots when they define the canonical contract.

### Implementations

| Layer | Entry point | Language | Contract |
|-------|-------------|----------|----------|
| C ABI | `n4m_pp_flex_pca` | C/C++ | Stable libn4m entry point family. |
| Python | `n4m.python.flexible_pca` | Python | ABI-close function backed by ctypes. |
| Python sklearn | `n4m.sklearn.FlexiblePCA` | Python | scikit-learn-compatible estimator backed by ctypes. |
| R | `flexible_pca(X, n_components = 5.0)` | R | Public package wrapper around the C ABI. |
| ref.sklearn | `sklearn.decomposition.PCA` | Python | canonical/comparator |
| ref.nirs4all | `nirs4all.FlexiblePCA` | Python | canonical/comparator |

### Usage

Every nirs4all-methods binding dispatches into the same C kernel. Registered comparator/source rows are listed in the benchmark card below.

::::{tab-set}
:class: nirs4all-methods-bindings


:::{tab-item} C ABI · libn4m
:sync: c
:class-label: lang-c

```c
n4m_status_t n4m_pp_flex_pca_create(n4m_pp_flex_pca_handle_t** out, double n_components);
void n4m_pp_flex_pca_destroy(n4m_pp_flex_pca_handle_t* handle);
n4m_status_t n4m_pp_flex_pca_fit(n4m_pp_flex_pca_handle_t* handle, n4m_matrix_view_t X);
n4m_status_t n4m_pp_flex_pca_is_fitted( const n4m_pp_flex_pca_handle_t* handle, int* out_fitted);
n4m_status_t n4m_pp_flex_pca_output_cols( const n4m_pp_flex_pca_handle_t* handle, int64_t* out_cols);
n4m_status_t n4m_pp_flex_pca_transform( const n4m_pp_flex_pca_handle_t* handle, n4m_matrix_view_t X, n4m_matrix_view_t out);
```

:::

:::{tab-item} Python ABI · n4m.python
:sync: python-abi
:class-label: lang-python

```python
from n4m import python as n4m

Xt = n4m.flexible_pca(X)
```

:::

:::{tab-item} Python sklearn · n4m.sklearn
:sync: python-sklearn
:class-label: lang-python

```python
from n4m.sklearn import FlexiblePCA

op = FlexiblePCA()
Xt = op.fit_transform(X)
```

:::

:::{tab-item} R · nirs4all-methods
:sync: r
:class-label: lang-r

```r
library(n4m)
res <- flexible_pca(X, n_components = 5.0)
```

:::

::::


**Benchmark Comparators And Sources** ◆

:::{card}
:class-card: external-refs

- ◆ **`ref.sklearn`** (Python · canonical) — `sklearn.decomposition.PCA` · scikit-learn 1.8.0
- ◆ **`ref.nirs4all`** (Python · comparator) — `nirs4all.FlexiblePCA` · nirs4all@cd731a23+dirty
:::

### Validation contract

- Operation: `cross_binding_callable` · comparator: `sign_invariant_columns` · tolerance: `rtol=1e-05`, `atol=1e-08` · quality: **contract-specific**
- Default validation dataset: `100×50` · seed `20260556`
- Suites: smoke `3` cells; benchmark `11` cells · Used by component projection methods.
- Metrics: `max_abs_diff`, `rel_l2_diff`, `rms_diff`, `shape_equal`
- Truth sources: cross-binding references declared directly in `benchmarks/cross_binding/orchestrator.py`.

| Backend | Library | Gate | Comparator | Note |
|---------|---------|------|------------|------|
| `ref.sklearn` | `sklearn.decomposition.PCA` | Python / parity | `sign_invariant_columns` |  |
| `ref.nirs4all` | `nirs4all.FlexiblePCA` | Python / parity | `sign_invariant_columns` |  |

### Benchmarks
Median wall-clock per cell from [`docs/_static/bench-data.json`](../benchmarks/overview.md). Divergence is the worst finite value over the visible sizes for each backend, preferring reference max-abs difference and falling back to binding max-abs difference when no reference comparison is recorded. Rows without a recorded comparison show `—`; the fastest backend per column is marked 🏆.
::::{tab-set}
:class: parity-tabs

:::{tab-item} 1 thread
:sync: threads-1

<div class="parity-table-wrap">
<table class="docutils parity-grouped">
<thead><tr><th>Backend</th><th>Divergence</th><th>100×50</th><th>100×500</th><th>100×2500</th></tr></thead>
<tbody class="lang-band lang-cpp"><tr class="lang-band-row" data-lang="cpp"><th colspan="5" scope="rowgroup"><span class="lang-band-dot"></span>C++ native · libn4m</th></tr>
<tr class="bk-row"><td class="bk-name"><code>N4M.cpp</code></td><td class="parity parity-divergence parity-exact" title="worst reference max abs diff over visible sizes">1.6e-13</td><td class="ms">0.220 ms</td><td class="ms">1.963 ms</td><td class="ms">4.828 ms</td></tr>
</tbody>
<tbody class="lang-band lang-python"><tr class="lang-band-row" data-lang="python"><th colspan="5" scope="rowgroup"><span class="lang-band-dot"></span>Python · nirs4all-methods</th></tr>
<tr class="bk-row"><td class="bk-name"><code>N4M.python</code></td><td class="parity parity-divergence parity-exact" title="worst reference max abs diff over visible sizes">1.6e-13</td><td class="ms">0.230 ms</td><td class="ms">1.977 ms</td><td class="ms">4.875 ms</td></tr>
<tr class="bk-row"><td class="bk-name"><code>N4M.sklearn</code></td><td class="parity parity-divergence parity-exact" title="worst reference max abs diff over visible sizes">1.6e-13</td><td class="ms">0.221 ms</td><td class="ms">1.910 ms</td><td class="ms ms-best">🏆 4.787 ms</td></tr>
</tbody>
<tbody class="lang-band lang-r"><tr class="lang-band-row" data-lang="r"><th colspan="5" scope="rowgroup"><span class="lang-band-dot"></span>R · nirs4all-methods</th></tr>
<tr class="bk-row"><td class="bk-name"><code>N4M.R</code></td><td class="parity parity-divergence parity-exact" title="worst reference max abs diff over visible sizes">1.6e-13</td><td class="ms ms-best">🏆 0.219 ms</td><td class="ms ms-best">🏆 1.844 ms</td><td class="ms">5.281 ms</td></tr>
</tbody>
<tbody class="lang-band lang-python"><tr class="lang-band-row" data-lang="python"><th colspan="5" scope="rowgroup"><span class="lang-band-dot"></span>Python · external</th></tr>
<tr class="bk-row truth-source-relaxed"><td class="bk-name"><span class="truth-mark" title="Registry parity reference (Python): nirs4all.FlexiblePCA · nirs4all@cd731a23+dirty — comparator">◆</span><code>ref.nirs4all</code></td><td class="parity parity-divergence parity-exact" title="worst reference max abs diff over visible sizes">2.0e-14</td><td class="ms">0.694 ms</td><td class="ms">3.103 ms</td><td class="ms">15.909 ms</td></tr>
<tr class="bk-row truth-source-strict"><td class="bk-name"><span class="truth-mark" title="Registry parity reference (Python): sklearn.decomposition.PCA · scikit-learn 1.8.0 — canonical">◆</span><code>ref.sklearn</code></td><td class="parity parity-divergence parity-exact" title="worst reference max abs diff over visible sizes">0</td><td class="ms">0.536 ms</td><td class="ms">2.971 ms</td><td class="ms">14.640 ms</td></tr>
</tbody>
</table>
</div>

:::
::::

---

_See also_: [benchmark overview](../benchmarks/overview.md) · [methods index](index.md) · [interactive dashboard](../landing/dashboard.md)
