# `flexible_svd` — Flexible SVD

_Group_: **Feature extraction** · _Registry tolerance_: sign-invariant `rtol=1e-5`, `atol=1e-8` · _Source_: [`docs/algorithms/flexible_svd.md`](../algorithms/flexible_svd.md)

## Description

From the `chemometrics4all.FlexibleSVD` Python wrapper docstring:

> SVD with integer component selection.

### Parameters

| Parameter      | Default | Meaning                                    |
| -------------- | ------- | ------------------------------------------ |
| `n_components` | —       | Flexible count or variance-ratio specifier |

The constructor rejects values `<= 0` or `NaN` with
`C4A_ERR_INVALID_ARGUMENT`.

## Explanations

### Bibliographic source

nirs4all 0.8.x ``operators/transforms/feature_selection.py``, class
``FlexibleSVD``.

### Mathematical principle

Truncated Singular Value Decomposition with a flexible component
specification. Unlike `FlexiblePCA`, `FlexibleSVD` does **not** centre
the data before computing the SVD. The number of retained components
can be supplied either as an explicit count or as a
cumulative-variance-ratio threshold.

Let $X \in \mathbb{R}^{m \times n}$ be the training matrix.

**Fit** runs a compact SVD on $X$ directly:

$$
X = U \, \mathrm{diag}(S) \, V^\top,
$$

with $k = \min(m, n)$. Signs are canonicalised so the
largest-absolute-value element of each $U_{:, j}$ is positive (with
ties broken by smallest index). This is the same `u_based_decision=True`
convention used by `FlexiblePCA`, deliberately diverging from
`sklearn.decomposition.TruncatedSVD` (which uses
`u_based_decision=False` because it has no centring step). The frozen
NumPy reference matches the c4a convention; sklearn does not.

Explained variance and the cumulative-variance ratio are computed
following sklearn's TruncatedSVD recipe (which is variance of the
projected scores divided by the total variance of the input):

$$
X_{\text{proj}} = U \, \mathrm{diag}(S), \qquad
\mathrm{evar}_j = \mathrm{Var}(X_{\text{proj}, j}), \qquad
\rho_j = \frac{\mathrm{evar}_j}{\sum_{i=1}^{n} \mathrm{Var}(X_{:, i})}.
$$

Variance uses `np.var(..., axis=0)` with `ddof=0`. The component
selection rule is identical to `FlexiblePCA`:

- **count mode** (when `n_components >= 1`):
  $k' = \min(\lfloor n_{components} \rfloor, k)$.
- **variance-ratio mode** (when `0 < n_components < 1`):
  smallest $k'$ such that $\sum_{j=1}^{k'} \rho_j \geq n_{components}$.

The kept components are stored as `components` $= V^\top_{[:k', :]}$.

**Transform** projects without centring:

$$
\text{out} = X_\text{new} \, \text{components}^\top.
$$

### Implementation

Stateful: `_create / _fit / _transform / _destroy` with companion
`_is_fitted` and `_output_cols` helpers. Same contract as
`FlexiblePCA` — refits replace the prior fit.

C ABI entry points used by the language bindings:

```c
c4a_status_t c4a_pp_flex_svd_create(c4a_pp_flex_svd_handle_t** out, double n_components);
void c4a_pp_flex_svd_destroy(c4a_pp_flex_svd_handle_t* handle);
c4a_status_t c4a_pp_flex_svd_fit(c4a_pp_flex_svd_handle_t* handle, c4a_matrix_view_t X);
c4a_status_t c4a_pp_flex_svd_is_fitted( const c4a_pp_flex_svd_handle_t* handle, int* out_fitted);
c4a_status_t c4a_pp_flex_svd_output_cols( const c4a_pp_flex_svd_handle_t* handle, int64_t* out_cols);
c4a_status_t c4a_pp_flex_svd_transform( const c4a_pp_flex_svd_handle_t* handle, c4a_matrix_view_t X, c4a_matrix_view_t out);
```

Benchmark comparator backends are registered in the matrix and stored as reproducible snapshots when they define the canonical contract.

### Implementations

| Layer | Entry point | Language | Contract |
|-------|-------------|----------|----------|
| C ABI | `c4a_pp_flex_svd` | C/C++ | Stable libc4a entry point family. |
| Python | `chemometrics4all.python.flexible_svd` | Python | ABI-close function backed by ctypes. |
| Python sklearn | `chemometrics4all.sklearn.FlexibleSVD` | Python | scikit-learn-compatible estimator backed by ctypes. |
| R | `flexible_svd(X, n_components = 5.0)` | R | Public package wrapper around the C ABI. |
| ref.sklearn | `sklearn.decomposition.TruncatedSVD` | Python | canonical/comparator |
| ref.nirs4all | `nirs4all.FlexibleSVD` | Python | canonical/comparator |

### Usage

Every chemometrics4all binding dispatches into the same C kernel. Registered comparator/source rows are listed in the benchmark card below.

::::{tab-set}
:class: chemometrics4all-bindings


:::{tab-item} C ABI · libc4a
:sync: c
:class-label: lang-c

```c
c4a_status_t c4a_pp_flex_svd_create(c4a_pp_flex_svd_handle_t** out, double n_components);
void c4a_pp_flex_svd_destroy(c4a_pp_flex_svd_handle_t* handle);
c4a_status_t c4a_pp_flex_svd_fit(c4a_pp_flex_svd_handle_t* handle, c4a_matrix_view_t X);
c4a_status_t c4a_pp_flex_svd_is_fitted( const c4a_pp_flex_svd_handle_t* handle, int* out_fitted);
c4a_status_t c4a_pp_flex_svd_output_cols( const c4a_pp_flex_svd_handle_t* handle, int64_t* out_cols);
c4a_status_t c4a_pp_flex_svd_transform( const c4a_pp_flex_svd_handle_t* handle, c4a_matrix_view_t X, c4a_matrix_view_t out);
```

:::

:::{tab-item} Python ABI · chemometrics4all.python
:sync: python-abi
:class-label: lang-python

```python
from chemometrics4all import python as c4a

Xt = c4a.flexible_svd(X)
```

:::

:::{tab-item} Python sklearn · chemometrics4all.sklearn
:sync: python-sklearn
:class-label: lang-python

```python
from chemometrics4all.sklearn import FlexibleSVD

op = FlexibleSVD()
Xt = op.fit_transform(X)
```

:::

:::{tab-item} R · chemometrics4all
:sync: r
:class-label: lang-r

```r
library(chemometrics4all)
res <- flexible_svd(X, n_components = 5.0)
```

:::

::::


**Benchmark Comparators And Sources** ◆

:::{card}
:class-card: external-refs

- ◆ **`ref.sklearn`** (Python · canonical) — `sklearn.decomposition.TruncatedSVD` · scikit-learn 1.8.0
- ◆ **`ref.nirs4all`** (Python · comparator) — `nirs4all.FlexibleSVD` · nirs4all@cd731a23+dirty
:::

### Validation contract

- Operation: `cross_binding_callable` · comparator: `sign_invariant_columns` · tolerance: `rtol=1e-05`, `atol=1e-08` · quality: **contract-specific**
- Default validation dataset: `100×50` · seed `20260556`
- Suites: smoke `3` cells; benchmark `11` cells · Used by component projection methods.
- Metrics: `max_abs_diff`, `rel_l2_diff`, `rms_diff`, `shape_equal`
- Truth sources: cross-binding references declared directly in `benchmarks/cross_binding/orchestrator.py`.

| Backend | Library | Gate | Comparator | Note |
|---------|---------|------|------------|------|
| `ref.sklearn` | `sklearn.decomposition.TruncatedSVD` | Python / parity | `sign_invariant_columns` |  |
| `ref.nirs4all` | `nirs4all.FlexibleSVD` | Python / parity | `sign_invariant_columns` |  |

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
<tr class="bk-row"><td class="bk-name"><code>C4A.cpp</code></td><td class="parity parity-divergence parity-exact" title="worst reference max abs diff over visible sizes">6.9e-11</td><td class="ms">0.212 ms</td><td class="ms">1.917 ms</td><td class="ms ms-best">🏆 4.122 ms</td></tr>
</tbody>
<tbody class="lang-band lang-python"><tr class="lang-band-row" data-lang="python"><th colspan="5" scope="rowgroup"><span class="lang-band-dot"></span>Python · chemometrics4all</th></tr>
<tr class="bk-row"><td class="bk-name"><code>C4A.python</code></td><td class="parity parity-divergence parity-exact" title="worst reference max abs diff over visible sizes">6.9e-11</td><td class="ms">0.222 ms</td><td class="ms">2.000 ms</td><td class="ms">4.508 ms</td></tr>
<tr class="bk-row"><td class="bk-name"><code>C4A.sklearn</code></td><td class="parity parity-divergence parity-exact" title="worst reference max abs diff over visible sizes">6.9e-11</td><td class="ms">0.219 ms</td><td class="ms">1.998 ms</td><td class="ms">5.220 ms</td></tr>
</tbody>
<tbody class="lang-band lang-r"><tr class="lang-band-row" data-lang="r"><th colspan="5" scope="rowgroup"><span class="lang-band-dot"></span>R · chemometrics4all</th></tr>
<tr class="bk-row"><td class="bk-name"><code>C4A.R</code></td><td class="parity parity-divergence parity-exact" title="worst reference max abs diff over visible sizes">6.9e-11</td><td class="ms ms-best">🏆 0.207 ms</td><td class="ms">2.016 ms</td><td class="ms">5.125 ms</td></tr>
</tbody>
<tbody class="lang-band lang-python"><tr class="lang-band-row" data-lang="python"><th colspan="5" scope="rowgroup"><span class="lang-band-dot"></span>Python · external</th></tr>
<tr class="bk-row truth-source-relaxed"><td class="bk-name"><span class="truth-mark" title="Registry parity reference (Python): nirs4all.FlexibleSVD · nirs4all@cd731a23+dirty — comparator">◆</span><code>ref.nirs4all</code></td><td class="parity parity-divergence parity-exact" title="worst reference max abs diff over visible sizes">6.4e-14</td><td class="ms">1.129 ms</td><td class="ms">2.370 ms</td><td class="ms">8.994 ms</td></tr>
<tr class="bk-row truth-source-strict"><td class="bk-name"><span class="truth-mark" title="Registry parity reference (Python): sklearn.decomposition.TruncatedSVD · scikit-learn 1.8.0 — canonical">◆</span><code>ref.sklearn</code></td><td class="parity parity-divergence parity-exact" title="worst reference max abs diff over visible sizes">0</td><td class="ms">1.006 ms</td><td class="ms ms-best">🏆 1.856 ms</td><td class="ms">7.367 ms</td></tr>
</tbody>
</table>
</div>

:::
::::

---

_See also_: [benchmark overview](../benchmarks/overview.md) · [methods index](index.md) · [interactive dashboard](../landing/dashboard.md)
