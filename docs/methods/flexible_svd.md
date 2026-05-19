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

Reference backends are registered in the benchmark matrix and stored as reproducible snapshots when they define the canonical contract.

### Implementations

| Layer | Entry point | Language | Contract |
|-------|-------------|----------|----------|
| C ABI | `c4a_pp_flex_svd` | C/C++ | Stable libc4a entry point family. |
| Python | `chemometrics4all.FlexibleSVD` | Python | sklearn-style wrapper backed by ctypes. |
| R | `flexible_svd(X, n_components = 5.0)` | R | Public package wrapper around the C ABI. |
| ref.nirs4all | `nirs4all.FlexibleSVD` | Python | canonical/comparator |
| ref.sklearn | `sklearn.decomposition.TruncatedSVD` | Python | canonical/comparator |

### Usage

Every chemometrics4all binding dispatches into the same C kernel. The registry references are listed in the parity card below.

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

:::{tab-item} Python · chemometrics4all
:sync: python
:class-label: lang-python

```python
from chemometrics4all import FlexibleSVD

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


**Registry parity references** 📐

:::{card}
:class-card: external-refs

- 📐 **`ref.nirs4all`** (Python · canonical) — `nirs4all.FlexibleSVD` · nirs4all@cd731a23+dirty
- 📐 **`ref.sklearn`** (Python · comparator) — `sklearn.decomposition.TruncatedSVD` · sklearn 1.8.0
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
<tr class="bk-row"><td class="bk-name"><code>C4A.cpp</code></td><td class="parity parity-exact">✓ exact</td><td class="ms">0.211 ms</td><td class="ms ms-best">🏆 1.869 ms</td><td class="ms">5.263 ms</td></tr>
</tbody>
<tbody class="lang-band lang-python"><tr class="lang-band-row" data-lang="python"><th colspan="5" scope="rowgroup"><span class="lang-band-dot"></span>Python · chemometrics4all</th></tr>
<tr class="bk-row"><td class="bk-name"><code>C4A.sklearn</code></td><td class="parity parity-exact">✓ bind</td><td class="ms">0.212 ms</td><td class="ms">1.950 ms</td><td class="ms ms-best">🏆 4.803 ms</td></tr>
</tbody>
<tbody class="lang-band lang-r"><tr class="lang-band-row" data-lang="r"><th colspan="5" scope="rowgroup"><span class="lang-band-dot"></span>R · chemometrics4all</th></tr>
<tr class="bk-row"><td class="bk-name"><code>C4A.R</code></td><td class="parity parity-exact">✓ bind</td><td class="ms ms-best">🏆 0.197 ms</td><td class="ms">1.875 ms</td><td class="ms">4.844 ms</td></tr>
</tbody>
<tbody class="lang-band lang-python"><tr class="lang-band-row" data-lang="python"><th colspan="5" scope="rowgroup"><span class="lang-band-dot"></span>Python · external</th></tr>
<tr class="bk-row truth-source-strict"><td class="bk-name"><span class="truth-mark" title="Registry parity reference (Python): nirs4all.FlexibleSVD · nirs4all@cd731a23+dirty — canonical">📐</span><code>ref.nirs4all</code></td><td class="parity parity-exact">✓ ref</td><td class="ms">1.581 ms</td><td class="ms">2.550 ms</td><td class="ms">10.994 ms</td></tr>
<tr class="bk-row truth-source-relaxed"><td class="bk-name"><span class="truth-mark" title="Registry parity reference (Python): sklearn.decomposition.TruncatedSVD · sklearn 1.8.0 — comparator">📐</span><code>ref.sklearn</code></td><td class="parity parity-exact">✓ ref</td><td class="ms">1.041 ms</td><td class="ms">2.108 ms</td><td class="ms">8.253 ms</td></tr>
</tbody>
</table>
</div>

:::
::::

---

_See also_: [benchmark overview](../benchmarks/overview.md) · [methods index](index.md) · [interactive dashboard](../landing/dashboard.md)
