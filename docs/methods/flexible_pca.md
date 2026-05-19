# `flexible_pca` — Flexible PCA

_Group_: **Feature extraction** · _Registry tolerance_: sign-invariant `rtol=1e-5`, `atol=1e-8` · _Source_: [`docs/algorithms/flexible_pca.md`](../algorithms/flexible_pca.md)

## Description

From the `chemometrics4all.FlexiblePCA` Python wrapper docstring:

> PCA with integer or explained-variance component selection.

### Parameters

| Parameter      | Default | Meaning                                    |
| -------------- | ------- | ------------------------------------------ |
| `n_components` | —       | Flexible count or variance-ratio specifier |

The constructor rejects values `<= 0` or `NaN` with
`C4A_ERR_INVALID_ARGUMENT`.

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

The c4a engine uses a one-sided Jacobi SVD (``cpp/src/core/common/svd.c``)
which agrees with ``np.linalg.svd`` to within a few ULPs on the singular
values; the resulting transform parity tolerance is ``1e-10`` absolute /
``1e-11`` relative against the frozen NumPy reference.

### Implementation

Stateful: `_create / _fit / _transform / _destroy` with companion
`_is_fitted` and `_output_cols` helpers. `_output_cols` reads the
learned $k'$ off the fitted state. Calling `_transform` or
`_output_cols` before `_fit` returns `C4A_ERR_NOT_FITTED`. Calling
`_fit` again replaces the prior fit (sklearn-style refit semantics).

C ABI entry points used by the language bindings:

```c
c4a_status_t c4a_pp_flex_pca_create(c4a_pp_flex_pca_handle_t** out, double n_components);
void c4a_pp_flex_pca_destroy(c4a_pp_flex_pca_handle_t* handle);
c4a_status_t c4a_pp_flex_pca_fit(c4a_pp_flex_pca_handle_t* handle, c4a_matrix_view_t X);
c4a_status_t c4a_pp_flex_pca_is_fitted( const c4a_pp_flex_pca_handle_t* handle, int* out_fitted);
c4a_status_t c4a_pp_flex_pca_output_cols( const c4a_pp_flex_pca_handle_t* handle, int64_t* out_cols);
c4a_status_t c4a_pp_flex_pca_transform( const c4a_pp_flex_pca_handle_t* handle, c4a_matrix_view_t X, c4a_matrix_view_t out);
```

Reference backends are registered in the benchmark matrix and stored as reproducible snapshots when they define the canonical contract.

### Implementations

| Layer | Entry point | Language | Contract |
|-------|-------------|----------|----------|
| C ABI | `c4a_pp_flex_pca` | C/C++ | Stable libc4a entry point family. |
| Python | `chemometrics4all.FlexiblePCA` | Python | sklearn-style wrapper backed by ctypes. |
| R | `flexible_pca(X, n_components = 5.0)` | R | Public package wrapper around the C ABI. |
| ref.nirs4all | `nirs4all.FlexiblePCA` | Python | canonical/comparator |
| ref.sklearn | `sklearn.decomposition.PCA` | Python | canonical/comparator |

### Usage

Every chemometrics4all binding dispatches into the same C kernel. The registry references are listed in the parity card below.

::::{tab-set}
:class: chemometrics4all-bindings


:::{tab-item} C ABI · libc4a
:sync: c
:class-label: lang-c

```c
c4a_status_t c4a_pp_flex_pca_create(c4a_pp_flex_pca_handle_t** out, double n_components);
void c4a_pp_flex_pca_destroy(c4a_pp_flex_pca_handle_t* handle);
c4a_status_t c4a_pp_flex_pca_fit(c4a_pp_flex_pca_handle_t* handle, c4a_matrix_view_t X);
c4a_status_t c4a_pp_flex_pca_is_fitted( const c4a_pp_flex_pca_handle_t* handle, int* out_fitted);
c4a_status_t c4a_pp_flex_pca_output_cols( const c4a_pp_flex_pca_handle_t* handle, int64_t* out_cols);
c4a_status_t c4a_pp_flex_pca_transform( const c4a_pp_flex_pca_handle_t* handle, c4a_matrix_view_t X, c4a_matrix_view_t out);
```

:::

:::{tab-item} Python · chemometrics4all
:sync: python
:class-label: lang-python

```python
from chemometrics4all import FlexiblePCA

op = FlexiblePCA()
Xt = op.fit_transform(X)
```

:::

:::{tab-item} R · chemometrics4all
:sync: r
:class-label: lang-r

```r
library(chemometrics4all)
res <- flexible_pca(X, n_components = 5.0)
```

:::

::::


**Registry parity references** 📐

:::{card}
:class-card: external-refs

- 📐 **`ref.nirs4all`** (Python · canonical) — `nirs4all.FlexiblePCA` · nirs4all@cd731a23+dirty
- 📐 **`ref.sklearn`** (Python · comparator) — `sklearn.decomposition.PCA` · sklearn 1.8.0
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
<tr class="bk-row"><td class="bk-name"><code>C++</code></td><td class="parity parity-exact">✓ exact</td><td class="ms">0.214 ms</td><td class="ms ms-best">🏆 1.907 ms</td><td class="ms ms-best">🏆 4.992 ms</td></tr>
</tbody>
<tbody class="lang-band lang-python"><tr class="lang-band-row" data-lang="python"><th colspan="5" scope="rowgroup"><span class="lang-band-dot"></span>Python · chemometrics4all</th></tr>
<tr class="bk-row"><td class="bk-name"><code>Py</code></td><td class="parity parity-exact">✓ bind</td><td class="ms">0.220 ms</td><td class="ms">1.953 ms</td><td class="ms">5.152 ms</td></tr>
</tbody>
<tbody class="lang-band lang-r"><tr class="lang-band-row" data-lang="r"><th colspan="5" scope="rowgroup"><span class="lang-band-dot"></span>R · chemometrics4all</th></tr>
<tr class="bk-row"><td class="bk-name"><code>R</code></td><td class="parity parity-exact">✓ bind</td><td class="ms ms-best">🏆 0.211 ms</td><td class="ms">1.953 ms</td><td class="ms">5.219 ms</td></tr>
</tbody>
<tbody class="lang-band lang-python"><tr class="lang-band-row" data-lang="python"><th colspan="5" scope="rowgroup"><span class="lang-band-dot"></span>Python · external</th></tr>
<tr class="bk-row truth-source-strict"><td class="bk-name"><span class="truth-mark" title="Registry parity reference (Python): nirs4all.FlexiblePCA · nirs4all@cd731a23+dirty — canonical">📐</span><code>ref.nirs4all</code></td><td class="parity parity-exact">✓ ref</td><td class="ms">0.680 ms</td><td class="ms">3.277 ms</td><td class="ms">20.428 ms</td></tr>
<tr class="bk-row truth-source-relaxed"><td class="bk-name"><span class="truth-mark" title="Registry parity reference (Python): sklearn.decomposition.PCA · sklearn 1.8.0 — comparator">📐</span><code>ref.sklearn</code></td><td class="parity parity-exact">✓ ref</td><td class="ms">0.567 ms</td><td class="ms">2.940 ms</td><td class="ms">17.032 ms</td></tr>
</tbody>
</table>
</div>

:::
::::

---

_See also_: [benchmark overview](../benchmarks/overview.md) · [methods index](index.md) · [interactive dashboard](../landing/dashboard.md)
