# `wavelet_pca` — Wavelet PCA

_Group_: **Wavelet** · _Registry tolerance_: `rtol=1e-5`, `atol=1e-8` · _Source_: [`docs/algorithms/wavelet_pca.md`](../algorithms/wavelet_pca.md)

## Description

From the `n4m.WaveletPCA` Python wrapper docstring:

> DWT coefficient projection through PCA scores.

### Parameters

| Name | Type | Default | Notes |
|------|------|---------|-------|
| `family` | `str` | `'haar'` | Wavelet family. |
| `mode` | `str` | `'periodization'` | Boundary handling mode. |
| `max_level` | `int` | `2` |  |
| `n_components` | `float` | `5.0` | Number of latent components or projected columns. |

## Explanations

### Bibliographic source

`nirs4all.operators.transforms.nirs.WaveletPCA` (Trygg & Wold 1998,
"PLS regression on wavelet compressed NIR spectra").  The nirs4all
reference fits a separate `sklearn.decomposition.PCA` per level;
nirs4all-methods fits a single PCA on the concatenated coefficient
stack, matching the simpler "flatten + PCA" path documented in the
v1 brief.

### Mathematical principle

Two-stage stateful dimensionality reduction.  Each row is passed
through a multi-level DWT, the resulting coefficient arrays are
flattened in the canonical
$[c_A^{(K)} \,\Vert\, c_D^{(K)} \,\Vert\, \ldots \,\Vert\, c_D^{(1)}]$
order, and the per-row coefficient matrix is fed to the same PCA
pipeline as `FlexiblePCA` (centre, compact SVD, sign-canonicalised
components).

Concretely, given a training matrix $X \in \mathbb{R}^{m \times n}$ and
an effective level $K$ (clamped to `dwt_max_level(n, family)`):

1. Build $F \in \mathbb{R}^{m \times D}$ where $D = \sum_k |c^{(k)}|$
   is the flat coefficient dimension and $F_i = \text{wavedec}(X_i)$
   flattened.
2. $\mu = \frac{1}{m} \sum_i F_i$, $F_c = F - \mathbf{1}\mu^\top$.
3. Compact SVD $F_c = U S V^\top$, signs canonicalised via
   `svd_flip(U, V^\top, u_based=True)`.
4. Explained-variance ratio $\rho_j = S_j^2 / \sum_i S_i^2$, multiplied
   by $(m - 1)^{-1}$ for the variance scale.
5. Component count $k'$ selected via the flexible specifier:
   integer $\geq 1$ (count mode) or float in $(0, 1)$ (variance ratio).
6. Components $= V^\top_{[:k', :]}$, mean $= \mu$.

`_transform` then re-applies the same DWT-flatten step to fresh rows
and projects them through $(F_{\text{new}} - \mu) V^{\top \top}_{[:k', :]}$.

### Implementation

Stateful: `_create / _fit / _transform / _destroy` plus `_is_fitted`
and `_output_cols`.  Calling `_transform` or `_output_cols` before
`_fit` returns `N4M_ERR_NOT_FITTED`.

C ABI entry points used by the language bindings:

```c
n4m_status_t n4m_pp_wavelet_pca_create( n4m_pp_wavelet_pca_handle_t** out, n4m_pp_wavelet_family_t family, n4m_pp_wavelet_boundary_t mode, int32_t max_level, double n_components);
void n4m_pp_wavelet_pca_destroy( n4m_pp_wavelet_pca_handle_t* handle);
n4m_status_t n4m_pp_wavelet_pca_fit( n4m_pp_wavelet_pca_handle_t* handle, n4m_matrix_view_t X);
n4m_status_t n4m_pp_wavelet_pca_is_fitted( const n4m_pp_wavelet_pca_handle_t* handle, int* out_fitted);
n4m_status_t n4m_pp_wavelet_pca_output_cols( const n4m_pp_wavelet_pca_handle_t* handle, int64_t* out_cols);
n4m_status_t n4m_pp_wavelet_pca_transform( const n4m_pp_wavelet_pca_handle_t* handle, n4m_matrix_view_t X, n4m_matrix_view_t out);
```

Benchmark comparator backends are registered in the matrix and stored as reproducible snapshots when they define the canonical contract.

### Implementations

| Layer | Entry point | Language | Contract |
|-------|-------------|----------|----------|
| C ABI | `n4m_pp_wavelet_pca` | C/C++ | Stable libn4m entry point family. |
| Python | `n4m.python.wavelet_pca` | Python | ABI-close function backed by ctypes. |
| Python sklearn | `n4m.sklearn.WaveletPCA` | Python | scikit-learn-compatible estimator backed by ctypes. |
| ref.pywavelets | `PyWavelets.wavedec(haar, periodization)+sklearn.PCA(full)` | Python | canonical/comparator |
| ref.nirs4all | `nirs4all.WaveletPCA(per-level contract)` | Python | context only; nirs4all fits per-level PCA blocks; n4m gates the flattened-DWT PCA contract with PyWavelets+sklearn |

### Usage

Every nirs4all-methods binding dispatches into the same C kernel. Registered comparator/source rows are listed in the benchmark card below.

::::{tab-set}
:class: nirs4all-methods-bindings


:::{tab-item} C ABI · libn4m
:sync: c
:class-label: lang-c

```c
n4m_status_t n4m_pp_wavelet_pca_create( n4m_pp_wavelet_pca_handle_t** out, n4m_pp_wavelet_family_t family, n4m_pp_wavelet_boundary_t mode, int32_t max_level, double n_components);
void n4m_pp_wavelet_pca_destroy( n4m_pp_wavelet_pca_handle_t* handle);
n4m_status_t n4m_pp_wavelet_pca_fit( n4m_pp_wavelet_pca_handle_t* handle, n4m_matrix_view_t X);
n4m_status_t n4m_pp_wavelet_pca_is_fitted( const n4m_pp_wavelet_pca_handle_t* handle, int* out_fitted);
n4m_status_t n4m_pp_wavelet_pca_output_cols( const n4m_pp_wavelet_pca_handle_t* handle, int64_t* out_cols);
n4m_status_t n4m_pp_wavelet_pca_transform( const n4m_pp_wavelet_pca_handle_t* handle, n4m_matrix_view_t X, n4m_matrix_view_t out);
```

:::

:::{tab-item} Python ABI · n4m.python
:sync: python-abi
:class-label: lang-python

```python
from n4m import python as n4m

Xt = n4m.wavelet_pca(X)
```

:::

:::{tab-item} Python sklearn · n4m.sklearn
:sync: python-sklearn
:class-label: lang-python

```python
from n4m.sklearn import WaveletPCA

op = WaveletPCA(family='haar', mode='periodization', max_level=2, n_components=5.0)
Xt = op.fit_transform(X)
```

:::

::::


**Benchmark Comparators And Sources** ◆

:::{card}
:class-card: external-refs

- ◆ **`ref.pywavelets`** (Python · canonical) — `PyWavelets.wavedec(haar, periodization)+sklearn.PCA(full)` · PyWavelets 1.9.0
- ℹ **`ref.nirs4all`** (Python · context) — `nirs4all.WaveletPCA(per-level contract)` · nirs4all@cd731a23+dirty — nirs4all fits per-level PCA blocks; n4m gates the flattened-DWT PCA contract with PyWavelets+sklearn
:::

### Validation contract

- Operation: `cross_binding_callable` · comparator: `sign_invariant_columns` · tolerance: `rtol=1e-05`, `atol=1e-08` · quality: **contract-specific**
- Default validation dataset: `100×50` · seed `20260556`
- Suites: smoke `3` cells; benchmark `11` cells · Used by component projection methods.
- Metrics: `max_abs_diff`, `rel_l2_diff`, `rms_diff`, `shape_equal`
- Truth sources: cross-binding references declared directly in `benchmarks/cross_binding/orchestrator.py`.

| Backend | Library | Gate | Comparator | Note |
|---------|---------|------|------------|------|
| `ref.pywavelets` | `PyWavelets.wavedec(haar, periodization)+sklearn.PCA(full)` | Python / parity | `sign_invariant_columns` |  |
| `ref.nirs4all` | `nirs4all.WaveletPCA(per-level contract)` | Python / context | `sign_invariant_columns` | nirs4all fits per-level PCA blocks; n4m gates the flattened-DWT PCA contract with PyWavelets+sklearn |

### Benchmarks
Median wall-clock per cell from [`docs/_static/bench-data.json`](../benchmarks/overview.md). Divergence is the worst finite value over the visible sizes for each backend, preferring reference max-abs difference and falling back to binding max-abs difference when no reference comparison is recorded. Rows without a recorded comparison show `—`; the fastest backend per column is marked 🏆.
::::{tab-set}
:class: parity-tabs

:::{tab-item} 1 thread
:sync: threads-1

<div class="parity-table-wrap">
<table class="docutils parity-grouped">
<thead><tr><th>Backend</th><th>Divergence</th><th>100×50</th></tr></thead>
<tbody class="lang-band lang-cpp"><tr class="lang-band-row" data-lang="cpp"><th colspan="3" scope="rowgroup"><span class="lang-band-dot"></span>C++ native · libn4m</th></tr>
<tr class="bk-row"><td class="bk-name"><code>N4M.cpp</code></td><td class="parity parity-divergence parity-exact" title="worst reference max abs diff over visible sizes">2.2e-15</td><td class="ms">1.512 ms</td></tr>
</tbody>
<tbody class="lang-band lang-python"><tr class="lang-band-row" data-lang="python"><th colspan="3" scope="rowgroup"><span class="lang-band-dot"></span>Python · nirs4all-methods</th></tr>
<tr class="bk-row"><td class="bk-name"><code>N4M.python</code></td><td class="parity parity-divergence parity-exact" title="worst reference max abs diff over visible sizes">2.2e-15</td><td class="ms">1.506 ms</td></tr>
<tr class="bk-row"><td class="bk-name"><code>N4M.sklearn</code></td><td class="parity parity-divergence parity-exact" title="worst reference max abs diff over visible sizes">2.2e-15</td><td class="ms">1.509 ms</td></tr>
</tbody>
<tbody class="lang-band lang-python"><tr class="lang-band-row" data-lang="python"><th colspan="3" scope="rowgroup"><span class="lang-band-dot"></span>Python · external</th></tr>
<tr class="bk-row truth-source-relaxed"><td class="bk-name"><span class="truth-mark" title="Registry parity reference (Python): nirs4all.WaveletPCA(per-level contract) · nirs4all@cd731a23+dirty — context">◆</span><code>ref.nirs4all</code></td><td class="parity parity-divergence parity-context" title="no divergence recorded">—</td><td class="ms">25.720 ms</td></tr>
<tr class="bk-row truth-source-strict"><td class="bk-name"><span class="truth-mark" title="Registry parity reference (Python): PyWavelets.wavedec(haar, periodization)+sklearn.PCA(full) · PyWavelets 1.9.0 — canonical">◆</span><code>ref.pywavelets</code></td><td class="parity parity-divergence parity-exact" title="worst reference max abs diff over visible sizes">0</td><td class="ms ms-best">🏆 1.343 ms</td></tr>
</tbody>
</table>
</div>

:::
::::

---

_See also_: [benchmark overview](../benchmarks/overview.md) · [methods index](index.md) · [interactive dashboard](../landing/dashboard.md)
