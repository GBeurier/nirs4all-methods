# `emsc` — EMSC

_Group_: **Preprocessing** · _Registry tolerance_: `rtol=1e-5`, `atol=1e-8` · _Source_: [`docs/algorithms/emsc.md`](../algorithms/emsc.md)

## Description

From the `n4m.EMSC` Python wrapper docstring:

> Extended Multiplicative Scatter Correction (polynomial).

### Parameters

| Name | Type | Default | Notes |
|------|------|---------|-------|
| `degree` | `int` | `2` |  |

## Explanations

### Bibliographic source

Martens, H., Stark, E. (1991). "Extended Multiplicative Signal Correction
and Spectral Interference Subtraction: New Preprocessing Methods for Near
Infrared Spectroscopy." *Journal of Pharmaceutical and Biomedical Analysis*
9 (8), 625-635.

Implementation matches `nirs4all.operators.transforms.nirs.ExtendedMultiplicativeScatterCorrection`
with `scale=False`.

### Mathematical principle

Per-sample scatter correction with polynomial wavelength terms modelling
physical and chemical interferences. Let $X \in \mathbb{R}^{n \times p}$ be
the training spectra and $d$ the polynomial degree.

**Fit** caches the reference spectrum (per-column mean) and the wavelength
axis:

$$
\mathrm{reference}[j] = \overline{X_{:, j}}, \qquad
\mathrm{wavelengths}[j] = j \in \{0, 1, \ldots, p - 1\}.
$$

The basis matrix is

$$
B = \bigl[\, \mathrm{reference}, \; w^1, \; w^2, \; \ldots, \; w^d \,\bigr]
\in \mathbb{R}^{p \times (d + 1)}.
$$

We factor $B = QR$ once via Householder reflections at fit time. The
$(d+1) \times (d+1)$ triangular factor $R$ and the implicit Householder
vectors are stored on the state.

**Transform** runs per row: for each $x_i \in \mathbb{R}^p$, solve the
least-squares system $B c = x_i$ by applying $Q^\top$ then
back-substituting against $R$. The output is

$$
x'_{i,j} = \frac{x_{i,j} - \sum_{d=1}^{D} c_d \cdot j^d}{c_0}.
$$

No inverse transform: the polynomial subtraction is data-driven per row
and discards the row-specific coefficients.

### Implementation

* The basis uses **raw integer wavelengths** (`np.arange(p)`), matching
  nirs4all. For `degree=3` with $p = 200$, the basis values span
  $[0, 7.9 \times 10^6]$ — the system is moderately ill-conditioned, which
  is why we use Householder QR instead of a Cholesky-on-normal-equations
  shortcut.
* `_fit` requires `rows >= 1` and `cols >= degree + 2`. Returns
  `N4M_ERR_NUMERICAL_FAILURE` when the QR factor's diagonal is zero
  (linearly dependent basis columns).
* `_transform` returns `N4M_ERR_NOT_FITTED` before fit, and
  `N4M_ERR_NUMERICAL_FAILURE` when the per-row reference coefficient
  $c_0 = 0$.
* Tolerance against nirs4all + `np.linalg.lstsq` (LAPACK gelsd): 5e-10
  absolute / 5e-10 relative. The handrolled Householder QR vs gelsd's SVD
  agree to ~8 decimal digits on this basis; the gap is structural to the
  conditioning of $B$, not a bug.

C ABI entry points used by the language bindings:

```c
n4m_status_t n4m_pp_emsc_create(n4m_pp_emsc_handle_t** out, int32_t degree);
void n4m_pp_emsc_destroy(n4m_pp_emsc_handle_t* handle);
n4m_status_t n4m_pp_emsc_fit(n4m_pp_emsc_handle_t* handle, n4m_matrix_view_t X);
n4m_status_t n4m_pp_emsc_is_fitted(const n4m_pp_emsc_handle_t* handle, int* out_fitted);
n4m_status_t n4m_pp_emsc_transform(const n4m_pp_emsc_handle_t* handle, n4m_matrix_view_t X, n4m_matrix_view_t out);
```

Benchmark comparator backends are registered in the matrix and stored as reproducible snapshots when they define the canonical contract.

### Implementations

| Layer | Entry point | Language | Contract |
|-------|-------------|----------|----------|
| C ABI | `n4m_pp_emsc` | C/C++ | Stable libn4m entry point family. |
| Python | `n4m.python.emsc` | Python | ABI-close function backed by ctypes. |
| Python sklearn | `n4m.sklearn.EMSC` | Python | scikit-learn-compatible estimator backed by ctypes. |
| R | `emsc(X, X_fit = X, degree = 1L)` | R | Public package wrapper around the C ABI. |
| ref.nirs4all | `nirs4all.ExtendedMultiplicativeScatterCorrection(scale=False)` | Python | canonical/comparator |

### Usage

Every nirs4all-methods binding dispatches into the same C kernel. Registered comparator/source rows are listed in the benchmark card below.

::::{tab-set}
:class: nirs4all-methods-bindings


:::{tab-item} C ABI · libn4m
:sync: c
:class-label: lang-c

```c
n4m_status_t n4m_pp_emsc_create(n4m_pp_emsc_handle_t** out, int32_t degree);
void n4m_pp_emsc_destroy(n4m_pp_emsc_handle_t* handle);
n4m_status_t n4m_pp_emsc_fit(n4m_pp_emsc_handle_t* handle, n4m_matrix_view_t X);
n4m_status_t n4m_pp_emsc_is_fitted(const n4m_pp_emsc_handle_t* handle, int* out_fitted);
n4m_status_t n4m_pp_emsc_transform(const n4m_pp_emsc_handle_t* handle, n4m_matrix_view_t X, n4m_matrix_view_t out);
```

:::

:::{tab-item} Python ABI · n4m.python
:sync: python-abi
:class-label: lang-python

```python
from n4m import python as n4m

Xt = n4m.emsc(X)
```

:::

:::{tab-item} Python sklearn · n4m.sklearn
:sync: python-sklearn
:class-label: lang-python

```python
from n4m.sklearn import EMSC

op = EMSC(degree=2)
Xt = op.fit_transform(X)
```

:::

:::{tab-item} R · nirs4all-methods
:sync: r
:class-label: lang-r

```r
library(n4m)
res <- emsc(X, degree = 1L)
```

:::

::::


**Benchmark Comparators And Sources** ◆

:::{card}
:class-card: external-refs

- ◆ **`ref.nirs4all`** (Python · canonical) — `nirs4all.ExtendedMultiplicativeScatterCorrection(scale=False)` · nirs4all@cd731a23+dirty
:::

### Validation contract

- Operation: `cross_binding_callable` · comparator: `default_allclose` · tolerance: `rtol=1e-05`, `atol=1e-08` · quality: **strict**
- Default validation dataset: `100×50` · seed `20260556`
- Suites: smoke `3` cells; benchmark `11` cells · Default C/Python/reference parity comparator.
- Metrics: `max_abs_diff`, `rel_l2_diff`, `rms_diff`, `shape_equal`
- Truth sources: cross-binding references declared directly in `benchmarks/cross_binding/orchestrator.py`.

| Backend | Library | Gate | Comparator | Note |
|---------|---------|------|------------|------|
| `ref.nirs4all` | `nirs4all.ExtendedMultiplicativeScatterCorrection(scale=False)` | Python / parity | `default_allclose` |  |

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
<tr class="bk-row"><td class="bk-name"><code>N4M.cpp</code></td><td class="parity parity-divergence parity-exact" title="worst reference max abs diff over visible sizes">3.8e-15</td><td class="ms">0.019 ms</td><td class="ms">0.141 ms</td><td class="ms">0.692 ms</td></tr>
</tbody>
<tbody class="lang-band lang-python"><tr class="lang-band-row" data-lang="python"><th colspan="5" scope="rowgroup"><span class="lang-band-dot"></span>Python · nirs4all-methods</th></tr>
<tr class="bk-row"><td class="bk-name"><code>N4M.python</code></td><td class="parity parity-divergence parity-exact" title="worst reference max abs diff over visible sizes">3.8e-15</td><td class="ms ms-best">🏆 0.018 ms</td><td class="ms ms-best">🏆 0.132 ms</td><td class="ms">0.705 ms</td></tr>
<tr class="bk-row"><td class="bk-name"><code>N4M.sklearn</code></td><td class="parity parity-divergence parity-exact" title="worst reference max abs diff over visible sizes">3.8e-15</td><td class="ms">0.021 ms</td><td class="ms">0.144 ms</td><td class="ms ms-best">🏆 0.677 ms</td></tr>
</tbody>
<tbody class="lang-band lang-r"><tr class="lang-band-row" data-lang="r"><th colspan="5" scope="rowgroup"><span class="lang-band-dot"></span>R · nirs4all-methods</th></tr>
<tr class="bk-row"><td class="bk-name"><code>N4M.R</code></td><td class="parity parity-divergence parity-exact" title="worst reference max abs diff over visible sizes">4.2e-15</td><td class="ms">0.040 ms</td><td class="ms">0.637 ms</td><td class="ms">3.562 ms</td></tr>
</tbody>
<tbody class="lang-band lang-python"><tr class="lang-band-row" data-lang="python"><th colspan="5" scope="rowgroup"><span class="lang-band-dot"></span>Python · external</th></tr>
<tr class="bk-row truth-source-strict"><td class="bk-name"><span class="truth-mark" title="Registry parity reference (Python): nirs4all.ExtendedMultiplicativeScatterCorrection(scale=False) · nirs4all@cd731a23+dirty — canonical">◆</span><code>ref.nirs4all</code></td><td class="parity parity-divergence parity-exact" title="worst reference max abs diff over visible sizes">0</td><td class="ms">1.207 ms</td><td class="ms">1.837 ms</td><td class="ms">4.649 ms</td></tr>
</tbody>
</table>
</div>

:::
::::

---

_See also_: [benchmark overview](../benchmarks/overview.md) · [methods index](index.md) · [interactive dashboard](../landing/dashboard.md)
