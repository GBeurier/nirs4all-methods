# `msc` — MSC

_Group_: **Preprocessing** · _Registry tolerance_: `rtol=1e-5`, `atol=1e-8` · _Source_: [`docs/algorithms/msc.md`](../algorithms/msc.md)

## Description

From the `chemometrics4all.MSC` Python wrapper docstring:

> Multiplicative Scatter Correction.

<details><summary>Full wrapper docstring</summary>

```text
Multiplicative Scatter Correction.

Fit learns the mean reference spectrum from the training matrix. Transform
regresses each row against that reference and applies the conventional
row-wise MSC correction used by prospectr and pls.
```

</details>

### Parameters

This operator takes no constructor parameters. Use `_create()`, then
`_fit(X)`, then `_transform(X)` (and optionally `_inverse_transform(X)` on the
same handle after a transform).

## Explanations

### Bibliographic source

Geladi, P., MacDougall, D., Martens, H. (1985). "Linearization and
Scatter-Correction for Near-Infrared Reflectance Spectra of Meat."
*Applied Spectroscopy* 39 (3), 491-500.

Implementation follows the conventional reference-spectrum MSC contract used by
`prospectr::msc` and `pls::msc`. The local development checkout of `nirs4all`
currently exposes a historical column-regression MSC variant, so it is kept as
performance context rather than a parity gate until that package is updated.

### Mathematical principle

Conventional row-wise scatter correction calibrated against the mean reference
spectrum of the training matrix. Let
$X_\mathrm{fit} \in \mathbb{R}^{n \times p}$ be the training spectra matrix.

**Fit** computes the reference spectrum:

$$
r_j = \frac{1}{n} \sum_{i=1}^{n} X_{\mathrm{fit}, i,j}
$$

and its centered denominator:

$$
\bar{r} = \frac{1}{p} \sum_{j=1}^{p} r_j, \qquad
d_r = \sum_{j=1}^{p} (r_j - \bar{r})^2.
$$

**Transform** regresses each input row $x_i$ against the fitted reference
spectrum:

$$
\bar{x}_i = \frac{1}{p} \sum_{j=1}^{p} x_{i,j}, \qquad
s_i = \frac{\sum_j (x_{i,j} - \bar{x}_i)(r_j - \bar{r})}{d_r},
\qquad
o_i = \bar{x}_i - s_i \bar{r}.
$$

The corrected spectrum is:

$$
X'_{i,j} = \frac{x_{i,j} - o_i}{s_i}.
$$

This is the default contract used by `prospectr::msc` and `pls::msc`: the fit
stores the reference spectrum, while the offset and slope are estimated for
each transformed spectrum.

**Inverse transform** needs the row coefficients from the forward transform:

$$
x_{i,j} = X'_{i,j} \cdot s_i + o_i.
$$

The C ABI stores those row coefficients on the handle during `_transform`, so
`_inverse_transform(_transform(X))` is covered for the same handle. Stateless
wrappers must preserve those coefficients separately; the R wrapper stores them
as attributes on the matrix returned by `msc()`.

### Implementation

* `_fit` requires `rows >= 1` and `cols >= 2`.
* `_fit` returns `C4A_ERR_NUMERICAL_FAILURE` when the reference spectrum has
  zero variance.
* `_transform` returns `C4A_ERR_NUMERICAL_FAILURE` when a row slope is zero or
  non-finite.
* `_inverse_transform` returns `C4A_ERR_SHAPE_MISMATCH` unless the input shape
  matches the previous transform on the same handle.
* The active benchmark gate compares against `prospectr::msc`; observed
  full-matrix differences are at double-precision roundoff.

C ABI entry points used by the language bindings:

```c
c4a_status_t c4a_pp_msc_create(c4a_pp_msc_handle_t** out);
void c4a_pp_msc_destroy(c4a_pp_msc_handle_t* handle);
c4a_status_t c4a_pp_msc_fit(c4a_pp_msc_handle_t* handle, c4a_matrix_view_t X);
c4a_status_t c4a_pp_msc_inverse_transform( const c4a_pp_msc_handle_t* handle, c4a_matrix_view_t X, c4a_matrix_view_t out);
c4a_status_t c4a_pp_msc_is_fitted(const c4a_pp_msc_handle_t* handle, int* out_fitted);
c4a_status_t c4a_pp_msc_transform(c4a_pp_msc_handle_t* handle, c4a_matrix_view_t X, c4a_matrix_view_t out);
```

Benchmark comparator backends are registered in the matrix and stored as reproducible snapshots when they define the canonical contract.

### Implementations

| Layer | Entry point | Language | Contract |
|-------|-------------|----------|----------|
| C ABI | `c4a_pp_msc` | C/C++ | Stable libc4a entry point family. |
| Python | `chemometrics4all.python.msc` | Python | ABI-close function backed by ctypes. |
| Python sklearn | `chemometrics4all.sklearn.MSC` | Python | scikit-learn-compatible estimator backed by ctypes. |
| R | `msc(X, X_fit = X)` | R | Public package wrapper around the C ABI. |
| ref.r.prospectr | `prospectr.msc` | R | canonical/comparator; c4a MSC follows the conventional row-wise reference-spectrum contract used by prospectr |
| ref.nirs4all | `nirs4all.MultiplicativeScatterCorrection(scale=False)` | Python | context only; current local nirs4all checkout uses a historical column-regression MSC variant |

### Usage

Every chemometrics4all binding dispatches into the same C kernel. Registered comparator/source rows are listed in the benchmark card below.

::::{tab-set}
:class: chemometrics4all-bindings


:::{tab-item} C ABI · libc4a
:sync: c
:class-label: lang-c

```c
c4a_status_t c4a_pp_msc_create(c4a_pp_msc_handle_t** out);
void c4a_pp_msc_destroy(c4a_pp_msc_handle_t* handle);
c4a_status_t c4a_pp_msc_fit(c4a_pp_msc_handle_t* handle, c4a_matrix_view_t X);
c4a_status_t c4a_pp_msc_inverse_transform( const c4a_pp_msc_handle_t* handle, c4a_matrix_view_t X, c4a_matrix_view_t out);
c4a_status_t c4a_pp_msc_is_fitted(const c4a_pp_msc_handle_t* handle, int* out_fitted);
c4a_status_t c4a_pp_msc_transform(c4a_pp_msc_handle_t* handle, c4a_matrix_view_t X, c4a_matrix_view_t out);
```

:::

:::{tab-item} Python ABI · chemometrics4all.python
:sync: python-abi
:class-label: lang-python

```python
from chemometrics4all import python as c4a

Xt = c4a.msc(X)
```

:::

:::{tab-item} Python sklearn · chemometrics4all.sklearn
:sync: python-sklearn
:class-label: lang-python

```python
from chemometrics4all.sklearn import MSC

op = MSC()
Xt = op.fit_transform(X)
```

:::

:::{tab-item} R · chemometrics4all
:sync: r
:class-label: lang-r

```r
library(chemometrics4all)
res <- msc(X)
```

:::

::::


**Benchmark Comparators And Sources** ◆

:::{card}
:class-card: external-refs

- ◆ **`ref.r.prospectr`** (R · canonical) — `prospectr.msc` · prospectr 0.2.8 — c4a MSC follows the conventional row-wise reference-spectrum contract used by prospectr
- ℹ **`ref.nirs4all`** (Python · context) — `nirs4all.MultiplicativeScatterCorrection(scale=False)` · nirs4all@cd731a23+dirty — current local nirs4all checkout uses a historical column-regression MSC variant
:::

### Validation contract

- Operation: `fit_transform` · comparator: `default_allclose` · tolerance: `rtol=1e-05`, `atol=1e-08` · quality: **strict**
- Default validation dataset: `64×128` · seed `1234567891`
- Suites: smoke `3` cells; benchmark `11` cells · Default C/Python/reference parity comparator.
- Metrics: `finite_output`, `max_abs_diff`, `shape_equal`
- Truth sources:
  - `nirs4all_msc` — nirs4all.operators.preprocessing.msc (nirs4all, git-pinned by benchmark environment)

| Backend | Library | Gate | Comparator | Note |
|---------|---------|------|------------|------|
| `ref.r.prospectr` | `prospectr.msc` | R / parity | `default_allclose` | c4a MSC follows the conventional row-wise reference-spectrum contract used by prospectr |
| `ref.nirs4all` | `nirs4all.MultiplicativeScatterCorrection(scale=False)` | Python / context | `default_allclose` | current local nirs4all checkout uses a historical column-regression MSC variant |

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
<tr class="bk-row"><td class="bk-name"><code>C4A.cpp</code></td><td class="parity parity-divergence parity-exact" title="worst reference max abs diff over visible sizes">8.0e-15</td><td class="ms ms-best">🏆 0.013 ms</td><td class="ms ms-best">🏆 0.080 ms</td><td class="ms">0.384 ms</td></tr>
</tbody>
<tbody class="lang-band lang-python"><tr class="lang-band-row" data-lang="python"><th colspan="5" scope="rowgroup"><span class="lang-band-dot"></span>Python · chemometrics4all</th></tr>
<tr class="bk-row"><td class="bk-name"><code>C4A.python</code></td><td class="parity parity-divergence parity-exact" title="worst reference max abs diff over visible sizes">8.0e-15</td><td class="ms">0.015 ms</td><td class="ms">0.082 ms</td><td class="ms ms-best">🏆 0.380 ms</td></tr>
<tr class="bk-row"><td class="bk-name"><code>C4A.sklearn</code></td><td class="parity parity-divergence parity-exact" title="worst reference max abs diff over visible sizes">8.0e-15</td><td class="ms">0.015 ms</td><td class="ms">0.080 ms</td><td class="ms">0.382 ms</td></tr>
</tbody>
<tbody class="lang-band lang-r"><tr class="lang-band-row" data-lang="r"><th colspan="5" scope="rowgroup"><span class="lang-band-dot"></span>R · chemometrics4all</th></tr>
<tr class="bk-row"><td class="bk-name"><code>C4A.R</code></td><td class="parity parity-divergence parity-exact" title="worst reference max abs diff over visible sizes">8.0e-15</td><td class="ms">0.101 ms</td><td class="ms">0.699 ms</td><td class="ms">4.500 ms</td></tr>
</tbody>
<tbody class="lang-band lang-python"><tr class="lang-band-row" data-lang="python"><th colspan="5" scope="rowgroup"><span class="lang-band-dot"></span>Python · external</th></tr>
<tr class="bk-row truth-source-relaxed"><td class="bk-name"><span class="truth-mark" title="Registry parity reference (Python): nirs4all.MultiplicativeScatterCorrection(scale=False) · nirs4all@cd731a23+dirty — context">◆</span><code>ref.nirs4all</code></td><td class="parity parity-divergence parity-context" title="no divergence recorded">—</td><td class="ms">0.879 ms</td><td class="ms">8.318 ms</td><td class="ms">42.063 ms</td></tr>
</tbody>
<tbody class="lang-band lang-r"><tr class="lang-band-row" data-lang="r"><th colspan="5" scope="rowgroup"><span class="lang-band-dot"></span>R · external</th></tr>
<tr class="bk-row truth-source-strict"><td class="bk-name"><span class="truth-mark" title="Registry parity reference (R): prospectr.msc · prospectr 0.2.8 — canonical">◆</span><code>ref.r.prospectr</code></td><td class="parity parity-divergence parity-exact" title="worst reference max abs diff over visible sizes">0</td><td class="ms">0.093 ms</td><td class="ms">0.469 ms</td><td class="ms">2.906 ms</td></tr>
</tbody>
</table>
</div>

:::
::::

---

_See also_: [benchmark overview](../benchmarks/overview.md) · [methods index](index.md) · [interactive dashboard](../landing/dashboard.md)
