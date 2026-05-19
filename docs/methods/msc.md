# `msc` — MSC

_Group_: **Preprocessing** · _Registry tolerance_: `rtol=1e-5`, `atol=1e-8` · _Source_: [`docs/algorithms/msc.md`](../algorithms/msc.md)

## Description

From the `chemometrics4all.MSC` Python wrapper docstring:

> Multiplicative Scatter Correction.

<details><summary>Full wrapper docstring</summary>

```text
Multiplicative Scatter Correction.

Fit learns the per-column linear coefficients against the per-row mean of
the training matrix.
```

</details>

### Parameters

This operator takes no constructor parameters. Use `_create()`, then
`_fit(X)`, then `_transform(X)` (and optionally `_inverse_transform(X)`).

## Explanations

### Bibliographic source

Geladi, P., MacDougall, D., Martens, H. (1985). "Linearization and
Scatter-Correction for Near-Infrared Reflectance Spectra of Meat."
*Applied Spectroscopy* 39 (3), 491-500.

Implementation matches `nirs4all.operators.transforms.nirs.MultiplicativeScatterCorrection`
with `scale=False`.

### Mathematical principle

Per-column scatter correction calibrated against the per-row mean of the
training matrix. Let $X \in \mathbb{R}^{n \times p}$ be the training spectra
matrix.

**Fit** computes per-row means and per-column linear regression coefficients:

$$
r_i = \frac{1}{p} \sum_{j=1}^{p} X_{i,j}, \qquad
(a_j, b_j) = \mathrm{polyfit}(r, X_{:, j}, \mathrm{deg}=1)
$$

For `deg=1`, `np.polyfit` returns `[slope, intercept]`. We use the
closed-form least-squares formulae directly:

$$
\bar{r} = \tfrac{1}{n} \sum_i r_i, \quad
a_j = \frac{\sum_i (r_i - \bar{r}) X_{i,j}}{\sum_i (r_i - \bar{r})^2}, \quad
b_j = \overline{X_{:, j}} - a_j \bar{r}.
$$

**Transform** divides each column by its slope and subtracts the
intercept:

$$
X'_{i,j} = \frac{X_{i,j} - b_j}{a_j}.
$$

**Inverse transform** reverses the operation:

$$
X_{i,j} = X'_{i,j} \cdot a_j + b_j.
$$

The fitted parameters $(a_j, b_j)$ are independent of the transform input —
calling `_fit(X_train)` then `_transform(X_test)` applies the calibration
learned on $X_\text{train}$ to a fresh $X_\text{test}$ of the same column
count.

### Implementation

* `_fit` requires `rows >= 2` and `cols >= 1`. Returns
  `C4A_ERR_NUMERICAL_FAILURE` when the per-row means form a constant vector
  (zero variance — slope undefined).
* `_transform` and `_inverse_transform` return `C4A_ERR_NOT_FITTED` before
  the first successful `_fit`, and `C4A_ERR_SHAPE_MISMATCH` if the input
  column count differs from the fitted count.
* Tolerance against nirs4all + numpy 1.26.4: 1e-10 absolute / 1e-11
  relative. Our closed-form `(a_j, b_j)` differ from `np.polyfit`'s
  Vandermonde + lstsq path by a couple of ULPs.

C ABI entry points used by the language bindings:

```c
c4a_status_t c4a_pp_msc_create(c4a_pp_msc_handle_t** out);
void c4a_pp_msc_destroy(c4a_pp_msc_handle_t* handle);
c4a_status_t c4a_pp_msc_fit(c4a_pp_msc_handle_t* handle, c4a_matrix_view_t X);
c4a_status_t c4a_pp_msc_inverse_transform( const c4a_pp_msc_handle_t* handle, c4a_matrix_view_t X, c4a_matrix_view_t out);
c4a_status_t c4a_pp_msc_is_fitted(const c4a_pp_msc_handle_t* handle, int* out_fitted);
c4a_status_t c4a_pp_msc_transform(const c4a_pp_msc_handle_t* handle, c4a_matrix_view_t X, c4a_matrix_view_t out);
```

Reference backends are registered in the benchmark matrix and stored as reproducible snapshots when they define the canonical contract.

### Implementations

| Layer | Entry point | Language | Contract |
|-------|-------------|----------|----------|
| C ABI | `c4a_pp_msc` | C/C++ | Stable libc4a entry point family. |
| Python | `chemometrics4all.MSC` | Python | sklearn-style wrapper backed by ctypes. |
| R | `msc(X, X_fit = X)` | R | Public package wrapper around the C ABI. |
| ref.nirs4all | `nirs4all.MultiplicativeScatterCorrection(scale=False)` | Python | canonical/comparator |

### Usage

Every chemometrics4all binding dispatches into the same C kernel. The registry references are listed in the parity card below.

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
c4a_status_t c4a_pp_msc_transform(const c4a_pp_msc_handle_t* handle, c4a_matrix_view_t X, c4a_matrix_view_t out);
```

:::

:::{tab-item} Python · chemometrics4all
:sync: python
:class-label: lang-python

```python
from chemometrics4all import MSC

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


**Registry parity references** 📐

:::{card}
:class-card: external-refs

- 📐 **`ref.nirs4all`** (Python · canonical) — `nirs4all.MultiplicativeScatterCorrection(scale=False)` · nirs4all@cd731a23+dirty
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
<tr class="bk-row"><td class="bk-name"><code>C++</code></td><td class="parity parity-exact">✓ exact</td><td class="ms ms-best">🏆 0.012 ms</td><td class="ms ms-best">🏆 0.069 ms</td><td class="ms">0.345 ms</td></tr>
</tbody>
<tbody class="lang-band lang-python"><tr class="lang-band-row" data-lang="python"><th colspan="5" scope="rowgroup"><span class="lang-band-dot"></span>Python · chemometrics4all</th></tr>
<tr class="bk-row"><td class="bk-name"><code>Py</code></td><td class="parity parity-exact">✓ bind</td><td class="ms">0.012 ms</td><td class="ms">0.073 ms</td><td class="ms ms-best">🏆 0.331 ms</td></tr>
</tbody>
<tbody class="lang-band lang-r"><tr class="lang-band-row" data-lang="r"><th colspan="5" scope="rowgroup"><span class="lang-band-dot"></span>R · chemometrics4all</th></tr>
<tr class="bk-row"><td class="bk-name"><code>R</code></td><td class="parity parity-exact">✓ bind</td><td class="ms">0.088 ms</td><td class="ms">0.449 ms</td><td class="ms">2.719 ms</td></tr>
</tbody>
<tbody class="lang-band lang-python"><tr class="lang-band-row" data-lang="python"><th colspan="5" scope="rowgroup"><span class="lang-band-dot"></span>Python · external</th></tr>
<tr class="bk-row truth-source-strict"><td class="bk-name"><span class="truth-mark" title="Registry parity reference (Python): nirs4all.MultiplicativeScatterCorrection(scale=False) · nirs4all@cd731a23+dirty — canonical">📐</span><code>ref.nirs4all</code></td><td class="parity parity-exact">✓ ref</td><td class="ms">0.872 ms</td><td class="ms">8.687 ms</td><td class="ms">42.919 ms</td></tr>
</tbody>
</table>
</div>

:::
::::

---

_See also_: [benchmark overview](../benchmarks/overview.md) · [methods index](index.md) · [interactive dashboard](../landing/dashboard.md)
