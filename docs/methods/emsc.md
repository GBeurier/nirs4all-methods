# `emsc` — EMSC

_Group_: **Preprocessing** · _Registry tolerance_: `rtol=1e-5`, `atol=1e-8` · _Source_: [`docs/algorithms/emsc.md`](../algorithms/emsc.md)

## Description

From the `chemometrics4all.EMSC` Python wrapper docstring:

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
  `C4A_ERR_NUMERICAL_FAILURE` when the QR factor's diagonal is zero
  (linearly dependent basis columns).
* `_transform` returns `C4A_ERR_NOT_FITTED` before fit, and
  `C4A_ERR_NUMERICAL_FAILURE` when the per-row reference coefficient
  $c_0 = 0$.
* Tolerance against nirs4all + `np.linalg.lstsq` (LAPACK gelsd): 5e-10
  absolute / 5e-10 relative. The handrolled Householder QR vs gelsd's SVD
  agree to ~8 decimal digits on this basis; the gap is structural to the
  conditioning of $B$, not a bug.

C ABI entry points used by the language bindings:

```c
c4a_status_t c4a_pp_emsc_create(c4a_pp_emsc_handle_t** out, int32_t degree);
void c4a_pp_emsc_destroy(c4a_pp_emsc_handle_t* handle);
c4a_status_t c4a_pp_emsc_fit(c4a_pp_emsc_handle_t* handle, c4a_matrix_view_t X);
c4a_status_t c4a_pp_emsc_is_fitted(const c4a_pp_emsc_handle_t* handle, int* out_fitted);
c4a_status_t c4a_pp_emsc_transform(const c4a_pp_emsc_handle_t* handle, c4a_matrix_view_t X, c4a_matrix_view_t out);
```

Reference backends are registered in the benchmark matrix and stored as reproducible snapshots when they define the canonical contract.

### Implementations

| Layer | Entry point | Language | Contract |
|-------|-------------|----------|----------|
| C ABI | `c4a_pp_emsc` | C/C++ | Stable libc4a entry point family. |
| Python | `chemometrics4all.EMSC` | Python | sklearn-style wrapper backed by ctypes. |
| R | `emsc(X, X_fit = X, degree = 1L)` | R | Public package wrapper around the C ABI. |
| ref.nirs4all | `nirs4all.ExtendedMultiplicativeScatterCorrection(scale=False)` | Python | canonical/comparator |

### Usage

Every chemometrics4all binding dispatches into the same C kernel. The registry references are listed in the parity card below.

::::{tab-set}
:class: chemometrics4all-bindings


:::{tab-item} C ABI · libc4a
:sync: c
:class-label: lang-c

```c
c4a_status_t c4a_pp_emsc_create(c4a_pp_emsc_handle_t** out, int32_t degree);
void c4a_pp_emsc_destroy(c4a_pp_emsc_handle_t* handle);
c4a_status_t c4a_pp_emsc_fit(c4a_pp_emsc_handle_t* handle, c4a_matrix_view_t X);
c4a_status_t c4a_pp_emsc_is_fitted(const c4a_pp_emsc_handle_t* handle, int* out_fitted);
c4a_status_t c4a_pp_emsc_transform(const c4a_pp_emsc_handle_t* handle, c4a_matrix_view_t X, c4a_matrix_view_t out);
```

:::

:::{tab-item} Python · chemometrics4all
:sync: python
:class-label: lang-python

```python
from chemometrics4all import EMSC

op = EMSC(degree=2)
Xt = op.fit_transform(X)
```

:::

:::{tab-item} R · chemometrics4all
:sync: r
:class-label: lang-r

```r
library(chemometrics4all)
res <- emsc(X, degree = 1L)
```

:::

::::


**Registry parity references** 📐

:::{card}
:class-card: external-refs

- 📐 **`ref.nirs4all`** (Python · canonical) — `nirs4all.ExtendedMultiplicativeScatterCorrection(scale=False)` · nirs4all@cd731a23+dirty
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
<tr class="bk-row"><td class="bk-name"><code>C4A.cpp</code></td><td class="parity parity-exact">✓ exact</td><td class="ms ms-best">🏆 0.018 ms</td><td class="ms">0.129 ms</td><td class="ms ms-best">🏆 0.695 ms</td></tr>
</tbody>
<tbody class="lang-band lang-python"><tr class="lang-band-row" data-lang="python"><th colspan="5" scope="rowgroup"><span class="lang-band-dot"></span>Python · chemometrics4all</th></tr>
<tr class="bk-row"><td class="bk-name"><code>C4A.sklearn</code></td><td class="parity parity-exact">✓ bind</td><td class="ms">0.019 ms</td><td class="ms ms-best">🏆 0.128 ms</td><td class="ms">0.712 ms</td></tr>
</tbody>
<tbody class="lang-band lang-r"><tr class="lang-band-row" data-lang="r"><th colspan="5" scope="rowgroup"><span class="lang-band-dot"></span>R · chemometrics4all</th></tr>
<tr class="bk-row"><td class="bk-name"><code>C4A.R</code></td><td class="parity parity-exact">✓ bind</td><td class="ms">0.095 ms</td><td class="ms">0.523 ms</td><td class="ms">3.062 ms</td></tr>
</tbody>
<tbody class="lang-band lang-python"><tr class="lang-band-row" data-lang="python"><th colspan="5" scope="rowgroup"><span class="lang-band-dot"></span>Python · external</th></tr>
<tr class="bk-row truth-source-strict"><td class="bk-name"><span class="truth-mark" title="Registry parity reference (Python): nirs4all.ExtendedMultiplicativeScatterCorrection(scale=False) · nirs4all@cd731a23+dirty — canonical">📐</span><code>ref.nirs4all</code></td><td class="parity parity-exact">✓ ref</td><td class="ms">1.157 ms</td><td class="ms">1.741 ms</td><td class="ms">4.463 ms</td></tr>
</tbody>
</table>
</div>

:::
::::

---

_See also_: [benchmark overview](../benchmarks/overview.md) · [methods index](index.md) · [interactive dashboard](../landing/dashboard.md)
