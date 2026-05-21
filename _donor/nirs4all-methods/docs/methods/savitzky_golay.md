# `savitzky_golay` — Savitzky-Golay

_Group_: **Preprocessing** · _Registry tolerance_: `rtol=1e-5`, `atol=1e-8` · _Source_: [`docs/algorithms/savitzky_golay.md`](../algorithms/savitzky_golay.md)

## Description

From the `n4m.SavitzkyGolay` Python wrapper docstring:

> scipy.signal.savgol_filter parity.

### Parameters

| Name | Type | Default | Notes |
|------|------|---------|-------|
| `window_length` | `int` | `5` | Odd Savitzky-Golay window length. |
| `polyorder` | `int` | `2` | Polynomial order. |
| `deriv` | `int` | `0` | Derivative order. |
| `delta` | `float` | `1.0` | Sample spacing along the wavelength axis. |
| `mode` | `str` | `'mirror'` | Boundary handling mode. |
| `cval` | `float` | `0.0` |  |

## Explanations

### Bibliographic source

Savitzky, A. & Golay, M. J. E. (1964).  "Smoothing and Differentiation of
Data by Simplified Least Squares Procedures."  *Analytical Chemistry*
36 (8), 1627-1639.

### Mathematical principle

A 1-D Savitzky-Golay FIR filter applied per row along the wavelength axis,
matching `scipy.signal.savgol_filter` from the current SciPy 1.17.1 parity
pin. Earlier SciPy releases on the same code path produced identical bytes.

The filter is a fixed convolution kernel of length `window_length` that is
the least-squares pseudo-inverse of a Vandermonde basis through positions
$x_i = i - \mathrm{pos}$ (centred filter, $\mathrm{pos} = (\mathrm{window\_length} - 1) / 2$).
Given `polyorder` $p$ and derivative order $d$, scipy builds:

$$
A \in \mathbb{R}^{(p + 1) \times w},\quad A_{k, i} = (x_i^{(\mathrm{rev})})^k,
\qquad
y \in \mathbb{R}^{p + 1},\quad y_k = \begin{cases} d! / \delta^d & k = d \\ 0 & \text{else} \end{cases}
$$

and solves $A c = y$ via `numpy.linalg.lstsq` (SVD).  The resulting $c \in \mathbb{R}^w$
is the convolution kernel.  Our engine produces the mathematically
equivalent Vandermonde-QR pseudo-inverse — `V = QR`, solve $R^T z = y$
then $R w = z$, then $c = V w$ — agreeing with the SciPy SVD path to
within $\sim 10^{-11}$ on the conditioning of typical small Vandermonde
matrices.

After the kernel is built, the transform is a `scipy.ndimage.convolve1d`
along axis=1:

$$
\hat{X}_{i, j} = \sum_{k=0}^{w - 1} c_k \, X_{i,\; j + \mathrm{pos} - k}.
$$

The kernel is the same for every row; the convolution is the only
per-row work at transform time.

### Implementation

- `_create` validates parameters; returns `N4M_ERR_INVALID_ARGUMENT` for
  even windows, $p \geq w$, $d < 0$, $\delta = 0$, or an unknown mode.
- `_transform` requires `cols >= window_length`; otherwise
  `N4M_ERR_INVALID_ARGUMENT`.
- Tolerance vs scipy: $10^{-9}$ absolute / $10^{-10}$ relative.  The gap is
  structural to the QR-vs-SVD lstsq comparison on the Vandermonde matrix.

C ABI entry points used by the language bindings:

```c
n4m_status_t n4m_pp_savgol_create(n4m_pp_savgol_handle_t** out, int32_t window_length, int32_t polyorder, int32_t deriv, double delta, n4m_pp_savgol_mode_t mode, double cval);
void n4m_pp_savgol_destroy(n4m_pp_savgol_handle_t* handle);
n4m_status_t n4m_pp_savgol_transform( const n4m_pp_savgol_handle_t* handle, n4m_matrix_view_t X, n4m_matrix_view_t out);
```

Benchmark comparator backends are registered in the matrix and stored as reproducible snapshots when they define the canonical contract.

### Implementations

| Layer | Entry point | Language | Contract |
|-------|-------------|----------|----------|
| C ABI | `n4m_pp_savgol` | C/C++ | Stable libn4m entry point family. |
| Python | `n4m.python.savitzky_golay` | Python | ABI-close function backed by ctypes. |
| Python sklearn | `n4m.sklearn.SavitzkyGolay` | Python | scikit-learn-compatible estimator backed by ctypes. |
| R | `savitzky_golay(X, window_length, polyorder, deriv = 0L, delta = 1.0, mode = "mirror", cval = 0.0)` | R | Public package wrapper around the C ABI. |
| ref.scipy | `scipy.signal.savgol_filter` | Python | canonical/comparator |
| ref.r.prospectr | `prospectr.savitzkyGolay` | R | canonical/comparator; prospectr drops boundary columns; parity compares the shared valid interior window |
| ref.nirs4all | `nirs4all.SavitzkyGolay(default edge mode)` | Python | context only; nirs4all does not expose n4m's mirror edge mode; the default scipy edge contract changes boundary samples |

### Usage

Every nirs4all-methods binding dispatches into the same C kernel. Registered comparator/source rows are listed in the benchmark card below.

::::{tab-set}
:class: nirs4all-methods-bindings


:::{tab-item} C ABI · libn4m
:sync: c
:class-label: lang-c

```c
n4m_status_t n4m_pp_savgol_create(n4m_pp_savgol_handle_t** out, int32_t window_length, int32_t polyorder, int32_t deriv, double delta, n4m_pp_savgol_mode_t mode, double cval);
void n4m_pp_savgol_destroy(n4m_pp_savgol_handle_t* handle);
n4m_status_t n4m_pp_savgol_transform( const n4m_pp_savgol_handle_t* handle, n4m_matrix_view_t X, n4m_matrix_view_t out);
```

:::

:::{tab-item} Python ABI · n4m.python
:sync: python-abi
:class-label: lang-python

```python
from n4m import python as n4m

Xt = n4m.savitzky_golay(X)
```

:::

:::{tab-item} Python sklearn · n4m.sklearn
:sync: python-sklearn
:class-label: lang-python

```python
from n4m.sklearn import SavitzkyGolay

op = SavitzkyGolay(window_length=5, polyorder=2, deriv=0, delta=1.0, mode='mirror', cval=0.0)
Xt = op.fit_transform(X)
```

:::

:::{tab-item} R · nirs4all-methods
:sync: r
:class-label: lang-r

```r
library(n4m)
res <- savitzky_golay(X, window_length = 11L, polyorder = 2L, mode = 'mirror')
```

:::

::::


**Benchmark Comparators And Sources** ◆

:::{card}
:class-card: external-refs

- ◆ **`ref.scipy`** (Python · canonical) — `scipy.signal.savgol_filter` · scipy 1.17.1
- ◆ **`ref.r.prospectr`** (R · comparator) — `prospectr.savitzkyGolay` · prospectr 0.2.8 — prospectr drops boundary columns; parity compares the shared valid interior window
- ℹ **`ref.nirs4all`** (Python · context) — `nirs4all.SavitzkyGolay(default edge mode)` · nirs4all@cd731a23+dirty — nirs4all does not expose n4m's mirror edge mode; the default scipy edge contract changes boundary samples
:::

### Validation contract

- Operation: `transform` · comparator: `default_allclose` · tolerance: `rtol=1e-05`, `atol=1e-08` · quality: **strict**
- Default validation dataset: `64×129` · seed `1234567892`
- Suites: smoke `3` cells; benchmark `11` cells · Default C/Python/reference parity comparator.
- Metrics: `finite_output`, `max_abs_diff`, `shape_equal`
- Truth sources:
  - `nirs4all_savitzky_golay` — nirs4all.operators.preprocessing.savitzky_golay (nirs4all, git-pinned by benchmark environment)
  - `scipy_savgol_filter` — scipy.signal.savgol_filter (scipy, 1.17.1)

| Backend | Library | Gate | Comparator | Note |
|---------|---------|------|------------|------|
| `ref.scipy` | `scipy.signal.savgol_filter` | Python / parity | `default_allclose` |  |
| `ref.r.prospectr` | `prospectr.savitzkyGolay` | R / parity | `savgol_valid_window` | prospectr drops boundary columns; parity compares the shared valid interior window |
| `ref.nirs4all` | `nirs4all.SavitzkyGolay(default edge mode)` | Python / context | `default_allclose` | nirs4all does not expose n4m's mirror edge mode; the default scipy edge contract changes boundary samples |

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
<tr class="bk-row"><td class="bk-name"><code>N4M.cpp</code></td><td class="parity parity-divergence parity-exact" title="worst reference max abs diff over visible sizes">2.3e-15</td><td class="ms ms-best">🏆 0.013 ms</td><td class="ms ms-best">🏆 0.051 ms</td><td class="ms ms-best">🏆 0.218 ms</td></tr>
</tbody>
<tbody class="lang-band lang-python"><tr class="lang-band-row" data-lang="python"><th colspan="5" scope="rowgroup"><span class="lang-band-dot"></span>Python · nirs4all-methods</th></tr>
<tr class="bk-row"><td class="bk-name"><code>N4M.python</code></td><td class="parity parity-divergence parity-exact" title="worst reference max abs diff over visible sizes">2.3e-15</td><td class="ms">0.021 ms</td><td class="ms">0.060 ms</td><td class="ms">0.258 ms</td></tr>
<tr class="bk-row"><td class="bk-name"><code>N4M.sklearn</code></td><td class="parity parity-divergence parity-exact" title="worst reference max abs diff over visible sizes">2.3e-15</td><td class="ms">0.022 ms</td><td class="ms">0.067 ms</td><td class="ms">0.265 ms</td></tr>
</tbody>
<tbody class="lang-band lang-r"><tr class="lang-band-row" data-lang="r"><th colspan="5" scope="rowgroup"><span class="lang-band-dot"></span>R · nirs4all-methods</th></tr>
<tr class="bk-row"><td class="bk-name"><code>N4M.R</code></td><td class="parity parity-divergence parity-exact" title="worst reference max abs diff over visible sizes">2.7e-15</td><td class="ms">0.055 ms</td><td class="ms">0.287 ms</td><td class="ms">1.719 ms</td></tr>
</tbody>
<tbody class="lang-band lang-python"><tr class="lang-band-row" data-lang="python"><th colspan="5" scope="rowgroup"><span class="lang-band-dot"></span>Python · external</th></tr>
<tr class="bk-row truth-source-relaxed"><td class="bk-name"><span class="truth-mark" title="Registry parity reference (Python): nirs4all.SavitzkyGolay(default edge mode) · nirs4all@cd731a23+dirty — context">◆</span><code>ref.nirs4all</code></td><td class="parity parity-divergence parity-context" title="no divergence recorded">—</td><td class="ms">0.266 ms</td><td class="ms">0.449 ms</td><td class="ms">1.215 ms</td></tr>
<tr class="bk-row truth-source-strict"><td class="bk-name"><span class="truth-mark" title="Registry parity reference (Python): scipy.signal.savgol_filter · scipy 1.17.1 — canonical">◆</span><code>ref.scipy</code></td><td class="parity parity-divergence parity-exact" title="worst reference max abs diff over visible sizes">0</td><td class="ms">0.068 ms</td><td class="ms">0.187 ms</td><td class="ms">0.718 ms</td></tr>
</tbody>
<tbody class="lang-band lang-r"><tr class="lang-band-row" data-lang="r"><th colspan="5" scope="rowgroup"><span class="lang-band-dot"></span>R · external</th></tr>
<tr class="bk-row truth-source-relaxed"><td class="bk-name"><span class="truth-mark" title="Registry parity reference (R): prospectr.savitzkyGolay · prospectr 0.2.8 — comparator">◆</span><code>ref.r.prospectr</code></td><td class="parity parity-divergence parity-exact" title="worst reference max abs diff over visible sizes">3.0e-15</td><td class="ms">0.106 ms</td><td class="ms">0.938 ms</td><td class="ms">5.344 ms</td></tr>
</tbody>
</table>
</div>

:::
::::

---

_See also_: [benchmark overview](../benchmarks/overview.md) · [methods index](index.md) · [interactive dashboard](../landing/dashboard.md)
