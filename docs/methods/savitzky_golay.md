# `savitzky_golay` — Savitzky-Golay

_Group_: **Preprocessing** · _Registry tolerance_: `rtol=1e-5`, `atol=1e-8` · _Source_: [`docs/algorithms/savitzky_golay.md`](../algorithms/savitzky_golay.md)

## Description

From the `chemometrics4all.SavitzkyGolay` Python wrapper docstring:

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

- `_create` validates parameters; returns `C4A_ERR_INVALID_ARGUMENT` for
  even windows, $p \geq w$, $d < 0$, $\delta = 0$, or an unknown mode.
- `_transform` requires `cols >= window_length`; otherwise
  `C4A_ERR_INVALID_ARGUMENT`.
- Tolerance vs scipy: $10^{-9}$ absolute / $10^{-10}$ relative.  The gap is
  structural to the QR-vs-SVD lstsq comparison on the Vandermonde matrix.

C ABI entry points used by the language bindings:

```c
c4a_status_t c4a_pp_savgol_create(c4a_pp_savgol_handle_t** out, int32_t window_length, int32_t polyorder, int32_t deriv, double delta, c4a_pp_savgol_mode_t mode, double cval);
void c4a_pp_savgol_destroy(c4a_pp_savgol_handle_t* handle);
c4a_status_t c4a_pp_savgol_transform( const c4a_pp_savgol_handle_t* handle, c4a_matrix_view_t X, c4a_matrix_view_t out);
```

Reference backends are registered in the benchmark matrix and stored as reproducible snapshots when they define the canonical contract.

### Implementations

| Layer | Entry point | Language | Contract |
|-------|-------------|----------|----------|
| C ABI | `c4a_pp_savgol` | C/C++ | Stable libc4a entry point family. |
| Python | `chemometrics4all.SavitzkyGolay` | Python | sklearn-style wrapper backed by ctypes. |
| R | `savitzky_golay(X, window_length, polyorder, deriv = 0L, delta = 1.0, mode = "mirror", cval = 0.0)` | R | Public package wrapper around the C ABI. |
| ref.scipy | `scipy.signal.savgol_filter` | Python | canonical/comparator |
| ref.nirs4all | `nirs4all.SavitzkyGolay(default edge mode)` | Python | context only; nirs4all does not expose c4a's mirror edge mode; the default scipy edge contract changes boundary samples |

### Usage

Every chemometrics4all binding dispatches into the same C kernel. The registry references are listed in the parity card below.

::::{tab-set}
:class: chemometrics4all-bindings


:::{tab-item} C ABI · libc4a
:sync: c
:class-label: lang-c

```c
c4a_status_t c4a_pp_savgol_create(c4a_pp_savgol_handle_t** out, int32_t window_length, int32_t polyorder, int32_t deriv, double delta, c4a_pp_savgol_mode_t mode, double cval);
void c4a_pp_savgol_destroy(c4a_pp_savgol_handle_t* handle);
c4a_status_t c4a_pp_savgol_transform( const c4a_pp_savgol_handle_t* handle, c4a_matrix_view_t X, c4a_matrix_view_t out);
```

:::

:::{tab-item} Python · chemometrics4all
:sync: python
:class-label: lang-python

```python
from chemometrics4all import SavitzkyGolay

op = SavitzkyGolay(window_length=5, polyorder=2, deriv=0, delta=1.0, mode='mirror', cval=0.0)
Xt = op.fit_transform(X)
```

:::

:::{tab-item} R · chemometrics4all
:sync: r
:class-label: lang-r

```r
library(chemometrics4all)
res <- savitzky_golay(X, window_length = 11L, polyorder = 2L, mode = 'mirror')
```

:::

::::


**Registry parity references** 📐

:::{card}
:class-card: external-refs

- 📐 **`ref.scipy`** (Python · canonical) — `scipy.signal.savgol_filter` · scipy 1.17.1
- ℹ **`ref.nirs4all`** (Python · context) — `nirs4all.SavitzkyGolay(default edge mode)` · nirs4all@cd731a23+dirty — nirs4all does not expose c4a's mirror edge mode; the default scipy edge contract changes boundary samples
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
<tr class="bk-row"><td class="bk-name"><code>C++</code></td><td class="parity parity-exact">✓ exact</td><td class="ms ms-best">🏆 0.013 ms</td><td class="ms ms-best">🏆 0.052 ms</td><td class="ms ms-best">🏆 0.227 ms</td></tr>
</tbody>
<tbody class="lang-band lang-python"><tr class="lang-band-row" data-lang="python"><th colspan="5" scope="rowgroup"><span class="lang-band-dot"></span>Python · chemometrics4all</th></tr>
<tr class="bk-row"><td class="bk-name"><code>Py</code></td><td class="parity parity-exact">✓ bind</td><td class="ms">0.019 ms</td><td class="ms">0.058 ms</td><td class="ms">0.252 ms</td></tr>
</tbody>
<tbody class="lang-band lang-r"><tr class="lang-band-row" data-lang="r"><th colspan="5" scope="rowgroup"><span class="lang-band-dot"></span>R · chemometrics4all</th></tr>
<tr class="bk-row"><td class="bk-name"><code>R</code></td><td class="parity parity-exact">✓ bind</td><td class="ms">0.043 ms</td><td class="ms">0.272 ms</td><td class="ms">1.672 ms</td></tr>
</tbody>
<tbody class="lang-band lang-python"><tr class="lang-band-row" data-lang="python"><th colspan="5" scope="rowgroup"><span class="lang-band-dot"></span>Python · external</th></tr>
<tr class="bk-row truth-source-relaxed"><td class="bk-name"><span class="truth-mark" title="Registry parity reference (Python): nirs4all.SavitzkyGolay(default edge mode) · nirs4all@cd731a23+dirty — context">📐</span><code>ref.nirs4all</code></td><td class="parity parity-context">✓ ref</td><td class="ms">0.250 ms</td><td class="ms">0.425 ms</td><td class="ms">1.237 ms</td></tr>
<tr class="bk-row truth-source-strict"><td class="bk-name"><span class="truth-mark" title="Registry parity reference (Python): scipy.signal.savgol_filter · scipy 1.17.1 — canonical">📐</span><code>ref.scipy</code></td><td class="parity parity-exact">✓ ref</td><td class="ms">0.063 ms</td><td class="ms">0.190 ms</td><td class="ms">0.934 ms</td></tr>
</tbody>
</table>
</div>

:::
::::

---

_See also_: [benchmark overview](../benchmarks/overview.md) · [methods index](index.md) · [interactive dashboard](../landing/dashboard.md)
