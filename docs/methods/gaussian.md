# `gaussian` — Gaussian

_Group_: **Preprocessing** · _Registry tolerance_: `rtol=1e-5`, `atol=1e-8` · _Source_: [`docs/algorithms/gaussian.md`](../algorithms/gaussian.md)

## Description

From the `chemometrics4all.Gaussian` Python wrapper docstring:

> SciPy-compatible 1-D Gaussian filter along the wavelength axis.

### Parameters

| Name | Type | Default | Notes |
|------|------|---------|-------|
| `sigma` | `float` | `1.0` | Gaussian standard deviation or kernel scale, depending on method. |
| `order` | `int` | `0` |  |
| `mode` | `str` | `'reflect'` | Boundary handling mode. |
| `cval` | `float` | `0.0` |  |
| `truncate` | `float` | `4.0` |  |

## Explanations

### Bibliographic source

`scipy.ndimage.gaussian_filter1d`
([source](https://github.com/scipy/scipy/blob/v1.17.1/scipy/ndimage/_filters.py)).

### Mathematical principle

1-D Gaussian filter / Gaussian derivative along the wavelength axis,
matching `scipy.ndimage.gaussian_filter1d(X, sigma, order, axis=1, mode,
cval, truncate)`.

The kernel of half-width $\ell = \lfloor \text{truncate} \cdot \sigma + 0.5 \rfloor$
is constructed as the natural Gaussian times an analytic Hermite
polynomial (matching `_gaussian_kernel1d` in `scipy/ndimage/_filters.py`):

$$
\phi(x) = \frac{1}{Z}\exp\!\left(-\frac{x^2}{2 \sigma^2}\right),
\quad
g(x) = q(x) \cdot \phi(x).
$$

For `order = 0`, $q = 1$.  For higher orders the Hermite recursion runs
`order` steps of the matrix $(D + P)$ on the coefficients of $q$, where

$$
D_{i, i+1} = i + 1,\quad P_{i+1, i} = -1 / \sigma^2,
$$

i.e., $q$ encodes the derivative of $\log\phi$ scaled per the Leibniz
rule.

After the kernel is built it is reversed and applied via SciPy's
cross-correlation convention:

$$
\hat{X}_{i, j} = \sum_{k=0}^{2\ell} \mathrm{rev}(g)_k \cdot X_{i,\; j + k - \ell}.
$$

### Implementation

- `_create` validates each parameter; returns `C4A_ERR_INVALID_ARGUMENT`
  on $\sigma \leq 0$, $\text{order} < 0$, $\text{truncate} < 0$, or an
  unknown mode.
- Tolerance vs scipy: $10^{-9}$ absolute / $10^{-10}$ relative.  The gap
  is structural to the small differences in how scipy's `correlate1d`
  accumulates the kernel sum (vectorised) vs our scalar loop.

C ABI entry points used by the language bindings:

```c
c4a_status_t c4a_pp_gaussian_create( c4a_pp_gaussian_handle_t** out, double sigma, int32_t order, c4a_pp_gaussian_mode_t mode, double cval, double truncate);
void c4a_pp_gaussian_destroy(c4a_pp_gaussian_handle_t* handle);
c4a_status_t c4a_pp_gaussian_transform( const c4a_pp_gaussian_handle_t* handle, c4a_matrix_view_t X, c4a_matrix_view_t out);
```

Benchmark comparator backends are registered in the matrix and stored as reproducible snapshots when they define the canonical contract.

### Implementations

| Layer | Entry point | Language | Contract |
|-------|-------------|----------|----------|
| C ABI | `c4a_pp_gaussian` | C/C++ | Stable libc4a entry point family. |
| Python | `chemometrics4all.python.gaussian` | Python | ABI-close function backed by ctypes. |
| Python sklearn | `chemometrics4all.sklearn.Gaussian` | Python | scikit-learn-compatible estimator backed by ctypes. |
| R | `gaussian(X, sigma = 1.0, order = 0L, mode = "reflect", cval = 0.0, truncate = 4.0)` | R | Public package wrapper around the C ABI. |
| ref.scipy | `scipy.ndimage.gaussian_filter1d` | Python | canonical/comparator |
| ref.nirs4all | `nirs4all.Gaussian` | Python | canonical/comparator |

### Usage

Every chemometrics4all binding dispatches into the same C kernel. Registered comparator/source rows are listed in the benchmark card below.

::::{tab-set}
:class: chemometrics4all-bindings


:::{tab-item} C ABI · libc4a
:sync: c
:class-label: lang-c

```c
c4a_status_t c4a_pp_gaussian_create( c4a_pp_gaussian_handle_t** out, double sigma, int32_t order, c4a_pp_gaussian_mode_t mode, double cval, double truncate);
void c4a_pp_gaussian_destroy(c4a_pp_gaussian_handle_t* handle);
c4a_status_t c4a_pp_gaussian_transform( const c4a_pp_gaussian_handle_t* handle, c4a_matrix_view_t X, c4a_matrix_view_t out);
```

:::

:::{tab-item} Python ABI · chemometrics4all.python
:sync: python-abi
:class-label: lang-python

```python
from chemometrics4all import python as c4a

Xt = c4a.gaussian(X)
```

:::

:::{tab-item} Python sklearn · chemometrics4all.sklearn
:sync: python-sklearn
:class-label: lang-python

```python
from chemometrics4all.sklearn import Gaussian

op = Gaussian(sigma=1.0, order=0, mode='reflect', cval=0.0, truncate=4.0)
Xt = op.fit_transform(X)
```

:::

:::{tab-item} R · chemometrics4all
:sync: r
:class-label: lang-r

```r
library(chemometrics4all)
res <- gaussian(X, sigma = 1.0)
```

:::

::::


**Benchmark Comparators And Sources** ◆

:::{card}
:class-card: external-refs

- ◆ **`ref.scipy`** (Python · canonical) — `scipy.ndimage.gaussian_filter1d` · scipy 1.17.1
- ◆ **`ref.nirs4all`** (Python · comparator) — `nirs4all.Gaussian` · nirs4all@cd731a23+dirty
:::

### Validation contract

- Operation: `cross_binding_callable` · comparator: `default_allclose` · tolerance: `rtol=1e-05`, `atol=1e-08` · quality: **strict**
- Default validation dataset: `100×50` · seed `20260556`
- Suites: smoke `3` cells; benchmark `11` cells · Default C/Python/reference parity comparator.
- Metrics: `max_abs_diff`, `rel_l2_diff`, `rms_diff`, `shape_equal`
- Truth sources: cross-binding references declared directly in `benchmarks/cross_binding/orchestrator.py`.

| Backend | Library | Gate | Comparator | Note |
|---------|---------|------|------------|------|
| `ref.scipy` | `scipy.ndimage.gaussian_filter1d` | Python / parity | `default_allclose` |  |
| `ref.nirs4all` | `nirs4all.Gaussian` | Python / parity | `default_allclose` |  |

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
<tr class="bk-row"><td class="bk-name"><code>C4A.cpp</code></td><td class="parity parity-divergence parity-exact" title="worst reference max abs diff over visible sizes">3.3e-16</td><td class="ms ms-best">🏆 0.009 ms</td><td class="ms ms-best">🏆 0.040 ms</td><td class="ms ms-best">🏆 0.177 ms</td></tr>
</tbody>
<tbody class="lang-band lang-python"><tr class="lang-band-row" data-lang="python"><th colspan="5" scope="rowgroup"><span class="lang-band-dot"></span>Python · chemometrics4all</th></tr>
<tr class="bk-row"><td class="bk-name"><code>C4A.python</code></td><td class="parity parity-divergence parity-exact" title="worst reference max abs diff over visible sizes">3.3e-16</td><td class="ms">0.015 ms</td><td class="ms">0.047 ms</td><td class="ms">0.206 ms</td></tr>
<tr class="bk-row"><td class="bk-name"><code>C4A.sklearn</code></td><td class="parity parity-divergence parity-exact" title="worst reference max abs diff over visible sizes">3.3e-16</td><td class="ms">0.017 ms</td><td class="ms">0.060 ms</td><td class="ms">0.272 ms</td></tr>
</tbody>
<tbody class="lang-band lang-r"><tr class="lang-band-row" data-lang="r"><th colspan="5" scope="rowgroup"><span class="lang-band-dot"></span>R · chemometrics4all</th></tr>
<tr class="bk-row"><td class="bk-name"><code>C4A.R</code></td><td class="parity parity-divergence parity-exact" title="worst reference max abs diff over visible sizes">7.8e-16</td><td class="ms">0.035 ms</td><td class="ms">0.225 ms</td><td class="ms">1.555 ms</td></tr>
</tbody>
<tbody class="lang-band lang-python"><tr class="lang-band-row" data-lang="python"><th colspan="5" scope="rowgroup"><span class="lang-band-dot"></span>Python · external</th></tr>
<tr class="bk-row truth-source-relaxed"><td class="bk-name"><span class="truth-mark" title="Registry parity reference (Python): nirs4all.Gaussian · nirs4all@cd731a23+dirty — comparator">◆</span><code>ref.nirs4all</code></td><td class="parity parity-divergence parity-exact" title="worst reference max abs diff over visible sizes">0</td><td class="ms">0.023 ms</td><td class="ms">0.150 ms</td><td class="ms">0.875 ms</td></tr>
<tr class="bk-row truth-source-strict"><td class="bk-name"><span class="truth-mark" title="Registry parity reference (Python): scipy.ndimage.gaussian_filter1d · scipy 1.17.1 — canonical">◆</span><code>ref.scipy</code></td><td class="parity parity-divergence parity-exact" title="worst reference max abs diff over visible sizes">0</td><td class="ms">0.020 ms</td><td class="ms">0.121 ms</td><td class="ms">0.690 ms</td></tr>
</tbody>
</table>
</div>

:::
::::

---

_See also_: [benchmark overview](../benchmarks/overview.md) · [methods index](index.md) · [interactive dashboard](../landing/dashboard.md)
