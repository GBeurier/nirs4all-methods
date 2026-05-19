# `fck_static` — FCK static

_Group_: **Feature extraction** · _Registry tolerance_: `rtol=1e-5`, `atol=1e-8` · _Source_: [`docs/algorithms/fck_static.md`](../algorithms/fck_static.md)

## Description

From the `chemometrics4all.FCKStaticTransformer` Python wrapper docstring:

> Static fractional convolutional kernel bank transformer.

### Parameters

| Name | Type | Default | Notes |
|------|------|---------|-------|
| `kernel_size` | `int` | `required` | Convolution kernel width. |
| `alphas` | `Sequence[float] \| None` | `None` |  |
| `sigmas` | `Sequence[float] \| None` | `None` |  |
| `filter_orders` | `Sequence[float] \| None` | `None` | Fractional derivative orders in the kernel bank. |
| `filter_scales` | `Sequence[float] \| None` | `None` | Scale factors in the kernel bank. |

## Explanations

### Bibliographic source

- `nirs4all.operators.transforms.fck_static.FCKStaticTransformer`
  ([source](https://github.com/GBeurier/nirs4all/blob/main/nirs4all/operators/transforms/fck_static.py)) —
  Phase-21 ABI follows the simpler $(\alpha, \sigma, K)$ interface of
  `fractional_kernel_1d` rather than the Python transformer's
  $(\alpha, \mathrm{scale}, K, \sigma)$ four-parameter form.
- `nirs4all.operators.models.sklearn.fckpls.fractional_kernel_1d`
  ([source](https://github.com/GBeurier/nirs4all/blob/main/nirs4all/operators/models/sklearn/fckpls.py)) —
  canonical spatial-domain kernel builder mirrored by
  `c4a_fck_kernel_1d`.

### Mathematical principle

A small, fixed bank of fractional convolutional kernels is applied as 1-D
convolutions across the wavelength axis. The bank is the Cartesian product
of a list of fractional orders ($\alpha$) and a list of Gaussian envelope
widths ($\sigma$), with $\alpha$ varying slowest:

$$
h_\ell = \mathrm{kernel}(\alpha_a, \sigma_s, K),
\quad \ell = a \cdot |\sigma| + s,
\quad a \in [0, |\alpha|), \; s \in [0, |\sigma|).
$$

For an input matrix $X \in \mathbb{R}^{n \times p}$, the output of the
transformer is the horizontal concatenation of the $L = |\alpha| \cdot
|\sigma|$ convolved bands:

$$
\Phi(X)_{i, l, j} = \sum_{k=0}^{K-1} \mathrm{rev}(h_l)_k \cdot
                     X_{i,\; \mathrm{idx}(j + k - \ell_w)},
\quad \ell_w = \lfloor (K - 1) / 2 \rfloor,
$$

flattened to shape $(n, L \cdot p)$ in $(i, l, j)$-major order, where
$\mathrm{idx}(\cdot)$ applies the reflect padding rule on out-of-range
source indices. The kernel is reversed before the cross-correlation sum
to match `scipy.ndimage.convolve1d(X, h, axis=1, mode='reflect')` exactly.

### Kernel construction

Each kernel $h_{\alpha, \sigma}$ of odd length $K$ is built directly in
the spatial domain, matching
`nirs4all.operators.models.sklearn.fckpls.fractional_kernel_1d`:

$$
\mathrm{idx}_i = i - \tfrac{K - 1}{2}, \quad i \in [0, K),
$$

$$
g_i = \exp\!\left(-\tfrac{1}{2} \left(\tfrac{\mathrm{idx}_i}{\max(\sigma, 10^{-6})}\right)^2\right).
$$

For $\alpha < 10^{-10}$ the kernel is the pure Gaussian envelope
$h = g$. Otherwise a signed fractional-power modulation is applied:

$$
h_i = g_i \cdot \mathrm{sign}(\mathrm{idx}_i) \cdot |\mathrm{idx}_i|^\alpha,
$$

with $h_i$ forced to zero when $|\mathrm{idx}_i| < 10^{-10}$ (centre point).
When $\alpha > 0.1$ the kernel is zero-meaned to ensure derivative-like
behaviour:

$$
h \leftarrow h - \mathrm{mean}(h).
$$

Finally the kernel is L1-normalised:

$$
h \leftarrow h / \sum_i |h_i|
$$

provided $\sum_i |h_i| > 10^{-12}$ (the guard is the same one the Python
reference uses; below that threshold the un-normalised kernel is kept).

### Frequency-domain interpretation

Although the implementation works entirely in the spatial domain, the
kernel approximates an operator whose Fourier response is

$$
H(\omega) \propto |\omega|^\alpha \cdot \exp(-\sigma \omega^2),
$$

so $\alpha \approx 0$ yields a smoother, $\alpha \approx 1$ a
first-derivative-like operator, and $\alpha \approx 2$ a
second-derivative-like operator, with non-integer values giving
fractional-order behaviour.

### Implementation

- `_create` validates each parameter and returns
  `C4A_ERR_INVALID_ARGUMENT` on `kernel_size <= 0`, `n_orders <= 0`,
  `n_scales <= 0`, any non-positive sigma, or a bank size that overflows
  `int32_t`. Returns `C4A_ERR_NULL_POINTER` on NULL `out`,
  `filter_orders`, or `filter_scales`.
- `_transform` returns `C4A_ERR_SHAPE_MISMATCH` if `out.cols != L *
  X.cols` or `out.rows != X.rows`.
- Tolerance vs the canonical Python reference: $10^{-12}$ absolute /
  $10^{-13}$ relative. The operator is pure arithmetic (one Gaussian, one
  `pow`, one L1 reduction per kernel, then a small dot-product per
  output sample) so parity is bit-tight on the same inputs.

C ABI entry points used by the language bindings:

```c
c4a_status_t c4a_pp_fck_static_create( c4a_pp_fck_static_handle_t** out, int32_t kernel_size, const double* filter_orders, int32_t n_orders, const double* filter_scales, int32_t n_scales);
void c4a_pp_fck_static_destroy(c4a_pp_fck_static_handle_t* handle);
c4a_status_t c4a_pp_fck_static_output_cols(int32_t n_kernels, int32_t n_features, int32_t* out);
c4a_status_t c4a_pp_fck_static_transform( const c4a_pp_fck_static_handle_t* handle, c4a_matrix_view_t X, c4a_matrix_view_t out);
```

Reference backends are registered in the benchmark matrix and stored as reproducible snapshots when they define the canonical contract.

### Implementations

| Layer | Entry point | Language | Contract |
|-------|-------------|----------|----------|
| C ABI | `c4a_pp_fck_static` | C/C++ | Stable libc4a entry point family. |
| Python | `chemometrics4all.FCKStaticTransformer` | Python | sklearn-style wrapper backed by ctypes. |
| R | `fck_static(X, kernel_size = 5L, filter_orders = c(0.5, 1.0), filter_scales = c(1.0, 2.0))` | R | Public package wrapper around the C ABI. |
| ref.frozen | `c4a frozen FCKStatic(fckpls kernel)` | Python | canonical/comparator |
| ref.nirs4all | `nirs4all.FCKStaticTransformer(current)` | Python | context only; current nirs4all transformer uses a different scaled kernel bank; c4a gates the frozen fckpls fractional_kernel_1d contract |

### Usage

Every chemometrics4all binding dispatches into the same C kernel. The registry references are listed in the parity card below.

::::{tab-set}
:class: chemometrics4all-bindings


:::{tab-item} C ABI · libc4a
:sync: c
:class-label: lang-c

```c
c4a_status_t c4a_pp_fck_static_create( c4a_pp_fck_static_handle_t** out, int32_t kernel_size, const double* filter_orders, int32_t n_orders, const double* filter_scales, int32_t n_scales);
void c4a_pp_fck_static_destroy(c4a_pp_fck_static_handle_t* handle);
c4a_status_t c4a_pp_fck_static_output_cols(int32_t n_kernels, int32_t n_features, int32_t* out);
c4a_status_t c4a_pp_fck_static_transform( const c4a_pp_fck_static_handle_t* handle, c4a_matrix_view_t X, c4a_matrix_view_t out);
```

:::

:::{tab-item} Python · chemometrics4all
:sync: python
:class-label: lang-python

```python
from chemometrics4all import FCKStaticTransformer

op = FCKStaticTransformer(alphas=None, sigmas=None, filter_orders=None, filter_scales=None)
Xt = op.fit_transform(X)
```

:::

:::{tab-item} R · chemometrics4all
:sync: r
:class-label: lang-r

```r
library(chemometrics4all)
res <- fck_static(X, kernel_size = 5L, filter_orders = c(0.5, 1.0), filter_scales = c(1.0, 2.0))
```

:::

::::


**Registry parity references** 📐

:::{card}
:class-card: external-refs

- 📐 **`ref.frozen`** (Python · canonical) — `c4a frozen FCKStatic(fckpls kernel)` · chemometrics4all frozen reference
- ℹ **`ref.nirs4all`** (Python · context) — `nirs4all.FCKStaticTransformer(current)` · nirs4all@cd731a23+dirty — current nirs4all transformer uses a different scaled kernel bank; c4a gates the frozen fckpls fractional_kernel_1d contract
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
<tr class="bk-row"><td class="bk-name"><code>C++</code></td><td class="parity parity-exact">✓ exact</td><td class="ms ms-best">🏆 0.066 ms</td><td class="ms">0.423 ms</td><td class="ms">2.119 ms</td></tr>
</tbody>
<tbody class="lang-band lang-python"><tr class="lang-band-row" data-lang="python"><th colspan="5" scope="rowgroup"><span class="lang-band-dot"></span>Python · chemometrics4all</th></tr>
<tr class="bk-row"><td class="bk-name"><code>Py</code></td><td class="parity parity-exact">✓ bind</td><td class="ms">0.066 ms</td><td class="ms ms-best">🏆 0.417 ms</td><td class="ms ms-best">🏆 2.119 ms</td></tr>
</tbody>
<tbody class="lang-band lang-r"><tr class="lang-band-row" data-lang="r"><th colspan="5" scope="rowgroup"><span class="lang-band-dot"></span>R · chemometrics4all</th></tr>
<tr class="bk-row"><td class="bk-name"><code>R</code></td><td class="parity parity-exact">✓ bind</td><td class="ms">0.104 ms</td><td class="ms">0.992 ms</td><td class="ms">6.562 ms</td></tr>
</tbody>
<tbody class="lang-band lang-python"><tr class="lang-band-row" data-lang="python"><th colspan="5" scope="rowgroup"><span class="lang-band-dot"></span>Python · external</th></tr>
<tr class="bk-row truth-source-strict"><td class="bk-name"><span class="truth-mark" title="Registry parity reference (Python): c4a frozen FCKStatic(fckpls kernel) · chemometrics4all frozen reference — canonical">📐</span><code>ref.frozen</code></td><td class="parity parity-exact">✓ ref</td><td class="ms">0.098 ms</td><td class="ms">0.529 ms</td><td class="ms">2.621 ms</td></tr>
<tr class="bk-row truth-source-relaxed"><td class="bk-name"><span class="truth-mark" title="Registry parity reference (Python): nirs4all.FCKStaticTransformer(current) · nirs4all@cd731a23+dirty — context">📐</span><code>ref.nirs4all</code></td><td class="parity parity-context">✓ ref</td><td class="ms">0.156 ms</td><td class="ms">0.603 ms</td><td class="ms">3.023 ms</td></tr>
</tbody>
</table>
</div>

:::
::::

---

_See also_: [benchmark overview](../benchmarks/overview.md) · [methods index](index.md) · [interactive dashboard](../landing/dashboard.md)
