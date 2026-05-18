# FCKStaticTransformer (fractional convolutional kernel bank, static)

## Algorithm

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

## Boundary handling

The convolution uses the Phase 4 padding helper with mode
`C4A_PAD_REFLECT` (no edge-repeat reflection), matching
`scipy.ndimage.convolve1d(..., mode='reflect')`.

## Parameters

| Parameter        | Constraint | Meaning                                        |
|------------------|------------|------------------------------------------------|
| `kernel_size`    | $> 0$      | Length $K$ of each kernel (odd recommended).   |
| `filter_orders`  | non-empty  | Vector of fractional orders ($\alpha$ values). |
| `n_orders`       | $> 0$      | Length of `filter_orders`.                     |
| `filter_scales`  | each $> 0$ | Vector of Gaussian envelope widths ($\sigma$). |
| `n_scales`       | $> 0$      | Length of `filter_scales`.                     |

The bank size is $L = n_{\text{orders}} \cdot n_{\text{scales}}$. The
output column count is $L \cdot p$; the helper
`c4a_pp_fck_static_output_cols(L, p, &out)` computes this and detects
`int32_t` overflow.

## Numerical contract

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

## Reference

- `nirs4all.operators.transforms.fck_static.FCKStaticTransformer`
  ([source](https://github.com/GBeurier/nirs4all/blob/main/nirs4all/operators/transforms/fck_static.py)) —
  Phase-21 ABI follows the simpler $(\alpha, \sigma, K)$ interface of
  `fractional_kernel_1d` rather than the Python transformer's
  $(\alpha, \mathrm{scale}, K, \sigma)$ four-parameter form.
- `nirs4all.operators.models.sklearn.fckpls.fractional_kernel_1d`
  ([source](https://github.com/GBeurier/nirs4all/blob/main/nirs4all/operators/models/sklearn/fckpls.py)) —
  canonical spatial-domain kernel builder mirrored by
  `c4a_fck_kernel_1d`.
