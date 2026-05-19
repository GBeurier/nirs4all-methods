# Gaussian (1-D Gaussian filter)

## Algorithm

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

## Boundary modes

Five modes mirror `scipy.ndimage`:

| Mode | Semantics |
|------|-----------|
| `reflect`  | Edge-repeating reflection (default). |
| `constant` | Pad with `cval`. |
| `nearest`  | Replicate the edge sample. |
| `mirror`   | Non-edge-repeating reflection. |
| `wrap`     | Cyclic extension. |

## Parameters

| Parameter | Constraint | Meaning |
|-----------|------------|---------|
| `sigma`    | $> 0$    | Standard deviation. |
| `order`    | $\geq 0$ | Derivative order. |
| `mode`     | enum     | Boundary handling. |
| `cval`     | any      | Fill value for mode = `constant`. |
| `truncate` | $\geq 0$ | Kernel half-width in standard deviations. |

## Numerical contract

- `_create` validates each parameter; returns `C4A_ERR_INVALID_ARGUMENT`
  on $\sigma \leq 0$, $\text{order} < 0$, $\text{truncate} < 0$, or an
  unknown mode.
- Tolerance vs scipy: $10^{-9}$ absolute / $10^{-10}$ relative.  The gap
  is structural to the small differences in how scipy's `correlate1d`
  accumulates the kernel sum (vectorised) vs our scalar loop.

## Reference

`scipy.ndimage.gaussian_filter1d`
([source](https://github.com/scipy/scipy/blob/v1.17.1/scipy/ndimage/_filters.py)).
