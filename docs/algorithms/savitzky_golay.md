# SavitzkyGolay

## Algorithm

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

## Boundary modes

Five modes mirror SciPy:

| Mode | Semantics |
|------|-----------|
| `mirror`   | Reflection without repeating the edge.  Default. |
| `constant` | Pad with `cval`. |
| `nearest`  | Replicate the edge sample. |
| `wrap`     | Cyclic extension. |
| `interp`   | Fit a `polyorder` polynomial to the first and last `window_length` samples and evaluate it at the edge positions (matches `_fit_edge` byte-for-byte). |

## Parameters

| Parameter | Constraint | Meaning |
|-----------|------------|---------|
| `window_length` | odd, $\geq 1$ | Number of taps. |
| `polyorder`     | $0 \leq p < w$ | Polynomial degree. |
| `deriv`         | $\geq 0$ | Derivative order ($d > p$ produces the all-zero kernel). |
| `delta`         | $\neq 0$ | Sample spacing for the derivative scaling. |
| `mode`          | enum | Boundary handling (see above). |
| `cval`          | any double | Fill for mode = `constant`. |

## Numerical contract

- `_create` validates parameters; returns `C4A_ERR_INVALID_ARGUMENT` for
  even windows, $p \geq w$, $d < 0$, $\delta = 0$, or an unknown mode.
- `_transform` requires `cols >= window_length`; otherwise
  `C4A_ERR_INVALID_ARGUMENT`.
- Tolerance vs scipy: $10^{-9}$ absolute / $10^{-10}$ relative.  The gap is
  structural to the QR-vs-SVD lstsq comparison on the Vandermonde matrix.

## Reference

Savitzky, A. & Golay, M. J. E. (1964).  "Smoothing and Differentiation of
Data by Simplified Least Squares Procedures."  *Analytical Chemistry*
36 (8), 1627-1639.
