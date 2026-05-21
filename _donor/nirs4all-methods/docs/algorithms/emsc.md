# Extended Multiplicative Scatter Correction (EMSC)

## Algorithm

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

## Parameters

| Parameter | Default | Meaning |
| --- | --- | --- |
| `degree` | — | Polynomial degree $d \geq 1$. |

`_fit` requires `cols >= degree + 2` to keep the basis full-rank.

## Numerical contract

* The basis uses **raw integer wavelengths** (`np.arange(p)`), matching
  nirs4all. For `degree=3` with $p = 200$, the basis values span
  $[0, 7.9 \times 10^6]$ — the system is moderately ill-conditioned, which
  is why we use Householder QR instead of a Cholesky-on-normal-equations
  shortcut.
* `_fit` requires `rows >= 1` and `cols >= degree + 2`. Returns
  `N4M_ERR_NUMERICAL_FAILURE` when the QR factor's diagonal is zero
  (linearly dependent basis columns).
* `_transform` returns `N4M_ERR_NOT_FITTED` before fit, and
  `N4M_ERR_NUMERICAL_FAILURE` when the per-row reference coefficient
  $c_0 = 0$.
* Tolerance against nirs4all + `np.linalg.lstsq` (LAPACK gelsd): 5e-10
  absolute / 5e-10 relative. The handrolled Householder QR vs gelsd's SVD
  agree to ~8 decimal digits on this basis; the gap is structural to the
  conditioning of $B$, not a bug.

## Reference

Martens, H., Stark, E. (1991). "Extended Multiplicative Signal Correction
and Spectral Interference Subtraction: New Preprocessing Methods for Near
Infrared Spectroscopy." *Journal of Pharmaceutical and Biomedical Analysis*
9 (8), 625-635.

Implementation matches `nirs4all.operators.transforms.nirs.ExtendedMultiplicativeScatterCorrection`
with `scale=False`.
