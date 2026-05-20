# Multiplicative Scatter Correction (MSC)

## Algorithm

Conventional row-wise scatter correction calibrated against the mean reference
spectrum of the training matrix. Let
$X_\mathrm{fit} \in \mathbb{R}^{n \times p}$ be the training spectra matrix.

**Fit** computes the reference spectrum:

$$
r_j = \frac{1}{n} \sum_{i=1}^{n} X_{\mathrm{fit}, i,j}
$$

and its centered denominator:

$$
\bar{r} = \frac{1}{p} \sum_{j=1}^{p} r_j, \qquad
d_r = \sum_{j=1}^{p} (r_j - \bar{r})^2.
$$

**Transform** regresses each input row $x_i$ against the fitted reference
spectrum:

$$
\bar{x}_i = \frac{1}{p} \sum_{j=1}^{p} x_{i,j}, \qquad
s_i = \frac{\sum_j (x_{i,j} - \bar{x}_i)(r_j - \bar{r})}{d_r},
\qquad
o_i = \bar{x}_i - s_i \bar{r}.
$$

The corrected spectrum is:

$$
X'_{i,j} = \frac{x_{i,j} - o_i}{s_i}.
$$

This is the default contract used by `prospectr::msc` and `pls::msc`: the fit
stores the reference spectrum, while the offset and slope are estimated for
each transformed spectrum.

**Inverse transform** needs the row coefficients from the forward transform:

$$
x_{i,j} = X'_{i,j} \cdot s_i + o_i.
$$

The C ABI stores those row coefficients on the handle during `_transform`, so
`_inverse_transform(_transform(X))` is covered for the same handle. Stateless
wrappers must preserve those coefficients separately; the R wrapper stores them
as attributes on the matrix returned by `msc()`.

## Parameters

This operator takes no constructor parameters. Use `_create()`, then
`_fit(X)`, then `_transform(X)` (and optionally `_inverse_transform(X)` on the
same handle after a transform).

## Numerical contract

* `_fit` requires `rows >= 1` and `cols >= 2`.
* `_fit` returns `C4A_ERR_NUMERICAL_FAILURE` when the reference spectrum has
  zero variance.
* `_transform` returns `C4A_ERR_NUMERICAL_FAILURE` when a row slope is zero or
  non-finite.
* `_inverse_transform` returns `C4A_ERR_SHAPE_MISMATCH` unless the input shape
  matches the previous transform on the same handle.
* The active benchmark gate compares against `prospectr::msc`; observed
  full-matrix differences are at double-precision roundoff.

## Reference

Geladi, P., MacDougall, D., Martens, H. (1985). "Linearization and
Scatter-Correction for Near-Infrared Reflectance Spectra of Meat."
*Applied Spectroscopy* 39 (3), 491-500.

Implementation follows the conventional reference-spectrum MSC contract used by
`prospectr::msc` and `pls::msc`. The local development checkout of `nirs4all`
currently exposes a historical column-regression MSC variant, so it is kept as
performance context rather than a parity gate until that package is updated.
