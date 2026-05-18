# Multiplicative Scatter Correction (MSC)

## Algorithm

Per-column scatter correction calibrated against the per-row mean of the
training matrix. Let $X \in \mathbb{R}^{n \times p}$ be the training spectra
matrix.

**Fit** computes per-row means and per-column linear regression coefficients:

$$
r_i = \frac{1}{p} \sum_{j=1}^{p} X_{i,j}, \qquad
(a_j, b_j) = \mathrm{polyfit}(r, X_{:, j}, \mathrm{deg}=1)
$$

For `deg=1`, `np.polyfit` returns `[slope, intercept]`. We use the
closed-form least-squares formulae directly:

$$
\bar{r} = \tfrac{1}{n} \sum_i r_i, \quad
a_j = \frac{\sum_i (r_i - \bar{r}) X_{i,j}}{\sum_i (r_i - \bar{r})^2}, \quad
b_j = \overline{X_{:, j}} - a_j \bar{r}.
$$

**Transform** divides each column by its slope and subtracts the
intercept:

$$
X'_{i,j} = \frac{X_{i,j} - b_j}{a_j}.
$$

**Inverse transform** reverses the operation:

$$
X_{i,j} = X'_{i,j} \cdot a_j + b_j.
$$

The fitted parameters $(a_j, b_j)$ are independent of the transform input —
calling `_fit(X_train)` then `_transform(X_test)` applies the calibration
learned on $X_\text{train}$ to a fresh $X_\text{test}$ of the same column
count.

## Parameters

This operator takes no constructor parameters. Use `_create()`, then
`_fit(X)`, then `_transform(X)` (and optionally `_inverse_transform(X)`).

## Numerical contract

* `_fit` requires `rows >= 2` and `cols >= 1`. Returns
  `C4A_ERR_NUMERICAL_FAILURE` when the per-row means form a constant vector
  (zero variance — slope undefined).
* `_transform` and `_inverse_transform` return `C4A_ERR_NOT_FITTED` before
  the first successful `_fit`, and `C4A_ERR_SHAPE_MISMATCH` if the input
  column count differs from the fitted count.
* Tolerance against nirs4all + numpy 1.26.4: 1e-10 absolute / 1e-11
  relative. Our closed-form `(a_j, b_j)` differ from `np.polyfit`'s
  Vandermonde + lstsq path by a couple of ULPs.

## Reference

Geladi, P., MacDougall, D., Martens, H. (1985). "Linearization and
Scatter-Correction for Near-Infrared Reflectance Spectra of Meat."
*Applied Spectroscopy* 39 (3), 491-500.

Implementation matches `nirs4all.operators.transforms.nirs.MultiplicativeScatterCorrection`
with `scale=False`.
