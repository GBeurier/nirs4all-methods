# Baseline (Column-Mean Centering)

## Algorithm

Per-column mean centering. Let $X \in \mathbb{R}^{n \times p}$ be the
training matrix.

**Fit** computes per-column means:

$$
\mu_j = \overline{X_{:, j}} = \frac{1}{n} \sum_{i=1}^{n} X_{i,j}.
$$

**Transform**:

$$
X'_{i,j} = X_{i,j} - \mu_j.
$$

**Inverse transform**:

$$
X_{i,j} = X'_{i,j} + \mu_j.
$$

## Naming note

The class name **Baseline** is historical. This operator is column-mean
centering — equivalent to `sklearn.preprocessing.StandardScaler(with_std=False)`.
It is **not** a baseline correction in the spectroscopic sense (a polynomial
or asymmetric-LS baseline removal lives in Phase 5).

## Parameters

This operator takes no constructor parameters. Use `_create()`, then
`_fit(X)`, then `_transform(X)` / `_inverse_transform(X)`.

## Numerical contract

* `_fit` requires `rows >= 1` and `cols >= 1`.
* `_transform` and `_inverse_transform` return `C4A_ERR_NOT_FITTED` before
  fit, and `C4A_ERR_SHAPE_MISMATCH` on column-count drift.
* Tolerance against nirs4all + numpy: 1e-12 absolute / 1e-13 relative
  (closed-form arithmetic — no least-squares involvement).

## Reference

Implementation matches `nirs4all.operators.transforms.signal.Baseline`.
The arithmetic is the same as scikit-learn's
`sklearn.preprocessing.StandardScaler(with_std=False)`.
