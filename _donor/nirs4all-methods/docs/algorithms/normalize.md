# Normalize (column-wise spectral normalisation)

## Algorithm

Column-wise (axis=0) normalisation with two modes selected by the desired output range $(\ell, u)$:

* **linalg-norm mode** (`(ell, u) == (-1, 1)`, the default in nirs4all): each column is divided by its L2 norm.
  $$
  n_j = \sqrt{\sum_{i=1}^{n} x_{i,j}^2}, \qquad
  x'_{i,j} = \frac{x_{i,j}}{n_j}
  $$

* **user-defined-range mode** (any other $(\ell, u)$): each column is min-max scaled into $[\ell, u]$.
  $$
  \mu_j = \min_i x_{i,j}, \;\; M_j = \max_i x_{i,j}, \;\;
  f_j = \frac{u - \ell}{M_j - \mu_j}, \;\;
  x'_{i,j} = \ell + f_j (x_{i,j} - \mu_j)
  $$

## Parameters

| Parameter | Default | Meaning |
| --- | --- | --- |
| `feature_min` | `-1.0` | Lower bound of the target range. |
| `feature_max` | `+1.0` | Upper bound of the target range. |

Mode selection is `(feature_min != -1.0) || (feature_max != 1.0)` — exactly matching nirs4all's `user_defined` boolean.

## Numerical contract

* Column sum-of-squares accumulates left-to-right with the same rounding sequence as `np.linalg.norm(X, axis=0)`.
* All divisions are evaluated per element (no precomputed reciprocal-multiplication), matching `X / linalg_norm_` in nirs4all.
* The user-defined-range arithmetic exactly mirrors `imin + f * (X - min_)` from nirs4all (single multiplication, single addition).

## Reference

Implementation matches `nirs4all.operators.transforms.scalers.Normalize`. Note this is **column-wise** — distinct from `sklearn.preprocessing.normalize`, which is row-wise.
