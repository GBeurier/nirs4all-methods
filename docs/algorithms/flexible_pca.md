# FlexiblePCA

## Algorithm

PCA (Principal Component Analysis) with a flexible component
specification. The number of retained components can be supplied either
as an explicit count or as a cumulative-variance-ratio threshold.

Let $X \in \mathbb{R}^{m \times n}$ be the training matrix.

**Fit** centres the data and computes a compact SVD:

$$
\mu = \frac{1}{m} \sum_{i=1}^{m} X_{i, :}, \qquad
X_c = X - \mathbf{1}_m \mu^\top, \qquad
X_c = U \, \mathrm{diag}(S) \, V^\top
$$

with $U \in \mathbb{R}^{m \times k}$, $S \in \mathbb{R}^{k}$,
$V^\top \in \mathbb{R}^{k \times n}$, $k = \min(m, n)$.

Signs are canonicalised so the largest-absolute-value element of each
$U_{:, j}$ is positive (with ties broken by the smallest index — i.e.
``np.argmax(|U|, axis=0)``). The flip is propagated to $V^\top$.

Explained variance and the cumulative-variance ratio are:

$$
\mathrm{evar}_j = \frac{S_j^2}{m - 1}, \qquad
\rho_j = \frac{\mathrm{evar}_j}{\sum_{i=1}^{k} \mathrm{evar}_i}.
$$

The number of components $k'$ retained is:

- **count mode** (when ``n_components >= 1``):
  $k' = \min(\lfloor n_{components} \rfloor, k)$.
- **variance-ratio mode** (when ``0 < n_components < 1``):
  smallest $k'$ such that $\sum_{j=1}^{k'} \rho_j \geq n_{components}$.

The kept components are stored as ``components`` $= V^\top_{[:k', :]}$.

**Transform** projects a fresh matrix onto the learned components:

$$
\text{out} = (X_\text{new} - \mathbf{1}\mu^\top)\,\text{components}^\top.
$$

The c4a engine uses a one-sided Jacobi SVD (``cpp/src/core/common/svd.c``)
which agrees with ``np.linalg.svd`` to within a few ULPs on the singular
values; the resulting transform parity tolerance is ``1e-10`` absolute /
``1e-11`` relative against the internal parity fixture.

## Parameters

| Parameter      | Default | Meaning                                    |
| -------------- | ------- | ------------------------------------------ |
| `n_components` | —       | Flexible count or variance-ratio specifier |

The constructor rejects values `<= 0` or `NaN` with
`C4A_ERR_INVALID_ARGUMENT`.

## Lifecycle

Stateful: `_create / _fit / _transform / _destroy` with companion
`_is_fitted` and `_output_cols` helpers. `_output_cols` reads the
learned $k'$ off the fitted state. Calling `_transform` or
`_output_cols` before `_fit` returns `C4A_ERR_NOT_FITTED`. Calling
`_fit` again replaces the prior fit (sklearn-style refit semantics).

## Numerical contract

- `_fit` requires `rows >= 2` and `cols >= 1`. Returns
  `C4A_ERR_INVALID_ARGUMENT` otherwise.
- `_transform` returns `C4A_ERR_SHAPE_MISMATCH` if the output column
  count is not exactly `n_components_learned`, or the input column count
  is not exactly `n_features_in`.
- Tolerance against the internal parity fixture: 1e-10 absolute / 1e-11
  relative — within the structural gap between Jacobi SVD and LAPACK
  gesdd on well-conditioned NIR matrices.

## Reference

nirs4all 0.8.x ``operators/transforms/feature_selection.py``, class
``FlexiblePCA``, with ``whiten=False``.
