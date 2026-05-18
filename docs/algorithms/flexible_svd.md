# FlexibleSVD

## Algorithm

Truncated Singular Value Decomposition with a flexible component
specification. Unlike `FlexiblePCA`, `FlexibleSVD` does **not** centre
the data before computing the SVD. The number of retained components
can be supplied either as an explicit count or as a
cumulative-variance-ratio threshold.

Let $X \in \mathbb{R}^{m \times n}$ be the training matrix.

**Fit** runs a compact SVD on $X$ directly:

$$
X = U \, \mathrm{diag}(S) \, V^\top,
$$

with $k = \min(m, n)$. Signs are canonicalised so the
largest-absolute-value element of each $U_{:, j}$ is positive (with
ties broken by smallest index). This is the same `u_based_decision=True`
convention used by `FlexiblePCA`, deliberately diverging from
`sklearn.decomposition.TruncatedSVD` (which uses
`u_based_decision=False` because it has no centring step). The frozen
NumPy reference matches the c4a convention; sklearn does not.

Explained variance and the cumulative-variance ratio are computed
following sklearn's TruncatedSVD recipe (which is variance of the
projected scores divided by the total variance of the input):

$$
X_{\text{proj}} = U \, \mathrm{diag}(S), \qquad
\mathrm{evar}_j = \mathrm{Var}(X_{\text{proj}, j}), \qquad
\rho_j = \frac{\mathrm{evar}_j}{\sum_{i=1}^{n} \mathrm{Var}(X_{:, i})}.
$$

Variance uses `np.var(..., axis=0)` with `ddof=0`. The component
selection rule is identical to `FlexiblePCA`:

- **count mode** (when `n_components >= 1`):
  $k' = \min(\lfloor n_{components} \rfloor, k)$.
- **variance-ratio mode** (when `0 < n_components < 1`):
  smallest $k'$ such that $\sum_{j=1}^{k'} \rho_j \geq n_{components}$.

The kept components are stored as `components` $= V^\top_{[:k', :]}$.

**Transform** projects without centring:

$$
\text{out} = X_\text{new} \, \text{components}^\top.
$$

## Parameters

| Parameter      | Default | Meaning                                    |
| -------------- | ------- | ------------------------------------------ |
| `n_components` | —       | Flexible count or variance-ratio specifier |

The constructor rejects values `<= 0` or `NaN` with
`C4A_ERR_INVALID_ARGUMENT`.

## Lifecycle

Stateful: `_create / _fit / _transform / _destroy` with companion
`_is_fitted` and `_output_cols` helpers. Same contract as
`FlexiblePCA` — refits replace the prior fit.

## Numerical contract

- `_fit` requires `rows >= 2` and `cols >= 1`. Returns
  `C4A_ERR_INVALID_ARGUMENT` otherwise.
- `_transform` returns `C4A_ERR_SHAPE_MISMATCH` if the output column
  count is not exactly `n_components_learned`, or the input column count
  is not exactly `n_features_in`.
- Tolerance against the frozen NumPy reference: 1e-10 absolute / 1e-11
  relative — same as `FlexiblePCA`.

## Reference

nirs4all 0.8.x ``operators/transforms/feature_selection.py``, class
``FlexibleSVD``.
