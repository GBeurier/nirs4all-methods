# WaveletSVD

## Algorithm

Two-stage stateful dimensionality reduction matching `WaveletPCA`,
but the projection step uses Truncated SVD (no mean subtraction).

Given a training matrix $X \in \mathbb{R}^{m \times n}$ and an
effective level $K$ (clamped to `dwt_max_level(n, family)`):

1. Build the flat coefficient matrix $F \in \mathbb{R}^{m \times D}$
   exactly as for `WaveletPCA`.
2. Compact SVD $F = U S V^\top$ with the same sign-canonicalisation as
   `FlexibleSVD` (`u_based_decision = True`).
3. Energy ratio $\rho_j = S_j^2 / \sum_i S_i^2$.
4. Component count $k'$ via the flexible specifier (integer
   $\geq 1$ for count, float in $(0, 1)$ for ratio).
5. Components $= V^\top_{[:k', :]}$.  No mean is stored.

`_transform` re-applies the same DWT-flatten step on incoming rows and
projects them through $F_{\text{new}} V^{\top \top}_{[:k', :]}$.

## Parameters

| Parameter      | Default       | Meaning                                  |
| -------------- | ------------- | ---------------------------------------- |
| `family`       | (db4)         | Wavelet family                           |
| `mode`         | (per.)        | Boundary mode                            |
| `max_level`    | 4             | Maximum decomposition depth             |
| `n_components` | —             | Flexible count or variance-ratio specifier |

## Lifecycle

Stateful: `_create / _fit / _transform / _destroy` plus `_is_fitted`
and `_output_cols`.  Same lifecycle and shape rules as `WaveletPCA`.

## Numerical contract

- `_fit` requires `rows >= 2` and `cols >= 1`.
- `_transform` requires `out.cols == n_components_learned` and
  `X.cols == n_features_in_at_fit_time`.
- Parity tolerance against the internal parity fixture: `1e-10` absolute
  / `1e-11` relative.

## Reference

`nirs4all.operators.transforms.nirs.WaveletSVD` (Trygg & Wold 1998).
Implementation simplifies the nirs4all per-level Truncated SVD into a
single SVD on the flattened coefficient stack.
