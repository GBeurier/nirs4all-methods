# WaveletPCA

## Algorithm

Two-stage stateful dimensionality reduction.  Each row is passed
through a multi-level DWT, the resulting coefficient arrays are
flattened in the canonical
$[c_A^{(K)} \,\Vert\, c_D^{(K)} \,\Vert\, \ldots \,\Vert\, c_D^{(1)}]$
order, and the per-row coefficient matrix is fed to the same PCA
pipeline as `FlexiblePCA` (centre, compact SVD, sign-canonicalised
components).

Concretely, given a training matrix $X \in \mathbb{R}^{m \times n}$ and
an effective level $K$ (clamped to `dwt_max_level(n, family)`):

1. Build $F \in \mathbb{R}^{m \times D}$ where $D = \sum_k |c^{(k)}|$
   is the flat coefficient dimension and $F_i = \text{wavedec}(X_i)$
   flattened.
2. $\mu = \frac{1}{m} \sum_i F_i$, $F_c = F - \mathbf{1}\mu^\top$.
3. Compact SVD $F_c = U S V^\top$, signs canonicalised via
   `svd_flip(U, V^\top, u_based=True)`.
4. Explained-variance ratio $\rho_j = S_j^2 / \sum_i S_i^2$, multiplied
   by $(m - 1)^{-1}$ for the variance scale.
5. Component count $k'$ selected via the flexible specifier:
   integer $\geq 1$ (count mode) or float in $(0, 1)$ (variance ratio).
6. Components $= V^\top_{[:k', :]}$, mean $= \mu$.

`_transform` then re-applies the same DWT-flatten step to fresh rows
and projects them through $(F_{\text{new}} - \mu) V^{\top \top}_{[:k', :]}$.

## Parameters

| Parameter      | Default       | Meaning                                  |
| -------------- | ------------- | ---------------------------------------- |
| `family`       | (db4)         | Wavelet family                           |
| `mode`         | (per.)        | Boundary mode                            |
| `max_level`    | 4             | Maximum decomposition depth             |
| `n_components` | —             | Flexible count or variance-ratio specifier |

`n_components` rules match `FlexiblePCA`:

- `>= 1.0`: count mode, truncated to `min(int(value), min(m, D))`.
- `(0, 1)`: variance-ratio mode, smallest $k'$ such that
  $\sum_{j=1}^{k'} \rho_j \geq \text{value}$.

## Lifecycle

Stateful: `_create / _fit / _transform / _destroy` plus `_is_fitted`
and `_output_cols`.  Calling `_transform` or `_output_cols` before
`_fit` returns `C4A_ERR_NOT_FITTED`.

## Numerical contract

- `_fit` requires `rows >= 2` and `cols >= 1`.
- `_transform` requires `out.cols == n_components_learned` and
  `X.cols == n_features_in_at_fit_time`.
- Parity tolerance against the frozen NumPy reference: `1e-10` absolute
  / `1e-11` relative (DWT is closed-form; the SVD gap against LAPACK
  `gesdd` is the dominant source of ULP-level noise).

## Reference

`nirs4all.operators.transforms.nirs.WaveletPCA` (Trygg & Wold 1998,
"PLS regression on wavelet compressed NIR spectra").  The nirs4all
reference fits a separate `sklearn.decomposition.PCA` per level;
chemometrics4all fits a single PCA on the concatenated coefficient
stack, matching the simpler "flatten + PCA" path documented in the
v1 brief.
