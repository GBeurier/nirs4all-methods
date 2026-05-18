# HighLeverageFilter (hat-matrix sample leverage)

Identifies samples with high leverage (influence) on a linear regression fit. The leverage of sample i is the i-th diagonal of the hat matrix `H = X (X^T X)^{-1} X^T`. Samples whose leverage exceeds a threshold are excluded; the remaining samples form the training set for downstream models.

## Methods

Two computation paths, selected by the `method` constructor parameter:

1. **`hat`** (`method == 0`) — direct hat-matrix diagonal in feature space. Requires `rows > cols`. When `rows <= cols` the engine auto-falls back to the PCA path to keep `X^T X` non-singular.
2. **`pca`** (`method == 1`) — explicit PCA / score-space leverage. The training matrix is decomposed via Jacobi eigendecomposition of `X_c^T X_c` (where `X_c` is the centred matrix). The top `n_components` eigenvectors form the projection; in score space the precision matrix is diagonal so leverage reduces to `h_i = sum_j scores_ij^2 / (lambda_j + reg)`.

The default `n_components` (when set to 0 or a non-positive value) is `min(rows - 1, cols, 50)` — matching the nirs4all reference.

## Threshold

Two threshold modes:

- **multiplier** (default): `threshold = multiplier * mean(training_leverages)`. The training leverages sum to the number of effective features (`trace(H) = p`), so the average is `p/n`. A multiplier of 2.0 is lenient (rejects only the most extreme leverages); 3.0 is strict.
- **absolute**: `threshold = absolute_threshold`. Must satisfy `0 < absolute_threshold < 1` because leverage is bounded by `[0, 1]`.

A sample i is KEPT iff `leverage[i] <= threshold`. The `apply` entry point requires the input column count to match the fit-time count (returns `C4A_ERR_SHAPE_MISMATCH` otherwise).

## Numerical contract

- Tikhonov regularisation: `reg = 1e-10 * trace(X_c^T X_c) / cols`. Identical to the nirs4all reference.
- Hat path: Householder QR factorisation of `X_c^T X_c + reg * I` via the shared `c4a_householder_qr` helper; per-row leverage computed by `(R^{-1} Q^T x_i) . x_i` (back-substitution, no explicit inverse).
- PCA path: cyclic Jacobi eigendecomposition of `X_c^T X_c` (no scipy dependency). Sign convention matches sklearn (each component's maximum-absolute element is non-negative).
- Parity tolerance: `1e-9 abs / 1e-10 rel` against the frozen NumPy reference. Mask equality is exact at the contracted tolerance because the threshold lies far from any data point in the fixture.

## ABI

```c
c4a_status_t c4a_filter_leverage_create(c4a_filter_leverage_handle_t** out,
                                         int32_t method,
                                         double  threshold_multiplier,
                                         int     use_absolute_threshold,
                                         double  absolute_threshold,
                                         int32_t n_components,
                                         int     center);
void         c4a_filter_leverage_destroy(c4a_filter_leverage_handle_t* h);
c4a_status_t c4a_filter_leverage_fit(c4a_filter_leverage_handle_t* h,
                                      c4a_matrix_view_t X);
c4a_status_t c4a_filter_leverage_is_fitted(const c4a_filter_leverage_handle_t* h,
                                            int* out_fitted);
c4a_status_t c4a_filter_leverage_apply(const c4a_filter_leverage_handle_t* h,
                                        c4a_matrix_view_t X,
                                        uint8_t* mask_out,
                                        c4a_filter_stats_t* stats_out);
double       c4a_filter_leverage_threshold(const c4a_filter_leverage_handle_t* h);
```

## Reference

- nirs4all 0.8.x — `nirs4all/operators/filters/high_leverage.py`
- `parity/python_generator/src/c4a_parity_filters_ref/high_leverage.py`
