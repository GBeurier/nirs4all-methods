# EPO — External Parameter Orthogonalization

## Purpose

Remove variation in `X` that is linearly correlated with a known
external parameter `d` (e.g. temperature, humidity, batch ID). Unlike
OSC, EPO is unsupervised wrt the target `y` and therefore cannot
accidentally remove `y`-relevant information unless that information
also correlates with `d`.

## Algorithm

For a univariate external parameter `d` (length `n_samples`):

1. Compute calibration means (when `scale != 0`):
   `X_mean = mean(X, axis=0)`,  `d_mean = mean(d)`.
   When `scale == 0` both means are taken as zero.
2. Solve the per-feature scalar regression
   `B[j] = (d − d_mean)^T (X[:, j] − X_mean[j]) / (d − d_mean)^T (d − d_mean)`,
   for `j = 0..cols-1`.

At fit time the n4m engine stores `X_mean`, `d_mean`, and `B`. The
training-time output `X - outer(d - d_mean, B)` is what nirs4all's
`fit_transform(X, d)` returns.

## Transform contracts

The core engine exposes two apply paths:

* `apply(X)` has no `d` vector. It assumes the new samples sit at the
  calibration mean (`d = d_mean`), so `d - d_mean = 0` and the projection is
  an identity/pass-through.
* `apply_with_d(X, d)` performs the actual EPO correction:
  `X - outer(d - d_mean, B)`.

Both contracts are public ABI. Use `n4m_pp_epo_transform(...)` when the
external parameter is not available for new samples; use
`n4m_pp_epo_transform_with_d(...)` for training-time projection or any
inference batch where `d` is known.

## C ABI

```c
typedef struct n4m_pp_epo_handle_t n4m_pp_epo_handle_t;

N4M_API n4m_status_t n4m_pp_epo_create(n4m_pp_epo_handle_t** out, int scale);
N4M_API void         n4m_pp_epo_destroy(n4m_pp_epo_handle_t* h);
N4M_API n4m_status_t n4m_pp_epo_fit(n4m_pp_epo_handle_t* h,
                                     n4m_matrix_view_t X,
                                     const double* d, int64_t d_len);
N4M_API n4m_status_t n4m_pp_epo_transform(const n4m_pp_epo_handle_t* h,
                                           n4m_matrix_view_t X,
                                           n4m_matrix_view_t out);
N4M_API n4m_status_t n4m_pp_epo_transform_with_d(
    const n4m_pp_epo_handle_t* h,
    n4m_matrix_view_t X,
    const double* d, int64_t d_len,
    n4m_matrix_view_t out);
N4M_API n4m_status_t n4m_pp_epo_inverse_transform(const n4m_pp_epo_handle_t* h,
                                                   n4m_matrix_view_t X,
                                                   n4m_matrix_view_t out);
N4M_API n4m_status_t n4m_pp_epo_is_fitted(const n4m_pp_epo_handle_t* h,
                                           int* out_fitted);
```

`scale != 0` enables centering of both `X` and `d`. The
`_inverse_transform` returns `N4M_ERR_UNSUPPORTED` (the projection
removes the `d`-correlated subspace, which is lossy).

`_fit` returns `N4M_ERR_NUMERICAL_FAILURE` when `d` has zero variance
(the per-feature regression denominator vanishes).

`_transform_with_d` returns `N4M_ERR_SHAPE_MISMATCH` when `d_len` does not
match the number of rows in `X`.

## Parity tolerance

| Reference | Abs tol | Rel tol |
|-----------|---------|---------|
| Internal parity fixture (`n4m_parity_orthog_ref.epo`) | 1e-9 | 1e-10 |

The internal parity fixture matches nirs4all 0.8.x bit-for-bit (≤ 2 ULPs)
across the four scale/variant combinations covered in `epo_v1.json`.

## References

1. Roger, J. M., Chauchard, F., Bellon-Maurel, V. (2003). EPO–PLS:
   external parameter orthogonalisation of PLS. Application to
   temperature-independent measurement of sugar content of intact
   fruits. *Chemometrics and Intelligent Laboratory Systems*, 66(2),
   191-204.
