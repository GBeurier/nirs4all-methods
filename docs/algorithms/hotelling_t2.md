# Hotelling T²

Multivariate outlier statistic on a PCA-reduced subspace.

## Algorithm

For a `(n_samples × n_features)` matrix `X`:

1. Centre `X` by column means: `Xc = X - mean(X, axis=0)`.
2. Compact SVD: `Xc = U Σ V^T` via the one-sided Jacobi solver in
   `core/common/svd.{c,h}`.
3. Take the first `k = n_components` columns of the score matrix
   `T = U Σ`.
4. The corresponding sample-covariance eigenvalues are
   `λ_j = σ_j² / (n_samples − 1)`.
5. Per-sample statistic:
   `T²[i] = Σ_{j<k} (T[i, j]² / λ_j)`.

## Upper control limit (UCL)

The F-distribution exact form:

```
UCL = k (n − 1)(n + 1) / (n (n − k)) · F⁻¹(1 − alpha; k, n − k)
```

`F⁻¹` is computed via the inverse of the regularised incomplete beta
function (Numerical Recipes §6.4 continued fraction + bisection).

## ABI

```c
C4A_API c4a_status_t c4a_util_hotelling_t2(c4a_matrix_view_t X,
                                            int32_t n_components,
                                            double alpha,
                                            double* t2_per_sample,
                                            int64_t n_samples,
                                            double* ucl_out);
```

`n_samples` must equal the row count of the view (returns
`C4A_ERR_SHAPE_MISMATCH` otherwise). `alpha` must be in (0, 1).
`n_components` must satisfy `1 ≤ n_components ≤ cols` and `< n_samples`.

## Numerical contract

- The PCA uses a Jacobi SVD with default tolerance `1e-14` and at most 50
  sweeps. For the typical NIRS PCA sizes (`p` up to a few hundred) this
  converges in 5–10 sweeps.
- F-quantile inverter: bisection to relative tolerance `1e-12`.
- Per-sample T² parity vs LAPACK reference (`numpy.linalg.svd`):
  **1e-10 abs / 1e-11 rel** — the structural gap between Jacobi rotations
  and Householder bidiagonal SVD.
- UCL parity vs `scipy.stats.f.ppf`: **1e-6 abs / 1e-6 rel** — the
  bisection inverter is intentionally simple and falls below the LAPACK
  / scipy approximation level.

## Reference

- Hotelling, H. (1931). "The Generalization of Student's Ratio".
- Fixture: `parity/fixtures/hotelling_t2_v1.json`.
