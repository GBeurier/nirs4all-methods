# Q-residuals (Squared Prediction Error)

Per-sample residual energy outside the first `n_components` PCA
directions. Complementary to Hotelling T²: T² measures variation inside
the model, Q-residuals measure variation outside.

## Algorithm

For a `(n_samples × n_features)` matrix `X`:

1. Centre `X` by column means: `Xc = X - mean(X, axis=0)`.
2. Compact SVD: `Xc = U Σ V^T`.
3. Reconstruct using only the first `k = n_components` directions:
   `Xc_hat[i, :] = Σ_{j<k} U[i, j] · σ_j · V[:, j]^T`.
4. Per-sample residual:
   `Q[i] = Σ_d (Xc[i, d] − Xc_hat[i, d])²`.

## Upper control limit (UCL) — Jackson-Mudholkar approximation

Let `θ_a = Σ_{j ≥ k} (σ_j² / (n − 1))^a` for `a = 1, 2, 3`. Define
`h0 = 1 − (2 θ_1 θ_3) / (3 θ_2²)`. Then:

```
ca  = Φ⁻¹(1 − alpha)
UCL = θ_1 · ( ca · sqrt(2 θ_2 h0²) / θ_1
              + 1
              + θ_2 h0 (h0 − 1) / θ_1² )^{1 / h0}
```

`Φ⁻¹` (inverse standard-normal CDF) uses Wichura's AS241 rational
approximation. If fewer than `k + 1` non-zero singular values are
available the UCL is reported as zero.

## ABI

```c
N4M_API n4m_status_t n4m_util_q_residuals(n4m_matrix_view_t X,
                                           int32_t n_components,
                                           double alpha,
                                           double* q_per_sample,
                                           int64_t n_samples,
                                           double* ucl_out);
```

Same validation as `n4m_util_hotelling_t2`: `n_samples` must equal
`X.rows`, `alpha` ∈ (0, 1), `1 ≤ n_components ≤ cols` and `< n_samples`.

## Numerical contract

- Same Jacobi SVD as Hotelling T² (see
  [hotelling_t2.md](hotelling_t2.md) for the SVD numerics).
- Per-sample Q parity vs LAPACK reference: **1e-10 abs / 1e-11 rel**.
- UCL parity vs `scipy.stats.norm.ppf`-based reference: **1e-6 abs /
  1e-5 rel**. The Wichura AS241 inverse-normal is accurate to ~7 digits
  in the central region (Φ⁻¹(0.95)) — sufficient for a statistical
  control limit but not bit-exact against the scipy LAPACK-quality
  inverter.

## Reference

- Jackson, J. E., Mudholkar, G. S. (1979). "Control procedures for residuals
  associated with principal component analysis."
- Wichura, M. J. (1988). "Algorithm AS 241: The percentage points of the
  normal distribution." Applied Statistics 37, 477-484.
- Fixture: `parity/fixtures/q_residuals_v1.json`.
