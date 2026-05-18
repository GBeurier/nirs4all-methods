# OSC — Orthogonal Signal Correction (Direct OSC)

## Purpose

Remove systematic variation in `X` that is orthogonal (statistically
unrelated) to a target `y`. Useful for NIRS pipelines where scattering,
temperature drift or instrumental effects obscure chemical information
that correlates with `y`.

## Algorithm

This is the **Direct OSC (DOSC)** variant (Westerhuis 2001), more
numerically stable than the original Wold 1998 iteration.

For each row pair `(X, y)` (univariate target):

1. Center + scale `X` and `y` to their per-column / scalar mean and
   sample standard deviation (ddof=1).
2. Compute the Y-predictive direction
   `w_pls = X^T y / ||X^T y||`
   (this is the SVD-fallback path documented below).
3. For each orthogonal component `i = 1..n_components`:
   - `w = X_res^T y / ||X_res^T y||`
   - `t = X_res w`,  `p = X_res^T t / ||t||²`
   - Subtract the `w_pls` projection from the loading: `w_ortho = p − (p · w_pls) w_pls`, normalise.
   - `t_ortho = X_res w_ortho`,  `p_ortho = X_res^T t_ortho / ||t_ortho||²`
   - Deflate: `X_res ← X_res − t_ortho p_ortho^T`
   - Store `W_ortho[:, i] = w_ortho`, `P_ortho[:, i] = p_ortho`

At transform time the same deflation is replayed against the centered +
scaled input, then the result is unscaled back to the original units.

## SVD-fallback vs PLS-weight extraction

chemometrics4all does **not** ship embedded PLS models (Phase 0 / 1 design
ban). The original nirs4all 0.8.x reference computes the PLS weight via
the first iteration of NIPALS, which for a univariate target `y` reduces
to the normalised cross-correlation `X^T y / ||X^T y||`. We use that
direction directly — equivalently the leading right-singular vector of
the rank-1 matrix `y x^T`. The two formulations are bit-equal for
univariate `y`; we verified parity against the nirs4all 0.8.x reference
in the Phase 8 brief (`max_abs ≤ 2 ULPs`).

For multivariate `y` (deferred — not exposed at the public ABI today)
the SVD fallback would require computing the leading singular pair of
the cross-product matrix `X^T Y`. We do not ship that path today since
the c4a OSC ABI accepts only a univariate `y` (length `n_samples`).

## Invertibility

OSC is **not** invertible. Each orthogonal-component deflation step
projects samples onto the subspace orthogonal to the stored components,
which is many-to-one — the original point cannot be uniquely recovered
from the transformed output. The public ABI's
`c4a_pp_osc_inverse_transform` returns `C4A_ERR_UNSUPPORTED`.

## C ABI

```c
typedef struct c4a_pp_osc_handle_t c4a_pp_osc_handle_t;

C4A_API c4a_status_t c4a_pp_osc_create(c4a_pp_osc_handle_t** out,
                                        int32_t n_components, int scale);
C4A_API void         c4a_pp_osc_destroy(c4a_pp_osc_handle_t* h);
C4A_API c4a_status_t c4a_pp_osc_fit(c4a_pp_osc_handle_t* h,
                                     c4a_matrix_view_t X,
                                     const double* y, int64_t y_len);
C4A_API c4a_status_t c4a_pp_osc_transform(const c4a_pp_osc_handle_t* h,
                                           c4a_matrix_view_t X,
                                           c4a_matrix_view_t out);
C4A_API c4a_status_t c4a_pp_osc_inverse_transform(const c4a_pp_osc_handle_t* h,
                                                   c4a_matrix_view_t X,
                                                   c4a_matrix_view_t out);
C4A_API c4a_status_t c4a_pp_osc_is_fitted(const c4a_pp_osc_handle_t* h,
                                           int* out_fitted);
```

`scale != 0` enables center+scale of both `X` and `y` (ddof=1
sample standard deviation, zero-std features replaced by 1.0).
`scale == 0` disables centering and scaling.

`n_components` is the requested number of orthogonal components.
Effective count is `min(n_components, cols - 1, rows - 2)`. The
algorithm short-circuits on degenerate norms (≤ 1e-10) and stops at
the last successful component — the surviving count is queryable via
the internal state and is the stride of the stored `W_ortho` / `P_ortho`
matrices.

## Parity tolerance

| Reference | Abs tol | Rel tol |
|-----------|---------|---------|
| Frozen NumPy ref (`c4a_parity_orthog_ref.osc`) | 1e-9 | 1e-10 |

The frozen NumPy reference matches nirs4all 0.8.x bit-for-bit (≤ 2 ULPs)
across the four parameter combinations covered in `osc_v1.json`.

## References

1. Wold, S., Antti, H., Lindgren, F., Öhman, J. (1998). Orthogonal
   signal correction of near-infrared spectra. *Chemometrics and
   Intelligent Laboratory Systems*, 44(1-2), 175-185.
2. Westerhuis, J. A., de Jong, S., Smilde, A. K. (2001). Direct
   orthogonal signal correction. *Chemometrics and Intelligent
   Laboratory Systems*, 56(1), 13-25.
3. Fearn, T. (2000). On orthogonal signal correction. *Chemometrics
   and Intelligent Laboratory Systems*, 50(1), 47-52.
