# Phase 9 — Feature selection & dimensionality reduction (PARTIAL)

## Goal

Add SVD-based dimensionality-reduction operators to the c4a preprocessing
toolkit, alongside a portable in-process SVD primitive that other phases
(Phase 8 OSC, Phase 19 T²/Q-residuals) can share.

This phase is intentionally **partial**: only the two stateful
SVD-based operators (FlexiblePCA, FlexibleSVD) are implemented. The
two PLS-driven feature selectors (CARS, MCUVE) require an internal PLS
callback that the binding surface cannot yet wire — they are deferred
to a later phase when the PLS engine lands.

## Out of scope

- CARS (Competitive Adaptive Reweighted Sampling) — requires PLS
  Monte-Carlo loop.
- MCUVE (Monte-Carlo Uninformative Variable Elimination) — requires
  PLS bootstrap loop.
- Whitening in `FlexiblePCA` (sklearn ``whiten=True`` flavour) — the
  reference uses ``whiten=False`` exclusively.

## Operators

| Operator      | What `_fit` does | What `_transform` does | Stateful fields |
|---------------|------------------|------------------------|-----------------|
| **FlexiblePCA** | centre + SVD; select k' by count or variance ratio; canonicalise signs | `(X - mean) @ components.T` | `mean[n]`, `components[k', n]` |
| **FlexibleSVD** | SVD without centring; select k' by count or variance ratio; canonicalise signs | `X @ components.T` | `components[k', n]` |

Both expose `_output_cols` as a runtime getter — the learned $k'$ is
data-dependent (variance-ratio mode) so it cannot be precomputed
without inspecting the training data.

## Sign convention

The SVD signs are canonicalised so the largest-absolute-value element
of each `U[:, j]` column is positive (sklearn ``svd_flip`` with
``u_based_decision=True``).

Note this differs from sklearn's ``TruncatedSVD`` which uses
``u_based_decision=False``: the c4a engine uses the U-based decision
uniformly across FlexiblePCA and FlexibleSVD so the same SVD primitive
can serve both. The frozen NumPy reference matches the c4a convention.

## SVD primitive

`cpp/src/core/common/svd.c` exposes `c4a_svd_compact(A, m, n, U, S, Vt)`
— a one-sided Jacobi SVD with these properties:

- pure C, no LAPACK / BLAS dependency,
- compact form: `A = U @ diag(S) @ Vt`, `k = min(m, n)`,
- consumes `A` in place,
- canonicalises signs via U-based argmax,
- transposes internally for the wide (`m < n`) case so the
  convergence loop always runs on a tall matrix (otherwise the
  rank-deficient columns prevent the Jacobi sweep from converging).

Other agents working on Phase 8 (OSC) or Phase 19 (T²/Q residuals) may
also need an SVD helper. The API contract is the function signature
`c4a_svd_compact(A, m, n, U, S, Vt)`; conflicts will be resolved at
merge time by keeping the most general implementation.

## C ABI surface added

Per-operator (using FlexiblePCA as example):

```c
typedef struct c4a_pp_flex_pca_handle_t c4a_pp_flex_pca_handle_t;
C4A_API c4a_status_t c4a_pp_flex_pca_create(c4a_pp_flex_pca_handle_t** out,
                                             double n_components);
C4A_API void         c4a_pp_flex_pca_destroy(c4a_pp_flex_pca_handle_t* h);
C4A_API c4a_status_t c4a_pp_flex_pca_fit(c4a_pp_flex_pca_handle_t* h,
                                          c4a_matrix_view_t X);
C4A_API c4a_status_t c4a_pp_flex_pca_transform(
    const c4a_pp_flex_pca_handle_t* h,
    c4a_matrix_view_t X,
    c4a_matrix_view_t out);
C4A_API c4a_status_t c4a_pp_flex_pca_is_fitted(
    const c4a_pp_flex_pca_handle_t* h, int* out_fitted);
C4A_API c4a_status_t c4a_pp_flex_pca_output_cols(
    const c4a_pp_flex_pca_handle_t* h, int64_t* out_cols);
```

Twelve symbols total (six per operator). These live in
`cpp/src/c_api/c_api_feature_selection.cpp`; declarations move to
`c4a.h` §9.x at integration time.

## n_components encoding

Both operators take a single `double n_components`:

- `>= 1.0` → integer count mode (truncated)
- `(0, 1)` → variance-ratio mode

Values `<= 0` or `NaN` are rejected with `C4A_ERR_INVALID_ARGUMENT`.

## Parity

Frozen NumPy reference under
`parity/python_generator/src/c4a_parity_orthog_ref/` — independent of
sklearn so upstream changes can't drift the fixture. Tolerance budget:
`1e-10` absolute / `1e-11` relative (per brief), accommodating the
Jacobi vs LAPACK gesdd ULP gap on mid-to-low singular values.

Fixtures:
- `parity/fixtures/flexible_pca_v1.json` (3 cases: count_5, variance_0_95,
  count_15)
- `parity/fixtures/flexible_svd_v1.json` (3 cases: count_5, variance_0_99,
  count_15)
