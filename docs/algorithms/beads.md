# BEADS — Baseline Estimation And Denoising with Sparsity (simplified)

Ning & Selesnick 2014 BEADS. **Phase 5b ships the simplified pentadiagonal variant**; see "Simplification" below for what is deferred.

For each row `y` of length `n`:

```
w := ones(n)
z := zeros(n)
for iter in 0..max_iter:
    A := diag(lam_0 * w) + (lam_1 + lam_2) * D_2^T D_2
    rhs := lam_0 * w * y
    z_new := A^{-1} rhs                              # pentadiagonal LDLT
    d := y - z_new
    w[i] := 1 / (|d[i]| + eps)                       # reweighted L2 sparsity
    w := w * n / sum(w)                              # renormalise to mean 1
    if rel_l2_diff(z, z_new) < tol: break
    z := z_new
out := y - z
```

`eps = 1e-3` prevents division by zero on perfectly matched residuals.

## Parameters
- `lam_0: double` (default 1e2) — data-fidelity weight.
- `lam_1: double` (default 0.5) — 1st-difference smoothness contribution (folded into the pentadiagonal in the simplified variant).
- `lam_2: double` (default 0.5) — 2nd-difference smoothness contribution.
- `max_iter: int` (default 50).
- `tol: double` (default 1e-3) — relative baseline-change convergence threshold.

## ABI
```c
c4a_status_t c4a_pp_beads_create(c4a_pp_beads_handle_t** out,
                                  double lam_0, double lam_1, double lam_2,
                                  int32_t max_iter, double tol);
void         c4a_pp_beads_destroy(c4a_pp_beads_handle_t* h);
c4a_status_t c4a_pp_beads_transform(const c4a_pp_beads_handle_t* h,
                                     c4a_matrix_view_t X, c4a_matrix_view_t out);
```

## Numerical contract
- Pentadiagonal LDLT (`core/common/banded_solver.h`) with row-scope `L` / `D` scratch buffers.
- Penalty: `(lam_1 + lam_2) * D_2^T D_2` (the simplification — see below).
- Convergence on `rel_l2_diff(z_old, z_new)`. Renormalising `w` to mean 1 keeps the effective lam_0 stable across iterations.
- On `max_iter` reached without convergence: silently returns the last iterate.
- Parity tolerance vs frozen NumPy reference: `1e-6 abs / 1e-7 rel`.

## Simplification

The full Ning/Selesnick BEADS algorithm uses a **7-diagonal system** combining 1st- and 2nd-difference penalties (`A = diag(w) + lam_0 * I + lam_1 * D_1^T D_1 + lam_2 * D_2^T D_2`) with a **Chebyshev polynomial approximation of the absolute value** function so that the sparsity penalty can be incorporated quadratically.

Phase 5b ships a simplification that:

1. Folds `lam_1 + lam_2` into the existing pentadiagonal `D_2^T D_2` penalty (drops the separate 1st-difference contribution).
2. Replaces the Chebyshev `|·|` approximation with a vanilla **reweighted L2** sparsity surrogate (`w[i] = 1 / (|d[i]| + eps)`).

This keeps the solve on the existing banded5 LDLT solver and stays close enough to the full algorithm to agree with pybaselines on smooth NIRS-like spectra at ~1e-3-1e-4 RMSE. The 7-diagonal BEADS plus Chebyshev approximation is deferred to a later phase if the simplified version proves insufficient for downstream pipelines.

## Reference
- Ning, X., Selesnick, I. W. & Duval, L. (2014). "Chromatogram baseline estimation and denoising using sparsity (BEADS)." Chemometrics and Intelligent Laboratory Systems 139, 156-167.
- Frozen Python reference (simplified variant): `parity/python_generator/src/c4a_parity_pybaselines_ref/beads.py`.
