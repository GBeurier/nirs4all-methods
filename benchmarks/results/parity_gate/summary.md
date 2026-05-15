# Parity gate report

Host: `Linux-6.6.114.1-microsoft-standard-WSL2-x86_64-with-glibc2.35`
pls4all: `0.74.0+abi.1.5.0`
Python: `3.11.14`
NumPy: `1.26.4`

Each method is compared against a Python reference and an R reference. Methods without a widely installable external reference are flagged `numpy-mirror` in the lib column.

| Method | Description | Reference (lang / lib / version) | Parity | RMSE rel | Tolerance | Status |
|--------|-------------|----------------------------------|--------|----------|-----------|--------|
| `sparse_simpls` | Sparse SIMPLS with soft-threshold lambda | python / `numpy-mirror` 1.26.4 | ✓ | 5.670e-03 | 1e+00 | ok |
| `sparse_simpls` | Sparse SIMPLS with soft-threshold lambda | R / `spls` 2.3.2 | ✓ | 2.512e-05 | 1e+00 | ok |
| `di_pls` | Domain-invariant PLS | python / `numpy-mirror` 1.26.4 | ✓ | 4.820e-03 | 1e-02 | ok |
| `di_pls` | Domain-invariant PLS | R / `(none)` - | ✗ | — | 1e-02 | no_r_reference |
| `recursive_pls` | Recursive (moving-window) PLS | python / `scikit-learn` 1.4.2 | ✓ | 1.226e-02 | 1e-01 | ok |
| `recursive_pls` | Recursive (moving-window) PLS | R / `pls` 2.8.5 | ✓ | 1.226e-02 | 1e-01 | ok |
| `cppls` | CPPLS (column-power-rescaled SIMPLS) | python / `numpy-mirror` 1.26.4 | ✓ | 9.884e-14 | 5e-02 | ok |
| `cppls` | CPPLS (column-power-rescaled SIMPLS) | R / `(none)` - | ✗ | — | 5e-02 | no_r_reference |
| `weighted_pls` | Sample-weighted PLS (sqrt(w)-prescaled SIMPLS) | python / `numpy-mirror` 1.26.4 | ✓ | 3.457e-13 | 5e-02 | ok |
| `weighted_pls` | Sample-weighted PLS (sqrt(w)-prescaled SIMPLS) | R / `(none)` - | ✗ | — | 5e-02 | no_r_reference |
| `robust_pls` | Robust PLS (Huber IRLS over weighted SIMPLS) | python / `numpy-mirror` 1.26.4 | ✓ | 5.737e-06 | 5e-02 | ok |
| `robust_pls` | Robust PLS (Huber IRLS over weighted SIMPLS) | R / `(none)` - | ✗ | — | 5e-02 | no_r_reference |
| `ridge_pls` | Ridge-augmented PLS | python / `numpy-mirror` 1.26.4 | ✓ | 7.218e-13 | 5e-02 | ok |
| `ridge_pls` | Ridge-augmented PLS | R / `(none)` - | ✗ | — | 5e-02 | no_r_reference |
| `continuum_regression` | Continuum regression (interpolates PLS / OLS) | python / `numpy-mirror` 1.26.4 | ✓ | 9.884e-14 | 5e-02 | ok |
| `continuum_regression` | Continuum regression (interpolates PLS / OLS) | R / `(none)` - | ✗ | — | 5e-02 | no_r_reference |
| `n_pls` | N-PLS — 3-way tensor PLS (Bro 1996) | python / `numpy-mirror` 1.26.4 | ✓ | 1.360e-07 | 5e-02 | ok |
| `n_pls` | N-PLS — 3-way tensor PLS (Bro 1996) | R / `(none)` - | ✗ | — | 5e-02 | no_r_reference |
| `kernel_pls_rbf` | Non-linear kernel PLS (RBF kernel) | python / `numpy-mirror` 1.26.4 | ✓ | 2.318e-15 | 5e-02 | ok |
| `kernel_pls_rbf` | Non-linear kernel PLS (RBF kernel) | R / `(none)` - | ✗ | — | 5e-02 | no_r_reference |
| `o2pls` | O2PLS — bi-directional OPLS (Trygg & Wold 2003) | python / `numpy-mirror` 1.26.4 | ✓ | 7.974e-02 | 1e+00 | ok |
| `o2pls` | O2PLS — bi-directional OPLS (Trygg & Wold 2003) | R / `OmicsPLS` 2.1.0 | ✓ | 4.541e-01 | 1e+00 | ok |
| `approximate_press` | Approximate-PRESS component selection (§29) | python / `numpy-mirror` 1.26.4 | ✓ | 1.634e-03 | 5e-02 | ok |
| `approximate_press` | Approximate-PRESS component selection (§29) | R / `(none)` - | ✗ | — | 5e-02 | no_r_reference |

## Reference caveats

- **`sparse_simpls`** (python / `numpy-mirror`): Soft-threshold SIMPLS following Chun & Keles (2010).
- **`di_pls`** (python / `numpy-mirror`): Domain-invariant PLS — no widely installable external Python or R reference (nirs4all DiPLS is Dynamic Inner PLS; mdatools::ipls is Interval PLS). NumPy mirror of the same direction-projection formula as pls4all.
- **`cppls`** (python / `numpy-mirror`): Column-power-rescaled SIMPLS — pls4all CPPLS scales each X column by std^gamma before SIMPLS then unscales coefficients. R `pls::cppls` is named similarly but implements Liland 2009 Canonical Powered PLS, a different algorithm.
- **`weighted_pls`** (python / `numpy-mirror`): Sample-weighted SIMPLS — pre-multiplies centered rows by sqrt(w) and runs SIMPLS. R `pls::plsr` does not accept a `weights` argument and no widely installable Python or R port for this exact algorithm exists.
- **`robust_pls`** (python / `numpy-mirror`): Huber IRLS wrapped around weighted SIMPLS. Mirrors pls4all::fit_robust_pls. R `chemometrics` ships robust PCR but no widely installable robust-PLS C / R port of this specific IRLS+SIMPLS variant.
- **`ridge_pls`** (python / `numpy-mirror`): Ridge-augmented SIMPLS — augments (X, Y) with sqrt(lambda)*I and zero blocks then runs SIMPLS. No widely installable R / Python port for this specific variant.
- **`continuum_regression`** (python / `numpy-mirror`): Column-power-rescaled SIMPLS interpolating PLS (tau=0) and OLS (tau=1). R `chemometrics::pls.cv` covers regular PLS; no widely installable continuum-regression variant matches this exact rescaling formulation.
- **`n_pls`** (python / `numpy-mirror`): 3-way N-PLS (Bro 1996). NumPy mirror of pls4all::fit_n_pls. MATLAB `nplstoolbox` is the canonical reference; no widely installable R or Python port exists for the rank-1 Khatri-Rao decomposition variant.
- **`kernel_pls_rbf`** (python / `numpy-mirror`): Kernel NIPALS (Rosipal & Trejo 2001) mirror. sklearn does not ship a non-linear kernel PLS; R `pls` only covers linear kernel-algorithm. No widely installable external reference for RBF / polynomial / sigmoid kernel PLS exists.
- **`o2pls`** (python / `numpy-mirror`): O2PLS (Trygg & Wold 2003). NumPy mirror of pls4all::fit_o2pls — peels n_x_orthogonal X-orthogonal components, then n_y_orthogonal Y-orthogonal components, then runs NIPALS PLS for n_predictive joint components.
- **`approximate_press`** (python / `numpy-mirror`): Approximate PRESS (§29) — leverage-inflated residual PRESS for component selection. No widely installable LOO-PRESS equivalent — R `pls::plsr(..., validation='LOO')` returns true LOO-PRESS, which is a different quantity.
