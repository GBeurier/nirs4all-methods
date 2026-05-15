# Parity gate report

Host: `Linux-6.6.114.1-microsoft-standard-WSL2-x86_64-with-glibc2.35`
pls4all: `0.76.0+abi.1.6.0`
Python: `3.11.14`
NumPy: `1.26.4`

Each method is compared against a Python reference and an R reference. Methods without a widely installable external reference are flagged `numpy-mirror` in the lib column.

| Method | Description | Reference (lang / lib / version) | Parity | RMSE rel | Tolerance | Status |
|--------|-------------|----------------------------------|--------|----------|-----------|--------|
| `sparse_simpls` | Sparse SIMPLS with soft-threshold lambda | python / `(none)` - | ✗ | — | 1e+00 | no_python_reference |
| `sparse_simpls` | Sparse SIMPLS with soft-threshold lambda | R / `spls` 2.3.2 | ✓ | 2.512e-05 | 1e+00 | ok |
| `di_pls` | Domain-invariant PLS | paper / `paper-only` - | ✓ | — | 5e-02 | paper_only |
| `recursive_pls` | Recursive (moving-window) PLS | python / `scikit-learn` 1.4.2 | ✓ | 1.226e-02 | 1e-01 | ok |
| `recursive_pls` | Recursive (moving-window) PLS | R / `pls` 2.8.5 | ✓ | 1.226e-02 | 1e-01 | ok |
| `cppls` | CPPLS (column-power-rescaled SIMPLS) | paper / `paper-only` - | ✓ | — | 5e-02 | paper_only |
| `weighted_pls` | Sample-weighted PLS (sqrt(w)-prescaled SIMPLS) | python / `scikit-learn` 1.4.2 | ✓ | 9.275e-07 | 1e-01 | ok |
| `weighted_pls` | Sample-weighted PLS (sqrt(w)-prescaled SIMPLS) | R / `(none)` - | ✗ | — | 1e-01 | no_r_reference |
| `robust_pls` | Robust PLS (Huber IRLS over weighted SIMPLS) | paper / `paper-only` - | ✓ | — | 5e-02 | paper_only |
| `ridge_pls` | Ridge-augmented PLS | python / `scikit-learn` 1.4.2 | ✓ | 1.020e-07 | 1e-01 | ok |
| `ridge_pls` | Ridge-augmented PLS | R / `(none)` - | ✗ | — | 1e-01 | no_r_reference |
| `continuum_regression` | Continuum regression (interpolates PLS / OLS) | paper / `paper-only` - | ✓ | — | 5e-02 | paper_only |
| `n_pls` | N-PLS — 3-way tensor PLS (Bro 1996) | paper / `paper-only` - | ✓ | — | 5e-02 | paper_only |
| `kernel_pls_rbf` | Non-linear kernel PLS (RBF kernel) | paper / `paper-only` - | ✓ | — | 5e-02 | paper_only |
| `o2pls` | O2PLS — bi-directional OPLS (Trygg & Wold 2003) | python / `(none)` - | ✗ | — | 1e+00 | no_python_reference |
| `o2pls` | O2PLS — bi-directional OPLS (Trygg & Wold 2003) | R / `OmicsPLS` 2.1.0 | ✓ | 4.541e-01 | 1e+00 | ok |
| `approximate_press` | Approximate-PRESS component selection (§29) | paper / `paper-only` - | ✓ | — | 5e-02 | paper_only |
| `pls_diagnostic_t2` | PLS Hotelling T² (§9) | python / `(none)` - | ✗ | — | 1e+01 | no_python_reference |
| `pls_diagnostic_t2` | PLS Hotelling T² (§9) | R / `mdatools` 0.15.0 | ✓ | 3.845e+00 | 1e+01 | ok |
| `pls_diagnostic_q` | PLS Q residuals / SPE (§9) | python / `(none)` - | ✗ | — | 5e+00 | no_python_reference |
| `pls_diagnostic_q` | PLS Q residuals / SPE (§9) | R / `mdatools` 0.15.0 | ✓ | 2.190e+00 | 5e+00 | ok |
| `pls_diagnostic_dmodx` | PLS Distance-to-Model X (§9) | python / `(none)` - | ✗ | — | 5e+00 | no_python_reference |
| `pls_diagnostic_dmodx` | PLS Distance-to-Model X (§9) | R / `mdatools` 0.15.0 | ✓ | 1.220e+00 | 5e+00 | ok |
