# Parity gate report

Host: `Linux-6.6.114.1-microsoft-standard-WSL2-x86_64-with-glibc2.35`
pls4all: `0.71.0+abi.1.2.0`
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

## Reference caveats

- **`sparse_simpls`** (python / `numpy-mirror`): Soft-threshold SIMPLS following Chun & Keles (2010).
- **`di_pls`** (python / `numpy-mirror`): Domain-invariant PLS — no widely installable external Python or R reference (nirs4all DiPLS is Dynamic Inner PLS; mdatools::ipls is Interval PLS). NumPy mirror of the same direction-projection formula as pls4all.
