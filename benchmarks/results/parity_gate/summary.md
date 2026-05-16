# Parity gate report

Host: `Linux-6.6.114.1-microsoft-standard-WSL2-x86_64-with-glibc2.35`
pls4all: `0.97.0+abi.1.13.0`
Python: `3.11.14`
NumPy: `2.4.5`

Each method is compared against a Python reference and an R reference. Methods without a widely installable external reference are flagged `numpy-mirror` in the lib column.

| Method | Description | Reference (lang / lib / version) | Parity | RMSE rel | Tolerance | Status |
|--------|-------------|----------------------------------|--------|----------|-----------|--------|
| `vissa_select` | VISSA-PLS — Variable Iterative Space Shrinkage (§49) | paper / `paper-only` - | ✓ | — | 1e+00 | paper_only |
| `pso_select` | PSO-PLS — Binary Particle Swarm variable selection (§48) | paper / `paper-only` - | ✓ | — | 1e+00 | paper_only |
| `gpr_pls` | GPR-on-PLS — RBF Gaussian Process on PLS scores (§47) | python / `scikit-learn` 1.4.2 | ✓ | 2.259e-10 | 1e-08 | ok |
| `gpr_pls` | GPR-on-PLS — RBF Gaussian Process on PLS scores (§47) | R / `(none)` - | ✗ | — | 1e-08 | no_r_reference |
