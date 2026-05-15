# Phase 31c — Batch 3: N-PLS + non-linear kernel PLS

**Status:** shipped — local tag `phase-31c-batch-3-n-pls-kernel-pls`
(`0.73.0+abi.1.4.0`).

## Methods

| Method | Internal kernel | Public C ABI entry | Reference |
|--------|-----------------|--------------------|-----------|
| 3-way N-PLS (Bro 1996) | `fit_n_pls / predict_n_pls` | `p4a_n_pls_fit` | numpy-mirror |
| Non-linear kernel PLS (RBF / poly / sigmoid; Rosipal & Trejo 2001) | `fit_kernel_pls / predict_kernel_pls` | `p4a_kernel_pls_fit` | numpy-mirror |

## R reference gap (documented)

- N-PLS: MATLAB `nplstoolbox` (Bro) is the canonical reference; the
  algorithm is not in any widely installable R/Python port. R `multiway`
  ships PARAFAC / Tucker / SCA decompositions but no N-PLS regression.
- Kernel PLS (non-linear): R `pls` only implements the linear
  `kernel-algorithm` solver. `kernlab` covers KPCA / KSVM but not KPLS.
  sklearn has `KernelPCA` (PCA, not PLS).

Both NumPy mirrors reproduce the exact algorithm used by the C++ core,
so parity gates catch transcription / compiler bugs even without an
independent third-party reference.

## Parity gate (Batch 3 entries)

| Method | Py RMSE rel | R | Status |
|--------|-------------|---|--------|
| `n_pls` | 1.36e-07 | — | Py PASS · R none |
| `kernel_pls_rbf` | 2.32e-15 | — | Py PASS · R none |

Tolerance is the default 5e-2; both methods reach numerical floor.

## ABI delta

- 2 new public symbols; total 138 (sorted, all `p4a_*` prefixed).
- C ABI minor 3 → 4.
- `ldd` audit: only libc/libstdc++/libgcc/libm/loader.

## Local gate (this checkpoint)

- 256 internal C++ tests (dev-release, local-asan-ubsan-gcc, local-ubsan-gcc).
- ABI symbol diff vs expected list: clean.
- `git diff --check`: clean.
- Parity gate: 12 PASS, 8 `no_r_reference` (documented).
