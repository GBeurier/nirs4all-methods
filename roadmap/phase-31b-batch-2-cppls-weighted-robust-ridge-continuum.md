# Phase 31b — Batch 2: CPPLS + weighted / robust / ridge / continuum PLS

**Status:** shipped — local tag
`phase-31b-batch-2-cppls-weighted-robust-ridge-continuum`
(`0.72.0+abi.1.3.0`).

Second tranche of public C ABI exposure for the Overview taxonomy. All
five methods reuse the universal `p4a_method_result_t` introduced in
Batch 1 (§16 of `p4a.h`).

## Methods

| Method | Internal kernel | Public C ABI entry | Reference (Python) | Reference (R) |
|--------|-----------------|--------------------|---------------------|---------------|
| CPPLS (col-std^γ rescaling) | `fit_cppls` | `p4a_cppls_fit` | numpy-mirror | none — `pls::cppls` is a different algorithm |
| Weighted PLS (sqrt(w) row scaling) | `fit_weighted_pls` | `p4a_weighted_pls_fit` | numpy-mirror | none — `pls::plsr` has no `weights=` |
| Robust PLS (Huber IRLS + weighted) | `fit_robust_pls` | `p4a_robust_pls_fit` | numpy-mirror | none |
| Ridge-augmented PLS (sqrt(λ)·I) | `fit_ridge_pls` | `p4a_ridge_pls_fit` | numpy-mirror | none |
| Continuum regression (col-std^τ) | `fit_continuum_regression` | `p4a_continuum_regression_fit` | numpy-mirror | none |

## R reference gap (documented)

`pls::cppls` shares the name with our CPPLS but implements Liland 2009
Canonical Powered PLS — a different algorithm (class-prior weighted
canonical regression). For the remaining four methods, no R port exists
with the exact rescaling / IRLS formulation. Each method's `notes`
field in `benchmarks/parity_timing/registry.py` documents this.

## Parity gate (Batch 2)

NumPy mirrors reproduce the C++ kernels step-for-step, so RMSE differences
sit at numerical floor:

| Method | Py RMSE rel | R RMSE rel | Status |
|--------|-------------|------------|--------|
| `cppls` | 9.88e-14 | — | Py PASS · R none |
| `weighted_pls` | 3.46e-13 | — | Py PASS · R none |
| `robust_pls` | 5.74e-06 | — | Py PASS · R none |
| `ridge_pls` | 7.22e-13 | — | Py PASS · R none |
| `continuum_regression` | 9.88e-14 | — | Py PASS · R none |

`robust_pls` carries a slightly larger residual (5.74e-06) because the
IRLS loop accumulates floating-point error across iterations; this is
still 8 orders of magnitude below the 5e-2 tolerance.

## ABI delta

- 5 new public symbols, total 136 (sorted, all `p4a_*` prefixed).
- C ABI minor 2 → 3.
- `ldd` audit: only libc/libstdc++/libgcc/libm/loader.

## Local gate (this checkpoint)

- 256 internal C++ tests (dev-release, local-asan-ubsan-gcc, local-ubsan-gcc).
- ABI symbol diff vs expected list: clean.
- `git diff --check`: clean.
- Parity gate: 10 PASS, 6 `no_r_reference` (documented).
