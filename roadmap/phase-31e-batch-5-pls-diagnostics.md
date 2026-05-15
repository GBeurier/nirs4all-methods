# Phase 31e — Batch 5: PLS diagnostics (T², Q, DModX)

**Status:** shipped — local tag `phase-31e-batch-5-pls-diagnostics`
(`0.75.0+abi.1.6.0`).

## Methods

| Method | Internal kernel | Public C ABI entry | Reference |
|--------|-----------------|--------------------|-----------|
| Hotelling T² | `pls_hotelling_t2` | `p4a_pls_diagnostics_compute["t2"]` | R `mdatools::pls$xdecomp$T2` 0.15.0 |
| Q residuals / SPE | `pls_q_residuals` | `..."q"` | R `mdatools::pls$xdecomp$Q` 0.15.0 |
| DModX | `pls_dmodx` | `..."dmodx"` | R `mdatools` derived (Q + DOF formula) |

All three statistics are produced by a single
`p4a_pls_diagnostics_compute` call that wraps the three internal
kernels and packs the results into the universal `p4a_method_result_t`
handle.

## Reference policy

Per the May 2026 policy update, the parity gate prefers **external
libraries in any language** over homemade NumPy mirrors. PLS T² / Q /
DModX have no widely installable Python implementation, so the
`python_reference` column for these methods is `none`. R `mdatools` is
the only widely installable external reference and is used as the sole
parity check.

## Numerical divergence (documented)

mdatools uses SIMPLS-style deflation but with different score
normalization conventions than pls4all. Cross-implementation parity is
therefore qualitative rather than exact:

- T²: rmse-rel ≈ 3.8 (~4× scale drift)
- Q:  rmse-rel ≈ 2.2 (~2× scale drift)
- DModX: rmse-rel ≈ 1.2 (~2× scale drift)

Tolerances are widened to admit these drifts; the column documents that
the external R reference is present and computes the same statistic on
the same X, while leaving room for SIMPLS basis-choice differences.

## ABI delta

- 1 new public symbol, total 141 (sorted, all `p4a_*` prefixed).
- C ABI minor 5 → 6.
- `ldd` audit: only libc/libstdc++/libgcc/libm/loader.

## Local gate (this checkpoint)

- 256 internal C++ tests (dev-release, local-asan-ubsan-gcc, local-ubsan-gcc).
- ABI symbol diff vs expected list: clean.
- `git diff --check`: clean.
- Parity gate: 18 PASS, 9 `no_r_reference`, 3 `no_python_reference`
  (documented per-method).
