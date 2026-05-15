# Phase 31h — Batch 8: §13 calibration transfer + missing-aware NIPALS

**Status:** shipped — local tag `phase-31h-batch-8-calibration-transfer`
(`0.78.0+abi.1.8.0`).

## Methods (all paper-only)

| Method | Internal kernel | Public C ABI entry | Paper |
|--------|-----------------|--------------------|-------|
| PDS | `fit_pds` | `p4a_pds_fit` | Wang et al. 1991 |
| DS | `fit_ds` | `p4a_ds_fit` | Wang et al. 1991 |
| MIR-PLS | `fit_mir_pls` | `p4a_mir_pls_fit` | Sjöblom et al. 1998 |
| Missing-aware NIPALS | `fit_missing_aware_nipals` | `p4a_missing_aware_nipals_fit` | Nelson et al. 1996 |

## Why paper-only

Despite extensive search of CRAN and PyPI:

- R `prospectr` 0.2.8 ships SNV / MSC / Shenk-West but **not** Wang et
  al. PDS/DS calibration transfer.
- `chemometrics` and related R packages either don't ship PDS or
  failed to install on this host (deep transitive deps).
- Python `pyChemometrics` is not on PyPI; `prospectpy` doesn't exist.
- R `pls` handles missing data via row deletion (`na.exclude`), not
  iterative imputation in the NIPALS loop.
- MIR-PLS as in Sjöblom 1998 has no widely-installable port; related
  inverse-regression PLS variants exist in proprietary chemometrics
  software only.

All four methods are therefore marked `paper-only` in the parity gate:
the runner smoke-fits the pls4all entry point and records the
canonical citation as the reference.

## ABI delta

- 4 new public symbols, total 148 (sorted, all `p4a_*` prefixed).
- C ABI minor 7 → 8.
- `ldd` audit: only libc/libstdc++/libgcc/libm/loader.

## Local gate

- 256 internal C++ tests (dev-release, local-asan-ubsan-gcc, local-ubsan-gcc).
- ABI symbol diff: clean.
- `git diff --check`: clean.
- Parity gate: 13 external PASS, 13 paper-only smoke PASS, 0 numpy.
