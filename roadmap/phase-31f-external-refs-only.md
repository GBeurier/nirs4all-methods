# Phase 31f — Reference-policy refactor: external refs only

**Status:** shipped — local tag `phase-31f-external-refs-only`
(`0.76.0+abi.1.6.0`).

## Policy

Per user direction May 2026, parity-gate references are restricted to
**external libraries in any language**:

1. R packages on CRAN: `pls`, `spls`, `OmicsPLS`, `mdatools`, …
2. Python packages on PyPI: `scikit-learn`, …
3. (Future) MATLAB / Octave packages.

Hand-written NumPy mirrors are **not** accepted as parity references —
they re-implement the algorithm in the same project that ships it, so
they only catch transcription errors, not algorithmic divergences.

When a method has no widely installable external implementation, it is
marked **paper-only** with the canonical citation. The runner
smoke-fits the pls4all side (to confirm the C ABI entry point is
callable) but skips the numerical parity comparison.

## Changes

### Removed (~600 lines)

- `_numpy_simpls` helper.
- 9 `*NumpyReference` adapter classes in
  `benchmarks/parity_timing/registry.py`.

### Added

- `WeightedPlsSklearnReference` — sklearn 1.4.2 PLSRegression on
  sqrt(w)-prescaled centered data (mathematically equivalent to
  weighted PLS).
- `RidgePlsSklearnReference` — sklearn 1.4.2 PLSRegression on
  (X augmented with sqrt(λ)·I, Y augmented with zeros) — standard
  data-augmentation trick for L2-penalized PLS.
- `MethodSpec.paper_only` — citation string when no external reference
  exists.
- Runner `paper_only` status path: smoke-fit pls4all side and record
  the citation; do not fail the gate on missing parity.

### Parity table (post-refactor)

| Method | Reference | RMSE rel |
|--------|-----------|----------|
| `sparse_simpls` | R `spls` 2.3.2 | 2.51e-05 |
| `di_pls` | paper-only (Nikzad-Langerodi 2018) | — |
| `recursive_pls` | sklearn 1.4.2 + R `pls` 2.8.5 | 1.23e-02 / 1.23e-02 |
| `cppls` | paper-only (Indahl 2005) | — |
| `weighted_pls` | sklearn 1.4.2 | 9.28e-07 |
| `robust_pls` | paper-only (Cummins 1995) | — |
| `ridge_pls` | sklearn 1.4.2 | 1.02e-07 |
| `continuum_regression` | paper-only (Stone & Brooks 1990) | — |
| `n_pls` | paper-only (Bro 1996) | — |
| `kernel_pls_rbf` | paper-only (Rosipal & Trejo 2001) | — |
| `o2pls` | R `OmicsPLS` 2.1.0 | 4.54e-01 (qualitative) |
| `approximate_press` | paper-only (Eastment & Krzanowski 1982) | — |
| `pls_diagnostic_t2 / q / dmodx` | R `mdatools` 0.15.0 | 3.84 / 2.19 / 1.22 (qualitative) |

## Local gate (this checkpoint)

- 256 internal C++ tests (dev-release, local-asan-ubsan-gcc, local-ubsan-gcc).
- ABI symbol diff vs expected list: 141 (unchanged — no new symbols
  in this commit; pure parity-policy refactor).
- `ldd` audit: only libc/libstdc++/libgcc/libm/loader.
- `git diff --check`: clean.
- Parity gate: 12 external PASS, 7 paper-only smoke PASS, 0 numpy.
