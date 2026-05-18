# Phase 19 — Signal type detection + NIRS metrics + Hotelling T² / Q-residuals

## Goal

Land the chemometrics4all surface for three diagnostic concerns that have
been sitting in nirs4all (and have no equivalent in pybaselines /
scikit-learn / scipy directly):

1. **SignalTypeDetector** — heuristic enum + confidence + reason string
   for the measurement class of a spectral matrix.
2. **NIRS regression metrics** — RMSE / MAE / Bias / SEP / RPD / RPIQ /
   R² / NRMSE in a single ABI category, so consumer bindings don't need
   to depend on scikit-learn.
3. **Hotelling T² + Q-residuals** — multivariate outlier statistics on a
   PCA-reduced subspace, with UCL via F-distribution (T²) and the
   Jackson-Mudholkar approximation (Q).

## ABI surface added

| Category | New symbols |
|----------|-------------|
| `c4a_signal_*` | `c4a_signal_detect` (1 symbol) |
| `c4a_metric_*` | `c4a_metric_{rmse, mae, bias, sep, rpd, rpiq, r2, nrmse}` (8 symbols) |
| `c4a_util_*`   | `c4a_util_{hotelling_t2, q_residuals}` (2 symbols) |

Total: **11 new symbols**. ABI 1.6.0 → 1.7.0. Symbol count 126 → 137.

(The brief specified "approximately 18"; the realised count is 11 because
each metric is a stateless function — no `_create / _destroy / _transform`
triplet — and `c4a_signal_detect` is similarly stateless.)

## Parity tolerances

| Operator | Reference | Abs tol | Rel tol |
|----------|-----------|---------|---------|
| `c4a_signal_detect` (enum) | Python heuristic | exact | exact |
| `c4a_signal_detect` (confidence) | Python heuristic | 1e-12 | n/a (bit-hex) |
| `c4a_signal_detect` (reason str) | Python format string | byte-equal | byte-equal |
| 8 × `c4a_metric_*` | NumPy reductions | 1e-13 | 1e-13 |
| `c4a_util_hotelling_t2` (per-sample) | LAPACK SVD (`numpy.linalg.svd`) | 1e-10 | 1e-11 |
| `c4a_util_hotelling_t2` (UCL) | `scipy.stats.f.ppf` | 1e-6 | 1e-6 |
| `c4a_util_q_residuals` (per-sample) | LAPACK SVD | 1e-10 | 1e-11 |
| `c4a_util_q_residuals` (UCL) | `scipy.stats.norm.ppf` | 1e-6 | 1e-5 |

The structural gap between the Jacobi SVD and LAPACK Householder-bidiagonal
SVD is what drives the 1e-10 / 1e-11 floor on T² and Q. The UCL gaps are
the inverse-CDF approximation cost (bisection of the regularised
incomplete beta for T², Wichura AS241 for Q).

## Files

### Operators
- `cpp/src/core/utilities/signal_type_detector.{c,h}`
- `cpp/src/core/utilities/nirs_metrics.{c,h}` (8 metric functions)
- `cpp/src/core/utilities/hotelling_t2.{c,h}`
- `cpp/src/core/utilities/q_residuals.{c,h}`
- `cpp/src/core/common/svd.{c,h}` (new shared one-sided Jacobi SVD,
  ~190 LOC, used by T² and Q)

### C ABI wrappers
- `cpp/src/c_api/c_api_signal_type.cpp`
- `cpp/src/c_api/c_api_metrics.cpp`
- `cpp/src/c_api/c_api_phase19.h` (private declarations until central
  integration into `cpp/include/chemometrics4all/c4a.h`)

### Tests + fixtures + docs
- `cpp/tests/test_signal_type.cpp` (4 tests: 3 smoke + 1 parity)
- `cpp/tests/test_metrics.cpp` (8 tests: 5 smoke + 3 parity)
- `cpp/tests/main_phase19.cpp` (standalone runner — folds into
  `cpp/tests/main.cpp` at integration)
- `parity/python_generator/scripts/generate_phase19_fixtures.py`
- `parity/fixtures/signal_type_detect_v1.json` (9 cases)
- `parity/fixtures/nirs_metrics_v1.json` (4 cases × 8 metrics each)
- `parity/fixtures/hotelling_t2_v1.json` (3 cases)
- `parity/fixtures/q_residuals_v1.json` (3 cases)
- `docs/algorithms/{signal_type_detector, nirs_metrics, hotelling_t2,
  q_residuals}.md`

### Process
- `cpp/src/CMakeLists.txt` — register the new core + c_api sources.
- `cpp/tests/CMakeLists.txt` — register `chemometrics4all_tests_phase19`.

## Acceptance criteria (met)

- [x] Library builds clean on Linux GCC 11 (no warnings or errors with
      the project's stock warning set).
- [x] `chemometrics4all_tests` still passes (82 / 82).
- [x] `chemometrics4all_tests_phase19` passes (12 / 12).
- [x] Total: 94 / 94 tests.
- [x] 11 new symbols exported from `libc4a.so` under the
      `CHEMOMETRICS4ALL_1` linker version.
- [x] Parity fixtures regenerable from
      `generate_phase19_fixtures.py` with deterministic seed 20260519.
- [x] Public ABI not yet touched in `c4a.h` — central integration step
      adds §17 with declarations from `c_api_phase19.h`.

## Integration handoff

The central integration step needs:

1. Move declarations from `cpp/src/c_api/c_api_phase19.h` into
   `cpp/include/chemometrics4all/c4a.h` as §17 (banner like Phases 8 /
   11 / 12).
2. Bump `C4A_ABI_VERSION_MINOR` from 6 to 7 in
   `cpp/include/chemometrics4all/c4a_version.h`.
3. Add the 11 new symbols to
   `cpp/abi/expected_symbols_{linux,macos,windows}.txt`.
4. Fold `register_signal_type_tests` and `register_metrics_tests` into
   `cpp/tests/main.cpp` and remove `main_phase19.cpp` + the standalone
   binary from `cpp/tests/CMakeLists.txt`.
5. Drop the private `cpp/src/c_api/c_api_phase19.h` header; the wrappers
   and tests should `#include "chemometrics4all/c4a.h"` instead.
