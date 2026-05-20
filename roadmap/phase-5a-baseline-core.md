# Phase 5a — Baseline correction core: Detrend + AsLS family

## Goal

1. **Phase 5 opener cleanup** (per Codex Phases 3+4 review M1, P2):
   - Extract Householder QR → `cpp/src/core/common/linalg.{c,h}` (used by EMSC, SavitzkyGolay, SG-edge — 3 copies today).
   - Extract JSON fixture parser → `cpp/tests/fixture_parser.hpp` (3 copies × ~270 LOC today).
   - Vendor frozen `pybaselines==1.1.4` NumPy reference → `parity/python_generator/src/n4m_parity_pybaselines_ref/` (~500 LOC + algorithm comments).
   - Fix `n4m_pp_derivate_*` documentation in n4m.h §9 (engine stores `cols` at fit time — banner currently claims no-op).
2. **Implement 4 baseline operators**: Detrend, AsLS, AirPLS, ArPLS.
3. **Native banded LDLT solver** for 2nd-order penalty matrices in `cpp/src/core/common/banded_solver.{c,h}`.

## Out of scope (Phase 5b)

- ModPoly, IModPoly, SNIP, RollingBall, IAsLS, BEADS — saved for Phase 5b.
- LogTransform `_fit/_transform` split (carried forward; decided in Phase 5b alongside the remaining baseline ops which need similar fit-time logic).
- pls4all re-pull (Phase 26).

## Operators

| Operator | Algorithm | Parameters | Iterative? | Banded? |
|----------|-----------|------------|:---:|:---:|
| **Detrend** | Polynomial least-squares baseline subtraction along axis=1 | `polyorder: int` (default 1), `weight: optional mask` (deferred) | no | no (uses QR) |
| **AsLS** | Asymmetric Least Squares baseline (`Eilers & Boelens 2005`). Solves `(W + λ D²ᵀD) z = W y` iteratively; weights update as `p` if `y > z` else `1-p` | `lam: double` (default 1e6), `p: double` (default 0.001), `max_iter: int` (default 50), `tol: double` (default 1e-3) | yes | yes (3-diag) |
| **AirPLS** | Adaptive iteratively reweighted penalized least squares (`Zhang 2010`). Same banded solve; weights = `exp(t * |d_neg| / sum(|d_neg|))` where `d_neg = -d.clip(max=0)` | `lam: double` (default 1e6), `max_iter: int` (default 50), `tol: double` (default 1e-2) | yes | yes (3-diag) |
| **ArPLS** | Asymmetrically reweighted penalized least squares (`Baek 2015`). Same banded solve; weights = `logistic((y - z + 2σ⁻) / σ⁻)` where σ⁻ = std of negative residuals | `lam: double` (default 1e5), `max_iter: int` (default 50), `tol: double` (default 1e-3) | yes | yes (3-diag) |

All 4 operators are **stateless** w.r.t. fit — they recompute baseline per row per call. Same lifecycle as Phase 4 (no `_fit`).

## Banded LDLT solver design (Codex P3 pin)

For all three iterative ops (AsLS, AirPLS, ArPLS), the per-iteration system is:

```
(W + λ D²ᵀD) z = W y
```

where:
- `W = diag(w[0..n-1])` (diagonal, updated per iteration)
- `D²` is the 2nd-order difference matrix (n-2) × n: `D²[i,i] = 1, D²[i,i+1] = -2, D²[i,i+2] = 1`
- `D²ᵀD²` is **pentadiagonal** (5-diagonal) symmetric positive semi-definite

The system `(W + λ D²ᵀD²)` is pentadiagonal symmetric. We use **LDLT factorization** (not Cholesky) because AsLS' W can be near-singular and ArPLS' iteration can produce indefinite intermediate states — LDLT handles indefinite cases via 1×1 / 2×2 pivots.

### Storage layout

Symmetric pentadiagonal stored as `(n, 3)` upper-triangular row-major:
```
banded[i, 0] = main diagonal
banded[i, 1] = first super-diagonal (i-th element couples to i+1)
banded[i, 2] = second super-diagonal (i-th element couples to i+2)
```
For row i ∈ [0, n-1], unused elements at the trailing edge (`i+1 >= n` or `i+2 >= n`) are zero.

### Algorithm

LDLT of a symmetric pentadiagonal matrix: O(n) flops per factorization, O(n) per solve. Total iterative AsLS cost: O(n × max_iter) — vs pls4all's dense `(cols × cols)` which is O(n² × max_iter).

### API

```c
typedef struct n4m_banded5_t {
    int64_t n;
    double* L;  /* (n, 2) — sub-diagonals of L, row-major */
    double* D;  /* (n,) — diagonal */
} n4m_banded5_t;

n4m_status_t n4m_banded5_factor(const double* main_diag,
                                const double* super1, const double* super2,
                                int64_t n, n4m_banded5_t* out);
n4m_status_t n4m_banded5_solve(const n4m_banded5_t* fact, const double* b, double* x);
void         n4m_banded5_free(n4m_banded5_t* fact);
```

These are INTERNAL helpers (not in n4m.h public surface). Live under `cpp/src/core/common/banded_solver.{c,h}`.

## Pybaselines reference vendoring

Pinned `pybaselines==1.1.4` per `parity/python_generator/pyproject.toml`. To insulate the parity floor from upstream churn, vendor a frozen NumPy reference under `parity/python_generator/src/n4m_parity_pybaselines_ref/`:

- `asls.py`: explicit AsLS algorithm (Eilers 2005 paper formulation) in ~80 LOC of pure NumPy + scipy.sparse.
- `airpls.py`: AirPLS (Zhang 2010 paper) in ~80 LOC.
- `arpls.py`: ArPLS (Baek 2015 paper) in ~80 LOC.
- `detrend.py`: polynomial detrending via `np.polyfit` per row in ~30 LOC.
- `_banded.py`: scipy.sparse import + 2nd-order difference matrix builder.

These are imported by `parity/python_generator/scripts/generate_phase5a_fixtures.py` and validated **once** to bit-exact match `pybaselines==1.1.4` output (4 anchor cases per op), after which they become the frozen ground truth.

## Files to create

### Core infrastructure
- `cpp/src/core/common/linalg.{c,h}` — Householder QR (moved from emsc.c / savitzky_golay.c / sg-edge).
- `cpp/src/core/common/banded_solver.{c,h}` — pentadiagonal LDLT solver.
- `cpp/tests/fixture_parser.hpp` — shared JSON parser (extracted from 3 test files).

### Operators
- `cpp/src/core/preprocessing/baselines/detrend.{c,h}`
- `cpp/src/core/preprocessing/baselines/asls.{c,h}`
- `cpp/src/core/preprocessing/baselines/airpls.{c,h}`
- `cpp/src/core/preprocessing/baselines/arpls.{c,h}`

### Tests + fixtures + docs
- `cpp/tests/test_preprocessing_baselines.cpp` — 8 tests (4 smoke + 4 parity).
- `parity/python_generator/scripts/generate_phase5a_fixtures.py`
- `parity/fixtures/{detrend,asls,airpls,arpls}_v1.json`
- `docs/algorithms/{detrend,asls,airpls,arpls}.md`
- `roadmap/phase-5a-baseline-core.md` (this file)
- `docs/reviews/phase-5a/` — Opus + Codex transcripts

### Vendored references
- `parity/python_generator/src/n4m_parity_pybaselines_ref/{__init__,asls,airpls,arpls,detrend,_banded}.py`

## Files to modify

- `cpp/src/core/preprocessing/scatter/emsc.c` — replace local QR with `n4m_householder_qr` calls.
- `cpp/src/core/preprocessing/derivatives/savitzky_golay.c` — same.
- `cpp/tests/test_preprocessing_{stateless,stateful,smoothing}.cpp` — replace per-file JSON parser with `#include "fixture_parser.hpp"`.
- `cpp/include/n4m/n4m.h` — append §11 "Phase 5a — Baseline correction" (12 new symbols = 4 ops × 3); fix Derivate banner in §9 to honestly state input-shape memoization.
- `cpp/src/c_api/c_api_preprocessing.cpp` — append 12 wrappers.
- `cpp/src/CMakeLists.txt` — add new source files (4 ops + linalg + banded_solver).
- `cpp/tests/{main.cpp, CMakeLists.txt}` — register `register_preprocessing_baselines_tests`.
- `cpp/include/n4m/n4m_version.h` — bump MINOR 4 → 5.
- `cpp/abi/expected_symbols_{linux,macos,windows}.txt` — regenerate (94 → 106).
- `CHANGELOG.md` — Phase 5a section.

## ABI surface added

```c
/* §11 - Phase 5a Baseline correction (4 ops, 12 symbols) */
typedef struct n4m_pp_detrend_handle_t n4m_pp_detrend_handle_t;
N4M_API n4m_status_t n4m_pp_detrend_create(n4m_pp_detrend_handle_t** out,
                                            int32_t polyorder);
N4M_API void         n4m_pp_detrend_destroy(n4m_pp_detrend_handle_t* h);
N4M_API n4m_status_t n4m_pp_detrend_transform(const n4m_pp_detrend_handle_t* h,
                                               n4m_matrix_view_t X,
                                               n4m_matrix_view_t out);

/* AsLS, AirPLS, ArPLS — same shape, 3 symbols each */
typedef struct n4m_pp_asls_handle_t   n4m_pp_asls_handle_t;
typedef struct n4m_pp_airpls_handle_t n4m_pp_airpls_handle_t;
typedef struct n4m_pp_arpls_handle_t  n4m_pp_arpls_handle_t;

N4M_API n4m_status_t n4m_pp_asls_create(n4m_pp_asls_handle_t** out,
                                         double lam, double p,
                                         int32_t max_iter, double tol);
N4M_API void         n4m_pp_asls_destroy(n4m_pp_asls_handle_t* h);
N4M_API n4m_status_t n4m_pp_asls_transform(const n4m_pp_asls_handle_t* h,
                                            n4m_matrix_view_t X,
                                            n4m_matrix_view_t out);

N4M_API n4m_status_t n4m_pp_airpls_create(n4m_pp_airpls_handle_t** out,
                                           double lam,
                                           int32_t max_iter, double tol);
N4M_API void         n4m_pp_airpls_destroy(n4m_pp_airpls_handle_t* h);
N4M_API n4m_status_t n4m_pp_airpls_transform(const n4m_pp_airpls_handle_t* h,
                                              n4m_matrix_view_t X,
                                              n4m_matrix_view_t out);

N4M_API n4m_status_t n4m_pp_arpls_create(n4m_pp_arpls_handle_t** out,
                                          double lam,
                                          int32_t max_iter, double tol);
N4M_API void         n4m_pp_arpls_destroy(n4m_pp_arpls_handle_t* h);
N4M_API n4m_status_t n4m_pp_arpls_transform(const n4m_pp_arpls_handle_t* h,
                                             n4m_matrix_view_t X,
                                             n4m_matrix_view_t out);
```

12 new symbols. Total: 94 → **106**. ABI bump 1.4.0 → 1.5.0.

## Parity tolerances

| Operator | Reference | Abs tol | Rel tol | Notes |
|----------|-----------|---------|---------|-------|
| Detrend | Frozen NumPy ref + `np.polyfit` | 1e-11 | 1e-12 | Closed-form, dominated by QR-vs-SVD gap |
| AsLS | Frozen NumPy ref + scipy.sparse | 1e-7 | 1e-8 | Iterative; convergence-bound. Bit-exact factorization comparison would need `splu` floating-point parity which is OS-dependent |
| AirPLS | Frozen NumPy ref | 1e-7 | 1e-8 | Same |
| ArPLS | Frozen NumPy ref | 1e-7 | 1e-8 | Same |

## Acceptance criteria

- ✅ Build clean. Zero warnings.
- ✅ All 4 operators implement stateless lifecycle.
- ✅ Tests: 61 → 61 + 8 + (refactor tests still passing) = **69/69**.
- ✅ ABI: 94 → **106 symbols**. ABI 1.4.0 → 1.5.0.
- ✅ Linalg + banded_solver + fixture_parser extractions done; 3 QR copies + 3 JSON parser copies eliminated.
- ✅ Frozen pybaselines reference vendored and validated.
- ✅ Opus post-review at `docs/reviews/phase-5a/opus-post.md`.
- ✅ CI green, commit + push.

## Verification

```bash
cd /home/delete/nirs4all/nirs4all-methods
cmake --build --preset dev-debug
./build/dev-debug/cpp/tests/n4m_tests       # 69/69 pass
./build/dev-debug/cpp/cli/n4m_cli --selfcheck
nm -D --defined-only build/dev-debug/cpp/src/libn4m.so.1.5.0 | awk '$2=="T" {print $3}' | sort -u | wc -l   # 106
```

## Next phase

[Phase 5b — Baseline correction (rest)](phase-5b-baseline-rest.md): ModPoly, IModPoly, SNIP, RollingBall, IAsLS, BEADS — plus the LogTransform `_fit/_transform` split decision.
