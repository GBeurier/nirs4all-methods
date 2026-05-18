# Phase 5b — Baseline correction (rest) + LogTransform split

## Goal

Complete the baseline-correction family started in Phase 5a, refactor the iterative LDLT pattern, and close the LogTransform deferral carried since Phase 2.

## Operators (6 baseline + 1 split + 1 refactor)

| Operator | Algorithm | Type |
|----------|-----------|------|
| **ModPoly** | Iterative polynomial baseline; replace `y` with `min(y, poly_fit)` each iteration | iterative, dense QR |
| **IModPoly** | Improved ModPoly with σ-based stopping criterion (Lieber 2003 / Gan 2006) | iterative, dense QR |
| **SNIP** | Statistics-sensitive Non-linear Iterative Peak-clipping (Ryan 1988, Morháč 1997). For each width w in `1..max_half_window`, replace `y[i] ← min(y[i], (y[i-w] + y[i+w]) / 2)` | iterative, no linear algebra |
| **RollingBall** | Morphological rolling-ball baseline (Kneen & Annegarn 1996). Min over window then max over window | non-iterative, morphological |
| **IAsLS** | Improved AsLS (He 2014). Pre-fit polynomial baseline then run AsLS-style reweighting on residual | iterative, banded LDLT |
| **BEADS** | Baseline Estimation And Denoising with Sparsity (Ning, Selesnick 2014). Simplified version: TV-regularised baseline via banded solve | iterative, banded |

## Refactor + carry-forward

| Item | Status from earlier reviews |
|---|---|
| **LogTransform `_fit/_transform` split** | Opened Phase 2, deferred Phase 3, Phase 4 Codex M3 said "must land Phase 5". Now lands. |
| **`c4a_banded5_factor_into(...)` (in-place factor)** | Phase 5a Opus #10 — lifts L/D buffers to row-scope, saves max_iter mallocs per row. |
| **Shared `c4a_second_diff_penalty_pent5(n, lam, m, s1, s2)` in banded_solver** | Phase 5a Opus #11 — dedup penalty builder across AsLS / AirPLS / ArPLS / IAsLS / BEADS. |
| **Shared `c4a_relative_l2_diff(w_old, w_new, n)` helper** | Phase 5a Opus #11 — dedup convergence check across AsLS / ArPLS / IAsLS. |
| **AirPLS smoke test strengthening** (bounded-value on constant input) | Phase 5a #7. |
| **CONTRIBUTING.md review order codification** | Phase 4 Codex P1. |

## C ABI surface added

6 baseline ops × 3 symbols (`_create / _destroy / _transform`) = **18 new symbols**.

LogTransform split adds `c4a_pp_log_fit` (and changes `_transform` semantics slightly — `_transform` reads `_fitted_offset` instead of recomputing). **No new ABI symbol** (just one new function alongside existing) — well, technically `_fit` is new, so +1. Net Phase 5b: **+19 symbols**.

Total: 106 → **125 symbols**. ABI 1.5.0 → 1.6.0.

```c
/* §12 — Phase 5b Baseline correction (rest) */
typedef struct c4a_pp_modpoly_handle_t       c4a_pp_modpoly_handle_t;
typedef struct c4a_pp_imodpoly_handle_t      c4a_pp_imodpoly_handle_t;
typedef struct c4a_pp_snip_handle_t          c4a_pp_snip_handle_t;
typedef struct c4a_pp_rolling_ball_handle_t  c4a_pp_rolling_ball_handle_t;
typedef struct c4a_pp_iasls_handle_t         c4a_pp_iasls_handle_t;
typedef struct c4a_pp_beads_handle_t         c4a_pp_beads_handle_t;

C4A_API c4a_status_t c4a_pp_modpoly_create(c4a_pp_modpoly_handle_t** out,
                                            int32_t polyorder, int32_t max_iter, double tol);
C4A_API void         c4a_pp_modpoly_destroy(c4a_pp_modpoly_handle_t* h);
C4A_API c4a_status_t c4a_pp_modpoly_transform(...);

/* ... same shape for IModPoly, SNIP, RollingBall, IAsLS, BEADS ... */

/* §8 update — LogTransform stateful split */
C4A_API c4a_status_t c4a_pp_log_fit(c4a_pp_log_handle_t* h, c4a_matrix_view_t X);
/* _transform behavior change: now reads handle->_fitted_offset (set by _fit
 * if auto_offset=1) instead of recomputing per call. */
```

## Parity tolerances

| Operator | Reference | Abs tol | Rel tol |
|----------|-----------|---------|---------|
| ModPoly | Frozen NumPy ref | 1e-9 | 1e-10 |
| IModPoly | Frozen NumPy ref | 1e-9 | 1e-10 |
| SNIP | Frozen NumPy ref | 1e-12 | 1e-13 (pure arithmetic, no LDLT) |
| RollingBall | Frozen NumPy ref | 1e-12 | 1e-13 |
| IAsLS | Frozen NumPy ref | 1e-7 | 1e-8 |
| BEADS | Frozen NumPy ref | 1e-6 | 1e-7 (more iterations, larger sparsity terms) |

## Files to create

### Operators
- `cpp/src/core/preprocessing/baselines/modpoly.{c,h}`
- `cpp/src/core/preprocessing/baselines/imodpoly.{c,h}`
- `cpp/src/core/preprocessing/baselines/snip.{c,h}`
- `cpp/src/core/preprocessing/baselines/rolling_ball.{c,h}`
- `cpp/src/core/preprocessing/baselines/iasls.{c,h}`
- `cpp/src/core/preprocessing/baselines/beads.{c,h}`

### Frozen Python references (under `c4a_parity_pybaselines_ref/`)
- `modpoly.py`, `imodpoly.py`, `snip.py`, `rolling_ball.py`, `iasls.py`, `beads.py`

### Tests + fixtures + docs
- `cpp/tests/test_preprocessing_baselines.cpp` — extend with 12 new tests (6 smoke + 6 parity).
- `parity/python_generator/scripts/generate_phase5b_fixtures.py`
- `parity/fixtures/{modpoly,imodpoly,snip,rolling_ball,iasls,beads}_v1.json`
- `docs/algorithms/{modpoly,imodpoly,snip,rolling_ball,iasls,beads}.md`

### Refactors
- Update `cpp/src/core/common/banded_solver.{c,h}` — add `c4a_banded5_factor_into(...)`, `c4a_second_diff_penalty_pent5(...)`.
- Update `cpp/src/core/preprocessing/baselines/{asls,airpls,arpls}.c` — use the shared helpers and in-place factor.
- Update `cpp/src/core/preprocessing/scaling/log_transform.{c,h}` — add `_fit` step + change `_transform` to read cached `_fitted_offset`.

### Process
- `cpp/include/chemometrics4all/c4a.h` — append §12 + update §8 LogTransform banner.
- `cpp/src/c_api/c_api_preprocessing.cpp` — append wrappers.
- `CHANGELOG.md` — `[0.1.0+abi.1.6.0]` section.

## Acceptance criteria

- ✅ Build clean.
- ✅ Tests: 69 → 69 + 12 + 1 (LogTransform `_fit` semantic check) = **82/82**.
- ✅ ABI: 106 → **125 symbols** (or 124 if BEADS gets simpler). ABI bump 1.5.0 → 1.6.0.
- ✅ AsLS / AirPLS / ArPLS refactored to use shared helpers; parity tests still pass at original tolerances.
- ✅ LogTransform old per-call behaviour preserved via `auto_offset=1` default + new `_fit` step that captures the offset.
- ✅ Frozen pybaselines ref extended with 6 new modules.
- ✅ Opus + Codex (Phases 5a+5b combined) reviews completed and committed.

## Verification

```bash
cmake --build --preset dev-debug
./build/dev-debug/cpp/tests/chemometrics4all_tests   # 82/82
nm -D --defined-only build/dev-debug/cpp/src/libc4a.so.1.6.0 | awk '$2=="T" {print $3}' | sort -u | wc -l   # 125
```

## Next phase

[Phase 6 — Wavelets & denoising](phase-6-wavelets.md): Wavelet, Haar, WaveletDenoise + family.
