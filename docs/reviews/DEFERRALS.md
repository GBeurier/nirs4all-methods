# Cross-Phase Deferral Tracker

Items intentionally left to a future phase. Each entry records:
* **From**: the phase where the item was identified.
* **Topic**: short description.
* **Target**: the phase scheduled to address it (or "TBD" if still open).
* **Status**: open / closed.

## Open

| From | Topic | Target | Status |
| --- | --- | --- | --- |
| 2 | `LogTransform` `_fit/_transform` split — the Phase 2 implementation recomputes auto_offset on every `_transform`. A proper fit-time offset belongs to the stateful contract (Phase 3) but was deferred to keep Phase 2 stateless across all 7 operators. | Phase 4+ (scaling family) | open |
| 0/2 | Reintroduce `c4a_array_t` opaque type + 4 accessors. Removed in Phase 3 as orphan declarations with no implementations. Will return when an operator genuinely needs variable-shape output (e.g. CARS / MCUVE / IRF feature selection). | Phase 9 | open |
| 0 | Operator `_state_to_bytes / _state_from_bytes` serialization helpers. Defer until pipeline-level serialization is designed end-to-end so we don't lock in a per-operator format prematurely. | Phase 11 (splitters / pipeline serialization) | open |
| 0 | CI sanitizer preset (ASAN + UBSAN). Disabled in Phase 0 to keep the bootstrap minimal. Re-enable when the operator surface stabilises enough that flaky sanitizer noise won't dominate review effort. | Phase 5 | open |
| 0 | Fixture compression (gzip / zstd) for the larger parity JSON blobs. Current per-fixture size is 500 kB–1 MB; gzip would compress 5–10×. Not urgent until the fixtures directory exceeds the GitHub LFS / repo-size budget. | Phase 6 | open |
| 0 | Build under sanitizers and run all tests under the sanitizer-on preset. Same prerequisite as the CI sanitizer item above. | Phase 5 | open |
| 0 | `cpp/include/chemometrics4all/c4a_windows.def` for the Windows DLL surface (symbol export). Generated from the Linux version script. Required before the first Windows CI build hits ABI-check. | Phase 22 (Python binding) | open |
| 2 | Hand-rolled JSON parser in the test harness — already over-engineered for a temporary solution. Consider switching to `nlohmann/json` (header-only, BSL-licensed, ~24k LOC) once the fixture format stabilises. | Phase 4+ if maintenance cost grows | open |
| 3 | Per-row scaled MSC (nirs4all `MultiplicativeScatterCorrection(scale=True)`). The c4a Phase 3 implementation matches `scale=False`. The `scale=True` path adds a per-column StandardScaler pre-centering step which is rarely used in NIRS practice but exists in nirs4all. | TBD | open |
| 3 | `LogTransform` `_fit/_transform` split (carried forward from Phase 2). | Phase 5 (per Codex P3+4 review M3: must land in P5 or have concrete later target with rationale) | open |
| 3 | `c4a_pp_derivate_*` diverges from `nirs4all.Derivate` (uses `np.diff axis=1` instead of `np.gradient axis=0`). Documented in algorithm docs + this tracker. Future Phase considers shipping `c4a_pp_gradient_*` axis=0 operator for full nirs4all parity. | TBD | open |
| 4 | **Shared Householder QR helper.** Phase 3's EMSC + Phase 4's SavitzkyGolay (main coefs + interp edge fit) each carry a local `householder_qr` (~50 LOC × 3 copies = 150 LOC duplicated). Promote to `cpp/src/core/common/linalg.{c,h}` at Phase 5 opener — Phase 5's banded LDLT solver belongs in the same module. | Phase 5 (opener, before any AsLS code lands) | open |
| 4 | **Shared JSON fixture parser** for tests. `test_preprocessing_{stateless,stateful,smoothing}.cpp` each contain ~270 LOC of nearly-identical JSON parser + big-endian hex helpers. Extract to `cpp/tests/fixture_parser.hpp`. Defer to Phase 5 opener (same window as the linalg promotion). | Phase 5 (opener) | open |
| 4 | SG `cols < window_length` rejected for all modes (scipy only rejects for `interp`). Documented contract is consistent but stricter than scipy on non-interp modes. Cosmetic — add a docs note. | Phase 6+ | open |
| 4 | SG `interp` mode duplicates Vandermonde-QR per row. Performance, not correctness. Precompute edge QR factors at `_create` time when `mode == interp`. | Phase 5+ | open |
| 4 | **Fixture compression urgency moved up**. Phase 4 fixtures hit 4–5 MB each (savgol, gaussian). Phase 5 banded solvers will want longer signals. Originally targeted Phase 6, but Phase 5 is the natural alignment with the LogTransform `_fit/_transform` split decision (M3 from Codex Phases 3+4 review). | Phase 5 (alongside LogTransform split decision) | open |
| 4 | **Smoke tests lack error-path coverage** for all 5 Phase 4 ops (no NULL transform, no dtype-mismatch, no shape-mismatch). Wrappers do enforce these; tests don't exercise them. Phase 5 brief should establish a shared error-path smoke pattern as first-class requirement. | Phase 5 (brief requirement) | open |
| 4 | `c4a_pp_derivate_*` mis-documented as no-op `_fit` in c4a.h §9 banner. Engine actually stores `cols` at fit time and uses it for shape validation in `_transform`. Update the doc. | Phase 5 (cleanup) | open |
| 4 | **Pybaselines reference vendoring** — Codex P2 from Phases 3+4 review. Vendor a frozen NumPy reference for AsLS / airPLS / arPLS / etc. (~500 LOC + algorithm comments) under `parity/python_generator/src/parity/pybaselines_ref/` BEFORE Phase 5 implementation, not as part of it. Insulates against pybaselines upstream churn. | Phase 5 (opener, ~1 day work) | open |
| 4 | **Phase 5 brief must pin banded-solver design upfront**: banded storage layout (3-diagonal for D²ᵀD), LDLT vs Cholesky (LDLT supports indefinite W for arPLS), in-place reweighting, per-row state cleanup. pls4all's `apply_asls` is dense O(N²) — chemometrics4all's opportunity to do better. | Phase 5 (brief design) | open |
| 4 | **Codex review timing** must be codified in CONTRIBUTING.md (currently silent): Opus reviews each phase pre-commit; PHASES/DEFERRALS/opus-post.md commit with the phase; Codex reviews every two phases post-commit against the latest tag. | Phase 5 (CONTRIBUTING.md update) | open |
| 5a | **Per-iteration banded LDLT allocations**. AsLS/AirPLS/ArPLS engines call `c4a_banded5_factor` inside the iteration loop (50 mallocs/frees per row). Lift L/D buffers to row-scope via `c4a_banded5_factor_into(...)` for ~5–10× call-overhead reduction. | Phase 5b (banded solver refactor window) | open |
| 5a | **Duplicated `build_penalty_diagonals` / `relative_l2_diff`** across asls.c / airpls.c / arpls.c. Extract to `core/common/banded_solver.{c,h}` as `c4a_second_diff_penalty_pent5` + shared utils. | Phase 5b (banded solver refactor window) | open |
| 5a | **AirPLS LDLT-vs-spsolve compounding divergence** at `lam=1e7` and `tight_tol_short` (1e-6 abs / 1e-5 rel). Structural — banded LDLT vs SuperLU pivot strategies. Document residual-check in debug builds. | Phase 5b (or follow-up) | open |
| 5a | **ArPLS extreme-arg clip** at `>700` is more aggressive than `scipy.special.expit` (which preserves `1e-305` representable values). No practical effect on parity. Cosmetic two-branch logistic stabilisation. | Phase 5b+ | open |
| 5a | **Smoke tests for baseline ops only check `isfinite`**. Strengthen with bounded-value assertions on constant inputs (baseline≈y, output≈0). | Phase 5b cleanup | open |
| 7-21 (parallel batch) | **Per-phase Opus / Codex post-reviews skipped** because the ten worktrees landed in a single batch commit. Phase 7-21 review transcripts not produced. Follow-up review pass to run on the integrated state before Phase 22 (Python binding) starts. | Pre-Phase 22 | open |
| 9 / 19 | **Shared `c4a_svd_compact` signature is destructive** (`A` is consumed in place). Phase 19's `q_residuals.c` now copies `Xc` before calling the SVD; this is a workaround. Cleaner alternative is to widen the SVD signature to take a `const double*` input and copy internally. Both options change either the call sites or the helper API. | Phase 22 or later | open |
| 20 | **Transfer metrics keeps its private Jacobi-PCA** instead of using the shared `c4a_svd_compact`. Different internal API surface (`transfer_metrics.c` rotates a private symmetric Gram block, returns eigenvalues + eigenvectors, no thin-SVD path). Refactor would require either a richer shared helper or a Phase-20-specific eigen routine. | Phase 22 or later | open |
| 12 / 14 | **Filter operators' `c4a_filter_stats_t` is shared** (4 fields, 32 bytes). Phase 14 wrappers do not yet populate the `exclusion_rate` field — only the integer counts. Fixed at integration time by routing all three Phase 14 apply calls through a `fill_stats` helper. Code is committed but the fix never ran under the original Phase 14 worktree's review. | Pre-Phase 22 review | open |

## Closed

| From | Topic | Closed in | Resolution |
| --- | --- | --- | --- |
| 4 (carried from 3+4 Codex M1) | **Shared Householder QR helper extraction**. Was 3 copies in EMSC + SG-main + SG-edge. | Phase 5a opener | Extracted to `cpp/src/core/common/linalg.{c,h}` (`c4a_householder_qr` / `c4a_apply_qt` / `c4a_back_solve_R`). EMSC + SG migrated; all 61 prior tests pass at original tolerances. |
| 4 (carried from 3+4 Codex M1) | **Shared JSON fixture parser**. Was 3 copies of ~270 LOC each in test_preprocessing_{stateless,stateful,smoothing}.cpp. | Phase 5a opener | Extracted to `cpp/tests/fixture_parser.hpp` (header-only, templated). Net reduction: 1087 LOC eliminated from test files. |
| 4 (carried from 3+4 Codex M2) | **`c4a_pp_derivate_*` doc fix**. c4a.h §9 banner claimed no-op `_fit`; engine actually memoises `cols` for shape validation. | Phase 5a opener | §9 banner updated to honestly state input-shape memoization. |
| 4 (carried from 3+4 Codex P2) | **Vendor pybaselines frozen reference** at Phase 5 opener before any AsLS code lands. | Phase 5a opener | Created `parity/python_generator/src/c4a_parity_pybaselines_ref/` with frozen NumPy implementations of AsLS / AirPLS / ArPLS / Detrend matching `pybaselines==1.1.4`. Validation script at `parity/python_generator/scripts/validate_frozen_pybaselines_ref.py`. |
| 4 (carried from 3+4 Codex P3) | **Phase 5 brief banded-solver design pinned upfront** (storage layout, LDLT vs Cholesky, in-place reweighting, per-row cleanup). | Phase 5a brief | Roadmap `phase-5a-baseline-core.md` explicitly fixes pentadiagonal storage `(n, 3)`, symmetric LDLT (handles indefinite W via fallback), row-scope scratch (factor still iter-scope — tracked Phase 5b). |

| From | Topic | Closed in | Resolution |
| --- | --- | --- | --- |
| 0 | Pull nirs4all 0.8.x as the parity baseline. Pin via importlib in the fixture generator. | Phase 3 | Done — `parity/python_generator/scripts/generate_phase{2,3}_fixtures.py` load nirs4all directly from `/home/delete/nirs4all/nirs4all/nirs4all` via `importlib.util.spec_from_file_location`. Fixture generator dependencies are pinned in `parity/python_generator/pyproject.toml` (numpy 1.26.4, scipy 1.11.4, sklearn 1.4.2). |
| 0 | Codex-reviewed orphan PLS declarations in c4a.h (config, pipeline, model, operator_bank, gating_strategy, AOM, validation_plan, method_result, c4a_array_t accessors). Inherited from pls4all but never implemented in chemometrics4all. | Phase 3 | Removed entirely. c4a.h shrunk from 2208 to 680 lines. |
| 0 | `cpp/src/c_api/c4a_linux.map` linker version script for symbol scoping. | Phase 2 | Added as defense-in-depth on top of `-fvisibility=hidden`. |
| 2 | `parity/python_generator/pyproject.toml` with pinned reference dependencies. | Phase 2 | Done in Phase 2 cleanup. |
| 2 | Test runner suite name → `chemometrics4all` (was inherited as `pls4all`). | Phase 2 | Done. |

## Conventions

* Items are added in the phase where they are first articulated, never
  earlier. "Pre-planned" deferrals are out of scope for this document.
* Once an item ships, it moves from **Open** to **Closed** with the closing
  phase + a one-line resolution note. Do not delete entries — the audit
  trail across phases is the point.
