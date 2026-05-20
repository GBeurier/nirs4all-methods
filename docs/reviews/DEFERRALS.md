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
| 0/2 | Reintroduce `n4m_array_t` opaque type + 4 accessors. Removed in Phase 3 as orphan declarations with no implementations. Will return when an operator genuinely needs variable-shape output (e.g. CARS / MCUVE / IRF feature selection). | Phase 9 | open |
| 0 | Operator `_state_to_bytes / _state_from_bytes` serialization helpers. Defer until pipeline-level serialization is designed end-to-end so we don't lock in a per-operator format prematurely. | Phase 11 (splitters / pipeline serialization) | open |
| 0 | CI sanitizer preset (ASAN + UBSAN). Disabled in Phase 0 to keep the bootstrap minimal. Re-enable when the operator surface stabilises enough that flaky sanitizer noise won't dominate review effort. | Phase 5 | open |
| 0 | Fixture compression (gzip / zstd) for the larger parity JSON blobs. Current per-fixture size is 500 kB–1 MB; gzip would compress 5–10×. Not urgent until the fixtures directory exceeds the GitHub LFS / repo-size budget. | Phase 6 | open |
| 0 | Build under sanitizers and run all tests under the sanitizer-on preset. Same prerequisite as the CI sanitizer item above. | Phase 5 | open |
| 0 | `cpp/include/n4m/n4m_windows.def` for the Windows DLL surface (symbol export). Generated from the Linux version script. Required before the first Windows CI build hits ABI-check. | Phase 22 (Python binding) | open |
| 2 | Hand-rolled JSON parser in the test harness — already over-engineered for a temporary solution. Consider switching to `nlohmann/json` (header-only, BSL-licensed, ~24k LOC) once the fixture format stabilises. | Phase 4+ if maintenance cost grows | open |
| 3 | Per-row scaled MSC (nirs4all `MultiplicativeScatterCorrection(scale=True)`). The n4m Phase 3 implementation matches `scale=False`. The `scale=True` path adds a per-column StandardScaler pre-centering step which is rarely used in NIRS practice but exists in nirs4all. | TBD | open |
| 3 | `LogTransform` `_fit/_transform` split (carried forward from Phase 2). | Phase 5 (per Codex P3+4 review M3: must land in P5 or have concrete later target with rationale) | open |
| 3 | `n4m_pp_derivate_*` diverges from `nirs4all.Derivate` (uses `np.diff axis=1` instead of `np.gradient axis=0`). Documented in algorithm docs + this tracker. Future Phase considers shipping `n4m_pp_gradient_*` axis=0 operator for full nirs4all parity. | TBD | open |
| 4 | **Shared Householder QR helper.** Phase 3's EMSC + Phase 4's SavitzkyGolay (main coefs + interp edge fit) each carry a local `householder_qr` (~50 LOC × 3 copies = 150 LOC duplicated). Promote to `cpp/src/core/common/linalg.{c,h}` at Phase 5 opener — Phase 5's banded LDLT solver belongs in the same module. | Phase 5 (opener, before any AsLS code lands) | open |
| 4 | **Shared JSON fixture parser** for tests. `test_preprocessing_{stateless,stateful,smoothing}.cpp` each contain ~270 LOC of nearly-identical JSON parser + big-endian hex helpers. Extract to `cpp/tests/fixture_parser.hpp`. Defer to Phase 5 opener (same window as the linalg promotion). | Phase 5 (opener) | open |
| 4 | SG `cols < window_length` rejected for all modes (scipy only rejects for `interp`). Documented contract is consistent but stricter than scipy on non-interp modes. Cosmetic — add a docs note. | Phase 6+ | open |
| 4 | SG `interp` mode duplicates Vandermonde-QR per row. Performance, not correctness. Precompute edge QR factors at `_create` time when `mode == interp`. | Phase 5+ | open |
| 4 | **Fixture compression urgency moved up**. Phase 4 fixtures hit 4–5 MB each (savgol, gaussian). Phase 5 banded solvers will want longer signals. Originally targeted Phase 6, but Phase 5 is the natural alignment with the LogTransform `_fit/_transform` split decision (M3 from Codex Phases 3+4 review). | Phase 5 (alongside LogTransform split decision) | open |
| 4 | **Smoke tests lack error-path coverage** for all 5 Phase 4 ops (no NULL transform, no dtype-mismatch, no shape-mismatch). Wrappers do enforce these; tests don't exercise them. Phase 5 brief should establish a shared error-path smoke pattern as first-class requirement. | Phase 5 (brief requirement) | open |
| 4 | `n4m_pp_derivate_*` mis-documented as no-op `_fit` in n4m.h §9 banner. Engine actually stores `cols` at fit time and uses it for shape validation in `_transform`. Update the doc. | Phase 5 (cleanup) | open |
| 4 | **Pybaselines reference vendoring** — Codex P2 from Phases 3+4 review. Vendor a frozen NumPy reference for AsLS / airPLS / arPLS / etc. (~500 LOC + algorithm comments) under `parity/python_generator/src/parity/pybaselines_ref/` BEFORE Phase 5 implementation, not as part of it. Insulates against pybaselines upstream churn. | Phase 5 (opener, ~1 day work) | open |
| 4 | **Phase 5 brief must pin banded-solver design upfront**: banded storage layout (3-diagonal for D²ᵀD), LDLT vs Cholesky (LDLT supports indefinite W for arPLS), in-place reweighting, per-row state cleanup. pls4all's `apply_asls` is dense O(N²) — nirs4all-methods's opportunity to do better. | Phase 5 (brief design) | open |
| 4 | **Codex review timing** must be codified in CONTRIBUTING.md (currently silent): Opus reviews each phase pre-commit; PHASES/DEFERRALS/opus-post.md commit with the phase; Codex reviews every two phases post-commit against the latest tag. | Phase 5 (CONTRIBUTING.md update) | open |
| 5a | **Duplicated `build_penalty_diagonals` / `relative_l2_diff`** across asls.c / airpls.c / arpls.c. Extract to `core/common/banded_solver.{c,h}` as `n4m_second_diff_penalty_pent5` + shared utils. | Phase 5b (banded solver refactor window) | open |
| 5a | **AirPLS LDLT-vs-spsolve compounding divergence** at `lam=1e7` and `tight_tol_short` (1e-6 abs / 1e-5 rel). Structural — banded LDLT vs SuperLU pivot strategies. Document residual-check in debug builds. | Phase 5b (or follow-up) | open |
| 5a | **ArPLS extreme-arg clip** at `>700` is more aggressive than `scipy.special.expit` (which preserves `1e-305` representable values). No practical effect on parity. Cosmetic two-branch logistic stabilisation. | Phase 5b+ | open |
| 5a | **Smoke tests for baseline ops only check `isfinite`**. Strengthen with bounded-value assertions on constant inputs (baseline≈y, output≈0). | Phase 5b cleanup | open |
| 7-21 (parallel batch) | **Per-phase Opus / Codex post-reviews skipped** because the ten worktrees landed in a single batch commit. Phase 7-21 review transcripts not produced. Follow-up review pass to run on the integrated state before Phase 22 (Python binding) starts. | Pre-Phase 22 | open |
| 9 / 19 | **Shared `n4m_svd_compact` signature is destructive** (`A` is consumed in place). Phase 19's `q_residuals.c` now copies `Xc` before calling the SVD; this is a workaround. Cleaner alternative is to widen the SVD signature to take a `const double*` input and copy internally. Both options change either the call sites or the helper API. | Phase 22 or later | open |
| 20 | **Transfer metrics keeps its private Jacobi-PCA** instead of using the shared `n4m_svd_compact`. Different internal API surface (`transfer_metrics.c` rotates a private symmetric Gram block, returns eigenvalues + eigenvectors, no thin-SVD path). Refactor would require either a richer shared helper or a Phase-20-specific eigen routine. | Phase 22 or later | open |
| 11 (Codex M2 — post-batch review) | **Splitter opaque-handle naming inconsistent with the rest of the surface.** The nine Phase 11 splitters expose `n4m_split_*_t` opaque structs while every other category uses the `_handle_t` suffix (filters, preprocessing static handles, FCK static). The rename `n4m_split_*_t` → `n4m_split_*_handle_t` is source-breaking and therefore deferred from this review-cleanup commit. Must land before Phase 22 (Python binding) so the binding generator sees a uniform suffix scheme. | Phase 21.5 pre-binding-prep cleanup (before Phase 22) | resolved in phase-21.5-A |
| 12 (Codex M3 — post-batch review) | **Phase 12 Y-outlier filter exposes `_fit_get_mask`** while Phase 14 leverage / quality / composite filters expose `_fit` + `_apply` separately. The combined call contract is inconsistent across the filter category; the Python binding will want a uniform `_apply(...)` surface across all filters. Reconcile by splitting `n4m_filter_y_outlier_fit_get_mask` into `_fit` + `_apply` (ABI-breaking — drops one entry point, adds two). | Phase 21.5 pre-binding-prep cleanup (before Phase 22) | resolved in phase-21.5-A |
| 13 | **Robust-Mahalanobis MCD is a simplified shrinkage variant**, not the full FastMCD elemental-set search (Rousseeuw & Van Driessen 1999). Captured contamination-tolerant behaviour but doesn't reach sklearn's `MinCovDet` cluster-locating power. Cluster-locating FastMCD has substantial extra surface (elemental subsets, concentration steps, deterministic seeding) and is deferred to a v2 follow-up. | Phase 22+ | open |
| 13 | **LOF uses an approximate k-NN search**, not the brute-force kd-tree. Speed-up for medium n, but parity tolerance widens to 3 mask diffs at the percentile boundary. Bring back the brute-force fallback once a shared kd-tree primitive exists. | Phase 22+ | open |
| 15 | **PolynomialDrift / linear drift use a custom 1e-15 parity reference** rather than byte-exact pybaselines. pybaselines' polynomial baseline uses different basis conditioning that would require a separate reference module; the augmenter's deterministic intent is captured by the in-house frozen ref. | Future v2 | open |
| 16 | **GaussJitter / UnsharpMask use the augmenter's local Gaussian kernel** rather than the project-wide `n4m_pp_gaussian_*` operator. Behavioural intent is augmentation (not preprocessing), so we keep the local kernel; eventual unification with the shared smoothing path is deferred. | Phase 22+ | open |
| 17 | **InstrumentalBroadening** matches scipy `gaussian_filter1d` for the fixed-FWHM path. The use_fwhm_range path uses an internal randomised FWHM but does not pin a byte-exact ref (probabilistic by design). | Future v2 (parity reference if needed) | open |
| 18 | **SplineXSimplification / SplineCurveSimplification** both return `N4M_ERR_NOT_IMPLEMENTED` from their `_apply` entry points. The two operators require an `rng.choice(replace=False)` primitive (NumPy `Generator.choice`) that is not yet part of the public RNG surface. Symbols are exported (24 stubs) so the binding generator sees them and v2 lands the missing primitive plus the actual implementations without an ABI break. | v2 (when `rng.choice` lands) | open |

## Closed

| From | Topic | Closed in | Resolution |
| --- | --- | --- | --- |
| 8 / 26 | **EPO public ABI lacks `transform_with_d`.** | Phase 26 dashboard/gates cleanup | Added `n4m_pp_epo_transform_with_d(handle, X, d, d_len, out)`, exported it in the ABI symbol lists, bound it in Python, and moved the benchmark C++ EPO row to the d-aware path. |
| 4 (carried from 3+4 Codex M1) | **Shared Householder QR helper extraction**. Was 3 copies in EMSC + SG-main + SG-edge. | Phase 5a opener | Extracted to `cpp/src/core/common/linalg.{c,h}` (`n4m_householder_qr` / `n4m_apply_qt` / `n4m_back_solve_R`). EMSC + SG migrated; all 61 prior tests pass at original tolerances. |
| 4 (carried from 3+4 Codex M1) | **Shared JSON fixture parser**. Was 3 copies of ~270 LOC each in test_preprocessing_{stateless,stateful,smoothing}.cpp. | Phase 5a opener | Extracted to `cpp/tests/fixture_parser.hpp` (header-only, templated). Net reduction: 1087 LOC eliminated from test files. |
| 4 (carried from 3+4 Codex M2) | **`n4m_pp_derivate_*` doc fix**. n4m.h §9 banner claimed no-op `_fit`; engine actually memoises `cols` for shape validation. | Phase 5a opener | §9 banner updated to honestly state input-shape memoization. |
| 4 (carried from 3+4 Codex P2) | **Vendor pybaselines frozen reference** at Phase 5 opener before any AsLS code lands. | Phase 5a opener | Created `parity/python_generator/src/n4m_parity_pybaselines_ref/` with frozen NumPy implementations of AsLS / AirPLS / ArPLS / Detrend matching `pybaselines==1.1.4`. Validation script at `parity/python_generator/scripts/validate_frozen_pybaselines_ref.py`. |
| 4 (carried from 3+4 Codex P3) | **Phase 5 brief banded-solver design pinned upfront** (storage layout, LDLT vs Cholesky, in-place reweighting, per-row cleanup). | Phase 5a brief | Roadmap `phase-5a-baseline-core.md` explicitly fixes pentadiagonal storage `(n, 3)`, symmetric LDLT (handles indefinite W via fallback), row-scope scratch (factor still iter-scope — tracked Phase 5b). |
| 5a | **Per-iteration banded LDLT allocations** in AsLS / AirPLS / ArPLS. | Phase 5b refactor (landed during batch) | All three engines (`asls.c`, `airpls.c`, `arpls.c`) now call `n4m_banded5_factor_into(L_buf, D_buf, ...)` against row-scope scratch buffers hoisted before the iteration loop — confirmed at `asls.c:91-99,115-116`, `airpls.c:126`, `arpls.c:109`. No more per-iteration mallocs in the reweighting loop. |
| 12 / 14 | **Phase 14 `n4m_filter_stats_t.exclusion_rate` not populated.** | Phase 14 integration | All three Phase 14 apply paths (leverage, quality, composite) route through `fill_stats` in `cpp/src/c_api/c_api_filters_meta.cpp:95-103` which sets `exclusion_rate = nx / ns` whenever `ns > 0`. Verified at apply sites `:201, :283, :412`. |

| From | Topic | Closed in | Resolution |
| --- | --- | --- | --- |
| 0 | Pull nirs4all 0.8.x as the parity baseline. Pin via importlib in the fixture generator. | Phase 3 | Done — `parity/python_generator/scripts/generate_phase{2,3}_fixtures.py` load nirs4all directly from `/home/delete/nirs4all/nirs4all/nirs4all` via `importlib.util.spec_from_file_location`. Fixture generator dependencies are pinned in `parity/python_generator/pyproject.toml` (numpy 1.26.4, scipy 1.11.4, sklearn 1.4.2). |
| 0 | Codex-reviewed orphan PLS declarations in n4m.h (config, pipeline, model, operator_bank, gating_strategy, AOM, validation_plan, method_result, n4m_array_t accessors). Inherited from pls4all but never implemented in n4m. | Phase 3 | Removed entirely. n4m.h shrunk from 2208 to 680 lines. |
| 0 | `cpp/src/c_api/n4m_linux.map` linker version script for symbol scoping. | Phase 2 | Added as defense-in-depth on top of `-fvisibility=hidden`. |
| 2 | `parity/python_generator/pyproject.toml` with pinned reference dependencies. | Phase 2 | Done in Phase 2 cleanup. |
| 2 | Test runner suite name → `nirs4all-methods` (was inherited as `pls4all`). | Phase 2 | Done. |

## Conventions

* Items are added in the phase where they are first articulated, never
  earlier. "Pre-planned" deferrals are out of scope for this document.
* Once an item ships, it moves from **Open** to **Closed** with the closing
  phase + a one-line resolution note. Do not delete entries — the audit
  trail across phases is the point.
