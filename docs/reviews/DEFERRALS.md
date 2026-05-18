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

## Closed

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
