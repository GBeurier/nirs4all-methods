# Phases 3+4 combined — Codex code + architecture review transcript

**Date**: 2026-05-18
**Reviewer**: Codex (via `general-purpose` agent)
**Scope**: Phases 3 (stateful scatter) + 4 (derivatives & smoothing). Every-two-phases architectural review cadence.
**Verdict**: **ACCEPT.**

## Phase 1+2 Codex follow-through — all confirmed honoured

- **R1**: Test Runner renamed to `chemometrics4all` ✅
- **R2**: `parity/python_generator/pyproject.toml` with numpy 1.26.4 + scipy 1.11.4 + sklearn 1.4.2 + PyWavelets 1.6.0 + pybaselines 1.1.4 ✅
- **R3**: c4a.h trimmed from 2208 → 680 lines (Phase 3), now 831 (Phase 4 added §10 banner) ✅
- **R4**: `c4a_array_*` orphan declarations gone (one residual comment explaining the removal) ✅
- **R5**: `cpp/src/c_api/c4a_linux.map` linker version script in place with `c4a_*` filter ✅
- **`_fit/_transform` ABI contract** implemented uniformly across all 4 Phase 3 stateful ops ✅
- **No per-operator `_to_bytes/_from_bytes`** symbols leaked ✅

## Major architectural recommendations

### M1. Promote `padding.{c,h}` + Householder QR to `core/common/linalg.{c,h}` at Phase 5 OPENER

Phase 5 will need a new `banded_solver.{c,h}` (banded LDLT for 2nd-order penalty matrices in AsLS / airPLS / arPLS). Whittaker baselines reuse the same penalty (`D^T diag(w) D`) and compose naturally with QR. pls4all's `apply_asls` allocates a dense `cols × cols` matrix per row per iteration — that's the cost trap chemometrics4all should not inherit.

**Action**: Phase 5 opener (BEFORE any AsLS code lands) extracts:
- `cpp/src/core/common/linalg.{c,h}` — Householder QR (currently 3 copies: emsc.c, savitzky_golay.c, sg_fit_edge), shared.
- `cpp/src/core/common/banded_solver.{c,h}` — banded LDLT for the 2nd-order penalty (`D^T D`) matrix.

Doing it mid-phase risks rewriting working AsLS code. The "wait for 4th copy" deferral in `savitzky_golay.c:69-70` was correct logic but the trigger isn't "4th copy", it's "first phase whose math doesn't fit dense O(N²)" — Phase 5.

### M2. Reclassify `c4a_pp_derivate_*` as stateful with input-shape memory

`derivate.c:54-68` stores `cols` at fit time and `_apply` returns `C4A_ERR_SHAPE_MISMATCH` when `cols != state->cols`. That IS state — it enforces `_transform` matches the shape of `_fit`. The c4a.h §9 doc currently says "`_fit` is a no-op" — inaccurate.

**Action**: Update c4a.h §9 NorrisWilliams banner to honestly state that Derivate stores the fit-time cols for shape validation. OR drop the shape check (the output-cols helper already lets callers compute the expected `out` width). Option (a) preferred (simpler, matches actual behaviour).

### M3. LogTransform `_fit/_transform` split needs concrete target

`log_transform.c:76-88` recomputes `auto_offset` on every `_transform`, meaning (1) two transforms on different X don't share a baseline, (2) inference-time transform on a single new sample gets a *fresh* fitted offset rather than the one learned from training. Documented in c4a.h:484-487 and DEFERRALS lines 13+22 but with open-ended target ("Phase 4+"). Now 3 phases behind.

**Action**: Phase 5 brief MUST either (a) take the `_fit/_transform` split for LogTransform alongside the AsLS family, or (b) reaffirm a concrete later target with rationale. Risk of indefinite deferral with bindings shipping the per-call recompute frozen into a downstream API.

## Code quality recommendations applied

| # | Recommendation | Status |
|---|---|---|
| 1 | `docs/reviews/PHASES.md` Phase 4 row empty | ✅ Updated with verdict + tests + ABI |
| 2 | `docs/reviews/DEFERRALS.md` missing Phase 4 entries (QR helper extraction, JSON parser duplication) | ✅ Added |
| 3 | `docs/reviews/phase-4/opus-post.md` missing | ✅ Created (alongside this codex-post.md) |
| 4 | c4a.h §10 banner missing validation constraint docs (NorrisWilliams `delta != 0`, FirstDerivative `edge_order ∈ {1,2}`) | ⏳ Deferred — minor docs polish |

## Code quality observations — deferred

- **O1** JSON parser triplicated (~270 LOC × 3). Extract to `cpp/tests/fixture_parser.hpp` at Phase 5 opener (same window as M1 linalg promotion).
- **O2** `local_snv.c:71-95` rolls its own reflect/edge index resolution; now that `padding.{c,h}` exists could call `c4a_pad_resolve_index`. Defer until M1 cleanup window — don't shake byte-exact parity tests retroactively.
- **O3** Three Householder QR copies (EMSC + SG main + SG edge fit); addressed by M1.
- **O4** Phase 4 fixtures are 4.4–5.2 MB each (savgol, gaussian). Compress at Phase 5 alongside LogTransform split decision (M3) — same toolchain choice.
- **O5** Smoke tests for Phase 4 lack error-path coverage. Wrappers do enforce `require_rowmajor_f64` aggressively but no test exercises the rejection branches. Phase 5 brief should make error-path smoke tests a first-class requirement (cheaper than retrofitting Phase 2/3/4).

## Process & workflow recommendations

### P1. Codify review order in CONTRIBUTING.md

**Order**: Opus reviews each phase pre-commit; PHASES.md / DEFERRALS.md / opus-post.md commit WITH the phase code; Codex reviews every two phases post-commit against the latest tag.

### P2. Vendor pybaselines reference at Phase 5 opener (not Phase 5 implementation)

pybaselines 1.1.4 pinned in pyproject.toml but a moving upstream target. ~500 LOC of frozen NumPy reference (algorithm comments) under `parity/python_generator/src/parity/pybaselines_ref/`:
- Phase 5 implementation doesn't depend on a moving target.
- Future "improvements" to arPLS convergence in pybaselines don't silently shift the chemometrics4all parity floor.
- Algorithm docs cite the vendored ref directly.
~1 day of work; Phase 5 OPENER work, not Phase 5 implementation work.

### P3. Phase 5 brief should pin banded-solver design upfront

`pls4all/cpp/src/core/pipeline.cpp:766-813` shows the cost trap: `penalty(cols × cols)` allocated and copied per iteration per row. For NIRS spectra (1000–2000 wavelengths × 10–20 iterations), this is multi-MB allocator churn.

Phase 5 brief must lock:
- (a) banded storage layout (3-diagonal for D²ᵀD)
- (b) LDLT vs Cholesky (LDLT supports indefinite W which arPLS produces near peaks)
- (c) in-place reweighting (avoid per-iteration penalty alloc)
- (d) per-row state cleanup

Without these in the brief, Phase 5 risks two implementation drafts.

## Things done well

- Phase 3 → Phase 4 stylistic continuity excellent. Engine layering uniform, opaque-handle pattern uniform, status-code semantics uniform.
- `core/common/padding.{c,h}` is the right primitive at the right scope: pure index resolution, no allocation, sentinel-based constant mode.
- Static_asserts at c4a.h cover Phase 4's two new enums.
- Memory ownership consistency: every `_state_new` allocates, every wrapper `_create` allocates the public handle, both free paths handle partial-allocation cleanup.
- Symbol-list discipline mature: 94 symbols match expected_symbols_linux.txt byte-for-byte.
- Acknowledgement in `savitzky_golay.c:69-70` is the right deferral discipline — explicit comment, explicit trigger, no premature abstraction.
- Architectural runway: pls4all shipped one AsLS variant on dense O(N²); chemometrics4all is positioned to ship 10 baseline ops on banded infrastructure. **This is the opportunity to make Phase 5 a *better* implementation, not a port.**

## Open Phase 5 entry checklist (acceptance criteria)

Before any AsLS / airPLS / arPLS code lands in Phase 5:

- [ ] `cpp/src/core/common/linalg.{c,h}` extracted (Householder QR shared by EMSC + SG + SG-edge).
- [ ] `cpp/src/core/common/banded_solver.{c,h}` skeleton with banded LDLT for 3-diagonal matrices.
- [ ] `parity/python_generator/src/parity/pybaselines_ref/` vendored frozen reference (~500 LOC NumPy + algorithm comments).
- [ ] `cpp/tests/fixture_parser.hpp` extracted (3 test files dedup'd).
- [ ] Phase 5 brief explicitly locks banded storage layout + LDLT choice + in-place reweighting + per-row state cleanup.
- [ ] LogTransform `_fit/_transform` split decision made explicit (apply in Phase 5 OR reaffirm concrete later target).
- [ ] Error-path smoke tests made a first-class requirement.
- [ ] `c4a_pp_derivate_*` doc update (state honestly that input-shape is memoized).
