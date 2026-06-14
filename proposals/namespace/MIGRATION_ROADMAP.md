# Namespace migration roadmap (ML/DL, ABI 2.0) — Codex "before" gate

**Input contract:** `TARGET_NAMESPACE_ML.md` + `_target_ml_table.tsv` (208/208, validated).
**Strategy:** clean ABI break to **2.0.0** (rename canonical symbols, delete terse exports, no runtime
aliases — matches the repo's no-shim/no-dead-code rule). Ship a **migration guide**, not aliases.
**Discipline:** every wave gets a Codex review *after* (code/doc/tests/e2e); the biggest waves
(Phase 2 core ABI, Phase 6 downstream) also get a Codex *before* mini-plan. All work on a branch
`feat/namespace-ml-abi2`; rollback = drop the branch.

**Gate status — Codex `GO-WITH-CHANGES`** (`codex_review_roadmap.md`). Amendments applied:
(1) **WVC split → 209 methods**; (2) `utils` removed → `transform.signal_conversion`, `lowlevel.h`
added, `ensemble.aom` flattened; (3) ABI 2.0 must also update `N4M_ABI_VERSION_*`, Python ABI
constants, SONAME, `.github/workflows/abi-check.yml`, and `cpp/src/c_api/n4m_linux.map` `N4M_1→N4M_2`;
(4) **core ABI + all in-tree bindings = ONE merge unit** (no merge after Phase 2 alone — deleting
old exports breaks every binding until updated); (5) catalog keeps stable legacy `method_id`s + adds
`namespace/leaf/fq_name/c_surface/legacy_ids`, 10 zero-ABI rows → `c_surface: none`; (6) the per-phase
green-gate commands are taken verbatim from `codex_review_roadmap.md`; parity fixtures are NOT
numerically regenerated unless an algorithm changed.

## Downstream dependency map (measured)

| Repo | Coupling | What changes |
|---|---|---|
| `nirs4all-lite` | **heavy** (re-exporter; ~406 refs) | re-export surface, WASM symbol names, parity fixtures, tests, docs |
| `nirs4all-web` | **heavy** (~379 refs) | consumes lite/WASM; method names in UI + fixtures |
| `nirs4all-studio` | light (~28) | any direct method references + docs |
| `nirs4all-io` | light (~13) | references + docs |
| `nirs4all-datasets` | light (~9) | references + docs |
| `nirs4all` (main) | minimal (~3) | references + docs |

## Phase R — Reconcile (no rename yet) — Codex-after
- Resolve `selection.wvc` double symbol (`n4m_wvc_select` vs `n4m_wvc_threshold_select`).
- Decide C-symbol fate of the 10 zero-ABI methods + 16 `kind=null` rows (gain symbol vs stay Python-only → excluded from C headers).
- Add `namespace` + `leaf` fields to `catalog/methods/*.yaml` (drive everything from the catalog; `_target_ml_table.tsv` is the seed). Make a generator the single source of truth.
- **Verifiable here:** catalog validation (`catalog/scripts/validate.py`), Python.

## Phase 2 — `nirs4all-methods` core — Codex-before + Codex-after
- C ABI: split `n4m.h` into role headers (`transform.h`, `estimators.h`, `feature_selection.h`, `augmentation.h`, `model_selection.h`, `domain_adaptation.h`, `ensemble.h`, `compose.h`, `metrics.h`, `decomposition.h`, `outlier_detection.h`, `utils.h`) + 2nd-level sub-headers for `transform`/`estimators`/`augmentation`; remove the empty stub headers.
- C++ `c_api/`: rename exported symbols to `n4m_<role>_<method>_<verb>`; delete terse exports. Core numerics untouched (no parity change expected).
- Bump `N4M_ABI_VERSION_*`; `scripts/bump_version.sh`; regenerate `cpp/abi/expected_symbols_{linux,macos,windows}.txt` + `docs/abi/changes_log.md`; `n4m_cli`, doctest.
- **Green gate (sandbox-verifiable):** `cmake --preset dev-release` + `ctest` + ABI symbol diff (linux) + `n4m_cli --selfcheck/--abi-info`. macOS/Windows snapshots regenerated but CI-verified.

## Phase 3 — Python binding — Codex-after
- Replace flat `python.py` with `n4m/<role>/...` subpackages, public class names; regenerate `_ffi_decls.py` against the renamed symbols; update `__init__`, `sklearn/`, `aom/`, `moment/`.
- **Green gate:** `PYTHONPATH=… pytest bindings/python/tests`; ruff.

## Phase 4 — Other bindings — Codex-after (per binding)
- R (`bindings/r/n4m/`): `n4m_<role>_<method>()`; `Rscript bindings/r/test_parity.R`.
- MATLAB/Octave, Julia, JS-WASM, JNI/Android: regenerate dispatchers/decls against new symbols.
- **Note:** R/MATLAB/Octave/WASM/Android builds are largely **CI/env-bound, not fully sandbox-verifiable** — verify what runs locally (Octave smoke, Python), flag the rest for CI.

## Phase 5 — Web pages + docs — Codex-after  ← "notify the alternative web page"
- Regenerate `bench-data.json` / parity / `docs/benchmarks/*` so method IDs match the new namespace.
- Current dashboard (`render_matrix_dashboard.py`) + Sphinx docs (`docs/`) relabeled to the new tree.
- **`proposals/dashboard_v2/`**: re-brief the dashboard with `_target_ml_table.tsv`; relabel its method grouping/search to the Scheme-B namespace. (This is the explicit "notify the alt-page" step.)
- **Green gate:** `test_dashboard_contract`; docs build.

## Phase 6 — Downstream repos — Codex-before + Codex-after (per repo)
- `nirs4all-lite` → `nirs4all-web` first (heavy), then `nirs4all-studio`/`io`/`datasets`/`nirs4all` (light). Update code + docs; run each repo's own green gate for regression; add a cross-repo migration note.
- **Green gate:** each repo's test suite (Python verifiable; Rust/WASM/JS partly CI-bound).

## Cross-cutting
- **Migration guide** `docs/MIGRATION_ABI2.md` (old symbol/import → new) — shipped, not aliased.
- **Work log** `NAMESPACE_MIGRATION_LOG.md` updated each wave.
- **Verifiability honesty:** sandbox proves C++/Python/catalog/docs; R/MATLAB/Octave/WASM/CUDA/Windows/macOS are flagged for CI — never reported as locally green when they weren't.

## Top risks (Codex)
parity-fixture stability · ABI snapshot regen on 3 platforms with FFI in lockstep · bench-data/doc
method-ID churn · downstream breakage (expected for 2.0 → guide) · the zero-ABI/kind=null backfill.
