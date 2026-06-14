# Namespace migration — work log

**Goal:** reorganize all 208 `libn4m` methods under the **ML/DL pipeline namespace (Scheme B)**
across the C ABI, the C++ core public surface, every language binding, both web dashboards, and
all docs — then propagate the change to every downstream repo that depends on `nirs4all-methods`,
updating code + docs and proving no regression.

**Decision:** Scheme B (ML/DL) chosen by the maintainer. Reference: `proposals/namespace/NAMESPACE_PROPOSALS.md`
(`# Scheme B — ML / DL PIPELINE`), flat mapping `proposals/namespace/_ml_table.md`, ground-truth
inventory `proposals/namespace/_method_inventory.json` (208 ids, validated 208/208 mapped, no collisions).

**Review discipline (maintainer's rule):** every major step is reviewed by **Codex** *before*
(roadmap gate) and *after* (code / doc / tests / e2e gate). Multiple agents are used for execution.
The agent updating the **alternative web page** is explicitly briefed ("notified") with the final
namespace spec.

---

## Target namespace (Scheme B) — top level

`n4m.{ transform, decomposition, feature_selection, augmentation, estimators, ensemble,
model_selection, compose, domain_adaptation, outlier_detection, metrics, utils }`
— 12 top-level nodes, 39 leaf namespaces. Symbol convention `n4m_<role>_<method>_<verb>`;
Python mirrors sklearn (`from n4m.transform.scatter import SNV`,
`from n4m.estimators.regression.linear import Ridge`, ...).

---

## Phase plan

| # | Phase | Codex-before (roadmap) | Codex-after (review) | Status |
|---|-------|:---:|:---:|--------|
| 0 | Review the chosen ML table; finalize target spec | n/a | ✅ ml-table review | **done** |
| 1 | Master migration roadmap + downstream dependency map | ✅ GO-with-changes | n/a | **done** |
| 2 | Migrate `nirs4all-methods` core: C ABI headers + symbols, C++ registry, catalog | ☐ | ☐ | pending |
| 3 | Migrate bindings: Python, R, MATLAB/Octave, Julia, JS-WASM, JNI/Android | ☐ | ☐ | pending |
| 4 | Update both web dashboards (current + alternative `dashboard_v2`) + docs (Sphinx) | ☐ | ☐ | pending |
| 5 | Propagate to downstream repos (code + docs) + regression tests | ☐ | ☐ | pending |

> Execution of phases 2–5 is **held** until the running visualization workflow (`dashboard_v2`)
> completes, per maintainer instruction ("attends le workflow"). Phase 0 (namespace review) is
> independent of that workflow and proceeds now.

---

## Timeline

### 2026-06-13

- **Namespace proposals delivered** (prior workflow): `NAMESPACE_PROPOSALS.md` with Scheme A (stats)
  + Scheme B (ML/DL), both mapping all 208 methods 1:1; maintainer selected **Scheme B**.
- **Visualization workflow** (`wmff0y9nk`) still running — building the alternative parity/benchmark
  page `proposals/dashboard_v2/`. Migration execution is gated on its completion.
- **Visualization workflow result:** `build:dashboard` **FAILED** — *"Claude Fable 5 is currently
  unavailable."* The alternative page `proposals/dashboard_v2/` was never created. The namespace
  track of the same workflow **succeeded** (proposal + inventory complete, 208/208 validated).
- **Transport fix:** the Codex **MCP** call hung (~no response). Switched all Codex reviews to the
  documented `codex exec` **CLI** path (works: `codex-cli 0.139.0`), run **backgrounded with a hard
  timeout** so a stalled reviewer can never block the session again.
- **Model fix:** Fable is unavailable — **all agents now run on Opus** (dashboard included).

### Resume (2026-06-14) — two independent background tracks

- **Track A (namespace gate):** Codex reviewing Scheme B (the ML table) via `codex exec`
  → `/tmp/codex_ml_review.md`. Verdict + change list folded in here on return, then target spec is
  finalized and the migration roadmap written (Codex roadmap gate before any code mutation).
- **Track B (visualization):** rebuilding the alternative dashboard `proposals/dashboard_v2/` on
  **Opus** (self-contained, offline, no CDN), independent of the namespace work. Phase 4 will later
  relabel its method grouping to the final Scheme-B namespace — this is the point at which the
  "alternative web page" work is briefed ("notified") with the final namespace.

### Codex review of the ML table — verdict: **ADOPT WITH CHANGES** (transcript: `proposals/namespace/codex_review_ml_table.md`)

Folded the full change-list into the finalized contract **`proposals/namespace/TARGET_NAMESPACE_ML.md`**
(+ deterministic generator `_build_target_ml.py` → `_target_ml_table.tsv`, validated **208/208, 0 collisions**).
Headline deltas: split the 14-member `estimators.regression.linear` → `regression.{latent,regularized,robust,kernel,tensor,online,local,glm}`;
`pls_cox` → `estimators.survival`; redistribute the 18 AOM methods across `model_selection.{aom_search,aom_campaign}`,
`compose.aom_superblock`, `ensemble.aom`; EPO → `domain_adaptation.orthogonalization` (OSC stays in transform);
leaf renames (`pls_fit_simple→pls`, `kernel→kernel_pls`, `tensor_pls→n_pls`, `recursive→recursive_pls`,
`derivate→derivative`, `split_splitter→data_twinning`, diagnostics `model_selection→one_se_rule`); dissolved the `utils` grab-bag.
Final tree: **13 top-level · 51 leaf · 208 methods**.

**ABI strategy — Codex overruled the proposal: clean break, ABI 2.0** (rename canonical symbols, delete terse
exports, no runtime aliases, regen all 3 snapshots, remove stub headers; ship a migration guide). Recorded in the contract.

### Downstream dependency map (measured)
`nirs4all-lite` (heavy, re-exporter ~406) · `nirs4all-web` (heavy ~379) · `nirs4all-studio` (~28) ·
`nirs4all-io` (~13) · `nirs4all-datasets` (~9) · `nirs4all` (~3). → drives the Phase-6 order lite→web→rest.

### Roadmap written + at Codex gate
`proposals/namespace/MIGRATION_ROADMAP.md` (Phase R reconcile → core ABI → Python → other bindings →
web/docs → downstream, all on branch `feat/namespace-ml-abi2`, each wave Codex-reviewed after; core + downstream
also reviewed before). **Sent to Codex via `codex exec` for the "before" gate** → `/tmp/codex_roadmap_review.md`.
No code is mutated until this gate passes. (Track B: dashboard rebuild on Opus still running.)

### Track B (visualization) — **DONE** (Opus)

`proposals/dashboard_v2/` built + hardened: `index.html` + `styles.css` + `app.js` + `bench-data.js`
(full 2.9 MB payload inlined as `window.BENCH_DATA`) + `README.md`. Self-contained, **opens by
double-click (file://), zero network/CDN** (verified 0 external refs; system-font stack). 183 methods
as searchable cards in 18 categories; **binding-parity vs reference-parity shown as two distinct
gates**; per-method drawer with inline-SVG timing-vs-size curves (lin/log), multi-reference parity
table, ΔRMSE-vs-Jaccard handled correctly, NR vs n/a never coerced to a number; colorblind-safe
(color+glyph+text). Verify pass fixed a broken origin filter (offered `donor`/`external` but data is
`pls4all`/`n4m_donor`) + chip styling + `__reference` null-binding cells.
**Caveat:** no JS engine/browser in the sandbox → validated structurally + by a Python re-simulation
of the data model against the real payload, **not** by a real browser render. Open it in a browser to
confirm visuals. Phase 5 will relabel its grouping to the final ML namespace.

### Codex roadmap gate — verdict: **GO-WITH-CHANGES** (transcript: `proposals/namespace/codex_review_roadmap.md`)

Confirmed the ml-table fold-in is correct; required corrections now applied + regenerated:
- **WVC split** → `feature_selection.wrapper.wvc` + `…wvc_threshold` ⇒ catalog **208 → 209 methods**.
- **`utils` removed**: `signal_type_detector` → `transform.signal_conversion`. **`lowlevel.moments` kept** (+`lowlevel.h`). **`ensemble.aom` flattened** into `ensemble`.
- ABI 2.0 must also bump `N4M_ABI_VERSION_*`, Python ABI constants, SONAME, `abi-check.yml`, and `n4m_linux.map` `N4M_1→N4M_2`.
- **Core ABI + all in-tree bindings = one merge unit.** Catalog keeps stable legacy `method_id`s + adds `namespace/leaf/fq_name/c_surface/legacy_ids`. 10 zero-ABI rows → `c_surface: none`. No parity-fixture regen unless an algorithm changed.
- All Codex per-phase green-gate scripts verified present (`selftest.py`, `split_legacy_methods.py`, `validate.py`, `reconcile_abi.py`, `parity_timing/lockfile.py`, `regen_abi_snapshots.sh`, …).

Regenerated: **`_target_ml_table.tsv` = 209/209, 0 collisions, 12 top-level, 49 leaf.**
**Both Codex gates passed → plan locked, execution authorized** (clean ABI 2.0 break; reversible — all on a branch).

### Execution — Phase R (reconcile)
- Branch **`feat/namespace-ml-abi2`** cut.
- Phase R underway: add catalog `namespace/leaf/fq_name/c_surface/legacy_ids` from `_target_ml_table.tsv`,
  perform the WVC split, mark the 10 zero-ABI rows `c_surface: none`, then run the Phase-R green gate.
  Codex reviews Phase R *after*. Results recorded below.

### Phase R — **DONE, 6/6 gates green** (agent on `feat/namespace-ml-abi2`)

- `catalog/methods.yaml` (source): 5 fields on every method (`namespace/leaf/fq_name/c_surface/legacy_ids`), **WVC split**, `method_count: 209`.
- 209 split files regenerated via `split_legacy_methods.py --write`, incl. new `catalog/methods/selection.wvc_threshold.yaml`.
- `catalog/schema/method.json` + `method_v1.json`: 5 fields added (both are `additionalProperties:false`).
- `c_surface` from reconciled `abi_symbols`; **10 zero-ABI methods → `c_surface:"none"`**; legacy ids stable (no fixture/doc churn).
- **Green gate 6/6 PASS** (`_build_target_ml.py`, `selftest.py`, `split_legacy_methods.py --check`, `validate.py --strict-abi --check-references`, `reconcile_abi.py --check`, `parity_timing.lockfile --check`); baseline green first ⇒ no pre-existing & no introduced failures. ABI coverage 702/702.
- Caveat for Codex: `c_surface` schema uses `oneOf`, which the vendored validator skips (lenient).

→ **Codex combined gate** (Phase R after-review + Phase 2 before-gate) launched via `codex exec` → `/tmp/codex_phaseR_phase2.md`. No C-ABI code moves until it locks the symbol convention.

### Codex combined gate — Phase R after + Phase 2 before (transcript: `proposals/namespace/codex_phaseR_phase2.md`)

**Phase R after = NO-GO until one fix** (catalog data verified correct, incl. the WVC split & the 10 `c_surface:"none"`):
`catalog/abi_method_map.yaml` collapsed both WVC symbols under the `n4m_wvc` prefix. **Fixed:**
`selection.wvc → n4m_wvc_select` + added `selection.wvc_threshold → n4m_wvc_threshold_select`; and **tightened
`validate.py`** to enforce the five Phase-R field invariants (all required; `fq_name == n4m.<ns>.<leaf>`;
`c_surface == abi_symbols` else `"none"`). **Re-ran gates: `reconcile_abi` 566/566 ✅ · `validate --strict-abi
--check-references` PASS 209/209 ✅ · `split --check` ✅ → Phase R GO.**

**Phase 2 before = LOCKED:**
- C symbol role token = **top-level namespace**: `n4m_<role>_<leaf><tail>`, tails byte-for-byte. Examples:
  `n4m_ridge_fit→n4m_estimators_ridge_fit`, `n4m_pp_snv_create→n4m_transform_snv_create`,
  `n4m_cars_select→n4m_feature_selection_cars_select`, `n4m_pls_fit_simple→n4m_estimators_pls_fit`,
  `n4m_aug_gaussian_noise_apply→n4m_augmentation_gaussian_noise_apply`. Deeper namespace = headers/bindings only.
- Headers: `n4m.h` + 12 role headers + subheaders `transform`(10)/`estimators`(4)/`augmentation`(8) + `lowlevel.h`;
  **delete** the 9 category stubs + `context.h` + `pls.h` (no compat include).
- Lockstep files: catalog, headers, `cpp/src/c_api/*.cpp`, `n4m_version.h`→**2.0.0**, `n4m_linux.map` **N4M_1→N4M_2**,
  3 ABI snapshots, `regen_abi_snapshots.sh` (drop hardcoded N4M_1), `abi-check.yml` SONAME **libn4m.so.2**,
  Python `_ffi`/`_ffi_decls` + call sites, R/MATLAB/Octave/JS bindings, `docs/abi/*` + new `docs/MIGRATION_ABI2.md`.
- Rename map generated from catalog (`role=namespace.split('.')[0]`, `leaf`, tail from old symbol); skip the 10
  `c_surface:"none"`; fail on collision. **Core ABI + all in-tree bindings = one merge unit.**

### Phase 2 — NEXT (verifiable sub-waves on `feat/namespace-ml-abi2`)
2a rename-map generator (+validate vs `expected_symbols_linux.txt`) → 2b `c_api` symbol rename + version/map/snapshots
→ 2c header split (+delete stubs/`pls.h`) → build `dev-release` + `ctest` + ABI diff + `n4m_cli` green → 2d Python
binding → 2e R/MATLAB/JS-WASM. Codex-after on the green branch.

<!-- PHASE_2_RESULT -->

## Phase 2 core (C++ only) — DONE on `feat/namespace-ml-abi2`

ABI **2.0.0** clean break. C++ surface only (headers + `cpp/src/c_api` + version + linker map +
ABI snapshots + regen script + abi-check SONAME + catalog symbol records). Bindings deferred to a later wave.

- **2a rename map (keystone):** `proposals/namespace/_build_rename_map.py` generates `_rename_map.tsv`
  deterministically from `catalog/methods.yaml` (skip the 10 `c_surface:"none"`). **566 symbols mapped, 566
  unique new, 0 collisions**; all 20 Codex worked-examples match; validated against
  `expected_symbols_linux.txt` (566/566). Generator is idempotent (re-reads its artifact once the catalog is
  migrated). 4 irregulars handled explicitly (`pls_fit_simple→…_pls_fit`; the 3 no-verb utils).
- **2b symbol rename applied:** 2568 boundary-aware replacements across 57 files in `cpp/include/n4m/*.h`,
  `cpp/src/c_api/*.cpp`, `cpp/src/core/{config.hpp,extra_pls.cpp}`, and `cpp/tests/*.cpp`. Fixed the 8
  token-paste macro invocations in `c_api_advanced.cpp` (`DEFINE_MSC_API`/`DEFINE_ALIGN_API`/
  `DEFINE_SELECTOR_CREATE`). `n4m_version.h` → 2.0.0; `n4m_linux.map` N4M_1 → **N4M_2**. Exported surface =
  **702 (566 method + 136 infra)**, exact match vs the renamed expectation, 0 leaks.
- **2c header split:** umbrella `n4m.h` (keeps shared infra + ABI guard rails) + 12 role headers + 22 subheaders
  (`transform`/`estimators`/`augmentation`). Deleted `pls.h` + 10 stubs (`aom_pop, preprocessing, models,
  selection, splitters, filters, diagnostics, transfer, utilities, context`) — no compat include; pls.h infra
  moved into the umbrella. CMake glob auto-installs the new tree.
- **Snapshots/gate:** regenerated `expected_symbols_{linux,macos,windows}.txt` (N4M_2 node);
  `regen_abi_snapshots.sh` no longer hardcodes N4M_1; `abi-check.yml` SONAME → **libn4m.so.2**;
  `docs/abi/changes_log.md` ABI-2.0.0 entry. Catalog symbol records (`methods.yaml`, `methods/*.yaml`,
  `abi_method_map.yaml`, `abi_missing_methods.yaml`) updated in lockstep; `reconcile_abi.py` `is_infra` taught
  the `n4m_model_selection_` ≠ `n4m_model_` distinction.

**Green gate (dev-release, system gcc, FITPACK off):** build 0 errors · `ctest` 100% (2/2) ·
`n4m_cli --selfcheck` OK ABI 2.0.0 · `regen_abi_snapshots --check` up-to-date · `validate.py --strict-abi
--check-references` PASS · `reconcile_abi --check` 566/566+136 = 702/702. **`bump_version --check` reports the
single expected drift in `bindings/python/src/n4m/_ffi.py` (ABI 1.22.0 → 2.0.0)** — a bindings file, out of
scope for this C++-only wave; it re-syncs in the bindings wave (2d/2e).

**Independent re-verification (orchestrator):** `n4m_cli --abi-info` → ABI **2.0.0**; the built `.so`
exports **702** version-tagged (`@@N4M_2`) symbols == snapshot 702; new names present
(`n4m_estimators_ridge_fit`, `n4m_transform_snv_create`, `n4m_estimators_pls_fit`,
`n4m_ensemble_aom_ridge_blender_fit`) and **all old terse names gone**. 12 role headers + the
`transform/estimators/augmentation` subheaders + `lowlevel.h` present; `pls.h` + stubs removed. Diff vs
`main`: 296 files, +12955 / −13582.

→ **Codex combined gate** (Phase 2-core after-review + Phase 3 bindings before-gate) launched via
`codex exec` → `/tmp/codex_phase2_phase3.md`. The bindings wave (same merge unit) starts once it returns GO.

### Codex gate — Phase 2 after + Phase 3 before (transcript: `proposals/namespace/codex_phase2_phase3.md`)

**Phase 2 core = GO** (no blocker). Codex verified: rename map 566/566 unique + 0 collisions + 4 irregulars
correct; header split correct (umbrella + 12 role + 22 sub + `lowlevel.h`; `pls.h` + 10 stubs absent, no compat);
the 10 `c_surface:none` have no decl/export; ABI 2.0.0 / `N4M_2` / SONAME `libn4m.so.2` / generic regen script;
`.so` == Linux snapshot; macOS/Windows snapshots = Linux minus the `N4M_2` node. Only drift = `_ffi.py` (Phase 3).

**Phase 3 (bindings) = LOCKED** (one merge unit):
- **Python:** recover the **missing** `scripts/generate_ffi_decls.py` (parse `N4M_API` protos from headers → ctypes,
  validate names vs snapshot; `_rename_map.tsv` as backstop, not from the `.so`); bump `_ffi.py`→2.0.0; replace flat
  `python.py` with `n4m.<role>` subpackages + **public role class names** (Ridge/SNV/CARS/KennardStone/AOMRidgeBlender…),
  no `Native*`, no top-level compat; regen `python_nirs4all_methods`+`python_pls4all` mirrors via `make_python_package.py`.
- **R:** rewrite C calls via `_rename_map.tsv`, R role wrappers, gate `n4m_abi_version()[1:2]==c(2,0)`.
- **MATLAB/Octave:** MEX C-call rename (Octave local, MATLAB CI-bound). **JS/WASM:** ccall/cwrap + `wasm_entry.c` rename
  + regen dist (emscripten CI-bound). **JNI/Android:** archived, no gate.
- Risks: missing FFI generator (biggest); stale artifacts shadow tests (always `N4M_LIB_PATH`); clean break (no
  compat/`Native*`); the 10 `c_surface:none` methods stay Python-only.

### Phase 3 — executing: **Python binding (3a)** agent on the branch (keystone; locally verifiable).

### Phase 3a Python — part (i) FFI/ABI-2 **DONE & green**; part (ii) restructure **running**
- **Recovered `scripts/generate_ffi_decls.py`** (parses `N4M_API` protos from headers → ctypes, validates vs
  snapshot, `--check`). Regenerated `_ffi_decls.py` (702 symbols, ABI-2); `_ffi.py`→2.0.0; all C call sites in
  `bindings/python*` renamed (566 + 113 prefix roots) — **zero old symbols remain**; mirrors regenerated.
- **Gates:** `generate_ffi_decls --check` PASS · n4m pytest **455 passed / 1 skip / 1 fail** · mirrors PASS ·
  `bump_version --check` **PASS** (Phase-2 `_ffi` drift resolved). The 1 failure (`docs/methods/index.md` says
  208 vs catalog **209** — WVC split) + ruff are **pre-existing/out-of-scope** → `index.md` fix lands in Phase 4
  docs; no `[tool.ruff]` config exists in the repo. Binding makes **live ABI-2 C calls** through every path.
- **Part (ii) running:** split `python.py` → `n4m.<role>` subpackages + **public class names** (drop `Native*` /
  `n4m.python` / top-level re-exports), 10 `c_surface:none` stay Python-only, rewrite tests to the new surface.

### Phase 3a Python — part (ii) restructure **DONE & green**
- **12 role packages / 49 leaf modules** per `_target_ml_table.tsv` (`n4m.estimators.regression.regularized.Ridge`,
  `n4m.transform.scatter.SNV`, `n4m.feature_selection.wrapper.CARS`, `n4m.model_selection.splitters.KennardStone`,
  `n4m.ensemble.AOMRidgeBlenderRegressor`, `n4m.lowlevel.moments`, …). `n4m.__init__` exposes only metadata + subpackages.
- Impl re-homed into private `n4m/_impl/` (flat `python.py`→`_impl/native.py`; `sklearn/*`/`aom`/`moment`→`_impl/*`) via new
  `bindings/python/scripts/build_role_tree.py`. **No public `Native*`, no `n4m.python`, no top-level re-exports.** The 10
  `c_surface:none` stay Python-only. Fixed one runtime-built metric symbol (`n4m_metric_*`→`n4m_metrics_regression_metrics_*`).
- **Gates:** n4m pytest **455 passed / 1 skip / 1 pre-existing-fail** (208≠209 doc); mirrors PASS; `ruff` 12 **pre-existing**, 0 new.
  Tests ported faithfully (incl. 204KB `test_moment_model_wrappers.py` → 82 passed). Updated 9 bench scripts + 30 catalog
  `module:` fields + the packager smoke test. `pls4all` subset independent (untouched, passes). **Python binding 3a fully done.**

### Phase 3b/c/d — **launching** R + JS/WASM + MATLAB/Octave in parallel (mechanical rename via `_rename_map.tsv`; some gates CI-bound). One Codex-after on the whole bindings wave once they land.

### Phase 3b/c/d bindings — all renamed to ABI-2 (static proofs green; compile gates CI-bound)
- **R**: 229 subs across `r_dispatch/r_methods/r_preprocessing.c` (n4m + pls4all); forwarding role aliases (31+28) in new
  `role_aliases.R` + NAMESPACE + man `\alias`. **0 old symbols remain; all 141 referenced symbols in the snapshot.**
  `n4m_abi_version()` → c(2,0) at runtime. **Scope call (for Codex):** only the 4 locked role names + the fully-prescribed
  `*_select`/`*_transform`/`*_split` families were renamed; the contested ~100-name `estimators`/`regression` `*_fit`
  surface left idiomatic (R names underspecified). R toolchain absent → `R CMD INSTALL`/testthat/`test_parity.R` **CI-bound**.
- **JS/WASM**: `wasm_entry.c` 123 renames/117 symbols + `model.ts`/tests/README. **0 old remain; all in snapshot; `tsc`
  clean (exit 0).** `EXPORTED_FUNCTIONS` snapshot-driven (auto ABI-2). Emscripten build + `npm test` **CI-bound** (no emsdk).
- **MATLAB/Octave**: 5 files, 96 symbols across the MEX sources; version docstrings refreshed. **0 old remain; all MEX
  symbols in snapshot.** `bindings/octave/` is a 3rd-party corpus (nothing to migrate). Octave/MATLAB compile **CI-bound**.

→ **Codex after-review over the whole bindings wave** (Python 3a + R + JS + MATLAB) launched via `codex exec`
→ `/tmp/codex_bindings_review.md`. Then Phase 4 (web/docs) → **checkpoint before downstream repos**.

### Codex bindings after-review — verdict: **NO-GO** (transcript: `proposals/namespace/codex_bindings_review.md`)
Positives confirmed: `generate_ffi_decls --check` ✅, `_ffi_decls` 702 == snapshot, `_ffi.py` 2.0.0; R/JS/MATLAB
call-site scans 0 old symbols. **Five binding-scope blockers (all confirmed):**
1. **Python namespace incomplete** — `feature_selection/wrapper.py` is empty (3 lines) → `from n4m.feature_selection.wrapper import CARS` fails. Audit ALL leaf modules / the `build_role_tree.py` generator.
2. **ABI-1 dynamic lookup in `_impl/native.py:9635`** (`n4m_metric_<name>` vs ABI-2 `n4m_metrics_regression_metrics_<name>`) → `nirs_metrics` broken. Fix all runtime-built symbol names.
3. **Catalog SOURCE stale** — `catalog/methods.yaml` has **31** `n4m.python` module fields (splits edited, source not). Fix source + regenerate splits.
4. **Cross-binding scripts** still import removed surfaces (`Native*`, `n4m.sklearn/aom/moment/python`) — `donor_ops.py`, `bench_aom_robust_hpo_timing.py`, … Port them.
5. **R**: locked `n4m_regression_ridge` alias **missing** — add it (Codex accepts the partial R scope otherwise).
**Merge-gate list (CI-bound):** R CMD INSTALL + testthat + parity + ABI c(2,0); emscripten + `npm test`; MATLAB runtests. Static+tsc accepts the rename on-branch but those runtime gates must pass in CI before merge.

→ **Fix agent** launched on the branch to clear items 1–5 + re-run local gates + the locked-API smoke imports.

### Bindings fixes — **DONE, gates green** (fix agent + orchestrator verify)
1. **Python namespace completeness:** root cause = the 24 wrapper selectors + ranking had **no `n4m` Python impl**
   (existed only in the separate `pls4all` wheel). Agent added real impls to `n4m/_impl/{native.py,selection.py}`
   (CARS/UVE/SPA/… on n4m's own config/method-result plumbing, mirroring pls4all's ABI-2 contracts) + fixed
   `build_role_tree.py`. `feature_selection.wrapper` now exports all 24. **(New code → Codex re-confirm.)** Remaining
   empty leaves = methods with a C symbol but no `n4m` Python class (pre-existing; surface only in `pls4all`; catalog
   declares no python binding → no test break).
2. **`_impl/native.py:9635`** `n4m_metric_*`→`n4m_metrics_regression_metrics_*`; all runtime-built symbols audited. `nirs_metrics` live.
3. **`catalog/methods.yaml`**: 30 stale `module:` fields fixed + splits regenerated; `split --check` PASS.
4. **Cross-binding scripts**: 10 ported off removed surfaces → `n4m._impl.*`; cross-binding tests **123 fail → 1 fail** (282 passed; the 1 = pre-existing spline-parity).
5. **R**: added `n4m_regression_ridge` (dispatch case + `ridge_fit()` + alias + NAMESPACE + man + test).
**Gates:** `generate_ffi_decls --check` ✅ · pytest **455 passed/1 skip/1 fail** (208≠209 doc → Phase 4) · `selftest` ✅ ·
`validate --strict-abi` ✅ 702/702, 209/209 · `split --check` ✅ · `make_python_package --all` ✅ · **orchestrator smoke ✅**
(SNV / Ridge / CARS-fit / ensemble / nirs_metrics all live). Codex re-confirm running; CI runtime gates (R/MATLAB/WASM) still required before merge.

## ✅ `nirs4all-methods` repo migration (CODE) COMPLETE on `feat/namespace-ml-abi2` — catalog + C ABI 2.0 + all bindings, green on-branch.
Remaining: **Phase 4** (web/docs relabel + bench-data + `index.md` 208→209) · **Phase 5** (downstream repos).

### Codex bindings re-confirm — NO-GO on 2 residuals → **FIXED & verified** (orchestrator; transcript `proposals/namespace/codex_bindings_reconfirm.md`)
1. **Selector fold-plan parity:** `_impl/native.py:_fold_plan` shuffled folds on nonzero seed AND used a `(i*cv)//n`
   layout — both diverge from the canonical `pls4all`/R/MATLAB plan. Rewrote it to the EXACT `pls4all._default_plan`
   layout (`fold_size = n//n_folds`, last fold absorbs remainder, seed never affects composition) — provably identical
   fold composition (e.g. n=10,nf=3 → 3/3/4). CARS/UVE/… now fit on the canonical plan.
2. **4 stray scripts** (`bench_aom_{chain_ridge_pls,robust_hpo,ridge_blender}_timing.py`, `bench_aom_screen_refit_scaling.py`)
   still called removed top-level `n4m.aom_*`/`n4m.build_aom_*`/`n4m.NativeAOMFixedCandidateRegressor` at runtime →
   repointed to `n4m._impl.native` / `n4m._impl` (call-sites only; docstrings + valid `n4m.library_path/abi_version` kept).
**Verify (`../nirs4all/.venv` python):** n4m pytest **455 passed / 1 skip / 1 fail** (pre-existing 208≠209 doc) · cross-binding
**282 passed / 1 fail** (pre-existing spline-parity) — **no regression; baseline green restored**. 4 scripts import OK.
→ Codex re-confirm of these 2 fixes running; **Phase 4 (web/docs) launched in parallel** (user chose "continue all autonomously").

### ✅ Bindings wave — Codex **GO** (transcript `proposals/namespace/codex_bindings_reconfirm.md` → final GO in `_reconfirm2`)
`_fold_plan` confirmed byte-identical to `pls4all`'s contiguous plan (n=10,nf=3 → 3/3/4, seed unused); the 4 scripts call only `n4m._impl` + retained metadata helpers. **Bindings merge unit GO.** CI runtime gates (R CMD INSTALL+testthat+parity / emscripten+npm test / MATLAB runtests) remain mandatory before the branch merges.

### Phase 4 (web + docs) — **DONE & green** (agent)
- **Doc-count fixed:** `build_methods.py` made catalog-driven → `docs/methods/index.md` shows **209** grouped by the
  12 `n4m.<role>` roles (each links to a page; 10 migration-added methods get catalog stub pages).
  `test_methods_index_total_matches_catalog_file_count` **PASSES**.
- **bench-data + dashboard:** regenerated `docs/_static/bench-data.json`; remapped dashboard grouping
  (`algo_groups`/`algo_to_group`) to the 12 namespace roles. `test_dashboard_contract` + `test_raw_manifest_reconciliation`
  + `test_catalog_python_bindings` → **20 passed**.
- **`proposals/dashboard_v2/` relabeled:** data-driven categories → regenerated `bench-data.js` from the new JSON
  (offline/no-CDN preserved); README notes the 12 roles. (= the "notify the alternative web page" step.)
- **Sphinx docs:** new `docs/MIGRATION_ABI2.md` (old→new symbols/imports/headers, validated); `docs/abi/reference.md`
  + `docs/bindings/python.md` rewritten to the ABI-2 role surface; wired into `about.md`. `sphinx-build` exit 0
  (54 pre-existing warnings, none from the migration).

## ✅ `nirs4all-methods` repo migration FULLY COMPLETE on `feat/namespace-ml-abi2` — catalog + C ABI 2.0 + all bindings + web + docs, green on-branch (CI runtime gates pending merge).
Next: **Phase 5 — downstream repos** (`nirs4all-lite` → `nirs4all-web` → studio/io/datasets/nirs4all).

### Downstream coupling scout (real CODE-level, not raw hit-counts)
- **nirs4all-lite** — re-exporter: `compat/upstreams.toml` (`python_imports=[nirs4all_methods,pls4all,n4m,nirs4all.methods]`
  + version pin); WASM parity/execution tests consume `nirs4all-methods/bindings/js/dist`; `test_execution_parity.py`
  does top-level `import n4m`/`import pls4all` (still valid). `Makefile` path var.
- **nirs4all-web** (studio-lite): `scripts/validate-catalog.mjs` validates its catalog symbols against
  `nirs4all-methods/cpp/abi/expected_symbols_*.txt` → **web catalog symbol refs must move to ABI-2 names**;
  `build-wasm.sh` copies the methods WASM dist.
- **nirs4all-studio / -io / -datasets / nirs4all**: **no code coupling** — only doc / CI-comment mentions → no code changes (optional doc sweep).
→ **Phase 5 real scope = lite (upstreams + tests) + web (catalog ABI-2 symbols).** Codex Phase-4-after + Phase-5-before gate launched → `/tmp/codex_phase4_phase5.md`.

### Codex Phase-4-after + Phase-5-before gate (transcript `proposals/namespace/codex_phase4_phase5.md`)
**Phase 4 = NO-GO (1 blocker):** index/bench-data/dashboard/MIGRATION docs ✅, but **78 generated `docs/methods/*.md`
pages still render OLD ABI-1 symbols** (e.g. `boosting_pls.md` → `n4m_boosting_pls_fit`, should be
`n4m_ensemble_boosting_pls_fit`). Cause: `build_methods.py` per-method metadata reads removed
`bindings/_catalog/sklearn_tier2.yaml` + `usage_section` falls back to legacy `<method>_fit`/hardcoded AOM symbols.
Fix = wire catalog `c_surface`/`_rename_map.tsv` into per-method symbol rendering + regen + add a doc-lint.
**Phase 5 = LOCKED** (Codex found deeper sites than the scout):
- **nirs4all-web**: `studio-lite/src/catalog/nodes.ts` declares 56 symbols, **53 old ABI-1** → remap via `_rename_map.tsv`.
  `validate-catalog.mjs` unchanged; `src/engine/wasm/methods/*` restaged by `build-wasm.sh` (CI). Gates: `validate:catalog`/`typecheck`/`test`/`build` (+WASM CI).
- **nirs4all-lite**: `bindings/python/pyproject.toml` floor `0.98→0.99`; `_execution.py` imports →
  `n4m.transform.scatter.SNV`/`n4m.transform.smoothing.SavitzkyGolay`/`n4m.model_selection.splitters.KennardStoneSplitter`;
  **`bindings/rust/nirs4all/src/lib.rs` dynamic symbol loads** → ABI-2 (`n4m_split_kennard_stone_*`/`n4m_pp_snv_*`/`n4m_pp_savgol_*`);
  infra `n4m_model_fit` unchanged. Gates: cargo fmt/clippy/test, py import, make test (WASM/R/Octave CI).
- **No code change** confirmed: studio, io, datasets, nirs4all main (optional io doc sweep).
→ Executing in parallel: **A** methods doc fix · **B** nirs4all-web · **C** nirs4all-lite.

### Phase 5 downstream — **nirs4all-web ✅ & nirs4all-lite ✅** (each on branch `feat/n4m-abi2-namespace`; A methods-doc-fix still running)
- **nirs4all-web**: 53 symbols remapped in `studio-lite/src/catalog/nodes.ts` (infra `n4m_model_fit` + 2 wasm shims kept).
  **All gates GREEN locally** (node via nvm): `validate:catalog` PASS (catalog↔ABI-2 in sync), `typecheck` ✅,
  vitest **90 tests** ✅, `build` + `build:single` ✅. Zero old symbols (2 grep proofs). No CI-bound steps needed.
- **nirs4all-lite**: `pyproject.toml` floor 0.98→0.99 (both); `_execution.py` imports →
  `n4m.transform.scatter.SNV`/`n4m.transform.smoothing.SavitzkyGolay`/`n4m.model_selection.splitters.KennardStoneSplitter`;
  `bindings/rust/nirs4all/src/lib.rs` 9 dynamic symbol loads + error strings → ABI-2; `upstreams.toml` unchanged (correct).
  **Gates GREEN locally**: cargo fmt/clippy/check/test **8/8** incl. the **strict end-to-end parity test against
  `libn4m.so.2.0.0`** (loads every remapped symbol, matches the Python oracle); `make test` test-rust/test-python(13)/test-python-parity ✅.
  WASM/R/MATLAB CI-bound. Zero old symbols (full 566-scan).
- **No code change** (Codex-confirmed): nirs4all-studio, -io, -datasets, nirs4all main.

### Phase 4 doc fix (A) — **DONE & green**
`build_methods.py`: removed the dead `sklearn_tier2.yaml` symbol source; wired `_rename_map.tsv` as the authoritative
ABI-2 symbol source (`method_c_symbol`/`rename_symbols_in_text` + final per-page substitution); fixed the legacy
`<method>_fit` fallback + a stale `methods_bibliography.py` string. **Doc-lint added** (`build_methods.py --strict` +
`docs/_extras/test_methods_doc_lint.py`, wired into `.github/workflows/docs.yml`). 73 registry + 10 stub + 8 hand
pages regenerated; `boosting_pls.md` now renders `n4m_ensemble_boosting_pls_fit`.
**Gates:** `--strict` exit 0 (no ABI-1 symbols) · grep proof zero old symbols in `docs/methods` · pytest **18 passed**
(incl. new doc-lint) · `sphinx` exit 0 (42 pre-existing warnings). Lint fails-on-inject / passes-on-revert verified.

### → Final Codex review (Phase 4 fix + Phase 5 web + lite) launched → `/tmp/codex_final.md`.

### Final Codex review — **NO-GO (1 residual; web ✅ + lite ✅ both pass)** (transcript `proposals/namespace/codex_final.md`)
Web (53 symbols, infra+shims kept) and lite (Python floors/imports + Rust loads ABI-2, infra unchanged) both **PASS**.
Methods residual: the doc-lint catches *exact* old symbols (0 remain) but **118 `docs/methods/*.md` pages still publish
old wildcard ABI-1 PREFIXES** (e.g. `split_kennard_stone.md` → `n4m_split_kennard_stone_*`; also `n4m_pp_*`/`n4m_aug_*`/`n4m_split_*`).
A second operator-page rendering path still points at the removed `n4m/sklearn` source. Fix = render ABI-2 prefixes
from catalog/`_rename_map.tsv` + **extend the doc-lint to catch wildcard/family prefixes** + regen.
→ Fix agent launched.

### Final residual — **FIXED & verified** → migration program COMPLETE
`render_operator_page()` now translates `c_prefix` ABI-1→ABI-2 (via `load_prefix_old_to_new()` from `_rename_map.tsv`)
+ an idempotent in-place pass rewrote all 118 operator pages' `_C ABI_` wildcard lines
(`n4m_pp_snv_*`→`n4m_transform_snv_*`, `n4m_split_kennard_stone_*`→`n4m_model_selection_kennard_stone_*`,
`n4m_aug_mixup_*`→`n4m_augmentation_mixup_*`, …). **Doc-lint strengthened** to flag old per-method + coarse family
prefixes (`n4m_pp_`/`n4m_aug_`/`n4m_split_`/`n4m_aom_`) incl. wildcards, with a `*_t` type-name allow-list (Codex's
exact complaint is now an automated CI gate). **Independently verified:** `build_methods --strict` → "DOC-LINT: OK"
· grep wildcards (excl `*_t`) → none · grep exact old → none · `test_methods_doc_lint.py` → 3 passed. Final Codex GO re-confirm running.

## 🎉 MIGRATION COMPLETE (on branches; reversible)
- **nirs4all-methods** `feat/namespace-ml-abi2` — catalog **209** + C ABI **2.0** + all 4 bindings + web/docs, green on-branch.
- **nirs4all-web** + **nirs4all-lite** `feat/n4m-abi2-namespace` — green locally (Codex-passed).
- **No change**: nirs4all-studio, -io, -datasets, nirs4all main.
- **Remaining for MERGE = CI/env-bound only**: macOS/Windows ABI snapshots; emscripten/WASM; R CMD check; MATLAB; full cross-binding parity sweep.

### ✅ Final Codex re-confirm — **GO for the whole migration program** (transcript `proposals/namespace/codex_final_go.md`)
Verified the operator-page prefix translation + in-place pass + strengthened lint; **0 exact old symbols, 0 forbidden
prefixes** (excl `*_t` types) across the 118 operator pages. Non-blocking note: `docs/methods/filter_x_outlier.md`
mentions the public ABI-2 TYPE `n4m_filter_x_outlier_method_t` (a `_t` type, not a function/wildcard) — allowed, does not change GO.
**Merge order:** methods `feat/namespace-ml-abi2` first → web/lite `feat/n4m-abi2-namespace` pinned to the merged ABI-2
methods. Full CI/env-bound merge-gate list per branch is in the transcript.

— END: program complete, Codex GO, on branches, awaiting CI for the env-bound gates + merge. —

### Post-review dashboard polish (maintainer feedback on `dashboard_v2`)
Maintainer liked the design but flagged: (1) no doc links, (2) methods still showed legacy `aug_`/`pp_`/…
prefixes. Both fixed on `feat/namespace-ml-abi2`:
- **`docs/_extras/build_landing.py`**: new `_algo_display_fq()` resolves each benchmark algo → ABI-2 namespace
  (legacy symbol token → `_rename_map.tsv` → ABI-2 symbol → catalog `c_surface` → `leaf`/`fq_name`). Emits
  `algo_display` + `algo_fq` in `bench-data.json`. **0/183 displays keep a legacy prefix; 179/183 carry a full `fq_name`.**
- **`proposals/dashboard_v2/` (app.js + styles.css + README)**: cards now show the leaf as the title and the
  `n4m.<role>…<leaf>` path as the handle (drawer too); added **📄 docs ↗** (card) + **Open documentation ↗** (drawer)
  links to `methods.nirs4all.org/methods/<page>.html` via a `DOCS_BASE` const. Card `<button>`→`<div role="button">`
  + keydown so the nested doc link is valid; link `stopPropagation`. Regenerated `bench-data.js`.
- **Verified**: `node --check app.js` ✅; DOM-shim render of `methodCard`/`openDrawer` ✅ (clean title, fq handle,
  doc href, role=button, stopPropagation); dashboard contract + manifest **8 passed**; **zero external resource loads**
  (offline preserved — the doc links are navigational only).
(Underlying bench IDs / doc-page filenames keep legacy names for parity-data lineage; nothing user-facing shows them.)

### Publication readiness — coordinated **1.0.0** breaking release (maintainer-chosen)
- **methods → 1.0.0** via `scripts/bump_version.sh --bump 1.0.0` (R `n4m`/`pls4all` DESCRIPTION + cran-comments,
  dev pyproject, npm package.json/lock, CITATION.cff, docs, README; `--check` green; ABI stays 2.0.0). The two
  **PyPI wheel mirrors** (`python_nirs4all_methods`, `python_pls4all`) aren't tracked by `--check` →
  regenerated via `make_python_package.py`, now **1.0.0**. `CHANGELOG.md` `[1.0.0]` breaking entry added.
- **lite → 0.2.0** (minor bump from 0.1.0 — maintainer's choice; methods stays 1.0.0). Via its own
  `bump_version.sh` (Rust crate SoT → pyproject/DESCRIPTION/wasm + package-lock/Cargo.lock fixed); repinned
  `nirs4all-methods>=1.0.0`; CHANGELOG `[0.2.0]`. Gates green (cargo fmt/clippy/test 8 + py unittest 13; WASM/R/MATLAB CI-bound).
- **web**: tidied 3 stale-symbol *comments* (functional surface already correct).
- **lab** (`feat/n4m-abi2`): `gpu_pls_proto/n4m_oracle.py` ctypes calls → ABI-2 (8 symbols) + docstrings/notes;
  verified end-to-end against `libn4m.so.2.0.0` (real fits); `DEFAULT_LIB` 1.11.0 → 2.0.0.
- **Confirmed NOT methods consumers** (no change needed): io, datasets, aom, studio, formats, dag-ml*, nirs4all main, org, papers.
- **Final scan:** all SOURCE clean across every repo. Remaining hits are **generated artifacts** only —
  web's `src/engine/wasm/methods/*` + `dist-single/` (restaged by `build-wasm.sh`/`npm build` once the methods
  WASM dist is rebuilt — emscripten CI-bound) and one lab research **artifact log** (historical, harmless).
- **NOT done (your action):** actual publish to PyPI/CRAN/npm + `git push` of the branches + the CI/env-bound gates.
