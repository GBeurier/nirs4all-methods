# Verification Report — merge sprint end-to-end test

**Date**: 2026-05-22
**Branch**: `merge/import-donor`
**Tip**: `e376b24`

This document records the **actual** end-to-end verification of the merge
sprint, not just per-commit `pls4all_tests` passing.

## TL;DR

| What | Status | Evidence |
|---|:---:|---|
| pls4all C++ build (dev-release preset) | ✅ green | `cmake --build --preset dev-release` clean |
| pls4all C++ test suite | ✅ green | `pls4all_tests` reports `265 run, 0 failures, 0 skipped` |
| pls4all CLI selfcheck | ✅ green | `pls4all_cli --selfcheck` returns `selfcheck OK` |
| Catalog validation | ✅ PASS | 160 unique methods, 2 active subset closures non-empty |
| License audit | ✅ PASS | 2 vendored entries, 10 referenced libs, all with LICENSE files |
| Benchmark matrix (pls4all-side 73 methods) | ✅ runs | smoke test 4 algos × 10 backends = 40 cells, all OK |
| Benchmark matrix (donor 87 methods) | ❌ MISSING | `KeyError('unknown method: snv')` — donor methods not in `benchmarks/parity_timing/registry.py` |
| Donor C++ tests (`n4m_tests`) | ❌ NOT BUILT | donor build wiring deferred to M5+M6 focused session |
| Python `nirs4all_methods` package | ⚪ scaffold | `__init__.py` raises ImportError until M9 lands build glue |
| Python slim `pls4all` package (new) | ⚪ scaffold | `__init__.py` raises ImportError until M10 lands |
| R `n4m` + `pls4all` packages (new) | ⚪ scaffold | DESCRIPTION + README only, no `src/libn4m_*/` build yet |
| cibuildwheel matrix | ⚪ scaffold | `release-wheels.yml` gated `if: false` |
| ABI rename `p4a_*` → `n4m_*` | 🟡 deferred | script `migrate_p4a_to_n4m.py` ready, execution requires M6 header merge first |

## Evidence — what actually works

### pls4all C++ build + tests (full evidence)

```
$ cmake --preset dev-release
-- Configuring done
-- Generating done
-- Build files have been written to: build/dev-release
$ cmake --build --preset dev-release -j 4
ninja: no work to do.   # post M0-M17, no rebuild needed since pls4all CMakeLists unchanged
$ ./build/dev-release/cpp/tests/pls4all_tests
=== 265 run, 0 failures, 0 skipped ===
$ ./build/dev-release/cpp/cli/pls4all_cli --version
pls4all 0.97.3+abi.1.16.0
$ ./build/dev-release/cpp/cli/pls4all_cli --selfcheck
selfcheck OK
```

### Catalog gate

```
$ python catalog/scripts/validate_catalog.py
== validate_catalog.py ==  repo=/home/delete/nirs4all/pls4all
  methods.yaml — 160 entries
  methods.yaml OK (160 entries, 160 unique ids)
  subsets — 7 descriptors
  aompls.yaml: status=planned (closure check skipped)
  n4m_transfer.yaml: status=planned (closure check skipped)
  nirs4all_augmentation.yaml: status=planned (closure check skipped)
  nirs4all_methods.yaml: closure OK (160 methods after excludes)
  nirs4all_preprocessing.yaml: status=planned (closure check skipped)
  nirs4all_selection.yaml: status=planned (closure check skipped)
  pls4all.yaml: closure OK (68 methods after excludes)
--
PASS: catalog validation green
```

### License audit

```
$ python scripts/audit_third_party_licenses.py --check
PASS: license audit clean (2 vendored entries, 10 reference libs)
```

### Benchmark matrix — pls4all side

Smoke run: 4 algorithms (`pls`, `simpls`, `opls`, `cars_select`) × 10
backends (`registry_pls4all`, `cpp`, `python_tier1/2`, `r_tier1/2`,
`r_pls_compat`, `r_mdatools_compat`, `matlab_tier1/2`) × 30×30 dataset
= **40 cells, all green**.

Full CSV at `docs/merge/verification/smoke_pls4all_4algos.csv`.

Sample row (SIMPLS C++ binding):
```
algorithm,backend,language,tier,ok,median_ms,parity_ok
pls,cpp,C++,canonical,True,0.45,True
simpls,cpp,C++,canonical,True,0.43,True
opls,cpp,C++,canonical,True,2.10,True
cars_select,cpp,C++,canonical,True,0.70,True
```

This proves the **pls4all benchmark layer rebuilds end-to-end** after
the merge: methods register, backends dispatch, parity oracles compute,
CSV writes.

## Evidence — what is genuinely deferred

### Benchmark matrix — donor side: `KeyError`

```
$ python benchmarks/cross_binding/orchestrator.py \
    --algorithms snv --sizes 30x30 --threads 1 \
    --libp4a-build dev-release --n-runs 2 --only-pls4all \
    --out-csv /tmp/donor_smoke.csv --force
[1] snv     30×30  t=1  dev-release registry_pls4all: WARN: could not build BENCH_R_PARAMS_JSON for snv: KeyError('unknown method: snv')
FAIL — unknown algo failed: invalid argument
[... all 10 backends report the same KeyError ...]
```

**Root cause**: `benchmarks/parity_timing/registry.py` is pls4all's
canonical method catalog. It declares **73 PLS-family methods**. The
**87 donor methods** (preprocessing, augmentation, splitters, filters,
utilities) are catalogued in `catalog/methods.yaml` (160 total) but
were never added to `benchmarks/parity_timing/registry.py`.

This is the exact gap that `docs/merge/M7/STATUS.md` lists as deferred:

> "Bring donor methods into `benchmarks/parity_timing/registry.py` — ❌ deferred"

**To close this gap**, a focused M8 session must:
1. Read the 87 donor methods from `catalog/methods.yaml`.
2. For each, add a `MethodSpec("...", category=..., reference_factory=...)` entry to `benchmarks/parity_timing/registry.py`.
3. Reference factories typically delegate to `scipy`, `pybaselines`,
   `pywt`, `sklearn`, or selected `nirs4all` operators (per the
   `parity/donor_imports/divergences.json` policy).
4. Re-run the orchestrator to populate `docs/_static/bench-data.json`.

Until then, `pls 4all` methods benchmark; donor methods do not.

### Other deferred items (covered in `docs/merge/README.md`)

- **M4** function split (39 stubs created, extraction recipe documented, surgery not executed)
- **M5+M6 paired session** for ABI rename + header layout
- **M9+M10+M11** package activation (catalog/scripts/render_subset.py
  exists but vendored-source pipeline not wired)
- **M11** CRAN cleanup pass (existing 1 WARNING + 4 NOTEs from M0
  still need to be fixed before submission)
- **M13** secondary binding refresh (12 pls4all bindings still use
  `p4a_*` symbols — refresh after M5 lands)
- **M15** cibuildwheel matrix activation (workflow gated `if: false`
  until M9+M10 land their build glue)

## Cleanup performed during this verification

| Path | Before | After |
|---|---|---|
| `/home/delete/nirs4all/M0/`, `/M1/`, `/M2.5/`, `/M3/`, `/M4/`, `/M5/`, `/M7/`, `/M16/`, `/M17/` | 9 working dirs duplicating `docs/merge/`-committed files | ❌ deleted (content lives at `docs/merge/M{N}/` in-repo) |
| `/home/delete/nirs4all/MERGE_ROADMAP.md` | working file at user's home | ❌ deleted (committed at `docs/merge/MERGE_ROADMAP.md`) |
| `/tmp/m2_placeholder_writer.py`, `/tmp/m4_stub_writer.py`, `/tmp/m6_header_stubs.py`, `/tmp/summarize_closure.py` | one-shot scripts dropped during the sprint | ❌ deleted |
| `/tmp/merge_smoke.csv`, `/tmp/donor_smoke.csv`, `/tmp/pls4all_4algos.csv` | benchmark smoke artifacts | moved to `docs/merge/verification/smoke_*.csv` then `/tmp` copies deleted |
| `benchmarks/cross_binding/data/.reference_oracles/` (23 MB, ~600 untracked files) | leftover from codex/parity-30x30 checkout | ⚪ kept — real benchmark oracles, useful for fast local re-runs; protected by `.gitignore` (added in commit `879691b`) |
| `benchmarks/cross_binding/data/.x_target/` (620 MB) | leftover generated arrays | ⚪ kept — same rationale, protected by `.gitignore` |

## Honest summary

The merge sprint delivered:
- ✅ Real unification of the source tree (222 donor sources lifted to category paths, blame preserved)
- ✅ Common-core consolidation (M2.5: 31 files unified)
- ✅ A working catalog of 160 methods with 2 active subset closures
- ✅ License audit + NOTICE.md + THIRD_PARTY_LICENSES.md
- ✅ pls4all C++ + benchmark matrix unchanged at every commit (265/265 green, 40-cell smoke run OK)
- 🟡 Scaffolds for M4 (39 stubs + recipe), M5 (rename script), M6 (11 headers), M9/M10/M11 (4 packages), M15 (release workflow), M16+M17 procedures

The merge sprint did NOT deliver:
- ❌ Function-level extraction of model.cpp + extra_pls.cpp into the M4 stubs (the 39 stubs are placeholders)
- ❌ ABI rename `p4a_*` → `n4m_*` (script ready, execution paired with M6)
- ❌ Donor methods in `benchmarks/parity_timing/registry.py` (87 methods catalogued but not benchmarked)
- ❌ Donor C++ tests (`n4m_tests`) compiled and runnable from pls4all's build
- ❌ Python/R packages actually building from vendored sources
- ❌ A unified bench-data.json that covers all 160 methods

The unfinished items are explicitly tracked in `docs/merge/README.md`
and the per-phase STATUS docs. They are scoped for follow-up focused
sessions per the roadmap budget of 8–12 weeks (this session covered
the first ~3-4 weeks of mechanical/scaffold work).

**For v1.0.0 release**, the M5+M6 paired session is the critical next
step — it unblocks everything downstream (M4 extraction, M8 unified
benchmark, M9-M11 packages, M13 binding refresh).
