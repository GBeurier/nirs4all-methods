# M7 + M8 — parity gate + dashboard unification: STATUS

**Status**: 🟡 **partial — donor infrastructure lifted under unified paths; full runner merge deferred**
**Date**: 2026-05-21
**Pushed**: pending commit

## What was done

### M7 — parity infrastructure lift

Donor's `_donor/nirs4all-methods/parity/` was relocated to `parity/donor_imports/` so it sits alongside pls4all's existing `parity/{schema,fixtures,python_generator,r_generator}` tree, while keeping the two distinct (no namespace collision yet).

| Origin | Target | Files |
|---|---|--:|
| `_donor/.../parity/run_reference_parity.py` | `parity/donor_imports/run_reference_parity.py` | 1 (107.8 KB; the 5-stage gate driver) |
| `_donor/.../parity/binding_parity.py` | `parity/donor_imports/binding_parity.py` | 1 (the Stage 4 scaffold) |
| `_donor/.../parity/divergences.json` | `parity/donor_imports/divergences.json` | 1 (machine-readable skip registry, 3 classes) |
| `_donor/.../parity/nirs4all_source.py` | `parity/donor_imports/nirs4all_source.py` | 1 |
| `_donor/.../parity/reference_parity.py` | `parity/donor_imports/reference_parity.py` | 1 |
| `_donor/.../parity/python_generator/` | `parity/donor_imports/python_generator/` | full subtree (locked env + adapters for SciPy/sklearn/pybaselines/PyWavelets/nirs4all) |
| `_donor/.../parity/fixtures/` | `parity/donor_imports/fixtures/` | full subtree (~250 JSON fixtures) |
| `_donor/.../parity/schema/` | `parity/donor_imports/schema/` | `n4m_fixture_schema_v1.json` |
| `_donor/.../parity/tolerances.md` | `parity/donor_imports/tolerances.md` | 1 (the n4m-side tolerance table) |
| `_donor/.../parity/README.md` | `parity/donor_imports/README.md` | 1 |

### M8 — benchmark dashboard portability

Donor's `_donor/nirs4all-methods/benchmarks/` similarly relocated:

| Origin | Target | Files |
|---|---|--:|
| `_donor/.../benchmarks/cross_binding/` | `benchmarks/donor_imports/cross_binding/` | subtree (orchestrator, results, validate_dashboard_payload.py) |
| `_donor/.../benchmarks/reference_snapshots/` | `benchmarks/donor_imports/reference_snapshots/` | subtree |
| `_donor/.../benchmarks/validation/` | `benchmarks/donor_imports/validation/` | subtree (the v0 registry/{datasets,suites,references,…}.json) |
| `_donor/.../benchmarks/benchmark_registry.json` | `benchmarks/donor_imports/benchmark_registry.json` | 1 (donor's scaffold registry) |
| `_donor/.../benchmarks/check_truth_sources.py` | `benchmarks/donor_imports/check_truth_sources.py` | 1 |
| `_donor/.../benchmarks/truth_sources.lock.json` | `benchmarks/donor_imports/truth_sources.lock.json` | 1 |
| `_donor/.../benchmarks/README.md` | `benchmarks/donor_imports/README.md` | 1 |

### Dashboard orchestrator de-hardcoded

`benchmarks/cross_binding/orchestrator.py:142` was the only hardcoded
`/home/delete/...` path Codex flagged (the rest of the 31346 grep hits
are inside generated CSV result files, not source). Replaced the
hardcoded path with an env-var-driven fallback:

```python
PLS4ALL_R_ENV = os.environ.get(
    "N4M_R_ENV",
    os.environ.get("PLS4ALL_R_ENV", "/home/delete/miniconda3/envs/pls4all_r"),
)
```

A clean clone now works without editing the file. The maintainer's
historical fallback stays for backward-compat. A warning emits if the
fallback doesn't exist and `N4M_R_ENV_NOWARN` is unset.

## What is NOT done (carried to focused session)

The roadmap M7 requires more than file-relocation:

| Sub-goal | Status |
|---|---|
| Single unified runner driving both halves | ❌ — pls4all has no top-level runner; donor's `run_reference_parity.py` is at `parity/donor_imports/` parked. The merger is M7's main job and lands in a focused session. |
| Implement Stage 4 binding parity (donor's `binding_parity.py` is scaffolded but unwired) | ❌ — wiring happens after the runner is unified. |
| Merge `tolerances.md` (rename rows, calibrate per category per §6.3, add backend rows §6.4) | ❌ — two distinct tables coexist (`parity/tolerances.md` from pls4all, `parity/donor_imports/tolerances.md` from donor). |
| Port pls4all's `📜` documented-but-unwired rows into machine-readable `divergences.json` | ❌ — donor's `divergences.json` exists at `parity/donor_imports/`; pls4all-side rows still in MD only. |
| Sync sklearn version (base 1.4 vs donor 1.8.0); regenerate base fixtures | ❌ — needs a fixture regeneration pass in a locked env. |

The roadmap M8 requires the same kind of follow-on work:

| Sub-goal | Status |
|---|---|
| Bring donor methods into `benchmarks/parity_timing/registry.py` | ❌ — donor's `benchmark_registry.json` parked at `benchmarks/donor_imports/`; merger deferred. |
| Add `category` / `subcategory` / `subset` filters to dashboard UI | ❌ |
| Split `bench-data.json` per-category for lazy loading | ❌ — 829 KB → 2 MB issue remains. |
| Adopt `MERGE_ROADMAP.md` §6.5 container envs (ci-reference / ci-blas-omp / ci-cuda Dockerfiles) | ❌ |

## Build verification

`pls4all` build and tests **unchanged** at this commit:
```
$ cmake --build --preset dev-release && ./build/dev-release/cpp/tests/n4m_tests
=== 265 run, 0 failures, 0 skipped ===
```

The donor infrastructure parked at `parity/donor_imports/` and
`benchmarks/donor_imports/` is dormant: pls4all's CMakeLists doesn't
activate it and no Python entrypoint references the parked paths.

## Remaining `_donor/` footprint after this commit

`_donor/nirs4all-methods/` still contains:
- `cpp/include/n4m/` (donor's umbrella header — duplicates the one already at `cpp/include/n4m/` from M2.5; deletion deferred to M5+M6 focused session)
- `cpp/src/c_api/` (donor's c_api wrappers — pairs with cpp/include/n4m/)
- `cpp/cli/`, `cpp/tests/`, `cpp/abi/` (donor's CLI + test harness + ABI golden symbol tables — useful for the M5+M6 session)
- `docs/`, `roadmap/`, `bindings/{python,r}`, `scripts/`, `cmake/`, `.github/`
- Top-level config files (ARCHITECTURE.md, CHANGELOG.md, CITATION.cff, README.md, etc.)
- Total: ~815 files preserved as donor archeology

Final cleanup of `_donor/` lands when the donor repo is renamed in M16
(at which point the prefix becomes redundant — everything is in the
unified repo by name).
