# Phase A — Rename cleanup (non-destructive big-bang)

> Status: **landed on `main`** · 2026-05-23
>
> Canonical spec: [`docs/REFACTOR_PLAN.md`](../docs/REFACTOR_PLAN.md) §3.
> Founding doc for the broader refactor. Phases B–F follow.

## Goal

A single `n4m` namespace, no remaining `p4a` paths or symbols, no
duplicated folders. Preserve the numerical core (`cpp/src/core/`) and
the parity fixtures (`parity/fixtures/`); wipe duplicated ABI wrappers,
binding stubs, old CLI, and obsolete docs.

## Outcome

Phase A landed across **5 commits** on `main`:

| Commit | Scope |
|--------|-------|
| `75df87a` Baseline | Pre-Phase-A state staged + `REFACTOR_PLAN.md` + 660 MB of generated data gitignored |
| `252248f` A1 | 3808 token substitutions `p4a_*/pls4all_*` → `n4m_*/n4m_*` across 343 files via `scripts/migrate_p4a_to_n4m.py` |
| `f993b91` A2–A13 | Structural renames, single `c_api/` surface, unified CMake |
| `ba05aac` A14–A26 | Docs, issue/PR templates, dev shell, doctor, bootstrap scripts |
| `4762df8` A27 | Final audit — restored PLS umbrella, build green, snapshot refreshed |

## Per-task summary

### A1 — token migration

`scripts/migrate_p4a_to_n4m.py --apply --stage all` followed by a second
pass over the root (`CMakeLists.txt`, `CMakePresets.json`, `scripts/`,
`.github/`, `cmake/`). Total **3808 substitutions across 343 files**.
Skipped intentional carve-outs: `README.md`, `CHANGELOG.md`,
`ARCHITECTURE.md`, `DISTRIBUTION.md`, `ROADMAP.md`, `Backlog.md`,
`Overview.md`, `CITATION.cff`.

### A2–A6 — header / source / cli / tests / abi swaps

| | Action |
|---|---|
| A2 | Deleted `cpp/include/pls4all/` (`p4a.h`, `p4a_version.h`, `p4a_export.h.in`) |
| A3 | **Revised vs plan**: instead of deleting `cpp/src/c_api/` whole, merged `cpp/src/c_api_n4m/*.cpp` into `cpp/src/c_api/` (the two folders covered disjoint symbol surfaces, both linked into `libn4m.so`). Adopted the more elaborate `n4m_linux.map` from `c_api_n4m/` (carries the `N4M_1` version tag) |
| A4 | Deleted `cpp/cli/p4a_cli.cpp`; promoted `cpp/cli_n4m/n4m_cli.cpp` to `cpp/cli/n4m_cli.cpp` |
| A5 | Deleted `cpp/tests/` (legacy); promoted `cpp/tests_n4m/` to `cpp/tests/` (26 files) |
| A6 | Renamed `cpp/abi_n4m/` → `cpp/abi/` |

### A7–A11 — binding consolidation

| | Action |
|---|---|
| A7 | Deleted `bindings/python_pls4all/`, `bindings/python_nirs4all_methods/` stubs |
| A8 | Consolidated `bindings/python_n4m/src/n4m/` into `bindings/python/src/n4m/`; mature `bindings/python/src/pls4all/` kept (subset target). Dropped `_original_donor_python` merge artefact |
| A9 | Deleted `bindings/r_n4m/`, `bindings/r_pls4all/` stubs |
| A10 | Renamed `bindings/r/pls4all/` → `bindings/r/n4m/`. Token-renamed inside: R Package name, `useDynLib`, all `r_pls4all_*` / `pls4all_*` identifiers, S3 method classes, .Rd / .Rmd file names. github.com URL + Authors@R contributor tag preserved (project-identity carve-outs). 55 files rewritten + 28 follow-up files |
| A11 | Archived `bindings/{go,rust,dotnet,lua,nim,ruby}/` to `bindings/_archive/` |

### A12–A13 — CMake unification

Rewrote `cpp/src/n4m_targets.cmake` to a single-libn4m model: one
`n4m_core` OBJECT lib (defined in `cpp/src/CMakeLists.txt`, extended
with donor sources via `target_sources()` from `n4m_targets.cmake`),
one `n4m_c` shared lib globbing `cpp/src/c_api/*.cpp`, and an explicit
`n4m_c_static` when `N4M_BUILD_STATIC=ON`. Removed the dead
`_n4m_c_sources` var, the unused `n4m_apply_c_target()` function, and
the broken self-aliasing `add_library(n4m_c ALIAS n4m_c)`.

Renamed:
- `cmake/Pls4allConfig.cmake.in` → `cmake/N4mConfig.cmake.in`
- `cmake/Pls4allOptions.cmake` → `cmake/N4mOptions.cmake`
- All `PLS4ALL_*` cache vars → `N4M_*` (`PLS4ALL_WITH_BLAS`,
  `PLS4ALL_BUILD_SHARED`, ...), `Pls4all`-prefixed function/variable
  names → `N4m*` / `n4m_*`, `_PLS4ALL_*_INCLUDED` guards → `_N4M_*`,
  `project(pls4all)` → `project(n4m)`

### A14–A15 — cleanup

- Root planning docs (ARCHITECTURE / Backlog / Overview / ROADMAP)
  already deleted in baseline (now under `docs/`).
- Wiped merge artefacts: `pls4all.Rcheck/`,
  `bindings/_catalog/`, `bindings/r/pls4all.Rcheck/`,
  `bindings/r/pls4all_0.97.0.tar.gz`, `objectifs.txt`, `report.log`.

### A16–A17 — README + CONTRIBUTING

- `README.md`: header rewritten for `nirs4all-methods (n4m)`,
  Python/R/MATLAB quickstarts updated for the `n4m/` layout +
  `N4M_*` env vars, `pls4all` flagged as a slim subset package, link
  to `docs/REFACTOR_PLAN.md`.
- `CONTRIBUTING.md`: full rewrite around the closed-lib + PR-only
  extensibility model. Includes the triage table mapping issue
  templates → PR templates, devcontainer as the primary setup path,
  per-PR §2.10 invariants checklist.

### A18–A19 — issue + PR templates

8 issue templates under `.github/ISSUE_TEMPLATE/`:
`method-request`, `method-update`, `external-reference-request`,
`binding-request`, `binding-update`, `subset-request`, `bug-report`,
`parity-discrepancy`, plus `config.yml` to disable blank issues.

7 PR templates under `.github/PULL_REQUEST_TEMPLATE/`:
`method-add`, `method-update`, `external-reference`, `binding-add`,
`binding-update`, `subset`, `parity-fix`, plus a top-level generic
fallback `PULL_REQUEST_TEMPLATE.md`. Each ends with the §2.10
invariants checklist.

### A20–A24 — dev shell

- `.devcontainer/`: Ubuntu 24.04 image (C++17 + CMake + Ninja +
  OpenBLAS + Python 3.12 + uv + R 4.4 + Octave 9 + Node 22 +
  Emscripten 3.1.74 + Docker CLI), `devcontainer.json`,
  `docker-compose.yml` (named volumes for ccache / uv / R / npm /
  docker.sock), `post-create.sh` (rustup + juliaup best-effort,
  R site library pre-warm).
- `scripts/doctor.sh` — per-category required/optional tool check
  with OS-specific install hints; non-zero exit on missing required.
- `scripts/bootstrap-bare-metal-{linux,macos}.sh` + `windows.ps1`
  — apt / brew / choco best-effort installers.
- `Makefile` + `scripts/bootstrap.sh` — `make bootstrap` (devcontainer
  if Docker available, else bare-metal), `make shell` (proxy for
  `docker compose run --rm dev bash`), `make doctor`.
- `.github/codespaces/devcontainer.json` mirrors the local devcontainer.

### A25 — MATLAB/Octave policy

`bindings/matlab/COMPAT.md` declares the MATLAB↔Octave divergence
table (empty at v1; populated by the conformance runner in
F-bootstrap). CI runs Octave only (per `REFACTOR_PLAN.md` §1.1ter).

### A26 — CLAUDE.md

Rewritten for the post-Phase-A state: single libn4m target table,
new repo layout, dev-shell entry points, "things that will trip up
a fresh agent" updated around the completed rename + remaining
intentional `pls4all` references (slim subset, github URL, R
contributor tag).

### A27 — final audit

- **Build green**: `cmake --preset dev-debug && cmake --build`
  produces `libn4m.so.1.9.0`, `libn4m_static.a`, `n4m_cli`, `n4m_tests`.
- **Tests green**: `ctest --preset dev-debug --output-on-failure` →
  1/1 passed (266 doctest sub-cases).
- `n4m_cli --selfcheck` → OK; `--abi-info` clean; `--version` clean.
- ABI snapshot regenerated (`cpp/abi/expected_symbols_linux.txt`,
  670 symbols). `abi-check.yml` updated to strip the `@@N4M_1`
  version-tag suffix before diffing (the new linker version script
  attaches a tag to every export).
- `grep -rE '\b(p4a_|P4A_)\b'` in active code → 0 matches outside
  `bindings/_archive/`, `CHANGELOG.md`, historical `docs/merge/` and
  `docs/reviews/`, and the `migrate_p4a_to_n4m.py` comments.
- `make doctor` → all required tools resolved on the maintainer's
  bare metal.

## Header-recovery note (judgment call)

A2 (delete `cpp/include/pls4all/`) collided with `cpp/src/core/model.cpp`'s
transitive dependency on the PLS-domain declarations (`n4m_algorithm_t`,
`n4m_solver_t`, `N4M_MODEL_*` enums, the full `n4m_model_array_t`
getter family, `n4m_config_*`, `n4m_pipeline_*`, etc.) which the
donor-side `cpp/include/n4m/n4m.h` umbrella had pruned in its
"Phase 3 trim" (deferred to later phases).

To unblock the build without reverting A2, the deleted header is
restored as `cpp/include/n4m/pls.h` (post-rename N4M_* content) and
included unconditionally from `n4m.h` at the very end (so `pls.h`'s
static_asserts see the base types defined in `n4m.h` first).

**Phase B will re-split this into per-category headers per
`docs/REFACTOR_PLAN.md` §2.1** (`n4m.h` umbrella + `n4m_pls.h` +
`n4m_preprocessing.h` + `n4m_selection.h` + etc.). For now, the
2-header (`n4m.h` + `pls.h`) shape is the simplest stable state.

## Intentional `pls4all` carve-outs

These references stay until a separate decision retires them:

| Where | Why |
|-------|-----|
| `bindings/python/src/pls4all/` | Mature slim subset package (the long-standing `pls4all` PyPI name) — `n4m`'s pure PLS surface |
| `bindings/r/n4m/` Authors@R: `person("pls4all", "contributors", ...)` | Historical contributor tag — preserved per CRAN convention |
| URLs `github.com/GBeurier/pls4all` | The GitHub repo itself hasn't been renamed yet |
| `catalog/subsets/pls4all.yaml` | The packaging-subset manifest for the PLS-only re-export (Phase E) |
| `bindings/_archive/{go,rust,dotnet,lua,nim,ruby}/*` | Frozen PoCs — preserved as-is for future revival |
| `docs/merge/`, `docs/reviews/`, `CHANGELOG.md` | Historical record |

## Phase A exit criterion status

> `cmake --preset dev-release && ctest` is green inside the devcontainer
> (and on bare metal for the maintainer), `nm` on `libn4m.so` contains no
> `p4a_*`, the repo `tree -L 2` shows no legacy names, `make doctor` runs
> clean in a fresh devcontainer, every method-/binding-/subset-related
> issue and PR template renders correctly on GitHub.

- ✅ `cmake --preset dev-debug && ctest` green on bare metal (devcontainer build deferred to next image pull)
- ✅ `nm -D libn4m.so.1.9.0` exports 670 symbols, all `n4m_*` (no `p4a_*`); plus the `N4M_1` version tag
- ✅ `tree -L 2 bindings cpp` shows no `_n4m` / `p4a` / `cli_n4m` / `tests_n4m` / `abi_n4m` / `c_api_n4m` / `python_n4m` / `python_nirs4all_methods` / `python_pls4all` / `r_n4m` / `r_pls4all`
- ✅ `make doctor` green
- ✅ All templates present and YAML-valid
