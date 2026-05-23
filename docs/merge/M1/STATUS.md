# M1 — import donor: STATUS

**Date**: 2026-05-21 (continuing from M0 same day)
**Branch**: `merge/import-donor`
**Anchor**: branched off `pls4all-final/v1.0.0-pre-merge` (`3b99c0b`)
**Merge commit SHA**: `2ca4aa8f681472817e15288f8f83a8ff3ca3e968`
**Pushed**: `origin/merge/import-donor` = `2ca4aa8` ✅

## What was done

| # | Action | Outcome |
|--:|---|---|
| 1 | Branch `merge/import-donor` from `pls4all-final/v1.0.0-pre-merge` | ✅ HEAD now at clean baseline (`3b99c0b`) |
| 2 | Add donor as local-path remote: `git remote add donor /home/delete/nirs4all/nirs4all-methods/` | ✅ |
| 3 | `git fetch donor` | ✅ 45 commits + 18 archive tags fetched |
| 4 | `git subtree add --prefix=_donor/nirs4all-methods donor main` | ✅ Merge commit `2ca4aa8` created |
| 5 | `cmake --preset dev-release` configure | ✅ no collision; pls4all 0.97.3 configured cleanly |
| 6 | `cmake --build --preset dev-release -j 4` | ✅ 24 build steps; libn4m.so.1.16.0 + n4m_cli + n4m_tests + libp4a_static.a all linked |
| 7 | `ctest --preset dev-release` (n4m_tests binary) | ✅ **265 run, 0 failures, 0 skipped** |
| 8 | `n4m_cli --selfcheck` | ✅ "selfcheck OK", project version `0.97.3+abi.1.16.0` |
| 9 | `git push -u origin merge/import-donor` | ✅ |

## Merge commit structure

```
commit 2ca4aa8f681472817e15288f8f83a8ff3ca3e968
parents: 3b99c0b4a348c271b22deff1c4822709522f451b   (pls4all baseline: "Publish partial benchmark dashboard refresh")
         6195574a64b167cabdfa8d9a1d8bf3cd706a80da   (donor M0 baseline: "M0 baseline: unblock donor CI ahead of merge")

Add '_donor/nirs4all-methods/' from commit '6195574a64b167cabdfa8d9a1d8bf3cd706a80da'

git-subtree-dir: _donor/nirs4all-methods
git-subtree-mainline: 3b99c0b4a348c271b22deff1c4822709522f451b
```

Both histories are reachable from the merge commit. `git log --first-parent` walks pls4all history; `git log` (default = follow all parents) walks both.

## Tree layout after import (verified)

```
/home/delete/nirs4all/pls4all/                         (pls4all root — untouched)
├── cpp/include/pls4all/p4a.h        (86.8 KB, original)
├── bindings/                        (14 sub-bindings)
├── parity/                          (pls4all parity)
├── benchmarks/                      (pls4all benchmarks)
├── docs/, roadmap/, scripts/        (pls4all)
└── _donor/nirs4all-methods/         (NEW — fully donor tree)
    ├── cpp/include/n4m/n4m.h        (136.8 KB, donor ABI)
    ├── bindings/                    (python, r)
    ├── parity/                      (donor parity gate)
    ├── benchmarks/                  (donor benchmarks)
    ├── docs/, roadmap/, scripts/    (donor)
    ├── CMakeLists.txt               (donor — NOT activated by pls4all's root CMakeLists)
    └── ARCHITECTURE.md, CHANGELOG.md, README.md, …
```

## Roadmap gate satisfied

> M1 gate per `MERGE_ROADMAP.md` row M1: *"both symbol tables co-exist; ctest can build pls4all only at this stage"*

- Both ABI symbol tables coexist on disk: pls4all `n4m_*` + donor `n4m_*` (each in its own header tree)
- pls4all builds and tests at 265/265 green
- Donor tree is dormant — neither pls4all's `CMakeLists.txt` nor any other build glue picks it up yet. That's correct for M1; M2.5 will start consolidating the common-core under `cpp/src/core/common/`.

## What is NOT done (intentionally, per the roadmap)

- Donor tree's CMakeLists not wired into pls4all's build (M2.5 will start that)
- No source moves yet (M3 moves donor non-PLS sources out from `_donor/` to final paths; M4 splits pls4all's `model.cpp` etc.)
- No ABI rename (M5)
- No catalog yet (M2 lands it)

## Untracked working-tree noise

~596 files showing as `??` in `git status`. These are leftover oracle dumps + `.x_target/` from the earlier `codex/parity-30x30-dashboard-fixes` checkout. They are NOT tracked by any branch — they are just orphan files on disk. They will be cleaned up either:
- by checking out a clean state after work resumes, or
- by the M2 catalog scripts when they run their own `.gitignore` propagation

No impact on the merge or build.

## Next: M2 — target tree skeleton

- Create the namespaced category dirs (per `MERGE_ROADMAP.md` §2)
- Land `catalog/schema/method_v1.json` (the expanded schema from §5.4)
- Auto-generate `catalog/methods.yaml` from both source trees
- Land the 7 subset YAMLs (2 active: `nirs4all_methods` + `pls4all`; 5 planned: `aompls`, `nirs4all_preprocessing`, `nirs4all_augmentation`, `nirs4all_selection`, `n4m_transfer`)
- Write `catalog/scripts/{validate_catalog,build_closure,render_subset}.py`
- Gate: `validate_catalog.py` passes; the two active subsets close
