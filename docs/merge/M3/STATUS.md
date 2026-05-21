# M3 — donor non-PLS source move: STATUS

**Date**: 2026-05-21 (same merge sprint)
**Branch**: `merge/import-donor`
**Pushed**: pending commit (this report ships with it)

## What was done

222 donor source files were `git mv`-moved from
`_donor/nirs4all-methods/cpp/src/core/{preprocessing,augmentations,splitters,filters,utilities}/`
to their final positions under `cpp/src/core/<category>/`. Blame chain
preserved via `git mv` for every file (status `R` in `git status`).

| Category move | Files | From | To | Note |
|---|--:|---|---|---|
| Preprocessing | 94 | `_donor/.../cpp/src/core/preprocessing/{11 subcats}/` | `cpp/src/core/preprocessing/{11 subcats}/` | per-subcategory move (baselines, derivatives, feature_selection, orthogonalization, resampling, scaling, scatter, signal_conversion, smoothing, specialized, wavelets) |
| Augmentation | 82 | `_donor/.../cpp/src/core/augmentations/{10 subcats + 2 root files}/` | `cpp/src/core/augmentation/{10 subcats}/` + 2 root files | **plural → singular** rename per §11/M3 decision (drift, edge_artifacts, environmental, mixup, noise, random, scattering, spectral, splines, wavelength) |
| Splitters | 20 | `_donor/.../cpp/src/core/splitters/` | `cpp/src/core/splitters/` | flat |
| Filters | 16 | `_donor/.../cpp/src/core/filters/` | `cpp/src/core/filters/` | includes `_vendored/` subdir |
| Utilities | 10 | `_donor/.../cpp/src/core/utilities/` | `cpp/src/core/utilities/` | flat |
| **Total** | **222** | | | |

## Common-core duplicates deleted (M3.2)

Files that had landed at `cpp/src/core/common/` during M2.5 were removed
from their donor location:

```
_donor/nirs4all-methods/cpp/src/core/context.cpp           (deleted)
_donor/nirs4all-methods/cpp/src/core/context.hpp           (deleted)
_donor/nirs4all-methods/cpp/src/core/status.cpp            (deleted)
_donor/nirs4all-methods/cpp/src/core/status.hpp            (deleted)
_donor/nirs4all-methods/cpp/src/core/matrix_view.cpp       (deleted)
_donor/nirs4all-methods/cpp/src/core/matrix_view.hpp       (deleted)
_donor/nirs4all-methods/cpp/src/core/version.cpp           (deleted)
_donor/nirs4all-methods/cpp/src/core/version.hpp           (deleted)
_donor/nirs4all-methods/cpp/src/core/parallel.hpp          (deleted)
_donor/nirs4all-methods/cpp/src/core/linalg.hpp            (deleted)
_donor/nirs4all-methods/cpp/src/core/common/               (entire subtree deleted — landed at cpp/src/core/common/ in M2.5)
```

After M3, `_donor/nirs4all-methods/cpp/src/core/` is **empty** (verified
by `find ... -type f` returning 0).

## Stale-reference cleanup (M3.3)

| File | Reference | Action |
|------|-----------|--------|
| `_donor/.../parity/python_generator/src/n4m_parity_aug_spectral_ref/__init__.py` | `cpp/src/core/augmentations/{wavelength, spectral}/` (comment) | Updated to `cpp/src/core/augmentation/...` (singular) |
| `_donor/.../bindings/r/n4m/src/libn4m/src/c_api/c_api_preprocessing.cpp` | `cpp/src/core/preprocessing/...` (comment) | Already correct — no change |
| `_donor/.../bindings/r/n4m/src/libn4m/src/c_api/c_api_resampling.cpp` | `cpp/src/core/preprocessing/resampling/` (comment) | Already correct — no change |
| `_donor/.../cpp/src/CMakeLists.txt` | 89 `core/augmentations/...` and `core/preprocessing/...` source entries | **Marked BROKEN** with a header note; not rewritten (this CMakeLists is dormant — pls4all does not activate it, and the unified CMakeLists lands in M5/M6) |

## Build verification (M3.4)

`pls4all` build and tests **remain green** after the M3 moves, because the
M3 moves are purely additive from pls4all's build-system perspective:

- pls4all's `cpp/src/CMakeLists.txt` enumerates explicit source files under
  `cpp/src/core/<file>.cpp` (no `file(GLOB ...)` patterns), and the source
  list is unchanged.
- The moved donor sources now live at `cpp/src/core/<category>/<sub>/<file>.{c,h}`,
  paths that pls4all's CMakeLists does **not** reference.
- The deleted `_donor/.../cpp/src/core/{context,status,matrix_view,version,parallel,linalg.hpp,common/}`
  copies were not consumed by anything in pls4all's build either.

```
$ cmake --build --preset dev-release
ninja: no work to do.
$ ./build/dev-release/cpp/tests/pls4all_tests
=== 265 run, 0 failures, 0 skipped ===
```

## Roadmap gate evaluation

The roadmap M3 gate (`MERGE_ROADMAP.md` §9 row M3):

> donor fixtures still pass against the moved sources; `git blame -C -C -C`
> chains back to donor commits

| Sub-gate | Status | Notes |
|---|---|---|
| `git mv` preserves blame chain | ✅ verified | All 222 moves are `R` status (rename) in `git status`. `git blame -C -C -C cpp/src/core/preprocessing/baselines/airpls.c` reaches donor's original commits. |
| Donor fixtures pass against moved sources | ⚠️ deferred | The donor's C++ test binary (`n4m_tests` at `_donor/.../cpp/tests/`) is not wired into pls4all's build yet. The unified test runner lands in M5/M6. The Python parity generator (the M2.5-aware part) sees no path drift because Python imports work by symbol, not by C source path. |
| pls4all-fixture C++ tests still pass | ✅ 265/265 | unchanged by M3 |

## File-count budget after M3

```
cpp/src/core/
├── common/                    31 files       (landed in M2.5)
├── preprocessing/{11 subs}/   94 files       ← NEW from donor in M3
├── augmentation/{10 subs}/    82 files       ← NEW from donor in M3
├── splitters/                 20 files       ← NEW from donor in M3
├── filters/                   16 files       ← NEW from donor in M3 (incl. _vendored)
├── utilities/                 10 files       ← NEW from donor in M3
├── <models/, selection/, diagnostics/, aom_pop/, transfer/, pipeline/>     placeholders only (target for M4 split of pls4all's model.cpp + extra_pls.cpp)
└── <pls4all flat sources at this level>      58 files (unchanged in M3 — M4 splits them)
```

Total under `cpp/src/core/`: ~311 files post-M3 (was ~89 after M2).

## What is NOT done in M3

Per roadmap §9, the following stay open and land later:

- **M4 — pls4all source split**: 6188 lines across `model.cpp` (4541 L) +
  `extra_pls.cpp` (1647 L) split into ~28 target files. Detailed plan at
  `docs/merge/M4/SPLIT_PLAN.md` (produced by an Explore agent
  in parallel with M3).
- **M5 — ABI rename**: `p4a_*` → `n4m_*` mechanical pass. Until then,
  pls4all's flat sources still use `p4a_*` symbols.
- **M6 — unified C ABI headers**: split umbrella header into per-category
  headers.
- **M7 — unified parity gate**: wires both halves into one runner.
- The donor's broken `_donor/.../cpp/src/CMakeLists.txt` is marked broken
  but kept for archeology; deletion happens in M6 when the unified build
  wiring lands.

## Risk register update

| Risk | Status |
|---|---|
| Move breaks pls4all build | ✅ Confirmed safe (additive only — pls4all CMakeLists references unchanged) |
| Move breaks donor's standalone build | Accepted — donor build was already dormant in pls4all repo; flagged in CMakeLists header |
| Blame loss on rename | Mitigated — `git mv` used throughout; status shows R (rename), not D+A |
| Plural-singular rename causes downstream confusion | Mitigated — singular `augmentation` adopted everywhere; only stale reference (the Python parity doc-string) updated |

## Next: M4 — pls4all source split

Plan ready at `docs/merge/M4/SPLIT_PLAN.md`. 7-phase extraction
(~34 commits), starting with helper-extraction → core PLS → derived →
classification → sparse → specialized/ensembles/transfer → core dispatch.
