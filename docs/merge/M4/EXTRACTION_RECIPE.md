# M4 — pls4all source split: EXTRACTION RECIPE

**Status**: scaffolded — 39 target stub files landed at their final paths.
The actual function-body extraction is deferred to a focused refactor
session because each cut must keep `n4m_tests` (265/265) green
between commits and the surgery is genuinely 4 days of careful work per
the roadmap budget. This document is the maintenance protocol that
guides that session.

Source plan: `docs/merge/M4/SPLIT_PLAN.md` (produced by the
Explore agent in parallel with M3).

## Pre-flight per-extraction checklist

Before any single extraction commit:

1. `n4m_tests` is currently green (`./build/dev-release/cpp/tests/n4m_tests` reports `265 run, 0 failures, 0 skipped`).
2. Work on the `merge/import-donor` branch.
3. The pls4all CMakeLists at `cpp/src/CMakeLists.txt` still references the source monoliths (`core/model.cpp`, `core/extra_pls.cpp`).

## Phase 1 — extract helpers FIRST (load-bearing)

The agent's analysis flags four cross-cutting helper blocks that MUST be
extracted before any algorithm files, because anonymous-namespace
helpers cannot be called across translation units:

| Helper block | Lines in source | Target file | Linkage transition |
|--------------|-----------------|-------------|---------------------|
| model.cpp anon-namespace (L18–L1524, ~1500 lines) — `center_scale_in_place`, `copy_matrix_checked`, `validate_fit_request`, `invert_square`, `dot`, `matrix_vector_product`, `read_value`/`write_value` | model.cpp:18-1524 | `cpp/src/core/models/core/_common.{cpp,hpp}` | anonymous → `n4m::core::detail::` (or `n4m::core::detail::` pre-M5) namespace; declare in `_common.hpp` |
| model.cpp PLS-shared helpers (`simple_simpls` is the cross-call from extra_pls; `dominant_svd_pair`, `largest_symmetric_eigenvector`, `canonicalize_direction_sign`, `canonicalize_svd_pair_sign`) | model.cpp:281-1524 selectively | `cpp/src/core/models/pls/_common.{cpp,hpp}` | same |
| extra_pls.cpp anon-namespace (L18–L306, ~290 lines) — `soft_threshold`, `column_means`, `subtract_means`, `copy_matrix`, `squared_norm` | extra_pls.cpp:18-306 | `cpp/src/core/models/sparse/_common.{cpp,hpp}` | same |

**Critical: `simple_simpls`** (extra_pls.cpp:97) is called by both
`fit_pls_sparse_simpls` (model.cpp) AND extra_pls sparse variants. It
MUST be extracted to a shared location accessible to both — the
recommended location is `models/pls/_common.{cpp,hpp}` (since SIMPLS is
canonically PLS-side).

### Phase 1 commit pattern

```
M4 phase 1: extract models/core/_common — N anonymous-namespace helpers
  - cpp/src/core/models/core/_common.cpp (+~300 lines)
  - cpp/src/core/models/core/_common.hpp (+~80 lines, declarations)
  - cpp/src/core/model.cpp (−~300 lines, replaced by include of _common.hpp)
  - cpp/src/CMakeLists.txt (+1 entry: core/models/core/_common.cpp)
  - Verify: n4m_tests 265/265 still green
```

Repeat for `pls/_common` and `sparse/_common`. After Phase 1 (3
commits), 50 % of model.cpp's bulk is extracted but no functional
change.

## Phase 2 — core PLS solvers (8 commits)

In topological order (dependencies first):

| Order | Function | Source lines | Target | Helper deps |
|--:|---|--:|---|---|
| 1 | `fit_pls_canonical_nipals + _svd` | model.cpp:1819-1838 | `models/pls/canonical.cpp` | `core/_common`, `pls/_common` |
| 2 | `fit_pls_regression_nipals` | model.cpp:1528-1818 | `models/pls/nipals.cpp` | `core/_common`, `pls/_common` |
| 3 | `fit_pls_regression_simpls` | model.cpp:3685-3884 | `models/pls/simpls.cpp` | `core/_common`, `pls/_common` (simple_simpls) |
| 4 | `fit_pls_regression_svd` | model.cpp:2895-3060 | `models/pls/svd.cpp` | `core/_common`, `pls/_common` |
| 5 | `fit_pls_svd` (PLSSVD) | model.cpp:1839-1999 | `models/pls/plssvd.cpp` | `core/_common`, `pls/_common` |
| 6 | `fit_pls_regression_power` | model.cpp:3061-3226 | `models/pls/power.cpp` | `core/_common`, `pls/_common` |
| 7 | `fit_pls_regression_randomized_svd` | model.cpp:3227-3396 | `models/pls/randomized_svd.cpp` | `core/_common`, `pls/_common`, `common/svd.{c,h}` (post-M2.5) |
| 8 | `fit_pls_regression_kernel` | model.cpp:2615-2894 | `models/pls/kernel.cpp` | `core/_common`, `pls/_common` |
| 9 | `fit_pls_regression_orthogonal_scores` | model.cpp:2350-2614 | `models/pls/oscores.cpp` | `core/_common`, `pls/_common` |

## Phase 3 — derived (2 commits)

| # | Function | Source lines | Target |
|--:|---|--:|---|
| 1 | `fit_pcr_svd` | model.cpp:3397-3684 | `models/pcr/pcr.cpp` |
| 2 | `fit_opls_nipals` | model.cpp:2000-2349 | `models/opls/opls.cpp` |

## Phase 4 — classification (3 commits)

| # | Function | Source lines | Target |
|--:|---|--:|---|
| 1 | `fit_di_pls` | model.cpp:3921-4162 | `models/classification/pls_da.cpp` |
| 2 | `fit_pls_qda` | extra_pls.cpp:847-944 | `models/classification/pls_qda.cpp` |
| 3 | `fit_pls_cox` | extra_pls.cpp:945-1029 | `models/classification/pls_cox.cpp` |

## Phase 5 — sparse (4 commits)

| # | Function | Source lines | Target |
|--:|---|--:|---|
| 1 | `fit_pls_sparse_simpls` | model.cpp:4163-4387 | `models/sparse/sparse_simpls.cpp` |
| 2 | `fit_sparse_pls_da` | extra_pls.cpp:307-366 | `models/sparse/sparse_pls_da.cpp` |
| 3 | `fit_group_sparse_pls` | extra_pls.cpp:367-444 | `models/sparse/group_sparse.cpp` |
| 4 | `fit_fused_sparse_pls` | extra_pls.cpp:445-480 | `models/sparse/fused_sparse.cpp` |

## Phase 6 — specialized + ensembles + transfer (12 commits)

| # | Function | Source lines | Target |
|--:|---|--:|---|
| 1 | `fit_cppls` | extra_pls.cpp:481-550 | `models/specialized/cppls.cpp` |
| 2 | `fit_weighted_pls` | extra_pls.cpp:551-616 | `models/specialized/weighted.cpp` |
| 3 | `fit_robust_pls` | extra_pls.cpp:617-677 | `models/specialized/robust_huber.cpp` |
| 4 | `fit_ridge_pls` | extra_pls.cpp:678-736 | `models/specialized/ridge.cpp` |
| 5 | `fit_continuum_regression` | extra_pls.cpp:737-799 | `models/specialized/continuum.cpp` |
| 6 | `fit_missing_aware_nipals` | extra_pls.cpp:1333-1397 | `models/specialized/missing_aware_nipals.cpp` |
| 7 | `fit_pls_glm` | extra_pls.cpp:800-846 | `models/glm/pls_glm.cpp` |
| 8 | `fit_mir_pls` | extra_pls.cpp:1243-1332 | `models/local/mir_pls.cpp` |
| 9 | `fit_pds` | extra_pls.cpp:1030-1140 | `transfer/pds.cpp` |
| 10 | `fit_ds` | extra_pls.cpp:1141-1242 | `transfer/ds.cpp` |
| 11 | `fit_bagging_pls + fit_boosting_pls + fit_random_subspace_pls` | extra_pls.cpp:1477-1647 | `models/ensembles/{bagging,boosting,random_subspace}.cpp` (3 commits) |

## Phase 7 — core dispatch (2 commits)

| # | Function | Source lines | Target |
|--:|---|--:|---|
| 1 | `predict_into + fit_model + transform_into + model_array` | model.cpp:3885-3920 and 4388-4541 | `models/core/dispatch.cpp` |
| 2 | `approximate_press` | extra_pls.cpp:1398-1476 | `models/core/approximate_press.cpp` |

## Final cleanup

After all 7 phases land:
- `cpp/src/core/model.cpp` reduces to either an empty file (and is deleted) or to a tiny placeholder that just re-includes the new files (for backward-compat).
- Same for `cpp/src/core/extra_pls.cpp`.
- `cpp/src/CMakeLists.txt` no longer references the two monoliths; instead it picks up `cpp/src/core/models/**/*.cpp` and `cpp/src/core/transfer/*.cpp` (still explicit listing, no GLOB).

## Per-commit gate

Each phase's individual commit must pass:

1. `cmake --build --preset dev-release` builds without errors.
2. `./build/dev-release/cpp/tests/n4m_tests` reports `265 run, 0 failures, 0 skipped`.
3. `./build/dev-release/cpp/cli/n4m_cli --selfcheck` reports `selfcheck OK`.
4. Optional: `objdump -T build/dev-release/cpp/src/libn4m.so.1 | grep ' n4m_' | wc -l` returns the same count as before the extraction (no symbol drop or duplication).

## What is in M4 today (this commit)

- **39 stub files** at all target paths under `cpp/src/core/models/{pls,opls,pcr,sparse,multiblock,local,ensembles,specialized,classification,glm,core}/` and `cpp/src/core/transfer/`. Each stub carries a structured header naming the origin source, the line range to extract, and the function(s) it will receive.
- **Two reports** (`SPLIT_PLAN.md` from the Explore agent + this `EXTRACTION_RECIPE.md`).
- **No changes** to `model.cpp`, `extra_pls.cpp`, or `cpp/src/CMakeLists.txt`. pls4all builds unchanged, `n4m_tests` 265/265 green.

## What is NOT done in M4

- The actual function-body extraction (Phases 1–7 above). The 28 algorithm
  extractions plus 3 helper extractions plus the cleanup are roadmap-budgeted
  at 4 days of careful work. They are deferred to a focused refactor session
  that can dedicate a build/test cycle per extraction.
- Multiblock subcategory has no stubs because all multiblock methods (`mb_pls`,
  `n_pls`, `o2pls`, `so_pls`, `on_pls`, `rosa`, `multiblock_extensions`) live
  in separate `cpp/src/core/*.cpp` files already — they need re-categorisation
  rather than extraction. Those moves are simpler and can land in M5/M6.

## Sequencing within the merge

Per `MERGE_ROADMAP.md` §9 critical path:
`M3 → M4 → M5 → M6 → M7 → M8 → ...`

The M4 scaffold unblocks M5 (ABI rename) because M5 operates on the
monoliths in place. After M5, the post-rename monoliths can be extracted
into the M4 stubs as planned here. The execution becomes:

1. M5 mechanical `n4m_* → n4m_*` rename on `model.cpp`, `extra_pls.cpp`
   in place.
2. M4 Phase 1 (extract helpers) on the renamed monoliths.
3. M4 Phases 2–7 on each algorithm.

This is the order the agent's plan implies.
