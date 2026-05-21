# M2.5 — common-core unification: RECONCILIATION REPORT

**Date**: 2026-05-21 (same merge sprint as M0/M1/M2)
**Branch**: `merge/import-donor`
**Tip**: pushed as part of the M2.5 commit (this report ships with it)

## Decision summary

The donor (`nirs4all-methods`) and the base (`pls4all`) both ship a
"common-core" of infrastructure: `context`, `status`, `matrix_view`,
`version`, `parallel`, `linalg.hpp`. **A side-by-side comparison shows
the implementations are 100 % equivalent — the only difference is the
symbol/namespace prefix (`p4a_*` / `pls4all::core` vs `n4m_*` /
`n4m::core`).** Enums, struct fields, error-buffer size, and backend
codes all map 1-for-1.

Per roadmap §11.4, the canonical prefix going forward is **`n4m_*`**
(versioned ELF symbols `@@N4M_1` on Linux). The donor's files are
therefore the unified winners as-is — no semantic rewrite needed.

## Comparison evidence

```
context.cpp      pls4all=2632  donor=2620   delta=12 bytes (p4a_ vs n4m_ on each token)
context.hpp      pls4all=3173  donor=3157   delta=16 bytes (same prefix-only diff)
status.cpp       pls4all=1752  donor=1744   delta=8  bytes
status.hpp       pls4all=437   donor=425    delta=12 bytes
matrix_view.cpp  pls4all=1484  donor=1476   delta=8  bytes
matrix_view.hpp  pls4all=948   donor=932    delta=16 bytes
version.cpp      pls4all=1110  donor=1098   delta=12 bytes
version.hpp      pls4all=995   donor=979    delta=16 bytes
parallel.hpp     pls4all=1422  donor=1410   delta=12 bytes
linalg.hpp       pls4all=10968 donor=10936  delta=32 bytes
```

After normalising the pls4all-side prefix (`p4a/n4m`, `P4A/N4M`,
`pls4all::core/n4m::core`, `pls4all::cuda_dispatch/n4m::cuda_dispatch`,
`pls4all/p4a.h/n4m/n4m.h`) the diffs vanish on every pair. The pls4all
files are a mechanical rename of the same source.

## Enum reconciliation

| Code | Donor | pls4all |
|------|-------|---------|
| 0    | `N4M_OK` | `P4A_OK` |
| 1    | `N4M_ERR_INVALID_ARGUMENT` | `P4A_ERR_INVALID_ARGUMENT` |
| 2    | `N4M_ERR_NULL_POINTER` | `P4A_ERR_NULL_POINTER` |
| 3    | `N4M_ERR_SHAPE_MISMATCH` | `P4A_ERR_SHAPE_MISMATCH` |
| 4    | `N4M_ERR_DTYPE_MISMATCH` | `P4A_ERR_DTYPE_MISMATCH` |
| 5    | `N4M_ERR_STRIDE_INVALID` | `P4A_ERR_STRIDE_INVALID` |
| 6    | `N4M_ERR_NOT_FITTED` | `P4A_ERR_NOT_FITTED` |
| 7    | `N4M_ERR_NUMERICAL_FAILURE` | `P4A_ERR_NUMERICAL_FAILURE` |
| 8    | `N4M_ERR_CONVERGENCE_FAILED` | `P4A_ERR_CONVERGENCE_FAILED` |
| 9    | `N4M_ERR_OUT_OF_MEMORY` | `P4A_ERR_OUT_OF_MEMORY` |
| 10   | `N4M_ERR_UNSUPPORTED` | `P4A_ERR_UNSUPPORTED` |
| 11   | `N4M_ERR_NOT_IMPLEMENTED` | `P4A_ERR_NOT_IMPLEMENTED` |
| 12   | `N4M_ERR_ABI_MISMATCH` | `P4A_ERR_ABI_MISMATCH` |
| 13   | `N4M_ERR_IO` | `P4A_ERR_IO` |
| 14   | `N4M_ERR_CORRUPT_BUFFER` | `P4A_ERR_CORRUPT_BUFFER` |
| 15   | `N4M_ERR_VERSION_INCOMPATIBLE` | `P4A_ERR_VERSION_INCOMPATIBLE` |
| 16   | `N4M_ERR_BACKEND_UNAVAILABLE` | `P4A_ERR_BACKEND_UNAVAILABLE` |
| 17   | `N4M_ERR_CANCELLED` | `P4A_ERR_CANCELLED` |
| 255  | `N4M_ERR_INTERNAL` | `P4A_ERR_INTERNAL` |

Identical numeric values across the board. M5 (ABI rename) will collapse
this into the `n4m_*` names; until then the M5-generated mapping table
(in `scripts/migrate_p4a_to_n4m.py`) treats them as straight renames.

## Backend enum reconciliation

| Code | Name | Same in both sides |
|------|------|---|
| 0    | `*_BACKEND_AUTO` | ✓ |
| 1    | `*_BACKEND_REFERENCE_CPU` | ✓ |
| 2    | `*_BACKEND_NATIVE_CPU` | ✓ |
| 3    | `*_BACKEND_BLAS` | ✓ |
| 4    | `*_BACKEND_OPENMP` | ✓ |
| 5    | `*_BACKEND_CUDA` | ✓ |

## Struct reconciliation (`matrix_view_t`)

Both repos define this struct with **identical field order, types, and
semantics**:

```c
struct *_matrix_view_t {
    void*        data;
    int64_t      rows;
    int64_t      cols;
    int64_t      row_stride;   /* elements between row i and i+1 */
    int64_t      col_stride;   /* elements between col j and j+1 */
    *_dtype_t    dtype;
    int32_t      reserved0;    /* keep struct size stable */
};
```

`row_stride` / `col_stride` field names match (a verification the
roadmap M2.5 row called out specifically). `reserved0` is in the same
position. The struct is wire-compatible across the two implementations.

## Context error-buffer

Both repos document a **4 KiB** per-context error buffer with
`snprintf`-style writes and NUL-termination. Same size, same write
discipline. No reconciliation needed.

## Files landed

Donor's versions (already `n4m_*`) copied verbatim to the unified
target locations:

| File | Source (donor) | Target (this repo) |
|------|----------------|--------------------|
| context.cpp | `_donor/nirs4all-methods/cpp/src/core/context.cpp` | `cpp/src/core/common/context.cpp` |
| context.hpp | `_donor/nirs4all-methods/cpp/src/core/context.hpp` | `cpp/src/core/common/context.hpp` |
| status.cpp | `_donor/nirs4all-methods/cpp/src/core/status.cpp` | `cpp/src/core/common/status.cpp` |
| status.hpp | `_donor/nirs4all-methods/cpp/src/core/status.hpp` | `cpp/src/core/common/status.hpp` |
| matrix_view.cpp | donor → | `cpp/src/core/common/matrix_view.cpp` |
| matrix_view.hpp | donor → | `cpp/src/core/common/matrix_view.hpp` |
| version.cpp | donor → | `cpp/src/core/common/version.cpp` |
| version.hpp | donor → | `cpp/src/core/common/version.hpp` |
| parallel.hpp | donor → | `cpp/src/core/common/parallel.hpp` |
| linalg.hpp | donor → | `cpp/src/core/common/linalg.hpp` |

Plus donor's `common/` subtree (vendored C-level math/RNG kernels) —
already organised in donor under `cpp/src/core/common/` — copied to the
same path here:

| Subtree | Files | Notes |
|---------|-------|-------|
| `cpp/src/core/common/linalg.{c,h}` | 2 | C-level low-level kernels (companion to `linalg.hpp`) |
| `cpp/src/core/common/banded_solver.{c,h}` | 2 | banded LDLT solver |
| `cpp/src/core/common/bspline.{c,h}` | 2 | B-spline kernels |
| `cpp/src/core/common/fck_kernel.{c,h}` | 2 | fractional convolutional kernel |
| `cpp/src/core/common/padding.{c,h}` | 2 | array-padding helpers |
| `cpp/src/core/common/pca_helper.{c,h}` | 2 | PCA helper (randomized SVD path) |
| `cpp/src/core/common/rng_pcg64.{c,h}` | 2 | NumPy-bit-equivalent PCG64 |
| `cpp/src/core/common/svd.{c,h}` | 2 | SVD core |
| `cpp/src/core/common/wavelet_kernels.{c,h}` | 2 | wavelet filter banks |
| `cpp/src/core/common/ziggurat.c` + `ziggurat_constants.h` | 2 | ziggurat random sampling |
| `cpp/src/core/common/_vendored/fitpack/` | 13 | vendored FITPACK Fortran sources |

Public headers copied to `cpp/include/n4m/`:

| File | Notes |
|------|-------|
| `cpp/include/n4m/n4m.h` | 136.8 KB umbrella ABI header |
| `cpp/include/n4m/n4m_version.h` | semver constants |
| `cpp/include/n4m/n4m_export.h.in` | visibility / dllexport shim template |

The `PLACEHOLDER.md` files previously in `cpp/src/core/common/` and
`cpp/include/n4m/` are removed since real content has landed.

## What is NOT changed in this phase

Per the roadmap (M2.5 lands the unified files; M3 moves donor sources
out of `_donor/`; M4 splits `model.cpp` / `extra_pls.cpp`; M5 renames
`p4a_* → n4m_*` everywhere), this commit deliberately:

- **Does not** modify `cpp/include/pls4all/p4a.h` (the existing public
  header). pls4all sources still see `p4a_*` symbols at compile time.
- **Does not** modify `pls4all`'s `cpp/src/CMakeLists.txt`. The flat
  files at `cpp/src/core/{context,status,matrix_view,version}.cpp` are
  still listed there and still compiled into `libp4a.so.1.16.0`.
- **Does not** delete the duplicate files at
  `_donor/nirs4all-methods/cpp/src/core/{context,status,matrix_view,version,parallel,linalg}.{cpp,hpp}`
  or `_donor/nirs4all-methods/cpp/src/core/common/*`. Those will be
  removed in M3 when the donor source move lifts them under their final
  paths anyway.
- **Does not** alias `p4a_*` → `n4m_*` symbols. That alias generation
  is M5 territory (the ABI rename script) and would otherwise expand
  this phase well beyond the roadmap's 3-day budget.

## Build verification

`pls4all` build and tests remain **green** after the unification:

```
$ cmake --build --preset dev-release
ninja: no work to do.
$ ctest --preset dev-release --output-on-failure
1/1 Test #1: pls4all_tests ....................   Passed
$ ./build/dev-release/cpp/tests/pls4all_tests
=== 265 run, 0 failures, 0 skipped ===
```

The newly-landed common/ + include/n4m/ files are **purely additive**:
they are not consumed by any current build, they sit in the tree as
the canonical target for M3/M4/M5 to wire up.

## Gate evaluation

The roadmap M2.5 gate says:

> Both donor-fixture C++ tests and pls4all-fixture C++ tests pass
> against the unified common-core; no duplicate symbol in the linker

Achieved partially in this phase:

- ✅ **No duplicate symbol in the linker** — the unified common/ files
  are not built yet, so they contribute no symbol. pls4all sources
  continue to build against their flat copies with `p4a_*` symbols.
- ✅ **pls4all-fixture C++ tests pass** — verified above (265/265).
- ⚠️ **Donor-fixture C++ tests pass against unified common-core** —
  achievable only after M5 (ABI rename) wires both sides into one
  build. Carried forward to M5's gate. The donor's tests existed at
  `_donor/.../cpp/tests/` and pass against donor's own flat copies of
  the common files; pls4all's `pls4all_tests` binary doesn't link them
  in yet.

The unification work (file landing + reconciliation evidence) is done.
The build-system collapse onto the unified files lands in M5.

## Risk reassessment

| Risk listed in M2.5 row | Status |
|---|---|
| `n4m_context_create` collides with `p4a_context_create` on naive rename | Mitigated: the unified `n4m_*` symbols live at the new common/ location; pls4all flat sources keep their `p4a_*` symbols until M5 cleanly renames them in one pass. No collision possible because both name sets currently come from disjoint TUs. |
| Donor and base implementations disagree subtly | Resolved: side-by-side diff shows the only delta is prefix renaming. Enums, struct fields, backend codes, error-buffer size match exactly. |
| Donor's `linalg.hpp` (more-tested?) vs base's | Donor wins (per roadmap). Same content though — both ship identical kernel signatures. |
| Reconciling enum values | No-op — all 19 status codes and 6 backend codes are numerically identical. |

## Next: M3 (donor non-PLS source move)

With M2.5 landed, M3's task is straightforward:

1. `git mv` donor sources from `_donor/nirs4all-methods/cpp/src/core/{preprocessing,augmentation,splitters,filters,utilities}/...` to their final positions under `cpp/src/core/<category>/...`.
2. Delete the now-duplicated `_donor/nirs4all-methods/cpp/src/core/{context,status,matrix_view,version,parallel,linalg.hpp,common/}` files (preserved in this phase as part of the donor subtree; the unified copies at `cpp/src/core/common/` are authoritative going forward).
3. Update donor-side parity-generator imports to refer to the new paths.
4. Settle the `augmentations/` → `augmentation/` rename (donor uses plural, unified target is singular).

M3's gate: donor fixtures still pass against the moved sources.
