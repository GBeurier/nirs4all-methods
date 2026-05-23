# M4 — pls4all source split plan

Source: Explore-agent analysis of `cpp/src/core/model.cpp` (4541 lines, 180 KB) +
`cpp/src/core/extra_pls.cpp` (1647 lines, 80 KB).

## Totals

| File | Lines | Functions | Helpers (anon namespace) |
|------|-------|-----------|--------------------------|
| model.cpp     | 4541 | 18 | ~1500 lines (L18–L1524) |
| extra_pls.cpp | 1647 | 19 | ~290 lines (L18–L306) |
| **Total** | **6188** | **37** | **~1790** |

## Per-file function table

### model.cpp

| # | Function | Line | Length | Target |
|--:|---|---:|---:|---|
| 1 | `fit_pls_regression_nipals` | 1528 | 291 | `models/pls/nipals.cpp` |
| 2 | `fit_pls_canonical_nipals` | 1819 | 10 | `models/pls/canonical.cpp` |
| 3 | `fit_pls_canonical_svd` | 1829 | 10 | `models/pls/canonical.cpp` |
| 4 | `fit_pls_svd` | 1839 | 161 | `models/pls/plssvd.cpp` |
| 5 | `fit_opls_nipals` | 2000 | 350 | `models/opls/opls.cpp` |
| 6 | `fit_pls_regression_orthogonal_scores` | 2350 | 265 | `models/pls/oscores.cpp` |
| 7 | `fit_pls_regression_kernel` | 2615 | 280 | `models/pls/kernel.cpp` |
| 8 | `fit_pls_regression_svd` | 2895 | 166 | `models/pls/svd.cpp` |
| 9 | `fit_pls_regression_power` | 3061 | 166 | `models/pls/power.cpp` |
| 10 | `fit_pls_regression_randomized_svd` | 3227 | 170 | `models/pls/randomized_svd.cpp` |
| 11 | `fit_pcr_svd` | 3397 | 288 | `models/pcr/pcr.cpp` |
| 12 | `fit_pls_regression_simpls` | 3685 | 200 | `models/pls/simpls.cpp` |
| 13 | `predict_into` | 3885 | 36 | `models/core/dispatch.cpp` |
| 14 | `fit_di_pls` | 3921 | 242 | `models/classification/pls_da.cpp` |
| 15 | `fit_pls_sparse_simpls` | 4163 | 225 | `models/sparse/sparse_simpls.cpp` |
| 16 | `fit_model` | 4388 | 46 | `models/core/dispatch.cpp` |
| 17 | `transform_into` | 4434 | 36 | `models/core/dispatch.cpp` |
| 18 | `model_array` | 4470 | 72 | `models/core/dispatch.cpp` |

### extra_pls.cpp

| # | Function | Line | Length | Target |
|--:|---|---:|---:|---|
| 1 | `fit_sparse_pls_da` | 307 | 60 | `models/sparse/sparse_pls_da.cpp` |
| 2 | `fit_group_sparse_pls` | 367 | 78 | `models/sparse/group_sparse.cpp` |
| 3 | `fit_fused_sparse_pls` | 445 | 36 | `models/sparse/fused_sparse.cpp` |
| 4 | `fit_cppls` | 481 | 70 | `models/specialized/cppls.cpp` |
| 5 | `fit_weighted_pls` | 551 | 66 | `models/specialized/weighted.cpp` |
| 6 | `fit_robust_pls` | 617 | 61 | `models/specialized/robust_huber.cpp` |
| 7 | `fit_ridge_pls` | 678 | 59 | `models/specialized/ridge.cpp` |
| 8 | `fit_continuum_regression` | 737 | 63 | `models/specialized/continuum.cpp` |
| 9 | `fit_pls_glm` | 800 | 47 | `models/glm/pls_glm.cpp` |
| 10 | `fit_pls_qda` | 847 | 98 | `models/classification/pls_qda.cpp` |
| 11 | `fit_pls_cox` | 945 | 85 | `models/classification/pls_cox.cpp` |
| 12 | `fit_pds` | 1030 | 111 | `transfer/pds.cpp` |
| 13 | `fit_ds` | 1141 | 102 | `transfer/ds.cpp` |
| 14 | `fit_mir_pls` | 1243 | 90 | `models/local/mir_pls.cpp` |
| 15 | `fit_missing_aware_nipals` | 1333 | 65 | `models/specialized/missing_aware_nipals.cpp` |
| 16 | `approximate_press` | 1398 | 79 | `models/core/dispatch.cpp` |
| 17 | `fit_bagging_pls` | 1477 | 54 | `models/ensembles/bagging.cpp` |
| 18 | `fit_boosting_pls` | 1531 | 52 | `models/ensembles/boosting.cpp` |
| 19 | `fit_random_subspace_pls` | 1583 | 65 | `models/ensembles/random_subspace.cpp` |

## High-risk cross-cutting helpers

The ~1790 lines of helpers (anonymous namespaces) MUST be extracted FIRST,
into shared `_common.cpp` files, before any algorithm extraction:

| Helper | Location | Used by | Extract to |
|--------|----------|---------|------------|
| `simple_simpls` | extra_pls.cpp:97 | model.cpp `fit_pls_sparse_simpls` AND extra_pls sparse variants | `models/pls/_common.cpp` |
| `center_scale_in_place` | model.cpp:161 | ~15 model.cpp functions | `models/core/_common.cpp` |
| `dominant_svd_pair`, `largest_symmetric_eigenvector` | model.cpp helpers | SVD / Kernel / Canonical / OPLS | `models/pls/_common.cpp` |
| `canonicalize_direction_sign`, `canonicalize_svd_pair_sign` | model.cpp:274,691,772,919,953,1064,1086,1175,3801 | SVD variants + OPLS chassis | `models/pls/_common.cpp` |
| `soft_threshold` | extra_pls.cpp:87 | all 4 sparse variants | `models/sparse/_common.cpp` |
| `copy_matrix`, `column_means`, `subtract_means`, `squared_norm` | extra_pls.cpp helpers | sparse + specialized | `models/sparse/_common.cpp` |
| `invert_square`, `dot`, `matrix_vector_product` | model.cpp helpers | most algorithms | `models/core/_common.cpp` |
| `validate_fit_request`, `copy_matrix_checked` | model.cpp helpers | all fit_* functions | `models/core/_common.cpp` |

## Proposed extraction order (7 phases)

| Phase | Scope | Commits |
|--:|---|---:|
| 1 | Helpers → `_common.cpp` (3 files) | 3 |
| 2 | Core PLS solvers (nipals, simpls, svd, plssvd, power, randomized_svd, kernel, canonical) | 8 |
| 3 | PCR + OPLS | 2 |
| 4 | Classification (pls_da, pls_qda, pls_cox) | 3 |
| 5 | Sparse (4 files) | 4 |
| 6 | Specialized (6) + ensembles (3) + transfer (2) + local + glm | 12 |
| 7 | Core dispatch + header consolidation | 2 |

Total: **~34 commits** if each extraction is its own commit (best for blame
preservation). Per-commit gate: `n4m_tests` 265/265 still green and no
function moves are functional changes.

## Notes from the agent's analysis

- Lines 18–1524 of model.cpp form a tight ~1500-line anonymous-namespace
  helper block with a clear boundary; lines 18–306 of extra_pls.cpp form a
  ~290-line one. Both extract cleanly.
- No explicit `// === ALGORITHM ===` section dividers exist; functions appear
  in topological order by dependency.
- The 18 model.cpp functions reach line 4541; the 19 extra_pls.cpp functions
  end at line 1647. Both files have no unaccounted top-level definitions.
- Helpers do **not** cross between the two source files (each set is
  file-local). The `simple_simpls` cross-call from model.cpp → extra_pls.cpp
  via `fit_pls_sparse_simpls` is the ONE exception — extracting it first to
  `models/pls/_common.cpp` resolves it cleanly.
