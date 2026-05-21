# Phase 13 — X-feature outlier filter (XOutlierFilter)

## Goal

Implement a stateful X-based outlier filter (`n4m_filter_x_outlier_*`)
that detects samples whose spectral features deviate from the bulk via
one of six selectable methods. Mirrors
`nirs4all.operators.filters.XOutlierFilter` and completes the Phase 12 /
Phase 13 / Phase 14 sample-filter trio (Y-outlier / X-outlier /
HighLeverage+SpectralQuality+Composite).

## Operators

One operator, six methods (`method` is `int32_t` internally; the public
enum lands in `n4m.h §18` at the integration commit):

| Method id | Algorithm | Threshold |
|:---:|---|---|
| 0 `mahalanobis` | Empirical covariance, root-Mahalanobis distance | `sqrt(chi2_inv(0.975, p))` |
| 1 `robust_mahalanobis` | Simplified FAST-MCD covariance, root-Mahalanobis | Same chi-squared threshold |
| 2 `pca_residual` | Q-statistic (PCA reconstruction error) | 95th percentile of training Q |
| 3 `pca_leverage` | Hotelling T² in PCA score space | 95th percentile of training T² |
| 4 `isolation_forest` | Vendored PCG64-seeded IF ensemble | Contamination quantile |
| 5 `lof` | Vendored k-NN-based LOF (k = min(20, n-1)) | Contamination quantile |

## Files

### Core engine (1 + 1 helper + 3 vendored)

- `cpp/src/core/filters/x_outlier.{c,h}` — 6-method orchestrator.
- `cpp/src/core/common/pca_helper.{c,h}` — shared SVD-based PCA wrapper.
- `cpp/src/core/filters/_vendored/isolation_forest.{c,h}` — ~400 LOC
  vendored Isolation Forest with PCG64-seeded trees.
- `cpp/src/core/filters/_vendored/lof.{c,h}` — ~250 LOC LOF with
  brute-force k-NN.
- `cpp/src/core/filters/_vendored/min_cov_det.{c,h}` — ~300 LOC
  simplified FAST-MCD (single seeded start + 30 concentration steps).

### C API wrapper (1 file)

- `cpp/src/c_api/c_api_filter_x_outlier.cpp` — 5 extern "C" entry
  points. Public ABI declarations live inline as forward-declared
  prototypes until n4m.h §18 lands at the integration commit.

### Tests (1 file)

- `cpp/tests/test_filter_x_outlier.cpp` — 12 tests (6 smoke + 6 parity).
  Lives in its own executable
  `n4m_tests_filter_x_outlier` so this worktree leaves
  `cpp/tests/main.cpp` and the expected-symbols files frozen.

### Parity (1 file)

- `parity/python_generator/src/n4m_parity_filters_ref/x_outlier.py` —
  frozen sklearn reference. Updated `__init__.py` to re-export
  `x_outlier_fit_get_mask`.

### Fixtures (6 files)

- `parity/fixtures/filter_x_outlier_{method}_v1.json` for each of the
  six methods (~23 KB each).

### Generator (1 file)

- `parity/python_generator/scripts/generate_phase13_fixtures.py`.

### Docs (1 file)

- `docs/algorithms/x_outlier_filter.md`.

### Build (2 files)

- `cpp/src/CMakeLists.txt` — append the new engine + vendored sources
  to `n4m_core` and the C ABI wrapper to
  `n4m_c{,_static}`.
- `cpp/tests/CMakeLists.txt` — register the new test executable +
  CTest target.

### Intentionally NOT modified

Per the phase brief, this worktree does not edit:

- `cpp/include/n4m/n4m.h` (integration adds §18 enum).
- `cpp/include/n4m/n4m_version.h` (ABI still 1.7.0).
- `cpp/abi/expected_symbols_{linux,macos,windows}.txt` (regeneration
  belongs to the merge commit).
- Top-level `CMakeLists.txt`.
- `cpp/tests/main.cpp`.

## ABI surface added

```c
typedef struct n4m_filter_x_outlier_handle_t n4m_filter_x_outlier_handle_t;

n4m_status_t n4m_filter_x_outlier_create(
    n4m_filter_x_outlier_handle_t** out,
    int32_t  method,                // 0..5
    int      use_threshold,
    double   threshold,
    int32_t  n_components,
    double   contamination,
    uint64_t seed,
    int32_t  n_estimators,
    int64_t  max_samples);
void         n4m_filter_x_outlier_destroy(n4m_filter_x_outlier_handle_t*);
n4m_status_t n4m_filter_x_outlier_fit(n4m_filter_x_outlier_handle_t* h,
                                       n4m_matrix_view_t X);
n4m_status_t n4m_filter_x_outlier_is_fitted(
    const n4m_filter_x_outlier_handle_t* h, int* out_fitted);
n4m_status_t n4m_filter_x_outlier_apply(
    const n4m_filter_x_outlier_handle_t* h,
    n4m_matrix_view_t X,
    uint8_t* mask_out,
    n4m_filter_stats_t* stats_out);
```

5 new exported symbols. Total: 251 → **256**. ABI MINOR remains at 1.7
until the next coordinating phase regenerates `expected_symbols_*` (this
worktree is forbidden from touching them).

## Parity tolerances

| Method | Tolerance | Notes |
|--------|-----------|-------|
| `mahalanobis` | exact mask | 1e-10 abs / 1e-11 rel on the Mahalanobis distance |
| `robust_mahalanobis` | ≤ 8 mask flips | Simplified FAST-MCD (see DEFERRALS) |
| `pca_residual` | exact mask | 1e-10 abs / 1e-11 rel on Q |
| `pca_leverage` | exact mask | 1e-10 abs / 1e-11 rel on T² |
| `isolation_forest` | ≤ 8 mask flips | PCG64 vs RandomState bit-stream mismatch |
| `lof` | ≤ 3 mask flips | Percentile interpolation edge |

## DEFERRALS

- **FAST-MCD ensemble**. sklearn runs `n_trials = 500` random starts; we
  run a single PCG64-seeded start. Asymptotic robustness is unchanged
  but mask flips by O(8) samples. Tracked for a follow-up phase.
- **Bit-exact sklearn `IsolationForest`**. sklearn uses NumPy
  RandomState; n4m uses PCG64. Tree structures differ. Counts agree
  within tolerance.
- **kd-tree LOF**. Brute-force k-NN is fine for NIRS-typical `n < 1000`.

## Acceptance criteria

- Build clean (no new warnings).
- 6 fixtures generated.
- 12 new tests, all passing (6 smoke + 6 parity).
- 5 new ABI symbols exported via `n4m_*` version script.
- No edits to `n4m.h` / `n4m_version.h` / `expected_symbols_*` / top
  `CMakeLists.txt` / `tests/main.cpp`.
- Existing 171 tests still pass.

## Verification

```bash
cd /home/delete/nirs4all/nirs4all-methods/.claude/worktrees/agent-a402c58f3cc0f19b4
cmake --preset dev-debug
cmake --build --preset dev-debug

build/dev-debug/cpp/tests/n4m_tests                       # 171 / 171
build/dev-debug/cpp/tests/n4m_tests_filter_x_outlier      # 12 / 12

ctest --test-dir build/dev-debug                                       # 2 / 2 pass

nm -D --defined-only build/dev-debug/cpp/src/libn4m.so.1.7.0 \
    | awk '$2=="T" {print $3}' | grep n4m_filter_x_outlier_ | sort
# n4m_filter_x_outlier_apply
# n4m_filter_x_outlier_create
# n4m_filter_x_outlier_destroy
# n4m_filter_x_outlier_fit
# n4m_filter_x_outlier_is_fitted
```
