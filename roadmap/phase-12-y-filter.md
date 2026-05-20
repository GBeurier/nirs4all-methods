# Phase 12 — Y-based outlier filter (`n4m_filter_*` opener)

## Goal

Land the first member of the **filter** family — `YOutlierFilter` — and open
the new ABI prefix `n4m_filter_*` (reserved in `n4m.h` §5 but unused before
this phase). One operator, four detection methods (IQR, Z-score, percentile,
MAD), single-call `_fit_get_mask` lifecycle.

## Out of scope (Phase 13+)

- `XOutlierFilter` — X-based outlier detection (Phase 13).
- `SpectralQualityFilter` — Phase 14.
- Sample-quality / data-completeness filters — later.

## Operator

| Operator        | Methods                              | Lifecycle                |
| --------------- | ------------------------------------ | ------------------------ |
| `YOutlierFilter` | iqr / zscore / percentile / mad     | `create / fit_get_mask / destroy` |

The `_fit_get_mask` single-call form is correct here: bounds are a
deterministic function of the input `y` (no train/test split, no stochastic
state). A separate `_fit` step would just be a placeholder.

## Mask + stats contract

```c
typedef struct n4m_filter_stats_t {
    int64_t n_samples;
    int64_t n_kept;
    int64_t n_excluded;
    double  exclusion_rate;
} n4m_filter_stats_t;
```

The **mask** is a caller-allocated `uint8_t*` buffer of length `n`:
`mask[i] == 1` keeps sample `i`, `mask[i] == 0` excludes it. NaN samples
are always excluded.

The **stats** struct is caller-allocated, populated on success. Layout is
fixed at 32 bytes (3 × `int64_t` + 1 × `double`) and asserted at compile time.

## Files created

### Public headers
- `cpp/include/n4m/n4m_filters.h` — new ABI surface for the
  `n4m_filter_*` category. Standalone header (includes `n4m.h` for the
  status / export macros) so the existing `n4m.h` does not need to grow.

### Core engine
- `cpp/src/core/filters/y_outlier.{c,h}` — 1-D outlier detection. Pure
  arithmetic, `qsort` for percentile / MAD methods.

### C ABI wrapper
- `cpp/src/c_api/c_api_filters.cpp` — `extern "C"` wrappers that match the
  established pattern (opaque handle, try/catch on every entry, NULL-safe
  destroy, byte-compatible stats reinterpret).

### Frozen NumPy reference
- `parity/python_generator/src/n4m_parity_filters_ref/__init__.py`
- `parity/python_generator/src/n4m_parity_filters_ref/y_outlier.py`

New parity package (`n4m_parity_filters_ref/`) — distinct from
`n4m_parity_pybaselines_ref` because the upstream parity target is
`nirs4all`, not `pybaselines`.

### Fixture generator
- `parity/python_generator/scripts/generate_phase12_fixtures.py` — synthesises
  four distinct `y` distributions (Gaussian-with-outliers, log-normal,
  uniform, bimodal) and a four-case parameter sweep per method.

### Fixtures
- `parity/fixtures/filter_y_outlier_iqr_v1.json`        (~7.6 KB, 4 cases)
- `parity/fixtures/filter_y_outlier_zscore_v1.json`     (~7.6 KB, 4 cases)
- `parity/fixtures/filter_y_outlier_percentile_v1.json` (~7.7 KB, 4 cases)
- `parity/fixtures/filter_y_outlier_mad_v1.json`        (~7.6 KB, 4 cases)

Schema differs from preprocessing fixtures: 1-D `y_hex` input, per-case
`expected_mask` as an integer array and `expected_stats` as an object.

### Tests
- `cpp/tests/test_filters_y.cpp` — 12 tests (4 smoke per method + 4 parity +
  4 cross-cutting: param validation × 2, NaN exclusion, constant input).
  Standalone executable (`n4m_filters_tests`) to avoid touching
  the existing Phase 0+ test harness in `cpp/tests/main.cpp`.

### Docs
- `docs/algorithms/y_outlier_filter.md`

## Files modified

- `cpp/src/CMakeLists.txt` — append `core/filters/y_outlier.c` to
  `n4m_core` and `c_api/c_api_filters.cpp` to the C ABI source
  list. No changes to top-level CMake.
- `cpp/tests/CMakeLists.txt` — add new test executable
  `n4m_filters_tests` with its own CTest entry. The existing
  `n4m_tests` executable stays exactly as-is (no edits to
  `cpp/tests/main.cpp`).

Per the phase brief, the following files are intentionally **not** modified:

- `cpp/include/n4m/n4m.h` (the new ABI lives in the dedicated
  `n4m_filters.h` instead — first non-preprocessing surface header).
- `cpp/include/n4m/n4m_version.h`
  (ABI version stays at 1.6.0 for this worktree; downstream phases bundle
  the version bump with the symbol-list regeneration).
- `cpp/abi/expected_symbols_{linux,macos,windows}.txt` (worktree-local, no
  regeneration step in this brief — the next CI run regenerates from the
  built `.so`).
- Top-level `CMakeLists.txt`.
- `cpp/tests/main.cpp`.

## ABI surface added

```c
/* n4m_filters.h §2 - Phase 12 YOutlierFilter (1 op, 3 symbols) */
typedef enum n4m_y_outlier_method_t {
    N4M_Y_OUTLIER_IQR        = 0,
    N4M_Y_OUTLIER_ZSCORE     = 1,
    N4M_Y_OUTLIER_PERCENTILE = 2,
    N4M_Y_OUTLIER_MAD        = 3
} n4m_y_outlier_method_t;

typedef struct n4m_filter_stats_t {
    int64_t n_samples;
    int64_t n_kept;
    int64_t n_excluded;
    double  exclusion_rate;
} n4m_filter_stats_t;

typedef struct n4m_filter_y_outlier_handle_t n4m_filter_y_outlier_handle_t;

n4m_status_t n4m_filter_y_outlier_create(n4m_filter_y_outlier_handle_t** out,
                                          n4m_y_outlier_method_t method,
                                          double threshold,
                                          double lower_pct, double upper_pct);
void         n4m_filter_y_outlier_destroy(n4m_filter_y_outlier_handle_t* h);
n4m_status_t n4m_filter_y_outlier_fit_get_mask(
    const n4m_filter_y_outlier_handle_t* h,
    const double* y, int64_t n,
    uint8_t* mask_out,
    n4m_filter_stats_t* stats_out);
```

3 new exported symbols. Total: 126 → **129**. ABI MINOR remains at 1.6 until
the next coordinating phase regenerates `expected_symbols_*` (since this
worktree is forbidden from touching them).

## Parity tolerances

| Field           | Tolerance       | Notes                                       |
| --------------- | --------------- | ------------------------------------------- |
| `mask[i]`       | exact equality  | uint8 byte-for-byte                         |
| `n_samples`     | exact equality  | always == `n`                               |
| `n_kept`        | exact equality  | derived from mask                           |
| `n_excluded`    | exact equality  | == `n_samples - n_kept`                     |
| `exclusion_rate`| 1e-12 abs       | single floating divide, rounding-bounded    |

## Acceptance criteria

- Build clean.
- 4 new fixtures, ~30 KB total.
- 12 new tests, all passing.
- 3 new ABI symbols exported via `n4m_*` version script.
- No edits to `n4m.h` / `n4m_version.h` / `expected_symbols_*` / top
  `CMakeLists.txt` / `tests/main.cpp`.
- Existing 82 tests still pass (zero regressions).

## Verification

```bash
cd /home/delete/nirs4all/nirs4all-methods/.claude/worktrees/agent-a40029a940a5f7368
cmake --preset dev-debug
cmake --build --preset dev-debug
ctest --test-dir build/dev-debug
# 2/2 tests passed (82 in n4m_tests + 12 in n4m_filters_tests)

nm -D --defined-only build/dev-debug/cpp/src/libn4m.so.1.6.0 \
    | awk '$2=="T" {print $3}' \
    | grep '^n4m_filter_y_outlier_'
# n4m_filter_y_outlier_create
# n4m_filter_y_outlier_destroy
# n4m_filter_y_outlier_fit_get_mask
```

## Next phase

Phase 13 — `XOutlierFilter`: X-based (spectral) outlier detection with
distance-based and statistical methods. Reuses `n4m_filter_stats_t` and
the new `n4m_filters.h` header.
