# Phase 12 — Y-based outlier filter (`c4a_filter_*` opener)

## Goal

Land the first member of the **filter** family — `YOutlierFilter` — and open
the new ABI prefix `c4a_filter_*` (reserved in `c4a.h` §5 but unused before
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
typedef struct c4a_filter_stats_t {
    int64_t n_samples;
    int64_t n_kept;
    int64_t n_excluded;
    double  exclusion_rate;
} c4a_filter_stats_t;
```

The **mask** is a caller-allocated `uint8_t*` buffer of length `n`:
`mask[i] == 1` keeps sample `i`, `mask[i] == 0` excludes it. NaN samples
are always excluded.

The **stats** struct is caller-allocated, populated on success. Layout is
fixed at 32 bytes (3 × `int64_t` + 1 × `double`) and asserted at compile time.

## Files created

### Public headers
- `cpp/include/chemometrics4all/c4a_filters.h` — new ABI surface for the
  `c4a_filter_*` category. Standalone header (includes `c4a.h` for the
  status / export macros) so the existing `c4a.h` does not need to grow.

### Core engine
- `cpp/src/core/filters/y_outlier.{c,h}` — 1-D outlier detection. Pure
  arithmetic, `qsort` for percentile / MAD methods.

### C ABI wrapper
- `cpp/src/c_api/c_api_filters.cpp` — `extern "C"` wrappers that match the
  established pattern (opaque handle, try/catch on every entry, NULL-safe
  destroy, byte-compatible stats reinterpret).

### Frozen NumPy reference
- `parity/python_generator/src/c4a_parity_filters_ref/__init__.py`
- `parity/python_generator/src/c4a_parity_filters_ref/y_outlier.py`

New parity package (`c4a_parity_filters_ref/`) — distinct from
`c4a_parity_pybaselines_ref` because the upstream parity target is
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
  Standalone executable (`chemometrics4all_filters_tests`) to avoid touching
  the existing Phase 0+ test harness in `cpp/tests/main.cpp`.

### Docs
- `docs/algorithms/y_outlier_filter.md`

## Files modified

- `cpp/src/CMakeLists.txt` — append `core/filters/y_outlier.c` to
  `chemometrics4all_core` and `c_api/c_api_filters.cpp` to the C ABI source
  list. No changes to top-level CMake.
- `cpp/tests/CMakeLists.txt` — add new test executable
  `chemometrics4all_filters_tests` with its own CTest entry. The existing
  `chemometrics4all_tests` executable stays exactly as-is (no edits to
  `cpp/tests/main.cpp`).

Per the phase brief, the following files are intentionally **not** modified:

- `cpp/include/chemometrics4all/c4a.h` (the new ABI lives in the dedicated
  `c4a_filters.h` instead — first non-preprocessing surface header).
- `cpp/include/chemometrics4all/c4a_version.h`
  (ABI version stays at 1.6.0 for this worktree; downstream phases bundle
  the version bump with the symbol-list regeneration).
- `cpp/abi/expected_symbols_{linux,macos,windows}.txt` (worktree-local, no
  regeneration step in this brief — the next CI run regenerates from the
  built `.so`).
- Top-level `CMakeLists.txt`.
- `cpp/tests/main.cpp`.

## ABI surface added

```c
/* c4a_filters.h §2 - Phase 12 YOutlierFilter (1 op, 3 symbols) */
typedef enum c4a_y_outlier_method_t {
    C4A_Y_OUTLIER_IQR        = 0,
    C4A_Y_OUTLIER_ZSCORE     = 1,
    C4A_Y_OUTLIER_PERCENTILE = 2,
    C4A_Y_OUTLIER_MAD        = 3
} c4a_y_outlier_method_t;

typedef struct c4a_filter_stats_t {
    int64_t n_samples;
    int64_t n_kept;
    int64_t n_excluded;
    double  exclusion_rate;
} c4a_filter_stats_t;

typedef struct c4a_filter_y_outlier_handle_t c4a_filter_y_outlier_handle_t;

c4a_status_t c4a_filter_y_outlier_create(c4a_filter_y_outlier_handle_t** out,
                                          c4a_y_outlier_method_t method,
                                          double threshold,
                                          double lower_pct, double upper_pct);
void         c4a_filter_y_outlier_destroy(c4a_filter_y_outlier_handle_t* h);
c4a_status_t c4a_filter_y_outlier_fit_get_mask(
    const c4a_filter_y_outlier_handle_t* h,
    const double* y, int64_t n,
    uint8_t* mask_out,
    c4a_filter_stats_t* stats_out);
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
- 3 new ABI symbols exported via `c4a_*` version script.
- No edits to `c4a.h` / `c4a_version.h` / `expected_symbols_*` / top
  `CMakeLists.txt` / `tests/main.cpp`.
- Existing 82 tests still pass (zero regressions).

## Verification

```bash
cd /home/delete/nirs4all/chemometrics4all/.claude/worktrees/agent-a40029a940a5f7368
cmake --preset dev-debug
cmake --build --preset dev-debug
ctest --test-dir build/dev-debug
# 2/2 tests passed (82 in chemometrics4all_tests + 12 in chemometrics4all_filters_tests)

nm -D --defined-only build/dev-debug/cpp/src/libc4a.so.1.6.0 \
    | awk '$2=="T" {print $3}' \
    | grep '^c4a_filter_y_outlier_'
# c4a_filter_y_outlier_create
# c4a_filter_y_outlier_destroy
# c4a_filter_y_outlier_fit_get_mask
```

## Next phase

Phase 13 — `XOutlierFilter`: X-based (spectral) outlier detection with
distance-based and statistical methods. Reuses `c4a_filter_stats_t` and
the new `c4a_filters.h` header.
