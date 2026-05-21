# Phase 14 — Meta sample filters: HighLeverage, SpectralQuality, Composite

## Goal

Implement three sample-filtering operators that complement the Phase 12
outlier filters (developed in parallel):

1. **HighLeverageFilter** — detects samples with disproportionate influence
   on a linear-model fit via the hat-matrix diagonal or PCA-based score-
   space leverage.
2. **SpectralQualityFilter** — rejects samples whose spectra fail multi-
   criteria quality checks (NaN ratio, Inf presence, zero ratio, variance
   floor, value bounds).
3. **CompositeFilter** — aggregates the keep-masks of any number of sub-
   filters via ANY or ALL semantics.

## Out of scope (handled by Phase 12 or later)

- YOutlierFilter / XOutlierFilter / MetadataFilter (Phase 12).
- IsolationForest / Mahalanobis / LOF outlier methods.
- The public ABI surface in `cpp/include/n4m/n4m.h` §13+
  (Phase 12 owns the section banner and the shared `n4m_filter_stats_t`
  declaration). Phase 14 emits the same symbols through a side-channel
  header (`cpp/src/c_api/c_api_filters_meta.h`) until Phase 12 lands.

## Operators

| Operator | Algorithm | Stateful? |
|----------|-----------|:---:|
| **HighLeverageFilter** (`n4m_filter_leverage_*`) | Hat-matrix diag `h_ii = X_i (X^T X)^{-1} X_i^T` via Householder QR (shared with EMSC / Detrend) on `X^T X + reg * I`; OR PCA-based leverage using Jacobi eigendecomposition of `X^T X` (top-k components). Threshold = `multiplier * mean(training_leverages)` or `absolute_threshold` in `(0, 1)`. | yes (fit / apply) |
| **SpectralQualityFilter** (`n4m_filter_quality_*`) | Six all-or-nothing per-row threshold checks: `max_nan_ratio`, `check_inf`, `max_zero_ratio`, `min_variance` (np.nanvar), `max_value`, `min_value`. | no (apply only) |
| **CompositeFilter** (`n4m_filter_composite_*`) | Boolean AND (`mode=ANY`) or OR (`mode=ALL`) of sub-filter keep-masks. Holds references to sub-filter handles; caller owns lifetimes. | yes (delegates to each sub-filter's `_apply`) |

## Phase 12 / Phase 14 coordination

Phase 12 introduces the `n4m_filter_*` ABI category, the shared
`n4m_filter_stats_t` result struct, and the §13+ banner in `n4m.h`.
Phase 14 lands in parallel and reuses the same category. To avoid edit
conflicts on `n4m.h`, `n4m_version.h`, the expected-symbols files, and
`tests/main.cpp`:

- Phase 14 declarations live in `cpp/src/c_api/c_api_filters_meta.h`
  (guarded behind `N4M_FILTER_STATS_T_DEFINED` so Phase 12 can drop its
  own definition into `n4m.h` without conflicting at merge time).
- The new symbols are still exported by the shared library through the
  same `n4m_*` wildcard version script (`cpp/src/c_api/n4m_linux.map`).
- Phase 14 tests live in their own executable
  (`n4m_tests_filters_meta`) attached to the existing test
  driver via `cpp/tests/CMakeLists.txt`. Once Phase 12 lands, the
  side-channel header is removed and the two executables are merged into
  the canonical `n4m_tests` runner.

## Files

### Core engines (6 files)

- `cpp/src/core/filters/high_leverage.{c,h}`
- `cpp/src/core/filters/spectral_quality.{c,h}`
- `cpp/src/core/filters/composite.{c,h}`

### C API wrappers (2 files)

- `cpp/src/c_api/c_api_filters_meta.cpp` — extern "C" wrappers.
- `cpp/src/c_api/c_api_filters_meta.h`   — Phase 12 coordination header.

### Tests (1 file)

- `cpp/tests/test_filters_meta.cpp`       — 6 tests (3 smoke + 3 parity).

### Parity (4 files)

- `parity/python_generator/src/n4m_parity_filters_ref/__init__.py`
- `parity/python_generator/src/n4m_parity_filters_ref/high_leverage.py`
- `parity/python_generator/src/n4m_parity_filters_ref/spectral_quality.py`
- `parity/python_generator/src/n4m_parity_filters_ref/composite.py`

### Fixtures (4 files)

- `parity/fixtures/filter_leverage_v1.json`       — tall-matrix cases
  (rows > cols, hat + explicit PCA), 3 cases.
- `parity/fixtures/filter_leverage_wide_v1.json`  — wide-matrix case
  (rows < cols → auto-fallback to PCA), 1 case.
- `parity/fixtures/filter_quality_v1.json`        — 4 cases.
- `parity/fixtures/filter_composite_v1.json`      — 2 cases (ANY / ALL).

### Generator + docs

- `parity/python_generator/scripts/generate_phase14_fixtures.py`
- `docs/algorithms/high_leverage.md`
- `docs/algorithms/spectral_quality.md`
- `docs/algorithms/composite_filter.md`
- `roadmap/phase-14-filters-meta.md` (this file)

## Files modified

- `cpp/src/CMakeLists.txt`     — add `core/filters/*.c` and `c_api_filters_meta.cpp`.
- `cpp/tests/CMakeLists.txt`   — add `n4m_tests_filters_meta` executable.

No edits to `n4m.h`, `n4m_version.h`, `cpp/abi/expected_symbols_*.txt`,
top-level `CMakeLists.txt`, or `cpp/tests/main.cpp` — these belong to
Phase 12 / the post-merge integration commit.

## ABI surface added

14 new symbols (no new public section in `n4m.h` until Phase 12 lands):

```
n4m_filter_leverage_create
n4m_filter_leverage_destroy
n4m_filter_leverage_fit
n4m_filter_leverage_is_fitted
n4m_filter_leverage_apply
n4m_filter_leverage_threshold

n4m_filter_quality_create
n4m_filter_quality_destroy
n4m_filter_quality_apply

n4m_filter_composite_create
n4m_filter_composite_destroy
n4m_filter_composite_add_leverage
n4m_filter_composite_add_quality
n4m_filter_composite_apply
```

Phase 12 will extend the composite's `add_*` family with
`add_y_outlier`, `add_x_outlier`, `add_metadata` to cover its own filter
kinds.

## Parity tolerances

| Operator | Reference | Abs tol | Rel tol | Notes |
|----------|-----------|---------|---------|-------|
| HighLeverage     | Frozen NumPy + `np.linalg.inv` | mask equality | mask equality | Threshold lies far from any data point in the fixture; bit-exact mask. Underlying leverages match within 1e-9 abs / 1e-10 rel (matrix-inverse conditioning bound). |
| SpectralQuality | Frozen NumPy thresholds | exact | exact | Pure threshold comparisons. Mask equality. |
| Composite       | Frozen NumPy boolean aggregator | exact | exact | Boolean AND / OR of sub-filter masks. |

## Acceptance criteria

- Build clean. Zero warnings.
- All 3 operators implement their respective lifecycles cleanly.
- Tests: 82 → 82 + 6 = **88/88**.
- 14 new ABI symbols (the canonical expected-symbols update lands with
  Phase 12 to avoid touching the abi/ directory in two parallel branches).
- Frozen NumPy reference vendored under
  `parity/python_generator/src/n4m_parity_filters_ref/` and validated
  against the nirs4all 0.8.x reference behaviour.
- CI green when run against the Phase 14 worktree.

## Verification

```bash
cd /home/delete/nirs4all/nirs4all-methods/.claude/worktrees/agent-a62f804de4a887c03
cmake --preset dev-debug
cmake --build --preset dev-debug
./build/dev-debug/cpp/tests/n4m_tests                # 82/82
./build/dev-debug/cpp/tests/n4m_tests_filters_meta   # 6/6

nm -D --defined-only build/dev-debug/cpp/src/libn4m.so.1.6.0 | \
    awk '$2=="T" {print $3}' | grep "n4m_filter" | wc -l         # 14
```

## Next phase

[Phase 15 — Augmentation (noise & spectral)](phase-15-augmentation.md):
Begin the augmentation surface (`n4m_aug_*`) — `GaussianAdditiveNoise`,
`MultiplicativeNoise`, `SpikeNoise`, `BandPerturbation`, etc.
