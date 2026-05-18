# Phase 20 — TransferMetricsComputer (9-metric dataset alignment vector)

## Goal

Introduce a single utility that quantifies the alignment between a source and a target dataset in a shared PCA-truncated space. Output is a 9-field POD struct mirroring `nirs4all.analysis.transfer_metrics.TransferMetricsComputer`.

This is the **first member of the `c4a_transfer_*` ABI category** (reserved in `c4a.h §5` under the `c4a_metric_*` reservation umbrella alongside the Phase 19 diagnostics).

## Operator (1 utility — pure function)

| Function | Inputs | Outputs |
|---|---|---|
| `c4a_transfer_metrics_compute` | `X_source`, `X_target` (row-major contiguous F64 views), `n_components`, `k_neighbors`, `seed` | `c4a_transfer_metrics_t` (9 doubles) |

The nine metrics:

| Field | Formula |
|---|---|
| `centroid_distance`    | \|\|mean(Z_s) − mean(Z_t)\|\|_2 in PCA space |
| `cka_similarity`       | Linear CKA on covariance matrices (`\|\|C_s C_t\|\|_F² / (\|\|C_s²\|\|_F·\|\|C_t²\|\|_F)`) |
| `grassmann_distance`   | √(Σ θ_i²) of principal angles between PCA subspaces |
| `rv_coefficient`       | `trace(C_s C_t) / √(trace(C_s²)·trace(C_t²))` |
| `procrustes_disparity` | Standardised Procrustes disparity on first 2 PCA components (matches `scipy.spatial.procrustes`) |
| `trustworthiness`      | sklearn formula, neighbours = `min(k_neighbors, n − 2)` |
| `spread_distance`      | `\|\|cov(Z_s) − cov(Z_t)\|\|_F + ½·(mean of min pairwise dists)` |
| `evr_source`           | Σλ_kept / Σλ_all on the source PCA |
| `evr_target`           | Σλ_kept / Σλ_all on the target PCA |

`grassmann_distance` is NaN when the two datasets do not share the same feature count.

## C ABI surface added

**One symbol** in a brand-new `c4a_transfer_*` category:

```c
typedef struct c4a_transfer_metrics_t {
    double centroid_distance;
    double cka_similarity;
    double grassmann_distance;
    double rv_coefficient;
    double procrustes_disparity;
    double trustworthiness;
    double spread_distance;
    double evr_source;
    double evr_target;
} c4a_transfer_metrics_t;

C4A_API c4a_status_t c4a_transfer_metrics_compute(
    c4a_matrix_view_t X_source,
    c4a_matrix_view_t X_target,
    int32_t n_components,
    int32_t k_neighbors,
    uint64_t seed,
    c4a_transfer_metrics_t* out);
```

Symbol count: 126 → **127**. ABI 1.6.0 → **1.7.0** (additive).

The struct and function declaration live in **`cpp/src/c_api/c_api_transfer_metrics_decl.h`** until merge time — the merge agent moves the §13 block into `cpp/include/chemometrics4all/c4a.h` and deletes the worktree-local header. This pattern lets the Phase 20 worktree co-exist with the parallel Phase 8/9/19 worktrees (which also append to `c4a.h`'s trailing section).

## Files created

### C engine

- `cpp/src/core/utilities/transfer_metrics.{c,h}` — reference C engine. Cyclic Jacobi PCA + metric kernels. Self-contained, no external linalg helper (Phase 20 is the first user of any SVD-like primitive; sharing the Jacobi sweep with future Phase 8/9/19 work is deferred).

### C ABI wrapper

- `cpp/src/c_api/c_api_transfer_metrics.cpp` — single extern "C" entry point.
- `cpp/src/c_api/c_api_transfer_metrics_decl.h` — worktree-local declaration of `c4a_transfer_metrics_t` + `c4a_transfer_metrics_compute`. Merge agent moves the §13 banner into `c4a.h`.

### Tests

- `cpp/tests/test_transfer_metrics.cpp` — smoke + invalid-args + parity tests (3 tests total).
- `cpp/tests/test_transfer_metrics_main.cpp` — worktree-only standalone driver. Lets the isolated worktree run the parity test without modifying `tests/main.cpp`. Deleted at merge time when `main.cpp` gains the standard `register_transfer_metrics_tests(r);` call.

### Frozen Python reference

- `parity/python_generator/src/c4a_parity_transfer_ref/__init__.py`
- `parity/python_generator/src/c4a_parity_transfer_ref/transfer_metrics.py`

Mirrors `nirs4all/analysis/transfer_metrics.py` with one intentional deviation: `spread_distance` subsamples via SplitMix64 instead of `numpy.random.RandomState.choice` (the C engine cannot bit-replicate NumPy's legacy MT19937 stream; SplitMix64 is trivial to mirror on both sides).

### Parity fixture + generator

- `parity/python_generator/scripts/generate_phase20_fixtures.py`
- `parity/fixtures/transfer_metrics_v1.json` — 5 cases:

| Name | Source shape | Target shape | n_components | k_neighbors | seed |
|---|---|---|---|---|---|
| `aligned_default`      | (40, 30) | (35, 30) | 10 | 10 | 0    |
| `aligned_tight`        | (40, 30) | (35, 30) | 3  | 5  | 17   |
| `aligned_wide`         | (40, 30) | (35, 30) | 8  | 8  | 42   |
| `mismatched_features`  | (40, 30) | (35, 25) | 5  | 6  | 7    |
| `small_samples`        | (12, 15) | (10, 15) | 4  | 3  | 2026 |

The `mismatched_features` case exercises the Grassmann NaN branch; `small_samples` exercises the n − 2 trustworthiness clamp.

### Docs

- `docs/algorithms/transfer_metrics.md` — operator reference (pipeline, parameters, tolerances, RNG choice, ABI).
- `roadmap/phase-20-transfer-metrics.md` — this brief.

## Files to update at merge time

The merge agent (or follow-up commit) is responsible for:

1. `cpp/include/chemometrics4all/c4a.h` — append a new §13 banner with the contents of `cpp/src/c_api/c_api_transfer_metrics_decl.h`.
2. `cpp/include/chemometrics4all/c4a_version.h` — bump `C4A_ABI_VERSION_MINOR` to 7.
3. `cpp/abi/expected_symbols_linux.txt` / `_macos.txt` / `_windows.txt` — append `c4a_transfer_metrics_compute`.
4. `cpp/tests/main.cpp` — declare and register the test suite:
   ```c++
   void register_transfer_metrics_tests(c4a_testing::Runner& r);
   /* ... in main() ... */
   register_transfer_metrics_tests(r);
   ```
5. Delete `cpp/src/c_api/c_api_transfer_metrics_decl.h` and `cpp/tests/test_transfer_metrics_main.cpp` (now obsolete).
6. `CHANGELOG.md` — add a `[0.1.0+abi.1.7.0]` section.

## Parity tolerances

| Metric | Abs tol | Rel tol |
|---|---|---|
| All 9 fields | `1e-9` | `1e-10` |

Iterative Jacobi diagonalisation + multiple matrix compositions (CKA / RV / Procrustes / Grassmann) drive a hard floor on bit-exact-with-NumPy parity. The bound is consistent with the LDLT-iterative baseline operators (Phase 5).

## Acceptance criteria

- ✅ Build clean (`cmake --build build/dev-debug`).
- ✅ Existing 82 tests still pass.
- ✅ 3 new tests pass via the standalone runner (`chemometrics4all_transfer_metrics_standalone`). After merge, those 3 tests join the main `chemometrics4all_tests` binary → 82 → **85** total.
- ✅ ABI: 126 → **127** symbols (additive). ABI bump 1.6.0 → **1.7.0**.
- ✅ Frozen Python reference under `c4a_parity_transfer_ref/` validates against the canonical `nirs4all` implementation in spirit (centroid, EVR, CKA, RV, Grassmann, Procrustes, Trustworthiness, Spread are mathematically equivalent; only the spread RNG path differs by design).

## Verification

```bash
# Configure
cmake --preset dev-debug

# Build core + ABI + tests
cmake --build build/dev-debug

# Run the existing suite (must stay at 82/82).
LD_LIBRARY_PATH=build/dev-debug/cpp/src \
    build/dev-debug/cpp/tests/chemometrics4all_tests
# expected: 82 passed, 0 failed

# Run the Phase 20 standalone runner.
LD_LIBRARY_PATH=build/dev-debug/cpp/src \
    build/dev-debug/cpp/tests/chemometrics4all_transfer_metrics_standalone
# expected: 3 passed, 0 failed
#   transfer_metrics_smoke         ok
#   transfer_metrics_invalid_args  ok
#   transfer_metrics_parity        ok

# Symbol count.
nm -D --defined-only build/dev-debug/cpp/src/libc4a.so.1.6.0 \
    | awk '$2=="T" {print $3}' | awk -F@ '{print $1}' | sort -u | wc -l
# expected: 127

# Regenerate the parity fixture (deterministic — produces a byte-identical file).
parity/python_generator/.venv/bin/python \
    parity/python_generator/scripts/generate_phase20_fixtures.py
```

## Next phase

To be decided.
