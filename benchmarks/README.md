# pls4all benchmarks

Phase 7a — first benchmark suite.

## Scope

Compares the **public C ABI** (`libp4a` via the ctypes Python wrapper) with
reference implementations. The first benchmark targets AOM-PLS global
selection against the local `nirs4all/bench/AOM_v0/aompls` oracle.

| Benchmark    | pls4all surface            | Reference                                |
|--------------|----------------------------|------------------------------------------|
| `aom_global` | `p4a_aom_global_select`    | `nirs4all/bench/AOM_v0/aompls`           |

The AOM-PLS benchmark drives selection from the public C ABI: it builds an
`OperatorBank` and a `ValidationPlan`, feeds row-major X / Y through the
`p4a_matrix_view_t` boundary, and compares results against the oracle on the
same data with identical fold indices.

## Run

Build `libp4a` first:

```bash
cmake --build --preset dev-release --parallel
```

Then run:

```bash
# Writes results/aom_global_accuracy.csv (gated),
#        results/aom_global_timing.csv (informational),
#        results/aom_global_summary.md.
PLS4ALL_LIB_PATH=$(pwd)/build/dev-release/cpp/src/libp4a.so \
  python benchmarks/run.py

# Assert the committed accuracy CSV matches a fresh run.
PLS4ALL_LIB_PATH=$(pwd)/build/dev-release/cpp/src/libp4a.so \
  python benchmarks/run.py --check
```

If the bench oracle is not in the default layout (`pls4all/` and `nirs4all/`
as sibling directories), point at its parent:

```bash
# Expects ${PLS4ALL_AOM_BENCH_DIR}/nirs4all/bench/AOM_v0/aompls/ to exist.
PLS4ALL_AOM_BENCH_DIR=/parent/of/nirs4all-checkout \
  python benchmarks/run.py
```

## Layout

```
benchmarks/
├── README.md                              this file
├── run.py                                 driver (writes CSV + markdown)
├── runners/
│   ├── _harness.py                        timing + table helpers
│   └── aom_global.py                      AOM-PLS global benchmark
└── results/
    ├── aom_global_accuracy.csv            COMMITTED — gated by --check
    ├── aom_global_timing.csv              COMMITTED, informational
    └── aom_global_summary.md              COMMITTED, informational
```

## What is gated, what is not

- **Gated** (`--check` will fail on drift): `aom_global_accuracy.csv`. The
  columns `selected_operator_index_match`, `selected_n_components_match`,
  `best_score_abs_delta`, `prediction_rmse_abs_delta`,
  `prediction_rmse_rel_delta` are the canonical regression signal.
- **Not gated**: `aom_global_timing.csv` and `aom_global_summary.md`. Wall
  times are platform-dependent. The runner still writes them on every
  invocation so trends are visible, but `--check` ignores them.

The runner exits non-zero if any case's `accuracy_pass` is `False`, even
without `--check`.

## What the timing numbers do and do not show

- `pls4all_end_to_end_ms_*` is the wall time of one full `fit + predict`
  call through the public C ABI, **including Python list <-> ctypes
  marshalling overhead** (Python lists are copied into / out of
  `ctypes.c_double` arrays on every call).
- `bench_end_to_end_ms_*` is the wall time of one full `fit + predict`
  through the NumPy bench oracle.
- Speedup is the ratio of the medians. It reflects the combined effect of
  the C++ kernel, ctypes marshalling, and the bench oracle's NumPy
  implementation overhead — it is **not** a pure kernel comparison.
- Datasets are intentionally small (≤ 16 samples × 10 features). Larger
  shapes will exercise binding overhead more heavily; that benchmark
  variant is deferred until the Python binding gains NumPy zero-copy
  matrix views (Phase 2 of the binding roadmap).

## Adding a new benchmark

1. Add a runner module under `benchmarks/runners/`.
2. Expose a `run(*, repeats: int = 5) -> tuple[list[AccuracyRecord],
   list[TimingRecord]]` function.
3. Wire it into `benchmarks/run.py`.
4. Commit the resulting deterministic CSV + summary markdown.

Numerical regressions should fail loudly via `accuracy_pass`. Timing
regressions should be tracked in the summary markdown but never gate the
release.
