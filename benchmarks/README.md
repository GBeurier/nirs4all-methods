# pls4all benchmarks

The benchmark suite times every shipped PLS solver across language paths
(native C++, pls4all-Python via ctypes, pls4all-R via `.Call`, reference
scikit-learn) and dataset sizes, then writes per-benchmark CSV + a
summary markdown.

## Suites

| Suite             | What it measures                                                                 | Reference                              |
|-------------------|----------------------------------------------------------------------------------|----------------------------------------|
| `aom_global`      | AOM-PLS global selection via `p4a_aom_global_select`                             | `nirs4all/bench/AOM_v0/aompls`         |
| `pls_regression`  | NIPALS / SIMPLS / SVD / kernel / wide-kernel / orth-scores / power / rand-SVD / PCR via `Model.fit/predict` | `sklearn.cross_decomposition.PLSRegression` (and `Pipeline([PCA, LinearRegression])` for PCR) |
| `matrix`          | Same algos across `(n_samples, n_features)` cells, all language paths            | sklearn (per-language accuracy gate)  |

## Build prerequisites

`libp4a` and `pls4all_cli` must be built (the matrix benchmark spawns
the CLI for the native C++ timing column):

```bash
cmake --build --preset dev-release --parallel
```

Python: `numpy` and (for `pls_regression` / `matrix`) `scikit-learn`.

R (optional): `Rscript` on PATH and the `pls4all` R package installed
from `bindings/r/pls4all/`. See [`../bindings/r/README.md`](../bindings/r/README.md).
If R or the package is missing, the matrix runner records
`r_not_available` in the timing CSV and skips R-side accuracy.

## Run

```bash
# Default: run aom_global + pls_regression + matrix at the smoke scale,
# write live CSV + summary, exit non-zero on any accuracy gate failure.
PLS4ALL_LIB_PATH=$(pwd)/build/dev-release/cpp/src/libp4a.so \
  python benchmarks/run.py

# Explicit subset.
python benchmarks/run.py --benchmark aom_global
python benchmarks/run.py --benchmark pls_regression --scale smoke
python benchmarks/run.py --benchmark matrix --scale smoke

# Assert committed accuracy CSVs match a fresh run (drift gate).
python benchmarks/run.py --check
```

The `--scale full` flag enables the published target matrix for the
`pls_regression` and `matrix` suites:

```
algos      : 9 (NIPALS, OScores, SIMPLS, Kernel, WideKernel, SVD, Power,
              RandomizedSVD, PCR)
n_samples  : 200, 500, 1000, 2000, 10000
n_features : 100, 1000, 10000
```

The full grid is 9 × 5 × 3 = 135 cells. It is **not** run by default —
launch it when the host machine is free. Memory and runtime grow quickly
on the (10000, 10000) cell; expect several GB of RAM per side.

If the bench oracle is not in the default layout (`pls4all/` and
`nirs4all/` as sibling directories), point at its parent:

```bash
PLS4ALL_AOM_BENCH_DIR=/parent/of/nirs4all-checkout \
  python benchmarks/run.py
```

## CPU pinning

The runner records `OMP_NUM_THREADS`, `OPENBLAS_NUM_THREADS`,
`MKL_NUM_THREADS` and `PLS4ALL_BENCH_THREADS` per row. Pin the thread
count externally:

```bash
# Single-core baseline.
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  python benchmarks/run.py --benchmark matrix --scale full --repeats 5

# Five-core run.
OMP_NUM_THREADS=5 OPENBLAS_NUM_THREADS=5 MKL_NUM_THREADS=5 \
  python benchmarks/run.py --benchmark matrix --scale full --repeats 5

# Ten-core run.
OMP_NUM_THREADS=10 OPENBLAS_NUM_THREADS=10 MKL_NUM_THREADS=10 \
  python benchmarks/run.py --benchmark matrix --scale full --repeats 5
```

`pls4all` is currently single-threaded internally (the parallel backend is
in the Acceleration Roadmap), so the core-pinning variables only affect
the reference implementations and any BLAS used by the kernel-algorithm
PLS variants.

For the cross-binding orchestrator, `--threads N` sets `OMP_NUM_THREADS`,
`OPENBLAS_NUM_THREADS`, `MKL_NUM_THREADS`, `BLIS_NUM_THREADS`, and the
binding context thread budget for each launched cell. Sklearn-compatible
reference adapters that expose `n_jobs` read `BENCH_SKLEARN_N_JOBS`, falling
back to that same `BENCH_THREADS` value:

```bash
# Run registry-backed sklearn/auswahl references with eight jobs per cell.
python benchmarks/cross_binding/orchestrator.py \
  --algorithms all --registry-cells --threads 8 \
  --reference-backends registry --workers 1

# Override only sklearn n_jobs while keeping BLAS/OpenMP at one thread.
BENCH_SKLEARN_N_JOBS=8 \
  python benchmarks/cross_binding/orchestrator.py \
  --algorithms selection.random_frog --registry-cells --threads 1 \
  --reference-backends registry --workers 1
```

## Layout

```
benchmarks/
├── README.md                                  this file
├── run.py                                     unified driver
├── runners/
│   ├── _harness.py                            AccuracyRecord, TimingRecord, helpers
│   ├── aom_global.py                          AOM-PLS global vs nirs4all/bench
│   ├── pls_regression.py                      PLS regression matrix (Python)
│   └── matrix.py                              cross-language matrix
└── results/
    ├── aom_global/                            committed CSVs + summary
    │   ├── accuracy.csv                       gated by --check
    │   ├── timing.csv                         informational
    │   └── summary.md                         informational
    ├── pls_regression/
    │   ├── accuracy.csv                       gated
    │   ├── timing.csv                         informational
    │   └── summary.md                         informational
    └── matrix/
        ├── accuracy.csv                       gated
        ├── timing.csv                         informational
        ├── summary.md                         informational
        └── metadata.json                      versions + host info + env
```

## What is gated, what is not

- **Gated** (drift fails `--check`): every `accuracy.csv`. Columns track
  numerical equivalence between pls4all and the reference (operator
  match, component-count match, prediction RMSE delta, best-score delta).
- **Not gated**: every `timing.csv`, `summary.md`, `metadata.json`. Wall
  times are platform-dependent.

The runner exits non-zero on any case where `accuracy_pass` is `False`,
with or without `--check`.

## What the timing numbers do and do not show

- `pls4all_python_ms_*`: full `fit + predict` time including Python ↔
  ctypes marshalling. With the NumPy zero-copy `MatrixView` constructor
  shipped in Phase 7b, large matrices avoid the per-call list copy and
  the overhead is near-zero — but R-side wrappers still pay a row-major
  conversion cost (R is column-major).
- `native_cpp_ms_*`: pure C++ fit + predict from `pls4all_cli --bench`.
  Comes closest to the kernel cost.
- `pls4all_r_ms_*`: full `fit + predict` from R, including the column ↔
  row conversion in the `.Call` gateway. Will be similar to or slightly
  slower than Python for the same size.
- `sklearn_ms_*`: reference implementation, includes its own Python
  overhead.

Comparing pls4all vs sklearn at small (n, p) tends to under-show pls4all
because Python overhead dominates both sides. The native C++ column is
the right column to read for pure kernel performance.

## Adding a new benchmark

1. Add a runner module under `benchmarks/runners/`.
2. Either expose `run(*, repeats: int = 5) -> tuple[list[AccuracyRecord],
   list[TimingRecord], dict]` (for record-style benchmarks) or return a
   `dict` with `timing`, `accuracy`, `metadata` keys (for matrix-style).
3. Wire it into `benchmarks/run.py`'s `_BENCHMARKS` table.
4. Commit the resulting deterministic CSV + summary markdown.

Numerical regressions should fail loudly via `accuracy_pass`. Timing
regressions should be tracked in the summary markdown but never gate the
release.
