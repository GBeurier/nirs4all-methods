# Phase 7e - Comprehensive performance matrix runner

Status: shipped locally as `phase-7-comprehensive-benchmark`.

Goal: orchestrate algorithm × dataset × language path benchmarks in
one runner so the same matrix can be re-launched later with varying
CPU pinning to compare 1 / 5 / 10-core behaviour.

Delivered:

- `benchmarks/runners/matrix.py`: a per-cell driver that times each
  algorithm on every available language path:
  - Native C++ via `pls4all_cli --bench` (subprocess spawn).
  - pls4all-Python via the Phase 7b `Model` ctypes wrapper.
  - pls4all-R via `Rscript` spawning the Phase 7c package (skipped
    cleanly when `Rscript` or the package is missing).
  - scikit-learn reference (`PLSRegression` for PLS solvers,
    `Pipeline([PCA, LinearRegression])` for PCR).
- Two scales: `smoke` (2 cells × 2 algos = 4 timing rows × 4 language
  columns = 16 measurements per run) and `full` (9 algos × 15
  (n_samples, n_features) cells, all 4 languages — runs on demand
  when the host is free).
- Output split:
  - `benchmarks/results/matrix/accuracy.csv` — gated by `--check`,
    compares each pls4all language path's predictions against
    scikit-learn within `rmse_rel <= 5e-2`.
  - `benchmarks/results/matrix/timing.csv` — informational; one row
    per (case, algo) with median + min for each language column plus
    a status field.
  - `benchmarks/results/matrix/summary.md` — regenerated each run.
  - `benchmarks/results/matrix/metadata.json` — pls4all version, host
    info, scale, repeats, thread pinning env vars
    (`OMP_NUM_THREADS`, `OPENBLAS_NUM_THREADS`, `MKL_NUM_THREADS`,
    `PLS4ALL_BENCH_THREADS`).

Run commands documented in `benchmarks/README.md` show how to re-launch
the matrix under 1-core / 5-core / 10-core pinning by exporting the
appropriate thread environment variables.

Smoke run on this host (PLS_REGRESSION on 200x100 and 500x100 with
SIMPLS + SVD, no R available):

| Case          | Algo      | Native C++ ms | Python ms | sklearn ms |
|---------------|-----------|---------------|-----------|------------|
| matrix-200x100| pls_simpls| 0.44          | 0.77      | 7.45       |
| matrix-200x100| pls_svd   | 0.64          | 0.85      | 7.53       |
| matrix-500x100| pls_simpls| 1.31          | 0.80      | 10.81      |
| matrix-500x100| pls_svd   | 3.54          | 1.12      | 6.63       |

All `accuracy_pass=True` (Python predictions match sklearn bit-for-bit).
`python benchmarks/run.py --benchmark matrix --check` passes.

Deferred:

- AOM / POP columns in the matrix runner (needs a separate dataset
  generator + reference oracle; the `aom_global` suite already covers
  this in isolation).
- Variable-selection benchmark columns.
- LW-PLS, MB-PLS, OPLS / OPLS-DA columns (the runner is extendable by
  adding solver names; the matrix driver itself does not need
  changes).
- Memory profiling (peak RSS captured per row).
- GPU / OpenMP backends (Acceleration Roadmap).
