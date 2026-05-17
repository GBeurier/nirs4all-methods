# Cross-binding benchmark — methodology

## Goal

For each `(algorithm, backend, n, p, threads)` cell, report:

1. **Parity**: max absolute difference of predictions vs a reference
   backend, with a verdict (`✓` if ≤ tolerance, `✗` otherwise).
2. **Timing**: median, min, max wall-clock milliseconds across `n_runs`
   runs (first discarded as warmup).
3. **Versions metadata**: language, BLAS implementation, binding /
   external library versions.

The combination supports the project's core claim: *same numerical
result as the established external implementation, often faster*.

## Cell composition

| Axis | Values |
|---|---|
| **Algorithms** | Canonical `benchmarks.parity_timing.registry.METHODS` catalog (`--algorithms all`) |
| **Backends** | pls4all bindings + registry-driven external reference columns (`ref.<library>`) |
| **Sizes** | Default 11-size sweep, or one canonical MethodSpec cell per method with `--registry-cells` |
| **Thread counts** | 1, 3, 10 |
| **libp4a build** | `blas-omp` by default (OpenBLAS + OpenMP); `dev-release` available for the single-thread reference column |

A "skip" record is emitted (instead of a failure) when an external
backend doesn't implement a given algorithm (e.g. sklearn has no native
sparse PLS, plsregress has no CPPLS).

## Determinism

Each cell runs `n_runs = 5` times with seeds `[BASE, BASE+1, BASE+2,
BASE+3, BASE+4]`. The base seed is `1_234_567_890` — a uint32-safe
integer that round-trips losslessly through R/Octave doubles and is
accepted as `sklearn`'s `random_state`.

**All backends in the same cell read the same orchestrator-generated
CSV** (`benchmarks/cross_binding/data/data_<n>x<p>_seed<seed>.csv`).
This is essential because Python NumPy, R `set.seed()` and Octave
`randn("state", ...)` produce different streams from the same seed —
sharing the CSV bytes is the only way to make cross-language parity
meaningful.

## Reference policy

For each `(algorithm, n, p)` group, the parity reference is:

1. **`cpp` at 1 thread, `blas-omp` build** when present (default for all
   algos with a libp4a entry point); else
2. **`python_tier1` at 1 thread** as fallback (covers algos that don't
   have a direct ctypes path on the C++ side).

The reference's predictions are saved to
`benchmarks/cross_binding/data/.predictions/*.npy` and compared
element-wise to every other backend's `.npy` for that cell.

## Parity tolerance

Default tolerance: `1e-6` absolute. Per-algorithm overrides exist for
inherently noisier algorithms:

| Algorithm | Tolerance | Reason |
|---|---|---|
| `gpr_pls` | 1e-3 | Iterative GP solver, different convergence criteria across libs |
| `bagging_pls`, `boosting_pls`, ensembles | 1e-3 | Stochastic averaging; per-implementation RNG differences |
| `GA`, `PSO`, `VISSA` selectors | non-applicable | Stochastic feature selection; per-implementation RNG streams |

## Thread control

The orchestrator sets the following env vars **before** spawning each
backend subprocess:

```
OMP_NUM_THREADS      = N
OPENBLAS_NUM_THREADS = N
MKL_NUM_THREADS      = N
BLIS_NUM_THREADS     = N
BENCH_THREADS        = N
```

In addition:
- Python pls4all calls `Context.num_threads = N` for belt-and-braces.
- Octave bench scripts call `maxNumCompThreads(N)` at start.
- Externals (sklearn, pls::plsr, plsregress) rely on the env vars only.

`OPENBLAS_NUM_THREADS == OMP_NUM_THREADS` (i.e. not OMP×BLAS) to avoid
oversubscription.

## Notes on observed parity gaps

The smoke runs surfaced a recurring `0.054` divergence among three
backends: `ikpls`, `r_tier2`, `matlab_tier2`. Root cause: those wrappers
default to `scale_x=True / scale_y=True` (unit-variance scaling), while
`cpp`, `python_tier1`, `python_tier2`, `r_tier1`, `r_pls`, `r_mixomics`,
`matlab_tier1`, `matlab_pls` default to `scale_x=False / scale_y=False`
(centring only — the spectroscopy convention).

This is **not a bug**: both conventions are valid. The benchmark
surfaces it as a parity ✗ because the predictions DO differ. Users
should pick the convention matching their reference paper.

## Timeout

Per-cell wall-clock timeout: **300 s**. Cells that time out are marked
with the ⏳ icon in the rendered Markdown. Empty / failed cells are
marked `—`.

## Hardware context

Captured per run in the rendered Markdown header (host platform string,
BLAS impl + version, run date). For the headline runs documented in
this repo, the host is reproducible from the commit SHA + the
`results/full_matrix.csv` `versions_json` column.

## Re-running

```bash
# Complete canonical method/reference matrix, including build + docs render.
# Existing cells in results/full_matrix.csv are skipped by default.
benchmarks/cross_binding/run_overnight.sh

# Same run, then commit/push docs/_static + benchmark markdown and trigger
# the GitHub Pages docs workflow on the current branch.
PUBLISH_WEB=1 benchmarks/cross_binding/run_overnight.sh

# Recompute after a pls4all optimization or dependency update.
FORCE=1 CLEAN_BUILD=1 benchmarks/cross_binding/run_overnight.sh

# Only retry cells that previously failed, preserving successful timings.
RERUN_FAILED=1 benchmarks/cross_binding/run_overnight.sh

# PLS headline sweep only.
python benchmarks/cross_binding/orchestrator.py \
  --algorithms pls --threads 1 3 10 --n-runs 5 \
  --resume-existing \
  --libp4a-build blas-omp --reference-backends all \
  --out-csv benchmarks/cross_binding/results/full_matrix.csv

# Render
python benchmarks/cross_binding/combine_and_render.py \
  --csvs benchmarks/cross_binding/results/full_matrix.csv \
  --out docs/benchmarks/cross_binding.md
```
