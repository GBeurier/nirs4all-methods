# Benchmark overview

chemometrics4all carries a **cross-binding benchmark** as a hard project
invariant: every algorithm is run *in every binding* against the same
data bytes and compared element-wise to an external reference library
that "owns" that algorithm. Each method's per-page benchmark block in
[the methods catalogue](../methods/index.md) is a slice of this same
matrix.

This page is the single explainer covering:

1. [What gets measured](#what-gets-measured)
2. [How a cell is computed](#how-a-cell-is-computed)
3. [Verdicts and their meaning](#verdicts-and-their-meaning)
4. [Reading a per-method benchmark table](#reading-a-per-method-benchmark-table)
5. [The interactive dashboard](#the-interactive-dashboard)
6. [How to reproduce a run](#how-to-reproduce-a-run)

## What gets measured

Three axes, every cell:

| Axis | Values |
|---|---|
| **Algorithm** | The 71 canonical methods in `benchmarks.parity_timing.registry.METHODS` |
| **Backend**   | chemometrics4all bindings (`chemometrics4all.cpp.*`, `chemometrics4all.python`, `chemometrics4all.sklearn`, `chemometrics4all.R`, `chemometrics4all.R.formula`, `chemometrics4all.matlab`, `chemometrics4all.matlab.classdef`, `chemometrics4all.registry`) plus the **registry-declared external reference column** per language (`ref.python_sklearn`, `ref.r_pls`, `ref.r_mixomics`, `ref.matlab_libpls`, ...) |
| **Cell**      | A fixed `(n, p, threads)` triple. Default: `n=200, p=50, threads∈{1, 3, 10}`. A sweep mode also covers an 11-size grid. |

The registry is the benchmark contract in
`benchmarks/parity_timing/registry.py`: it lists each method, its canonical
cell parameters, prediction key, chemometrics4all route and supported external
references. `--reference-backends registry` is the default so only valid
external references are scheduled.

For each `(algorithm, backend, n, p, threads)` cell the orchestrator
records:

- **`median_ms`** — wall-clock median over `n_runs=5` runs, first
  discarded as warmup. Also `min_ms` / `max_ms`.
- **`parity_max_diff`** — max absolute element-wise difference of the
  predictions vs the reference for that cell.
- **`parity_ok`** — true iff `parity_max_diff ≤ tolerance`.
- **`versions_json`** — language, BLAS impl + version, binding /
  external library versions, chemometrics4all version. Always captured per
  row so a CSV row is reproducible six months later.

## How a cell is computed

```
                       same (n, p, seed)
                              │
              ┌───────────────┴───────────────┐
              ▼                               ▼
   orchestrator generates ONE                load the same CSV in
   CSV: data_<n>x<p>_seed<s>.csv             every backend subprocess
              │
              ├─→ chemometrics4all.cpp     (libc4a, ctypes)
              ├─→ chemometrics4all.python  (tier-1 ctypes wrapper)
              ├─→ chemometrics4all.sklearn (tier-2 sklearn-style)
              ├─→ chemometrics4all.R       (Rscript via rpy2)
              ├─→ chemometrics4all.matlab  (oct2py to Octave)
              ├─→ chemometrics4all.registry (canonical registry entry point)
              ├─→ sklearn.PLSRegression  (external)
              ├─→ pls::plsr              (external, via Rscript)
              ├─→ plsregress             (external, via Octave)
              └─→ ... (every ref column declared in the registry)
              │
              ▼
   each backend writes its predictions to .npy
              │
              ▼
   compare every backend's .npy vs the chosen reference's .npy
              │
              ▼
   record parity_max_diff, parity_ok, median_ms, versions_json
```

The point of generating one CSV and reading it in *every* backend is
that R's `set.seed`, NumPy's `default_rng` and Octave's `randn` produce
**different streams** from the same seed. Sharing CSV bytes is the
only way to make cross-language parity meaningful.

### Reference policy

For each `(algorithm, n, p)` group the parity reference is, in order:

1. **`chemometrics4all.cpp.ref` at 1 thread** when the algorithm has a libc4a
   entry point. This is the scalar reference build — no BLAS, no
   OpenMP — used as the deterministic baseline.
2. **`chemometrics4all.python` at 1 thread** otherwise.

References are *external libraries*, sanctioned for use either because
they are the canonical implementation of a paper (sklearn for PLS,
ropls for OPLS, spls for sparse PLS, ...) or because they are an
in-tree paper port with no maintained external package. AOM/POP now uses
the unreleased `nirs4all.operators.models.sklearn.aom_pls` module from the
git-pinned `nirs4all` dependency in
`benchmarks/cross_binding/requirements-git.txt`; MB-PLS / LW-PLS use the
same `nirs4all.operators.models.sklearn.*` family. These sanctioned
exceptions are documented in the registry header.

### Parity tolerance

Default: `1e-6` absolute. Per-algorithm overrides:

| Algorithm | Tolerance | Reason |
|---|---|---|
| `gpr_pls`         | 1e-3     | Iterative GP solver; different convergence criteria across libs |
| `bagging_pls`, `boosting_pls`, ensemble selectors | 1e-3 | Stochastic averaging; per-implementation RNG differences |
| `GA`, `PSO`, `VISSA` selectors | not applicable | Stochastic feature selection; per-implementation RNG streams |
| Selectors with stochastic resampling | scaled to ~1e-1 | RNG streams differ across languages |

### Thread control

Every backend reads the same environment vars before subprocess spawn:

```
OMP_NUM_THREADS      = N
OPENBLAS_NUM_THREADS = N
MKL_NUM_THREADS      = N
BLIS_NUM_THREADS     = N
BENCH_THREADS        = N
```

In addition:
- Python chemometrics4all also sets `Context.num_threads = N`.
- Octave scripts call `maxNumCompThreads(N)`.
- `OPENBLAS_NUM_THREADS == OMP_NUM_THREADS` (i.e. **not** OMP × BLAS)
  to avoid oversubscription.

Timeout per cell: **300 s**. Cells that time out are marked with the
⏳ icon.

## Verdicts and their meaning

| Verdict | Icon | What it means |
|---|---|---|
| **exact**         | ✓  | `parity_max_diff ≤ tolerance`. Bit-for-bit (within FP) agreement with the reference. |
| **drift**         | ≈  | Ran successfully but max diff `> tolerance`. Typically a convention mismatch (NIPALS vs SIMPLS, unit-variance vs centring-only, orthogonal-component ordering). |
| **divergent**     | ✗  | Max diff is much larger than tolerance — different algorithm under same name. The dashboard tooltip explains. |
| **not_available** | ⊘  | The library does not implement this algorithm (e.g. sklearn has no sparse PLS, plsregress has no CPPLS). In default registry mode these rows are normally not scheduled; they mostly appear in legacy `fixed` / `all` audit runs. |
| **not_run**       | —  | Timeout, infrastructure error, or the cell was skipped by the orchestrator. |
| **error**         | ⚠  | Runtime exception (`ModuleNotFoundError`, segfault, etc.). The tooltip carries the reason. |

`drift` is the most interesting verdict in practice: it surfaces
*legitimate* convention differences between libraries. E.g.
`ikpls` and `r_tier2` default to `scale_x=TRUE / scale_y=TRUE`
while chemometrics4all defaults to centring-only. Both are valid; users
should pick the convention matching their reference paper.

## Reading a per-method benchmark table

Every method's documentation page ends with a `Benchmarks` block in
the user-specified format:

```
+----------+----------+----------+----------+----------+
| backend  | parity   | n=200    | n=500    | n=1000   |
| (1 thr)  |          | median   | median   | median   |
+----------+----------+----------+----------+----------+
| chemometrics4all.cpp.ref      ✓  ref    | 0.85 ms  | 4.20 ms  | 16.3 ms  |
| chemometrics4all.cpp.blas+omp ✓  +0e+0  | 0.41 ms  | 1.92 ms  |  6.1 ms  |
| chemometrics4all.python       ✓  +0e+0  | 1.97 ms  | 5.14 ms  | 18.0 ms  |
| chemometrics4all.sklearn      ✓  +0e+0  | 2.10 ms  | 5.50 ms  | 18.4 ms  |
| chemometrics4all.R            ✓  +0e+0  | 14.2 ms  | 38.4 ms  | 84.5 ms  |
| sklearn              ≈  +2e-3  | 2.30 ms  | 6.40 ms  | 19.8 ms  |
| pls::plsr            ≈  +1e-2  | 12.5 ms  | 33.0 ms  | 74.0 ms  |
+----------+----------+----------+----------+----------+
```

Read it left to right:

- **backend** — the implementation. chemometrics4all rows on top, external
  reference rows below.
- **parity** — verdict + signed `parity_max_diff` vs the reference.
- **median** — median wall-clock ms across `n_runs` runs.

Each row's full provenance (BLAS version, library version, host CPU)
is captured in `versions_json` and shown by the dashboard.

## The interactive dashboard

The repo's landing page is a live, filterable dashboard built from
the same CSV that drives these tables. See the
[dashboard explainer](../landing/dashboard.md).

Highlights:

- Filter by algorithm group, by language band, by thread count.
- Toggle column presets: "Headline", "chemometrics4all only", "C++ tiers",
  "Externals", "Thread sweep".
- Hover a cell to see the parity max diff, the library version, the
  failure reason if any.
- Click an algorithm to drill into its per-method page in this
  documentation.

## How to reproduce a run

```bash
# Install git-only external refs not released on PyPI yet.
python -m pip install -r benchmarks/cross_binding/requirements-git.txt

# Canonical method/reference matrix, including build + docs render.
# Existing cells in results/full_matrix.csv are skipped by default.
benchmarks/cross_binding/run_overnight.sh

# Exhaustive stress matrix: all chemometrics4all bindings/tiers, all CPU libc4a
# backend variants, all default dataset sizes, threads 1/3/10, and
# registry-declared external references.
FULL_MATRIX=1 benchmarks/cross_binding/run_overnight.sh

# Legacy external coverage audit: includes fixed refs and can emit
# NOT_IMPLEMENTED for unsupported third-party library/algorithm pairs.
FULL_MATRIX=1 REFERENCE_BACKENDS=all benchmarks/cross_binding/run_overnight.sh

# Include CUDA when that preset is available.
FULL_MATRIX=1 LIBC4A_BUILD=all benchmarks/cross_binding/run_overnight.sh

# On the Pages branch (main), also publish the refreshed dashboard.
PUBLISH_WEB=1 benchmarks/cross_binding/run_overnight.sh

# Recompute after a chemometrics4all optimisation or dependency update.
FORCE=1 CLEAN_BUILD=1 benchmarks/cross_binding/run_overnight.sh

# Only retry cells that previously failed, preserving successful timings.
RERUN_FAILED=1 benchmarks/cross_binding/run_overnight.sh

# PLS headline sweep only.
python benchmarks/cross_binding/orchestrator.py \
  --algorithms pls --threads 1 3 10 --n-runs 5 \
  --resume-existing \
  --libc4a-build blas-omp --reference-backends registry \
  --out-csv benchmarks/cross_binding/results/full_matrix.csv

# Render an existing CSV only.
python benchmarks/cross_binding/combine_and_render.py \
  --csvs benchmarks/cross_binding/results/full_matrix.csv \
  --out docs/benchmarks/cross_binding.md
```

See [`methodology.md`](methodology.md) for fine details on tolerances,
hardware capture, and the resume / rerun semantics.
