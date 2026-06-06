# Cross-binding benchmark

This runner is the operational benchmark for full pls4all binding coverage.
It exercises the canonical method catalog across C++/Python/R/MATLAB
bindings and the external references that are actually valid for each
method.

## Registry

`benchmarks/parity_timing/registry.py` is the source of truth. For each
method it declares:

- canonical benchmark parameters and dataset shape;
- the pls4all call path and prediction key;
- supported external references, when an external library exists;
- tolerance and method-specific notes.

`run_overnight.sh` sets `REFERENCE_BACKENDS=registry` by default. The
lower-level `orchestrator.py` still defaults to `all` for legacy audits, so
pass `--reference-backends registry` explicitly when calling it directly.
Registry mode runs only the external references declared for the method
under test, so unsupported library/method combinations are not scheduled.

`--reference-backends fixed` and `--reference-backends all` are legacy
cross-product audit modes. They intentionally try fixed sklearn/R/MATLAB
scripts against broad algorithm sets. `NOT_IMPLEMENTED` rows in those
modes usually mean the external library does not support the algorithm, not
that a pls4all binding is missing.

## Run

```bash
# Canonical clean sweep. Defaults:
# CANONICAL_ONLY=1 REGISTRY_CELLS=1 REFERENCE_BACKENDS=registry
# LIBP4A_BUILD=blas-omp RESUME=1
benchmarks/cross_binding/run_overnight.sh

# Internal pls4all smoke across the complete catalog.
python benchmarks/cross_binding/orchestrator.py \
  --algorithms all --sizes 100x50 --threads 1 \
  --libp4a-build blas-omp --n-runs 2 --only-pls4all \
  --timeout 120 \
  --out-csv /tmp/pls4all_only.csv

# Registry-declared external references at canonical cells.
python benchmarks/cross_binding/orchestrator.py \
  --algorithms all --registry-cells --canonical-pls4all-only \
  --reference-backends registry --threads 1 \
  --libp4a-build blas-omp --n-runs 2 \
  --timeout 180 \
  --out-csv /tmp/pls4all_registry_refs.csv

# Legacy fixed-reference audit, expected to include NOT_IMPLEMENTED rows.
FULL_MATRIX=1 REFERENCE_BACKENDS=all \
  benchmarks/cross_binding/run_overnight.sh
```

For R/Octave-backed references, either activate the conda environment that
contains `Rscript` and `octave`, or set `PLS4ALL_R_ENV=/path/to/env`.
Several registry helpers still contain workstation-specific fallback paths;
the release-quality path is to make every reference resolve from the active
environment or from a pinned package.

## Oracle snapshots

External reference predictions are treated as method oracles, not as
throwaway timing by-products. When a canonical `ref_*` backend succeeds,
the orchestrator snapshots its prediction vector under
`benchmarks/cross_binding/data/.reference_oracles/`, keyed by algorithm,
reference id, canonical cell and prediction seed.

Later `--only-pls4all` runs still evaluate reference parity by loading
that stored oracle. If no snapshot exists yet, `reference_parity_ok=False`
with `reference oracle missing; run canonical reference backend`. Refresh
the snapshot only when the reference library, oracle code, benchmark cell
or fixture seed is intentionally updated.

## Dashboard refresh rows

The dashboard builder reads `results/full_matrix.csv` first and then any
`results/dashboard_refresh_*.csv` files. Refresh files are small deltas
keyed by `(algorithm, backend, libp4a_build, n, p, threads)`; later rows
replace stale cells without rewriting the full historical timing matrix.
Use them for targeted gate fixes such as a corrected oracle snapshot or a
binding adapter repair, and regenerate the full matrix when publishing a
new benchmark baseline.

Refresh rows used for the dashboard must be produced by the current
`adaptive-v1` timing schema. Do not keep old `n_runs=1` cold-start or
`warmup-v*` refresh rows in these files: they measure a different
protocol and can make fast C++ cells look hundreds of milliseconds
slower than the actual warm fit/predict path.

## Standalone interactive matrix

For local inspection of a single orchestrator CSV, render a standalone HTML
dashboard without rebuilding Sphinx:

```bash
python benchmarks/cross_binding/render_matrix_dashboard.py \
  --csv /tmp/n4m_full_parity.csv \
  --out build/cross_binding_dashboard/index.html
```

Open `build/cross_binding_dashboard/index.html` in a browser. The page reuses
the rich docs dashboard shell, with column presets, language-grouped backend
bands, method filters, failure focus mode, tooltips, and per-cell timing/parity
details. Matrix interactions are self-contained; method/footer navigation links
still target the docs tree and may not resolve from the `build/` output path.

## Timing protocol

Every scheduled cell starts with run #1 as a timed warmstart. If this
warmstart exceeds 5 min, the benchmark reports that single time and stops.
Otherwise run #2 is the first scored run, and the warmstart is excluded
from all reported statistics.

The run #2 duration chooses the total execution count:

| Run #2 duration | Total executions | Reported statistic |
|---|---:|---|
| `> 30 s` | 2 | run #2 only |
| `> 5 s` | 3 | mean of runs #2-#3 |
| `> 1 s` | 10 | median after warmstart |
| `> 0.1 s` | 20 | median after warmstart |
| `<= 0.1 s` | 40 | median after warmstart |

`reported_ms` is the dashboard score. `n_runs` counts scored samples
after dropping the warmstart; `total_runs` includes it. `median_ms` is a
compatibility alias for `reported_ms` in `adaptive-v1` CSVs.

## Result semantics

- `ok=False` with a `reason` means no timing/prediction was produced for
  that scheduled cell.
- `binding_parity_ok=False` means a pls4all core/binding row produced
  predictions and timing, but differs from the native C++ baseline beyond
  tolerance.
- `reference_parity_ok=False` means a successful row differs from the
  registry-declared method oracle beyond tolerance.
- External libraries are not pls4all bindings. Treat their binding parity
  fields as not applicable even if older CSVs contain legacy values.
- Dashboard and static tables show one relevant marker per cell: reference
  parity for C++/external rows, binding parity for internal pls4all bindings.
- `NOT_IMPLEMENTED` is expected only for legacy fixed/all reference modes
  when a third-party library does not implement the algorithm.
- In `--only-pls4all` runs, reference parity is evaluated from the stored
  oracle snapshot. Missing snapshots are blocking setup failures, not a
  reason to skip Gate 2.

## Current coverage

The previous 2026-05-18 coverage claim was produced by the registry sweep,
but the follow-up audit found dashboard/gate interpretation bugs. Treat the
numbers as historical smoke evidence until the dual-gate fixes land:

- 568/568 scheduled internal pls4all cells OK;
- 143/143 registry-declared external reference cells OK.

PCR, OPLS and AOM preprocessing are wired through R/MATLAB tier1/tier2.
Complex methods such as DI-PLS, N-PLS, SO-PLS, ON-PLS, ROSA, GPR-PLS,
MB-PLS, selectors and diagnostics route through registry parameters instead
of ad hoc wrapper signatures.

## AOM reference

Cross-binding AOM/POP references should import the unreleased nirs4all
operator implementation from a git-pinned dependency, not from a workstation
path:

```bash
python -m pip install -r benchmarks/cross_binding/requirements-git.txt
```

The pinned dependency currently resolves to
`git+https://github.com/GBeurier/nirs4all.git@f07f5472f73d43d886be989226b3be4b68d7846b`.
The imported module is `nirs4all.operators.models.sklearn.aom_pls`.
The registry reference id is
`ref_python_nirs4all_operators_models_sklearn_aom_pls`.

## AOM/Moment CUDA Facade Smoke

`aom_moment_cuda_facade_smoke.py` runs in a child Python process with
`N4M_LIB_PATH` set to the CUDA-enabled `libn4m`, because the binding loads one
shared library per process. It verifies the logical `n4m.aom` and `n4m.moment`
facades, moment/AOM PLS device routing, staged train-only selection, the
SavGol-focused staged preset, and the reusable AOM-PLS / POP-PLS selector
wrappers:

```bash
CUDA_VISIBLE_DEVICES=0 \
PYTHONPATH=bindings/python/src \
N4M_LIB_PATH=build/cuda-on/cpp/src/libn4m.so \
  /home/delete/.venv/bin/python benchmarks/cross_binding/aom_moment_cuda_facade_smoke.py \
  --cuda-lib build/cuda-on/cpp/src/libn4m.so \
  --output benchmarks/cross_binding/aom_moment_cuda_facade_smoke.json \
  --cuda-visible-devices 0 --n-samples 80 --n-features 1024 --cv 4
```

The JSON should keep nonzero CUDA-device PLS CV counters for `moment`, custom
`aom` chains, `aom_profile_sweep`, `staged_estimator` and
`savgol_focus_estimator` / `strict_family_lite_estimator`, with host PLS CV
counters at zero. It also records `prediction_replay_max_abs_error` for
`aom_pls` and `pop_pls`; those replay errors should stay at numerical noise
level (`<= 1e-10`).

## AOM Diversity CUDA Wrapper Smokes

The Ridge blender and operator PLS stack timing scripts can now benchmark the
ABI-close function, the native sklearn replay wrapper, or both with
`--mode native|wrapper|both`. The CUDA smoke CSVs are generated with one
visible GPU and `--mode both`, so they prove both the native function row and
the sklearn wrapper replay row on the CUDA-enabled shared library:

For release artifacts, the base `*_timing.csv` files are the CPU/dev-release
pairs and the `*_timing_cuda_smoke.csv` files are the one-GPU CUDA pairs. Use
the same command shape with `N4M_LIB_PATH=build/dev-release/cpp/src/libn4m.so`
and without `CUDA_VISIBLE_DEVICES` / CUDA-only flags to refresh a CPU pair.

```bash
CUDA_VISIBLE_DEVICES=0 \
PYTHONPATH=bindings/python/src \
N4M_LIB_PATH=build/cuda-on/cpp/src/libn4m.so \
  /home/delete/.venv/bin/python benchmarks/cross_binding/bench_aom_ridge_blender_timing.py \
  --output benchmarks/cross_binding/aom_ridge_blender_timing_cuda_smoke.csv \
  --repeats 1 --profile compact --cv 5 --mode both

CUDA_VISIBLE_DEVICES=0 \
PYTHONPATH=bindings/python/src \
N4M_LIB_PATH=build/cuda-on/cpp/src/libn4m.so \
  /home/delete/.venv/bin/python benchmarks/cross_binding/bench_aom_operator_pls_stack_timing.py \
  --output benchmarks/cross_binding/aom_operator_pls_stack_timing_cuda_smoke.csv \
  --repeats 1 --profile compact --cv 4 --mode both
```

The wrapper rows are `native_aom_ridge_blender_sklearn` and
`native_aom_operator_pls_stack_sklearn`. Their
`prediction_replay_max_abs_error` column should stay at numerical noise level
(`<= 1e-10`) because `predict(X)` must replay the native folded model state.
The same artifacts also guard deterministic cost telemetry. Ridge blender rows
must satisfy `n_ridge_blender_cv_fits = n_candidates * cv`,
`n_ridge_blender_final_fits = n_candidates`, and
`n_ridge_blender_fit_calls = n_candidates * (cv + 1)`. Operator PLS stack rows
must satisfy `n_operator_pls_stack_fit_calls = (n_specs + 1) * cv + 1`, with
PLS projector fits equal to that count times `n_operators` and Ridge head fits
equal to that count.

`bench_aom_ridge_superblock_timing.py` covers the catalogued strict-linear
donor-style AOM Ridge superblock reference. It benchmarks the Python function
and sklearn wrapper over the same strict operator bank, while each fold/final
Ridge fit goes through the native `n4m.ridge` binding:

```bash
CUDA_VISIBLE_DEVICES=0 \
PYTHONPATH=bindings/python/src \
N4M_LIB_PATH=build/cuda-on/cpp/src/libn4m.so \
  /home/delete/.venv/bin/python benchmarks/cross_binding/bench_aom_ridge_superblock_timing.py \
  --output benchmarks/cross_binding/aom_ridge_superblock_timing_cuda_smoke.csv \
  --repeats 1 --cv 4 --mode both
```

The expected rows are `native_aom_ridge_superblock` and
`native_aom_ridge_superblock_sklearn`; both should report
`ridge_backend=native` and replay error at numerical noise level.

`bench_aom_ridge_active_superblock_timing.py` covers the catalogued
strict-linear donor-style AOM Ridge active-superblock reference. It screens an
active operator subset fold-locally during alpha CV, then benchmarks the Python
function and sklearn wrapper over the same replayable final model:

```bash
CUDA_VISIBLE_DEVICES=0 \
PYTHONPATH=bindings/python/src \
N4M_LIB_PATH=build/cuda-on/cpp/src/libn4m.so \
  /home/delete/.venv/bin/python benchmarks/cross_binding/bench_aom_ridge_active_superblock_timing.py \
  --output benchmarks/cross_binding/aom_ridge_active_superblock_timing_cuda_smoke.csv \
  --repeats 1 --cv 4 --mode both
```

The expected rows are `native_aom_ridge_active_superblock` and
`native_aom_ridge_active_superblock_sklearn`; both should report
`ridge_backend=native` and replay error at numerical noise level.

`bench_aom_ridge_mkl_superblock_timing.py` covers the catalogued strict-linear
AOM Ridge MKL-light weighted-superblock reference. It learns fold-local KTA
operator weights during alpha CV, relearns weights on the full calibration
rows, and records replay from folded input-space coefficients:

```bash
CUDA_VISIBLE_DEVICES=0 \
PYTHONPATH=bindings/python/src \
N4M_LIB_PATH=build/cuda-on/cpp/src/libn4m.so \
  /home/delete/.venv/bin/python benchmarks/cross_binding/bench_aom_ridge_mkl_superblock_timing.py \
  --output benchmarks/cross_binding/aom_ridge_mkl_superblock_timing_cuda_smoke.csv \
  --repeats 1 --cv 4 --mode both
```

The expected rows are `native_aom_ridge_mkl_superblock` and
`native_aom_ridge_mkl_superblock_sklearn`; both should report
`mkl_mode=alignment`, `ridge_backend=native` and replay error at numerical
noise level.

`bench_aom_pls_superblock_timing.py` covers the catalogued strict-linear
donor-style AOM-PLS superblock reference. It concatenates strict operator
views, selects the PLS component count by train CV, and records the native PLS
route counters from the final fit:

```bash
CUDA_VISIBLE_DEVICES=0 \
PYTHONPATH=bindings/python/src \
N4M_LIB_PATH=build/cuda-on/cpp/src/libn4m.so \
  /home/delete/.venv/bin/python benchmarks/cross_binding/bench_aom_pls_superblock_timing.py \
  --output benchmarks/cross_binding/aom_pls_superblock_timing_cuda_smoke.csv \
  --repeats 1 --cv 4 --mode both --cuda-pls-min-device-features 1
```

The expected rows are `native_aom_pls_superblock` and
`native_aom_pls_superblock_sklearn`; both should report `pls_backend=native`
and replay error at numerical noise level.

`bench_aom_ridge_pls_superblock_timing.py` covers the catalogued strict-linear
donor-style AOM Ridge-PLS superblock reference. It concatenates strict operator
views, selects the PLS component count and Ridge-PLS penalty by train CV, and
records replay from folded input-space coefficients:

```bash
CUDA_VISIBLE_DEVICES=0 \
PYTHONPATH=bindings/python/src \
N4M_LIB_PATH=build/cuda-on/cpp/src/libn4m.so \
  /home/delete/.venv/bin/python benchmarks/cross_binding/bench_aom_ridge_pls_superblock_timing.py \
  --output benchmarks/cross_binding/aom_ridge_pls_superblock_timing_cuda_smoke.csv \
  --repeats 1 --cv 4 --mode both
```

The expected rows are `native_aom_ridge_pls_superblock` and
`native_aom_ridge_pls_superblock_sklearn`; both should report
`ridge_pls_backend=native` and replay error at numerical noise level.

`bench_aom_chain_ridge_pls_timing.py` covers the catalogued strict-linear
donor-style single-chain AOM Ridge-PLS selector. It applies each candidate
strict AOM chain sequentially, selects the chain, PLS component count and
Ridge-PLS penalty by train CV, fits through native `ridge_pls`, and records
replay from the folded raw input-space coefficients:

```bash
CUDA_VISIBLE_DEVICES=0 \
PYTHONPATH=bindings/python/src \
N4M_LIB_PATH=build/cuda-on/cpp/src/libn4m.so \
  /home/delete/.venv/bin/python benchmarks/cross_binding/bench_aom_chain_ridge_pls_timing.py \
  --output benchmarks/cross_binding/aom_chain_ridge_pls_timing_cuda_smoke.csv \
  --repeats 1 --cv 4 --mode both
```

The expected rows are `native_aom_chain_ridge_pls` and
`native_aom_chain_ridge_pls_sklearn`; both should report
`selection_mode=chain_ridge_pls`, `ridge_pls_backend=native` and replay error
at numerical noise level.

`bench_aom_ridge_global_timing.py` covers the donor-style AOM Ridge global
selector: one strict operator and one Ridge alpha selected by native CV, then
replayed from folded input coefficients:

```bash
CUDA_VISIBLE_DEVICES=0 \
PYTHONPATH=bindings/python/src \
N4M_LIB_PATH=build/cuda-on/cpp/src/libn4m.so \
  /home/delete/.venv/bin/python benchmarks/cross_binding/bench_aom_ridge_global_timing.py \
  --output benchmarks/cross_binding/aom_ridge_global_timing_cuda_smoke.csv \
  --repeats 1 --cv 4 --mode both
```

The expected rows are `native_aom_ridge_global` and
`native_aom_ridge_global_sklearn`; both should report
`ridge_backend=native_aom_chain_sweep` and replay error at numerical noise
level.

## AOM Robust-HPO Timing Smoke

`bench_aom_robust_hpo_timing.py` times the `n4m.aom_robust_hpo` native ABI
path and the `NativeAOMRobustHPORegressor` sklearn wrapper in the same run.
The CUDA smoke artifact covers both backends (`native_abi` and
`native_sklearn`) on the CUDA-enabled shared library across three synthetic
shapes (6 rows total):

```bash
CUDA_VISIBLE_DEVICES=0 \
PYTHONPATH=bindings/python/src \
N4M_LIB_PATH=build/cuda-on/cpp/src/libn4m.so \
  /home/delete/.venv/bin/python benchmarks/cross_binding/bench_aom_robust_hpo_timing.py \
  --output benchmarks/cross_binding/aom_robust_hpo_timing_cuda_smoke.csv \
  --repeats 1
```

The CSV must contain 6 rows: 3 shapes x 2 backends (`native_abi` +
`native_sklearn`), all with `profile=compact` and `library_path` pointing to
`build/cuda-on`. The `native_sklearn` rows carry
`prediction_replay_max_abs_error` (should stay at numerical noise level,
`<= 1e-10`), proving the wrapper replays the selected winner through
`X @ input_coefficients + intercept` on the CUDA-enabled build path.

## Moment Stack CUDA Smoke

`bench_moment_stack_timing.py` records `NativeMomentStackRegressor` timing and
aggregate route counters for its OOF and final base fits. A one-GPU PLS-only
smoke proves the stack can route its PLS base through the CUDA moment path:

```bash
CUDA_VISIBLE_DEVICES=0 \
PYTHONPATH=bindings/python/src \
N4M_LIB_PATH=build/cuda-on/cpp/src/libn4m.so \
  /home/delete/.venv/bin/python benchmarks/cross_binding/bench_moment_stack_timing.py \
  --output benchmarks/cross_binding/moment_stack_timing_cuda_smoke.csv \
  --repeats 1 --shapes 80x1024 --base-models pls --cv 4 --inner-cv 4 \
  --n-components 1 --cuda-pls-min-device-features 1 \
  --cuda-pls-parallel-folds
```

The CSV should show nonzero `n_base_oof_pls_moment_cuda_device_cv_fits` and
`n_base_final_pls_moment_cuda_device_cv_fits`, with the corresponding host PLS
CV counters equal to zero.

## PLS Cross-Validate Reference ABI Smoke

`bench_pls_cross_validate_timing.py` records the reserved
`n4m.pls_cross_validate` / `n4m_pls_cross_validate` reference surface that the
future fused PLS grinder is expected to accelerate. The current implementation
delegates to `sweep_run(heads=("pls",))`, so each row records max absolute
candidate-score and prediction deltas against that sweep reference.

```bash
PYTHONPATH=bindings/python/src \
N4M_LIB_PATH=build/dev-release/cpp/src/libn4m.so \
  /home/delete/.venv/bin/python benchmarks/cross_binding/bench_pls_cross_validate_timing.py \
  --output benchmarks/cross_binding/pls_cross_validate_timing.csv \
  --repeats 1
```

The one-GPU smoke lowers the PLS device threshold and enables bounded
parallel-fold scheduling:

```bash
CUDA_VISIBLE_DEVICES=0 \
PYTHONPATH=bindings/python/src \
N4M_LIB_PATH=build/cuda-on/cpp/src/libn4m.so \
  /home/delete/.venv/bin/python benchmarks/cross_binding/bench_pls_cross_validate_timing.py \
  --output benchmarks/cross_binding/pls_cross_validate_timing_cuda_smoke.csv \
  --repeats 1 --cuda-pls-min-device-features 1 --cuda-pls-parallel-folds
```

Both CSVs contain six rows: three shapes × full/score-only calls. The
equivalence deltas should stay at numerical zero. CPU rows prove host exact-CV
routing, while CUDA rows prove the same reference ABI can route PLS CV/final
fits through the CUDA device path. This is still a reference hook and timing
baseline, not the final fused many-chain IKPLS executor.

## Direct Moment Heads CUDA Smoke

`bench_direct_moment_heads_timing.py` times the reusable moment/linear heads
as both ABI-close functions and sklearn-style replay wrappers. The one-GPU
CUDA smoke artifact covers all 9 direct heads across three synthetic shapes
(54 rows total):

Each head appears as both a `native_function` row and a `sklearn_fit_predict`
row; both rows carry `surface_status=function_and_sklearn_replay`. The direct
head set is Ridge, PLS, PCR, CPPLS, weighted PLS, robust PLS, ridge-augmented
PLS, continuum regression and ECR.

```bash
CUDA_VISIBLE_DEVICES=0 \
PYTHONPATH=bindings/python/src \
N4M_LIB_PATH=build/cuda-on/cpp/src/libn4m.so \
  /home/delete/.venv/bin/python benchmarks/cross_binding/bench_direct_moment_heads_timing.py \
  --output benchmarks/cross_binding/direct_moment_heads_timing_cuda_smoke.csv \
  --repeats 1 --cuda-pls-min-device-features 1 --cuda-pls-parallel-folds
```

The CSV should contain 54 rows: 9 methods × 3 shapes × 2 backends. The
`replay_max_abs_error` column should stay at numerical noise level
(`<= 1e-10`), proving the wrappers replay the native fitted state through the
CUDA build path. PLS rows should also report nonzero
`n_pls_moment_cuda_device_cv_fits`, zero `n_pls_moment_host_cv_fits`, and
device CV fits equal to total PLS moment CV fits. The direct-head CPU artifact
uses the same schema and is regenerated with:

```bash
PYTHONPATH=bindings/python/src \
N4M_LIB_PATH=build/dev-release/cpp/src/libn4m.so \
  /home/delete/.venv/bin/python benchmarks/cross_binding/bench_direct_moment_heads_timing.py \
  --output benchmarks/cross_binding/direct_moment_heads_timing.csv \
  --repeats 1
```

Both CSVs must carry the current PLS route fields, including
`n_pls_moment_cuda_many_batched_batches` and
`n_pls_moment_cuda_many_batched_jobs`, so the artifact schema proves whether the
experimental many-batched route was requested or stayed at zero.

## Sweep And Selector CUDA Smokes

The committed sweep and selector CUDA smoke CSVs should be regenerated against
the current `build/cuda-on` shared library, not an older ABI symlink. These
smokes pin the reusable moment sweep, global AOM sweep, and AOM/POP selector
surfaces:

```bash
CUDA_VISIBLE_DEVICES=0 \
PYTHONPATH=bindings/python/src \
N4M_LIB_PATH=build/cuda-on/cpp/src/libn4m.so \
  /home/delete/.venv/bin/python benchmarks/cross_binding/bench_moment_sweep_timing.py \
  --output benchmarks/cross_binding/moment_sweep_timing_cuda_smoke.csv \
  --repeats 1 --native-only \
  --cuda-pls-min-device-features 1 --cuda-pls-parallel-folds

CUDA_VISIBLE_DEVICES=0 \
PYTHONPATH=bindings/python/src \
N4M_LIB_PATH=build/cuda-on/cpp/src/libn4m.so \
  /home/delete/.venv/bin/python benchmarks/cross_binding/bench_moment_sweep_timing.py \
  --output benchmarks/cross_binding/moment_sweep_timing_cuda_many_batched_smoke.csv \
  --repeats 1 --native-only --cv 5 \
  --cuda-pls-min-device-features 1 --cuda-pls-many-batched

CUDA_VISIBLE_DEVICES=0 \
PYTHONPATH=bindings/python/src \
N4M_LIB_PATH=build/cuda-on/cpp/src/libn4m.so \
  /home/delete/.venv/bin/python benchmarks/cross_binding/bench_aom_sweep_timing.py \
  --output benchmarks/cross_binding/aom_sweep_timing_cuda_smoke.csv \
  --repeats 1 --cv 4 --profile compact \
  --cuda-pls-min-device-features 1 --cuda-pls-parallel-folds

CUDA_VISIBLE_DEVICES=0 \
PYTHONPATH=bindings/python/src \
N4M_LIB_PATH=build/cuda-on/cpp/src/libn4m.so \
  /home/delete/.venv/bin/python benchmarks/cross_binding/bench_aom_selector_timing.py \
  --output benchmarks/cross_binding/aom_selector_timing_cuda_smoke.csv \
  --repeats 1
```

For the sweep CSVs, every exact PLS moment CV row should have host CV fits at
zero and CUDA-device CV fits equal to total PLS moment CV fits. The moment
sweep `many_batched` smoke keeps `cuda_pls_parallel_folds=False` and proves
the alternate CUDA route directly: PLS rows should report
`n_pls_moment_cuda_many_batched_batches=1`,
`n_pls_moment_cuda_many_batched_jobs=n_pls_moment_cv_fits`, and zero
parallel-fold batches/jobs. The moment
sweep smoke also includes a tall Ridge cell (`96x48`) so at least one Ridge
row reports nonzero `n_ridge_moment_cv_fits`; this keeps the multi-lambda
Ridge moment scorer covered separately from the wide dual/materialized Ridge
route. The artifact guard also compares the tall Ridge full-sweep row against
the tall Ridge `score_only` row, so the general sweep and score-only scorer
must agree on the selected lambda and CV RMSE. The tall Ridge rows should also
report `n_ridge_moment_eigen_path_preparations=5`,
`n_ridge_moment_eigen_path_cv_fits=25` and
`n_ridge_moment_direct_cv_fits=0`, proving that the multi-lambda moment route
is using the fold-local eigen-path rather than the per-lambda direct QR solve.
For selector rows, `replay_max_abs` should stay at numerical noise level
(`<= 1e-10`) for both function and sklearn surfaces.

AOM Ridge-containing timing smokes also expose the same Ridge moment route
telemetry (`n_ridge_moment_eigen_path_preparations`,
`n_ridge_moment_eigen_path_cv_fits`, `n_ridge_moment_direct_cv_fits`) so
campaign CSVs can distinguish logical Ridge moment CV work from the physical
eigen-path/direct-solve route.

## AOM Preprocess Timing Smoke

`bench_aom_preprocess_timing.py` covers the currently supported reusable
`n4m.aom_preprocess` primitive contract: direct single-operator `identity`,
`detrend_poly_1`, `savgol_smooth_5_2`, `savgol_derivative_5_2_1`,
`norris_williams_5_5_1`, `finite_difference_1`, `gaussian_1`,
`whittaker_100` and `fck_1` rows in both `soft` and `hard` gating modes.
Strict chains and model-scoring diversity remain covered by the AOM sweep and
staged campaign helpers.

```bash
CUDA_VISIBLE_DEVICES=0 \
PYTHONPATH=bindings/python/src \
N4M_LIB_PATH=build/cuda-on/cpp/src/libn4m.so \
  /home/delete/.venv/bin/python benchmarks/cross_binding/bench_aom_preprocess_timing.py \
  --output benchmarks/cross_binding/aom_preprocess_timing_cuda_smoke.csv \
  --repeats 1
```

The CSV should contain three shapes times two gating modes times nine direct
operators, with expected operator kinds `0`, `7`, `8`, `9`, `10`, `15`, `16`,
`17`, `18`, `weight_shape=1x1`, `weight_sum=1`, and
`replay_max_abs_error <= 1e-12` for every row.

## AOM Staged Oracle Comparison

Use `run_aom_staged_real_cohort.py` to generate held-out target rows for the
staged AOM/moment workflow on the local train/test NIRS splits:

```bash
PYTHONPATH=bindings/python/src N4M_LIB_PATH=build/dev-release/cpp/src/libn4m.so \
  /home/delete/.venv/bin/python benchmarks/cross_binding/run_aom_staged_real_cohort.py \
  --output /tmp/aom_staged_real_cohort_results.csv \
  --limit 10 --plan compact --max-chains 12 --top-k 12 --refit-top-k 6
```

For a one-GPU calibration run, use the CUDA build and force the exact PLS moment
route onto device. The output CSV records screen/refit PLS moment route counters
so the run can be audited without reading logs:

```bash
CUDA_VISIBLE_DEVICES=0 \
PYTHONPATH=bindings/python/src \
N4M_LIB_PATH=build/cuda-on/cpp/src/libn4m.so \
  /home/delete/.venv/bin/python benchmarks/cross_binding/run_aom_staged_real_cohort.py \
  --output /tmp/n4m_aom_staged_real_cohort_10_cuda.csv \
  --limit 10 --plan compact --cv 4 --max-chains 12 \
  --chain-chunk-size 6 --top-k 12 --refit-top-k 6 \
  --refit-per-head-top-k 2 --heads ridge,pls --components 1,2 \
  --ridge-lambdas 0.1,1.0,10.0 --moment-policy auto \
  --cuda-pls-min-device-features 1 --cuda-pls-parallel-folds \
  --backend-min-cuda-product 1
```

For mixed `--heads ridge,pls` campaigns, the real-cohort runner defaults to
`--split-head-scoring auto`. This is score-preserving: each mixed chunk is
scored as separate Ridge-only and PLS-only native calls, then rows are merged
back with the same `(chain, head, param)` scores. The point is launch shape, not
selection policy; it lets the existing Ridge and PLS head-homogeneous fast paths
run during broad preprocessing screens. Use `--split-head-scoring off` only when
you need the legacy single native call per mixed chunk for timing comparison.
Use `--pls-score-mode cv` (the default) for exact-CV PLS screens, or
`--pls-score-mode gcv_proxy` for explicit proxy-vs-exact recall campaigns; the
retained-candidate refit remains exact-CV either way and the CSV/diagnostics
record the requested mode.

The checked 10-row CUDA audit artifact
`aom_staged_real_cohort_compact10_split_head_auto_20260606.csv` validates that
this default is score-preserving on the compact mixed profile: against the
compact mixed diagnostic baseline it has 8 paired ties, median ratio `1`, no
test-set selection, and clean one-GPU PLS routing (`screen=960/0`,
`refit=220/0` device/host CV fits). Because the artifact uses
`--scale-x-grid false,true`, split-head counters are summed across both model
configs: the 8 OK rows report 32 split chunks and 64 score calls. The matching
diagnostics directory is `aom_staged_compact10_split_head_auto_20260606/`.

To compare global model preprocessing choices without using the test split for
selection, use `--scale-x-grid`. The staged campaign runs each listed value,
selects the model config by train exact-CV refit, and records
`scale_x`, `scale_x_values`, `selected_model_config_id` and
`split_head_scoring` in the output CSV:

```bash
CUDA_VISIBLE_DEVICES=0 \
PYTHONPATH=bindings/python/src \
N4M_LIB_PATH=build/cuda-on/cpp/src/libn4m.so \
  /home/delete/.venv/bin/python benchmarks/cross_binding/run_aom_staged_real_cohort.py \
  --output /tmp/n4m_aom_staged_real_cohort_10_cuda_scalex_grid.csv \
  --limit 10 --plan compact --cv 5 --max-chains 12 \
  --chain-chunk-size 6 --top-k 12 --refit-top-k 6 \
  --refit-per-head-top-k 2 --heads ridge,pls --components 1,2 \
  --ridge-lambdas 0.1,1.0,10.0 --moment-policy auto \
  --scale-x-grid false,true \
  --cuda-pls-min-device-features 1 --cuda-pls-parallel-folds \
  --backend-min-cuda-product 1
```

On the local diverse-10 cohort, this compact scale-grid run selected
`scale_x=True` for 8/10 rows and kept PLS exact-CV routing on one GPU
(`screen_cuda=1200`, host `0`; `refit_cuda=280`, host `0`). The oracle summary
was: AOM-PLS paired median ratio `1.03079` with 1 target win, AOM-Ridge paired
median ratio `1.05918` with 0 target wins, and TabPFN paired median ratio
`1.05956` with 4 target wins. The large remaining AOM-Ridge gap is concentrated
where the oracle uses SNV/MSC/EMSC/ASLS-family variants outside the strict
moment/linear scope.

A bounded `compact_wide` follow-up with `--max-features 1200`, `--max-chains
24`, `--top-k 16`, `--refit-top-k 8` produced 8 OK rows and 2 property-skipped
rows. Against the compact scale-grid baseline on the 8 paired rows it had 2
wins, 1 loss and 5 ties, median paired ratio `1.0`, mean paired ratio
`0.994179`, while increasing screen work to `2880` CUDA PLS CV fits. Treat that
as evidence to target specific preprocessing families/options before expanding
the full cartesian again.

Use the focused plans when the question is family recall rather than raw
cartesian width. Start with `savgol_focus`: it screens compact plus SavGol
smooth, SavGol-derivative and SavGol-combination stages early. Use
`strict_family_focus` as a heavier audit profile: it adds separate
finite-difference, Norris-Williams, Gaussian, FCK and Whittaker stages so a
small `--max-chains` budget reaches those families instead of stopping in the
early `lab` ordering, but those structured stages can dominate wall time on
some datasets.
For a reusable low-cost sklearn audit, use
`n4m.NativeAOMStrictFamilyLiteRegressor`: it keeps the strict-family stage
coverage but defaults to a tiny retained refit budget and no `scale_x` grid.

```bash
CUDA_VISIBLE_DEVICES=0 \
PYTHONPATH=bindings/python/src \
N4M_LIB_PATH=build/cuda-on/cpp/src/libn4m.so \
  /home/delete/.venv/bin/python benchmarks/cross_binding/run_aom_staged_real_cohort.py \
  --output /tmp/n4m_aom_staged_real_cohort_10_cuda_family_focus.csv \
  --limit 10 --plan strict_family_focus --cv 4 --max-chains 8 \
  --chain-chunk-size 4 --top-k 12 --refit-top-k 8 \
  --refit-per-head-top-k 2 --heads ridge,pls --components 1,2 \
  --ridge-lambdas 0.1,1.0,10.0 --moment-policy auto \
  --scale-x-grid false,true \
  --cuda-pls-min-device-features 1 --cuda-pls-parallel-folds \
  --backend-min-cuda-product 1 --max-features 1200
```

On the local diverse-10 cohort with `--max-features 1200`,
`savgol_focus --max-chains 6 --scale-x-grid false,true` produced 8 OK rows and
2 property-skipped rows. Against the compact scale-grid baseline on the 8
paired rows it had 5 wins, 2 losses and 1 tie, median paired ratio `0.995259`,
with median fit time `16.06s`; all PLS screen/refit CV fits stayed on CUDA
(host `0`). A broader `strict_family_focus --max-chains 4` partial run showed
family-stage wins but stalled during a MANURE refit after all checkpoints were
written, so treat it as a targeted audit profile rather than the fast default.

For incremental campaigns, prefer property filters over hand-picking dataset
names. The runner can skip rows by measured training size, feature count or
`n_train * n_features` while still writing auditable `skipped` rows:

```bash
CUDA_VISIBLE_DEVICES=0 \
PYTHONPATH=bindings/python/src \
N4M_LIB_PATH=build/cuda-on/cpp/src/libn4m.so \
  /home/delete/.venv/bin/python benchmarks/cross_binding/run_aom_staged_real_cohort.py \
  --output /tmp/n4m_aom_staged_real_cohort_10_cuda_compact_wide_p1200.csv \
  --limit 10 --plan compact_wide --cv 4 --max-chains 24 \
  --chain-chunk-size 8 --top-k 16 --refit-top-k 8 \
  --refit-per-head-top-k 3 --heads ridge,pls --components 1,2 \
  --ridge-lambdas 0.1,1.0,10.0 --moment-policy auto \
  --cuda-pls-min-device-features 1 --cuda-pls-parallel-folds \
  --backend-min-cuda-product 1 --max-features 1200
```

For preprocessing-family experiments, pass explicit staged screens as JSON.
The JSON must be a non-empty list of profile strings or stage objects accepted
by `n4m.aom_staged_chain_campaign`; the runner records the compact JSON in each
output row as `stages_json`:

```bash
CUDA_VISIBLE_DEVICES=0 \
PYTHONPATH=bindings/python/src \
N4M_LIB_PATH=build/cuda-on/cpp/src/libn4m.so \
  /home/delete/.venv/bin/python benchmarks/cross_binding/run_aom_staged_real_cohort.py \
  --output /tmp/n4m_aom_staged_real_cohort_3_cuda_custom_stages.csv \
  --limit 3 --plan compact \
  --stages-json '[{"name":"compact","profile":"compact","max_chains":8,"top_k":8},{"name":"sg_lab","profile":"lab","families":{"identity":[["identity",[]]],"savgol_smooth":[["savgol_smooth",[5,2]],["savgol_smooth",[9,2]],["savgol_smooth",[15,3]]],"savgol_derivative":[["savgol_derivative",[7,2,1]],["savgol_derivative",[15,3,1]]],"finite_difference":[["finite_difference",[1]]]},"templates":[["identity"],["savgol_smooth"],["savgol_derivative"],["savgol_smooth","finite_difference"]],"max_chains":8,"top_k":8}]' \
  --cv 4 --max-chains 8 --chain-chunk-size 4 --top-k 8 \
  --refit-top-k 6 --refit-per-head-top-k 2 \
  --heads ridge,pls --components 1,2 --ridge-lambdas 0.1,1.0,10.0 \
  --moment-policy auto --cuda-pls-min-device-features 1 \
  --cuda-pls-parallel-folds --backend-min-cuda-product 1 \
  --max-features 1200
```

Add `--diagnostics-dir /path/to/diag` when the campaign is meant to explain
which preprocessing families/options paid off. For every `ok` dataset row, the
runner writes `<safe_dataset_key>.diagnostics.json` with the selected train-CV
candidate, `impact`, `rank_diagnostics`, selected model config and route/counter
fields. It also appends one row per impact group to `impact_groups.csv`, with
`group_kind` values such as `by_operator`, `by_stage_family`,
`by_stage_option` and `by_head_stage_option`. This is offline audit output only:
production selection remains the campaign's train-CV refit winner and the
diagnostics writer does not use held-out/test scores for model choice.

Use `summarize_aom_impact_groups.py` to turn that raw impact table into a
cross-dataset preprocessing summary:

```bash
python benchmarks/cross_binding/summarize_aom_impact_groups.py \
  --diagnostics-dir /path/to/diag \
  --output /tmp/aom_impact_summary.csv \
  --summary-output /tmp/aom_impact_summary.md
```

The summary ranks impact groups by `dataset_wins`, rank-1 occurrences and
average train-CV rank. Dataset keys remain audit labels only; no production
model selection or routing is performed by the summarizer.

The synthetic staged timing smoke can also be run against the CUDA build on one
GPU. The release-readiness smoke output is
`aom_staged_chain_campaign_timing_cuda_smoke.csv`; it should show
`selection_uses_test_set=False`, nonzero `n_pls_moment_cuda_device_cv_fits`,
and zero `n_pls_moment_host_cv_fits`.

```bash
CUDA_VISIBLE_DEVICES=0 \
PYTHONPATH=bindings/python/src \
N4M_LIB_PATH=build/cuda-on/cpp/src/libn4m.so \
  /home/delete/.venv/bin/python benchmarks/cross_binding/bench_aom_staged_chain_campaign_timing.py \
  --output benchmarks/cross_binding/aom_staged_chain_campaign_timing_cuda_smoke.csv \
  --repeats 1 --plans compact --n-samples 96 --n-features 128 --cv 3 \
  --heads pls --components 1 --ridge-lambdas 0.1 --max-chains 4 \
  --chain-chunk-size 2 --top-k 4 --refit-top-k 3 \
  --refit-per-head-top-k 1 --moment-policy auto \
  --pls-score-mode cv \
  --cuda-pls-min-device-features 1 --cuda-pls-parallel-folds
```

Use `compare_aom_staged_to_oracles.py` after a staged AOM/moment campaign has
produced per-dataset scores. It is an offline CSV join: it does not fit models,
and it chooses the best score per dataset inside each oracle source.

```bash
python benchmarks/cross_binding/compare_aom_staged_to_oracles.py \
  --target /path/to/staged_results.csv \
  --target-score-column rmsep \
  --output /tmp/aom_staged_oracle_comparison.csv \
  --summary-output /tmp/aom_staged_oracle_comparison.md
```

By default it compares against the local AOM-PLS oracle, AOM-Ridge oracle and
TabPFN master results found under `/home/delete/nirs4all/nirs4all-aom` and
`/home/delete/nirs4all/nirs4all-lab`. Plain PLS/Ridge baselines are filtered
out of the AOM-family oracles, and the TabPFN oracle uses the best available
Raw/Opt reference per dataset.

Use `compare_aom_staged_variants.py` to compare multiple staged campaign CSVs
against each other before going back to the external oracles. It groups rows by
campaign configuration columns (`plan`, `stages_json`, heads, budget, scoring
knobs like `pls_score_mode`, property filters and CUDA knobs), not by dataset
identity. Exact-CV (`cv`) and proxy (`gcv_proxy`) screens therefore stay in
separate variants and are labeled accordingly. Dataset keys are used only for
offline paired score comparisons:

```bash
python benchmarks/cross_binding/compare_aom_staged_variants.py \
  --input compact=/tmp/n4m_aom_staged_real_cohort_10_cuda.csv \
  --input wide=/tmp/n4m_aom_staged_real_cohort_10_cuda_compact_wide_p1200.csv \
  --baseline-label compact \
  --output /tmp/n4m_aom_staged_variant_summary.csv \
  --summary-output /tmp/n4m_aom_staged_variant_summary.md
```

The summary includes OK/skipped/error counts, median scores/timings,
screen/refit PLS moment route totals, and paired win/loss/tie ratios against the
chosen baseline.

Fixture generation still has a separate dependency on the historical
`AOM_v0` bench oracle path. A clean CI/release gate must either vendor that
oracle, pin it through nirs4all, or skip those fixtures with an explicit
policy marker.
