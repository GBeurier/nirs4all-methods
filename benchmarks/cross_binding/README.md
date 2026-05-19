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

Fixture generation still has a separate dependency on the historical
`AOM_v0` bench oracle path. A clean CI/release gate must either vendor that
oracle, pin it through nirs4all, or skip those fixtures with an explicit
policy marker.
