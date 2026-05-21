# Benchmark overview

pls4all has one benchmark matrix with two different parity questions:

1. **Binding parity**: do all pls4all language bindings call the C++ core
   and return the same result as the canonical native row?
2. **Reference parity**: does each implementation, including external
   libraries, agree with the registry-declared oracle for that method?

Those two questions must stay separate. An external library is not a
pls4all binding, so it should not fail binding parity just because it
uses a different convention. Conversely, a pls4all binding can match C++
and still fail reference parity if the C++ algorithm does not match the
method oracle.

This page is the single explainer covering:

1. [What gets measured](#what-gets-measured)
2. [How a cell is computed](#how-a-cell-is-computed)
3. [Reference policy](#reference-policy)
4. [Verdicts and tolerances](#verdicts-and-tolerances)
5. [Reading benchmark tables](#reading-benchmark-tables)
6. [Current caveats](#current-caveats)
7. [How to reproduce a run](#how-to-reproduce-a-run)

## What gets measured

| Axis | Values |
|---|---|
| **Algorithm** | The canonical methods in `benchmarks.parity_timing.registry.METHODS`. |
| **Backend** | pls4all core/bindings (`cpp`, `python_tier1`, `python_tier2`, `r_tier1`, `r_tier2`, `matlab_tier1`, `matlab_tier2`, `registry_pls4all`) plus registry-declared external references (`ref_python_*`, `ref_r_*`, `ref_matlab_*`). |
| **Cell** | A fixed `(algorithm, n, p, threads, libp4a_build)` tuple. Registry mode uses each method's canonical cell; full-matrix mode sweeps dataset sizes, builds and thread counts. |

The registry in `benchmarks/parity_timing/registry.py` is the contract:
method parameters, prediction key, canonical external references,
reference type and method-specific RMSE-relative tolerance all live there.

Each scheduled row records:

- `median_ms`, `min_ms`, `max_ms`: wall-clock timings for the backend.
- `binding_parity_ok`, `binding_parity_max_diff`,
  `binding_parity_note`: pls4all binding consistency against native C++.
- `reference_parity_ok`, `reference_parity_rmse_rel`,
  `reference_parity_note`: agreement with the canonical method oracle.
- `is_canonical_reference`, `reference_id`, `reference_kind`: provenance
  for the external oracle column.
- `versions_json`: library, BLAS and binding versions captured per row.

Legacy `parity_*` aliases still exist for older renderers. New code and
docs should use the explicit `binding_parity_*` and
`reference_parity_*` fields.

## How a cell is computed

```text
                       same (n, p, seed)
                              |
              +---------------+---------------+
              |                               |
              v                               v
   orchestrator writes one CSV         every backend subprocess
   data_<n>x<p>_seed<s>.csv            reads those same bytes
              |
              +-> pls4all.cpp / Python / R / MATLAB
              +-> sklearn-style, formula, classdef bindings
              +-> registry-declared external references
              |
              v
       each backend writes predictions to .npy
              |
              +-> Gate 1: pls4all bindings vs native C++ baseline
              |
              +-> Gate 2: all successful rows vs canonical external oracle
```

The shared CSV is mandatory. NumPy, R and Octave do not produce the same
random stream from the same integer seed, so comparing generated data
across languages would not be meaningful.

## Reference policy

### Gate 1: binding parity

Binding parity is only for pls4all rows:

- `pls4all_core`
- `pls4all_binding`

The comparator is the canonical native row for the same
`(algorithm, n, p)` group, currently `cpp` at one thread from the
`blas-omp` build when available, with `python_tier1` as fallback.

The expected tolerance is strict max-absolute agreement (`1e-6`) because
these rows should call the same C ABI with the same parameters. External
rows should be marked not applicable for this gate.

### Gate 2: reference parity

Reference parity is for every successful row: pls4all core, pls4all
bindings and external libraries. The comparator is the method's
canonical registry reference, for example sklearn for baseline PLS,
`pls` for some R-owned methods, libPLS for selected MATLAB references,
or a sanctioned in-tree/nirs4all oracle when no installable third-party
package exists.

This is the gate that answers "does the implementation under this method
name behave like the known reference?" When several external libraries
exist, they are also compared to the same canonical oracle so the
dashboard can show library divergence.

Runs that intentionally schedule only pls4all rows, such as
`--only-pls4all`, cannot evaluate Gate 2 unless the canonical external
reference row is also present. Those rows should be interpreted as
`reference parity not run`, not as a numerical failure.

## Verdicts and tolerances

| Verdict | Meaning |
|---|---|
| `exact` | Within the relevant tolerance. For Gate 1 this should normally be max-abs `<= 1e-6`. |
| `drift` | Ran successfully but differs beyond tolerance. Often a convention mismatch such as scaling, component ordering or stochastic plan choice. |
| `divergent` | Large numerical disagreement. Treat as a bug or a different algorithm under the same name until proven otherwise. |
| `not_available` | The backend/library does not implement that method. This is expected in legacy fixed/all external-reference sweeps. |
| `not_run` | The comparator was not scheduled, the backend timed out, or the orchestrator skipped the cell. |
| `error` | Runtime failure. The `reason` field should carry the process error. |

Selector methods may still declare RMSE-relative mask tolerances as
diagnostic budgets. Any tolerance wide enough to accept almost-disjoint
masks is a smoke threshold, not a release-quality oracle. The release
dashboard keeps the strict Gate 2 threshold at `rmse_rel <= 1e-3`; rows
above that threshold must stay visible as divergent unless a
package-compatible variant is implemented and tested separately.

## Reading benchmark tables

Per-method pages and the dashboard display one relevant gate per timed cell:

- C++ and external library columns show **reference parity** against the
  method oracle.
- Internal pls4all binding columns show **binding parity** against the C++
  baseline.

For external libraries, the binding gate is not applicable; the reference
gate is the meaningful one. A good external row can therefore be
reference-exact while having no binding verdict.

The dashboard also has a synthetic reference column that names the
canonical oracle for each method. It is provenance, not a timed backend.

## Current caveats

The May 2026 stabilization work fixed the main dual-gate reporting issues:
external rows no longer carry a binding-parity verdict, dashboard/static
tables render only the relevant gate per cell, and `--only-pls4all` runs
evaluate Gate 2 from a stored oracle snapshot. Remaining caveats:

- Older CSVs may still contain legacy external `parity_*` values. Regenerate
  benchmark CSVs before treating dashboard totals as release evidence.
- A missing `.reference_oracles` snapshot is a setup failure. Run the
  canonical reference backend after intentional reference-library updates;
  do not skip Gate 2 for pls4all-only runs.
- Fixture regeneration currently still requires the historical AOM bench
  oracle path when AOM/POP fixtures are checked. Until that oracle is
  packaged or pinned through nirs4all, fixture determinism is not fully
  reproducible from a clean clone.
- Tier-2 selector adapters now fail closed on unknown registry parameters,
  but any newly added selector must add explicit aliases/tests before it is
  accepted into release gates.

## How to reproduce a run

```bash
# Canonical registry cells, pls4all + declared external references.
benchmarks/cross_binding/run_overnight.sh

# Internal pls4all smoke only. Gate 2 uses stored oracle snapshots.
python benchmarks/cross_binding/orchestrator.py \
  --algorithms all --sizes 100x50 --threads 1 \
  --libp4a-build blas-omp --n-runs 2 --only-pls4all \
  --timeout 120 --out-csv /tmp/pls4all_only.csv

# Small reference-parity sample.
python benchmarks/cross_binding/orchestrator.py \
  --algorithms pls pcr --registry-cells \
  --canonical-pls4all-only --reference-backends registry \
  --threads 1 --libp4a-build blas-omp --n-runs 2 \
  --timeout 180 --out-csv /tmp/pls4all_refs.csv --force

# Exhaustive stress matrix.
FULL_MATRIX=1 REFERENCE_BACKENDS=registry \
  benchmarks/cross_binding/run_overnight.sh

# Legacy external coverage audit; NOT_IMPLEMENTED rows are expected.
FULL_MATRIX=1 REFERENCE_BACKENDS=all \
  benchmarks/cross_binding/run_overnight.sh

# Render an existing CSV only.
python benchmarks/cross_binding/combine_and_render.py \
  --csvs benchmarks/cross_binding/results/full_matrix.csv \
  --out docs/benchmarks/cross_binding.md
```
