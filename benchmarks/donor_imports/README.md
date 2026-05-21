# Benchmarks

This directory contains the nirs4all-methods benchmark and reference registry,
plus an initial cross-binding timing runner.

nirs4all-methods is not pls4all: benchmark claims must be tied to real runs for
preprocessing, splitting, filtering, augmentation, diagnostics, and utility
operators. The initial generated matrix covers Python tier-2 bindings for a
conservative subset of already-gated methods; C++/R/MATLAB and external rows
use the same CSV contract and can be added incrementally.

## Files

| File | Purpose |
|---|---|
| `benchmark_registry.json` | Minimal human-readable catalog of benchmarkable non-model methods. |
| `truth_sources.lock.json` | Pinned reference sources used to justify future parity or accuracy claims. |
| `check_truth_sources.py` | Local consistency checker for the registry and lockfile. |
| `cross_binding/orchestrator.py` | Initial timing matrix generator. |
| `reference_snapshots/cross_binding/` | Stored outputs from canonical external reference methods, with version metadata. |
| `cross_binding/results/full_matrix.csv` | Generated dashboard input when the runner has been executed. |

## Current status

- Cross-binding timing results are generated from the runner, not inferred
  from pls4all.
- The current matrix is Python-only and should be treated as a bootstrap
  dashboard, not a final performance study.
- PLS regression/classification references belong in `pls4all`, not here.
- AirPLS and ArPLS remain in scope because they are baseline correction
  algorithms, not PLS models.

## Validation

Run:

```bash
python benchmarks/check_truth_sources.py
N4M_LIB_PATH=$PWD/build/dev-release/cpp/src/libn4m.so \
  PYTHONPATH=$PWD/bindings/python/src \
  python benchmarks/cross_binding/orchestrator.py --write-reference-snapshots
python docs/_extras/build_landing.py
```

The checker verifies that every registry entry points to locked truth sources,
that required fields are present, and that locked external packages/projects
are declared.

## Adding a benchmark entry

1. Add a method to `benchmark_registry.json` with a stable `method_id`.
2. Add every external reference method to `truth_sources.lock.json`.
3. Run `python benchmarks/check_truth_sources.py`.
4. Wire the method into `benchmarks/cross_binding/orchestrator.py` or a
   backend-specific runner.
5. Generate or refresh its stored external-reference output with
   `--write-reference-snapshots` only when the reference contract intentionally
   changes.
