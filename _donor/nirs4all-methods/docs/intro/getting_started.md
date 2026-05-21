# Getting started

nirs4all-methods is currently documented as a chemometrics / NIRS operator
library. PLS model examples belong in `pls4all` unless and until this repository
adds a supported model surface.

## Build locally

```bash
cmake --preset dev-release
cmake --build --preset dev-release --parallel
ctest --preset dev-release --output-on-failure
```

## Explore the operator docs

The current generated pages live under `docs/algorithms/` and cover
preprocessing, baseline correction, splitters, filters, diagnostics,
augmenters, and utility methods.

## Benchmark status

The benchmark dashboard is a scaffold. It does not publish timings, speedups, or
cross-binding parity counts. The current benchmark contract is:

- `benchmarks/benchmark_registry.json`
- `benchmarks/truth_sources.lock.json`
- `benchmarks/check_truth_sources.py`

Validate registry edits with:

```bash
python benchmarks/check_truth_sources.py
```
