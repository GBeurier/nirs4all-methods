# Benchmarks

chemometrics4all now has an initial cross-binding timing matrix.

The previous PLS dashboard text came from the pls4all template and is not a
valid claim for this repository. The committed benchmark surface starts with a
small reference registry, a Python tier-2 timing runner, and a dashboard payload
generated from `benchmarks/cross_binding/results/full_matrix.csv`.

## Current assets

| Asset | Status |
|---|---|
| `benchmarks/benchmark_registry.json` | Initial non-model method registry. |
| `benchmarks/truth_sources.lock.json` | Initial lockfile for package, project, and frozen local references. |
| `benchmarks/check_truth_sources.py` | Consistency checker for registry and lockfile. |
| `benchmarks/cross_binding/orchestrator.py` | Initial Python tier-2 timing runner. |
| `benchmarks/cross_binding/results/full_matrix.csv` | Generated CSV consumed by the dashboard. |
| `docs/_static/bench-data.json` | Dashboard payload generated from the CSV when present. |

## What is in scope

- Preprocessing operators such as SNV, MSC, Savitzky-Golay, and baseline
  correction.
- Splitters, filters, diagnostics, transfer metrics, augmenters, and utilities.
- Local frozen references and pinned external references.

PLS regression and classification benchmarks belong in `pls4all`. AirPLS and
ArPLS remain in scope here because they are baseline correction methods, not PLS
models.

## Pages

- [Overview](overview.md)
- [Methodology](methodology.md)
