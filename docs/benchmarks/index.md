# Benchmarks

chemometrics4all now has an initial cross-binding timing matrix.

The previous PLS dashboard text came from the pls4all template and is not a
valid claim for this repository. The committed benchmark surface starts with a
reference registry, split Python timing runners (`C4A.python` ABI-close
functions and `C4A.sklearn` estimators), and a dashboard payload generated from
`benchmarks/cross_binding/results/full_matrix.csv`.

## Current assets

| Asset | Status |
|---|---|
| `benchmarks/benchmark_registry.json` | Initial non-model method registry. |
| `benchmarks/truth_sources.lock.json` | Initial lockfile for external package/project references and literature sources. |
| `benchmarks/check_truth_sources.py` | Consistency checker for registry and lockfile. |
| `benchmarks/cross_binding/orchestrator.py` | Cross-binding timing runner for C ABI, Python direct functions, sklearn estimators, R bindings, and external references. |
| `benchmarks/cross_binding/results/full_matrix.csv` | Generated CSV consumed by the dashboard. |
| `docs/_static/bench-data.json` | Dashboard payload generated from the CSV when present. |

## What is in scope

- Preprocessing operators such as SNV, MSC, Savitzky-Golay, and baseline
  correction.
- Splitters, filters, diagnostics, transfer metrics, augmenters, and utilities.
- Pinned external references, cross-binding parity, and literature-backed method entries.

PLS regression and classification benchmarks belong in `pls4all`. AirPLS and
ArPLS remain in scope here because they are baseline correction methods, not PLS
models.

## Pages

- [Overview](overview.md)
- [Methodology](methodology.md)
- [External reference audit](external_reference_audit.md)
