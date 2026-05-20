# nirs4all-methods validation registry

This directory contains the deterministic validation snapshot used by the
documentation and dashboard.

The exporter reads the cross-binding orchestrator structurally with
Python's `ast` module. It does not import `benchmarks.cross_binding.orchestrator`,
so refreshing the registry does not load numpy, ctypes, `libn4m`, or
optional external reference packages.

## Refresh workflow

```bash
python -m benchmarks.validation.export_registry --write
python -m benchmarks.validation.validate_registry
PYTHONPATH=. python -m pytest benchmarks/validation/tests -q
```

The committed JSON files under `registry/` are the stable interface for
docs and dashboards:

- `methods.json`: method ids, labels, ABI entries, comparators, reference
  backends, registry truth-source ids, default datasets, and metrics.
- `references.json`: aggregate backend/language coverage.
- `datasets.json`: deterministic `(n, p)` cells and seed-base policy.
- `comparators.json`: numerical parity comparators and tolerance defaults.
- `tolerances.json`: per-method comparator/tolerance band.
- `suites.json`: smoke and full benchmark validation suites.
- `manifest.json`: schema metadata and counts.

After running cross-binding benchmarks, rebuild the docs payloads so the
same validation contracts are visible in method pages and in
`docs/_static/bench-data.json`.
