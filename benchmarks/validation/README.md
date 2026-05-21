# `benchmarks/validation/` - declarative validation framework

Internal framework that wraps `benchmarks/parity_timing/registry.py`
and exposes its validation contracts as a deterministic JSON snapshot.
This is Phase PLS-1 of `VALIDATION_FRAMEWORK_ROADMAP.md`: the package
adds a declarative layer *on top of* the existing parity/timing
orchestrator without rewriting it.

## Purpose

The parity / timing registry is the source of truth for
"what pls4all algorithm must agree with which external library, to
what tolerance, on which dataset cell." Until now that contract lived
only in Python (`MethodSpec`) and a host-insensitive structural
lockfile (`benchmarks/parity_timing/truth_sources.lock.json`).

This package projects the registry onto a wider deterministic JSON
surface so:

- the docs (Phase PLS-2) can render a `### Validation contract`
  block per method without re-reading Python sources;
- the dashboard payload (`bench-data.json`) can carry validation
  metadata indirectly;
- a future suite runner (Phase PLS-3) can drive the orchestrator from
  the declarative bundle instead of duplicated CLI flags;
- CI can gate "registry changed but snapshot did not" without
  installing R / MATLAB / Octave.

The framework does NOT replace `benchmarks/parity_timing/lockfile.py`.
That lockfile is the host-insensitive structural snapshot used by the
parity-timing gate; the validation framework's `methods.json`
superset covers the same surface plus comparator / cell_params /
tolerance-quality fields.

## Files

| File | Purpose |
| ---- | ------- |
| `__init__.py` | Package entry point. |
| `models.py` | Stdlib-only dataclasses describing each JSON file. |
| `export_registry.py` | Build / write the snapshot from `parity_timing.registry`. |
| `validate_registry.py` | Fail non-zero on snapshot drift. |
| `tests/test_export_registry.py` | Determinism + drift-detection tests. |
| `registry/methods.json` | Method records - name, cell_params, tolerance, comparator, reference structure. |
| `registry/references.json` | Declared reference role slots (`python`, `r`, `ikpls`, `mixOmics`, ...) with the methods that declare each. |
| `registry/datasets.json` | Synthetic dataset cells (`100x50`, `500x500`, ...) sourced from `cross_binding/orchestrator.DEFAULT_SIZES`. |
| `registry/comparators.json` | The two parity gates from `parity_timing/_comparator.py` (`binding_parity`, `reference_parity`). |
| `registry/tolerances.json` | Per-method tolerance band + quality classification (`strict` / `relaxed` / `qualitative`). |
| `registry/suites.json` | Named bundles of (methods x datasets x comparators). Phase PLS-1 declares `smoke` and `benchmark`. |
| `registry/manifest.json` | Schema version, source path, and counts for each sibling file. |

Every JSON file is written with `indent=2`, `sort_keys=True`,
`ensure_ascii=True`, and a trailing newline. Re-running the export
on an unchanged registry must produce byte-identical output.

## Commands

```bash
# Refresh the committed snapshot after editing the parity registry.
python -m benchmarks.validation.export_registry --write

# Print payload counts without writing anything.
python -m benchmarks.validation.export_registry

# Fail non-zero on drift (snapshot != registry) or missing references.
python -m benchmarks.validation.validate_registry

# Run the focused pytest test.
python -m pytest benchmarks/validation/tests -q
```

The export reads `benchmarks/parity_timing/registry.py` directly. It
does not call MethodSpec reference factories, so it works on a host
without R / MATLAB / Octave installed - the structural data
(method names, cell params, comparator binding, declared reference
role slots, tolerance band) is enough for the Phase PLS-1 surface.

## Relationship to `parity_timing/truth_sources.lock.json`

Both files are host-insensitive structural snapshots of the parity
registry. They overlap, but they are not redundant:

- `truth_sources.lock.json` is the existing minimal contract - one
  row per `MethodSpec`, declares which reference slots are wired and
  the `rmse_rel_tol`. The parity-timing gate already uses it.
- `validation/registry/methods.json` carries the same row PLUS
  `description`, `cell_params`, `comparator`, `tolerance_quality`,
  the `needs_*` flags, and the canonical reference role. It exists
  because docs / dashboard need the wider surface, not because the
  parity-timing gate does.

The two files derive from the same source (`METHODS` in
`parity_timing/registry.py`) so they cannot drift independently as
long as both `python -m benchmarks.parity_timing.lockfile --write`
and `python -m benchmarks.validation.export_registry --write` are
run together after registry edits.

## Host-insensitivity contract

The Phase PLS-1 snapshot is reproducible on any host that can import
`benchmarks.parity_timing.registry`. Reference resolution - calling
factories to read `library_version` - is host-sensitive (R / MATLAB
packages may be missing) and is *deliberately deferred* to Phase
PLS-2 doc rendering, where it runs on a fully-provisioned host that
also drives the actual benchmark suite. The doc-time payload can add
resolved metadata; the committed Phase PLS-1 snapshot must not.

## Adding a new MethodSpec

After editing `benchmarks/parity_timing/registry.py`:

```bash
python -m benchmarks.parity_timing.lockfile --write
python -m benchmarks.validation.export_registry --write
python -m benchmarks.validation.validate_registry
```

Then commit `registry.py` along with the regenerated lockfile and
validation snapshot in the same change.

## Roadmap follow-up

Phase PLS-1 only ships the declarative layer. Items intentionally
*not* included in this pass:

- docs / dashboard wiring -> Phase PLS-2;
- `run_suite` wrapper around the orchestrator -> Phase PLS-3;
- CI lock and Sphinx build gate -> Phase PLS-4;
- portage to `chemometrics4all` -> Phase C4A-1.

See `VALIDATION_FRAMEWORK_ROADMAP.md` for the full plan.
