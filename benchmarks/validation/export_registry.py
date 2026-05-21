"""Export `benchmarks.parity_timing.registry` to deterministic JSON.

Phase PLS-1 of `VALIDATION_FRAMEWORK_ROADMAP.md`: surface the parity /
timing registry as a structural JSON snapshot under
`benchmarks/validation/registry/`. The snapshot drives Phase PLS-2
(docs + dashboard payload) and Phase PLS-3 (run wrapper).

Usage
-----

    # CI / pre-commit gate: build payload in memory and print counts.
    python -m benchmarks.validation.export_registry

    # Refresh the committed snapshot.
    python -m benchmarks.validation.export_registry --write

The export is host-insensitive: it does NOT call MethodSpec reference
factories (those probe R / MATLAB / Python package versions). The
structural data - method names, cell parameters, tolerance, comparator
binding, declared reference role slots - is enough for the docs layer.
Host-sensitive resolution of `library_name` / `version` happens later,
at doc-render time on a fully-provisioned host.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from benchmarks.parity_timing.registry import (
    METHODS,
    _truth_source_quality,
    iter_reference_factories,
)

from .models import (
    SCHEMA_DESCRIPTION,
    SCHEMA_VERSION,
    ComparatorRecord,
    DatasetRecord,
    Manifest,
    MethodRecord,
    ReferenceRecord,
    SuiteRecord,
    ToleranceRecord,
)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

REGISTRY_DIR = Path(__file__).resolve().parent / "registry"
METHODS_PATH = REGISTRY_DIR / "methods.json"
REFERENCES_PATH = REGISTRY_DIR / "references.json"
DATASETS_PATH = REGISTRY_DIR / "datasets.json"
COMPARATORS_PATH = REGISTRY_DIR / "comparators.json"
TOLERANCES_PATH = REGISTRY_DIR / "tolerances.json"
SUITES_PATH = REGISTRY_DIR / "suites.json"
MANIFEST_PATH = REGISTRY_DIR / "manifest.json"

REGISTRY_FILES: tuple[Path, ...] = (
    METHODS_PATH,
    REFERENCES_PATH,
    DATASETS_PATH,
    COMPARATORS_PATH,
    TOLERANCES_PATH,
    SUITES_PATH,
    MANIFEST_PATH,
)

# ---------------------------------------------------------------------------
# Cross-binding orchestrator dataset surface
# ---------------------------------------------------------------------------
#
# `benchmarks/cross_binding/orchestrator.py` exposes `DEFAULT_SIZES` and
# `DEFAULT_SEED_BASE`. Importing that module here would run its top-level
# side effects (creates `data/`, `results/` directories, imports numpy)
# and is not necessary for the Phase PLS-1 declarative snapshot. We mirror
# the constants explicitly; the focused test under `tests/` cross-checks
# them against the orchestrator at runtime when it is importable.

DEFAULT_SIZES: tuple[tuple[int, int], ...] = (
    (100, 50), (100, 500), (100, 2500),
    (500, 50), (500, 500), (500, 2500),
    (2500, 50), (2500, 500), (2500, 2500),
    (10000, 50), (10000, 500),
)
DEFAULT_SEED_BASE: int = 1_234_567_890

# Smoke = the lightest cross-binding cells; large enough to exercise every
# code path, small enough to run within seconds. The orchestrator's
# fastest cell is (100, 50); pair it with (100, 500) so dimension-
# dependent code paths (block partitioning, interval selectors) still
# fire.
SMOKE_SIZES: tuple[tuple[int, int], ...] = ((100, 50), (100, 500))

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _structural_reference_kind(method: Any) -> str:
    """Host-insensitive analogue of
    `registry.reference_kind(method)`.

    The runtime classifier inspects RESOLVED reference factories, which
    means hosts without R installed report fewer references and may
    collapse `external` to `paper_only`. The validation snapshot must be
    reproducible, so we classify on declared structure only:

    - `paper_only` when `method.paper_only` is truthy *and* no executable
      reference factory is declared.
    - `external` otherwise (covers `nirs4all_sanctioned` too - the docs
      layer refines that distinction on a fully-provisioned host).
    """
    has_any = (
        method.python_reference is not None
        or method.r_reference is not None
        or bool(method.extra_references)
    )
    if not has_any and method.paper_only:
        return "paper_only"
    return "external"


def _canonical_reference_role(method: Any) -> str | None:
    """First declared reference factory slot, host-insensitive.

    Mirrors the resolution order used by
    `iter_reference_factories(method)` (python -> r -> extras) but stops
    at the first declared slot regardless of whether the factory
    resolves on this host. Returns None when the method is paper-only
    with no declared factories.
    """
    if method.python_reference is not None:
        return "python"
    if method.r_reference is not None:
        return "r"
    for role, _factory in method.extra_references:
        return role
    return None


def _comparator_for(method: Any) -> str:
    """The comparator a parity check uses for this method.

    Every MethodSpec is gated by the relative-RMSE `reference_parity`
    comparator (see `benchmarks/parity_timing/_comparator.py`). The
    `binding_parity` (max_abs_diff <= 1e-6) comparator gates
    pls4all-binding consistency separately - it is the same comparator
    for every method and lives in the comparators JSON file.
    """
    if method.paper_only and not (
        method.python_reference is not None
        or method.r_reference is not None
        or bool(method.extra_references)
    ):
        # Pure paper_only methods have no executable parity target; the
        # framework still records the binding comparator for the
        # pls4all-vs-pls4all gate, but the reference comparator is not
        # exercised.
        return "binding_parity"
    return "reference_parity"


# ---------------------------------------------------------------------------
# Builders
# ---------------------------------------------------------------------------


def build_methods() -> list[MethodRecord]:
    """Build the `methods.json` payload from the parity registry."""
    out: list[MethodRecord] = []
    for method in METHODS:
        extra_roles = [role for role, _f in method.extra_references]
        out.append(MethodRecord(
            name=method.name,
            description=method.description,
            cell_params=dict(method.cell_params),
            rmse_rel_tol=float(method.rmse_rel_tol),
            prediction_key=method.prediction_key,
            paper_only=method.paper_only,
            notes=method.notes,
            needs_x_target=method.needs_x_target,
            needs_sample_weights=method.needs_sample_weights,
            needs_labels=method.needs_labels,
            needs_group_assignment=method.needs_group_assignment,
            has_python_reference=method.python_reference is not None,
            has_r_reference=method.r_reference is not None,
            extra_reference_roles=list(extra_roles),
            reference_kind=_structural_reference_kind(method),
            canonical_reference_role=_canonical_reference_role(method),
            comparator=_comparator_for(method),
            tolerance_quality=_truth_source_quality(
                float(method.rmse_rel_tol)),
        ))
    out.sort(key=lambda m: m.name)
    return out


def build_references() -> list[ReferenceRecord]:
    """Union of declared reference role slots across the registry.

    The roles are host-insensitive: `python`, `r`, plus any extra slot
    names declared via `extra_references` (e.g. `ikpls`, `mixOmics`).
    Library / version metadata is host-sensitive and intentionally not
    captured here; the docs layer resolves it at render time.
    """
    by_role: dict[str, list[str]] = {}
    extra_roles: set[str] = set()
    for method in METHODS:
        for role, _factory in iter_reference_factories(method):
            by_role.setdefault(role, []).append(method.name)
            if role not in {"python", "r"}:
                extra_roles.add(role)
    out: list[ReferenceRecord] = []
    for role in sorted(by_role):
        out.append(ReferenceRecord(
            role=role,
            methods=sorted(set(by_role[role])),
            is_extra=role in extra_roles,
        ))
    return out


def build_datasets() -> list[DatasetRecord]:
    """Datasets the cross-binding orchestrator sweeps.

    The id format `{n}x{p}` matches the on-disk cache name produced by
    `gen_dataset_csv(n, p, seed)` in `cross_binding/orchestrator.py`,
    so anything consuming this file can locate the materialised CSV
    without re-deriving the convention.
    """
    out: list[DatasetRecord] = []
    for n, p in DEFAULT_SIZES:
        out.append(DatasetRecord(
            id=f"{n}x{p}",
            n=n,
            p=p,
            seed_base=DEFAULT_SEED_BASE,
        ))
    out.sort(key=lambda d: (d.n, d.p))
    return out


def build_comparators() -> list[ComparatorRecord]:
    """Declarative view of `benchmarks/parity_timing/_comparator.py`.

    Two comparators gate the pls4all parity surface:

    - `binding_parity` - element-wise max|pred-ref| <= 1e-6. Used to
      assert every pls4all binding (cpp, python_tier1, registry,
      r_tier1, matlab_tier1, ...) produces the same numbers as the C
      kernel.
    - `reference_parity` - relative RMSE <= MethodSpec.rmse_rel_tol.
      Used to assert pls4all's algorithm output matches an external
      "ground truth" implementation (sklearn / ropls / R::pls /
      ikpls / mixOmics / ...) within the declared tolerance band.
    """
    return [
        ComparatorRecord(
            id="binding_parity",
            name="binding-consistency gate",
            description=(
                "Element-wise max|pred-ref| <= tolerance. Asserts every "
                "pls4all binding produces bit-identical outputs to the "
                "C kernel."
            ),
            metric="max_abs_diff",
            default_tolerance=1e-6,
            tolerance_field="binding_tolerance",
        ),
        ComparatorRecord(
            id="reference_parity",
            name="external-reference validity gate",
            description=(
                "Relative-RMSE ||pred-ref||_RMS / ||ref||_RMS <= "
                "tolerance, with an absolute-RMSE <= 1e-9 escape near "
                "zero. Asserts pls4all's algorithm output matches an "
                "external ground-truth implementation within the "
                "method's declared tolerance band."
            ),
            metric="rmse_rel",
            default_tolerance=5e-2,
            tolerance_field="reference_tolerance",
        ),
    ]


def build_tolerances() -> list[ToleranceRecord]:
    """Per-method tolerance band + quality classification.

    `quality` derives from `MethodSpec.rmse_rel_tol` via the same band
    cutoffs the parity registry uses (`QUALITY_STRICT`,
    `QUALITY_RELAXED`, `QUALITY_QUALITATIVE`). The strict release gate
    further clamps executable references to rmse_rel <= 1e-3; that
    secondary gate is not encoded here - the snapshot records the
    MethodSpec's declared band.
    """
    out: list[ToleranceRecord] = []
    for method in METHODS:
        out.append(ToleranceRecord(
            method=method.name,
            binding_tolerance=1e-6,
            reference_tolerance=float(method.rmse_rel_tol),
            quality=_truth_source_quality(float(method.rmse_rel_tol)),
            comparator=_comparator_for(method),
        ))
    out.sort(key=lambda t: t.method)
    return out


def build_suites(
    methods: list[MethodRecord],
    datasets: list[DatasetRecord],
) -> list[SuiteRecord]:
    """Two suites for Phase PLS-1: smoke + benchmark.

    - `smoke` - every method against the lightest datasets. Designed
      to fail fast in CI / pre-commit; turnaround in tens of seconds.
    - `benchmark` - every method against every default dataset size.
      The publication / dashboard surface.
    """
    method_names = [m.name for m in methods]
    smoke_dataset_ids = [f"{n}x{p}" for n, p in SMOKE_SIZES]
    all_dataset_ids = [d.id for d in datasets]
    comparator_ids = ["binding_parity", "reference_parity"]
    return [
        SuiteRecord(
            id="smoke",
            description=(
                "Fastest cross-binding cells for every MethodSpec; "
                "used by pre-commit / CI smoke gates."
            ),
            methods=method_names,
            datasets=smoke_dataset_ids,
            comparators=comparator_ids,
        ),
        SuiteRecord(
            id="benchmark",
            description=(
                "Full cross-binding sweep - every MethodSpec across "
                "every default dataset size. Mirrors the cross-binding "
                "orchestrator's `DEFAULT_SIZES` surface."
            ),
            methods=method_names,
            datasets=all_dataset_ids,
            comparators=comparator_ids,
        ),
    ]


def build_manifest(counts: dict[str, int]) -> Manifest:
    return Manifest(
        schema_version=SCHEMA_VERSION,
        description=SCHEMA_DESCRIPTION,
        source="benchmarks/parity_timing/registry.py",
        files=[p.name for p in REGISTRY_FILES if p.name != "manifest.json"],
        counts=dict(counts),
    )


# ---------------------------------------------------------------------------
# Payload assembly + serialisation
# ---------------------------------------------------------------------------


def build_payload() -> dict[str, Any]:
    """Build the full registry payload, keyed by output filename."""
    methods = build_methods()
    references = build_references()
    datasets = build_datasets()
    comparators = build_comparators()
    tolerances = build_tolerances()
    suites = build_suites(methods, datasets)

    counts = {
        "methods": len(methods),
        "references": len(references),
        "datasets": len(datasets),
        "comparators": len(comparators),
        "tolerances": len(tolerances),
        "suites": len(suites),
    }
    manifest = build_manifest(counts)

    return {
        METHODS_PATH.name: {
            "schema_version": SCHEMA_VERSION,
            "methods": [m.to_dict() for m in methods],
        },
        REFERENCES_PATH.name: {
            "schema_version": SCHEMA_VERSION,
            "references": [r.to_dict() for r in references],
        },
        DATASETS_PATH.name: {
            "schema_version": SCHEMA_VERSION,
            "datasets": [d.to_dict() for d in datasets],
        },
        COMPARATORS_PATH.name: {
            "schema_version": SCHEMA_VERSION,
            "comparators": [c.to_dict() for c in comparators],
        },
        TOLERANCES_PATH.name: {
            "schema_version": SCHEMA_VERSION,
            "tolerances": [t.to_dict() for t in tolerances],
        },
        SUITES_PATH.name: {
            "schema_version": SCHEMA_VERSION,
            "suites": [s.to_dict() for s in suites],
        },
        MANIFEST_PATH.name: manifest.to_dict(),
    }


def dumps(value: Any) -> str:
    """Deterministic JSON encoding used by every write site.

    The trailing newline keeps the file POSIX-friendly and stops
    `git diff` from flagging "no newline at end of file" on every
    regeneration.
    """
    return json.dumps(
        value, indent=2, sort_keys=True, ensure_ascii=True) + "\n"


def write_payload(payload: dict[str, Any],
                   registry_dir: Path = REGISTRY_DIR) -> list[Path]:
    """Write every file in `payload` to `registry_dir` deterministically.

    Returns the list of files written, in writing order.
    """
    registry_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    for name, data in sorted(payload.items()):
        path = registry_dir / name
        path.write_text(dumps(data), encoding="utf-8")
        written.append(path)
    return written


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _cmd_write() -> int:
    payload = build_payload()
    written = write_payload(payload)
    for path in written:
        print(f"wrote {path}")
    return 0


def _cmd_print() -> int:
    payload = build_payload()
    manifest = payload[MANIFEST_PATH.name]
    counts = manifest.get("counts", {})
    print("validation registry payload (in-memory)")
    print(f"  schema_version: {manifest.get('schema_version')}")
    print(f"  source: {manifest.get('source')}")
    for key in sorted(counts):
        print(f"  {key}: {counts[key]}")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--write", action="store_true",
        help=(
            "Refresh the committed JSON snapshot under "
            f"{REGISTRY_DIR.relative_to(Path.cwd()) if Path.cwd() in REGISTRY_DIR.parents else REGISTRY_DIR}."
        ),
    )
    args = parser.parse_args(argv)
    if args.write:
        return _cmd_write()
    return _cmd_print()


if __name__ == "__main__":
    sys.exit(main())
