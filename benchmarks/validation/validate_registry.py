"""Validate the committed validation registry snapshot.

Phase PLS-1 gate. Three checks, in order:

1. Every MethodSpec must declare at least one reference factory OR be
   marked `paper_only`. Mirrors
   `benchmarks.parity_timing.registry.validate_methods_have_references`
   but reported through the validation framework's CLI.
2. Every JSON file under `benchmarks/validation/registry/` must exist
   and be parseable.
3. Re-running the in-memory export must produce a byte-identical copy
   of each committed file (deterministic JSON, sorted keys, indent=2,
   final newline).

Exit codes
----------

- 0  - snapshot is in sync with the registry.
- 1  - drift detected, or required fields missing. The CLI prints
       actionable instructions: which file changed, which method
       lacks a reference, how to regenerate.

Usage
-----

    python -m benchmarks.validation.validate_registry
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from benchmarks.parity_timing.registry import (
    validate_methods_have_references,
)

from .export_registry import (
    MANIFEST_PATH,
    METHODS_PATH,
    REGISTRY_DIR,
    REGISTRY_FILES,
    build_payload,
    dumps,
)

REGISTRY_FILENAMES: tuple[str, ...] = tuple(p.name for p in REGISTRY_FILES)

REQUIRED_METHOD_FIELDS: tuple[str, ...] = (
    "name", "description", "cell_params", "rmse_rel_tol",
    "prediction_key", "paper_only", "needs_x_target",
    "needs_sample_weights", "needs_labels", "needs_group_assignment",
    "has_python_reference", "has_r_reference", "extra_reference_roles",
    "reference_kind", "comparator", "tolerance_quality",
)

REQUIRED_TOLERANCE_FIELDS: tuple[str, ...] = (
    "method", "binding_tolerance", "reference_tolerance",
    "quality", "comparator",
)

REQUIRED_DATASET_FIELDS: tuple[str, ...] = (
    "id", "n", "p", "seed_base",
)


def _diff_summary(expected: str, committed: str, *, max_lines: int = 12,
                  context: int = 1) -> str:
    """Compact unified-diff style report for a JSON file mismatch.

    `difflib.unified_diff` works on lines; our JSON files are line-
    delimited by construction, so the line-level diff is human-readable
    even for nested structures. Truncated to `max_lines` so the
    validator stays usable as a pre-commit gate.
    """
    import difflib

    expected_lines = expected.splitlines(keepends=True)
    committed_lines = committed.splitlines(keepends=True)
    diff_lines = list(difflib.unified_diff(
        committed_lines, expected_lines,
        fromfile="committed", tofile="expected",
        n=context,
    ))
    if not diff_lines:
        return "(no textual diff - likely a trailing newline difference)"
    if len(diff_lines) > max_lines:
        head = "".join(diff_lines[:max_lines])
        return head + f"  ... (+{len(diff_lines) - max_lines} more lines)\n"
    return "".join(diff_lines)


def _check_reference_completeness() -> list[str]:
    """Surface MethodSpecs missing a reference / paper_only citation."""
    bad = validate_methods_have_references()
    if not bad:
        return []
    return [
        "MethodSpec(s) without any reference factory or `paper_only` "
        "citation: " + ", ".join(sorted(bad)) + ".",
        (
            "Add a `python_reference=` / `r_reference=` factory, an "
            "entry in `extra_references`, or set `paper_only` to a "
            "citation string."
        ),
    ]


def _check_required_fields(committed: dict[str, Any]) -> list[str]:
    """Schema check on the committed payload - invariant per file kind.

    Runs against the on-disk JSON because that is where missing fields
    could realistically leak in (the in-memory builder always produces
    schema-clean records, by construction). The committed dict mirrors
    `build_payload()`'s shape: filename -> loaded JSON document.
    """
    errors: list[str] = []
    methods = committed.get(METHODS_PATH.name, {}).get("methods", [])
    for record in methods:
        missing = [f for f in REQUIRED_METHOD_FIELDS if f not in record]
        if missing:
            errors.append(
                f"methods.json: method '{record.get('name', '?')}' is "
                f"missing required field(s): {', '.join(missing)}"
            )

    tolerances = committed.get(
        "tolerances.json", {}).get("tolerances", [])
    for record in tolerances:
        missing = [f for f in REQUIRED_TOLERANCE_FIELDS if f not in record]
        if missing:
            errors.append(
                f"tolerances.json: row '{record.get('method', '?')}' is "
                f"missing required field(s): {', '.join(missing)}"
            )

    datasets = committed.get("datasets.json", {}).get("datasets", [])
    for record in datasets:
        missing = [f for f in REQUIRED_DATASET_FIELDS if f not in record]
        if missing:
            errors.append(
                f"datasets.json: row '{record.get('id', '?')}' is "
                f"missing required field(s): {', '.join(missing)}"
            )
    return errors


def _check_manifest_counts(payload: dict[str, Any]) -> list[str]:
    """The manifest counts must match the lengths of every sibling file."""
    manifest = payload.get(MANIFEST_PATH.name)
    if not isinstance(manifest, dict):
        return []
    counts = manifest.get("counts")
    if not isinstance(counts, dict):
        return ["manifest.json: missing required `counts` object"]
    sibling_lengths = {
        "methods": len(payload.get("methods.json", {}).get("methods", [])),
        "references": len(payload.get("references.json", {}).get(
            "references", [])),
        "datasets": len(payload.get("datasets.json", {}).get(
            "datasets", [])),
        "comparators": len(payload.get("comparators.json", {}).get(
            "comparators", [])),
        "tolerances": len(payload.get("tolerances.json", {}).get(
            "tolerances", [])),
        "suites": len(payload.get("suites.json", {}).get("suites", [])),
    }
    errors: list[str] = []
    for key, expected in sibling_lengths.items():
        got = counts.get(key)
        if got != expected:
            errors.append(
                f"manifest.json: counts[{key!r}] = {got!r}, expected "
                f"{expected!r} (length of {key}.json)"
            )
    return errors


def _display_path(path: Path, registry_dir: Path) -> str:
    """Render a registry file path in error messages relative to the
    package root when possible, so the operator sees `registry/foo.json`
    instead of an absolute path that may differ between hosts."""
    try:
        return str(path.relative_to(registry_dir.parent))
    except ValueError:
        return str(path)


def _check_committed_against_expected(
    payload: dict[str, Any],
    registry_dir: Path = REGISTRY_DIR,
) -> list[str]:
    """Compare each committed file to the expected serialised form."""
    errors: list[str] = []
    for name in REGISTRY_FILENAMES:
        path = registry_dir / name
        if not path.exists():
            errors.append(
                f"{_display_path(path, registry_dir)} is missing - "
                "run `python -m benchmarks.validation.export_registry "
                "--write`."
            )
            continue
        expected = dumps(payload[name])
        committed = path.read_text(encoding="utf-8")
        if committed != expected:
            errors.append(
                f"{_display_path(path, registry_dir)} is out of date - "
                "registry changed. Run `python -m "
                "benchmarks.validation.export_registry --write` and "
                "commit. Diff (expected vs committed):\n"
                + _diff_summary(expected, committed)
            )
    return errors


def validate(registry_dir: Path = REGISTRY_DIR) -> list[str]:
    """Run every gate and collect actionable error messages.

    An empty list means the snapshot is in sync. Each entry is a
    self-contained, human-readable message - printed verbatim by the
    CLI and consumed as-is by the focused pytest test.
    """
    errors: list[str] = []
    errors.extend(_check_reference_completeness())

    # Parse every committed file once. JSON errors surface with a
    # precise path instead of being swallowed by the textual diff
    # gate further down.
    committed: dict[str, Any] = {}
    for name in REGISTRY_FILENAMES:
        path = registry_dir / name
        if not path.exists():
            continue
        try:
            committed[name] = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            errors.append(
                f"{_display_path(path, registry_dir)} is not valid "
                f"JSON: {exc}"
            )

    errors.extend(_check_required_fields(committed))

    payload = build_payload()
    errors.extend(_check_manifest_counts(committed))
    errors.extend(_check_committed_against_expected(payload, registry_dir))
    return errors


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--registry-dir", default=str(REGISTRY_DIR),
        help=f"Directory containing the snapshot (default: {REGISTRY_DIR}).",
    )
    args = parser.parse_args(argv)
    registry_dir = Path(args.registry_dir)
    errors = validate(registry_dir)
    if errors:
        print(
            f"validation registry snapshot FAILED ({len(errors)} "
            "issue(s)):",
            file=sys.stderr,
        )
        for message in errors:
            print(f"  - {message}", file=sys.stderr)
        return 1
    print(
        "validation registry snapshot OK - "
        f"{len(list(registry_dir.glob('*.json')))} file(s) in sync."
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
