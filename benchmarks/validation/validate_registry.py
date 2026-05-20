"""Validate the committed chemometrics4all validation snapshot."""

from __future__ import annotations

import argparse
import difflib
import json
import sys
from pathlib import Path
from typing import Any

from .export_registry import (
    MANIFEST_PATH,
    REGISTRY_DIR,
    REGISTRY_FILES,
    build_payload,
    dumps,
)

REGISTRY_FILENAMES: tuple[str, ...] = tuple(path.name for path in REGISTRY_FILES)

REQUIRED_METHOD_FIELDS: tuple[str, ...] = (
    "method_id",
    "label",
    "family",
    "py_class",
    "c_prefix",
    "r_expr",
    "max_cols",
    "compare_internal",
    "comparator",
    "tolerance_quality",
    "operation",
    "truth_sources",
    "metrics",
    "default_dataset",
    "references",
)

REQUIRED_REFERENCE_FIELDS: tuple[str, ...] = (
    "backend",
    "library",
    "language",
    "compare",
    "gate_c4a",
    "contract_note",
    "max_cols",
    "comparator",
    "r_expr",
)

REQUIRED_TOLERANCE_FIELDS: tuple[str, ...] = (
    "method",
    "comparator",
    "binding_tolerance",
    "reference_rtol",
    "reference_atol",
    "quality",
)

REQUIRED_DATASET_FIELDS: tuple[str, ...] = (
    "id",
    "n",
    "p",
    "seed_base",
    "in_smoke_preset",
)


def _diff_summary(
    expected: str,
    committed: str,
    *,
    max_lines: int = 12,
    context: int = 1,
) -> str:
    diff_lines = list(difflib.unified_diff(
        committed.splitlines(keepends=True),
        expected.splitlines(keepends=True),
        fromfile="committed",
        tofile="expected",
        n=context,
    ))
    if not diff_lines:
        return "(no textual diff - likely a trailing newline difference)"
    if len(diff_lines) > max_lines:
        return "".join(diff_lines[:max_lines]) + (
            f"  ... (+{len(diff_lines) - max_lines} more lines)\n"
        )
    return "".join(diff_lines)


def _display_path(path: Path, registry_dir: Path) -> str:
    try:
        return str(path.relative_to(registry_dir.parent))
    except ValueError:
        return str(path)


def _check_required_fields(committed: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    methods = committed.get("methods.json", {}).get("methods", [])
    for record in methods:
        missing = [field for field in REQUIRED_METHOD_FIELDS if field not in record]
        method_id = record.get("method_id", "?")
        if missing:
            errors.append(
                f"methods.json: method '{method_id}' is missing required "
                f"field(s): {', '.join(missing)}"
            )
        for ref in record.get("references", []):
            ref_missing = [
                field for field in REQUIRED_REFERENCE_FIELDS if field not in ref
            ]
            if ref_missing:
                errors.append(
                    f"methods.json: method '{method_id}' reference "
                    f"'{ref.get('backend', '?')}' is missing required "
                    f"field(s): {', '.join(ref_missing)}"
                )

    for record in committed.get("tolerances.json", {}).get("tolerances", []):
        missing = [
            field for field in REQUIRED_TOLERANCE_FIELDS if field not in record
        ]
        if missing:
            errors.append(
                f"tolerances.json: row '{record.get('method', '?')}' is "
                f"missing required field(s): {', '.join(missing)}"
            )

    for record in committed.get("datasets.json", {}).get("datasets", []):
        missing = [field for field in REQUIRED_DATASET_FIELDS if field not in record]
        if missing:
            errors.append(
                f"datasets.json: row '{record.get('id', '?')}' is missing "
                f"required field(s): {', '.join(missing)}"
            )
    return errors


def _check_manifest_counts(payload: dict[str, Any]) -> list[str]:
    manifest = payload.get(MANIFEST_PATH.name)
    if not isinstance(manifest, dict):
        return [f"{MANIFEST_PATH.name}: missing manifest object"]
    counts = manifest.get("counts")
    if not isinstance(counts, dict):
        return [f"{MANIFEST_PATH.name}: missing required `counts` object"]
    sibling_lengths = {
        "methods": len(payload.get("methods.json", {}).get("methods", [])),
        "references": len(payload.get("references.json", {}).get("references", [])),
        "datasets": len(payload.get("datasets.json", {}).get("datasets", [])),
        "comparators": len(payload.get("comparators.json", {}).get("comparators", [])),
        "tolerances": len(payload.get("tolerances.json", {}).get("tolerances", [])),
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


def _check_committed_against_expected(
    payload: dict[str, Any],
    registry_dir: Path,
) -> list[str]:
    errors: list[str] = []
    for name in REGISTRY_FILENAMES:
        path = registry_dir / name
        if not path.exists():
            errors.append(
                f"{_display_path(path, registry_dir)} is missing - run "
                "`python -m benchmarks.validation.export_registry --write`."
            )
            continue
        expected = dumps(payload[name])
        committed = path.read_text(encoding="utf-8")
        if committed != expected:
            errors.append(
                f"{_display_path(path, registry_dir)} is out of date - run "
                "`python -m benchmarks.validation.export_registry --write`. "
                "Diff (expected vs committed):\n"
                + _diff_summary(expected, committed)
            )
    return errors


def _load_json(path: Path) -> tuple[dict[str, Any], str | None]:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return {}, f"missing required file: {path}"
    except json.JSONDecodeError as exc:
        return {}, f"invalid JSON in {path}: {exc}"
    return data if isinstance(data, dict) else {}, None


def _check_benchmark_registry() -> list[str]:
    root = REGISTRY_DIR.parents[2]
    registry_path = root / "benchmarks" / "benchmark_registry.json"
    lock_path = root / "benchmarks" / "truth_sources.lock.json"
    registry, registry_error = _load_json(registry_path)
    lockfile, lock_error = _load_json(lock_path)
    errors = [msg for msg in (registry_error, lock_error) if msg]
    if errors:
        return errors

    methods = registry.get("methods")
    sources = lockfile.get("sources")
    if not isinstance(methods, list) or not methods:
        errors.append("benchmark_registry.json: methods must be a non-empty list")
        methods = []
    if not isinstance(sources, dict) or not sources:
        errors.append("truth_sources.lock.json: sources must be a non-empty object")
        sources = {}

    seen_methods: set[str] = set()
    referenced_sources: set[str] = set()
    for idx, method in enumerate(methods):
        if not isinstance(method, dict):
            errors.append(f"benchmark_registry.json: methods[{idx}] must be an object")
            continue
        method_id = method.get("method_id", f"methods[{idx}]")
        for field in (
            "method_id",
            "label",
            "family",
            "operation",
            "truth_sources",
            "metrics",
            "default_dataset",
        ):
            if field not in method:
                errors.append(f"benchmark_registry.json: {method_id}: missing {field}")
        if method_id in seen_methods:
            errors.append(f"benchmark_registry.json: duplicate method_id {method_id}")
        seen_methods.add(str(method_id))
        truth_sources = method.get("truth_sources")
        if not isinstance(truth_sources, list) or not truth_sources:
            errors.append(
                f"benchmark_registry.json: {method_id}: truth_sources must "
                "be a non-empty list"
            )
        else:
            for source_id in truth_sources:
                if source_id not in sources:
                    errors.append(
                        f"benchmark_registry.json: {method_id}: unknown "
                        f"truth source {source_id!r}"
                    )
                else:
                    referenced_sources.add(str(source_id))
        dataset = method.get("default_dataset")
        if not isinstance(dataset, dict):
            errors.append(
                f"benchmark_registry.json: {method_id}: default_dataset must "
                "be an object"
            )
        else:
            for field in ("n_samples", "n_features", "seed"):
                if field not in dataset:
                    errors.append(
                        f"benchmark_registry.json: {method_id}: "
                        f"default_dataset missing {field}"
                    )

    for source_id, source in sources.items():
        if not isinstance(source, dict):
            errors.append(
                f"truth_sources.lock.json: {source_id}: source must be an object"
            )
            continue
        if not source.get("type"):
            errors.append(f"truth_sources.lock.json: {source_id}: missing type")
        if not source.get("package") and not source.get("project"):
            errors.append(
                f"truth_sources.lock.json: {source_id}: source must name "
                "package or project"
            )

    unused = sorted(set(sources) - referenced_sources)
    if unused:
        errors.append("truth_sources.lock.json: unused locked sources: " + ", ".join(unused))
    return errors


def validate(registry_dir: Path = REGISTRY_DIR) -> list[str]:
    errors: list[str] = []
    committed: dict[str, Any] = {}
    for name in REGISTRY_FILENAMES:
        path = registry_dir / name
        if not path.exists():
            continue
        try:
            committed[name] = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            errors.append(f"{_display_path(path, registry_dir)} is invalid JSON: {exc}")

    errors.extend(_check_required_fields(committed))
    errors.extend(_check_manifest_counts(committed))
    errors.extend(_check_benchmark_registry())
    errors.extend(_check_committed_against_expected(build_payload(), registry_dir))
    return errors


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--registry-dir",
        default=str(REGISTRY_DIR),
        help=f"Directory containing the snapshot (default: {REGISTRY_DIR}).",
    )
    args = parser.parse_args(argv)
    errors = validate(Path(args.registry_dir))
    if errors:
        for message in errors:
            print(f"ERROR: {message}", file=sys.stderr)
        return 1
    print(f"OK: validation registry is in sync ({args.registry_dir})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
