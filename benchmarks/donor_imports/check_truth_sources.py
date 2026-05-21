#!/usr/bin/env python3
"""Validate the benchmark registry against the truth-source lockfile."""

from __future__ import annotations

import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
REGISTRY = ROOT / "benchmarks" / "benchmark_registry.json"
LOCKFILE = ROOT / "benchmarks" / "truth_sources.lock.json"

REQUIRED_METHOD_FIELDS = {
    "method_id",
    "label",
    "family",
    "operation",
    "truth_sources",
    "metrics",
    "default_dataset",
}
REQUIRED_DATASET_FIELDS = {"n_samples", "n_features", "seed"}


def load_json(path: Path) -> dict:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        raise SystemExit(f"missing required file: {path}") from None
    except json.JSONDecodeError as exc:
        raise SystemExit(f"invalid JSON in {path}: {exc}") from None


def error(errors: list[str], message: str) -> None:
    errors.append(message)


def validate() -> list[str]:
    registry = load_json(REGISTRY)
    lockfile = load_json(LOCKFILE)
    errors: list[str] = []

    methods = registry.get("methods")
    sources = lockfile.get("sources")
    if not isinstance(methods, list) or not methods:
        error(errors, "registry must contain a non-empty methods list")
        methods = []
    if not isinstance(sources, dict) or not sources:
        error(errors, "lockfile must contain a non-empty sources object")
        sources = {}

    seen_methods: set[str] = set()
    referenced_sources: set[str] = set()
    for idx, method in enumerate(methods):
        if not isinstance(method, dict):
            error(errors, f"methods[{idx}] must be an object")
            continue
        missing = sorted(REQUIRED_METHOD_FIELDS - set(method))
        method_id = method.get("method_id", f"methods[{idx}]")
        if missing:
            error(errors, f"{method_id}: missing fields: {', '.join(missing)}")
        if method_id in seen_methods:
            error(errors, f"{method_id}: duplicate method_id")
        seen_methods.add(method_id)

        truth_sources = method.get("truth_sources")
        if not isinstance(truth_sources, list) or not truth_sources:
            error(errors, f"{method_id}: truth_sources must be a non-empty list")
        else:
            for source_id in truth_sources:
                if source_id not in sources:
                    error(errors, f"{method_id}: unknown truth source {source_id!r}")
                else:
                    referenced_sources.add(source_id)

        metrics = method.get("metrics")
        if not isinstance(metrics, list) or not metrics:
            error(errors, f"{method_id}: metrics must be a non-empty list")

        dataset = method.get("default_dataset")
        if not isinstance(dataset, dict):
            error(errors, f"{method_id}: default_dataset must be an object")
        else:
            missing_dataset = sorted(REQUIRED_DATASET_FIELDS - set(dataset))
            if missing_dataset:
                error(errors, f"{method_id}: default_dataset missing fields: {', '.join(missing_dataset)}")

    for source_id, source in sources.items():
        if not isinstance(source, dict):
            error(errors, f"{source_id}: lockfile source must be an object")
            continue
        source_type = source.get("type")
        if not source_type:
            error(errors, f"{source_id}: missing type")
        if not source.get("package") and not source.get("project"):
            error(errors, f"{source_id}: external source must name package or project")

    unused = sorted(set(sources) - referenced_sources)
    if unused:
        error(errors, "unused locked sources: " + ", ".join(unused))

    return errors


def main() -> int:
    errors = validate()
    if errors:
        for message in errors:
            print(f"ERROR: {message}", file=sys.stderr)
        return 1
    print(f"OK: {REGISTRY.relative_to(ROOT)} matches {LOCKFILE.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
