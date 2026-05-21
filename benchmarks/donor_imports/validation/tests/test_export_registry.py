"""Focused tests for the deterministic validation registry."""

from __future__ import annotations

import json
import shutil
from pathlib import Path

from benchmarks.validation.export_registry import (
    DATASETS_PATH,
    METHODS_PATH,
    REGISTRY_DIR,
    REGISTRY_FILES,
    build_payload,
    dumps,
)
from benchmarks.validation.validate_registry import (
    REQUIRED_DATASET_FIELDS,
    REQUIRED_METHOD_FIELDS,
    REQUIRED_REFERENCE_FIELDS,
    REQUIRED_TOLERANCE_FIELDS,
    validate,
)


def test_build_payload_is_idempotent() -> None:
    first = build_payload()
    second = build_payload()
    assert set(first) == set(second)
    for name in first:
        assert dumps(first[name]) == dumps(second[name]), (
            f"non-deterministic export for {name}"
        )


def test_committed_snapshot_matches_in_memory_payload() -> None:
    payload = build_payload()
    for path in REGISTRY_FILES:
        assert path.exists(), (
            f"{path} missing - run `python -m "
            "benchmarks.validation.export_registry --write`"
        )
        assert path.read_text(encoding="utf-8") == dumps(payload[path.name])


def test_validator_passes_on_committed_snapshot() -> None:
    assert validate() == []


def test_validator_detects_diff_drift(tmp_path: Path) -> None:
    shadow = tmp_path / "registry"
    shutil.copytree(REGISTRY_DIR, shadow)
    methods_file = shadow / "methods.json"
    payload = json.loads(methods_file.read_text(encoding="utf-8"))
    payload["methods"][0]["label"] = "drifted"
    methods_file.write_text(dumps(payload), encoding="utf-8")
    errors = validate(registry_dir=shadow)
    assert errors
    assert any("methods.json" in message for message in errors)


def test_validator_detects_missing_file(tmp_path: Path) -> None:
    shadow = tmp_path / "registry"
    shutil.copytree(REGISTRY_DIR, shadow)
    (shadow / "manifest.json").unlink()
    errors = validate(registry_dir=shadow)
    assert errors
    assert any("manifest.json" in message for message in errors)


def test_every_record_has_required_fields() -> None:
    payload = build_payload()
    for record in payload[METHODS_PATH.name]["methods"]:
        missing = [field for field in REQUIRED_METHOD_FIELDS if field not in record]
        assert not missing, (
            f"method '{record.get('method_id', '?')}' missing fields: {missing}"
        )
        for ref in record["references"]:
            ref_missing = [
                field for field in REQUIRED_REFERENCE_FIELDS if field not in ref
            ]
            assert not ref_missing, (
                f"reference '{ref.get('backend', '?')}' missing fields: "
                f"{ref_missing}"
            )
    for record in payload["tolerances.json"]["tolerances"]:
        missing = [
            field for field in REQUIRED_TOLERANCE_FIELDS if field not in record
        ]
        assert not missing, (
            f"tolerance '{record.get('method', '?')}' missing fields: {missing}"
        )
    for record in payload[DATASETS_PATH.name]["datasets"]:
        missing = [field for field in REQUIRED_DATASET_FIELDS if field not in record]
        assert not missing, (
            f"dataset '{record.get('id', '?')}' missing fields: {missing}"
        )


def test_manifest_counts_match_sibling_files() -> None:
    payload = build_payload()
    counts = payload["manifest.json"]["counts"]
    assert counts["methods"] == len(payload["methods.json"]["methods"])
    assert counts["references"] == len(payload["references.json"]["references"])
    assert counts["datasets"] == len(payload["datasets.json"]["datasets"])
    assert counts["comparators"] == len(payload["comparators.json"]["comparators"])
    assert counts["tolerances"] == len(payload["tolerances.json"]["tolerances"])
    assert counts["suites"] == len(payload["suites.json"]["suites"])


def test_benchmark_registry_subset_is_covered() -> None:
    root = Path(__file__).resolve().parents[3]
    registry = json.loads(
        (root / "benchmarks" / "benchmark_registry.json").read_text(
            encoding="utf-8"
        )
    )
    expected = {record["method_id"] for record in registry["methods"]}
    payload = build_payload()
    got = {record["method_id"] for record in payload["methods.json"]["methods"]}
    assert expected <= got


def test_dumps_sorts_keys_and_keeps_trailing_newline() -> None:
    text = dumps({"b": 1, "a": 2})
    assert text.endswith("\n")
    assert text.index('"a"') < text.index('"b"')
