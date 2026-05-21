"""Phase PLS-1 gate - deterministic export + drift detection.

Pure pytest, stdlib-only assertions. Verifies:

1. The in-memory export is byte-stable: calling `build_payload()` twice
   produces identical serialised JSON. No timestamps, no host-dependent
   ordering.
2. The committed snapshot under `benchmarks/validation/registry/`
   matches the in-memory payload byte-for-byte (the same gate
   `validate_registry.py` enforces from the CLI).
3. The validator returns a non-empty list of actionable errors when
   any one file drifts - both for missing-field drift and for diff
   drift.
4. Every required field defined in `validate_registry.py` is present
   in every record of the relevant file.
5. The mirror of `cross_binding/orchestrator.DEFAULT_SIZES` /
   `DEFAULT_SEED_BASE` agrees with the orchestrator at runtime
   (skipped if the orchestrator is not importable on this host -
   numpy is the only extra dep but the test stays defensive).
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest

from benchmarks.validation import export_registry
from benchmarks.validation.export_registry import (
    DEFAULT_SEED_BASE,
    DEFAULT_SIZES,
    METHODS_PATH,
    REGISTRY_DIR,
    REGISTRY_FILES,
    build_payload,
    dumps,
)
from benchmarks.validation.validate_registry import (
    REQUIRED_DATASET_FIELDS,
    REQUIRED_METHOD_FIELDS,
    REQUIRED_TOLERANCE_FIELDS,
    validate,
)


def test_build_payload_is_idempotent() -> None:
    """Two consecutive in-memory builds must produce identical JSON."""
    first = build_payload()
    second = build_payload()
    for name in first:
        assert dumps(first[name]) == dumps(second[name]), (
            f"non-deterministic export for {name}"
        )


def test_committed_snapshot_matches_in_memory_payload() -> None:
    """Every committed file must match the freshly built payload."""
    payload = build_payload()
    for path in REGISTRY_FILES:
        assert path.exists(), (
            f"{path} missing - run `python -m "
            "benchmarks.validation.export_registry --write`"
        )
        committed = path.read_text(encoding="utf-8")
        expected = dumps(payload[path.name])
        assert committed == expected, (
            f"{path.name} drifted from registry - re-export to refresh"
        )


def test_validator_passes_on_committed_snapshot() -> None:
    """The validator must accept the committed snapshot as-is."""
    errors = validate()
    assert errors == [], "validator rejected the committed snapshot: " + (
        " | ".join(errors)
    )


def test_validator_detects_diff_drift(tmp_path: Path) -> None:
    """Mutating a committed file under a temp copy must trigger drift."""
    shadow = tmp_path / "registry"
    shutil.copytree(REGISTRY_DIR, shadow)

    methods_file = shadow / "methods.json"
    payload = json.loads(methods_file.read_text(encoding="utf-8"))
    assert payload["methods"], "expected non-empty methods.json"
    # Mutate the first method's tolerance to force a textual diff.
    payload["methods"][0]["rmse_rel_tol"] = 9999.0
    methods_file.write_text(
        dumps(payload),
        encoding="utf-8",
    )

    errors = validate(registry_dir=shadow)
    assert errors, "validator failed to detect tolerance drift"
    assert any("methods.json" in msg for msg in errors), (
        "validator did not name the drifted file: " + " | ".join(errors)
    )


def test_validator_detects_missing_file(tmp_path: Path) -> None:
    """Removing a committed file must trigger an actionable error."""
    shadow = tmp_path / "registry"
    shutil.copytree(REGISTRY_DIR, shadow)

    (shadow / "manifest.json").unlink()
    errors = validate(registry_dir=shadow)
    assert errors, "validator failed to detect missing manifest.json"
    assert any("manifest.json" in msg for msg in errors), (
        "validator did not name the missing file: " + " | ".join(errors)
    )


def test_validator_detects_required_field_drift(tmp_path: Path) -> None:
    """Removing a required field from a record must trigger drift."""
    shadow = tmp_path / "registry"
    shutil.copytree(REGISTRY_DIR, shadow)

    methods_file = shadow / "methods.json"
    payload = json.loads(methods_file.read_text(encoding="utf-8"))
    # `comparator` is part of REQUIRED_METHOD_FIELDS.
    del payload["methods"][0]["comparator"]
    methods_file.write_text(
        dumps(payload),
        encoding="utf-8",
    )

    errors = validate(registry_dir=shadow)
    assert errors, "validator failed to detect missing field"
    assert any("missing required field" in msg for msg in errors), (
        "validator did not surface the missing field: "
        + " | ".join(errors)
    )


def test_every_method_has_required_fields() -> None:
    """Defense in depth - the in-memory payload should always be
    schema-clean even if the validator's required-field check
    regresses."""
    payload = build_payload()
    methods = payload[METHODS_PATH.name]["methods"]
    for record in methods:
        missing = [f for f in REQUIRED_METHOD_FIELDS if f not in record]
        assert not missing, (
            f"method '{record.get('name', '?')}' missing fields: "
            f"{missing}"
        )

    for record in payload["tolerances.json"]["tolerances"]:
        missing = [f for f in REQUIRED_TOLERANCE_FIELDS if f not in record]
        assert not missing, (
            f"tolerance '{record.get('method', '?')}' missing fields: "
            f"{missing}"
        )

    for record in payload["datasets.json"]["datasets"]:
        missing = [f for f in REQUIRED_DATASET_FIELDS if f not in record]
        assert not missing, (
            f"dataset '{record.get('id', '?')}' missing fields: "
            f"{missing}"
        )


def test_manifest_counts_match_sibling_files() -> None:
    """The manifest is the single human-eyeballable counts surface;
    if its counts go stale, every downstream consumer is wrong."""
    payload = build_payload()
    manifest = payload["manifest.json"]
    counts = manifest["counts"]

    assert counts["methods"] == len(payload[METHODS_PATH.name]["methods"])
    assert counts["references"] == len(
        payload["references.json"]["references"])
    assert counts["datasets"] == len(payload["datasets.json"]["datasets"])
    assert counts["comparators"] == len(
        payload["comparators.json"]["comparators"])
    assert counts["tolerances"] == len(
        payload["tolerances.json"]["tolerances"])
    assert counts["suites"] == len(payload["suites.json"]["suites"])


def test_dumps_emits_sorted_keys_and_trailing_newline() -> None:
    """The deterministic dumps contract must not regress.

    Sorted keys + trailing newline keep diffs minimal and prevent
    `git diff` from flagging "no newline at end of file" every time
    the snapshot is regenerated.
    """
    sample = {"b": 1, "a": 2}
    text = dumps(sample)
    assert text.endswith("\n"), "dumps() must emit a trailing newline"
    # `"a"` precedes `"b"` when keys are sorted.
    assert text.index('"a"') < text.index('"b"'), (
        "dumps() must sort keys"
    )


def test_mirror_dataset_constants_match_orchestrator() -> None:
    """The validation module mirrors `DEFAULT_SIZES` and
    `DEFAULT_SEED_BASE` from `cross_binding/orchestrator.py`. The
    mirror must not silently drift from the orchestrator.

    Skipped if the orchestrator is not importable (e.g. minimal CI
    without numpy) - the snapshot is still valid in isolation; this
    test just catches drift on the dev host.
    """
    orchestrator = pytest.importorskip(
        "benchmarks.cross_binding.orchestrator")
    expected_sizes = tuple(orchestrator.DEFAULT_SIZES)
    assert DEFAULT_SIZES == expected_sizes, (
        "validation.export_registry.DEFAULT_SIZES is out of sync with "
        "cross_binding.orchestrator.DEFAULT_SIZES"
    )
    assert DEFAULT_SEED_BASE == orchestrator.DEFAULT_SEED_BASE, (
        "validation.export_registry.DEFAULT_SEED_BASE is out of sync "
        "with cross_binding.orchestrator.DEFAULT_SEED_BASE"
    )


def test_method_count_matches_parity_lockfile() -> None:
    """`methods.json` and `truth_sources.lock.json` derive from the
    same `METHODS` list. They must therefore declare the same set of
    method names."""
    lockfile_path = (
        Path(export_registry.__file__).resolve().parents[1]
        / "parity_timing"
        / "truth_sources.lock.json"
    )
    if not lockfile_path.exists():
        pytest.skip("parity_timing lockfile not present")
    lockfile = json.loads(lockfile_path.read_text(encoding="utf-8"))
    expected = {entry["name"] for entry in lockfile["methods"]}
    payload = build_payload()
    got = {m["name"] for m in payload[METHODS_PATH.name]["methods"]}
    assert got == expected, (
        "methods.json and truth_sources.lock.json disagree on the "
        "MethodSpec set - run both `python -m "
        "benchmarks.parity_timing.lockfile --write` and `python -m "
        "benchmarks.validation.export_registry --write` after editing "
        "the registry."
    )
