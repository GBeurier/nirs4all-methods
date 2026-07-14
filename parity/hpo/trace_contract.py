# SPDX-License-Identifier: CECILL-2.1
"""Independent semantic oracle for the versioned HPO StudyTrace contract.

This module is test-only and deliberately imports no n4m production package.
It validates lifecycle, structured-error, intermediate-value, pruning, and
logical-concurrency invariants independently of the ABI 2.2 rich trace.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from collections import Counter
from pathlib import Path
from typing import Any

_TERMINAL = {"COMPLETED", "PRUNED", "FAILED", "CANCELLED"}
_ERROR_STATUSES = {"FAILED", "CANCELLED"}
_CODE_RE = re.compile(r"^[A-Z][A-Z0-9_]{1,63}$")


class TraceContractError(ValueError):
    """Raised when a conformance tape violates StudyTrace v1 semantics."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise TraceContractError(message)


def _is_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _is_finite_number(value: Any) -> bool:
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        return False
    try:
        return math.isfinite(float(value))
    except (OverflowError, ValueError):
        return False


def _require_keys(value: Any, expected: set[str], path: str) -> dict[str, Any]:
    _require(isinstance(value, dict), f"{path}: expected object")
    actual = set(value)
    _require(actual == expected, f"{path}: keys {sorted(actual)} != {sorted(expected)}")
    return value


def _validate_error(value: Any, path: str) -> None:
    error = _require_keys(
        value,
        {"category", "code", "details", "message", "retryable"},
        path,
    )
    _require(
        isinstance(error["category"], str) and error["category"],
        f"{path}.category: expected non-empty string",
    )
    _require(
        isinstance(error["code"], str) and _CODE_RE.fullmatch(error["code"]),
        f"{path}.code: expected stable UPPER_SNAKE_CASE code",
    )
    _require(
        isinstance(error["message"], str) and error["message"],
        f"{path}.message: expected non-empty string",
    )
    _require(
        isinstance(error["retryable"], bool),
        f"{path}.retryable: expected boolean",
    )
    _require(isinstance(error["details"], dict), f"{path}.details: expected object")


def _validate_params(value: Any, path: str) -> None:
    _require(isinstance(value, dict), f"{path}: expected object")
    for name, item in value.items():
        _require(isinstance(name, str) and name, f"{path}: empty parameter name")
        _require(
            isinstance(item, (str, bool)) or _is_finite_number(item),
            f"{path}.{name}: expected finite scalar",
        )


def validate_study_trace(document: Any) -> None:
    """Validate one complete, terminal StudyTrace v1 document."""
    root = _require_keys(
        document,
        {"$schema", "events", "format", "schema_version", "study", "summary", "trials"},
        "$",
    )
    _require(root["format"] == "n4m.hpo.study_trace", "$.format: unsupported format")
    _require(root["schema_version"] == 1, "$.schema_version: unsupported version")
    _require(isinstance(root["$schema"], str), "$.$schema: expected string")

    study = _require_keys(
        root["study"],
        {"direction", "execution", "id", "pruner", "sampler", "seed"},
        "$.study",
    )
    for key in ("id", "pruner", "sampler"):
        _require(
            isinstance(study[key], str) and study[key],
            f"$.study.{key}: expected non-empty string",
        )
    _require(
        study["direction"] in {"minimize", "maximize"}, "$.study.direction: invalid"
    )
    _require(_is_int(study["seed"]) and study["seed"] >= 0, "$.study.seed: invalid")
    execution = _require_keys(
        study["execution"],
        {"max_in_flight", "mode", "tell_order"},
        "$.study.execution",
    )
    _require(execution["mode"] == "ordered_events", "$.study.execution.mode: invalid")
    _require(
        execution["tell_order"] == "event_sequence",
        "$.study.execution.tell_order: invalid",
    )
    max_in_flight = execution["max_in_flight"]
    _require(_is_int(max_in_flight) and max_in_flight >= 1, "max_in_flight: invalid")

    trials = root["trials"]
    _require(isinstance(trials, list) and trials, "$.trials: expected non-empty array")
    by_id: dict[int, dict[str, Any]] = {}
    intermediate_by_sequence: dict[int, tuple[int, dict[str, Any]]] = {}
    for index, raw_trial in enumerate(trials):
        path = f"$.trials[{index}]"
        _require(isinstance(raw_trial, dict), f"{path}: expected object")
        status = raw_trial.get("status")
        _require(status in _TERMINAL, f"{path}.status: invalid terminal status")
        expected = {"id", "intermediates", "params", "status"}
        if status == "COMPLETED":
            expected.add("score")
        if status == "PRUNED":
            expected.add("pruned_at")
        if status in _ERROR_STATUSES:
            expected.add("error")
        trial = _require_keys(raw_trial, expected, path)
        trial_id = trial["id"]
        _require(_is_int(trial_id) and trial_id >= 0, f"{path}.id: invalid")
        _require(trial_id not in by_id, f"{path}.id: duplicate {trial_id}")
        by_id[trial_id] = trial
        _validate_params(trial["params"], f"{path}.params")
        if status == "COMPLETED":
            _require(_is_finite_number(trial["score"]), f"{path}.score: non-finite")
        if status in _ERROR_STATUSES:
            _validate_error(trial["error"], f"{path}.error")

        intermediates = trial["intermediates"]
        _require(
            isinstance(intermediates, list), f"{path}.intermediates: expected array"
        )
        previous_step = -1
        for item_index, raw_item in enumerate(intermediates):
            item_path = f"{path}.intermediates[{item_index}]"
            item = _require_keys(
                raw_item,
                {"score", "sequence", "should_prune", "step"},
                item_path,
            )
            _require(
                _is_int(item["step"]) and item["step"] > previous_step,
                f"{item_path}.step: must increase strictly",
            )
            previous_step = item["step"]
            _require(_is_int(item["sequence"]), f"{item_path}.sequence: invalid")
            _require(
                item["sequence"] not in intermediate_by_sequence,
                f"{item_path}.sequence: duplicate",
            )
            _require(_is_finite_number(item["score"]), f"{item_path}.score: non-finite")
            _require(
                isinstance(item["should_prune"], bool),
                f"{item_path}.should_prune: expected boolean",
            )
            intermediate_by_sequence[item["sequence"]] = (trial_id, item)
        prune_events = [item for item in intermediates if item["should_prune"]]
        if status == "PRUNED":
            _require(len(prune_events) == 1, f"{path}: PRUNED needs one prune decision")
            _require(
                intermediates and prune_events[0] is intermediates[-1],
                f"{path}: prune decision must be final intermediate",
            )
            _require(
                trial["pruned_at"] == prune_events[0]["step"],
                f"{path}.pruned_at: inconsistent",
            )
        else:
            _require(not prune_events, f"{path}: non-PRUNED trial has prune decision")

    events = root["events"]
    _require(isinstance(events, list), "$.events: expected array")
    _require(
        [event.get("sequence") for event in events] == list(range(len(events))),
        "$.events: sequence must be contiguous from zero",
    )
    state = {trial_id: "NOT_ASKED" for trial_id in by_id}
    in_flight = 0
    observed_max = 0
    seen_intermediates: set[int] = set()
    for index, raw_event in enumerate(events):
        path = f"$.events[{index}]"
        _require(isinstance(raw_event, dict), f"{path}: expected object")
        kind = raw_event.get("kind")
        common = {"kind", "sequence", "trial_id"}
        if kind == "ASK":
            event = _require_keys(raw_event, common, path)
        elif kind == "INTERMEDIATE":
            event = _require_keys(
                raw_event,
                common | {"score", "should_prune", "step"},
                path,
            )
        elif kind == "TERMINAL":
            status = raw_event.get("status")
            expected = common | {"status"}
            if status in _ERROR_STATUSES:
                expected.add("error_code")
            event = _require_keys(raw_event, expected, path)
        else:
            raise TraceContractError(f"{path}.kind: invalid event kind")

        trial_id = event["trial_id"]
        _require(trial_id in by_id, f"{path}.trial_id: unknown {trial_id}")
        if kind == "ASK":
            _require(state[trial_id] == "NOT_ASKED", f"{path}: duplicate ASK")
            state[trial_id] = "RUNNING"
            in_flight += 1
            observed_max = max(observed_max, in_flight)
            _require(in_flight <= max_in_flight, f"{path}: max_in_flight exceeded")
        elif kind == "INTERMEDIATE":
            _require(state[trial_id] == "RUNNING", f"{path}: trial is not RUNNING")
            expected = intermediate_by_sequence.get(event["sequence"])
            _require(expected is not None, f"{path}: no matching trial intermediate")
            expected_trial_id, expected_item = expected
            _require(
                expected_trial_id == trial_id, f"{path}: intermediate trial mismatch"
            )
            for key in ("score", "should_prune", "step"):
                _require(event[key] == expected_item[key], f"{path}.{key}: mismatch")
            seen_intermediates.add(event["sequence"])
        else:
            _require(state[trial_id] == "RUNNING", f"{path}: trial is not RUNNING")
            status = event["status"]
            _require(status == by_id[trial_id]["status"], f"{path}.status: mismatch")
            if status in _ERROR_STATUSES:
                _require(
                    event["error_code"] == by_id[trial_id]["error"]["code"],
                    f"{path}.error_code: mismatch",
                )
            state[trial_id] = status
            in_flight -= 1

    _require(
        all(value in _TERMINAL for value in state.values()),
        "$.events: open trials remain",
    )
    _require(
        seen_intermediates == set(intermediate_by_sequence),
        "$.events: missing intermediate event",
    )

    summary = _require_keys(
        root["summary"],
        {"max_observed_in_flight", "n_intermediates", "n_trials", "status_counts"},
        "$.summary",
    )
    counts = Counter(trial["status"] for trial in trials)
    _require(summary["n_trials"] == len(trials), "$.summary.n_trials: mismatch")
    _require(
        summary["n_intermediates"] == len(intermediate_by_sequence),
        "$.summary.n_intermediates: mismatch",
    )
    _require(
        summary["max_observed_in_flight"] == observed_max,
        "$.summary.max_observed_in_flight: mismatch",
    )
    _require(
        summary["status_counts"] == dict(counts), "$.summary.status_counts: mismatch"
    )


def sha256_file(path: Path) -> str:
    """Return the lowercase SHA-256 digest of an artifact."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def validate_conformance_pack(path: Path) -> None:
    """Validate all checksums in a conformance-pack manifest."""
    pack = json.loads(path.read_text(encoding="utf-8"))
    root = _require_keys(pack, {"artifacts", "format", "schema_version"}, "$")
    _require(root["format"] == "n4m.hpo.conformance_pack", "$.format: invalid")
    _require(root["schema_version"] == 1, "$.schema_version: invalid")
    _require(
        isinstance(root["artifacts"], dict) and root["artifacts"], "$.artifacts: empty"
    )
    hpo_root = path.parent.parent
    for relative, expected in root["artifacts"].items():
        _require(isinstance(relative, str) and relative, "$.artifacts: invalid path")
        artifact = (hpo_root / relative).resolve()
        _require(
            hpo_root.resolve() in artifact.parents,
            f"$.artifacts: unsafe path {relative}",
        )
        _require(artifact.is_file(), f"$.artifacts: missing {relative}")
        _require(
            isinstance(expected, str) and re.fullmatch(r"[0-9a-f]{64}", expected),
            f"$.artifacts.{relative}: invalid digest",
        )
        _require(
            sha256_file(artifact) == expected,
            f"$.artifacts: stale digest for {relative}",
        )
