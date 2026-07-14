# SPDX-License-Identifier: CECILL-2.1
"""Independent gates for lifecycle/concurrency HPO conformance tapes."""

from __future__ import annotations

import copy
import json
import shutil
import subprocess
import sys
from pathlib import Path

import pytest
from jsonschema import Draft202012Validator, ValidationError

_HPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_HPO))

from trace_contract import (  # noqa: E402
    TraceContractError,
    validate_conformance_pack,
    validate_study_trace,
)
from comparators import compare_traces  # noqa: E402


@pytest.fixture
def lifecycle_trace() -> dict:
    return json.loads((_HPO / "golden" / "lifecycle_states.v1.json").read_text())


def test_lifecycle_trace_covers_every_terminal_state(lifecycle_trace):
    validate_study_trace(lifecycle_trace)
    assert {trial["status"] for trial in lifecycle_trace["trials"]} == {
        "COMPLETED",
        "PRUNED",
        "FAILED",
        "CANCELLED",
    }
    assert lifecycle_trace["summary"]["max_observed_in_flight"] == 3


@pytest.mark.parametrize(
    "mutation, expected",
    [
        (lambda doc: doc["trials"][1].pop("error"), "keys"),
        (lambda doc: doc["trials"][3].__setitem__("score", 0.0), "keys"),
        (
            lambda doc: doc["events"][12].__setitem__("sequence", 99),
            "sequence",
        ),
        (
            lambda doc: doc["events"][10].__setitem__("trial_id", 2),
            "not RUNNING",
        ),
        (
            lambda doc: doc["summary"].__setitem__("max_observed_in_flight", 2),
            "max_observed_in_flight",
        ),
    ],
)
def test_lifecycle_trace_rejects_semantic_mutations(
    lifecycle_trace, mutation, expected
):
    mutated = copy.deepcopy(lifecycle_trace)
    mutation(mutated)
    with pytest.raises(TraceContractError, match=expected):
        validate_study_trace(mutated)


def test_json_schema_is_local_and_draft_2020_12(lifecycle_trace):
    schema = json.loads((_HPO / "contracts" / "study_trace.v1.schema.json").read_text())
    assert schema["$schema"] == "https://json-schema.org/draft/2020-12/schema"
    assert schema["$id"].endswith("/study_trace.v1.schema.json")
    Draft202012Validator.check_schema(schema)
    Draft202012Validator(schema).validate(lifecycle_trace)
    assert all(
        not value.startswith(("http://", "https://")) for value in _walk_refs(schema)
    )


def test_json_schema_enforces_status_specific_terminal_shape(lifecycle_trace):
    schema = json.loads((_HPO / "contracts" / "study_trace.v1.schema.json").read_text())
    validator = Draft202012Validator(schema)

    completed_with_error = copy.deepcopy(lifecycle_trace)
    completed_event = next(
        event
        for event in completed_with_error["events"]
        if event.get("status") == "COMPLETED"
    )
    completed_event["error_code"] = "IMPOSSIBLE"
    with pytest.raises(ValidationError):
        validator.validate(completed_with_error)

    failed_without_error = copy.deepcopy(lifecycle_trace)
    failed_event = next(
        event
        for event in failed_without_error["events"]
        if event.get("status") == "FAILED"
    )
    del failed_event["error_code"]
    with pytest.raises(ValidationError):
        validator.validate(failed_without_error)


def _walk_refs(value):
    if isinstance(value, dict):
        for key, item in value.items():
            if key == "$ref":
                yield item
            else:
                yield from _walk_refs(item)
    elif isinstance(value, list):
        for item in value:
            yield from _walk_refs(item)


def test_conformance_pack_checksums_are_fresh():
    validate_conformance_pack(
        _HPO / "contracts" / "study_trace_conformance_pack.v1.json"
    )


def test_trace_comparator_is_recursive_and_type_strict():
    golden = [
        {
            "error": {"retryable": False},
            "intermediates": [{"score": 1.0, "should_prune": False}],
        }
    ]
    assert compare_traces(copy.deepcopy(golden), golden) == []
    wrong_bool = copy.deepcopy(golden)
    wrong_bool[0]["error"]["retryable"] = 0
    assert compare_traces(wrong_bool, golden)
    missing_nested = copy.deepcopy(golden)
    del missing_nested[0]["intermediates"][0]["should_prune"]
    assert compare_traces(missing_nested, golden)


def test_javascript_consumer_independently_accepts_lifecycle_trace():
    node = shutil.which("node")
    if node is None:
        pytest.skip("Node.js is not installed; Python semantic gate still ran")
    subprocess.run(
        [
            node,
            str(_HPO / "consumers" / "validate_study_trace.mjs"),
            str(_HPO / "golden" / "lifecycle_states.v1.json"),
        ],
        check=True,
        capture_output=True,
        text=True,
    )


def test_ruby_consumer_independently_accepts_lifecycle_trace():
    ruby = shutil.which("ruby")
    if ruby is None:
        pytest.skip("Ruby is not installed; Python semantic gate still ran")
    subprocess.run(
        [
            ruby,
            str(_HPO / "consumers" / "validate_study_trace.rb"),
            str(_HPO / "golden" / "lifecycle_states.v1.json"),
            str(_HPO / "golden" / "racing_parallel.json"),
            str(_HPO / "golden" / "random_failed_trials.json"),
        ],
        check=True,
        capture_output=True,
        text=True,
    )


def test_ruby_consumer_rejects_corrupt_native_schedule(tmp_path):
    ruby = shutil.which("ruby")
    if ruby is None:
        pytest.skip("Ruby is not installed; Python semantic gate still ran")
    trace = json.loads((_HPO / "golden" / "racing_parallel.json").read_text())
    trace[1]["asked_at"] = trace[0]["asked_at"]
    corrupt = tmp_path / "corrupt-racing.json"
    corrupt.write_text(json.dumps(trace))
    completed = subprocess.run(
        [
            ruby,
            str(_HPO / "consumers" / "validate_study_trace.rb"),
            str(corrupt),
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode != 0
    assert "duplicate native event sequence" in completed.stderr
