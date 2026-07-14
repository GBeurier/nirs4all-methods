# SPDX-License-Identifier: CECILL-2.1
"""Fail-closed 9 x 5 sampler-pruner native conformance matrix.

This module deliberately separates two claims:

* the versioned JSON contract is a machine-readable inventory of all 45 plain
  numeric compositions plus stable transverse refusal vectors;
* the executable gate proves deterministic behavior through the native engine
  and Python binding.  It is not an independent sampler oracle and it is not a
  substitute for running the same tapes through another binding.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from specs import HpoSpec, ParamDecl

_HERE = Path(__file__).resolve().parent
CONTRACT_PATH = _HERE / "contracts" / "sampler_pruner_compatibility.v1.json"
SCHEMA_PATH = _HERE / "contracts" / "sampler_pruner_compatibility.v1.schema.json"

SAMPLERS = (
    "random",
    "sobol",
    "lhs",
    "ternary",
    "ga",
    "pso",
    "cmaes",
    "tpe",
    "gp_ei",
)
PRUNERS = ("none", "median", "asha", "hyperband", "racing")

_EXPECTED_SAMPLER_TRANSITIONS = {
    "random": ("sequential_history", "none", 0, 0),
    "sobol": ("sequential_history", "none", 0, 0),
    "lhs": ("sequential_history", "none", 0, 0),
    "ternary": ("sequential_history", "none", 0, 0),
    "ga": ("generation_synchronous", "generation", 16, 1),
    "pso": ("generation_synchronous", "generation", 16, 1),
    "cmaes": ("generation_synchronous", "generation", 6, 1),
    "tpe": ("sequential_history", "startup_history", 3, 3),
    "gp_ei": ("sequential_history", "startup_history", 3, 3),
}
_EXPECTED_PRUNER_OPTIONS = {
    "none": ("compatibility_curve", 4, 0, 0, 1, (0,)),
    "median": ("compatibility_curve", 5, 0, 0, 1, (0,)),
    "asha": ("compatibility_curve", 9, 0, 3, 1, (0,)),
    "hyperband": ("compatibility_curve", 9, 9, 3, 1, (0,)),
    "racing": ("racing_observation", 12, 0, 0, 2, (1, 0)),
}
_EXPECTED_REFUSALS = {
    "unknown_sampler_enum": (
        "unknown_sampler_enum",
        ("sampler:99",),
        "N4M_ERR_NOT_IMPLEMENTED",
        "UNKNOWN_SAMPLER_ENUM",
    ),
    "unknown_pruner_enum": (
        "unknown_pruner_enum",
        ("pruner:99",),
        "N4M_ERR_NOT_IMPLEMENTED",
        "UNKNOWN_PRUNER_ENUM",
    ),
    "adaptive_sampler_startup_zero": (
        "startup_zero",
        ("sampler:tpe", "sampler:gp_ei"),
        "N4M_ERR_INVALID_ARGUMENT",
        "ADAPTIVE_SAMPLER_REQUIRES_STARTUP",
    ),
    "median_startup_zero": (
        "startup_zero",
        ("pruner:median",),
        "N4M_ERR_INVALID_ARGUMENT",
        "MEDIAN_PRUNER_REQUIRES_STARTUP",
    ),
    "hyperband_without_max_resource": (
        "hyperband_without_max_resource",
        ("pruner:hyperband",),
        "N4M_ERR_INVALID_ARGUMENT",
        "HYPERBAND_REQUIRES_MAX_RESOURCE",
    ),
    "max_resource_outside_hyperband": (
        "max_resource_outside_hyperband",
        ("pruner:none", "pruner:median", "pruner:asha", "pruner:racing"),
        "N4M_ERR_INVALID_ARGUMENT",
        "MAX_RESOURCE_RESERVED_FOR_HYPERBAND",
    ),
    "reduction_factor_one": (
        "reduction_factor_one",
        ("pruner:asha", "pruner:hyperband"),
        "N4M_ERR_INVALID_ARGUMENT",
        "REDUCTION_FACTOR_MUST_BE_ZERO_OR_AT_LEAST_TWO",
    ),
    "reduction_factor_outside_successive_halving": (
        "reduction_factor_outside_successive_halving",
        ("pruner:none", "pruner:median", "pruner:racing"),
        "N4M_ERR_INVALID_ARGUMENT",
        "REDUCTION_FACTOR_RESERVED_FOR_ASHA_HYPERBAND",
    ),
    "non_mean_eval_mode": (
        "non_mean_eval_mode",
        ("eval_mode:best", "eval_mode:robust_best"),
        "N4M_ERR_NOT_IMPLEMENTED",
        "NON_MEAN_EVAL_MODE_NOT_IMPLEMENTED",
    ),
    "constant_liar_not_none": (
        "constant_liar_not_none",
        ("liar:min", "liar:mean", "liar:max"),
        "N4M_ERR_NOT_IMPLEMENTED",
        "CONSTANT_LIAR_NOT_IMPLEMENTED",
    ),
    "hard_constraint_unsupported_sampler": (
        "flat_hard_constraint",
        (
            "sampler:sobol",
            "sampler:ga",
            "sampler:pso",
            "sampler:cmaes",
            "sampler:gp_ei",
        ),
        "N4M_ERR_UNSUPPORTED",
        "SAMPLER_DOES_NOT_SUPPORT_HARD_CONSTRAINTS",
    ),
    "sorted_tuple_hard_constraint": (
        "sorted_tuple_hard_constraint",
        tuple(f"sampler:{sampler}" for sampler in SAMPLERS),
        "N4M_ERR_UNSUPPORTED",
        "SORTED_TUPLE_ROOT_HARD_CONSTRAINT_UNSUPPORTED",
    ),
}


class CompatibilityContractError(ValueError):
    """The matrix contract or one of its executable probes failed closed."""


@dataclass(frozen=True)
class CellReport:
    """Small CI-facing summary for one successfully executed matrix cell."""

    sampler: str
    pruner: str
    n_trials: int
    n_intermediates: int
    n_pruned: int


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise CompatibilityContractError(message)


def _is_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _require_keys(value: Any, expected: set[str], path: str) -> dict[str, Any]:
    _require(isinstance(value, dict), f"{path}: expected object")
    actual = set(value)
    _require(actual == expected, f"{path}: keys {sorted(actual)} != {sorted(expected)}")
    return value


def load_contract(path: Path = CONTRACT_PATH) -> dict[str, Any]:
    """Load and semantically validate compatibility contract v1."""

    document = json.loads(path.read_text(encoding="utf-8"))
    validate_contract(document)
    return document


def validate_contract(document: Any) -> None:
    """Validate exact coverage and stable meanings beyond JSON Schema shape."""

    root = _require_keys(
        document,
        {
            "$schema",
            "canonical_tape",
            "cells",
            "evidence_scope",
            "format",
            "pruners",
            "refusal_vectors",
            "samplers",
            "schema_version",
        },
        "$",
    )
    _require(
        root["$schema"] == SCHEMA_PATH.name,
        "$.$schema: unexpected local schema",
    )
    _require(
        root["format"] == "n4m.hpo.sampler_pruner_compatibility",
        "$.format: unsupported",
    )
    _require(
        _is_int(root["schema_version"]) and root["schema_version"] == 1,
        "$.schema_version: unsupported",
    )

    scope = _require_keys(
        root["evidence_scope"],
        {"does_not_prove", "level", "proves"},
        "$.evidence_scope",
    )
    _require(
        scope["level"] == "native_python_deterministic_conformance",
        "$.evidence_scope.level: unsupported",
    )
    for key in ("proves", "does_not_prove"):
        _require(
            isinstance(scope[key], list)
            and scope[key]
            and all(isinstance(item, str) and item for item in scope[key]),
            f"$.evidence_scope.{key}: expected non-empty strings",
        )
    _require(
        "cross_binding_45_of_45_parity" in scope["does_not_prove"],
        "$.evidence_scope: cross-binding non-claim is mandatory",
    )

    tape = _require_keys(
        root["canonical_tape"],
        {
            "checkpoint_after_trials",
            "checkpoint_state",
            "event_order",
            "n_startup_trials",
            "n_trials",
            "objective",
            "seed",
            "space",
        },
        "$.canonical_tape",
    )
    _require(tape["seed"] == 1701, "$.canonical_tape.seed: v1 drift")
    _require(_is_int(tape["seed"]), "$.canonical_tape.seed: expected integer")
    _require(tape["n_trials"] == 20, "$.canonical_tape.n_trials: v1 drift")
    _require(_is_int(tape["n_trials"]), "$.canonical_tape.n_trials: expected integer")
    _require(
        _is_int(tape["n_startup_trials"]) and tape["n_startup_trials"] == 3,
        "$.canonical_tape.n_startup_trials: v1 drift",
    )
    _require(
        _is_int(tape["checkpoint_after_trials"])
        and tape["checkpoint_after_trials"] == 14,
        "$.canonical_tape.checkpoint_after_trials: v1 drift",
    )
    _require(
        tape["checkpoint_state"] == "terminal_prefix_batch_boundary",
        "$.canonical_tape.checkpoint_state: v1 drift",
    )
    _require(tape["objective"] == "sphere", "$.canonical_tape.objective: v1 drift")
    _require(
        tape["event_order"] == "ASK_INTERMEDIATE_TERMINAL",
        "$.canonical_tape.event_order: v1 drift",
    )
    _require(
        tape["space"]
        == [
            {"kind": "int", "name": "k", "args": [1, 30]},
            {"kind": "float", "name": "x", "args": [-5.0, 5.0]},
        ],
        "$.canonical_tape.space: v1 drift",
    )
    _require(
        all(
            isinstance(arg, (int, float)) and not isinstance(arg, bool)
            for axis in tape["space"]
            for arg in axis["args"]
        ),
        "$.canonical_tape.space: boolean is not a numeric bound",
    )

    sampler_rows = root["samplers"]
    _require(isinstance(sampler_rows, list), "$.samplers: expected array")
    _require(
        all(isinstance(row, dict) for row in sampler_rows),
        "$.samplers: expected objects",
    )
    _require(
        tuple(row.get("id") for row in sampler_rows) == SAMPLERS,
        "$.samplers: expected each canonical sampler exactly once and in order",
    )
    sampler_by_id: dict[str, dict[str, Any]] = {}
    for index, raw in enumerate(sampler_rows):
        row = _require_keys(
            raw, {"adaptive_transition", "id", "schedule"}, f"$.samplers[{index}]"
        )
        transition = _require_keys(
            row["adaptive_transition"],
            {"after_trials", "kind", "minimum_completed"},
            f"$.samplers[{index}].adaptive_transition",
        )
        _require(
            _is_int(transition["after_trials"])
            and _is_int(transition["minimum_completed"]),
            f"$.samplers[{index}].adaptive_transition: expected integers",
        )
        expected = _EXPECTED_SAMPLER_TRANSITIONS[row["id"]]
        actual = (
            row["schedule"],
            transition["kind"],
            transition["after_trials"],
            transition["minimum_completed"],
        )
        _require(actual == expected, f"$.samplers[{index}]: transition contract drift")
        _require(
            transition["after_trials"] < tape["n_trials"]
            or transition["kind"] == "none",
            f"$.samplers[{index}]: canonical tape misses transition",
        )
        sampler_by_id[row["id"]] = row

    pruner_rows = root["pruners"]
    _require(isinstance(pruner_rows, list), "$.pruners: expected array")
    _require(
        all(isinstance(row, dict) for row in pruner_rows),
        "$.pruners: expected objects",
    )
    _require(
        tuple(row.get("id") for row in pruner_rows) == PRUNERS,
        "$.pruners: expected each canonical pruner exactly once and in order",
    )
    pruner_by_id: dict[str, dict[str, Any]] = {}
    for index, raw in enumerate(pruner_rows):
        row = _require_keys(
            raw,
            {
                "id",
                "intermediate",
                "intermediate_steps",
                "max_in_flight",
                "max_resource",
                "reduction_factor",
                "tell_order",
            },
            f"$.pruners[{index}]",
        )
        _require(
            all(
                _is_int(row[key])
                for key in (
                    "intermediate_steps",
                    "max_in_flight",
                    "max_resource",
                    "reduction_factor",
                )
            )
            and isinstance(row["tell_order"], list)
            and all(_is_int(item) for item in row["tell_order"]),
            f"$.pruners[{index}]: expected strict integer options",
        )
        actual = (
            row["intermediate"],
            row["intermediate_steps"],
            row["max_resource"],
            row["reduction_factor"],
            row["max_in_flight"],
            tuple(row["tell_order"]),
        )
        _require(
            actual == _EXPECTED_PRUNER_OPTIONS[row["id"]],
            f"$.pruners[{index}]: option contract drift",
        )
        _require(
            tape["checkpoint_after_trials"] % row["max_in_flight"] == 0,
            f"$.pruners[{index}]: checkpoint split crosses a logical batch",
        )
        pruner_by_id[row["id"]] = row

    cells = root["cells"]
    _require(isinstance(cells, list), "$.cells: expected array")
    expected_pairs = {(sampler, pruner) for sampler in SAMPLERS for pruner in PRUNERS}
    actual_pairs: set[tuple[str, str]] = set()
    for index, raw in enumerate(cells):
        cell = _require_keys(
            raw,
            {
                "checkpoint",
                "creation_status",
                "evidence",
                "pruner",
                "reason_code",
                "reproducibility",
                "sampler",
                "schedule",
                "status",
            },
            f"$.cells[{index}]",
        )
        pair = (cell["sampler"], cell["pruner"])
        _require(pair not in actual_pairs, f"$.cells[{index}]: duplicate pair {pair}")
        actual_pairs.add(pair)
        _require(cell["sampler"] in sampler_by_id, f"$.cells[{index}]: unknown sampler")
        _require(cell["pruner"] in pruner_by_id, f"$.cells[{index}]: unknown pruner")
        _require(cell["status"] == "supported", f"$.cells[{index}]: unsupported status")
        _require(
            cell["creation_status"] == "N4M_OK", f"$.cells[{index}]: creation drift"
        )
        _require(
            cell["reason_code"] == "NATIVE_COMPOSITION_SUPPORTED",
            f"$.cells[{index}]: reason drift",
        )
        _require(
            cell["schedule"] == sampler_by_id[cell["sampler"]]["schedule"],
            f"$.cells[{index}]: schedule drift",
        )
        _require(
            cell["reproducibility"] == "same_ordered_event_stream",
            f"$.cells[{index}]: reproducibility drift",
        )
        _require(
            cell["checkpoint"] == "save_restore_exact",
            f"$.cells[{index}]: checkpoint drift",
        )
        _require(
            cell["evidence"] == "native_python_deterministic_conformance",
            f"$.cells[{index}]: evidence drift",
        )
    _require(
        actual_pairs == expected_pairs,
        f"$.cells: missing={sorted(expected_pairs - actual_pairs)} "
        f"extra={sorted(actual_pairs - expected_pairs)}",
    )
    _require(len(cells) == 45, "$.cells: expected exactly 45 rows")

    refusals = root["refusal_vectors"]
    _require(isinstance(refusals, list), "$.refusal_vectors: expected array")
    by_id: dict[str, dict[str, Any]] = {}
    for index, raw in enumerate(refusals):
        vector = _require_keys(
            raw,
            {"applies_to", "expected_status", "fixture", "id", "reason_code"},
            f"$.refusal_vectors[{index}]",
        )
        vector_id = vector["id"]
        _require(vector_id not in by_id, f"$.refusal_vectors[{index}]: duplicate id")
        _require(
            vector_id in _EXPECTED_REFUSALS, f"$.refusal_vectors[{index}]: unknown id"
        )
        actual = (
            vector["fixture"],
            tuple(vector["applies_to"]),
            vector["expected_status"],
            vector["reason_code"],
        )
        _require(
            actual == _EXPECTED_REFUSALS[vector_id],
            f"$.refusal_vectors[{index}]: stable refusal drift",
        )
        by_id[vector_id] = vector
    _require(
        set(by_id) == set(_EXPECTED_REFUSALS),
        f"$.refusal_vectors: missing={sorted(set(_EXPECTED_REFUSALS) - set(by_id))}",
    )


def make_spec(document: dict[str, Any], cell: dict[str, Any]) -> HpoSpec:
    """Project one matrix row and its shared catalogs into an executable spec."""

    tape = document["canonical_tape"]
    pruner = next(row for row in document["pruners"] if row["id"] == cell["pruner"])
    return HpoSpec(
        id=f"compatibility_{cell['sampler']}_{cell['pruner']}",
        sampler=cell["sampler"],
        pruner=cell["pruner"],
        space=tuple(
            ParamDecl(axis["kind"], axis["name"], tuple(axis["args"]))
            for axis in tape["space"]
        ),
        objective=tape["objective"],
        n_trials=tape["n_trials"],
        seed=tape["seed"],
        n_startup_trials=tape["n_startup_trials"],
        intermediate=pruner["intermediate"],
        intermediate_steps=pruner["intermediate_steps"],
        max_resource=pruner["max_resource"],
        reduction_factor=pruner["reduction_factor"],
        max_in_flight=pruner["max_in_flight"],
        tell_order=tuple(pruner["tell_order"]),
    )


def _validate_trace(
    document: dict[str, Any], cell: dict[str, Any], trace: list[dict]
) -> CellReport:
    tape = document["canonical_tape"]
    _require(len(trace) == tape["n_trials"], f"{cell}: incomplete trial tape")
    _require(
        [record["id"] for record in trace] == list(range(tape["n_trials"])),
        f"{cell}: trial ids are not contiguous",
    )
    _require(
        all(record.get("status") in {1, 2} for record in trace),
        f"{cell}: a trial did not reach COMPLETED or PRUNED",
    )
    _require(
        all("terminal_at" in record for record in trace),
        f"{cell}: a terminal event is missing",
    )
    _require(
        not any("duration" in key for record in trace for key in record),
        f"{cell}: wall-clock duration leaked into the normative replay tape",
    )
    _require(
        all(record.get("intermediates") for record in trace),
        f"{cell}: every trial must evaluate at least one INTERMEDIATE",
    )

    sequences: list[int] = []
    pruner_row = next(row for row in document["pruners"] if row["id"] == cell["pruner"])
    for record in trace:
        asked = record["asked_at"]
        terminal = record["terminal_at"]
        intermediate_sequences = [item["sequence"] for item in record["intermediates"]]
        _require(
            asked < intermediate_sequences[0] <= intermediate_sequences[-1] < terminal,
            f"{cell}: expected ASK < INTERMEDIATE < TERMINAL for trial {record['id']}",
        )
        _require(
            intermediate_sequences == sorted(intermediate_sequences)
            and len(intermediate_sequences) == len(set(intermediate_sequences)),
            f"{cell}: intermediate event order is not strict",
        )
        steps = [item["step"] for item in record["intermediates"]]
        _require(
            steps == list(range(len(steps)))
            and len(steps) <= pruner_row["intermediate_steps"],
            f"{cell}: intermediate steps are not the canonical prefix",
        )
        if record["status"] == 2:
            _require(
                record["intermediates"][-1]["should_prune"] is True
                and record["pruned_at"] == steps[-1],
                f"{cell}: PRUNED trial lacks a final true decision",
            )
        else:
            _require(
                len(steps) == pruner_row["intermediate_steps"]
                and not any(item["should_prune"] for item in record["intermediates"]),
                f"{cell}: COMPLETED trial has a truncated/true intermediate tape",
            )
        sequences.extend((asked, *intermediate_sequences, terminal))
    _require(
        sorted(sequences) == list(range(len(sequences))),
        f"{cell}: global event sequence has a gap or duplicate",
    )

    lifecycle_events = sorted(
        [(record["asked_at"], 1) for record in trace]
        + [(record["terminal_at"], -1) for record in trace]
    )
    in_flight = 0
    observed_max = 0
    for _sequence, delta in lifecycle_events:
        in_flight += delta
        _require(in_flight >= 0, f"{cell}: terminal event preceded ASK")
        observed_max = max(observed_max, in_flight)
    _require(in_flight == 0, f"{cell}: lifecycle left a trial running")
    _require(
        observed_max == pruner_row["max_in_flight"],
        f"{cell}: observed max_in_flight {observed_max} != "
        f"{pruner_row['max_in_flight']}",
    )

    n_pruned = sum(record["status"] == 2 for record in trace)
    if cell["pruner"] == "none":
        _require(n_pruned == 0, f"{cell}: none pruner changed terminal status")
        _require(
            not any(
                item["should_prune"]
                for record in trace
                for item in record["intermediates"]
            ),
            f"{cell}: none pruner emitted a prune decision",
        )
    else:
        _require(n_pruned > 0, f"{cell}: pruner never exercised a true decision")

    sampler_row = next(
        row for row in document["samplers"] if row["id"] == cell["sampler"]
    )
    transition = sampler_row["adaptive_transition"]
    if transition["kind"] == "generation":
        generation_size = transition["after_trials"]
        first_generation = trace[:generation_size]
        _require(
            len(trace) > generation_size, f"{cell}: next generation was never asked"
        )
        _require(
            sum(record["status"] == 1 for record in first_generation)
            >= transition["minimum_completed"],
            f"{cell}: first generation had no scored member to learn from",
        )
        for boundary in range(generation_size, len(trace), generation_size):
            previous_generation = trace[boundary - generation_size : boundary]
            _require(
                trace[boundary]["asked_at"]
                > max(record["terminal_at"] for record in previous_generation),
                f"{cell}: population crossed boundary {boundary} before "
                "full terminalization",
            )
    elif transition["kind"] == "startup_history":
        completed = sorted(
            (record for record in trace if record["status"] == 1),
            key=lambda record: record["terminal_at"],
        )
        minimum = transition["minimum_completed"]
        _require(
            len(completed) >= minimum,
            f"{cell}: adaptive startup never accumulated {minimum} completed trials",
        )
        startup_terminal = completed[minimum - 1]["terminal_at"]
        _require(
            any(record["asked_at"] > startup_terminal for record in trace),
            f"{cell}: no ASK exercised the post-startup adaptive path",
        )

    if cell["pruner"] == "racing":
        for batch_start in range(0, len(trace), 2):
            first, second = trace[batch_start : batch_start + 2]
            _require(
                first["asked_at"]
                < second["asked_at"]
                < second["intermediates"][0]["sequence"],
                f"{cell}: logical two-in-flight batch drifted",
            )
            first_by_step = {
                item["step"]: item["sequence"] for item in first["intermediates"]
            }
            second_by_step = {
                item["step"]: item["sequence"] for item in second["intermediates"]
            }
            for step in first_by_step.keys() & second_by_step.keys():
                _require(
                    second_by_step[step] < first_by_step[step],
                    f"{cell}: racing tell_order=(1,0) drifted at step {step}",
                )

    return CellReport(
        sampler=cell["sampler"],
        pruner=cell["pruner"],
        n_trials=len(trace),
        n_intermediates=sum(len(record["intermediates"]) for record in trace),
        n_pruned=n_pruned,
    )


def execute_cell(document: dict[str, Any], cell: dict[str, Any]) -> CellReport:
    """Execute replay + checkpoint continuation for one supported matrix cell."""

    import run_native

    spec = make_spec(document, cell)
    direct = run_native.run_spec(spec)
    replay = run_native.run_spec(spec)
    resumed = run_native.run_spec(
        spec,
        checkpoint_after=document["canonical_tape"]["checkpoint_after_trials"],
    )
    _require(direct == replay, f"{spec.id}: exact deterministic replay diverged")
    _require(direct == resumed, f"{spec.id}: checkpoint continuation diverged")
    report = _validate_trace(document, cell, direct)
    if cell["sampler"] in {"tpe", "gp_ei"}:
        random_cell = next(
            row
            for row in document["cells"]
            if row["sampler"] == "random" and row["pruner"] == cell["pruner"]
        )
        random_trace = run_native.run_spec(make_spec(document, random_cell))
        completed = sorted(
            (record for record in direct if record["status"] == 1),
            key=lambda record: record["terminal_at"],
        )
        minimum = document["canonical_tape"]["n_startup_trials"]
        startup_terminal = completed[minimum - 1]["terminal_at"]
        first_adaptive = next(
            record["id"] for record in direct if record["asked_at"] > startup_terminal
        )
        _require(
            all(
                direct[index]["params"] == random_trace[index]["params"]
                for index in range(first_adaptive)
            ),
            f"{spec.id}: startup fallback diverged from random before its threshold",
        )
        _require(
            direct[first_adaptive]["params"] != random_trace[first_adaptive]["params"],
            f"{spec.id}: first post-startup ASK did not exercise adaptive sampling",
        )
    return report


def _refusal_kwargs(fixture: str, target: str) -> dict[str, Any]:
    from n4m.model_selection.optimizer import EvalMode, Pruner, Sampler

    kind, value = target.split(":", 1)
    samplers = {item.name.lower(): item for item in Sampler}
    pruners = {item.name.lower(): item for item in Pruner}
    kwargs: dict[str, Any] = {"n_startup_trials": 3}
    if fixture == "unknown_sampler_enum":
        kwargs["sampler"] = int(value)
    elif fixture == "unknown_pruner_enum":
        kwargs["pruner"] = int(value)
    elif fixture == "startup_zero":
        kwargs["n_startup_trials"] = 0
        kwargs[kind] = samplers[value] if kind == "sampler" else pruners[value]
    elif fixture == "hyperband_without_max_resource":
        kwargs["pruner"] = Pruner.HYPERBAND
    elif fixture == "max_resource_outside_hyperband":
        kwargs.update(pruner=pruners[value], max_resource=9)
    elif fixture == "reduction_factor_one":
        kwargs.update(pruner=pruners[value], reduction_factor=1)
        if value == "hyperband":
            kwargs["max_resource"] = 9
    elif fixture == "reduction_factor_outside_successive_halving":
        kwargs.update(pruner=pruners[value], reduction_factor=3)
    elif fixture == "non_mean_eval_mode":
        kwargs["eval_mode"] = {
            "best": EvalMode.BEST,
            "robust_best": EvalMode.ROBUST_BEST,
        }[value]
    elif fixture == "constant_liar_not_none":
        pass  # exercised through the C ABI because Python intentionally omits it
    elif fixture in {"flat_hard_constraint", "sorted_tuple_hard_constraint"}:
        kwargs["sampler"] = samplers[value]
    else:  # pragma: no cover - semantic validation rejects unknown fixtures
        raise CompatibilityContractError(f"unknown refusal fixture: {fixture}")
    return kwargs


def _make_refusal_space(fixture: str):
    from n4m.model_selection.optimizer import ConstraintKind, SearchSpace

    space = SearchSpace()
    if fixture == "flat_hard_constraint":
        space.add_categorical("a", ["off", "on"])
        space.add_categorical("b", ["off", "on"])
        space.add_constraint(ConstraintKind.EXCLUDE, ["a", "b"], ["on", "on"])
    elif fixture == "sorted_tuple_hard_constraint":
        space.add_sorted_tuple("knots", 2, 0.0, 1.0)
        space.add_categorical("flag", ["off", "on"])
        space.add_constraint(ConstraintKind.EXCLUDE, ["knots", "flag"], ["", "on"])
    else:
        space.add_float("x", 0.0, 1.0)
    return space


def execute_refusal_vector(vector: dict[str, Any]) -> int:
    """Execute every target in a refusal vector and compare only stable status."""

    from n4m._errors import N4MError
    from n4m._types import Status
    from n4m.model_selection.optimizer import Optimizer

    expected_status = {
        "N4M_ERR_INVALID_ARGUMENT": int(Status.ERR_INVALID_ARGUMENT),
        "N4M_ERR_UNSUPPORTED": int(Status.ERR_UNSUPPORTED),
        "N4M_ERR_NOT_IMPLEMENTED": int(Status.ERR_NOT_IMPLEMENTED),
    }[vector["expected_status"]]
    probes = 0
    for target in vector["applies_to"]:
        space = _make_refusal_space(vector["fixture"])
        try:
            if vector["fixture"] == "constant_liar_not_none":
                from ctypes import byref, c_void_p

                from n4m._context import Context
                from n4m._ffi import lib
                from n4m._types import OptimizerOptions

                _kind, value = target.split(":", 1)
                options = OptimizerOptions()
                lib.n4m_optimizer_options_init(byref(options))
                options.liar = {"min": 1, "mean": 2, "max": 3}[value]
                handle = c_void_p()
                context = Context()
                try:
                    status = int(
                        lib.n4m_optimizer_create(
                            context._handle,  # noqa: SLF001 - direct C ABI refusal probe
                            space._handle,  # noqa: SLF001 - direct C ABI refusal probe
                            byref(options),
                            byref(handle),
                        )
                    )
                finally:
                    if handle.value:
                        lib.n4m_optimizer_destroy(handle)
                    context.close()
                _require(
                    status == expected_status,
                    f"{vector['id']}[{target}]: status {status} != "
                    f"{vector['expected_status']}",
                )
                probes += 1
                continue
            try:
                optimizer = Optimizer(
                    space, **_refusal_kwargs(vector["fixture"], target)
                )
            except N4MError as error:
                _require(
                    error.status == expected_status,
                    f"{vector['id']}[{target}]: status {error.status} != "
                    f"{vector['expected_status']}",
                )
            else:
                optimizer.close()
                raise CompatibilityContractError(
                    f"{vector['id']}[{target}]: expected typed refusal, got N4M_OK"
                )
        finally:
            space.close()
        probes += 1
    return probes


def run_compatibility_gate(
    document: dict[str, Any] | None = None,
) -> tuple[list[CellReport], int]:
    """Execute all 45 cells and all declared typed refusal probes."""

    contract = load_contract() if document is None else document
    validate_contract(contract)
    reports = [execute_cell(contract, cell) for cell in contract["cells"]]
    refusal_probes = sum(
        execute_refusal_vector(vector) for vector in contract["refusal_vectors"]
    )
    _require(len(reports) == 45, "compatibility gate did not execute all 45 cells")
    return reports, refusal_probes


__all__ = [
    "CONTRACT_PATH",
    "PRUNERS",
    "SAMPLERS",
    "SCHEMA_PATH",
    "CellReport",
    "CompatibilityContractError",
    "execute_cell",
    "execute_refusal_vector",
    "load_contract",
    "make_spec",
    "run_compatibility_gate",
    "validate_contract",
]
