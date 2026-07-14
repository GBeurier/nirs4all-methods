# SPDX-License-Identifier: CECILL-2.1
"""Public Python/ABI edge contracts omitted by the original Track-Q matrix."""

from __future__ import annotations

import math
import os
import sys
from ctypes import POINTER, byref, c_double, c_int64, c_void_p
from pathlib import Path

import pytest

if "N4M_LIB_PATH" not in os.environ:
    pytest.skip("no libn4m build (set N4M_LIB_PATH)", allow_module_level=True)

_REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(_REPO / "bindings" / "python" / "src"))

from n4m._errors import N4MError, check  # noqa: E402
from n4m._ffi import lib  # noqa: E402
from n4m._types import Status  # noqa: E402
from n4m.model_selection.optimizer import (  # noqa: E402
    ConstraintKind,
    Direction,
    IntermediateValue,
    Optimizer,
    ParameterKind,
    Pruner,
    SearchSpace,
    TrialError,
    TrialParameter,
    TrialRecord,
    TrialStatus,
)


def _matrix(result: c_void_p, name: str) -> list[float]:
    data = POINTER(c_double)()
    rows = c_int64()
    cols = c_int64()
    check(
        lib.n4m_method_result_get_double_matrix(
            result, name.encode(), byref(data), byref(rows), byref(cols)
        ),
        f"n4m_method_result_get_double_matrix({name})",
    )
    assert rows.value == 1
    return [float(data[index]) for index in range(cols.value)]


def _trial_trace(optimizer: Optimizer, since_id: int = 0) -> dict[str, list[float]]:
    result = c_void_p()
    check(
        lib.n4m_optimizer_get_trials(
            optimizer._handle,
            since_id,
            byref(result),  # noqa: SLF001
        ),
        "n4m_optimizer_get_trials",
    )
    try:
        return {
            name: _matrix(result, name)
            for name in (
                "trial_ids",
                "trial_scores",
                "trial_status",
                "trial_rung",
                "trial_duration",
            )
        }
    finally:
        lib.n4m_method_result_destroy(result)


def test_mutex_group_rejects_only_the_all_present_label_combination():
    with SearchSpace() as space:
        space.add_categorical("a", ["off", "on"])
        space.add_categorical("b", ["off", "on"])
        space.add_constraint(
            ConstraintKind.MUTEX_GROUP,
            ["a", "b"],
            ["on", "on"],
        )
        seen: set[tuple[str, str]] = set()
        with Optimizer(space, seed=31) as optimizer:
            for _ in range(96):
                trial = optimizer.ask()
                a = trial.get_category("a")[1]
                b = trial.get_category("b")[1]
                assert not (a == "on" and b == "on")
                seen.add((a, b))
                optimizer.tell(trial.id, 0.0)
    assert seen == {("off", "off"), ("off", "on"), ("on", "off")}


def test_condition_not_in_controls_child_activation():
    with SearchSpace() as space:
        space.add_float("child", 0.0, 1.0)
        space.add_categorical("parent", ["off", "on"])
        space.add_constraint(
            ConstraintKind.CONDITION_NOT_IN,
            ["child", "parent"],
            ["", "off"],
        )
        seen: set[str] = set()
        with Optimizer(space, seed=37) as optimizer:
            for _ in range(64):
                trial = optimizer.ask()
                parent = trial.get_category("parent")[1]
                seen.add(parent)
                assert trial.is_active("child") is (parent != "off")
                record = optimizer.get_trials(since_id=trial.id)[0]
                assert "child" in record.params
                assert record.param_details["child"].active is (parent != "off")
                optimizer.tell(trial.id, 0.0)
    assert seen == {"off", "on"}


def test_terminal_lifecycle_unknown_ids_double_tell_and_best_before_tell():
    with (
        SearchSpace().add_int("k", 1, 3) as space,
        Optimizer(space, seed=41) as optimizer,
    ):
        assert optimizer.best() is None

        failed = optimizer.ask()
        with pytest.raises(N4MError) as unknown:
            optimizer.tell_result(999, TrialStatus.FAILED, error="unknown trial")
        assert unknown.value.status == Status.ERR_INVALID_ARGUMENT

        optimizer.tell_result(failed.id, TrialStatus.FAILED, error="fixture failure")
        assert failed.status == TrialStatus.FAILED
        assert optimizer.best() is None
        optimizer.tell_result(failed.id, TrialStatus.FAILED, error="fixture failure")
        with pytest.raises(N4MError) as changed_error:
            optimizer.tell_result(
                failed.id, TrialStatus.FAILED, error="rewritten failure"
            )
        assert changed_error.value.status == Status.ERR_INVALID_ARGUMENT
        with pytest.raises(N4MError) as failed_to_completed:
            optimizer.tell(failed.id, 0.0)
        assert failed_to_completed.value.status == Status.ERR_INVALID_ARGUMENT

        completed = optimizer.ask()
        optimizer.tell_result(completed.id, TrialStatus.COMPLETED, score=1.5)
        optimizer.tell_result(completed.id, TrialStatus.COMPLETED, score=1.5)
        with pytest.raises(N4MError) as completed_to_failed:
            optimizer.tell_result(
                completed.id, TrialStatus.FAILED, error="late failure"
            )
        assert completed_to_failed.value.status == Status.ERR_INVALID_ARGUMENT
        best, score = optimizer.best()
        assert best.id == completed.id
        assert score == 1.5


def test_get_trials_exposes_running_failed_completed_and_since_filter():
    with (
        SearchSpace().add_int("k", 1, 3) as space,
        Optimizer(space, seed=43) as optimizer,
    ):
        failed = optimizer.ask()
        optimizer.tell_result(failed.id, TrialStatus.FAILED, error="fixture failure")
        completed = optimizer.ask()
        optimizer.tell(completed.id, 2.5)
        running = optimizer.ask()

        trace = _trial_trace(optimizer)
        assert trace["trial_ids"] == [0.0, 1.0, 2.0]
        assert trace["trial_status"] == [3.0, 1.0, 0.0]
        assert math.isnan(trace["trial_scores"][0])
        assert trace["trial_scores"][1] == 2.5
        assert math.isnan(trace["trial_scores"][2])
        assert trace["trial_rung"] == [0.0, 0.0, 0.0]
        assert all(value >= 0.0 for value in trace["trial_duration"])

        filtered = _trial_trace(optimizer, since_id=running.id)
        assert filtered["trial_ids"] == [2.0]
        assert filtered["trial_status"] == [0.0]


def test_public_rich_trial_records_cover_errors_intermediates_and_ownership():
    with SearchSpace() as space:
        space.add_int("k", 1, 3)
        space.add_categorical("kind", ["alpha", "beta"])
        with Optimizer(space, seed=45) as optimizer:
            assert optimizer.get_trials() == []
            failed = optimizer.ask()
            assert optimizer.tell_intermediate(failed.id, 0, 4.0) is False
            assert optimizer.tell_intermediate(failed.id, 0, 4.0) is False
            with pytest.raises(N4MError) as rewritten_intermediate:
                optimizer.tell_intermediate(failed.id, 0, 5.0)
            assert rewritten_intermediate.value.status == Status.ERR_INVALID_ARGUMENT
            assert optimizer.tell_intermediate(failed.id, 2, 3.0) is False
            with pytest.raises(N4MError) as out_of_order:
                optimizer.tell_intermediate(failed.id, 1, 2.0)
            assert out_of_order.value.status == Status.ERR_INVALID_ARGUMENT
            failure = TrialError(
                code="OBJECTIVE_EXCEPTION",
                message="échec déterministe|with separator",
                retryable=False,
            )
            optimizer.tell_result(failed.id, TrialStatus.FAILED, error=failure)
            optimizer.tell_result(failed.id, TrialStatus.FAILED, error=failure)

            cancelled = optimizer.ask()
            cancellation = TrialError(
                code="BUDGET_CANCELLED",
                message="study timeout",
                retryable=True,
            )
            optimizer.tell_result(
                cancelled.id,
                TrialStatus.CANCELLED,
                error=cancellation,
            )
            optimizer.tell_result(
                cancelled.id,
                TrialStatus.CANCELLED,
                error=cancellation,
            )
            with pytest.raises(N4MError) as changed_cancellation:
                optimizer.tell_result(
                    cancelled.id,
                    TrialStatus.CANCELLED,
                    error=TrialError(
                        code="BUDGET_CANCELLED",
                        message="different timeout",
                        retryable=True,
                    ),
                )
            assert changed_cancellation.value.status == Status.ERR_INVALID_ARGUMENT
            assert cancelled.status == TrialStatus.CANCELLED

            completed = optimizer.ask()
            optimizer.tell(completed.id, 1.5)
            optimizer.ask()

            records = optimizer.get_trials()
            filtered = optimizer.get_trials(since_id=cancelled.id)
            assert [record.id for record in records] == [0, 1, 2, 3]
            assert [record.id for record in filtered] == [1, 2, 3]
            for invalid_since_id in (-1, True, 2**63):
                with pytest.raises(ValueError, match="since_id"):
                    optimizer.get_trials(invalid_since_id)

        # TrialRecord is owning; the borrowed Trial handles above are now closed.
        assert all(isinstance(record, TrialRecord) for record in records)
        assert records[0].status == TrialStatus.FAILED
        assert records[0].error == failure
        assert records[0].score is None
        assert records[0].intermediates == (
            IntermediateValue(sequence=1, step=0, score=4.0, should_prune=False),
            IntermediateValue(sequence=2, step=2, score=3.0, should_prune=False),
        )
        assert records[0].ask_sequence == 0
        assert records[0].terminal_sequence == 3
        assert isinstance(records[0].param_details["k"], TrialParameter)
        assert isinstance(records[0].params["k"], int)
        assert records[0].params["kind"] in {"alpha", "beta"}
        assert records[1].status == TrialStatus.CANCELLED
        assert records[1].error == cancellation
        assert records[2].status == TrialStatus.COMPLETED
        assert records[2].score == 1.5
        assert records[2].error is None
        assert records[3].status == TrialStatus.RUNNING
        assert records[3].score is None
        assert records[3].terminal_sequence is None


def test_legacy_and_malformed_trial_errors_fail_closed():
    with (
        SearchSpace().add_int("k", 1, 3) as space,
        Optimizer(space, seed=46) as optimizer,
    ):
        legacy = optimizer.ask()
        optimizer.tell_result(
            legacy.id,
            TrialStatus.FAILED,
            error="legacy objective failure",
        )
        record = optimizer.get_trials()[0]
        assert record.error == TrialError(
            code="OBJECTIVE_ERROR",
            message="legacy objective failure",
            retryable=False,
        )

        malformed = optimizer.ask()
        with pytest.raises(ValueError, match="TrialError.code"):
            optimizer.tell_result(
                malformed.id,
                TrialStatus.FAILED,
                error=TrialError("lowercase", "bad"),
            )
        assert malformed.status == TrialStatus.RUNNING
        with pytest.raises(N4MError) as unknown_version:
            optimizer.tell_result(
                malformed.id,
                TrialStatus.CANCELLED,
                error="n4m.error.v2|BUDGET_CANCELLED|1|future",
            )
        assert unknown_version.value.status == Status.ERR_INVALID_ARGUMENT
        assert malformed.status == TrialStatus.RUNNING


def test_rich_trace_preserves_declared_parameter_types():
    with SearchSpace() as space:
        space.add_int("integer", 1, 3)
        space.add_float("continuous", 0.0, 1.0)
        space.add_categorical("string_category", ["alpha", "beta"])
        space.add_categorical("int_category", [10, 20])
        space.add_categorical("float_category", [1.5, 2.5])
        space.add_categorical("bool_category", [False, True])
        space.add_ordinal("ordinal", [0.25, 0.75])
        space.add_sorted_tuple(
            "knots",
            length=2,
            low=0,
            high=4,
            element_is_int=True,
        )
        with Optimizer(space, seed=49) as optimizer:
            optimizer.ask()
            record = optimizer.get_trials()[0]

        assert isinstance(record.params["integer"], int)
        assert isinstance(record.params["continuous"], float)
        assert isinstance(record.params["string_category"], str)
        assert isinstance(record.params["int_category"], int)
        assert isinstance(record.params["float_category"], float)
        assert isinstance(record.params["bool_category"], bool)
        assert isinstance(record.params["ordinal"], float)
        assert isinstance(record.params["knots#0"], int)
        assert isinstance(record.params["knots#1"], int)
        assert record.param_details["integer"].kind is ParameterKind.INT
        assert record.param_details["knots#0"].kind is ParameterKind.SORTED_TUPLE
        assert record.param_details["knots#0"].integer is True


def test_prune_causing_intermediate_replay_returns_original_decision():
    with (
        SearchSpace().add_int("k", 1, 10) as space,
        Optimizer(
            space,
            pruner=Pruner.MEDIAN,
            direction=Direction.MINIMIZE,
            n_startup_trials=2,
            seed=51,
        ) as optimizer,
    ):
        trials = [optimizer.ask() for _ in range(3)]
        assert optimizer.tell_intermediate(trials[0].id, 0, 1.0) is False
        assert optimizer.tell_intermediate(trials[1].id, 0, 2.0) is False
        assert optimizer.tell_intermediate(trials[2].id, 0, 9.0) is True
        assert optimizer.tell_intermediate(trials[2].id, 0, 9.0) is True
        with pytest.raises(N4MError) as rewritten:
            optimizer.tell_intermediate(trials[2].id, 0, 8.0)
        assert rewritten.value.status == Status.ERR_INVALID_ARGUMENT
        pruned = optimizer.get_trials(since_id=trials[2].id)[0]
        assert pruned.status is TrialStatus.PRUNED
        assert len(pruned.intermediates) == 1
        assert pruned.intermediates[0].should_prune is True


def test_optimizer_persistence_round_trip_and_corruption_fail_closed():
    with (
        SearchSpace().add_int("k", 1, 3) as space,
        Optimizer(space, seed=47) as optimizer,
    ):
        trial = optimizer.ask()
        optimizer.tell(trial.id, float(trial.get_int("k")))
        checkpoint = optimizer.save()
        assert checkpoint.startswith(b"N4MOPT\r\n")
        with Optimizer.load(checkpoint) as loaded:
            left = optimizer.ask()
            right = loaded.ask()
            assert left.id == right.id
            assert left.get_int("k") == right.get_int("k")

        corrupt = bytearray(checkpoint)
        corrupt[len(corrupt) // 2] ^= 1
        with pytest.raises(N4MError) as excinfo:
            Optimizer.load(corrupt)
        assert excinfo.value.status == Status.ERR_CORRUPT_BUFFER
