# SPDX-License-Identifier: CECILL-2.1
"""Smoke test for the native HPO optimizer Python wrapper.

Run against a built libn4m:  N4M_LIB_PATH=/path/to/libn4m.so.2.1.0 \
    PYTHONPATH=bindings/python/src python bindings/python/tests/test_optimizer_smoke.py
(or via pytest with the same env).
"""

from __future__ import annotations

import copy
import gc
import struct
import weakref

import pytest

from n4m._errors import N4MError, PartialBatchError
from n4m._types import Status
from n4m.model_selection import (
    CategoryType,
    ConstraintKind,
    EvalMode,
    IntermediateValue,
    ParameterKind,
    TrialError,
    TrialParameter,
    TrialRecord,
)
from n4m.model_selection.optimizer import (
    Direction,
    Optimizer,
    Pruner,
    Sampler,
    SearchSpace,
    TrialStatus,
)


def test_role_package_reexports_public_optimizer_enums():
    assert ConstraintKind.EXCLUDE.value == 2
    assert EvalMode.MEAN.value == 1


def test_role_package_reexports_rich_trial_snapshot_types():
    assert TrialRecord.__module__.endswith("model_selection.optimizer")
    assert TrialParameter.__module__.endswith("model_selection.optimizer")
    assert TrialError.__module__.endswith("model_selection.optimizer")
    assert IntermediateValue.__module__.endswith("model_selection.optimizer")
    assert ParameterKind.CATEGORICAL.value == 4
    assert CategoryType.BOOL.value == 3


def test_random_quadratic():
    space = SearchSpace()
    space.add_float("x", -5.0, 5.0)
    assert space.num_params() == 1
    opt = Optimizer(
        space, sampler=Sampler.RANDOM, direction=Direction.MINIMIZE, seed=123
    )
    for _ in range(400):
        t = opt.ask()
        x = t.get_float("x")
        opt.tell(t.id, (x - 2.0) ** 2)
    best, score = opt.best()
    assert score < 0.25
    assert best.status == TrialStatus.COMPLETED


def test_tpe_mixed():
    space = SearchSpace()
    space.add_float("x", 0.0, 10.0)
    space.add_categorical("c", ["a", "b", "c"])
    opt = Optimizer(
        space,
        sampler=Sampler.TPE,
        direction=Direction.MINIMIZE,
        n_startup_trials=15,
        seed=2,
    )
    for _ in range(140):
        t = opt.ask()
        x = t.get_float("x")
        idx, label = t.get_category("c")
        pen = {0: 0.0, 1: 5.0, 2: 10.0}[idx]
        opt.tell(t.id, (x - 3.0) ** 2 + pen)
    best, score = opt.best()
    assert score < 1.0
    idx, label = best.get_category("c")
    assert label == "a"


def test_median_pruner():
    space = SearchSpace()
    space.add_int("k", 1, 10)
    opt = Optimizer(
        space,
        pruner=Pruner.MEDIAN,
        direction=Direction.MINIMIZE,
        n_startup_trials=2,
        seed=1,
    )
    ids = [opt.ask().id for _ in range(3)]
    assert opt.tell_intermediate(ids[0], 0, 1.0) is False
    assert opt.tell_intermediate(ids[1], 0, 2.0) is False
    assert opt.tell_intermediate(ids[2], 0, 5.0) is True  # worse than the peer median


def test_determinism():
    def first_x(seed):
        s = SearchSpace()
        s.add_float("x", -5.0, 5.0)
        o = Optimizer(s, seed=seed)
        return o.ask().get_float("x")

    assert first_x(42) == first_x(42)
    assert first_x(1) != first_x(2)


@pytest.mark.parametrize("sampler", [Sampler.RANDOM, Sampler.TPE, Sampler.PSO])
def test_checkpoint_resume_is_bit_exact_across_python_binding(sampler):
    with SearchSpace().add_float("x", -2.0, 2.0).add_int("k", 1, 7) as space:
        original = Optimizer(
            space,
            sampler=sampler,
            direction=Direction.MINIMIZE,
            n_startup_trials=2,
            seed=0x123456789,
        )
        try:
            prefix_trials = 32 if sampler is Sampler.PSO else 8
            for _ in range(prefix_trials):
                trial = original.ask()
                x = trial.get_float("x")
                k = trial.get_int("k")
                original.tell(trial.id, (x - 0.25) ** 2 + (k - 2) ** 2)
            checkpoint = original.save()
            assert checkpoint.startswith(b"N4MOPT\r\n")
            assert len(checkpoint) % 8 == 0
            restored = Optimizer.load(checkpoint)
            try:
                assert restored.get_trials() == original.get_trials()
                continuation_trials = 20 if sampler is Sampler.PSO else 4
                for _ in range(continuation_trials):
                    left = original.ask()
                    right = restored.ask()
                    assert left.id == right.id
                    assert left.get_float("x") == right.get_float("x")
                    assert left.get_int("k") == right.get_int("k")
                    score = left.get_float("x") ** 2 + left.get_int("k")
                    original.tell(left.id, score)
                    restored.tell(right.id, score)
            finally:
                restored.close()
        finally:
            original.close()


def _trace_without_duration(records):
    return [
        (
            record.id,
            record.ask_sequence,
            record.terminal_sequence,
            record.params,
            record.param_details,
            record.status,
            record.score,
            record.rung,
            record.intermediates,
            record.error,
        )
        for record in records
    ]


@pytest.mark.parametrize(
    ("pruner", "n_trials", "pre_reports", "post_reports", "max_resource"),
    [
        pytest.param(
            Pruner.NONE,
            3,
            [(0, 0, 1.0), (1, 0, 2.0)],
            [(2, 0, 9.0)],
            0,
            id="none",
        ),
        pytest.param(
            Pruner.MEDIAN,
            3,
            [(0, 0, 1.0), (1, 0, 2.0)],
            [(2, 0, 9.0)],
            0,
            id="median",
        ),
        pytest.param(
            Pruner.ASHA,
            3,
            [(0, 0, 1.0), (1, 0, 2.0)],
            [(2, 0, 9.0)],
            0,
            id="asha",
        ),
        pytest.param(
            Pruner.HYPERBAND,
            7,
            [(0, 0, 1.0), (3, 0, 2.0)],
            [(6, 0, 9.0)],
            9,
            id="hyperband",
        ),
        pytest.param(
            Pruner.RACING,
            3,
            [
                report
                for step in range(10)
                for report in ((0, step, 1.0), (1, step, 1.0))
            ]
            + [(2, 0, 10.0)],
            [(2, step, 10.0) for step in range(1, 10)],
            0,
            id="racing",
        ),
    ],
)
def test_checkpoint_preserves_all_pruner_decisions_and_terminal_trace(
    pruner, n_trials, pre_reports, post_reports, max_resource
):
    with SearchSpace().add_int("k", 1, 10) as space:
        original = Optimizer(
            space,
            pruner=pruner,
            direction=Direction.MINIMIZE,
            n_startup_trials=2,
            reduction_factor=(3 if pruner in {Pruner.ASHA, Pruner.HYPERBAND} else 0),
            max_resource=max_resource,
            seed=101,
        )
        try:
            for _ in range(n_trials):
                original.ask()
            for trial_id, step, score in pre_reports:
                assert original.tell_intermediate(trial_id, step, score) is False
            restored = Optimizer.load(original.save())
            try:
                assert restored.get_trials() == original.get_trials()
                decisions = []
                for trial_id, step, score in post_reports:
                    left = original.tell_intermediate(trial_id, step, score)
                    right = restored.tell_intermediate(trial_id, step, score)
                    assert left is right
                    decisions.append(left)
                    if left:
                        break

                target_id = post_reports[0][0]
                if pruner is Pruner.NONE:
                    assert decisions == [False]
                    original.tell(target_id, 9.0)
                    restored.tell(target_id, 9.0)
                    expected_status = TrialStatus.COMPLETED
                else:
                    assert decisions[-1] is True
                    expected_status = TrialStatus.PRUNED

                left_records = original.get_trials()
                right_records = restored.get_trials()
                assert _trace_without_duration(left_records) == _trace_without_duration(
                    right_records
                )
                assert left_records[target_id].status is expected_status
                assert right_records[target_id].status is expected_status
                assert (
                    left_records[target_id].terminal_sequence
                    == right_records[target_id].terminal_sequence
                )
            finally:
                restored.close()
        finally:
            original.close()


def _fnv1a64(data: bytes | bytearray) -> int:
    value = 14695981039346656037
    for byte in data:
        value ^= byte
        value = (value * 1099511628211) & 0xFFFFFFFFFFFFFFFF
    return value


def test_checkpoint_roundtrips_rich_ordered_space_and_rejects_invalid_utf8():
    space = (
        SearchSpace()
        .add_categorical("mode", ["café", "thé"])
        .add_categorical("gain", [1, 2])
        .add_ordinal("grade", [0.1, 1.5, 3.0])
        .add_float("alpha", 0.0, 1.0, step=0.125)
        .add_int("depth", 1, 9, step=2)
        .add_sorted_tuple("knots", 3, -1.0, 1.0)
        .add_constraint(
            ConstraintKind.CONDITION_IN,
            ["alpha", "mode"],
            ["", "café"],
        )
        .add_constraint(
            ConstraintKind.CONDITION_NOT_IN,
            ["depth", "mode"],
            ["", "café"],
        )
        .add_constraint(
            ConstraintKind.EXCLUDE,
            ["mode", "gain"],
            ["café", "2"],
        )
    )
    original = Optimizer(space, sampler=Sampler.RANDOM, seed=0xCAFE)
    try:

        def assert_rich_trial(trial):
            _, mode = trial.get_category("mode")
            _, gain = trial.get_category("gain")
            trial.get_category("grade")
            assert trial.is_active("alpha") is (mode == "café")
            assert trial.is_active("depth") is (mode != "café")
            assert not (mode == "café" and gain == "2")
            knots = [trial.get_float(f"knots#{index}") for index in range(3)]
            assert knots == sorted(knots)
            return mode, gain, knots

        for _ in range(12):
            trial = original.ask()
            mode, gain, knots = assert_rich_trial(trial)
            score = sum(value * value for value in knots) + (mode == "thé") + int(gain)
            original.tell(trial.id, score)

        checkpoint = original.save()
        records = original.get_trials()
        assert {record.params["mode"] for record in records} <= {"café", "thé"}
        assert all(record.params["gain"] in {1, 2} for record in records)
        assert all(record.params["grade"] in {0.1, 1.5, 3.0} for record in records)
        assert all(
            record.param_details["alpha"].active is (record.params["mode"] == "café")
            for record in records
        )
        assert all(
            record.param_details["depth"].active is (record.params["mode"] != "café")
            for record in records
        )

        restored = Optimizer.load(checkpoint)
        try:
            assert restored.get_trials() == records
            for _ in range(8):
                left = original.ask()
                right = restored.ask()
                assert left.id == right.id
                assert assert_rich_trial(left) == assert_rich_trial(right)
                assert left.get_float("alpha") == right.get_float("alpha")
                assert left.get_int("depth") == right.get_int("depth")
                score = left.get_float("alpha") + left.get_int("depth")
                original.tell(left.id, score)
                restored.tell(right.id, score)
        finally:
            restored.close()

        corrupt = bytearray(checkpoint)
        space_size = struct.unpack_from("<Q", corrupt, 40)[0]
        utf8_offset = corrupt.find("café".encode(), 48, 48 + space_size)
        assert utf8_offset >= 0
        corrupt[utf8_offset + 3] = 0xFF
        struct.pack_into("<Q", corrupt, 32, _fnv1a64(corrupt[48 : 48 + space_size]))
        struct.pack_into("<Q", corrupt, len(corrupt) - 8, _fnv1a64(corrupt[:-8]))
        with pytest.raises(N4MError) as excinfo:
            Optimizer.load(corrupt)
        assert excinfo.value.status == Status.ERR_CORRUPT_BUFFER
    finally:
        original.close()
        space.close()


def test_checkpoint_preserves_lifecycle_error_intermediate_and_warm_queue():
    with SearchSpace().add_int("k", 1, 7).add_float("x", -2.0, 2.0) as space:
        original = Optimizer(space, seed=77)
        try:
            completed, failed, running = [original.ask() for _ in range(3)]
            assert original.tell_intermediate(completed.id, 0, 2.5) is False
            original.tell(completed.id, 1.5)
            original.tell_result(
                failed.id,
                TrialStatus.FAILED,
                error=TrialError("EVAL_ERROR", "worker failed", retryable=True),
            )
            assert original.tell_intermediate(running.id, 3, 4.5) is False
            original.enqueue({"k": 3, "x": 0.25})
            restored = Optimizer.load(original.save())
            try:
                assert restored.get_trials() == original.get_trials()
                queued = restored.ask()
                assert queued.get_int("k") == 3
                assert queued.get_float("x") == 0.25
            finally:
                restored.close()
        finally:
            original.close()


def test_checkpoint_decode_is_transactional_and_fail_closed():
    with (
        SearchSpace().add_int("k", 1, 3) as space,
        Optimizer(space, seed=47) as optimizer,
    ):
        checkpoint = optimizer.save()
    corrupt = bytearray(checkpoint)
    corrupt[len(corrupt) // 2] ^= 1
    with pytest.raises(N4MError, match="n4m_optimizer_load") as excinfo:
        Optimizer.load(corrupt)
    assert excinfo.value.status == Status.ERR_CORRUPT_BUFFER
    with pytest.raises(ValueError, match="must not be empty"):
        Optimizer.load(b"")
    with pytest.raises(TypeError, match="bytes-like"):
        Optimizer.load("not bytes")


def test_trial_keeps_optimizer_alive_and_refuses_use_after_close():
    space = SearchSpace().add_int("k", 1, 3)
    optimizer = Optimizer(space, seed=11)
    trial = optimizer.ask()
    optimizer_ref = weakref.ref(optimizer)

    del optimizer
    gc.collect()

    parent = optimizer_ref()
    assert parent is not None
    assert trial.id == 0
    assert 1 <= trial.get_int("k") <= 3

    parent.close()
    with pytest.raises(RuntimeError, match="Optimizer is closed"):
        _ = trial.id


def test_optimizer_context_manager_closes_owned_context():
    space = SearchSpace().add_float("x", 0.0, 1.0)
    with Optimizer(space, seed=3) as optimizer:
        trial = optimizer.ask()
        assert 0.0 <= trial.get_float("x") <= 1.0
        owned_context = optimizer._ctx

    assert not optimizer._handle.value
    assert not owned_context.handle.value
    with pytest.raises(RuntimeError, match="Optimizer is closed"):
        optimizer.ask()


def test_owning_optimizer_handles_are_not_copyable():
    with SearchSpace().add_int("k", 1, 3) as space:
        for copier in (copy.copy, copy.deepcopy):
            with pytest.raises(TypeError, match="SearchSpace is not copyable"):
                copier(space)

        with Optimizer(space, seed=7) as optimizer:
            for copier in (copy.copy, copy.deepcopy):
                with pytest.raises(TypeError, match="Optimizer is not copyable"):
                    copier(optimizer)


def test_ask_batch_full_returns_trial_list():
    space = SearchSpace()
    space.add_int("k", 1, 100)
    with Optimizer(space, sampler=Sampler.RANDOM, seed=7) as opt:
        trials = opt.ask_batch(5)
        assert isinstance(trials, list)
        assert [t.id for t in trials] == [0, 1, 2, 3, 4]  # exactly n, in ask order
        assert opt.ask_batch(0) == []  # n == 0 -> empty list


def test_ask_batch_int32_validation_rejects_bool_and_overflow():
    space = SearchSpace()
    space.add_int("k", 1, 100)
    with Optimizer(space, seed=7) as opt:
        with pytest.raises(TypeError):
            opt.ask_batch(True)  # bool rejected even though bool is an int
        with pytest.raises(TypeError):
            opt.ask_batch(3.0)  # non-int rejected
        with pytest.raises(OverflowError):
            opt.ask_batch(2**31)  # outside int32
        with pytest.raises(OverflowError):
            opt.ask_batch(-(2**31) - 1)  # outside int32 on the negative side
        with pytest.raises(N4MError) as excinfo:
            opt.ask_batch(-1)  # native INVALID_ARGUMENT, zero progress
        assert excinfo.value.status == Status.ERR_INVALID_ARGUMENT
        assert not isinstance(excinfo.value, PartialBatchError)
        with pytest.raises(N4MError) as excinfo:
            opt.ask_batch(-(2**31))  # the exact int32 minimum reaches native code
        assert excinfo.value.status == Status.ERR_INVALID_ARGUMENT


def test_ask_batch_population_benign_partial_then_boundary():
    space = SearchSpace()
    space.add_float("x", 0.0, 1.0)
    with Optimizer(space, sampler=Sampler.GA, seed=1) as opt:
        first = opt.ask_batch(512)  # benign partial at the generation boundary
        assert 0 < len(first) < 512
        pop = len(first)
        with pytest.raises(N4MError) as excinfo:  # zero capacity -> stable status
            opt.ask_batch(4)
        assert excinfo.value.status == Status.ERR_INVALID_ARGUMENT
        assert not isinstance(excinfo.value, PartialBatchError)
        for trial in first:
            opt.tell(trial.id, 0.5)
        assert len(opt.ask_batch(512)) == pop  # a fresh full generation


def test_ask_batch_partial_batch_error_retains_committed_trials():
    space = SearchSpace()
    space.add_categorical("a", ["off", "on"])
    space.add_categorical("b", ["off", "on"])
    space.add_constraint(ConstraintKind.EXCLUDE, ["a", "b"], ["on", "on"])
    with Optimizer(space, sampler=Sampler.RANDOM, seed=3) as opt:
        opt.enqueue({"a": 0, "b": 0})  # valid: both off, satisfies EXCLUDE
        opt.enqueue({"a": 1, "b": 1})  # invalid: both on, violates EXCLUDE
        with pytest.raises(PartialBatchError) as excinfo:
            opt.ask_batch(4)
        err = excinfo.value
        assert isinstance(err, N4MError)  # typed subclass of N4MError
        assert err.status == Status.ERR_INVALID_ARGUMENT
        assert len(err.partial_trials) == 1  # the valid warm-start committed
        # The borrowed committed trial keeps the optimizer alive and is usable.
        assert err.partial_trials[0].id == 0
        recovered = opt.get_trials()
        assert len(recovered) == 1
        assert recovered[0].id == 0
        assert recovered[0].status == TrialStatus.RUNNING
        opt.tell(err.partial_trials[0].id, 0.0)  # terminal handling / recovery


def test_partial_batch_error_is_exported():
    import n4m
    import n4m.model_selection as model_selection

    assert n4m.PartialBatchError is PartialBatchError
    assert model_selection.PartialBatchError is PartialBatchError


if __name__ == "__main__":
    test_random_quadratic()
    test_tpe_mixed()
    test_median_pruner()
    test_determinism()
    test_ask_batch_full_returns_trial_list()
    test_ask_batch_int32_validation_rejects_bool_and_overflow()
    test_ask_batch_population_benign_partial_then_boundary()
    test_ask_batch_partial_batch_error_retains_committed_trials()
    print("optimizer python smoke: OK")
