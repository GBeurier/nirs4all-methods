# SPDX-License-Identifier: CECILL-2.1 OR AGPL-3.0-or-later
"""Logarithmic optimizer proposals must survive their own checkpoints."""

import math
import struct

import pytest
from n4m.model_selection.optimizer import (
    Direction,
    Optimizer,
    Sampler,
    SearchSpace,
    TrialStatus,
)


def bits(value):
    return struct.pack(">d", value)


def assert_staged_continuation(opt, trial, score, *, integer=False):
    running = Optimizer.load(opt.save())
    try:
        assert running.configuration_matches(opt)
        statuses = [record.status for record in running.get_trials()]
        assert statuses[-1] == TrialStatus.RUNNING
        assert all(status == TrialStatus.COMPLETED for status in statuses[:-1])
        opt.tell(trial.id, score)
        running.tell(trial.id, score)
        completed = Optimizer.load(opt.save())
        try:
            assert completed.configuration_matches(opt)
            assert all(
                record.status == TrialStatus.COMPLETED
                for record in completed.get_trials()
            )
            expected = opt.ask()
            for restored in (running, completed):
                actual = restored.ask()
                assert actual.id == expected.id
                if integer:
                    assert actual.get_int("scale") == expected.get_int("scale")
                else:
                    assert bits(actual.get_float("scale")) == bits(
                        expected.get_float("scale")
                    )
                assert bits(actual.get_float("aux")) == bits(expected.get_float("aux"))
        finally:
            completed.close()
    finally:
        running.close()


@pytest.mark.parametrize(
    "low,high",
    [
        (1e-5, 1e-2),
        (5.0, 5000.0),
        (7.0, 7000.0),
        (50.0, 50000.0),
        (1000.0, 1e6),
        (1e18, 9e18),
        (2e19, 9e20),
        (1e21, 9e21),
        (1e15, 1e16),
        (0.5, 2.5),
        (1.0, 1000.0),
    ],
)
def test_sobol_log_float_checkpoint(low, high):
    space = (
        SearchSpace().add_float("scale", low, high, log=True).add_float("aux", 0.0, 1.0)
    )
    opt = Optimizer(space, sampler=Sampler.SOBOL, direction=Direction.MINIMIZE, seed=19)
    try:
        trial = opt.ask()
        value = trial.get_float("scale")
        assert math.isfinite(value) and low <= value <= high
        if (low, high) in ((0.5, 2.5), (1.0, 1000.0)):
            assert bits(value) == bits(low)
        assert_staged_continuation(opt, trial, 0.5)
    finally:
        opt.close()
        space.close()


@pytest.mark.parametrize("sampler", [Sampler.CMAES, Sampler.TPE])
@pytest.mark.parametrize("low,high", [(1e15, 1e16), (2e19, 9e20)])
def test_adaptive_log_float_checkpoint(sampler, low, high):
    space = (
        SearchSpace().add_float("scale", low, high, log=True).add_float("aux", 0.0, 1.0)
    )
    opt = Optimizer(space, sampler=sampler, direction=Direction.MINIMIZE, seed=19)
    try:
        for i in range(60):
            trial = opt.ask()
            value = trial.get_float("scale")
            assert math.isfinite(value) and low <= value <= high
            if i == 59:
                assert_staged_continuation(opt, trial, -math.log(value))
            else:
                opt.tell(trial.id, -math.log(value))
    finally:
        opt.close()
        space.close()


def test_log_integer_checkpoint():
    low, high = 10**15, 2**53 - 1
    space = (
        SearchSpace().add_int("scale", low, high, log=True).add_float("aux", 0.0, 1.0)
    )
    opt = Optimizer(space, sampler=Sampler.SOBOL, seed=19)
    try:
        trial = opt.ask()
        assert low <= trial.get_int("scale") <= high
        assert_staged_continuation(opt, trial, 0.5, integer=True)
    finally:
        opt.close()
        space.close()


@pytest.mark.parametrize("sampler", list(Sampler))
def test_t4_log_float_with_large_integer_tuple(sampler):
    """The accepted log clamp also applies in mixed tuple studies."""
    low, high = 2**52, 2**52 + 1
    space = (
        SearchSpace()
        .add_float("scale", 5.0, 5000.0, log=True)
        .add_sorted_tuple("window", 3, low, high, element_is_int=True)
        .add_float("aux", 0.0, 1.0)
    )
    opt = Optimizer(space, sampler=sampler, direction=Direction.MINIMIZE, seed=19)
    try:
        for index in range(13):
            trial = opt.ask()
            value = trial.get_float("scale")
            assert math.isfinite(value) and 5.0 <= value <= 5000.0
            values = [trial.get_float(f"window#{j}") for j in range(3)]
            assert all(low <= item <= high for item in values)
            assert values == sorted(values)
            if index != 12:
                opt.tell(trial.id, 0.5)
        restored = Optimizer.load(opt.save())
        try:
            assert restored.configuration_matches(opt)
            opt.tell(trial.id, 0.5)
            restored.tell(trial.id, 0.5)
            for _ in range(6):
                expected, actual = opt.ask(), restored.ask()
                assert actual.id == expected.id
                for name in ("scale", "window#0", "window#1", "window#2", "aux"):
                    assert bits(actual.get_float(name)) == bits(
                        expected.get_float(name)
                    )
                opt.tell(expected.id, 0.5)
                restored.tell(actual.id, 0.5)
        finally:
            restored.close()
    finally:
        opt.close()
        space.close()
