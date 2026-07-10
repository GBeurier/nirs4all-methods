# SPDX-License-Identifier: CECILL-2.1
"""Smoke test for the native HPO optimizer Python wrapper.

Run against a built libn4m:  N4M_LIB_PATH=/path/to/libn4m.so.2.1.0 \
    PYTHONPATH=bindings/python/src python bindings/python/tests/test_optimizer_smoke.py
(or via pytest with the same env).
"""
from __future__ import annotations

from n4m.model_selection.optimizer import (
    Direction,
    Metric,
    Optimizer,
    Pruner,
    Sampler,
    SearchSpace,
    TrialStatus,
)


def test_random_quadratic():
    space = SearchSpace()
    space.add_float("x", -5.0, 5.0)
    assert space.num_params() == 1
    opt = Optimizer(space, sampler=Sampler.RANDOM, direction=Direction.MINIMIZE, seed=123)
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
    opt = Optimizer(space, sampler=Sampler.TPE, direction=Direction.MINIMIZE,
                    n_startup_trials=15, seed=2)
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
    opt = Optimizer(space, pruner=Pruner.MEDIAN, direction=Direction.MINIMIZE,
                    n_startup_trials=2, seed=1)
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


if __name__ == "__main__":
    test_random_quadratic()
    test_tpe_mixed()
    test_median_pruner()
    test_determinism()
    print("optimizer python smoke: OK")
