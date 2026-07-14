# SPDX-License-Identifier: CECILL-2.1
"""Tests for the native ``finetune_estimator`` Python wrapper (MT8).

Run against a built libn4m:  N4M_LIB_PATH=/path/to/libn4m.so.2.2.0 \
    PYTHONPATH=bindings/python/src python -m pytest \
    bindings/python/tests/test_finetune_estimator.py
"""

from __future__ import annotations

import ctypes
from dataclasses import replace

import numpy as np
import pytest

from n4m import N4MError
from n4m.model_selection import optimizer as optimizer_module
from n4m.model_selection import (
    Algorithm,
    FinetuneResult,
    Metric,
    Optimizer,
    Sampler,
    SearchSpace,
    TrialRecord,
    TrialStatus,
    ValidationPlan,
    finetune_estimator,
)


def _make_plan(n: int, folds: int = 2) -> ValidationPlan:
    """Build a public owning round-robin k-fold validation plan."""
    return ValidationPlan.from_fold_ids(np.arange(n, dtype=np.int32) % folds)


def _dataset(n: int = 40, p: int = 5, seed: int = 0):
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n, p))
    beta = rng.standard_normal(p)
    Y = X @ beta + 0.01 * rng.standard_normal(n)
    return X, Y


def test_algorithm_enum_matches_c_abi():
    assert Algorithm.PLS_REGRESSION == 0
    assert Algorithm.PLS_CANONICAL == 1
    assert Algorithm.PLS_SVD == 2
    assert Algorithm.PLS_DA == 3
    assert Algorithm.OPLS == 4
    assert Algorithm.OPLS_DA == 5
    assert Algorithm.SPARSE_PLS == 6
    assert Algorithm.MB_PLS == 7
    assert Algorithm.LW_PLS == 8
    assert Algorithm.AOM_PLS == 9
    assert Algorithm.PCR == 10


@pytest.mark.parametrize(
    "algo",
    [
        Algorithm.PLS_REGRESSION,
        Algorithm.PLS_CANONICAL,
        Algorithm.PLS_SVD,
        Algorithm.OPLS,
        Algorithm.PCR,
        Algorithm.SPARSE_PLS,
    ],
)
def test_eligible_routes_return_finite_result(algo):
    X, Y = _dataset()
    plan = _make_plan(len(X))
    try:
        space = SearchSpace()
        space.add_int("n_components", 1, 3)
        if algo is Algorithm.SPARSE_PLS:
            space.add_float("sparsity_lambda", 0.0, 0.9)
        result = finetune_estimator(
            algo, X, Y, plan, space, n_trials=8, metric=Metric.RMSE, seed=7
        )
        assert isinstance(result, FinetuneResult)
        assert np.isfinite(result.best_score) and result.best_score >= 0.0
        assert result.metric is Metric.RMSE
        assert result.estimator is algo
        assert result.requested_trials == 8
        assert result.timed_out is False
        assert result.best_params["n_components"] in (1, 2, 3)
        assert isinstance(result.best_params["n_components"], int)
        if algo is Algorithm.SPARSE_PLS:
            assert "sparsity_lambda" in result.best_params
            assert 0.0 <= result.best_params["sparsity_lambda"] < 1.0
        else:
            assert "sparsity_lambda" not in result.best_params
        # Owning ordered trace usable after the native result is destroyed.
        assert len(result.trials) == 8
        assert all(isinstance(t, TrialRecord) for t in result.trials)
        assert [t.id for t in result.trials] == sorted(t.id for t in result.trials)
        assert any(t.status is TrialStatus.COMPLETED for t in result.trials)
        space.close()
    finally:
        plan.close()


def test_sparse_lambda_recorded_in_trace_and_params():
    X, Y = _dataset()
    plan = _make_plan(len(X))
    try:
        space = SearchSpace()
        space.add_int("n_components", 1, 3)
        space.add_float("sparsity_lambda", 0.0, 0.9)
        result = finetune_estimator(
            Algorithm.SPARSE_PLS, X, Y, plan, space, n_trials=12, seed=3
        )
        # best.sparsity_lambda is surfaced and matches the corresponding trace trial.
        best_lambda = result.best_params["sparsity_lambda"]
        assert 0.0 <= best_lambda < 1.0
        completed = [t for t in result.trials if t.score is not None]
        assert completed
        best_trial = min(completed, key=lambda t: t.score)
        assert best_trial.params["sparsity_lambda"] == pytest.approx(best_lambda)
        assert result.best_score == pytest.approx(best_trial.score)
        space.close()
    finally:
        plan.close()


def test_sparse_accepts_single_axis_subset():
    X, Y = _dataset()
    plan = _make_plan(len(X))
    try:
        # only sparsity_lambda: n_components falls back to its documented default.
        space = SearchSpace()
        space.add_float("sparsity_lambda", 0.0, 0.5)
        result = finetune_estimator(Algorithm.SPARSE_PLS, X, Y, plan, space, n_trials=4)
        assert result.estimator is Algorithm.SPARSE_PLS
        assert "sparsity_lambda" in result.best_params
        assert "n_components" not in result.best_params
        space.close()
    finally:
        plan.close()


@pytest.mark.parametrize(
    "algo",
    [
        Algorithm.PLS_DA,
        Algorithm.OPLS_DA,
        Algorithm.MB_PLS,
        Algorithm.LW_PLS,
        Algorithm.AOM_PLS,
        99,  # invalid enum value
    ],
)
def test_ineligible_estimators_rejected(algo):
    X, Y = _dataset(n=12, p=3)
    plan = _make_plan(len(X))
    try:
        space = SearchSpace()
        space.add_int("n_components", 1, 2)
        with pytest.raises(N4MError) as excinfo:
            finetune_estimator(algo, X, Y, plan, space, n_trials=2)
        assert excinfo.value.status == 10
        space.close()
    finally:
        plan.close()


def test_schema_violation_rejected():
    X, Y = _dataset()
    plan = _make_plan(len(X))
    try:
        # sparsity_lambda is not tunable for a dense route.
        space = SearchSpace()
        space.add_int("n_components", 1, 3)
        space.add_float("sparsity_lambda", 0.0, 0.5)
        with pytest.raises(N4MError) as excinfo:
            finetune_estimator(Algorithm.PLS_REGRESSION, X, Y, plan, space, n_trials=2)
        assert excinfo.value.status == 10
        space.close()
    finally:
        plan.close()


def test_marshals_listlike_x_and_1d_y():
    X, Y = _dataset(n=30, p=4)
    plan = _make_plan(len(X))
    try:
        space = SearchSpace()
        space.add_int("n_components", 1, 2)
        # X as a nested Python list, Y as a 1-D column-less vector.
        result = finetune_estimator(
            Algorithm.PLS_REGRESSION,
            X.tolist(),
            np.asarray(Y).reshape(-1),
            plan,
            space,
            n_trials=4,
            sampler=Sampler.RANDOM,
        )
        assert np.isfinite(result.best_score)
        space.close()
    finally:
        plan.close()


def test_bad_plan_handle_raises_type_error():
    X, Y = _dataset(n=10, p=3)
    space = SearchSpace()
    space.add_int("n_components", 1, 2)
    with pytest.raises(TypeError):
        finetune_estimator(Algorithm.PLS_REGRESSION, X, Y, object(), space, n_trials=2)
    plan = _make_plan(len(X))
    plan.close()
    with pytest.raises(RuntimeError, match="closed"):
        finetune_estimator(Algorithm.PLS_REGRESSION, X, Y, plan, space, n_trials=2)
    space.close()


def test_bad_space_and_context_raise_stable_type_errors():
    X, Y = _dataset(n=10, p=3)
    with ValidationPlan.from_fold_ids(np.arange(10) % 2) as plan:
        with pytest.raises(TypeError, match="space must be"):
            finetune_estimator(
                Algorithm.PLS_REGRESSION, X, Y, plan, object(), n_trials=2
            )
        space = SearchSpace().add_int("n_components", 1, 2)
        with pytest.raises(TypeError, match="context must be"):
            finetune_estimator(
                Algorithm.PLS_REGRESSION,
                X,
                Y,
                plan,
                space,
                n_trials=2,
                context=object(),
            )
        space.close()


def test_repeated_calls_do_not_leak_or_corrupt():
    # Ownership / cleanup: many back-to-back calls (each allocates and frees a
    # native result) stay correct and independent.
    X, Y = _dataset(n=24, p=4)
    plan = _make_plan(len(X))
    try:
        scores = []
        for _ in range(5):
            space = SearchSpace()
            space.add_int("n_components", 1, 3)
            result = finetune_estimator(
                Algorithm.PLS_REGRESSION, X, Y, plan, space, n_trials=6, seed=42
            )
            scores.append(result.best_score)
            space.close()
        # Deterministic seed => identical best score every time.
        assert all(s == pytest.approx(scores[0]) for s in scores)
    finally:
        plan.close()


@pytest.mark.parametrize("n_trials", [0, -1, 2**31])
def test_n_trials_range_checked_before_ctypes(n_trials):
    X, Y = _dataset(n=10, p=3)
    with ValidationPlan.from_fold_ids(np.arange(10) % 2) as plan:
        space = SearchSpace().add_int("n_components", 1, 2)
        with pytest.raises(ValueError, match="INT32_MAX"):
            finetune_estimator(
                Algorithm.PLS_REGRESSION, X, Y, plan, space, n_trials=n_trials
            )
        space.close()


def test_n_trials_bool_rejected_before_ctypes():
    X, Y = _dataset(n=10, p=3)
    with ValidationPlan.from_fold_ids(np.arange(10) % 2) as plan:
        space = SearchSpace().add_int("n_components", 1, 2)
        with pytest.raises(TypeError, match="n_trials must be an int"):
            finetune_estimator(
                Algorithm.PLS_REGRESSION, X, Y, plan, space, n_trials=True
            )
        space.close()


@pytest.mark.parametrize("cv", [2.5, True])
def test_validation_plan_rejects_non_integer_cv(cv):
    with pytest.raises(TypeError, match="cv must be an int or None"):
        ValidationPlan.from_fold_ids(np.arange(10) % 2, cv=cv)


@pytest.mark.parametrize("cv", [-1, 0, 1, 11])
def test_validation_plan_rejects_out_of_range_explicit_cv(cv):
    with pytest.raises(ValueError, match=r"cv must be in \[2, n_samples\]"):
        ValidationPlan.from_fold_ids(np.arange(10) % 2, cv=cv)


@pytest.mark.parametrize(
    "fold_ids",
    [
        np.array([0.0, 1.0, 0.0, 1.0]),
        np.array([False, True, False, True]),
    ],
)
def test_validation_plan_rejects_non_integer_fold_ids(fold_ids):
    with pytest.raises(TypeError, match="fold_ids must contain integers"):
        ValidationPlan.from_fold_ids(fold_ids)


def test_validation_plan_checks_fold_id_range_before_int32_cast():
    fold_ids = np.array([0, 1, 0, 2**32], dtype=np.uint64)
    with pytest.raises(ValueError, match="fold_ids must be in"):
        ValidationPlan.from_fold_ids(fold_ids)


def test_optional_result_scalar_only_swallows_missing_key(monkeypatch):
    def fake_missing(_result, _name, _value):
        return 1  # N4M_ERR_INVALID_ARGUMENT: documented missing-key status

    monkeypatch.setattr(
        optimizer_module.lib, "n4m_method_result_get_scalar", fake_missing
    )
    assert optimizer_module._maybe_result_scalar(None, "missing") is None

    def fake_internal(_result, _name, _value):
        return 255  # N4M_ERR_INTERNAL

    monkeypatch.setattr(
        optimizer_module.lib, "n4m_method_result_get_scalar", fake_internal
    )
    with pytest.raises(N4MError) as excinfo:
        optimizer_module._maybe_result_scalar(None, "broken")
    assert excinfo.value.status == 255


def test_finetune_decoder_rejects_non_boolean_timeout(monkeypatch):
    X, Y = _dataset(n=12, p=3)
    original = optimizer_module.lib.n4m_method_result_get_scalar

    def corrupt_timed_out(result, name, value):
        if name == b"timed_out":
            ctypes.cast(value, ctypes.POINTER(ctypes.c_double))[0] = 2.0
            return 0
        return original(result, name, value)

    monkeypatch.setattr(
        optimizer_module.lib, "n4m_method_result_get_scalar", corrupt_timed_out
    )
    with ValidationPlan.from_fold_ids(np.arange(12) % 2) as plan:
        space = SearchSpace().add_int("n_components", 1, 2)
        with pytest.raises(RuntimeError, match="invalid timed_out"):
            finetune_estimator(Algorithm.PLS_REGRESSION, X, Y, plan, space, n_trials=2)
        space.close()


def test_finetune_decoder_rejects_unmarked_partial_trace(monkeypatch):
    X, Y = _dataset(n=12, p=3)
    original = optimizer_module._decode_trials_result

    def truncate_trace(result, since_id):
        return original(result, since_id)[:-1]

    monkeypatch.setattr(optimizer_module, "_decode_trials_result", truncate_trace)
    with ValidationPlan.from_fold_ids(np.arange(12) % 2) as plan:
        space = SearchSpace().add_int("n_components", 1, 2)
        with pytest.raises(RuntimeError, match="timeout/trial count mismatch"):
            finetune_estimator(Algorithm.PLS_REGRESSION, X, Y, plan, space, n_trials=2)
        space.close()


def test_finetune_decoder_rejects_non_numeric_active_parameter(monkeypatch):
    X, Y = _dataset(n=12, p=3)
    original = optimizer_module._decode_trials_result

    def corrupt_active_parameter(result, since_id):
        trials = original(result, since_id)
        first = trials[0]
        params = dict(first.params)
        active_name = next(
            name for name, detail in first.param_details.items() if detail.active
        )
        params[active_name] = "not numeric"
        trials[0] = replace(first, params=params)
        return trials

    monkeypatch.setattr(
        optimizer_module, "_decode_trials_result", corrupt_active_parameter
    )
    with ValidationPlan.from_fold_ids(np.arange(12) % 2) as plan:
        space = SearchSpace().add_int("n_components", 1, 2)
        with pytest.raises(RuntimeError, match="non-numeric active parameter"):
            finetune_estimator(Algorithm.PLS_REGRESSION, X, Y, plan, space, n_trials=2)
        space.close()


def test_finetune_decodes_real_mixed_completed_failed_trace():
    X, Y = _dataset(n=24, p=3, seed=11)
    with ValidationPlan.from_fold_ids(np.arange(24) % 2) as plan:
        space = SearchSpace().add_int("n_components", 1, 6)
        selected_seed = None
        for seed in range(1, 2048):
            with Optimizer(space, seed=seed) as preview:
                first = preview.ask().get_int("n_components")
                second = preview.ask().get_int("n_components")
            if first <= X.shape[1] and second > X.shape[1]:
                selected_seed = seed
                break
        assert selected_seed is not None

        result = finetune_estimator(
            Algorithm.PLS_REGRESSION,
            X,
            Y,
            plan,
            space,
            n_trials=2,
            seed=selected_seed,
        )
        assert result.timed_out is False
        assert len(result.trials) == result.requested_trials == 2
        completed, failed = result.trials
        assert completed.status is TrialStatus.COMPLETED
        assert failed.status is TrialStatus.FAILED
        assert completed.score == result.best_score
        assert failed.score is None
        assert result.best_params == {
            name: completed.params[name]
            for name, detail in completed.param_details.items()
            if detail.active
        }
        space.close()


def test_validation_plan_from_splits_public_api():
    X, Y = _dataset(n=12, p=3)
    even = np.arange(0, 12, 2)
    odd = np.arange(1, 12, 2)
    with ValidationPlan.from_splits(12, [(odd, even), (even, odd)]) as plan:
        space = SearchSpace().add_int("n_components", 1, 2)
        result = finetune_estimator(
            Algorithm.PLS_REGRESSION, X, Y, plan, space, n_trials=2
        )
        assert np.isfinite(result.best_score)
        space.close()


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
