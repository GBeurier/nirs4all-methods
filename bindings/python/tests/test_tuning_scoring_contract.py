# SPDX-License-Identifier: CECILL-2.1 OR AGPL-3.0-or-later
"""Bounded PLS_REGRESSION objective qualification through production native APIs.

The multi-target producer shares the fit engine; this is no independent PLS-fit
oracle. Constant-target producer refusal is recorded separately from parity.
"""

from __future__ import annotations

import json

import numpy as np
import pytest
from n4m import N4MError
from n4m._types import Status
from n4m.model_selection import (
    Algorithm,
    Direction,
    EvalMode,
    Metric,
    Sampler,
    SearchSpace,
    TrialStatus,
    ValidationPlan,
    finetune_estimator,
)
from n4m.model_selection.sweep import pls_cross_validate

COMPONENTS = (1, 2, 3)
N_TRIALS = 16
REGRESSION_METRICS = (Metric.RMSE, Metric.MSE, Metric.MAE, Metric.R2)
# Separate native moment/materialized arithmetic and NumPy reductions can round
# differently. This is far below the fixture's competing aggregation gaps.
RTOL = 1e-10
ATOL = 1e-12
EPS = np.finfo(np.float64).eps  # C++ DBL_EPSILON, absolute (not 1e-12).


def _data(n_targets: int = 1) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(908)
    x = rng.normal(size=(37, 5)) * [1, 3, 0.4, 2, 5]
    folds = np.repeat(np.arange(3, dtype=np.int32), [5, 11, 21])
    rng.shuffle(folds)
    y = (
        1.8 * x[:, 0]
        - 0.7 * x[:, 1]
        + 0.2 * x[:, 4]
        + rng.normal(size=37) * (0.3 + folds * 0.6)
    )[:, None]
    if n_targets == 2:
        y = np.column_stack(
            (y, 70 + 25 * (-x[:, 0] + 0.4 * x[:, 3] + rng.normal(size=37)))
        )
    return x, y, folds


def _oof(x: np.ndarray, y: np.ndarray, folds: np.ndarray, component: int) -> dict:
    cv = int(folds.max()) + 1
    result = pls_cross_validate(
        x,
        y,
        cv=cv,
        fold_ids=folds,
        component_grid=(component,),
        center_x=True,
        scale_x=True,
        center_y=True,
        scale_y=True,
        score_only=False,
    )
    assert result["cv"] == cv
    assert result["n_candidates"] == 1
    assert result["n_samples"] == len(x)
    assert result["n_features"] == x.shape[1]
    assert result["n_targets"] == y.shape[1]
    assert result["score_only"] == 0
    assert result["selected_candidate_id"] == 0
    assert result["selected_head_id"] == 1  # Native sweep head 1 = PLS.
    assert result["selected_param"] == component
    assert result["candidate_scores"].shape == (1, 4)
    np.testing.assert_array_equal(result["candidate_scores"][0, :3], [0, 1, component])
    assert result["oof_predictions"].shape == y.shape
    assert result["fold_ids"].shape == folds.shape
    np.testing.assert_array_equal(result["fold_ids"], folds)
    single = y.shape[1] == 1
    assert result["n_pls_moment_candidates"] == int(single)
    assert result["n_pls_moment_cv_fits"] == (cv if single else 0)
    assert result["n_pls_materialized_cv_fits"] == (0 if single else cv)
    assert np.isfinite(result["oof_predictions"]).all()
    return result


def _expected(y: np.ndarray, oof: np.ndarray) -> dict[Metric, float]:
    """Test arithmetic over real OOF entries, never a production evaluator."""
    error = oof - y
    sse = float(np.sum(error**2))
    tss = float(np.sum((y - y.mean()) ** 2))  # One grand mean over n*q.
    rmse = float(np.sqrt(sse / y.size))
    return {
        Metric.RMSE: rmse,
        Metric.MSE: rmse * rmse,
        Metric.MAE: float(np.mean(np.abs(error))),
        Metric.R2: (float(sse <= EPS) if tss <= EPS else 1 - sse / tss),
    }


def _tune(x: np.ndarray, y: np.ndarray, folds: np.ndarray, metric: Metric):
    cv = int(folds.max()) + 1
    with (
        ValidationPlan.from_fold_ids(folds, cv=cv) as plan,
        SearchSpace() as space,
    ):
        assert plan.n_folds == cv
        assert plan.n_samples == len(x)
        space.add_int("n_components", 1, 3)
        # No enqueue seam exists on this convenience driver. Fixed native RANDOM
        # seed + sufficient draws gives reproducible coverage, asserted below.
        result = finetune_estimator(
            Algorithm.PLS_REGRESSION,
            x,
            y,
            plan,
            space,
            n_trials=N_TRIALS,
            sampler=Sampler.RANDOM,
            seed=7,
            direction=Direction.AUTO,
            metric=metric,
            eval_mode=EvalMode.MEAN,
        )
    assert result.metric is metric
    assert result.estimator is Algorithm.PLS_REGRESSION
    assert result.requested_trials == N_TRIALS
    assert not result.timed_out
    assert len(result.trials) == N_TRIALS
    assert [t.id for t in result.trials] == sorted({t.id for t in result.trials})
    assert all(t.status is TrialStatus.COMPLETED for t in result.trials)
    assert all(t.error is None and np.isfinite(t.score) for t in result.trials)
    assert {t.params["n_components"] for t in result.trials} == set(COMPONENTS)
    return result


def _check_trials(result, expected: dict[int, dict[Metric, float]]) -> dict:
    for trial in result.trials:
        assert trial.score == pytest.approx(
            expected[trial.params["n_components"]][result.metric], rel=RTOL, abs=ATOL
        )
    choose = max if result.metric is Metric.R2 else min
    # Python min/max retain the first item on EXACT ties, like native better().
    oracle_best = choose(
        result.trials, key=lambda t: expected[t.params["n_components"]][result.metric]
    )
    actual_best = choose(result.trials, key=lambda t: t.score)
    assert result.best_score == actual_best.score
    assert result.best_params == actual_best.params == oracle_best.params
    assert result.best_score == pytest.approx(
        expected[oracle_best.params["n_components"]][result.metric], rel=RTOL, abs=ATOL
    )
    ties = [t for t in result.trials if t.score == actual_best.score]
    assert actual_best.id == ties[0].id
    assert result.best_params == ties[0].params
    return {
        "metric": result.metric.name,
        "trials": [
            {"id": t.id, "component": t.params["n_components"], "score": t.score}
            for t in result.trials
        ],
        "expected": {k: v[result.metric] for k, v in expected.items()},
        "best": result.best_params,
        "first_best_id": actual_best.id,
        "exact_best_tie_ids": [t.id for t in ties],
    }


@pytest.fixture(scope="module", params=(1, 2), ids=("single", "unequal_scale_multi"))
def cohort(request):
    x, y, folds = _data(request.param)
    assert np.bincount(folds).tolist() == [5, 11, 21]
    if request.param == 2:
        assert np.std(y[:, 1]) > 10 * np.std(y[:, 0])
    producers = {k: _oof(x, y, folds, k) for k in COMPONENTS}
    return x, y, folds, producers


@pytest.mark.parametrize("metric", REGRESSION_METRICS, ids=lambda m: m.name)
def test_pls_tuning_objectives_and_auto_winner(cohort, metric, record_property):
    x, y, folds, producers = cohort
    expected = {}
    fold_means = {}
    for k, producer in producers.items():
        pred = producer["oof_predictions"]
        expected[k] = _expected(y, pred)
        assert expected[k][Metric.RMSE] > 0.1  # Finite, nontrivial errors.
        fold_means[k] = float(
            np.mean(
                [_expected(y[folds == f], pred[folds == f])[metric] for f in range(3)]
            )
        )
        assert not np.isclose(expected[k][metric], fold_means[k], rtol=RTOL, atol=ATOL)
        if y.shape[1] == 2:
            macro_r2 = np.mean(
                [_expected(y[:, j], pred[:, j])[Metric.R2] for j in range(2)]
            )
            column_tss = np.sum((y - y.mean(axis=0)) ** 2)
            column_r2 = 1 - np.sum((pred - y) ** 2) / column_tss
            assert abs(expected[k][Metric.R2] - macro_r2) > 0.1
            assert abs(expected[k][Metric.R2] - column_r2) > 0.1
    result = _tune(x, y, folds, metric)
    evidence = _check_trials(result, expected)
    evidence["unweighted_fold_means"] = fold_means
    evidence["producer_routes"] = {
        k: {
            name: p[name]
            for name in (
                "n_pls_moment_candidates",
                "n_pls_moment_cv_fits",
                "n_pls_materialized_cv_fits",
            )
        }
        for k, p in producers.items()
    }
    record_property("qualification", json.dumps(evidence))


@pytest.mark.parametrize("case", ("small_sse", "large_sse", "above_epsilon"))
def test_solver_admitted_nearconstant_r2(case, record_property):
    x, y, folds = _data()
    y *= 1e-8 if case == "above_epsilon" else 1e-10
    if case == "large_sse":
        x[0, 0] = 1e4  # Real held-out leverage: small TSS, SSE above epsilon.
    tss = float(np.sum((y - y.mean()) ** 2))
    assert 0 < tss <= 1e-12
    assert (tss > EPS) == (case == "above_epsilon")
    expected = {}
    sses = {}
    for k in COMPONENTS:
        pred = _oof(x, y, folds, k)["oof_predictions"]
        sses[k] = float(np.sum((pred - y) ** 2))
        assert 0 < sses[k] < np.inf
        assert (sses[k] > EPS) == (case != "small_sse")
        expected[k] = _expected(y, pred)
        if case != "above_epsilon":
            assert expected[k][Metric.R2] == (0.0 if case == "large_sse" else 1.0)
    evidence = _check_trials(_tune(x, y, folds, Metric.R2), expected)
    evidence.update(tss=tss, sse=sses, dbl_epsilon=EPS)
    record_property("qualification", json.dumps(evidence))


def test_constant_r2_finetune_and_separate_producer_refusal(record_property):
    x, y, folds = _data()
    y.fill(0.0)
    refusals = []
    for k in COMPONENTS:
        with pytest.raises(N4MError) as exc:
            _oof(x, y, folds, k)
        assert exc.value.status == Status.ERR_NUMERICAL_FAILURE
        refusals.append(
            {"component": k, "status": exc.value.status, "error": str(exc.value)}
        )
    # The finetune fit route admits this cohort; its constant-zero prediction
    # has TSS=SSE=0. The refused producer provides NO OOF parity evidence here.
    result = _tune(x, y, folds, Metric.R2)
    assert all(t.score == 1.0 for t in result.trials)
    assert result.best_params == result.trials[0].params
    assert result.best_score == 1.0
    record_property("producer_refused_not_parity", json.dumps(refusals))
    record_property(
        "constant_finetune",
        json.dumps(
            {
                "scores": [t.score for t in result.trials],
                "components": [t.params["n_components"] for t in result.trials],
                "best": result.best_params,
            }
        ),
    )


@pytest.mark.parametrize(
    "metric",
    (Metric.ACCURACY, Metric.BALANCED_ACCURACY, Metric.F1, Metric.LOGLOSS),
    ids=lambda m: m.name,
)
def test_classification_metric_refused_not_numerical_parity(metric, record_property):
    x, y, folds = _data()
    with pytest.raises(N4MError) as exc:
        _tune(x, y, folds, metric)
    assert exc.value.status == Status.ERR_NOT_IMPLEMENTED
    record_property(
        "unsupported_not_parity",
        json.dumps(
            {
                "metric": metric.name,
                "status": exc.value.status,
                "error": str(exc.value),
            }
        ),
    )
