# SPDX-License-Identifier: CECILL-2.1 OR AGPL-3.0-or-later
"""Public Python-to-C ranking parity and missing-data boundary tests."""

import json
from pathlib import Path

import numpy as np
import pytest
from n4m import N4MError, ranking

_FIXTURE = json.loads(
    (
        Path(__file__).resolve().parents[3]
        / "parity"
        / "fixtures"
        / "ranking_reference_v1.json"
    ).read_text()
)


@pytest.mark.parametrize("case", _FIXTURE["cases"], ids=lambda c: c["name"])
@pytest.mark.parametrize("method", ["trueskill", "openskill", "bradley_terry"])
def test_independent_reference_parity(case, method):
    result = ranking.rank(case["scores"], present=case["present"], method=method)
    expected = np.asarray(case[method])
    actual = np.asarray(
        [[c["score"], c["uncertainty"]] for c in result["candidates"]]
        if method != "bradley_terry"
        else [c["score"] for c in result["candidates"]]
    )
    np.testing.assert_allclose(actual, expected, atol=2e-8, rtol=2e-9)
    assert result["convergence"]["converged"]
    json.dumps(result, allow_nan=False)


def test_friedman_reference_and_subset():
    case = _FIXTURE["friedman"]
    result = ranking.rank(case["scores"])
    assert result["inference"]["friedman_statistic"] == pytest.approx(case["statistic"])
    assert result["inference"]["friedman_p_value"] == pytest.approx(case["p_value"])
    scores = np.vstack([case["scores"], [100, 0, np.nan]])
    present = np.ones(scores.shape, dtype=bool)
    present[-1, -1] = False
    incomplete = ranking.rank(scores, present=present)
    assert incomplete["inference"] == result["inference"]
    assert incomplete["coverage"]["complete_blocks"] == 3
    assert (
        incomplete["candidates"][0]["average_rank"]
        != result["candidates"][0]["average_rank"]
    )


@pytest.mark.parametrize("method", ["elo", "bradley_terry", "trueskill", "openskill"])
def test_bridge_and_disconnected_global_ranks(method):
    x = [[1, 2, np.nan], [np.nan, 1, 2]]
    mask = [[1, 1, 0], [0, 1, 1]]
    result = ranking.rank(x, present=mask, method=method)
    assert result["coverage"]["connected_components"] == 1
    assert result["coverage"]["complete_blocks"] == 0
    assert [c["rank"] for c in result["candidates"]] == [1, 2, 3]
    separate = ranking.rank(
        [[1, 2, 0], [0, 0, 1]], present=[[1, 1, 0], [0, 0, 1]], method=method
    )
    assert separate["coverage"]["connected_components"] == 2
    assert all(c["rank"] is None for c in separate["candidates"])


def test_strided_layout_direction_and_explicit_absence():
    x = np.arange(24.0).reshape(4, 6)[:, ::2]
    for view in (x, np.asfortranarray(x), x[::-1]):
        assert (
            ranking.rank(view)["candidates"]
            == ranking.rank(np.array(view))["candidates"]
        )
    result = ranking.rank([[np.nan, 2, 1]], present=[[0, 1, 1]], higher_is_better=True)
    assert result["candidates"][0]["score"] is None
    assert result["candidates"][1]["average_rank"] == 1
    assert result["score_direction"] == "lower_is_better"
    with pytest.raises(N4MError):
        ranking.rank([[np.nan, 1]])
    with pytest.raises(ValueError):
        ranking.rank([[1, 2]], present=[[1, 256]])


def test_nontransitive_tolerance_and_all_ties():
    result = ranking.rank([[0, 0.75, 1.5]], tie_tolerance=1)
    assert [c["average_rank"] for c in result["candidates"]] == [1.5, 1.5, 3]
    assert [c["ties"] for c in result["candidates"]] == [1, 2, 1]
    ties = ranking.rank([[1, 1], [1, 1]])
    assert [c["rank"] for c in ties["candidates"]] == [1.5, 1.5]
    assert ties["inference"]["friedman_statistic"] is None
    assert ties["inference"]["friedman_p_value"] is None


def test_refuse_unsupported_and_report_nonconvergence():
    with pytest.raises(ValueError):
        ranking.rank([[1, 2]], method="trueskill2")
    with pytest.raises(ValueError):
        ranking.rank([[1, 2]], team_weights=[1, 2])
    with pytest.raises(N4MError):
        ranking.rank([[1, 2]], beta=1e300)
    result = ranking.rank([[1, 2]], method="bradley_terry", max_iterations=1)
    assert not result["convergence"]["converged"]
    assert all(c["rank"] is None for c in result["candidates"])


def test_relative_gain_variance_units_and_dl_reference():
    result = ranking.paired_relative_gain([2, 4, 10], [1, 3, 10])
    assert result["mean_gain"] == 0.25
    assert result["sample_sd"] == 0.25
    assert result["standard_error"] is None
    meta = ranking.paired_relative_gain(
        [2, 4, 10], [1, 3, 10], variances=[0.01, 0.04, 0.09]
    )
    # Independently computed inverse-variance DL formulas (fixed weights 100,25,100/9).
    assert meta["pooled_gain"] == pytest.approx(0.34741314101695414)
    assert meta["tau_squared"] == pytest.approx(0.023035714285714284)
    assert meta["standard_error"] == pytest.approx(0.13486320236692917)
    for baseline in ([0, 2], [-1, 2]):
        with pytest.raises(N4MError):
            ranking.paired_relative_gain(baseline, [1, 1])
    with pytest.raises(N4MError):
        ranking.paired_relative_gain([1, 2], [1, 1], variances=[0, 0.01])
    json.dumps(meta, allow_nan=False)


def test_relative_gain_preserves_near_equal_large_positive_losses():
    baseline = np.float64(1e308)
    candidate = np.nextafter(baseline, 0)
    result = ranking.paired_relative_gain([baseline], [candidate])
    assert result["mean_gain"] == (baseline - candidate) / baseline
    assert result["sample_sd"] is None


@pytest.mark.parametrize(
    "variances, expected_se",
    [([1e-16, 1], 1e-8), ([1e-16, 1e-16], np.sqrt(5e-17)), ([1e-300, 1e300], 1e-150)],
)
def test_dl_dominant_weight_avoids_cancellation(variances, expected_se):
    # Identical effects imply exactly Q=tau²=I²=0 for arbitrary positive variances.
    # Old sw - sw2/sw cancelled to zero for the first finite-input case.
    try:
        result = ranking.paired_relative_gain([2, 2], [1, 1], variances=variances)
    except N4MError as error:
        if np.finfo(np.longdouble).minexp == np.finfo(
            np.float64
        ).minexp and variances == [1e-300, 1e300]:
            assert error.status_name == "UNSUPPORTED"
            return
        raise
    assert result["pooled_gain"] == 0.5
    assert result["tau_squared"] == result["q"] == result["i_squared"] == 0
    assert result["standard_error"] == pytest.approx(expected_se, rel=2e-15, abs=0)
    json.dumps(result, allow_nan=False)


def test_dl_subnormal_variances_preserve_representable_standard_error():
    variance = np.nextafter(0.0, 1.0)
    try:
        result = ranking.paired_relative_gain(
            [2, 2], [1, 1], variances=[variance, variance]
        )
    except N4MError as error:
        if np.finfo(np.longdouble).minexp == np.finfo(np.float64).minexp:
            assert error.status_name == "UNSUPPORTED"
            return
        raise
    # Compute square root before the division to avoid underflow in the oracle.
    expected = np.sqrt(variance) / np.sqrt(2.0)
    assert result["standard_error"] > 0
    assert result["standard_error"] == pytest.approx(expected, rel=2e-15, abs=0)
