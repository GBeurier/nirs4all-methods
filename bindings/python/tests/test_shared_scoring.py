# SPDX-License-Identifier: CECILL-2.1
"""Shared analytical scoring cases through the public C-backed scalar wrappers."""

import json
from pathlib import Path

import pytest
from n4m._errors import N4MError
from n4m._impl import metrics
from n4m._types import Status

FIXTURE_PATH = (
    Path(__file__).resolve().parents[3]
    / "parity/fixtures/shared_scoring_numerical.v1.json"
)
assert FIXTURE_PATH.is_file(), (
    f"Required shared scoring fixture missing: {FIXTURE_PATH}"
)
FIXTURE = json.loads(FIXTURE_PATH.read_text(encoding="utf-8"))
CASES = {case["name"]: case for case in FIXTURE["cases"]}
FINITE_CASES = [
    case
    for case in FIXTURE["cases"]
    if case["applicability"]["methods_scalar_abi"] == "supported"
]
assert {case["name"] for case in FINITE_CASES} == {
    "finite_scalar",
    "constant_perfect",
    "constant_imperfect",
    "one_row_perfect",
    "one_row_imperfect",
}
NONFINITE_CASES = [
    case for case in FIXTURE["cases"] if case.get("operation") == "nonfinite_input"
]
assert {(case["input_field"], case["input_token"]) for case in NONFINITE_CASES} == {
    (field, token)
    for field in ("y_true", "y_pred")
    for token in ("NaN", "+Inf", "-Inf")
}


@pytest.mark.parametrize(
    "metric_name", ["rmse", "mae", "bias", "sep", "rpd", "rpiq", "r2", "nrmse"]
)
@pytest.mark.parametrize("case", NONFINITE_CASES, ids=lambda case: case["name"])
def test_scalar_metrics_reject_nonfinite_inputs(metric_name: str, case: dict) -> None:
    """Every scalar rejects each bad value in either array at every position."""
    assert case["applicability"]["methods_scalar_abi"] == "rejects"
    base = CASES[case["base_case"]]
    bad = {"NaN": float("nan"), "+Inf": float("inf"), "-Inf": -float("inf")}[
        case["input_token"]
    ]
    for position in range(len(base[case["input_field"]])):
        inputs = {field: list(base[field]) for field in ("y_true", "y_pred")}
        inputs[case["input_field"]][position] = bad
        with pytest.raises(N4MError) as error:
            getattr(metrics, metric_name)(inputs["y_true"], inputs["y_pred"])
        assert error.value.status == Status.ERR_INVALID_ARGUMENT


@pytest.mark.parametrize("metric_name", ["rmse", "mae", "r2"])
@pytest.mark.parametrize("case", FINITE_CASES, ids=lambda case: case["name"])
def test_scalar_metrics_match_analytical_fixture(metric_name: str, case: dict) -> None:
    """Read expected values from the fixture without recomputing them."""
    assert case["kind"] == "regression" and case["n_outputs"] == 1
    actual = getattr(metrics, metric_name)(case["y_true"], case["y_pred"])
    expected = case["expected"][metric_name]
    tolerance = FIXTURE["tolerance"]
    assert abs(actual - expected) <= tolerance["absolute"] + tolerance[
        "relative"
    ] * abs(expected)


@pytest.mark.parametrize(
    ("metric_name", "y_true", "y_pred"),
    [
        ("rpd", [1.0, 1.0], [1.0, 1.0]),
        ("rpiq", [1.0, 1.0], [1.0, 1.0]),
        ("nrmse", [0.0, 0.0], [1.0, 1.0]),
    ],
)
def test_finite_degenerate_ratios_remain_infinite(
    metric_name: str, y_true: list[float], y_pred: list[float]
) -> None:
    """Input refusal must preserve intentional positive infinity outputs."""
    assert getattr(metrics, metric_name)(y_true, y_pred) == float("inf")
