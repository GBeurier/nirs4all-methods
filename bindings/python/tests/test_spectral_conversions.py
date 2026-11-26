"""Spectral conversions against external references (frozen fixture).

``parity/spectral_conversions/generate.py`` computed every expected output
with an independent library (photobiology, mdatools, astropy, scipy, specdal,
spectrolab, prospectr, asdreader); this test replays the fixture through the
generic roles without those libraries installed.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from n4m import roles
from n4m.roles._base import _REGISTRY

FIXTURE = (
    Path(__file__).resolve().parents[3]
    / "parity"
    / "fixtures"
    / "spectral_conversions_external_v1.json"
)
CASES = json.loads(FIXTURE.read_text(encoding="utf-8"))["cases"]


def _unhex(values):
    if isinstance(values, list) and values and isinstance(values[0], list):
        return np.array([[float.fromhex(v) for v in row] for row in values])
    return np.array([float.fromhex(v) for v in values])


def _params(raw):
    return {
        key: [float.fromhex(v) for v in value] if isinstance(value, list) else value
        for key, value in raw.items()
    }


@pytest.mark.parametrize("case", CASES, ids=lambda case: case["name"])
def test_matches_external_reference(case):
    est = _REGISTRY[case["method_id"]](**_params(case["params"]))
    X = _unhex(case["X"])
    fit = {} if case["axis"] is None else {"axis": _unhex(case["axis"])}
    got = est.fit(X, **fit).transform(X)
    np.testing.assert_allclose(
        got,
        _unhex(case["expected"]),
        rtol=case["rtol"],
        atol=case["atol"],
        err_msg=case["reference"],
    )
    if "expected_axis" in case:
        axis = np.asarray(est.fit_result_["axis"]).ravel()
        np.testing.assert_allclose(
            axis, _unhex(case["expected_axis"]), rtol=case["axis_rtol"]
        )


def test_every_conversion_has_an_external_reference():
    covered = {case["method_id"] for case in CASES}
    assert covered == {
        "preprocessing.signal_conversion.to_absorbance",
        "preprocessing.signal_conversion.from_absorbance",
        "preprocessing.signal_conversion.kubelka_munk",
        "preprocessing.signal_conversion.from_kubelka_munk",
        "preprocessing.signal_conversion.percent_to_fraction",
        "preprocessing.signal_conversion.fraction_to_percent",
        "preprocessing.signal_conversion.reflectance_from_references",
        "preprocessing.signal_conversion.splice_correction",
        "preprocessing.resampling.resampler",
        "preprocessing.resampling.axis_unit_conversion",
    }


def test_fit_result_is_not_state():
    est = roles.AxisUnitConversion(source_unit="nm", target_unit="cm-1")
    X = np.array([[1.0, 2.0]])
    est.fit(X, axis=np.array([1000.0, 2000.0]))
    np.testing.assert_array_equal(est.fit_result_["axis"], [[5000.0, 10000.0]])
    np.testing.assert_array_equal(est.fit_result_["source_columns"], [1, 0])
    # The converted axis gives meaning to the output columns: an import rebuilds it.
    replay = type(est).from_n4me(est.to_n4me())
    np.testing.assert_array_equal(replay.fit_result_["axis"], [[5000.0, 10000.0]])
    assert roles.SNV().fit(X).fit_result_ is None
