# SPDX-License-Identifier: CECILL-2.1
"""Gate 1 (binding_parity) tests for the chemometrics4all Python binding.

For each of the twenty Tier-2 operators, we:

1. Load the matching ``parity/fixtures/<op>_v1.json`` fixture.
2. Reconstruct the input matrix from ``input_hex`` (IEEE 754 binary64
   big-endian hex strings, eight bytes per element).
3. Run the binding with the same constructor parameters as the fixture's
   first case.
4. Decode the expected output from ``output_hex`` (or its op-specific
   equivalent).
5. Compare the two arrays via :func:`parity.binding_parity` with the Gate 1
   tolerance of ``1e-6``.

A failure means the binding diverged from libc4a's bit-equivalent reference
— widening the tolerance is NOT the fix.
"""
from __future__ import annotations

import json
import struct
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from chemometrics4all import (
    EMSC,
    LSNV,
    MSC,
    RNV,
    SNV,
    AirPLS,
    ArPLS,
    AreaNormalization,
    AsLS,
    BEADS,
    BaselineCenter,
    CropTransformer,
    Derivate,
    Detrend,
    EPO,
    FirstDerivative,
    FCKStaticTransformer,
    FlexiblePCA,
    FlexibleSVD,
    FractionToPercent,
    FromAbsorbance,
    Gaussian,
    GaussianAdditiveNoise,
    Haar,
    IAsLS,
    IModPoly,
    IntegerKBinsDiscretizer,
    KBinsStratifiedSplitter,
    KennardStoneSplitter,
    KubelkaMunk,
    LogTransform,
    ModPoly,
    Normalize,
    NorrisWilliams,
    OSC,
    PCG64,
    PercentToFraction,
    RangeDiscretizer,
    ResampleTransformer,
    Resampler,
    RollingBall,
    SNIP,
    SavitzkyGolay,
    SecondDerivative,
    SimpleScale,
    SPXYSplitter,
    ToAbsorbance,
    Wavelet,
    WaveletDenoise,
    WaveletFeatures,
    WaveletPCA,
    WaveletSVD,
    XOutlierFilter,
    YOutlierFilter,
    bias,
    hotelling_t2,
    mae,
    nrmse,
    q_residuals,
    r2,
    rmse,
    rpd,
    rpiq,
    sep,
    transfer_metrics,
)
from parity.binding_parity import binding_parity

TOLERANCE = 1e-6


# ---------------------------------------------------------------------------
# Fixture helpers
# ---------------------------------------------------------------------------


def _decode_hex_array(hex_list: list[str]) -> np.ndarray:
    """Decode IEEE 754 binary64 big-endian hex strings into a flat float64 array."""
    out = np.empty(len(hex_list), dtype=np.float64)
    for i, h in enumerate(hex_list):
        out[i] = struct.unpack(">d", bytes.fromhex(h))[0]
    return out


def _decode_scalar(hex_value: str) -> float:
    return struct.unpack(">d", bytes.fromhex(hex_value))[0]


def _decode_matrix(hex_list: list[str], rows: int, cols: int) -> np.ndarray:
    flat = _decode_hex_array(hex_list)
    return flat.reshape(rows, cols).copy(order="C")


def _load_fixture(fixtures_dir: Path, name: str) -> dict[str, Any]:
    path = fixtures_dir / f"{name}_v1.json"
    if not path.exists():
        pytest.skip(f"fixture {path} missing")
    with path.open() as f:
        return json.load(f)


def _assert_parity(pred: np.ndarray, expected: np.ndarray, op_name: str,
                   tolerance: float = TOLERANCE) -> None:
    """Run binding_parity and raise with a friendly message on failure."""
    res = binding_parity(pred, expected, tolerance=tolerance)
    if not res.ok:
        msg = (
            f"binding_parity FAIL for {op_name}: {res.detail}, "
            f"shape_mismatch={res.shape_mismatch}, "
            f"max_abs_diff={res.max_abs_diff}, max_rel_diff={res.max_rel_diff}"
        )
        # Add a few sample pairs for triage.
        if pred.shape == expected.shape and pred.size > 0:
            diff = np.abs(pred - expected).ravel()
            worst = int(np.argmax(diff))
            flat_p = pred.ravel()
            flat_e = expected.ravel()
            msg += (
                f"\n  worst element: idx={worst}, pred={flat_p[worst]!r}, "
                f"ref={flat_e[worst]!r}, |diff|={diff[worst]!r}"
            )
        pytest.fail(msg)


def _assert_parity_allow_nonfinite(
    pred: np.ndarray, expected: np.ndarray, op_name: str
) -> None:
    """Parity helper for diagnostics where NaN/Inf is part of the contract."""
    pred = np.asarray(pred)
    expected = np.asarray(expected)
    if pred.shape != expected.shape:
        _assert_parity(pred, expected, op_name)
        return
    same_nan = np.isnan(pred) == np.isnan(expected)
    same_posinf = np.isposinf(pred) == np.isposinf(expected)
    same_neginf = np.isneginf(pred) == np.isneginf(expected)
    if not (np.all(same_nan) and np.all(same_posinf) and np.all(same_neginf)):
        _assert_parity(pred, expected, op_name)
        return
    finite = np.isfinite(pred) & np.isfinite(expected)
    _assert_parity(pred[finite], expected[finite], op_name)


# ---------------------------------------------------------------------------
# Preprocessing — stateless
# ---------------------------------------------------------------------------


def test_snv_binding_parity(fixtures_dir):
    fx = _load_fixture(fixtures_dir, "snv")
    X = _decode_matrix(fx["input_hex"], fx["rows"], fx["cols"])
    for case in fx["cases"]:
        params = case["params"]
        snv = SNV(
            with_mean=params["with_mean"],
            with_std=params["with_std"],
            ddof=params["ddof"],
        )
        pred = snv.fit_transform(X)
        expected = _decode_matrix(case["output_hex"], fx["rows"], fx["cols"])
        _assert_parity(pred, expected, f"SNV[{case['name']}]")


def test_lsnv_binding_parity(fixtures_dir):
    fx = _load_fixture(fixtures_dir, "lsnv")
    X = _decode_matrix(fx["input_hex"], fx["rows"], fx["cols"])
    for case in fx["cases"]:
        params = case["params"]
        op = LSNV(
            window=params["window"],
            pad_mode=params["pad_mode"],
            constant_value=params["constant_value"],
        )
        pred = op.fit_transform(X)
        expected = _decode_matrix(case["output_hex"], fx["rows"], fx["cols"])
        _assert_parity(pred, expected, f"LSNV[{case['name']}]")


def test_rnv_binding_parity(fixtures_dir):
    fx = _load_fixture(fixtures_dir, "rnv")
    X = _decode_matrix(fx["input_hex"], fx["rows"], fx["cols"])
    for case in fx["cases"]:
        params = case["params"]
        op = RNV(
            with_center=params["with_center"],
            with_scale=params["with_scale"],
            k=params["k"],
        )
        pred = op.fit_transform(X)
        expected = _decode_matrix(case["output_hex"], fx["rows"], fx["cols"])
        _assert_parity(pred, expected, f"RNV[{case['name']}]")


def test_area_normalization_binding_parity(fixtures_dir):
    fx = _load_fixture(fixtures_dir, "area_norm")
    X = _decode_matrix(fx["input_hex"], fx["rows"], fx["cols"])
    for case in fx["cases"]:
        op = AreaNormalization(method=case["params"]["method"])
        pred = op.fit_transform(X)
        expected = _decode_matrix(case["output_hex"], fx["rows"], fx["cols"])
        _assert_parity(pred, expected, f"AreaNormalization[{case['name']}]")


def test_normalize_binding_parity(fixtures_dir):
    fx = _load_fixture(fixtures_dir, "normalize")
    X = _decode_matrix(fx["input_hex"], fx["rows"], fx["cols"])
    for case in fx["cases"]:
        params = case["params"]
        op = Normalize(
            feature_min=params["feature_min"],
            feature_max=params["feature_max"],
        )
        pred = op.fit_transform(X)
        expected = _decode_matrix(case["output_hex"], fx["rows"], fx["cols"])
        _assert_parity(pred, expected, f"Normalize[{case['name']}]")


def test_simple_scale_binding_parity(fixtures_dir):
    fx = _load_fixture(fixtures_dir, "simple_scale")
    X = _decode_matrix(fx["input_hex"], fx["rows"], fx["cols"])
    for case in fx["cases"]:
        pred = SimpleScale().fit_transform(X)
        expected = _decode_matrix(case["output_hex"], fx["rows"], fx["cols"])
        _assert_parity(pred, expected, f"SimpleScale[{case['name']}]")


def test_log_transform_binding_parity(fixtures_dir):
    fx = _load_fixture(fixtures_dir, "log_transform")
    X = _decode_matrix(fx["input_hex"], fx["rows"], fx["cols"])
    for case in fx["cases"]:
        params = case["params"]
        op = LogTransform(
            base=params["base"],
            offset=params["offset"],
            auto_offset=params["auto_offset"],
            min_value=params["min_value"],
        )
        pred = op.fit_transform(X)
        expected = _decode_matrix(case["output_hex"], fx["rows"], fx["cols"])
        _assert_parity(pred, expected, f"LogTransform[{case['name']}]")


def test_to_absorbance_binding_parity(fixtures_dir):
    fx = _load_fixture(fixtures_dir, "to_absorbance")
    X_frac = _decode_matrix(fx["input_hex"], fx["rows"], fx["cols"])
    for case in fx["cases"]:
        params = case["params"]
        # Per the fixture generator (parity/python_generator/scripts/
        # generate_phase7_fixtures.py), is_percent=True cases are run on the
        # fractional input scaled by 100 — the C test does the same scaling
        # before calling the operator.
        X = X_frac * 100.0 if params["is_percent"] else X_frac
        op = ToAbsorbance(
            is_percent=params["is_percent"],
            epsilon=params["epsilon"],
            clip_negative=params.get("clip_negative", True),
        )
        pred = op.fit_transform(X)
        expected = _decode_matrix(case["output_hex"], fx["rows"], fx["cols"])
        _assert_parity(pred, expected, f"ToAbsorbance[{case['name']}]")


def test_from_absorbance_binding_parity(fixtures_dir):
    fx = _load_fixture(fixtures_dir, "from_absorbance")
    X = _decode_matrix(fx["input_hex"], fx["rows"], fx["cols"])
    for case in fx["cases"]:
        op = FromAbsorbance(is_percent=case["params"]["is_percent"])
        pred = op.fit_transform(X)
        expected = _decode_matrix(case["output_hex"], fx["rows"], fx["cols"])
        _assert_parity(pred, expected, f"FromAbsorbance[{case['name']}]")


def test_percent_to_fraction_binding_parity(fixtures_dir):
    fx = _load_fixture(fixtures_dir, "pct_to_frac")
    X = _decode_matrix(fx["input_hex"], fx["rows"], fx["cols"])
    for case in fx["cases"]:
        pred = PercentToFraction().fit_transform(X)
        expected = _decode_matrix(case["output_hex"], fx["rows"], fx["cols"])
        _assert_parity(pred, expected, f"PercentToFraction[{case['name']}]")


def test_fraction_to_percent_binding_parity(fixtures_dir):
    fx = _load_fixture(fixtures_dir, "frac_to_pct")
    X = _decode_matrix(fx["input_hex"], fx["rows"], fx["cols"])
    for case in fx["cases"]:
        pred = FractionToPercent().fit_transform(X)
        expected = _decode_matrix(case["output_hex"], fx["rows"], fx["cols"])
        _assert_parity(pred, expected, f"FractionToPercent[{case['name']}]")


def test_kubelka_munk_binding_parity(fixtures_dir):
    fx = _load_fixture(fixtures_dir, "kubelka_munk")
    X_frac = _decode_matrix(fx["input_hex"], fx["rows"], fx["cols"])
    for case in fx["cases"]:
        params = case["params"]
        # Same percent-scaling convention as ToAbsorbance (see generator).
        X = X_frac * 100.0 if params["is_percent"] else X_frac
        op = KubelkaMunk(is_percent=params["is_percent"], epsilon=params["epsilon"])
        pred = op.fit_transform(X)
        expected = _decode_matrix(case["output_hex"], fx["rows"], fx["cols"])
        _assert_parity(pred, expected, f"KubelkaMunk[{case['name']}]")


# ---------------------------------------------------------------------------
# Preprocessing — derivative / smoothing
# ---------------------------------------------------------------------------


def test_savgol_binding_parity(fixtures_dir):
    fx = _load_fixture(fixtures_dir, "savgol")
    X = _decode_matrix(fx["input_hex"], fx["rows"], fx["cols"])
    for case in fx["cases"]:
        params = case["params"]
        op = SavitzkyGolay(
            window_length=params["window_length"],
            polyorder=params["polyorder"],
            deriv=params["deriv"],
            delta=params["delta"],
            mode=params["mode"],
            cval=params["cval"],
        )
        pred = op.fit_transform(X)
        expected = _decode_matrix(case["output_hex"],
                                  case.get("output_rows", fx["rows"]),
                                  case.get("output_cols", fx["cols"]))
        _assert_parity(pred, expected, f"SavitzkyGolay[{case['name']}]")


def test_derivate_binding_parity(fixtures_dir):
    fx = _load_fixture(fixtures_dir, "derivate")
    X_fit = _decode_matrix(fx["fit_input_hex"], fx["fit_rows"], fx["fit_cols"])
    X_test = _decode_matrix(fx["input_hex"], fx["rows"], fx["cols"])
    for case in fx["cases"]:
        params = case["params"]
        op = Derivate(order=params["order"], delta=params["delta"])
        op.fit(X_fit)
        pred = op.transform(X_test)
        expected = _decode_matrix(
            case["output_hex"], case["output_rows"], case["output_cols"]
        )
        _assert_parity(pred, expected, f"Derivate[{case['name']}]")


def test_norris_williams_binding_parity(fixtures_dir):
    fx = _load_fixture(fixtures_dir, "norris_williams")
    X = _decode_matrix(fx["input_hex"], fx["rows"], fx["cols"])
    for case in fx["cases"]:
        params = case["params"]
        op = NorrisWilliams(
            segment=params["segment"],
            gap=params["gap"],
            derivative_order=params["derivative_order"],
            delta=params["delta"],
        )
        pred = op.fit_transform(X)
        expected = _decode_matrix(
            case["output_hex"], case["output_rows"], case["output_cols"]
        )
        _assert_parity(pred, expected, f"NorrisWilliams[{case['name']}]")


def test_gaussian_binding_parity(fixtures_dir):
    fx = _load_fixture(fixtures_dir, "gaussian")
    X = _decode_matrix(fx["input_hex"], fx["rows"], fx["cols"])
    for case in fx["cases"]:
        params = case["params"]
        op = Gaussian(
            sigma=params["sigma"],
            order=params["order"],
            mode=params["mode"],
            cval=params["cval"],
            truncate=params["truncate"],
        )
        pred = op.fit_transform(X)
        expected = _decode_matrix(
            case["output_hex"], case["output_rows"], case["output_cols"]
        )
        _assert_parity(pred, expected, f"Gaussian[{case['name']}]")


def test_first_derivative_binding_parity(fixtures_dir):
    fx = _load_fixture(fixtures_dir, "first_derivative")
    X = _decode_matrix(fx["input_hex"], fx["rows"], fx["cols"])
    for case in fx["cases"]:
        params = case["params"]
        op = FirstDerivative(delta=params["delta"], edge_order=params["edge_order"])
        pred = op.fit_transform(X)
        expected = _decode_matrix(case["output_hex"],
                                  case.get("output_rows", fx["rows"]),
                                  case.get("output_cols", fx["cols"]))
        _assert_parity(pred, expected, f"FirstDerivative[{case['name']}]")


def test_second_derivative_binding_parity(fixtures_dir):
    fx = _load_fixture(fixtures_dir, "second_derivative")
    X = _decode_matrix(fx["input_hex"], fx["rows"], fx["cols"])
    for case in fx["cases"]:
        params = case["params"]
        op = SecondDerivative(delta=params["delta"], edge_order=params["edge_order"])
        pred = op.fit_transform(X)
        expected = _decode_matrix(case["output_hex"],
                                  case.get("output_rows", fx["rows"]),
                                  case.get("output_cols", fx["cols"]))
        _assert_parity(pred, expected, f"SecondDerivative[{case['name']}]")


# ---------------------------------------------------------------------------
# Preprocessing — stateful (MSC, EMSC)
# ---------------------------------------------------------------------------


def test_msc_binding_parity(fixtures_dir):
    fx = _load_fixture(fixtures_dir, "msc")
    X_fit = _decode_matrix(fx["fit_input_hex"], fx["fit_rows"], fx["fit_cols"])
    X_test = _decode_matrix(fx["input_hex"], fx["rows"], fx["cols"])
    for case in fx["cases"]:
        op = MSC()
        op.fit(X_fit)
        params = case.get("params", {})
        if params.get("variant") == "inverse":
            # The fixture generator computes
            #   inverse(MSC.transform(test_X))
            # so the binding must do the same: forward-transform first, then
            # invert.
            X_fwd = op.transform(X_test)
            pred = op.inverse_transform(X_fwd)
        else:
            pred = op.transform(X_test)
        expected = _decode_matrix(case["output_hex"],
                                  case.get("output_rows", fx["rows"]),
                                  case.get("output_cols", fx["cols"]))
        _assert_parity(pred, expected, f"MSC[{case['name']}]")


def test_emsc_binding_parity(fixtures_dir):
    fx = _load_fixture(fixtures_dir, "emsc")
    X_fit = _decode_matrix(fx["fit_input_hex"], fx["fit_rows"], fx["fit_cols"])
    X_test = _decode_matrix(fx["input_hex"], fx["rows"], fx["cols"])
    for case in fx["cases"]:
        params = case["params"]
        op = EMSC(degree=params["degree"])
        op.fit(X_fit)
        pred = op.transform(X_test)
        expected = _decode_matrix(case["output_hex"],
                                  case.get("output_rows", fx["rows"]),
                                  case.get("output_cols", fx["cols"]))
        _assert_parity(pred, expected, f"EMSC[{case['name']}]")


def test_baseline_center_binding_parity(fixtures_dir):
    fx = _load_fixture(fixtures_dir, "baseline_center")
    X_fit = _decode_matrix(fx["fit_input_hex"], fx["fit_rows"], fx["fit_cols"])
    X_test = _decode_matrix(fx["input_hex"], fx["rows"], fx["cols"])
    for case in fx["cases"]:
        op = BaselineCenter()
        op.fit(X_fit)
        params = case.get("params", {})
        if params.get("variant") == "inverse":
            X_fwd = op.transform(X_fit)
            pred = op.inverse_transform(X_fwd)
        elif params.get("variant") == "test":
            pred = op.transform(X_test)
        else:
            pred = op.transform(X_fit)
        expected = _decode_matrix(
            case["output_hex"], case["output_rows"], case["output_cols"]
        )
        _assert_parity(pred, expected, f"BaselineCenter[{case['name']}]")


# ---------------------------------------------------------------------------
# Orthogonalization / feature extraction
# ---------------------------------------------------------------------------


def test_osc_binding_parity(fixtures_dir):
    fx = _load_fixture(fixtures_dir, "osc")
    X_fit = _decode_matrix(fx["fit_input_hex"], fx["fit_rows"], fx["fit_cols"])
    y_fit = _decode_hex_array(fx["fit_y_hex"])
    X_test = _decode_matrix(fx["input_hex"], fx["rows"], fx["cols"])
    for case in fx["cases"]:
        params = case["params"]
        op = OSC(n_components=params["n_components"], scale=params["scale"])
        op.fit(X_fit, y_fit)
        pred = op.transform(X_fit if params["variant"] == "fit_transform" else X_test)
        expected = _decode_matrix(
            case["output_hex"], case["output_rows"], case["output_cols"]
        )
        _assert_parity(pred, expected, f"OSC[{case['name']}]")


def test_epo_binding_parity(fixtures_dir):
    fx = _load_fixture(fixtures_dir, "epo")
    X_fit = _decode_matrix(fx["fit_input_hex"], fx["fit_rows"], fx["fit_cols"])
    d_fit = _decode_hex_array(fx["fit_d_hex"])
    X_test = _decode_matrix(fx["input_hex"], fx["rows"], fx["cols"])
    for case in fx["cases"]:
        params = case["params"]
        op = EPO(scale=params["scale"])
        if params["variant"] == "fit_transform":
            pred = op.fit_transform(X_fit, d_fit)
        else:
            op.fit(X_fit, d_fit)
            pred = op.transform(X_test)
        expected = _decode_matrix(
            case["output_hex"], case["output_rows"], case["output_cols"]
        )
        _assert_parity(pred, expected, f"EPO[{case['name']}]")


def test_flexible_pca_binding_parity(fixtures_dir):
    fx = _load_fixture(fixtures_dir, "flexible_pca")
    X_fit = _decode_matrix(fx["fit_input_hex"], fx["fit_rows"], fx["fit_cols"])
    X_test = _decode_matrix(fx["input_hex"], fx["rows"], fx["cols"])
    for case in fx["cases"]:
        op = FlexiblePCA(n_components=case["params"]["n_components"])
        op.fit(X_fit)
        pred = op.transform(X_test)
        expected = _decode_matrix(
            case["output_hex"], case["output_rows"], case["output_cols"]
        )
        _assert_parity(pred, expected, f"FlexiblePCA[{case['name']}]")


def test_flexible_svd_binding_parity(fixtures_dir):
    fx = _load_fixture(fixtures_dir, "flexible_svd")
    X_fit = _decode_matrix(fx["fit_input_hex"], fx["fit_rows"], fx["fit_cols"])
    X_test = _decode_matrix(fx["input_hex"], fx["rows"], fx["cols"])
    for case in fx["cases"]:
        op = FlexibleSVD(n_components=case["params"]["n_components"])
        op.fit(X_fit)
        pred = op.transform(X_test)
        expected = _decode_matrix(
            case["output_hex"], case["output_rows"], case["output_cols"]
        )
        _assert_parity(pred, expected, f"FlexibleSVD[{case['name']}]")


def test_fck_static_binding_parity(fixtures_dir):
    fx = _load_fixture(fixtures_dir, "fck_static")
    X = _decode_matrix(fx["input_hex"], fx["rows"], fx["cols"])
    for case in fx["cases"]:
        params = case["params"]
        op = FCKStaticTransformer(
            kernel_size=params["kernel_size"],
            alphas=params["alphas"],
            sigmas=params["sigmas"],
        )
        pred = op.fit_transform(X)
        expected = _decode_matrix(
            case["output_hex"], case["output_rows"], case["output_cols"]
        )
        _assert_parity(pred, expected, f"FCKStaticTransformer[{case['name']}]")


# ---------------------------------------------------------------------------
# Baseline
# ---------------------------------------------------------------------------


def _run_stateless_matrix_fixture(fixtures_dir, fixture_name, factory, label,
                                  tolerance: float = TOLERANCE):
    fx = _load_fixture(fixtures_dir, fixture_name)
    X = _decode_matrix(fx["input_hex"], fx["rows"], fx["cols"])
    for case in fx["cases"]:
        op = factory(case["params"])
        pred = op.fit_transform(X)
        expected = _decode_matrix(
            case["output_hex"],
            case.get("output_rows", fx["rows"]),
            case.get("output_cols", fx["cols"]),
        )
        _assert_parity(pred, expected, f"{label}[{case['name']}]", tolerance)


def test_detrend_binding_parity(fixtures_dir):
    fx = _load_fixture(fixtures_dir, "detrend")
    X = _decode_matrix(fx["input_hex"], fx["rows"], fx["cols"])
    for case in fx["cases"]:
        params = case["params"]
        op = Detrend(polyorder=params["polyorder"])
        pred = op.fit_transform(X)
        expected = _decode_matrix(case["output_hex"],
                                  case.get("output_rows", fx["rows"]),
                                  case.get("output_cols", fx["cols"]))
        _assert_parity(pred, expected, f"Detrend[{case['name']}]")


def test_asls_binding_parity(fixtures_dir):
    fx = _load_fixture(fixtures_dir, "asls")
    X = _decode_matrix(fx["input_hex"], fx["rows"], fx["cols"])
    for case in fx["cases"]:
        params = case["params"]
        op = AsLS(lam=params["lam"], p=params["p"],
                  max_iter=params["max_iter"], tol=params["tol"])
        pred = op.fit_transform(X)
        expected = _decode_matrix(case["output_hex"],
                                  case.get("output_rows", fx["rows"]),
                                  case.get("output_cols", fx["cols"]))
        _assert_parity(pred, expected, f"AsLS[{case['name']}]")


def test_airpls_binding_parity(fixtures_dir):
    fx = _load_fixture(fixtures_dir, "airpls")
    X = _decode_matrix(fx["input_hex"], fx["rows"], fx["cols"])
    for case in fx["cases"]:
        params = case["params"]
        op = AirPLS(lam=params["lam"], max_iter=params["max_iter"], tol=params["tol"])
        pred = op.fit_transform(X)
        expected = _decode_matrix(case["output_hex"],
                                  case.get("output_rows", fx["rows"]),
                                  case.get("output_cols", fx["cols"]))
        _assert_parity(pred, expected, f"AirPLS[{case['name']}]")


def test_arpls_binding_parity(fixtures_dir):
    _run_stateless_matrix_fixture(
        fixtures_dir,
        "arpls",
        lambda p: ArPLS(lam=p["lam"], max_iter=p["max_iter"], tol=p["tol"]),
        "ArPLS",
    )


def test_modpoly_binding_parity(fixtures_dir):
    _run_stateless_matrix_fixture(
        fixtures_dir,
        "modpoly",
        lambda p: ModPoly(polyorder=p["polyorder"], max_iter=p["max_iter"], tol=p["tol"]),
        "ModPoly",
    )


def test_imodpoly_binding_parity(fixtures_dir):
    _run_stateless_matrix_fixture(
        fixtures_dir,
        "imodpoly",
        lambda p: IModPoly(polyorder=p["polyorder"], max_iter=p["max_iter"], tol=p["tol"]),
        "IModPoly",
    )


def test_snip_binding_parity(fixtures_dir):
    _run_stateless_matrix_fixture(
        fixtures_dir,
        "snip",
        lambda p: SNIP(max_half_window=p["max_half_window"]),
        "SNIP",
    )


def test_rolling_ball_binding_parity(fixtures_dir):
    _run_stateless_matrix_fixture(
        fixtures_dir,
        "rolling_ball",
        lambda p: RollingBall(
            half_window=p["half_window"],
            smooth_half_window=p["smooth_half_window"],
        ),
        "RollingBall",
    )


def test_iasls_binding_parity(fixtures_dir):
    _run_stateless_matrix_fixture(
        fixtures_dir,
        "iasls",
        lambda p: IAsLS(
            lam=p["lam"],
            p=p["p"],
            lam_1=p.get("lam_1", 1e-4),
            polyorder=p["polyorder"],
            diff_order=p.get("diff_order", 2),
            max_iter=p["max_iter"],
            tol=p["tol"],
        ),
        "IAsLS",
        tolerance=5e-6,
    )


def test_beads_binding_parity(fixtures_dir):
    _run_stateless_matrix_fixture(
        fixtures_dir,
        "beads",
        lambda p: BEADS(
            lam_0=p["lam_0"],
            lam_1=p["lam_1"],
            lam_2=p["lam_2"],
            max_iter=p["max_iter"],
            tol=p["tol"],
        ),
        "BEADS",
    )


# ---------------------------------------------------------------------------
# Resampling / discretization
# ---------------------------------------------------------------------------


def test_crop_transformer_binding_parity(fixtures_dir):
    fx = _load_fixture(fixtures_dir, "crop")
    X = _decode_matrix(fx["input_hex"], fx["rows"], fx["cols"])
    for case in fx["cases"]:
        params = case["params"]
        op = CropTransformer(start=params["start"], end=params["end"])
        pred = op.fit_transform(X)
        expected = _decode_matrix(
            case["output_hex"], case["output_rows"], case["output_cols"]
        )
        _assert_parity(pred, expected, f"CropTransformer[{case['name']}]")


def test_resample_transformer_binding_parity(fixtures_dir):
    fx = _load_fixture(fixtures_dir, "resample")
    X = _decode_matrix(fx["input_hex"], fx["rows"], fx["cols"])
    for case in fx["cases"]:
        op = ResampleTransformer(num_samples=case["params"]["num_samples"])
        pred = op.fit_transform(X)
        expected = _decode_matrix(
            case["output_hex"], case["output_rows"], case["output_cols"]
        )
        _assert_parity(pred, expected, f"ResampleTransformer[{case['name']}]")


def _grid(start: float, step: float, n: int) -> np.ndarray:
    return start + step * np.arange(int(n), dtype=np.float64)


def test_resampler_binding_parity(fixtures_dir):
    fx = _load_fixture(fixtures_dir, "resampler")
    X = _decode_matrix(fx["input_hex"], fx["rows"], fx["cols"])
    for case in fx["cases"]:
        params = case["params"]
        source = _grid(params["src_min"], params["src_step"], params["src_n"])
        target = _grid(params["tgt_min"], params["tgt_step"], params["tgt_n"])
        op = Resampler(
            target_wavelengths=target,
            method=params["method"],
            crop_min=params["crop_min"],
            crop_max=params["crop_max"],
            use_crop=bool(params["use_crop"]),
            fill_value=params["fill_value"],
            bounds_error=bool(params["bounds_error"]),
            extrapolate=bool(params["extrapolate"]),
        )
        op.fit(source_wavelengths=source)
        pred = op.transform(X)
        expected = _decode_matrix(
            case["output_hex"], case["output_rows"], case["output_cols"]
        )
        _assert_parity(pred, expected, f"Resampler[{case['name']}]")


def test_integer_kbins_discretizer_binding_parity(fixtures_dir):
    fx = _load_fixture(fixtures_dir, "kbins_disc")
    X_fit = _decode_matrix(fx["fit_input_hex"], fx["fit_rows"], fx["fit_cols"])
    X_test = _decode_matrix(fx["input_hex"], fx["rows"], fx["cols"])
    for case in fx["cases"]:
        params = case["params"]
        op = IntegerKBinsDiscretizer(
            n_bins=params["n_bins"],
            strategy=params["strategy"],
        )
        op.fit(X_fit)
        pred = op.transform(X_test).astype(np.float64)
        expected = _decode_matrix(
            case["output_hex"], case["output_rows"], case["output_cols"]
        )
        _assert_parity(pred, expected, f"IntegerKBinsDiscretizer[{case['name']}]")


def test_range_discretizer_binding_parity(fixtures_dir):
    fx = _load_fixture(fixtures_dir, "range_disc")
    X = _decode_matrix(fx["input_hex"], fx["rows"], fx["cols"])
    for case in fx["cases"]:
        op = RangeDiscretizer(edges_csv=case["params"]["edges_csv"])
        pred = op.fit_transform(X).astype(np.float64)
        expected = _decode_matrix(
            case["output_hex"], case["output_rows"], case["output_cols"]
        )
        _assert_parity(pred, expected, f"RangeDiscretizer[{case['name']}]")


# ---------------------------------------------------------------------------
# Splitters — compare 1-D index arrays as float64 ``(N,)`` views.
# ---------------------------------------------------------------------------


def test_kennard_stone_binding_parity(fixtures_dir):
    fx = _load_fixture(fixtures_dir, "split_kennard_stone")
    X = _decode_matrix(fx["input_hex"], fx["rows"], fx["cols"])
    for case in fx["cases"]:
        params = case["params"]
        op = KennardStoneSplitter(test_size=params["test_size"])
        train, test = op.split(X)
        expected_train = np.asarray(params["train_idx"], dtype=np.int64)
        expected_test = np.asarray(params["test_idx"], dtype=np.int64)
        # Order matters for Kennard-Stone (it's deterministic + ordered).
        _assert_parity(train.astype(np.float64), expected_train.astype(np.float64),
                       f"KennardStone[{case['name']}] train_idx")
        _assert_parity(test.astype(np.float64), expected_test.astype(np.float64),
                       f"KennardStone[{case['name']}] test_idx")


def test_spxy_binding_parity(fixtures_dir):
    fx = _load_fixture(fixtures_dir, "split_spxy")
    X = _decode_matrix(fx["input_hex"], fx["rows"], fx["cols"])
    Y = _decode_matrix(fx["fit_input_hex"], fx["fit_rows"], fx["fit_cols"])
    for case in fx["cases"]:
        params = case["params"]
        op = SPXYSplitter(test_size=params["test_size"])
        train, test = op.split(X, Y)
        expected_train = np.asarray(params["train_idx"], dtype=np.int64)
        expected_test = np.asarray(params["test_idx"], dtype=np.int64)
        _assert_parity(train.astype(np.float64), expected_train.astype(np.float64),
                       f"SPXY[{case['name']}] train_idx")
        _assert_parity(test.astype(np.float64), expected_test.astype(np.float64),
                       f"SPXY[{case['name']}] test_idx")


def test_kbins_stratified_binding_parity(fixtures_dir):
    fx = _load_fixture(fixtures_dir, "split_kbins_stratified")
    Y = _decode_matrix(fx["fit_input_hex"], fx["fit_rows"], fx["fit_cols"])
    for case in fx["cases"]:
        params = case["params"]
        op = KBinsStratifiedSplitter(
            test_size=params["test_size"],
            seed=params["seed"],
            n_bins=params["n_bins"],
            strategy=int(params["strategy"]),
        )
        train, test = op.split(Y)
        expected_train = np.asarray(params["train_idx"], dtype=np.int64)
        expected_test = np.asarray(params["test_idx"], dtype=np.int64)
        # KBinsStratified returns sorted indices; just compare directly.
        _assert_parity(np.sort(train).astype(np.float64),
                       np.sort(expected_train).astype(np.float64),
                       f"KBinsStratified[{case['name']}] train_idx")
        _assert_parity(np.sort(test).astype(np.float64),
                       np.sort(expected_test).astype(np.float64),
                       f"KBinsStratified[{case['name']}] test_idx")


# ---------------------------------------------------------------------------
# Filters
# ---------------------------------------------------------------------------


def test_y_outlier_filter_binding_parity(fixtures_dir):
    fx = _load_fixture(fixtures_dir, "filter_y_outlier_iqr")
    y = _decode_hex_array(fx["y_hex"])
    for case in fx["cases"]:
        params = case["params"]
        op = YOutlierFilter(
            method=params["method"],
            threshold=params["threshold"],
            lower_percentile=params.get("lower_percentile", 1.0),
            upper_percentile=params.get("upper_percentile", 99.0),
        )
        mask, stats = op.fit_apply(y)
        expected_mask = np.asarray(case["expected_mask"], dtype=np.uint8)
        _assert_parity(mask.astype(np.float64), expected_mask.astype(np.float64),
                       f"YOutlierFilter[{case['name']}] mask")
        # Spot-check the stats payload.
        es = case["expected_stats"]
        assert stats.n_samples == es["n_samples"], (
            f"YOutlierFilter[{case['name']}] n_samples mismatch")
        assert stats.n_kept == es["n_kept"]
        assert stats.n_excluded == es["n_excluded"]
        # exclusion_rate is hex-encoded float.
        expected_rate = struct.unpack(">d", bytes.fromhex(es["exclusion_rate"]))[0]
        assert abs(stats.exclusion_rate - expected_rate) <= TOLERANCE


def test_x_outlier_filter_binding_parity(fixtures_dir):
    fx = _load_fixture(fixtures_dir, "filter_x_outlier_mahalanobis")
    X = _decode_matrix(fx["input_hex"], fx["rows"], fx["cols"])
    for case in fx["cases"]:
        params = case["params"]
        # Some cases include `threshold: null` and `n_components: null`.
        threshold = params.get("threshold")
        n_components = params.get("n_components")
        op = XOutlierFilter(
            method=params["method"],
            use_threshold=threshold is not None,
            threshold=(float(threshold) if threshold is not None else 0.0),
            n_components=(int(n_components) if n_components is not None else 0),
            contamination=params.get("contamination", 0.1),
            seed=int(params.get("random_state", 0)),
        )
        mask, _stats = op.fit_apply(X)
        # output_hex encodes the mask as F64 values (0.0/1.0).
        expected = _decode_hex_array(case["output_hex"])
        _assert_parity(mask.astype(np.float64), expected,
                       f"XOutlierFilter[{case['name']}] mask")


# ---------------------------------------------------------------------------
# Metrics / diagnostics
# ---------------------------------------------------------------------------


def test_nirs_metrics_binding_parity(fixtures_dir):
    fx = _load_fixture(fixtures_dir, "nirs_metrics")
    metric_fns = {
        "rmse": rmse,
        "mae": mae,
        "bias": bias,
        "sep": sep,
        "rpd": rpd,
        "rpiq": rpiq,
        "r2": r2,
        "nrmse": nrmse,
    }
    for case in fx["cases"]:
        y_true = _decode_hex_array(case["y_true_hex"])
        y_pred = _decode_hex_array(case["y_pred_hex"])
        for name, fn in metric_fns.items():
            pred = np.asarray([fn(y_true, y_pred)], dtype=np.float64)
            expected = np.asarray([_decode_scalar(case["metrics_hex"][name])])
            _assert_parity_allow_nonfinite(pred, expected, f"{name}[{case['name']}]")


def test_hotelling_t2_binding_parity(fixtures_dir):
    fx = _load_fixture(fixtures_dir, "hotelling_t2")
    for case in fx["cases"]:
        X = _decode_matrix(case["input_hex"], case["rows"], case["cols"])
        values, ucl = hotelling_t2(
            X,
            n_components=case["n_components"],
            alpha=case["alpha"],
        )
        expected_values = _decode_hex_array(case["t2_hex"])
        expected_ucl = np.asarray([_decode_scalar(case["ucl_hex"])])
        _assert_parity(values, expected_values, f"hotelling_t2[{case['name']}] values")
        _assert_parity(np.asarray([ucl]), expected_ucl, f"hotelling_t2[{case['name']}] ucl")


def test_q_residuals_binding_parity(fixtures_dir):
    fx = _load_fixture(fixtures_dir, "q_residuals")
    for case in fx["cases"]:
        X = _decode_matrix(case["input_hex"], case["rows"], case["cols"])
        values, ucl = q_residuals(
            X,
            n_components=case["n_components"],
            alpha=case["alpha"],
        )
        expected_values = _decode_hex_array(case["q_hex"])
        expected_ucl = np.asarray([_decode_scalar(case["ucl_hex"])])
        _assert_parity(values, expected_values, f"q_residuals[{case['name']}] values")
        _assert_parity(np.asarray([ucl]), expected_ucl, f"q_residuals[{case['name']}] ucl")


def test_transfer_metrics_binding_parity(fixtures_dir):
    fx = _load_fixture(fixtures_dir, "transfer_metrics")
    for case in fx["cases"]:
        p = case["params"]
        source = _decode_matrix(
            case["source_hex"], case["source_rows"], case["source_cols"]
        )
        target = _decode_matrix(
            case["target_hex"], case["target_rows"], case["target_cols"]
        )
        pred = transfer_metrics(
            source,
            target,
            n_components=p["n_components"],
            k_neighbors=p["k_neighbors"],
            seed=p["seed"],
        )
        for name, hex_value in case["expected"].items():
            _assert_parity_allow_nonfinite(
                np.asarray([pred[name]], dtype=np.float64),
                np.asarray([_decode_scalar(hex_value)]),
                f"transfer_metrics[{case['name']}] {name}",
            )


# ---------------------------------------------------------------------------
# Wavelet / Augmentation
# ---------------------------------------------------------------------------


def test_wavelet_binding_parity(fixtures_dir):
    fx = _load_fixture(fixtures_dir, "wavelet")
    X = _decode_matrix(fx["input_hex"], fx["rows"], fx["cols"])
    for case in fx["cases"]:
        params = case["params"]
        op = Wavelet(family=params["family"], mode=params["mode"])
        pred = op.fit_transform(X)
        expected = _decode_matrix(
            case["output_hex"], case["output_rows"], case["output_cols"]
        )
        _assert_parity(pred, expected, f"Wavelet[{case['name']}]")


def test_haar_binding_parity(fixtures_dir):
    fx = _load_fixture(fixtures_dir, "haar")
    X = _decode_matrix(fx["input_hex"], fx["rows"], fx["cols"])
    for case in fx["cases"]:
        pred = Haar().fit_transform(X)
        expected = _decode_matrix(
            case["output_hex"], case["output_rows"], case["output_cols"]
        )
        _assert_parity(pred, expected, f"Haar[{case['name']}]")


def test_wavelet_denoise_binding_parity(fixtures_dir):
    fx = _load_fixture(fixtures_dir, "wavelet_denoise")
    X = _decode_matrix(fx["input_hex"], fx["rows"], fx["cols"])
    for case in fx["cases"]:
        params = case["params"]
        op = WaveletDenoise(
            family=params["family"],
            mode=params["mode"],
            level=params["level"],
            threshold_mode=params["threshold_mode"],
            noise_estimator=params["noise_estimator"],
        )
        pred = op.fit_transform(X)
        expected = _decode_matrix(case["output_hex"],
                                  case.get("output_rows", fx["rows"]),
                                  case.get("output_cols", fx["cols"]))
        _assert_parity(pred, expected, f"WaveletDenoise[{case['name']}]")


def test_wavelet_features_binding_parity(fixtures_dir):
    fx = _load_fixture(fixtures_dir, "wavelet_features")
    X = _decode_matrix(fx["input_hex"], fx["rows"], fx["cols"])
    for case in fx["cases"]:
        params = case["params"]
        op = WaveletFeatures(
            family=params["family"],
            mode=params["mode"],
            max_level=params["max_level"],
        )
        pred = op.fit_transform(X)
        expected = _decode_matrix(
            case["output_hex"], case["output_rows"], case["output_cols"]
        )
        _assert_parity(pred, expected, f"WaveletFeatures[{case['name']}]")


def test_wavelet_pca_binding_parity(fixtures_dir):
    fx = _load_fixture(fixtures_dir, "wavelet_pca")
    X_fit = _decode_matrix(fx["fit_input_hex"], fx["fit_rows"], fx["fit_cols"])
    X_test = _decode_matrix(fx["input_hex"], fx["rows"], fx["cols"])
    for case in fx["cases"]:
        params = case["params"]
        op = WaveletPCA(
            family=params["family"],
            mode=params["mode"],
            max_level=params["max_level"],
            n_components=params["n_components"],
        )
        op.fit(X_fit)
        pred = op.transform(X_test)
        expected = _decode_matrix(
            case["output_hex"], case["output_rows"], case["output_cols"]
        )
        _assert_parity(pred, expected, f"WaveletPCA[{case['name']}]")


def test_wavelet_svd_binding_parity(fixtures_dir):
    fx = _load_fixture(fixtures_dir, "wavelet_svd")
    X_fit = _decode_matrix(fx["fit_input_hex"], fx["fit_rows"], fx["fit_cols"])
    X_test = _decode_matrix(fx["input_hex"], fx["rows"], fx["cols"])
    for case in fx["cases"]:
        params = case["params"]
        op = WaveletSVD(
            family=params["family"],
            mode=params["mode"],
            max_level=params["max_level"],
            n_components=params["n_components"],
        )
        op.fit(X_fit)
        pred = op.transform(X_test)
        expected = _decode_matrix(
            case["output_hex"], case["output_rows"], case["output_cols"]
        )
        _assert_parity(pred, expected, f"WaveletSVD[{case['name']}]")


def test_gaussian_noise_binding_parity(fixtures_dir):
    fx = _load_fixture(fixtures_dir, "aug_gaussian_noise")
    X = _decode_matrix(fx["X_hex"], fx["rows"], fx["cols"])
    for case in fx["cases"]:
        seed = int(case["seed"])
        params = case["params"]
        rng = PCG64(seed=seed)
        op = GaussianAdditiveNoise(sigma=params["sigma"], rng=rng)
        pred = op.fit_transform(X)
        expected = _decode_matrix(case["expected_out_hex"], fx["rows"], fx["cols"])
        _assert_parity(pred, expected, f"GaussianAdditiveNoise[{case['name']}]")
        # Keep `rng` alive for the duration of the test.
        del op
        del rng
