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
    AsLS,
    Detrend,
    FirstDerivative,
    GaussianAdditiveNoise,
    KBinsStratifiedSplitter,
    KennardStoneSplitter,
    KubelkaMunk,
    PCG64,
    SavitzkyGolay,
    SecondDerivative,
    SPXYSplitter,
    ToAbsorbance,
    WaveletDenoise,
    XOutlierFilter,
    YOutlierFilter,
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


def _decode_matrix(hex_list: list[str], rows: int, cols: int) -> np.ndarray:
    flat = _decode_hex_array(hex_list)
    return flat.reshape(rows, cols).copy(order="C")


def _load_fixture(fixtures_dir: Path, name: str) -> dict[str, Any]:
    path = fixtures_dir / f"{name}_v1.json"
    if not path.exists():
        pytest.skip(f"fixture {path} missing")
    with path.open() as f:
        return json.load(f)


def _assert_parity(pred: np.ndarray, expected: np.ndarray, op_name: str) -> None:
    """Run binding_parity and raise with a friendly message on failure."""
    res = binding_parity(pred, expected, tolerance=TOLERANCE)
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
            out = np.empty(X_fwd.shape, dtype=np.float64, order="C")
            from chemometrics4all._errors import check
            from chemometrics4all._ffi import lib
            from chemometrics4all._matrix import numpy_to_view
            check(
                lib.c4a_pp_msc_inverse_transform(
                    op._handle, numpy_to_view(X_fwd), numpy_to_view(out)
                ),
                "c4a_pp_msc_inverse_transform",
            )
            pred = out
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


# ---------------------------------------------------------------------------
# Baseline
# ---------------------------------------------------------------------------


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
# Wavelet / Augmentation
# ---------------------------------------------------------------------------


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
