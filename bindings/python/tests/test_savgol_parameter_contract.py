# SPDX-License-Identifier: CECILL-2.1 OR AGPL-3.0-or-later
"""Public SG parameter contract against independent SciPy 1.17.1 literals."""

import numpy as np
import pytest
from n4m import N4MError, roles
from n4m.transform.smoothing import SavitzkyGolay

# Root's frozen prepare_references.py SHA-256:
# ceab07b3ffcae1ea8678a56b6d2f68097b332a5fd0dc24aeb89f1df29775d5b5
# Literal inputs/outputs were copied from its independent SciPy calls, not libn4m.
SG_INPUT = [
    0.5, 1.25, -0.75, 3.0, 2.5, -1.0, 4.25, 0.0,
    5.5, 2.0, -2.25, 6.0, 1.5, 3.75, -0.5, 7.0,
]

CASES = [
    # scipy.signal.savgol_filter(
    #     np.array([0.5,1.25,-0.75,3,2.5,-1,4.25,0,5.5,2,-2.25,6,1.5,3.75,-0.5,7], dtype=np.float64),
    #     window_length=10, polyorder=3, deriv=0, delta=1.0,
    #     axis=-1, mode="interp", cval=0.0)
    pytest.param(
        {"window_length": 10, "polyorder": 3, "deriv": 0, "delta": 1.0, "mode": "interp", "cval": 0.0},
        [
            0.6000000000000022, 0.7136363636363636, 0.9181818181818171, 1.1909090909090896,
            1.5090909090909084, 2.4656250000000024, 1.987500000000021, 1.9468750000000061,
            2.0687500000000147, 2.8000000000000105, 1.7328125000000134, 1.681002331002326,
            1.816200466200459, 2.3671911421911345, 3.450815850815845, 5.183916083916084,
        ],
        id="even_interp",
    ),
    # scipy.signal.savgol_filter(
    #     np.array([0.5,1.25,-0.75,3,2.5,-1,4.25,0,5.5,2,-2.25,6,1.5,3.75,-0.5,7], dtype=np.float64),
    #     window_length=10, polyorder=3, deriv=1, delta=0.25,
    #     axis=-1, mode="interp", cval=0.0)
    pytest.param(
        {"window_length": 10, "polyorder": 3, "deriv": 1, "delta": 0.25, "mode": "interp", "cval": 0.0},
        [
            0.24242424242423105, 0.6515151515151453, 0.9696969696969675, 1.1969696969696977,
            1.3333333333333357, 2.894036519036521, -1.6588966588966545, 0.8190559440559428,
            -0.11344211344211136, 0.01418026418026308, -0.6744852369852319, -0.13500388500389632,
            1.294483294483288, 3.1913364413364427, 5.555555555555569, 8.387140637140666,
        ],
        id="even_derivative",
    ),
    # scipy.signal.savgol_filter(
    #     np.array([0.5,1.25,-0.75,3,2.5,-1,4.25,0,5.5,2], dtype=np.float64),
    #     window_length=10, polyorder=3, deriv=0, delta=1.0,
    #     axis=-1, mode="interp", cval=0.0)
    pytest.param(
        {"window_length": 10, "polyorder": 3, "deriv": 0, "delta": 1.0, "mode": "interp", "cval": 0.0},
        [
            0.6000000000000022, 0.7136363636363636, 0.9181818181818171, 1.1909090909090896,
            1.5090909090909084, 1.8499999999999996, 2.1909090909090914, 2.50909090909091,
            2.7818181818181826, 2.986363636363636,
        ],
        id="even_minimum_width",
    ),
    # scipy.signal.savgol_filter(
    #     np.array([0.5,1.25,-0.75,3,2.5,-1,4.25,0,5.5,2,-2.25,6,1.5,3.75,-0.5,7], dtype=np.float64),
    #     window_length=10, polyorder=3, deriv=0, delta=1.0,
    #     axis=-1, mode="mirror", cval=0.0)
    pytest.param(
        {"window_length": 10, "polyorder": 3, "deriv": 0, "delta": 1.0, "mode": "mirror", "cval": 0.0},
        [
            0.8609374999999931, 0.5828125000000072, 1.4390625000000108, 1.0484375000000181,
            1.678125000000017, 2.4656250000000024, 1.987500000000021, 1.9468750000000061,
            2.0687500000000147, 2.8000000000000105, 1.7328125000000134, 2.687500000000008,
            2.1171875000000027, 2.8500000000000134, 2.7343750000000284, 2.7343749999999893,
        ],
        id="even_mirror",
    ),
    # scipy.signal.savgol_filter(
    #     np.array([0.5,1.25,-0.75,3,2.5,-1,4.25,0,5.5,2,-2.25,6,1.5,3.75,-0.5,7], dtype=np.float64),
    #     window_length=10, polyorder=3, deriv=0, delta=1.0,
    #     axis=-1, mode="constant", cval=1.25)
    pytest.param(
        {"window_length": 10, "polyorder": 3, "deriv": 0, "delta": 1.0, "mode": "constant", "cval": 1.25},
        [
            1.1671874999999994, 0.8109375000000103, 1.264062500000004, 1.0484375000000181,
            1.678125000000017, 2.4656250000000024, 1.987500000000021, 1.9468750000000061,
            2.0687500000000147, 2.8000000000000105, 1.7328125000000134, 2.534375000000013,
            2.401562499999998, 3.007812500000011, 3.151562500000013, 2.1265624999999906,
        ],
        id="even_constant",
    ),
    # scipy.signal.savgol_filter(
    #     np.array([0.5,1.25,-0.75,3,2.5,-1,4.25,0,5.5,2,-2.25,6,1.5,3.75,-0.5,7], dtype=np.float64),
    #     window_length=10, polyorder=3, deriv=0, delta=1.0,
    #     axis=-1, mode="nearest", cval=0.0)
    pytest.param(
        {"window_length": 10, "polyorder": 3, "deriv": 0, "delta": 1.0, "mode": "nearest", "cval": 0.0},
        [
            0.9609375000000026, 0.7500000000000139, 1.3015625000000075, 1.1140625000000206,
            1.678125000000017, 2.4656250000000024, 1.987500000000021, 1.9468750000000061,
            2.0687500000000147, 2.8000000000000105, 1.7328125000000134, 2.03125000000003,
            2.1140625000000237, 3.4750000000000423, 4.732812500000049, 5.001562500000031,
        ],
        id="even_nearest",
    ),
    # scipy.signal.savgol_filter(
    #     np.array([0.5,1.25,-0.75,3,2.5,-1,4.25,0,5.5,2,-2.25,6,1.5,3.75,-0.5,7], dtype=np.float64),
    #     window_length=10, polyorder=3, deriv=0, delta=1.0,
    #     axis=-1, mode="wrap", cval=0.0)
    pytest.param(
        {"window_length": 10, "polyorder": 3, "deriv": 0, "delta": 1.0, "mode": "wrap", "cval": 0.0},
        [
            2.1234374999999974, 1.2812500000000029, 1.632812500000002, 0.5453124999999992,
            1.678125000000017, 2.4656250000000024, 1.987500000000021, 1.9468750000000061,
            2.0687500000000147, 2.8000000000000105, 1.7328125000000134, 2.600000000000011,
            2.373437499999997, 3.084375000000004, 2.7781250000000144, 1.6515624999999943,
        ],
        id="even_wrap",
    ),
    # scipy.signal.savgol_filter(
    #     np.array([0.5,1.25,-0.75,3,2.5,-1,4.25,0,5.5,2,-2.25,6,1.5,3.75,-0.5,7], dtype=np.float64),
    #     window_length=10, polyorder=3, deriv=0, delta=0.0,
    #     axis=-1, mode="interp", cval=0.0)
    pytest.param(
        {"window_length": 10, "polyorder": 3, "deriv": 0, "delta": 0.0, "mode": "interp", "cval": 0.0},
        [
            0.6000000000000022, 0.7136363636363636, 0.9181818181818171, 1.1909090909090896,
            1.5090909090909084, 2.4656250000000024, 1.987500000000021, 1.9468750000000061,
            2.0687500000000147, 2.8000000000000105, 1.7328125000000134, 1.681002331002326,
            1.816200466200459, 2.3671911421911345, 3.450815850815845, 5.183916083916084,
        ],
        id="zero_delta_smoothing",
    ),
    # scipy.signal.savgol_filter(
    #     np.array([0.5,1.25,-0.75,3,2.5,-1,4.25,0,5.5,2,-2.25,6,1.5,3.75,-0.5,7], dtype=np.float64),
    #     window_length=11, polyorder=3, deriv=0, delta=1.0,
    #     axis=-1, mode="interp", cval=0.0)
    pytest.param(
        {"window_length": 11, "polyorder": 3, "deriv": 0, "delta": 1.0, "mode": "interp", "cval": 0.0},
        [
            1.0489510489510474, 0.41433566433566105, 0.43181818181817677, 0.8916083916083842,
            1.5839160839160753, 2.298951048951052, 1.8141025641025799, 2.221445221445235,
            1.8846153846153932, 2.4959207459207504, 2.1853146853147076, 1.737762237762254,
            1.5891608391608565, 1.998251748251762, 3.2237762237762317, 5.52447552447552,
        ],
        id="odd_interp",
    ),
    # scipy.signal.savgol_filter(
    #     np.array([0.5,1.25,-0.75,3,2.5,-1,4.25,0,5.5,2,-2.25,6,1.5,3.75,-0.5,7], dtype=np.float64),
    #     window_length=10, polyorder=3, deriv=4, delta=1.0,
    #     axis=-1, mode="interp", cval=0.0)
    pytest.param(
        {"window_length": 10, "polyorder": 3, "deriv": 4, "delta": 1.0, "mode": "interp", "cval": 0.0},
        [
            0.0, 0.0, 0.0, 0.0,
            0.0, 0.0, 0.0, 0.0,
            0.0, 0.0, 0.0, 0.0,
            0.0, 0.0, 0.0, 0.0,
        ],
        id="derivative_above_polyorder",
    ),
]


@pytest.mark.parametrize("params,expected", CASES)
def test_public_savgol_independent_literals(params, expected):
    source = np.array(SG_INPUT[:len(expected)], dtype=np.float64)[None, :]
    before = source.copy()
    actual = SavitzkyGolay(**params).transform(source)
    assert actual.shape == source.shape and actual.dtype == np.float64
    np.testing.assert_array_equal(source, before)
    # Same SG tolerances as cpp/tests/test_preprocessing_smoothing.cpp.
    np.testing.assert_allclose(actual, [expected], atol=1e-9, rtol=1e-10)


@pytest.mark.parametrize("deriv", [1, 4])
def test_zero_spacing_derivative_is_refused(deriv):
    # SciPy 1.17.1: savgol_filter(np.array(SG_INPUT, dtype=np.float64),
    #     window_length=10, polyorder=3, deriv=1, delta=0.0,
    #     axis=-1, mode="interp", cval=0.0) raises ZeroDivisionError.
    # The owner consistently refuses zero spacing for every positive derivative.
    with pytest.raises(N4MError) as caught:
        SavitzkyGolay(window_length=10, polyorder=3, deriv=deriv, delta=0.0,
                      mode="interp", cval=0.0).transform([SG_INPUT])
    assert caught.value.status_name == "INVALID_ARGUMENT"


@pytest.mark.parametrize("params", [
    {"window_length": 0}, {"window_length": -2},
    {"window_length": 10, "polyorder": 10}, {"deriv": -1},
])
def test_invalid_configuration_remains_refused(params):
    with pytest.raises(N4MError) as caught:
        SavitzkyGolay(**params).transform([SG_INPUT])
    assert caught.value.status_name == "INVALID_ARGUMENT"


@pytest.mark.parametrize("parameter,value", [
    pytest.param("polyorder", 2**32 + 3, id="polyorder-wrap-to-3"),
    pytest.param("deriv", 2**32, id="deriv-wrap-to-0"),
])
def test_savgol_int32_parameters_refuse_before_native_call(parameter, value, monkeypatch):
    # Root's public-path counterexamples must not become another int32 algorithm.
    from n4m._ffi import lib

    def forbidden_native_create(*args):
        raise AssertionError("Out-of-int32 SG parameters reached the C ABI")

    monkeypatch.setattr(lib, "n4m_transform_savitzky_golay_create", forbidden_native_create)
    params = {"window_length": 11, "polyorder": 3, "deriv": 0, "mode": "interp"}
    invalid = {**params, parameter: value}
    with pytest.raises(ValueError, match=rf"{parameter}.*INT32"):
        SavitzkyGolay(**invalid)

    unfitted = SavitzkyGolay(**params)
    unfitted.set_params(**{parameter: value})
    with pytest.raises(ValueError, match=rf"{parameter}.*INT32"):
        unfitted.transform([SG_INPUT])


def test_generic_savgol_even_zero_spacing_n4me_round_trip():
    # Same independent SciPy 1.17.1 zero_delta_smoothing literal as above:
    # scipy.signal.savgol_filter(np.array(SG_INPUT, dtype=np.float64),
    #     window_length=10, polyorder=3, deriv=0, delta=0.0,
    #     axis=-1, mode="interp", cval=0.0).
    params = {
        "window_length": 10, "polyorder": 3, "deriv": 0,
        "delta": 0.0, "mode": "interp", "cval": 0.0,
    }
    expected = [[
        0.6000000000000022, 0.7136363636363636, 0.9181818181818171, 1.1909090909090896,
        1.5090909090909084, 2.4656250000000024, 1.987500000000021, 1.9468750000000061,
        2.0687500000000147, 2.8000000000000105, 1.7328125000000134, 1.681002331002326,
        1.816200466200459, 2.3671911421911345, 3.450815850815845, 5.183916083916084,
    ]]
    source = np.array([SG_INPUT], dtype=np.float64)
    before = source.copy()
    definition = roles.method_info("preprocessing.derivatives.savitzky_golay")
    assert definition.state_format == b"width"
    fitted = roles.SavitzkyGolay(**params).fit(source)
    assert fitted.get_params() == params
    assert fitted.capabilities_ == definition.capabilities
    assert fitted.n_features_in_ == source.shape[1]
    np.testing.assert_allclose(fitted.transform(source), expected, atol=1e-9, rtol=1e-10)

    payload = fitted.to_n4me()
    restored = roles.NativeEstimator.from_n4me(payload)
    assert isinstance(restored, roles.SavitzkyGolay)
    assert restored.get_params() == params
    assert restored.n_features_in_ == fitted.n_features_in_
    assert restored.capabilities_ == fitted.capabilities_
    np.testing.assert_allclose(restored.transform(source.copy()), expected, atol=1e-9, rtol=1e-10)
    np.testing.assert_array_equal(source, before)
