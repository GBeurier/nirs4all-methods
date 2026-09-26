# SPDX-License-Identifier: CECILL-2.1
"""Native fitted preprocessing and N4MP portability checks."""

from __future__ import annotations

import numpy as np
import pytest
from n4m._errors import N4MError
from n4m.compose import NativePreprocessingPipeline, PreprocessingOperatorSpec

_KINDS = (
    "identity", "center", "autoscale", "pareto_scale", "snv", "msc",
    "emsc", "detrend_poly", "savgol_smooth", "savgol_derivative",
    "norris_williams", "asls_baseline", "osc", "epo", "wavelet_denoise",
)


@pytest.fixture
def spectra():
    features = np.arange(1, 25, dtype=np.float64)
    train_rows = np.arange(1, 31, dtype=np.float64)
    held_rows = np.array([3.5, 17.5], dtype=np.float64)

    def signal(rows):
        return (np.sin(rows[:, None] * features / 31)
                + np.cos(rows[:, None] + features / 11)
                + rows[:, None] * features / 1000)

    return signal(train_rows), signal(held_rows), train_rows[:, None] / 30


@pytest.mark.parametrize("kind", _KINDS)
def test_all_15_native_kinds_round_trip_fitted_n4mp(spectra, kind):
    train, heldout, y = spectra
    target = y if kind in {"osc", "epo"} else None
    with NativePreprocessingPipeline([PreprocessingOperatorSpec(kind)]) as pipeline:
        pipeline.fit(train, target)
        transformed = pipeline.transform(heldout)
        assert transformed.shape == heldout.shape
        assert np.isfinite(transformed).all()
        assert pipeline.n_features_in_ == train.shape[1]
        assert pipeline.n_operators_ == 1
        assert pipeline.steps_ == (PreprocessingOperatorSpec(kind),)
        blob = pipeline.to_bytes()
        assert blob.startswith(b"N4MP")
    with NativePreprocessingPipeline.from_bytes(blob) as restored:
        assert restored.n_features_in_ == train.shape[1]
        assert restored.n_operators_ == 1
        assert restored.steps_ == (PreprocessingOperatorSpec(kind),)
        assert restored.operators == restored.steps_
        np.testing.assert_array_equal(restored.transform(heldout), transformed)
        assert restored.to_bytes() == blob
        with pytest.raises(ValueError, match="feature width"):
            restored.transform(heldout[:, :-1])


def test_ordered_pipeline_matches_independent_r_heldout_oracle():
    features = np.arange(1, 9, dtype=np.float64)

    def signal(rows):
        rows = np.asarray(rows, dtype=np.float64)
        return (np.sin(rows[:, None] * features / 9)
                + np.cos(rows[:, None] + features / 7)
                + rows[:, None] * features / 100)

    train = signal(np.arange(1, 7))
    heldout = signal([2.5, 7.5])
    expected = np.array([
        [-4.001193774003808, -2.2897281335515838, -0.7449819940745884,
         0.5119733321851917, 1.3908885370727357, 1.8394358338967038,
         1.8472190824668655, 1.446387116008484],
        [-2.2015814295950307, -2.6519026524658353, -1.5440849758961113,
         0.4949255610220601, 2.225790408871716, 2.604408534032623,
         1.461767129717428, -0.3893225756868496],
    ])
    pipeline = NativePreprocessingPipeline(["snv", ("msc", ())]).fit(train)
    try:
        np.testing.assert_allclose(pipeline.transform(heldout), expected,
                                   rtol=0, atol=1e-10)
        with NativePreprocessingPipeline.from_bytes(
            pipeline.to_bytes(), expected_operators=["snv", "msc"]
        ) as restored:
            np.testing.assert_allclose(restored.transform(heldout), expected,
                                       rtol=0, atol=1e-10)
            assert tuple(step.kind for step in restored.steps_) == ("snv", "msc")
            assert len(restored.steps_) == 2
            with pytest.raises(ValueError, match="differ"):
                NativePreprocessingPipeline.from_bytes(
                    pipeline.to_bytes(), expected_operators=["snv", "autoscale"])
    finally:
        pipeline.close()


def test_fitted_state_and_malformed_inputs(spectra):
    train, heldout, y = spectra
    pipeline = NativePreprocessingPipeline(["center"])
    with pytest.raises(RuntimeError, match="not fitted"):
        pipeline.to_bytes()
    with pytest.raises(RuntimeError, match="not fitted"):
        pipeline.transform(heldout)
    pipeline.fit(train)
    reference = pipeline.transform(heldout)
    pipeline.operators = [PreprocessingOperatorSpec("snv", (np.nan,))]
    with pytest.raises(ValueError, match="finite"):
        pipeline.fit(train)
    np.testing.assert_array_equal(pipeline.transform(heldout), reference)
    with pytest.raises(ValueError, match="rows"):
        pipeline.fit(train, y[:-1])
    with pytest.raises(ValueError, match="finite"):
        pipeline.transform(np.full_like(heldout, np.inf))
    for invalid in ("gaussian", "not_a_kind", 18, True):
        with pytest.raises(ValueError, match="kind"):
            NativePreprocessingPipeline([invalid]).fit(train)
    with pytest.raises(N4MError):
        NativePreprocessingPipeline.from_bytes(b"N4MPbad")
    pipeline.close()
    pipeline.close()
    with pytest.raises(RuntimeError, match="not fitted"):
        pipeline.transform(heldout)
