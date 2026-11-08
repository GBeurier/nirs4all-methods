# SPDX-License-Identifier: CECILL-2.1 OR AGPL-3.0-or-later
"""Focused StandardScale numerical and fitted-parameter contracts."""

from __future__ import annotations

import json
import os
import pickle
import struct
from pathlib import Path

import numpy as np
import pytest
from n4m import MultimodalClassifierPipeline, MultimodalPipeline, roles
from n4m._errors import N4MError
from n4m._types import Status


def scaler_rows():
    """Exactly representable offsets, a constant, and variance 5 / 20/3."""
    x = np.array([[2.0**40 + k, 7, k] for k in (0, 2, 4, 6)])
    heldout = np.array([[2.0**40 + k, 7, k] for k in (1, 5)])
    return x, heldout


@pytest.mark.parametrize("with_mean", [False, True])
@pytest.mark.parametrize("with_std", [False, True])
@pytest.mark.parametrize("ddof", [0, 1])
def test_standard_scale_ddof_analytic_state(with_mean, with_std, ddof):
    x, heldout = scaler_rows()
    mean = np.array([2.0**40 + 3, 7, 3])
    scale = np.array([np.sqrt(20 / (4 - ddof)), 1, np.sqrt(20 / (4 - ddof))])
    expected = (heldout - (mean if with_mean else 0)) / (scale if with_std else 1)
    est = roles.StandardScale(with_mean=with_mean, with_std=with_std, ddof=ddof)
    restored = checkpoint = None
    try:
        est.fit(x)
        out = est.transform(heldout)
        np.testing.assert_allclose(out, expected, rtol=1e-12, atol=1e-12)
        np.testing.assert_array_equal(out[:, 1], 0 if with_mean else 7)
        # Additional transform rows cannot refit the learned moments.
        combined = np.vstack([heldout, [2.0**40 + 999, 8, -1000]])
        np.testing.assert_array_equal(est.transform(combined)[:2], out)
        payload = est.to_n4me()
        restored = roles.NativeEstimator.from_n4me(payload)
        checkpoint = pickle.loads(pickle.dumps(est))
        assert restored.to_n4me() == payload
        assert restored.transform(heldout).tobytes() == out.tobytes()
        assert checkpoint.transform(heldout).tobytes() == out.tobytes()
    finally:
        est._release()
        if restored is not None:
            restored._release()
        if checkpoint is not None:
            checkpoint._release()


def test_standard_scale_default_zero_is_bitwise_and_singleton():
    x, heldout = scaler_rows()
    default, explicit = roles.StandardScale(), roles.StandardScale(ddof=0)
    try:
        default.fit(x)
        explicit.fit(x)
        assert default.transform(heldout).tobytes() == explicit.transform(heldout).tobytes()
        assert default.to_n4me() == explicit.to_n4me()
        default.fit([[0.1, 7]])
        np.testing.assert_array_equal(default.transform([[0.1, 7]]), [[0, 0]])
    finally:
        default._release()
        explicit._release()


@pytest.mark.parametrize("with_std", [False, True])
@pytest.mark.parametrize("ddof", [-1, 1, 2, 2**63 - 1])
def test_standard_scale_ddof_fit_domain(ddof, with_std):
    # A one-row sample fit is invalid even when scaling is disabled.
    est = roles.StandardScale(ddof=ddof, with_std=with_std)
    try:
        if ddof < 0:
            with pytest.raises(ValueError) as error:
                est.fit([[1, 7]])
            assert type(error.value) is ValueError
            assert str(error.value) == (
                f"StandardScale: invalid value {ddof!r} for parameter 'ddof'"
            )
        else:
            with pytest.raises(N4MError) as error:
                est.fit([[1, 7]])
            assert error.value.status == Status.ERR_INVALID_ARGUMENT
    finally:
        est._release()


def test_standard_scale_fractional_ddof_refused():
    est = roles.StandardScale(ddof=0.5)
    try:
        with pytest.raises((TypeError, ValueError, N4MError)):
            est.fit([[1], [2]])
    finally:
        est._release()


@pytest.fixture
def prechange_scaler_witness():
    path = os.environ.get("N4M_STANDARD_SCALE_WITNESS_DIR")
    if not path:
        pytest.skip("Root must supply the existing genuine pre-change witness directory")
    return Path(path)


@pytest.mark.parametrize("profile", ["normal", "large-offset"])
def test_standard_scale_default_matches_old_output_bits(prechange_scaler_witness, profile):
    # Root's existing supplemental receipt fixes actual public output bytes.
    root = prechange_scaler_witness / "supplement-population0"
    arrays = json.loads((root / "RECEIPT.json").read_text(encoding="utf-8"))["arrays"]

    def read_array(name):
        spec = arrays[name]
        assert spec["dtype"] == "float64.little_endian"
        return np.frombuffer((root / spec["file"]).read_bytes(), dtype="<f8").reshape(spec["shape"])

    training = "normal-training-input" if profile == "normal" else "large-offset-input"
    transform = "normal-heldout-input" if profile == "normal" else training
    expected = "normal-heldout-output" if profile == "normal" else "large-offset-output"
    output_bytes = (root / arrays[expected]["file"]).read_bytes()
    est = roles.StandardScale()
    try:
        est.fit(read_array(training))
        actual = est.transform(read_array(transform))
        assert list(actual.shape) == arrays[expected]["shape"]
        assert actual.astype("<f8").tobytes() == output_bytes
    finally:
        est._release()
    assert (root / arrays[expected]["file"]).read_bytes() == output_bytes


@pytest.mark.parametrize("kind", ["direct", "regression", "classification"])
def test_standard_scale_genuine_prechange_states_refused(prechange_scaler_witness, kind):
    root = prechange_scaler_witness
    filenames = {
        "direct": "standard-scale.n4me",
        "regression": "standard-scale.n4mf",
        "classification": "standard-scale.n4mc",
    }
    payload = (root / filenames[kind]).read_bytes()
    with pytest.raises(N4MError, match="parameter count") as error:
        if kind == "direct":
            est = roles.NativeEstimator.from_n4me(payload)
            est._release()  # Also release if a regression incorrectly accepts it.
        else:
            metadata = json.loads((root / f"{kind}.json").read_text(encoding="utf-8"))
            cls = (
                MultimodalPipeline if kind == "regression"
                else MultimodalClassifierPipeline
            )
            with cls.from_state(
                payload,
                recipe=metadata["recipe"],
                source_schemas=metadata["source_schemas"],
            ):
                pass
    assert error.value.status == Status.ERR_CORRUPT_BUFFER
    assert (root / filenames[kind]).read_bytes() == payload


@pytest.mark.parametrize("classifier", [False, True], ids=["N4MF", "N4MC"])
def test_standard_scale_multimodal_population_recipe_refuses_sample_state(classifier):
    # Exercise the existing serializer integrity check using a whole genuine
    # NEW owner packet, as in existing semantic-substitution tests. Never edit
    # or manufacture the pre-change compatibility witnesses above.
    from test_multimodal_classifier import case, reseal
    from test_multimodal_pipeline import selected_case

    if classifier:
        blocks, recipe, schemas = case(order=("nir",))
        y = [0, 1, 2] * 5
        cls = MultimodalClassifierPipeline
    else:
        blocks, y, recipe, schemas = selected_case(("nir",))
        cls = MultimodalPipeline
    with cls(recipe, schemas) as model:
        model.fit(blocks, y)
        original = model.export_state()
        with cls.from_state(original, recipe=recipe, source_schemas=schemas) as back:
            np.testing.assert_array_equal(back.transform(blocks), model.transform(blocks))
        sample = roles.StandardScale(ddof=1).fit(blocks["nir"])
        try:
            sample_packet = sample.to_n4me()
        finally:
            sample._release()
        # One selected nir source: recipe block then one nested N4ME block.
        offset = 28 + struct.unpack_from("<Q", original, 20)[0]
        size = struct.unpack_from("<Q", original, offset)[0]
        replacement = (
            original[:offset]
            + struct.pack("<Q", len(sample_packet))
            + sample_packet
            + original[offset + 8 + size:]
        )
        with (
            pytest.raises(N4MError, match="nested parameters") as error,
            cls.from_state(reseal(replacement), recipe=recipe, source_schemas=schemas),
        ):
            pass
        assert error.value.status == Status.ERR_CORRUPT_BUFFER
        assert model.export_state() == original
