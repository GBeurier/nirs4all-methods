# SPDX-License-Identifier: CECILL-2.1
"""Native classifier wiring, typed labels, train-only encoding and portable state."""

from __future__ import annotations

import copy
import ctypes as ct
import hashlib
import struct

import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from n4m import MultimodalClassifierPipeline, MultimodalPipeline, N4MError
from n4m._ffi import lib
from n4m._types import Status
from n4m.roles import PLSLogistic
from n4m.roles._base import _Context
from n4m.roles._multimodal import _views
from test_multimodal_pipeline import raw_case


def case(order=("nir", "metadata"), dtype=np.float64):
    blocks, _, recipe, schemas = raw_case(dtype)
    recipe["source_order"] = list(order)
    recipe["encoders"] = {name: recipe["encoders"][name] for name in order}
    recipe["source_weights"] = {name: recipe["source_weights"][name] for name in order}
    recipe["model"] = {
        "method_id": "models.classification.pls_logistic",
        "params": {"n_components": 2, "max_iter": 1000},
    }
    return {name: blocks[name] for name in order}, recipe, {name: schemas[name] for name in order}


def digest(blocks):
    return {
        name: hashlib.sha256(repr(value.tolist()).encode() if value.dtype == object else value.tobytes()).hexdigest()
        for name, value in blocks.items()
    }


@pytest.mark.parametrize("labels", [
    ["猫", "A", "é"] * 5,
    [-17, 90, 5] * 5,
    ["α", "β", "α"] * 5,
])
@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_train_only_encoders_genuine_classifier_and_typed_labels(labels, dtype):
    blocks, recipe, schemas = case(dtype=dtype)
    before = digest(blocks)
    heldout = {name: value[:3].copy() for name, value in blocks.items()}
    heldout["metadata"][:, 1] = ["🚀", "A", "é"]
    for value in blocks.values():
        value.setflags(write=False)
    # Independent sklearn encoders fit only original training rows.
    nir = StandardScaler().fit(blocks["nir"])
    numeric = StandardScaler().fit(blocks["metadata"][:, :1].astype(np.float64))
    categories = OneHotEncoder(handle_unknown="ignore", sparse_output=False).fit(blocks["metadata"][:, 1:2])

    def encode(values):
        return np.column_stack((
            nir.transform(values["nir"]) * recipe["source_weights"]["nir"],
            np.column_stack((
                numeric.transform(values["metadata"][:, :1].astype(np.float64)),
                categories.transform(values["metadata"][:, 1:2]),
            )) * recipe["source_weights"]["metadata"],
        ))

    expected_classes = np.asarray(sorted(set(labels)), dtype=object if isinstance(labels[0], str) else np.int64)
    with MultimodalClassifierPipeline(recipe, schemas) as pipeline:
        pipeline.fit(blocks, labels)
        assert_array_equal(pipeline.classes_, expected_classes)
        assert_array_equal(pipeline.label_names_, expected_classes)
        assert_allclose(pipeline.transform(blocks), encode(blocks), atol=2e-6 if dtype == np.float32 else 1e-12)
        assert_allclose(pipeline.transform(heldout), encode(heldout), atol=2e-6 if dtype == np.float32 else 1e-12)
        # Existing native PLS-logistic is an independent public fit route; its
        # kernel has separately preserved checked-in numerical parity fixtures.
        head = PLSLogistic(n_components=2, max_iter=1000).fit(encode(blocks), labels)
        try:
            assert_array_equal(pipeline.predict(heldout), head.predict(encode(heldout)))
            assert_allclose(pipeline.predict_proba(heldout), head.predict_proba(encode(heldout)), atol=2e-6 if dtype == np.float32 else 1e-8)
            assert_allclose(pipeline.decision_function(heldout), head.decision_function(encode(heldout)), atol=2e-6 if dtype == np.float32 else 1e-8)
        finally:
            head._release()
        proba = pipeline.predict_proba(heldout)
        assert proba.shape == (3, len(expected_classes))
        assert_allclose(proba.sum(axis=1), 1, atol=1e-12)
        assert np.all(np.isfinite(proba)) and np.all(proba >= 0)
        # Unknown held-out category cannot create a newly learned indicator.
        assert pipeline.transform(heldout)[0, -3:].tolist() == [0.0, 0.0, 0.0]
    assert digest(blocks) == before


@pytest.mark.parametrize("order", [
    ("nir",), ("metadata", "image", "series", "nir"),
])
def test_full_native_encoders_preserve_regression_profile(order):
    blocks, recipe, schemas = case(order=order)
    labels = np.asarray([0, 1, 2] * 5)
    old = copy.deepcopy(recipe)
    old["model"] = {"method_id": "models.regularized.ridge", "params": {
        "alpha": 0.3, "center_x": True, "center_y": True, "scale_x": False,
    }}
    with MultimodalClassifierPipeline(recipe, schemas) as classifier, MultimodalPipeline(old, schemas) as regression:
        classifier.fit(blocks, labels)
        regression.fit(blocks, np.arange(15, dtype=np.float64))
        assert_allclose(classifier.transform(blocks), regression.transform(blocks), rtol=0, atol=0)
        assert classifier.export_state()[:4] == b"N4MC"
        assert regression.export_state()[:4] == b"N4MF"
        # A synthetic older writer header exercises compatibility without
        # regenerating or editing any historical checked-in fixture.
        old_writer = bytearray(regression.export_state())
        struct.pack_into("<I", old_writer, 12, 16)
        with MultimodalPipeline.from_state(reseal(old_writer), recipe=old, source_schemas=schemas) as replay:
            assert_allclose(replay.predict(blocks), regression.predict(blocks), rtol=0, atol=0)
        with pytest.raises(N4MError):
            MultimodalClassifierPipeline.from_state(regression.export_state(), recipe=recipe, source_schemas=schemas)


def test_portable_roundtrip_without_fit_and_external_table(monkeypatch):
    blocks, recipe, schemas = case()
    names = [-17, 5, 90]
    with MultimodalClassifierPipeline(recipe, schemas) as trained:
        trained.fit(blocks, names * 5)
        state = trained.export_state()
        labels, proba = trained.predict(blocks), trained.predict_proba(blocks)
    assert struct.unpack_from("<4I", state, 4) == (1, 2, 17, 0)

    def forbidden(*args, **kwargs):
        raise AssertionError("replay must not fit")

    monkeypatch.setattr(lib, "n4m_multimodal_classifier_fit", forbidden)
    monkeypatch.setattr(PLSLogistic, "fit", forbidden)
    with MultimodalClassifierPipeline.from_state(state, recipe=recipe, source_schemas=schemas, class_names=names) as replay:
        assert_array_equal(replay.classes_, names)
        assert_array_equal(replay.predict(blocks), labels)
        assert_allclose(replay.predict_proba(blocks), proba, rtol=0, atol=0)
        assert replay.export_state() == state
    with MultimodalClassifierPipeline.from_state(state, recipe=recipe, source_schemas=schemas) as raw:
        assert_array_equal(raw.classes_, [0, 1, 2])
        assert raw.label_names_ is None


def test_import_arbitrary_native_ids_maps_by_column_position():
    blocks, recipe, schemas = case()
    native_ids = np.ascontiguousarray([-19, 5, 90] * 5, dtype=np.int64)
    with MultimodalClassifierPipeline(recipe, schemas) as raw:
        views, keep, rows = _views(blocks, schemas, tuple(recipe["source_order"]))
        with _Context() as context:
            context.check(lib.n4m_multimodal_classifier_fit(
                context.handle, raw._handle, len(views), views,
                native_ids.ctypes.data_as(ct.POINTER(ct.c_int64)), rows,
            ), "direct native arbitrary-ID classifier")
        assert_array_equal(raw.classes_, [-19, 5, 90])
        predicted = raw.predict(blocks)
        state, proba = raw.export_state(), raw.predict_proba(blocks)
    originals = ["negative", "middle", "positive"]
    with MultimodalClassifierPipeline.from_state(state, recipe=recipe, source_schemas=schemas, class_names=originals) as mapped:
        positions = {-19: 0, 5: 1, 90: 2}
        assert_array_equal(mapped.predict(blocks), [originals[positions[int(value)]] for value in predicted])
        assert_array_equal(mapped.classes_, originals)
        assert_allclose(mapped.predict_proba(blocks), proba, rtol=0, atol=0)


@pytest.mark.parametrize("bad", [[True, False] * 7 + [True], [1, "1"] * 7 + [1], [0.0, 1.0] * 7 + [0.0], [0, 2**63] * 7 + [0]])
def test_ambiguous_labels_refused_without_losing_previous_fit(bad):
    blocks, recipe, schemas = case()
    with MultimodalClassifierPipeline(recipe, schemas) as pipeline:
        pipeline.fit(blocks, [-7, 42, -7] * 5)
        before, classes = pipeline.export_state(), pipeline.classes_
        with pytest.raises((ValueError, TypeError, N4MError)):
            pipeline.fit(blocks, bad)
        assert pipeline.export_state() == before
        assert_array_equal(pipeline.classes_, classes)
        with pytest.raises(N4MError):
            pipeline.fit(blocks, [42] * 15)
        assert pipeline.export_state() == before


@pytest.mark.parametrize("table", [["A", "A"], ["A", 1], [True, False], [0.0, 1.0], ["only"], [0, 2**63]])
def test_import_table_validation_and_transactional_cleanup(table):
    blocks, recipe, schemas = case()
    with MultimodalClassifierPipeline(recipe, schemas) as pipeline:
        state = pipeline.fit(blocks, ["A", "B", "A"] * 5).export_state()
    with pytest.raises((ValueError, TypeError)):
        MultimodalClassifierPipeline.from_state(state, recipe=recipe, source_schemas=schemas, class_names=table)


def reseal(data):
    data = bytearray(data)
    checksum = 0xCBF29CE484222325
    for value in data[:-8]:
        checksum = ((checksum ^ value) * 0x100000001B3) & ((1 << 64) - 1)
    struct.pack_into("<Q", data, len(data) - 8, checksum)
    return bytes(data)


def class_offset(state, count):
    cursor = 20
    cursor += 8 + struct.unpack_from("<Q", state, cursor)[0]
    for _ in range(count):
        cursor += 8 + struct.unpack_from("<Q", state, cursor)[0]
        categories = struct.unpack_from("<Q", state, cursor)[0]
        cursor += 8
        for _ in range(categories):
            cursor += 8 + struct.unpack_from("<Q", state, cursor)[0]
    return cursor


def test_malformed_state_classes_head_recipe_and_source_identity_refused():
    blocks, recipe, schemas = case()
    with MultimodalClassifierPipeline(recipe, schemas) as pipeline:
        state = pipeline.fit(blocks, [0, 1, 2] * 5).export_state()
    bad = bytearray(state)
    struct.pack_into("<q", bad, class_offset(state, 2) + 8, -1)
    mismatch = reseal(bad)  # sorted IDs, but they disagree with fitted head.
    bad = bytearray(state)
    struct.pack_into("<q", bad, class_offset(state, 2) + 16, 0)
    duplicate = reseal(bad)
    for corrupt in (state[:-1], state + b"\0", b"N4MF" + state[4:], mismatch, duplicate):
        with pytest.raises(N4MError):
            MultimodalClassifierPipeline.from_state(corrupt, recipe=recipe, source_schemas=schemas)
    wrong = copy.deepcopy(recipe)
    wrong["model"]["params"]["max_iter"] += 1
    with pytest.raises(N4MError):
        MultimodalClassifierPipeline.from_state(state, recipe=wrong, source_schemas=schemas)
    wrong_schema = copy.deepcopy(schemas)
    wrong_schema["nir"]["identity"] += ":changed"
    with pytest.raises(N4MError):
        MultimodalClassifierPipeline.from_state(state, recipe=recipe, source_schemas=wrong_schema)
    # A different valid N4ME head contradicts the unchanged outer recipe.
    alternate = copy.deepcopy(recipe)
    alternate["model"]["params"]["n_components"] = 1
    with MultimodalClassifierPipeline(alternate, schemas) as pipeline:
        other = pipeline.fit(blocks, [0, 1, 2] * 5).export_state()
    original_head = class_offset(state, 2) + 8 + 3 * 8
    other_head = class_offset(other, 2) + 8 + 3 * 8
    substituted = state[:original_head] + other[other_head:-8] + b"\0" * 8
    with pytest.raises(N4MError):
        MultimodalClassifierPipeline.from_state(reseal(substituted), recipe=recipe, source_schemas=schemas)


def test_recipe_profile_is_closed():
    blocks, recipe, schemas = case()
    for field, value in (("method_id", "models.regularized.ridge"), ("params", {"n_components": 2, "max_iter": 1000, "C": 1.0})):
        wrong = copy.deepcopy(recipe)
        wrong["model"][field] = value
        with pytest.raises(ValueError):
            MultimodalClassifierPipeline(wrong, schemas)
    with MultimodalClassifierPipeline(recipe, schemas) as pipeline:
        with pytest.raises(N4MError):
            pipeline.predict(blocks)


def test_close_once_without_transient_allocation(monkeypatch):
    _, recipe, schemas = case()
    pipeline = MultimodalClassifierPipeline(recipe, schemas)
    handle = pipeline._handle.value
    destroy = lib.n4m_multimodal_classifier_destroy
    calls = []

    def witnessed(value):
        calls.append(value.value)
        return destroy(value)

    monkeypatch.setattr(lib, "n4m_multimodal_classifier_destroy", witnessed)
    pipeline.close()
    pipeline.close()
    pipeline.__del__()
    assert calls == [handle]
    with pytest.raises(RuntimeError, match="closed"):
        pipeline.export_state()


def test_missing_classifier_abi_is_an_honest_refusal(monkeypatch):
    _, recipe, schemas = case()
    monkeypatch.setattr(lib, "n4m_check_abi_compatibility", lambda major, minor: Status.ERR_ABI_MISMATCH)
    with pytest.raises(N4MError, match="2.17"):
        MultimodalClassifierPipeline(recipe, schemas)


@pytest.mark.parametrize("operation", ["predict_proba", "decision_function", "transform"])
@pytest.mark.parametrize("rows,cols", [(257, 65536), (1, 16777217), (1, 0), (1, -1)])
def test_matrix_budget_refuses_before_numpy_allocation(monkeypatch, operation, rows, cols):
    from n4m.roles import _multimodal_classifier as binding

    pipeline = object.__new__(MultimodalClassifierPipeline)
    pipeline._closed, pipeline._handle = False, ct.c_void_p()
    pipeline._order, pipeline.source_schemas = (), {}
    monkeypatch.setattr(binding, "_views", lambda *args: ([], [], rows))

    def width(handle, pointer):
        ct.cast(pointer, ct.POINTER(ct.c_int64))[0] = cols
        return Status.OK

    def forbidden(*args, **kwargs):
        pytest.fail("matrix allocation must not precede native element-budget refusal")

    monkeypatch.setattr(lib, "n4m_multimodal_classifier_transform_cols", width)
    monkeypatch.setattr(lib, "n4m_multimodal_classifier_n_outputs", width)
    monkeypatch.setattr(binding.np, "empty", forbidden)
    with pytest.raises(ValueError, match="16777216"):
        pipeline._matrix(operation, {}, None)


def test_matrix_budget_exact_boundary_has_no_numeric_allocation():
    from n4m.roles._multimodal_classifier import _bounded_shape

    _bounded_shape(256, 65536)
    _bounded_shape(0, 65536)
    with pytest.raises(ValueError, match="16777216"):
        _bounded_shape(257, 65536)


def test_class_count_budget_refuses_before_numpy_allocation(monkeypatch):
    from n4m.roles import _multimodal_classifier as binding

    pipeline = object.__new__(MultimodalClassifierPipeline)
    pipeline._closed, pipeline._handle = False, ct.c_void_p()

    def classes(handle, buffer, capacity, pointer):
        ct.cast(pointer, ct.POINTER(ct.c_int64))[0] = 65537
        return Status.OK

    def forbidden(*args, **kwargs):
        pytest.fail("class-table allocation must not precede bound refusal")

    monkeypatch.setattr(lib, "n4m_multimodal_classifier_classes", classes)
    monkeypatch.setattr(binding.np, "empty", forbidden)
    with pytest.raises(ValueError, match="class count"):
        pipeline._class_ids()


@pytest.mark.parametrize("rows,classes,components,admitted", [
    (3072, 33, 127, True),  # ((33-1)*(127+1))^2 ==16777216.
    (3072, 33, 128, False),
    (3072, 512, 128, False),  # Actual Astra physical case; onehot fits budget.
    (8388608, 2, 1, True),  # Onehot/design each exactly16777216.
    (6000000, 2, 2, False),  # Onehot fits; logistic design does not.
    (2**63 - 1, 2, 1, False),
    (2, 2**64 - 1, 1, False),
    (2, 2, 2**63 - 1, False),
])
def test_classifier_working_set_boundary_and_overflow_without_allocation(rows, classes, components, admitted):
    from n4m.roles._multimodal_classifier import _bounded_working_set

    if admitted:
        _bounded_working_set(rows, classes, components)
    else:
        with pytest.raises(ValueError, match="PLS-logistic working set"):
            _bounded_working_set(rows, classes, components)


def test_physical_astra_case_refused_before_source_marshalling_and_native_fit(monkeypatch):
    from n4m.roles import _multimodal_classifier as binding

    blocks, recipe, schemas = case(order=("nir",))
    blocks["nir"] = np.sin(np.arange(3072 * 128, dtype=np.float64) * 0.017).reshape(3072, 128)
    schemas["nir"]["input_shape"] = [128]
    recipe["model"]["params"]["n_components"] = 128
    labels = np.arange(3072, dtype=np.int64) % 512
    before = digest(blocks)

    def forbidden(*args, **kwargs):
        pytest.fail("working-set admission must precede source marshalling/native FIT")

    with MultimodalClassifierPipeline(recipe, schemas) as pipeline:
        monkeypatch.setattr(binding, "_views", forbidden)
        monkeypatch.setattr(lib, "n4m_multimodal_classifier_fit", forbidden)
        with pytest.raises(ValueError, match="PLS-logistic working set"):
            pipeline.fit(blocks, labels)
        assert pipeline.label_names_ is None
    assert digest(blocks) == before


def test_working_set_refusal_preserves_fitted_owner_and_original_labels(monkeypatch):
    from n4m.roles import _multimodal_classifier as binding

    blocks, recipe, schemas = case(order=("nir",))
    with MultimodalClassifierPipeline(recipe, schemas) as pipeline:
        pipeline.fit(blocks, [-7, 42, -7] * 5)
        before = pipeline.export_state()
        expected = pipeline.predict(blocks)
        classes = pipeline.classes_
        oversized = {"nir": np.tile(blocks["nir"][:1], (1367, 1))}
        labels = np.arange(1367, dtype=np.int64)
        # 1367^2 fits onehot; ((1367-1)*3)^2 exceeds Hessian budget.
        with monkeypatch.context() as guarded:
            guarded.setattr(binding, "_views", lambda *args: pytest.fail("source marshalling after rejected admission"))
            guarded.setattr(lib, "n4m_multimodal_classifier_fit", lambda *args: pytest.fail("native FIT after rejected admission"))
            with pytest.raises(ValueError, match="PLS-logistic working set"):
                pipeline.fit(oversized, labels)
        assert pipeline.export_state() == before
        assert_array_equal(pipeline.classes_, classes)
        assert_array_equal(pipeline.predict(blocks), expected)
