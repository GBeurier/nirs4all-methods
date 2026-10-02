# SPDX-License-Identifier: CECILL-2.1
"""Actual native raw encoders, independent sklearn oracle and state closure."""

from __future__ import annotations

import copy
import struct

import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from n4m import MultimodalPipeline
from n4m.roles import StandardScale


def raw_case(dtype=np.float64):
    # Binding-only diagnostic fixture, deliberately outside U07's exact shapes.
    rng = np.random.default_rng(901)
    blocks = {
        "nir": rng.normal(size=(15, 5)).astype(dtype),
        "image": rng.normal(size=(15, 2, 3, 2)).astype(dtype),
        "series": rng.normal(size=(15, 4, 2)).astype(dtype),
        "metadata": np.column_stack((np.arange(15) / 4, ["é", "A", "猫"] * 5)).astype(
            object
        ),
    }
    y = rng.normal(size=15)
    representations = ("signal_1d", "rgb_image", "series_mv", "tabular_mixed")
    schemas = {
        name: {
            "representation_id": representation,
            "input_shape": list(value.shape[1:]),
            "dtype": str(value.dtype),
            "identity": "original-io-descriptor:" + name + ":axes:µ",
        }
        for (name, value), representation in zip(
            blocks.items(), representations, strict=True
        )
    }
    recipe = {
        "schema_version": 1,
        "fusion": "early",
        "source_order": list(blocks),
        "encoders": {
            "nir": {"kind": "standard_scaler", "with_mean": True, "with_std": True},
            "image": {
                "kind": "tensor_pca",
                "n_components": 3,
                "whiten": False,
                "random_state": 42,
            },
            "series": {
                "kind": "tensor_pca",
                "n_components": 2,
                "whiten": False,
                "random_state": 31,
            },
            "metadata": {
                "kind": "column_transformer",
                "numeric_columns": [0],
                "categorical_columns": [1],
                "with_mean": True,
                "with_std": True,
                "handle_unknown": "ignore",
                "sparse_output": False,
                "drop": None,
            },
        },
        "source_weights": {"nir": 1.0, "image": 0.5, "series": 1.5, "metadata": 0.75},
        "model": {
            "method_id": "models.regularized.ridge",
            "params": {
                "alpha": 0.3,
                "center_x": True,
                "center_y": True,
                "scale_x": False,
            },
        },
    }
    return blocks, y, recipe, schemas


def oracle(blocks, y, recipe):
    transforms = [
        StandardScaler(),
        PCA(recipe["encoders"]["image"]["n_components"], svd_solver="full"),
        PCA(recipe["encoders"]["series"]["n_components"], svd_solver="full"),
        StandardScaler(),
        OneHotEncoder(handle_unknown="ignore", sparse_output=False),
    ]
    values = [
        blocks["nir"].astype(np.float64),
        blocks["image"].reshape(len(y), -1).astype(np.float64),
        blocks["series"].reshape(len(y), -1).astype(np.float64),
        blocks["metadata"][:, :1].astype(np.float64),
        blocks["metadata"][:, 1:2],
    ]
    learned = [t.fit(v) for t, v in zip(transforms, values, strict=True)]

    def encode(new):
        raw = [
            new["nir"].astype(np.float64),
            new["image"].reshape(len(new["nir"]), -1).astype(np.float64),
            new["series"].reshape(len(new["nir"]), -1).astype(np.float64),
            new["metadata"][:, :1].astype(np.float64),
            new["metadata"][:, 1:2],
        ]
        z = [t.transform(v) for t, v in zip(learned, raw, strict=True)]
        z[3] = np.column_stack((z[3], z[4]))
        return np.column_stack(
            [
                part * recipe["source_weights"][name]
                for name, part in zip(recipe["source_order"], z[:4], strict=True)
            ]
        )

    ridge = Ridge(alpha=recipe["model"]["params"]["alpha"]).fit(encode(blocks), y)
    return encode, ridge, learned


def subset_oracle(blocks, y, recipe):
    """Independent selected-name traversal; fit every learned state on train only."""
    learned = {}
    widths = {}
    for name in recipe["source_order"]:
        raw = blocks[name]
        settings = recipe["encoders"][name]
        if name == "metadata":
            scaler = StandardScaler().fit(raw[:, :1].astype(np.float64))
            categories = OneHotEncoder(handle_unknown="ignore", sparse_output=False).fit(raw[:, 1:2])
            learned[name] = (scaler, categories)
            widths[name] = 1 + len(categories.categories_[0])
        elif name == "nir":
            learned[name] = StandardScaler().fit(raw.astype(np.float64))
            widths[name] = raw.shape[1]
        else:
            learned[name] = PCA(settings["n_components"], svd_solver="full").fit(
                raw.reshape(len(y), -1).astype(np.float64)
            )
            widths[name] = settings["n_components"]

    def encode(new):
        parts = []
        for name in recipe["source_order"]:
            raw = new[name]
            if name == "metadata":
                scaler, categories = learned[name]
                part = np.column_stack((
                    scaler.transform(raw[:, :1].astype(np.float64)),
                    categories.transform(raw[:, 1:2]),
                ))
            else:
                part = learned[name].transform(raw.reshape(len(raw), -1).astype(np.float64))
            parts.append(part * recipe["source_weights"][name])
        return np.column_stack(parts)

    ridge = Ridge(alpha=recipe["model"]["params"]["alpha"]).fit(encode(blocks), y)
    return encode, ridge, widths


def selected_case(order, dtype=np.float64):
    blocks, y, recipe, schemas = raw_case(dtype)
    recipe["source_order"] = list(order)
    recipe["encoders"] = {name: recipe["encoders"][name] for name in order}
    recipe["source_weights"] = {name: recipe["source_weights"][name] for name in order}
    return {name: blocks[name] for name in order}, y, recipe, {name: schemas[name] for name in order}


@pytest.mark.parametrize("order", [
    ("nir",), ("metadata",), ("series", "nir"),
    ("metadata", "image", "nir"), ("metadata", "series", "image", "nir"),
])
@pytest.mark.parametrize("dtype", [np.float64, np.float32])
def test_selected_order_weighted_early_fusion_train_only_oracle_and_replay(order, dtype):
    blocks, y, recipe, schemas = selected_case(order, dtype)
    heldout = {name: value[:3].copy() for name, value in blocks.items()}
    if "metadata" in heldout:
        heldout["metadata"][:, 1] = ["🚀", "A", "é"]
    encode, ridge, widths = subset_oracle(blocks, y, recipe)
    # Named mappings may arrive in any insertion order; the recipe fixes fusion order.
    with MultimodalPipeline(recipe, dict(reversed(list(schemas.items())))) as native:
        native.fit(dict(reversed(list(blocks.items()))), y)
        actual, expected = native.transform(blocks), encode(blocks)
        start = 0
        for name in order:
            end = start + widths[name]
            a, b = actual[:, start:end], expected[:, start:end]
            if name in ("image", "series"):
                assert_allclose(a @ a.T, b @ b.T, rtol=1e-8, atol=1e-8)
            else:
                assert_allclose(a, b, rtol=1e-11, atol=1e-11)
            if name == "metadata":
                assert_array_equal(native.transform(heldout)[0, start + 1:end], np.zeros(widths[name] - 1))
            start = end
        assert_allclose(native.predict(blocks), ridge.predict(expected), rtol=1e-8, atol=1e-8)
        assert_allclose(native.predict(heldout), ridge.predict(encode(heldout)), rtol=1e-8, atol=1e-8)
        state = native.export_state()
        assert struct.unpack_from("<I", state, 4)[0] == 1
        assert struct.unpack_from("<I", state, 36)[0] == len(order)
        with MultimodalPipeline.from_state(state, recipe=recipe, source_schemas=schemas) as replay:
            assert_array_equal(replay.predict(heldout), native.predict(heldout))
            assert replay.export_state() == state
        altered = copy.deepcopy(recipe)
        altered["source_weights"][order[0]] += 0.25
        with pytest.raises(RuntimeError, match="recipe/source schema"):
            MultimodalPipeline.from_state(state, recipe=altered, source_schemas=schemas)
        if len(order) > 1:
            reordered = copy.deepcopy(recipe)
            reordered["source_order"] = list(reversed(order))
            with pytest.raises(RuntimeError, match="recipe/source schema"):
                MultimodalPipeline.from_state(state, recipe=reordered, source_schemas=schemas)
            shortened = copy.deepcopy(recipe)
            shortened["source_order"] = list(order[:-1])
            shortened["encoders"].pop(order[-1])
            shortened["source_weights"].pop(order[-1])
            with pytest.raises(RuntimeError, match="recipe/source schema"):
                MultimodalPipeline.from_state(
                    state, recipe=shortened,
                    source_schemas={name: schemas[name] for name in order[:-1]},
                )
        changed = copy.deepcopy(schemas)
        changed[order[0]]["identity"] += ":rebound"
        with pytest.raises(RuntimeError, match="recipe/source schema"):
            MultimodalPipeline.from_state(state, recipe=recipe, source_schemas=changed)


def test_absent_encoders_are_never_fit_but_selected_zero_weight_is_fit():
    blocks, y, recipe, schemas = selected_case(("nir",))
    with MultimodalPipeline(recipe, schemas) as native:
        native.fit({"nir": blocks["nir"][:1]}, y[:1])
        assert native.transform({"nir": blocks["nir"][:1]}).shape == (1, 5)
    blocks, y, recipe, schemas = selected_case(("image",))
    recipe["source_weights"]["image"] = 0.0
    with MultimodalPipeline(recipe, schemas) as native:
        with pytest.raises(RuntimeError, match="component count"):
            native.fit({"image": blocks["image"][:1]}, y[:1])


@pytest.mark.parametrize("order", [[], ["nir", "nir"], ["other"], ["nir"] * 5, "nir"])
def test_subset_recipe_rejects_invalid_order_before_native_fit(order):
    _, _, recipe, schemas = raw_case()
    recipe["source_order"] = order
    with pytest.raises(ValueError, match="source_order"):
        MultimodalPipeline(recipe, schemas)


def test_selected_schema_encoder_and_weight_maps_require_exact_coverage():
    blocks, y, recipe, schemas = selected_case(("nir",))
    full_blocks, _, full_recipe, full_schemas = raw_case()
    for field in ("encoders", "source_weights"):
        altered = copy.deepcopy(recipe)
        altered[field]["image"] = full_recipe[field]["image"]
        with pytest.raises(ValueError, match=field):
            MultimodalPipeline(altered, schemas)
    with pytest.raises(ValueError, match="source_schemas"):
        MultimodalPipeline(recipe, full_schemas)
    with MultimodalPipeline(recipe, schemas) as native:
        with pytest.raises(ValueError, match="blocks"):
            native.fit(full_blocks, y)
        native.fit(blocks, y)


def test_population_scaler_constants_large_offset_and_portable_state():
    x = np.column_stack(
        (np.full(37, 0.1), 2.0**40 + np.arange(37), np.arange(37) ** 2 / 7)
    )
    reference = StandardScaler().fit(x)
    native = StandardScale().fit(x)
    assert_allclose(native.transform(x), reference.transform(x), rtol=1e-12, atol=1e-12)
    assert_array_equal(native.transform(x)[:, 0], np.zeros(37))
    replay = StandardScale.from_n4me(native.to_n4me())
    assert_array_equal(replay.transform(x), native.transform(x))


@pytest.mark.parametrize("dtype", [np.float64, np.float32])
def test_actual_native_early_fusion_and_train_only_unicode_vocabulary(dtype):
    blocks, y, recipe, schemas = raw_case(dtype)
    heldout = {name: value[:3].copy() for name, value in blocks.items()}
    heldout["metadata"][:, 1] = ["🚀", "A", "é"]
    encode, ridge, learned = oracle(blocks, y, recipe)
    with MultimodalPipeline(recipe, schemas) as native:
        native.fit(blocks, y)
        native_z, oracle_z = native.transform(blocks), encode(blocks)
        # Native/sklearn SVD sign conventions may differ; compare PCA Gram
        # matrices (actual retained subspaces), leaving all other values exact.
        for start, width in [(5, 3), (8, 2)]:
            assert_allclose(
                native_z[:, start : start + width]
                @ native_z[:, start : start + width].T,
                oracle_z[:, start : start + width]
                @ oracle_z[:, start : start + width].T,
                rtol=1e-8,
                atol=1e-8,
            )
        assert_allclose(native_z[:, :5], oracle_z[:, :5], rtol=1e-11, atol=1e-11)
        assert_allclose(native_z[:, 10:], oracle_z[:, 10:], rtol=1e-11, atol=1e-11)
        assert_array_equal(native.transform(heldout)[0, -3:], np.zeros(3))
        assert_array_equal(learned[-1].categories_[0], ["A", "é", "猫"])
        assert_allclose(
            native.predict(blocks), ridge.predict(encode(blocks)), rtol=1e-8, atol=1e-8
        )
        assert_allclose(
            native.predict(heldout),
            ridge.predict(encode(heldout)),
            rtol=1e-8,
            atol=1e-8,
        )
        state = native.export_state()
        assert state[:4] == b"N4MF" and struct.unpack_from("<I", state, 4)[0] == 1
        assert len(state) <= 64 * 1024 * 1024
        with MultimodalPipeline.from_state(
            state, recipe=recipe, source_schemas=schemas
        ) as replay:
            assert_array_equal(replay.predict(heldout), native.predict(heldout))
            assert_array_equal(replay.transform(heldout), native.transform(heldout))
            assert replay.export_state() == state


@pytest.mark.parametrize("profile", ["rank_deficient", "all_zero_weights", "single_category"])
def test_zero_alpha_least_squares_oracle_and_transactional_state(profile):
    blocks, y, recipe, schemas = raw_case()
    recipe["model"]["params"]["alpha"] = 0.0
    if profile == "all_zero_weights":
        recipe["source_weights"] = dict.fromkeys(recipe["source_order"], 0.0)
    elif profile == "single_category":
        blocks["metadata"][:, 1] = "é"
    heldout = {name: value[:3].copy() for name, value in blocks.items()}
    heldout["metadata"][:, 1] = ["🚀", "é", "A"]
    # Fit the independent encoders using the existing oracle; the positive
    # penalty here affects only its discarded model, never the tested recipe.
    encoder_recipe = copy.deepcopy(recipe)
    encoder_recipe["model"]["params"]["alpha"] = 0.3
    encode, _, _ = oracle(blocks, y, encoder_recipe)
    fused = encode(blocks)
    centered = fused - fused.mean(axis=0)
    assert np.linalg.matrix_rank(centered) < fused.shape[1]
    reference = LinearRegression().fit(fused, y)
    with MultimodalPipeline(recipe, schemas) as native:
        native.fit(blocks, y)
        assert_allclose(native.predict(blocks), reference.predict(fused), rtol=1e-8, atol=1e-8)
        expected = reference.predict(encode(heldout))
        assert_allclose(native.predict(heldout), expected, rtol=1e-8, atol=1e-8)
        if profile == "all_zero_weights":
            assert_array_equal(native.transform(blocks), np.zeros_like(fused))
            assert_allclose(native.predict(heldout), np.full(3, y.mean()), rtol=1e-12, atol=1e-12)
        elif profile == "single_category":
            assert_array_equal(native.transform(heldout)[0, -1:], [0.0])
        before = native.predict(heldout)
        state = native.export_state()
        with MultimodalPipeline.from_state(state, recipe=recipe, source_schemas=schemas) as replay:
            assert_array_equal(replay.predict(heldout), before)
            assert replay.export_state() == state
        invalid_y = y.copy()
        invalid_y[0] = np.nan
        with pytest.raises(Exception, match="NaN|Inf"):
            native.fit(blocks, invalid_y)
        assert_array_equal(native.predict(heldout), before)
        assert native.export_state() == state


def test_length_delimited_embedded_nul_categories_remain_distinct_after_replay():
    blocks, y, recipe, schemas = raw_case()
    blocks["metadata"][:, 1] = ["A", "A\0é", "猫"] * 5
    heldout = {name: value[:3].copy() for name, value in blocks.items()}
    heldout["metadata"][:, 1] = ["A\0é", "A\0éx", "A"]
    encode, reference, learned = oracle(blocks, y, recipe)
    assert_array_equal(learned[-1].categories_[0], ["A", "A\0é", "猫"])
    with MultimodalPipeline(recipe, schemas) as native:
        native.fit(blocks, y)
        weight = recipe["source_weights"]["metadata"]
        assert_array_equal(native.transform(heldout)[:, -3:], np.array([[0, 1, 0], [0, 0, 0], [1, 0, 0]]) * weight)
        assert_allclose(native.predict(heldout), reference.predict(encode(heldout)), rtol=1e-8, atol=1e-8)
        state = native.export_state()
        with MultimodalPipeline.from_state(state, recipe=recipe, source_schemas=schemas) as replay:
            assert_array_equal(replay.transform(heldout), native.transform(heldout))
            assert_array_equal(replay.predict(heldout), native.predict(heldout))
            assert replay.export_state() == state


@pytest.mark.parametrize("change", ["identity", "shape", "dtype", "representation"])
def test_independently_supplied_schema_must_match_complete_native_state(change):
    blocks, y, recipe, schemas = raw_case()
    with MultimodalPipeline(recipe, schemas) as native:
        native.fit(blocks, y)
        state = native.export_state()
        different = copy.deepcopy(schemas)
        field, value = {
            "identity": ("identity", "changed-axis-order"),
            "shape": ("input_shape", [3, 2, 2]),
            "dtype": ("dtype", "float32"),
            "representation": ("representation_id", "series_mv"),
        }[change]
        different["image"][field] = value
        with pytest.raises(
            Exception, match="schema|recipe|identity|pattern|shape|dtype"
        ):
            native.predict(blocks, source_schemas=different)
        with pytest.raises(Exception, match="schema|recipe|pattern"):
            MultimodalPipeline.from_state(
                state, recipe=recipe, source_schemas=different
            )


def test_failed_refit_and_import_cleanup_preserve_previous_fitted_state():
    blocks, y, recipe, schemas = raw_case()
    with MultimodalPipeline(recipe, schemas) as native:
        native.fit(blocks, y)
        before = native.predict(blocks)
        bad = {name: value.copy() for name, value in blocks.items()}
        bad["series"][0, 0, 0] = np.nan
        with pytest.raises(Exception, match="NaN|Inf"):
            native.fit(bad, y)
        assert_array_equal(native.predict(blocks), before)
        state = native.export_state()
        for broken in (
            state[:-1],
            state[:60] + bytes([state[60] ^ 1]) + state[61:],
            state + b"x",
        ):
            with pytest.raises(Exception, match="checksum|size|magic"):
                MultimodalPipeline.from_state(
                    broken, recipe=recipe, source_schemas=schemas
                )
        altered = copy.deepcopy(recipe)
        altered["source_weights"]["image"] = 0.6
        with pytest.raises(Exception, match="recipe|schema"):
            MultimodalPipeline.from_state(state, recipe=altered, source_schemas=schemas)
        assert_array_equal(native.predict(blocks), before)
    native.close()
    with pytest.raises(RuntimeError, match="closed"):
        native.predict(blocks)


def test_rows_bounds_and_unsupported_declarations_are_refused_without_fallback():
    blocks, y, recipe, schemas = raw_case()
    with MultimodalPipeline(recipe, schemas) as native:
        with pytest.raises(Exception, match="not fitted"):
            native.predict(blocks)
        short = dict(blocks, image=blocks["image"][:-1])
        with pytest.raises(ValueError, match="row"):
            native.fit(short, y)
    wrong = copy.deepcopy(recipe)
    wrong["encoders"]["image"]["n_components"] = 12
    with MultimodalPipeline(wrong, schemas) as native:
        short = {name: value[:4] for name, value in blocks.items()}
        with pytest.raises(Exception, match="component count"):
            native.fit(short, y[:4])
    for key, value in [("whiten", True), ("random_state", -1), ("n_components", 0)]:
        wrong = copy.deepcopy(recipe)
        wrong["encoders"]["image"][key] = value
        with pytest.raises(Exception):
            MultimodalPipeline(wrong, schemas)
