"""Native role pipeline (ABI 2.14): Python facade over n4m_role_pipeline_*."""

from __future__ import annotations

import base64
import json
import pickle
import re
from pathlib import Path

import numpy as np
import pytest
from n4m import N4MError
from n4m.roles import (
    PLSLDA,
    SNV,
    PLSRegression,
    Ridge,
    RolePipeline,
    YOutlierFilter,
)


class _Named(np.ndarray):
    """A 2-D float table with column names (the facade reads ``.columns``)."""

    columns: list[str]


def frame(values, columns) -> _Named:
    table = np.asarray(values, dtype=np.float64).view(_Named)
    table.columns = list(columns)
    return table


FIXTURE = (
    Path(__file__).resolve().parents[3]
    / "parity"
    / "fixtures"
    / "role_pipeline_negative.json"
)
# Fixture outputs come from Linux x86-64; kernels drift by a few ulps on other
# platforms, so replays compare at 1e-9 (the exported bytes stay identical).
REPLAY_TOL = 1e-9
REFIT_TOL = 1e-9


@pytest.fixture(scope="module")
def doc() -> dict:
    return json.loads(FIXTURE.read_text(encoding="utf-8"))


def data(seed: int = 3, n: int = 40, p: int = 8):
    rng = np.random.default_rng(seed)
    X = 2.0 + rng.normal(size=(n, p))
    y = X @ rng.normal(size=p)
    return X, y


def test_recipe_is_validated_natively():
    with pytest.raises(N4MError, match="at least one step"):
        RolePipeline.from_steps([])
    with pytest.raises(N4MError, match="ends with one regressor or classifier"):
        RolePipeline.from_steps(["preprocessing.scatter.snv"])
    with pytest.raises(N4MError, match="only the last step"):
        RolePipeline.from_steps(
            ["models.regularized.ridge", "models.regularized.ridge"]
        )
    with pytest.raises(TypeError):
        RolePipeline.from_steps([("models.regularized.ridge", {"no_such_param": 1})])


def test_matches_the_chain_run_by_hand():
    X, y = data()
    y[4] = 80.0
    steps = [
        "filters.y_outlier",
        "n4m:preprocessing.scatter.snv",
        ("models.pls.pls_regression", {"n_components": 3}),
        {"class": "n4m:models.regularized.ridge", "params": {}},
    ]
    pipeline = RolePipeline.from_steps(steps).fit(X, y)
    keep = YOutlierFilter().fit(X, y).get_mask(X, y)
    assert not keep[4]
    snv = SNV().fit(X[keep])
    pls = PLSRegression(n_components=3).fit(snv.transform(X[keep]), y[keep])
    scores = pls.transform(snv.transform(X[keep]))
    ridge = Ridge().fit(scores, y[keep])
    expected = ridge.predict(pls.transform(snv.transform(X)))
    np.testing.assert_array_equal(pipeline.predict(X), expected)
    np.testing.assert_array_equal(
        pipeline.transform(X), pls.transform(snv.transform(X))
    )
    info = pipeline.steps_info_
    assert [s["role"] for s in info] == [
        "sample_filter",
        "transformer",
        "transformer",
        "regressor",
    ]
    assert [s["state_index"] for s in info] == [-1, 0, 1, 2]
    assert info[2]["n_features_out"] == 3


def test_multi_target_reaches_the_supervised_transformer():
    X, _ = data()
    Y = np.column_stack([X[:, 0] - X[:, 1], X[:, 2] + 0.5 * X[:, 3]])
    step = ("models.pls.pls_regression", {"n_components": 2})
    pipeline = RolePipeline.from_steps([step, step]).fit(X, Y)
    first = PLSRegression(n_components=2).fit(X, Y)
    last = PLSRegression(n_components=2).fit(first.transform(X), Y)
    np.testing.assert_array_equal(pipeline.predict(X), last.predict(first.transform(X)))
    assert pipeline.predict(X).shape == (X.shape[0], 2)


def test_classifier_labels_stay_facade_level():
    X, y = data()
    labels = np.where(y > np.median(y), "high", "low")
    pipeline = RolePipeline(
        [
            "preprocessing.scatter.snv",
            "models.pls.pls_regression",
            "models.classification.pls_lda",
        ]
    ).fit(X, labels)
    assert list(pipeline.classes_) == ["high", "low"]
    assert list(pipeline.label_names_) == ["high", "low"]
    assert RolePipeline(["models.regularized.ridge"]).fit(X, y).label_names_ is None
    assert set(pipeline.predict(X)) <= {"high", "low"}
    assert pipeline.decision_function(X).shape == (X.shape[0], 2)
    restored = RolePipeline.from_states(
        pipeline.steps, pipeline.export_states(), class_names=pipeline.classes_
    )
    np.testing.assert_array_equal(restored.predict(X), pipeline.predict(X))
    with pytest.raises(N4MError, match="step 2 \\(models.classification.pls_lda\\)"):
        pipeline._matrix("n4m_role_pipeline_predict", "n4m_role_pipeline_n_outputs", X)


def test_dataframe_column_identity_is_checked():
    X, y = data()
    table = frame(X, [f"w{j}" for j in range(X.shape[1])])
    pipeline = RolePipeline(
        ["preprocessing.scatter.snv", "models.regularized.ridge"]
    ).fit(table, y)
    assert list(pipeline.feature_names_in_) == list(table.columns)
    np.testing.assert_array_equal(
        pipeline.predict(table), pipeline.predict(X)
    )  # arrays are positional
    with pytest.raises(N4MError, match="the columns are reordered"):
        pipeline.predict(frame(np.asarray(X)[:, ::-1], table.columns[::-1]))
    with pytest.raises(N4MError, match="fitted with 'w0' there"):
        pipeline.predict(frame(X, ["other", *table.columns[1:]]))
    with pytest.raises(N4MError, match="7 columns; the pipeline was fitted on 8"):
        pipeline.predict(frame(np.asarray(X)[:, :-1], table.columns[:-1]))
    with pytest.raises(N4MError, match="duplicate feature name"):
        RolePipeline(["models.regularized.ridge"]).fit(frame(X, ["a"] * X.shape[1]), y)


def test_unused_input_is_refused():
    X, y = data()
    with pytest.raises(N4MError, match="not used by any step of the pipeline 'groups'"):
        RolePipeline(["models.regularized.ridge"]).fit(
            X, y, groups=np.arange(X.shape[0])
        )


def test_training_rows_export_needs_opt_in_and_states_round_trip():
    X, y = data()
    pipeline = RolePipeline(["preprocessing.scatter.snv", "models.pls.kernel"]).fit(
        X, y
    )
    assert [s["contains_training_rows"] for s in pipeline.steps_info_] == [False, True]
    with pytest.raises(N4MError, match="state retains training rows"):
        pipeline.export_states()
    states = pipeline.export_states(allow_training_rows=True)
    assert [(m, rows) for m, _, rows in states] == [
        ("preprocessing.scatter.snv", False),
        ("models.pls.kernel", True),
    ]
    restored = RolePipeline.from_states(pipeline.steps, states)
    np.testing.assert_array_equal(restored.predict(X), pipeline.predict(X))
    clone = pickle.loads(pickle.dumps(pipeline))
    np.testing.assert_array_equal(clone.predict(X), pipeline.predict(X))


def test_failed_refit_keeps_the_fitted_pipeline():
    X, y = data()
    pipeline = RolePipeline(["models.regularized.ridge"]).fit(X, y)
    before = pipeline.predict(X)
    with pytest.raises(N4MError):
        pipeline.fit(X, y, groups=np.arange(X.shape[0]))
    np.testing.assert_array_equal(pipeline.predict(X), before)


def _payloads(states: list) -> list[bytes]:
    return [
        base64.b64decode(s["n4me_base64"] if isinstance(s, dict) else s) for s in states
    ]


def test_shared_fixture_positive_pipelines_replay(doc):
    names = doc["feature_names"]
    X, X_test = np.asarray(doc["x_train"]), np.asarray(doc["x_test"])
    reg = doc["regression"]
    pipeline = RolePipeline.from_states(
        reg["steps"], _payloads(reg["states"]), feature_names=names
    )
    np.testing.assert_allclose(
        pipeline.predict(X_test), reg["predict"], rtol=REPLAY_TOL, atol=REPLAY_TOL
    )
    np.testing.assert_allclose(
        pipeline.transform(X_test), reg["transform"], rtol=REPLAY_TOL, atol=REPLAY_TOL
    )
    assert [m for m, _, _ in pipeline.export_states()] == [
        s["method_id"] for s in reg["states"]
    ]
    assert [p for _, p, _ in pipeline.export_states()] == _payloads(reg["states"])
    refit = RolePipeline(reg["steps"]).fit(frame(X, names), np.asarray(doc["y_train"]))
    np.testing.assert_allclose(
        refit.predict(X_test), reg["predict"], rtol=REFIT_TOL, atol=REFIT_TOL
    )

    cls = doc["classification"]
    pipeline = RolePipeline.from_states(
        cls["steps"],
        _payloads(cls["states"]),
        feature_names=names,
        class_names=cls["class_names"],
    )
    assert pipeline.predict(X_test).tolist() == cls["predict"]
    np.testing.assert_allclose(
        pipeline.decision_function(X_test),
        cls["decision_function"],
        rtol=REPLAY_TOL,
        atol=REPLAY_TOL,
    )
    refit = RolePipeline(cls["steps"]).fit(X, np.asarray(doc["labels_train"]))
    assert refit.predict(X_test).tolist() == cls["predict"]


def test_shared_fixture_negative_cases(doc):
    names = doc["feature_names"]
    X, X_test = np.asarray(doc["x_train"]), np.asarray(doc["x_test"])
    reg = doc["regression"]
    fitted = RolePipeline.from_states(
        reg["steps"], _payloads(reg["states"]), feature_names=names
    )
    for case in doc["cases"]:
        stage = case["stage"]
        if stage == "fit":
            pipeline = RolePipeline(case["steps"]).fit(X, np.asarray(doc[case["y"]]))
            np.testing.assert_allclose(
                pipeline.predict(X_test),
                case["predict"],
                rtol=REFIT_TOL,
                atol=REFIT_TOL,
            )
            continue
        with pytest.raises(N4MError) as error:
            if stage == "create":
                RolePipeline.from_steps(case["steps"])
            elif stage == "import":
                RolePipeline.from_states(case["steps"], _payloads(case["states"]))
            elif stage == "predict":
                columns = (
                    names[:-1]
                    if case.get("drop_last_column")
                    else case["feature_names"]
                )
                fitted.predict(frame(X_test[:, : len(columns)], columns))
            else:
                RolePipeline(case["steps"]).fit(
                    X, np.asarray(doc[case["y"]])
                ).export_states()
        assert error.value.status == case["status"], case["name"]
        assert case["message"] in str(error.value), case["name"]


def _inject(values: list, case: dict) -> list:
    """The case's labels / class_names with its non-finite value (JSON has none)."""
    values = list(values)
    if "inject" in case:
        values[case["inject"]["at"]] = float(case["inject"]["value"])
    return values


def test_shared_fixture_label_cases(doc):
    X, X_test = np.asarray(doc["x_train"]), np.asarray(doc["x_test"])
    for case in doc["label_cases"]:
        if case["stage"] == "import":
            source = doc[case["pipeline"]]
            class_names = _inject(case["class_names"], case)

            def load(source=source, class_names=class_names):
                return RolePipeline.from_states(
                    source["steps"],
                    _payloads(source["states"]),
                    class_names=class_names,
                )

            if case.get("accept"):
                assert load().predict(X_test).tolist() == source["predict"], case[
                    "name"
                ]
                continue
            with pytest.raises(ValueError, match=re.escape(case["message"])):
                load()
            continue
        labels = _inject(case["labels"], case)
        steps = doc["classification"]["steps"]
        with pytest.raises(ValueError, match=re.escape(case["message"])):
            RolePipeline(steps).fit(X, labels)
        # The estimator-level classifiers encode labels the same way.
        with pytest.raises(ValueError, match=re.escape(case["message"])):
            PLSLDA().fit(X, labels)


def test_shared_fixture_name_cases(doc):
    names = doc["feature_names"]
    X, X_test = np.asarray(doc["x_train"]), np.asarray(doc["x_test"])
    reg = doc["regression"]
    fitted = RolePipeline.from_states(
        reg["steps"], _payloads(reg["states"]), feature_names=names
    )
    for case in doc["name_cases"]:
        columns = case["feature_names"]
        with pytest.raises(ValueError, match=case["message"]):
            if case["stage"] == "fit":
                RolePipeline(case["steps"]).fit(
                    frame(X, columns), np.asarray(doc[case["y"]])
                )
            elif case["stage"] == "import":
                RolePipeline.from_states(
                    reg["steps"], _payloads(reg["states"]), feature_names=columns
                )
            else:
                fitted.predict(frame(X_test, columns))


def test_label_tables_round_trip_numbers_and_pickles():
    X, y = data()
    labels = np.where(y > np.median(y), 2.5, 0.5)
    steps = ["models.classification.pls_lda"]
    pipeline = RolePipeline(steps).fit(X, labels)
    assert pipeline.label_names_.tolist() == [0.5, 2.5]
    restored = RolePipeline.from_states(
        steps, pipeline.export_states(), class_names=pipeline.label_names_.tolist()
    )
    np.testing.assert_array_equal(restored.predict(X), pipeline.predict(X))
    clone = pickle.loads(pickle.dumps(pipeline))
    np.testing.assert_array_equal(clone.predict(X), pipeline.predict(X))
    # Integer labels are the class ids: no table, and uint64 beyond int64 is
    # refused instead of wrapping to a negative id.
    ids = np.where(y > np.median(y), 7, 3)
    assert RolePipeline(steps).fit(X, ids).label_names_ is None
    with pytest.raises(ValueError, match="class labels must fit int64"):
        RolePipeline(steps).fit(X, ids.astype(np.uint64) + np.uint64(2**63))
    # Beside a fraction, an integer beyond 2^53 would round and merge labels.
    mixed = [2**53 + 1 if v > np.median(y) else 0.5 for v in y]
    mixed[0] = 2**53
    with pytest.raises(ValueError, match="exactly representable as float64"):
        RolePipeline(steps).fit(X, np.asarray(mixed, dtype=object))
    with pytest.raises(
        ValueError, match="class labels must be integers, finite numbers"
    ):
        RolePipeline(steps).fit(X, ids > 5)


def test_integer_fit_inputs_must_fit_int64():
    X, y = data()
    groups = np.full(X.shape[0], 2**63, dtype=np.uint64)
    with pytest.raises(ValueError, match="groups must fit int64"):
        RolePipeline(["models.regularized.ridge"]).fit(X, y, groups=groups)


def test_zero_rows_give_empty_outputs_whatever_the_steps():
    X, y = data()
    for steps in (
        ["models.regularized.ridge"],
        ["preprocessing.scatter.snv", "models.regularized.ridge"],
        [
            "preprocessing.scatter.snv",
            "models.pls.pls_regression",
            "models.regularized.ridge",
        ],
    ):
        pipeline = RolePipeline(steps).fit(X, y)
        assert pipeline.predict(X[:0]).shape == (0,)
        assert pipeline.transform(X[:0]).shape == (0, pipeline.transform(X).shape[1])
        with pytest.raises(N4MError, match="fitted on 8"):
            pipeline.predict(X[:0, :-1])
    labels = np.where(y > np.median(y), "high", "low")
    classifier = RolePipeline(
        ["preprocessing.scatter.snv", "models.classification.pls_lda"]
    ).fit(X, labels)
    assert classifier.predict(X[:0]).shape == (0,)
    assert classifier.decision_function(X[:0]).shape == (0, 2)
