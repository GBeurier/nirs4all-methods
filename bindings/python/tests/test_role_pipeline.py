"""Native role pipeline (ABI 2.14): Python facade over n4m_role_pipeline_*."""

from __future__ import annotations

import base64
import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from n4m import N4MError
from n4m.roles import (
    SNV,
    PLSRegression,
    Ridge,
    RolePipeline,
    YOutlierFilter,
)

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
    frame = pd.DataFrame(X, columns=[f"w{j}" for j in range(X.shape[1])])
    pipeline = RolePipeline(
        ["preprocessing.scatter.snv", "models.regularized.ridge"]
    ).fit(frame, y)
    assert list(pipeline.feature_names_in_) == list(frame.columns)
    np.testing.assert_array_equal(
        pipeline.predict(frame), pipeline.predict(X)
    )  # arrays are positional
    with pytest.raises(N4MError, match="the columns are reordered"):
        pipeline.predict(frame[frame.columns[::-1]])
    with pytest.raises(N4MError, match="fitted with 'w0' there"):
        pipeline.predict(frame.rename(columns={"w0": "other"}))
    with pytest.raises(N4MError, match="7 columns; the pipeline was fitted on 8"):
        pipeline.predict(frame.iloc[:, :-1])
    with pytest.raises(N4MError, match="duplicate feature name"):
        RolePipeline(["models.regularized.ridge"]).fit(
            pd.DataFrame(X, columns=["a"] * X.shape[1]), y
        )


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
    refit = RolePipeline(reg["steps"]).fit(
        pd.DataFrame(X, columns=names), np.asarray(doc["y_train"])
    )
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
                fitted.predict(pd.DataFrame(X_test[:, : len(columns)], columns=columns))
            else:
                RolePipeline(case["steps"]).fit(
                    X, np.asarray(doc[case["y"]])
                ).export_states()
        assert error.value.status == case["status"], case["name"]
        assert case["message"] in str(error.value), case["name"]
