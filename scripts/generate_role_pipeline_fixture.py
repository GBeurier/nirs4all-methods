#!/usr/bin/env python3
"""Write the shared role-pipeline fixture (native ``n4m_role_pipeline_*``, ABI 2.14).

Two positive pipelines fitted by the Python binding (a regression recipe with
a sample filter and a supervised transformer, and a classification recipe
with string labels) carry their N4ME states and outputs; the Python, R,
JS/WASM and Rust suites import them, predict identically, refit the recipe on
the same data and reproduce the Python fit. The negative cases are the
pipeline-level refusals every binding must report with the same native
status and message.

    N4M_LIB_PATH=... PYTHONPATH=bindings/python/src \\
        python scripts/generate_role_pipeline_fixture.py
"""

from __future__ import annotations

import base64
import json
from pathlib import Path

import numpy as np

import n4m
from n4m.roles import RolePipeline

REPO = Path(__file__).resolve().parents[1]
OUTPUT = REPO / "parity" / "fixtures" / "role_pipeline_negative.json"
# Same content as R source: the R package tests read no JSON.
R_OUTPUT = (
    REPO / "bindings" / "r" / "n4m" / "tests" / "testthat" / "fixture-role-pipeline.R"
)

INVALID_ARGUMENT = 1
SHAPE_MISMATCH = 3


def step(method_id: str, **params) -> dict:
    """A recipe token, as the v8 envelopes write it."""
    return {"class": f"n4m:{method_id}", "params": params}


REGRESSION = [
    step("filters.y_outlier"),
    step("preprocessing.scatter.snv"),
    step("models.pls.pls_regression", n_components=3),
    step("models.regularized.ridge"),
]
CLASSIFICATION = [
    step("preprocessing.scatter.snv"),
    step("models.pls.pls_regression", n_components=2),
    step("models.classification.pls_lda"),
]


def dataset() -> dict:
    rng = np.random.default_rng(20260927)
    n, p = 48, 12
    scores = rng.normal(size=(n, 2)) * np.array([3.0, 2.0])
    X = 2.0 + scores @ rng.normal(size=(2, p)) * 0.1 + 0.01 * rng.normal(size=(n, p))
    y = scores[:, 0] - 0.5 * scores[:, 1] + 0.05 * rng.normal(size=n)
    y[5] = 40.0  # one gross outlier for the target filter
    y2 = np.column_stack([y, scores[:, 1] + 0.05 * rng.normal(size=n)])
    labels = np.array(["low", "mid", "high"])[np.digitize(scores[:, 0], [-1.5, 1.5])]
    train, test = slice(0, 36), slice(36, n)
    return {
        "feature_names": [f"nm{1000 + 2 * j}" for j in range(p)],
        "x_train": X[train],
        "y_train": y[train],
        "y2_train": y2[train],
        "labels_train": labels[train],
        "x_test": X[test],
    }


def positive(steps: list, X, y, names, x_test) -> tuple[dict, RolePipeline]:
    import pandas as pd

    pipeline = RolePipeline.from_steps(steps).fit(pd.DataFrame(X, columns=names), y)
    states = pipeline.export_states()
    case = {
        "steps": steps,
        "feature_names": names,
        "states": [
            {
                "method_id": method_id,
                "n4me_base64": base64.b64encode(payload).decode("ascii"),
                "contains_training_rows": rows,
            }
            for method_id, payload, rows in states
        ],
    }
    prediction = pipeline.predict(x_test)
    case["predict"] = prediction.tolist()
    return case, pipeline


def main() -> None:
    data = dataset()
    names = data["feature_names"]
    regression, reg = positive(
        REGRESSION, data["x_train"], data["y_train"], names, data["x_test"]
    )
    regression["transform"] = reg.transform(data["x_test"]).tolist()
    classification, cls = positive(
        CLASSIFICATION, data["x_train"], data["labels_train"], names, data["x_test"]
    )
    classification["class_names"] = cls.classes_.tolist()
    classification["decision_function"] = cls.decision_function(data["x_test"]).tolist()

    multi = RolePipeline.from_steps(
        [step("models.pls.pls_regression", n_components=2)] * 2
    ).fit(data["x_train"], data["y2_train"])
    cppls = RolePipeline.from_steps([step("models.pls.cppls", n_components=2)]).fit(
        data["x_train"], data["y_train"]
    )
    cppls_state = base64.b64encode(cppls.export_states()[0][1]).decode("ascii")
    reg_states = [s["n4me_base64"] for s in regression["states"]]
    swapped = list(names)
    swapped[2], swapped[5] = swapped[5], swapped[2]

    cases = [
        {
            "name": "empty_recipe",
            "stage": "create",
            "steps": [],
            "status": INVALID_ARGUMENT,
            "message": "at least one step",
        },
        {
            "name": "wrong_role_order",
            "stage": "create",
            "steps": [
                step("preprocessing.scatter.snv"),
                step("filters.y_outlier"),
                step("models.pls.cppls"),
            ],
            "status": INVALID_ARGUMENT,
            "message": "step 1 (filters.y_outlier): sample filters come before",
        },
        {
            "name": "missing_terminal",
            "stage": "create",
            "steps": [
                step("preprocessing.scatter.snv"),
                step("preprocessing.scatter.msc"),
            ],
            "status": INVALID_ARGUMENT,
            "message": "step 1 (preprocessing.scatter.msc): a role pipeline ends with one regressor or classifier",
        },
        {
            "name": "recipe_param_differs_from_state",
            "stage": "import",
            "steps": [step("models.pls.cppls", n_components=1)],
            "states": [cppls_state],
            "status": INVALID_ARGUMENT,
            "message": "parameter 'n_components' is 2 in the state but 1 in the recipe",
        },
        {
            "name": "method_mismatch",
            "stage": "import",
            "steps": REGRESSION,
            "states": [reg_states[1], reg_states[0], reg_states[2]],
            "status": INVALID_ARGUMENT,
            "message": "state 0 (step 1 (preprocessing.scatter.snv)): the state was fitted by 'models.pls.pls_regression'",
        },
        {
            "name": "state_count_mismatch",
            "stage": "import",
            "steps": REGRESSION,
            "states": reg_states[:2],
            "status": INVALID_ARGUMENT,
            "message": "the recipe has 3 stateful steps but 2 states were given",
        },
        {
            "name": "feature_name_permutation",
            "stage": "predict",
            "feature_names": swapped,
            "status": INVALID_ARGUMENT,
            "message": "input column 2 is 'nm1010', fitted at column 5: the columns are reordered",
        },
        {
            "name": "width_mismatch",
            "stage": "predict",
            "drop_last_column": True,
            "status": SHAPE_MISMATCH,
            "message": "the input has 11 columns; the pipeline was fitted on 12",
        },
        {
            "name": "training_rows_without_opt_in",
            "stage": "export",
            "steps": [step("preprocessing.scatter.snv"), step("models.pls.kernel")],
            "y": "y_train",
            "status": INVALID_ARGUMENT,
            "message": "state 1 (step 1 (models.pls.kernel)): state retains training rows",
        },
        {
            "name": "multi_target_supervised_transformer",
            "stage": "fit",
            "steps": [step("models.pls.pls_regression", n_components=2)] * 2,
            "y": "y2_train",
            "predict": multi.predict(data["x_test"]).tolist(),
        },
    ]
    doc = {
        "abi": ".".join(map(str, n4m.abi_version())),
        **{
            k: (v.tolist() if isinstance(v, np.ndarray) else v) for k, v in data.items()
        },
        "regression": regression,
        "classification": classification,
        "cases": cases,
    }
    OUTPUT.write_text(json.dumps(doc, indent=1) + "\n", encoding="utf-8")
    R_OUTPUT.write_text(render_r(doc), encoding="utf-8")
    print(
        f"wrote {OUTPUT.relative_to(REPO)} and {R_OUTPUT.relative_to(REPO)} ({len(cases)} cases)"
    )


# -- R rendering (the R package tests read no JSON) ------------------------------


def r_value(value) -> str:
    if isinstance(value, bool):
        return "TRUE" if value else "FALSE"
    if isinstance(value, str):
        return json.dumps(value)
    if isinstance(value, int):
        return f"{value}L"
    if isinstance(value, float):
        return repr(value)
    if isinstance(value, dict):
        return (
            "list(" + ", ".join(f"{k} = {r_value(v)}" for k, v in value.items()) + ")"
        )
    if not value:
        return "list()"
    if isinstance(value[0], list):
        arr = np.asarray(value, dtype=float)
        cells = ", ".join(repr(float(v)) for v in arr.ravel())
        return f"matrix(c({cells}), nrow = {arr.shape[0]}, byrow = TRUE)"
    if isinstance(value[0], dict):
        return "list(" + ", ".join(r_value(v) for v in value) + ")"
    if isinstance(value[0], (str, bool)):
        return "c(" + ", ".join(r_value(v) for v in value) + ")"
    return "c(" + ", ".join(repr(float(v)) for v in value) + ")"


def r_steps(steps: list) -> str:
    return (
        "list("
        + ", ".join(
            f"list(method_id = {json.dumps(s['class'][4:])}, params = {r_value(s['params'])})"
            for s in steps
        )
        + ")"
    )


def r_states(states: list) -> str:
    """N4ME payloads as hex strings (base64 needs no R dependency this way)."""
    return "c(" + ", ".join(f'"{base64.b64decode(s).hex()}"' for s in states) + ")"


def r_positive(case: dict) -> str:
    fields = [
        f"steps = {r_steps(case['steps'])}",
        f"states = {r_states([s['n4me_base64'] for s in case['states']])}",
        f"state_methods = {r_value([s['method_id'] for s in case['states']])}",
        f"state_training_rows = {r_value([s['contains_training_rows'] for s in case['states']])}",
        f"predict = {r_value(case['predict'])}",
    ]
    for name in ("transform", "decision_function", "class_names"):
        if name in case:
            fields.append(f"{name} = {r_value(case[name])}")
    return "list(\n    " + ",\n    ".join(fields) + ")"


def render_r(doc: dict) -> str:
    cases = []
    for case in doc["cases"]:
        fields = [
            f"name = {json.dumps(case['name'])}",
            f"stage = {json.dumps(case['stage'])}",
        ]
        if "steps" in case:
            fields.append(f"steps = {r_steps(case['steps'])}")
        if "states" in case:
            fields.append(f"states = {r_states(case['states'])}")
        for key in (
            "feature_names",
            "drop_last_column",
            "y",
            "status",
            "message",
            "predict",
        ):
            if key in case:
                fields.append(f"{key} = {r_value(case[key])}")
        cases.append("   list(" + ", ".join(fields) + ")")
    return "\n".join(
        [
            "# Generated by scripts/generate_role_pipeline_fixture.py from the Python binding. Do not edit.",
            f"# N4ME states written by n4m ABI {doc['abi']}.",
            "role_pipeline_fixture <- list(",
            f"  feature_names = {r_value(doc['feature_names'])},",
            f"  x_train = {r_value(doc['x_train'])},",
            f"  y_train = {r_value(doc['y_train'])},",
            f"  y2_train = {r_value(doc['y2_train'])},",
            f"  labels_train = {r_value(doc['labels_train'])},",
            f"  x_test = {r_value(doc['x_test'])},",
            f"  regression = {r_positive(doc['regression'])},",
            f"  classification = {r_positive(doc['classification'])},",
            "  cases = list(",
            ",\n".join(cases),
            "  )",
            ")",
            "",
        ]
    )


if __name__ == "__main__":
    main()
