#!/usr/bin/env python3
"""Write the cross-language N4ME fixture of the generic estimator roles.

Every estimator of the native manifest is fitted once on a fixed synthetic
dataset; the fixture stores its N4ME bytes, the held-out rows and the native
outputs of its roles (predictions, transformed rows, selected columns, class
labels, decision scores and probabilities). The R
and JS/WASM suites import the same bytes and must reproduce these outputs
(Level 2, trained-state portability), then refit and match the Python fit.

    N4M_LIB_PATH=... PYTHONPATH=bindings/python/src python scripts/generate_estimator_fixtures.py
"""

from __future__ import annotations

import base64
import json
from pathlib import Path

import n4m
import numpy as np
from n4m import roles
from n4m.roles._base import _REGISTRY

REPO = Path(__file__).resolve().parents[1]
OUTPUT = REPO / "parity" / "fixtures" / "estimator_roles_n4me.json"
# Same content as R source, so the R package tests need no JSON dependency.
R_OUTPUT = (
    REPO / "bindings" / "r" / "n4m" / "tests" / "testthat" / "fixture-estimator-roles.R"
)
N_FEATURES = 12

# Values for parameters the test data needs (required ones, or defaults too
# large for 12 columns).
EXPLICIT_PARAMS = {
    "mode_j": 3,
    "mode_k": 4,
    "top_k": 4,
    "thresholds": [0.05, 0.1, 0.3],
    "alpha_thresholds": [0.95, 0.99],
    "start": 2,
    "end": 10,
    "num_samples": 8,
    "edges": [0.25, 0.5, 0.75],
    "kernel_size": 5,
    "alphas": [0.0, 1.0],
    "sigmas": [1.0, 2.0],
    "n_neighbors": 10,
    "n_components_per_block": [1, 1, 1],
    "n_unique_per_block": [1, 1, 1],
    # AOM chains and operator banks: identity, then a first-order detrend.
    "chain_offsets": [0, 1, 2],
    "op_kinds": [0, 7],
    "param_offsets": [0, 0, 1],
    "chain_params": [1.0],
    "op_params": [1.0],
    # Linear stack compression of the 12 X columns (base outputs), one target.
    "base_intercepts": [0.1 * j - 0.5 for j in range(N_FEATURES)],
    "meta_weights": [0.05 * j + 0.1 for j in range(N_FEATURES)],
    "meta_intercept": [0.5],
}
# Fit input name -> n4m_fit_input_t index (n4m/estimator.h).
DATA_INPUTS = {"feature_groups": 4, "blocks": 5, "axis": 6, "X_target": 7}
AXIS = (1000.0 + 2.0 * np.arange(N_FEATURES)).tolist()
CAP_SERIALIZABLE = 1 << 7
# Procedure inputs -> fixture data; procedures reading another X.
INPUT_DATA = {
    "y": "y_train",
    "groups": "groups",
    "axis": "axis",
    "X_target": "x_target",
}
PROCEDURE_X = {"diagnostics.regression_metrics": "x_predictions"}


def dataset():
    rng = np.random.default_rng(20260927)
    scores = rng.normal(size=(48, 2)) * np.array([3.0, 2.0])
    loadings = rng.normal(size=(2, N_FEATURES))
    X = scores @ loadings + 0.1 * rng.normal(size=(48, N_FEATURES))
    y = scores[:, 0] - 0.5 * scores[:, 1] + 0.05 * rng.normal(size=48)
    # Paired transfer methods need one target row per training row. The noise
    # keeps it full rank: subspace metrics on a rank-2 target would compare
    # null-space directions that rounding alone decides.
    X_target = rng.normal(size=(36, 2)) @ loadings + 0.3 + 0.05 * rng.normal(size=(36, N_FEATURES))
    # Strictly positive, reflectance-like values keep every conversion defined.
    shift = 1.0 - min(X.min(), X_target.min())
    X, X_target = (X + shift) / (2 * shift), (X_target + shift) / (2 * shift)
    return X[:36], y[:36], X[36:], X_target, y[36:]


def class_labels(y) -> np.ndarray:
    """Three classes cut from the response; non-contiguous ids exercise the remap."""
    return 10 * (1 + np.digitize(y, np.quantile(y, [1 / 3, 2 / 3])))


def survival(y) -> np.ndarray:
    """PLS-Cox response: time falling with y, then the event flag (every fourth row censored)."""
    return np.column_stack(
        [np.exp(-0.3 * y), (np.arange(y.size) % 4 != 0).astype(float)]
    )


# Estimators fitted on another response than y (the case's "y" names it).
TARGETS = {"models.heads.pls_cox": "survival_train"}


def explicit_params(cls) -> dict:
    params = {k: v for k, v in EXPLICIT_PARAMS.items() if k in cls._param_types}
    if cls is roles.RandomFrog:
        params["initial_size"] = 6
    if cls is roles.RecursivePLS:
        params["window_size"] = 20  # IRF also has a window_size
    if cls is roles.EMCUVE:
        params["noise_features"] = 12  # 50 noise columns swamp 12 real ones
    if cls._method_id.startswith("aom_pop."):
        params.pop("alphas", None)  # the AOM Ridge grids keep their defaults
    return params


def fit_inputs(cls, X_target):
    """The data inputs the manifest declares required for ``cls``."""
    values = {
        "feature_groups": np.arange(N_FEATURES) // 4,
        "blocks": [4, 4, 4],
        "axis": AXIS,
        "X_target": X_target,
    }
    info = roles.method_info(cls._method_id)
    return {n: values[n] for n, i in DATA_INPUTS.items() if info.inputs[i] == 2}


def procedure_case(cls, data: dict) -> dict:
    """Default-parameter run of a procedure on the fixture data."""
    method_id = cls._method_id
    need = cls.input_requirements()
    names = [n for n in ("y", "groups", "axis", "X_target") if need[n] == "required"]
    x_name = PROCEDURE_X.get(method_id, "x_train")
    X = np.asarray(data[x_name])
    kw = {n: np.asarray(data[INPUT_DATA[n]]) for n in names}
    params = explicit_params(cls)
    proc = cls(**params)
    case = {"method_id": method_id, "x": x_name, "inputs": names, "params": params}
    if issubclass(cls, roles.NativeSplitter):
        folds = proc.split(X, kw.get("y"), kw.get("groups"))
        case["folds"] = [[tr.tolist(), te.tolist()] for tr, te in folds]
    elif issubclass(cls, roles.NativeAugmenter):
        out = proc.augment(X, kw.get("y"), axis=kw.get("axis"))
        if isinstance(out, tuple):  # target-mixing: y mixed with the draw of X
            out, case["Y"] = out[0], out[1].tolist()
        case["X"] = out.tolist()
    else:
        y = kw.pop("y", None)
        out = proc.run(X, y, **kw)
        case["outputs"] = {
            k: (v.tolist() if isinstance(v, np.ndarray) else v) for k, v in out.items()
        }
    return case


def main() -> None:
    X, y, X_test, X_target, y_test = dataset()
    labels = class_labels(y)
    data = {
        "x_train": X,
        "y_train": y,
        "groups": np.arange(X.shape[0]) % 18,
        "axis": AXIS,
        "x_target": X_target,
        # Regression metrics read a prediction column against y.
        "x_predictions": (y + 0.1 * np.sin(np.arange(y.size))).reshape(-1, 1),
    }
    procedures = [
        procedure_case(cls, data)
        for _, cls in sorted(_REGISTRY.items())
        if not issubclass(cls, roles.NativeEstimator)
    ]
    cases = []
    for method_id in sorted(_REGISTRY):
        cls = _REGISTRY[method_id]
        if not issubclass(cls, roles.NativeEstimator):
            continue  # procedures have no fitted state
        params = explicit_params(cls)
        target = labels if issubclass(cls, roles.NativeClassifier) else y
        if method_id in TARGETS:
            target = survival(y)
        est = cls(**params).fit(X, target, **fit_inputs(cls, X_target))
        serializable = roles.method_info(method_id).capabilities & CAP_SERIALIZABLE
        case = {
            "method_id": method_id,
            "fit_inputs": sorted(fit_inputs(cls, X_target)),
            "params": params,
            **({"y": TARGETS[method_id]} if method_id in TARGETS else {}),
            # Filters are train-only; the X-outlier state is not serializable.
            "n4me_base64": (
                base64.b64encode(est.to_n4me(allow_training_rows=True)).decode()
                if serializable
                else None
            ),
        }
        if isinstance(est, roles.NativeRegressor):
            case["predict"] = est.predict(X_test).tolist()
        if isinstance(est, (roles.NativeTransformer, roles.NativeSelector)):
            case["transform"] = est.transform(X_test).tolist()
        if isinstance(est, roles.NativeSelector):
            case["selected_indices"] = est.selected_indices_.tolist()
        if isinstance(est, roles.NativeSampleFilter):
            case["mask"] = est.get_mask(X_test, y_test).astype(int).tolist()
        if isinstance(est, roles.NativeClassifier):
            case["classes"] = est.classes_.tolist()
            case["predict_labels"] = est.predict(X_test).tolist()
            case["decision_function"] = est.decision_function(X_test).tolist()
            if hasattr(est, "predict_proba"):
                case["predict_proba"] = est.predict_proba(X_test).tolist()
        cases.append(case)
    doc = {
        "abi": ".".join(map(str, n4m.abi_version())),
        "x_train": X.tolist(),
        "y_train": y.tolist(),
        "labels_train": labels.tolist(),
        "survival_train": survival(y).tolist(),
        "x_target": X_target.tolist(),
        "feature_groups": (np.arange(N_FEATURES) // 4).tolist(),
        "blocks": [4, 4, 4],
        "axis": AXIS,
        "x_test": X_test.tolist(),
        "y_test": y_test.tolist(),
        "groups": data["groups"].tolist(),
        "x_predictions": data["x_predictions"].tolist(),
        "cases": cases,
        "procedures": procedures,
    }
    OUTPUT.write_text(json.dumps(doc, indent=1) + "\n", encoding="utf-8")
    R_OUTPUT.write_text(render_r(doc), encoding="utf-8")
    print(
        f"wrote {OUTPUT.relative_to(REPO)} and {R_OUTPUT.relative_to(REPO)} "
        f"({len(cases)} estimators, {len(procedures)} procedures)"
    )


def r_value(value) -> str:
    if isinstance(value, list):
        return r_vector(value)
    return f"{value}L" if isinstance(value, int) else repr(float(value))


def r_vector(values) -> str:
    return "c(" + ", ".join(repr(float(v)) for v in values) + ")"


def r_matrix(rows) -> str:
    arr = np.asarray(rows, dtype=float)
    return f"matrix({r_vector(arr.ravel())}, nrow = {arr.shape[0]}, byrow = TRUE)"


def render_r(doc: dict) -> str:
    out = [
        "# Generated by scripts/generate_estimator_fixtures.py from the Python binding. Do not edit.",
        f"# N4ME states written by n4m ABI {doc['abi']}.",
        "estimator_roles_fixture <- list(",
        f"  x_train = {r_matrix(doc['x_train'])},",
        f"  y_train = {r_vector(doc['y_train'])},",
        f"  labels_train = {r_vector(doc['labels_train'])},",
        f"  survival_train = {r_matrix(doc['survival_train'])},",
        f"  x_target = {r_matrix(doc['x_target'])},",
        f"  feature_groups = {r_vector(doc['feature_groups'])},",
        f"  blocks = {r_vector(doc['blocks'])},",
        f"  axis = {r_vector(doc['axis'])},",
        f"  x_test = {r_matrix(doc['x_test'])},",
        f"  groups = {r_vector(doc['groups'])},",
        f"  x_predictions = {r_matrix(doc['x_predictions'])},",
        f"  y_test = {r_vector(doc['y_test'])},",
        "  cases = list(",
    ]
    cases = []
    for case in doc["cases"]:
        hexa = (
            "NULL"
            if case["n4me_base64"] is None
            else f'"{base64.b64decode(case["n4me_base64"]).hex()}"'
        )
        inputs = ", ".join(f'"{n}"' for n in case["fit_inputs"])
        params = ", ".join(f"{k} = {r_value(v)}" for k, v in case["params"].items())
        fields = [
            f'    method_id = "{case["method_id"]}"',
            f"    n4me = {hexa}",
            f"    fit_inputs = c({inputs})",
            f"    params = list({params})",
        ]
        if "y" in case:
            fields.append(f'    y = "{case["y"]}"')
        if "predict" in case:
            fields.append(f"    predict = {r_vector(case['predict'])}")
        if "selected_indices" in case:
            fields.append(
                f"    selected_indices = {r_vector(case['selected_indices'])}"
            )
        if "transform" in case:
            fields.append(f"    transform = {r_matrix(case['transform'])}")
        for name in ("classes", "predict_labels", "mask"):
            if name in case:
                fields.append(f"    {name} = {r_vector(case[name])}")
        for name in ("decision_function", "predict_proba"):
            if name in case:
                fields.append(f"    {name} = {r_matrix(case[name])}")
        cases.append("   list(\n" + ",\n".join(fields) + ")")
    procedures = []
    for case in doc["procedures"]:
        inputs = ", ".join(f'"{n}"' for n in case["inputs"])
        params = ", ".join(f"{k} = {r_value(v)}" for k, v in case["params"].items())
        fields = [
            f'    method_id = "{case["method_id"]}"',
            f'    x = "{case["x"]}"',
            f"    inputs = c({inputs})",
            f"    params = list({params})",
        ]
        if "folds" in case:
            folds = ", ".join(
                f"list({r_int_vector(tr)}, {r_int_vector(te)})"
                for tr, te in case["folds"]
            )
            fields.append(f"    folds = list({folds})")
        if "X" in case:
            fields.append(f"    X = {r_matrix(case['X'])}")
        if "Y" in case:
            fields.append(f"    Y = {r_vector(case['Y'])}")
        if "outputs" in case:
            outputs = ", ".join(
                f"`{k}` = {r_output(v)}" for k, v in case["outputs"].items()
            )
            fields.append(f"    outputs = list({outputs})")
        procedures.append("   list(\n" + ",\n".join(fields) + ")")
    out += [
        ",\n".join(cases),
        "  ),",
        "  procedures = list(",
        ",\n".join(procedures),
        "  )",
        ")",
        "",
    ]
    return "\n".join(out)


def r_int_vector(values) -> str:
    return (
        "c(" + ", ".join(f"{int(v)}" for v in values) + ")" if values else "numeric(0)"
    )


def r_output(value) -> str:
    """A native output: matrix (nested list), vector or scalar."""
    if isinstance(value, list):
        if value and isinstance(value[0], list):
            return r_matrix(value)
        return r_vector(value) if value else "numeric(0)"
    return repr(float(value))


if __name__ == "__main__":
    main()
