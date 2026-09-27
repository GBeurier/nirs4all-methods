#!/usr/bin/env python3
"""Write the shared negative fixture of the generic estimator roles.

Each case is a call every binding must refuse, or a failure every binding
must survive with the same observable behaviour: targets and per-row inputs
whose length does not match X (never recycled or reshaped), an N4ME payload
whose recorded parameter contradicts its state (checksum recomputed), an
export of training rows without the explicit opt-in, and a failed classifier
refit that must leave the previous model usable, and output views narrower
or wider than the operation's output. The Python, R, JS/WASM and Rust suites
replay the same file (R, which has no raw view API, checks the widths its
facade allocates).

    N4M_LIB_PATH=... PYTHONPATH=bindings/python/src \
        python scripts/generate_estimator_negative_fixture.py
"""

from __future__ import annotations

import base64
import json
import struct
from pathlib import Path

import n4m
import numpy as np
from n4m import roles

REPO = Path(__file__).resolve().parents[1]
OUTPUT = REPO / "parity" / "fixtures" / "estimator_roles_negative.json"
# Same content as R source, so the R package tests need no JSON dependency.
R_OUTPUT = (
    REPO
    / "bindings"
    / "r"
    / "n4m"
    / "tests"
    / "testthat"
    / "fixture-estimator-roles-negative.R"
)
ROWS, COLS = 40, 8


def dataset():
    rng = np.random.default_rng(20260927)
    X = rng.normal(size=(ROWS, COLS))
    y = X[:, 0] - 0.5 * X[:, 1] + 0.05 * rng.normal(size=ROWS)
    X_test = rng.normal(size=(6, COLS))
    return X, y, X_test


def fnv1a64(data: bytes) -> int:
    h = 0xCBF29CE484222325
    for byte in data:
        h = ((h ^ byte) * 0x100000001B3) & ((1 << 64) - 1)
    return h


def mutate_param(payload: bytes, name: str, value: int) -> bytes:
    """Sets an integer N4ME parameter and recomputes the checksum."""
    out = bytearray(payload)
    pos = 20
    pos += 4 + struct.unpack_from("<I", out, pos)[0]
    (count,) = struct.unpack_from("<I", out, pos)
    pos += 4
    for _ in range(count):
        (length,) = struct.unpack_from("<I", out, pos)
        key = bytes(out[pos + 4 : pos + 4 + length]).decode()
        pos += 4 + length + 4
        (n_values,) = struct.unpack_from("<Q", out, pos)
        pos += 8
        if key == name:
            struct.pack_into("<q", out, pos, value)
            struct.pack_into("<Q", out, len(out) - 8, fnv1a64(bytes(out[:-8])))
            return bytes(out)
        pos += 8 * n_values
    raise KeyError(name)


def _b64(raw: bytes) -> str:
    return base64.b64encode(raw).decode()


def main() -> None:
    X, y, X_test = dataset()

    pls = roles.PLSRegression(n_components=2).fit(X, y)
    state = pls.to_n4me()
    kernel = roles.KernelPLS(n_components=2).fit(X, y)
    labels = (X[:, 0] > 0).astype(np.int64)
    lda = roles.PLSLDA(n_components=2).fit(X, labels)

    cases = [
        # Targets: one row per row of X, whatever the length divides.
        {
            "id": "target_length_divides",
            "step": "fit",
            "method_id": "models.regularized.ridge",
            "y": y[:20].tolist(),
            "mentions": "y",
        },
        {
            "id": "target_scalar",
            "step": "fit",
            "method_id": "models.regularized.ridge",
            "y": [7.0],
            "mentions": "y",
        },
        {
            "id": "target_length_short",
            "step": "fit",
            "method_id": "models.regularized.ridge",
            "y": y[:-1].tolist(),
            "mentions": "y",
        },
        {
            # The same 40 values as a 2 x 20 matrix: never reinterpreted.
            "id": "target_transposed",
            "step": "fit",
            "method_id": "models.pls.pls_regression",
            "y_matrix": y.reshape(2, 20).tolist(),
            "mentions": "y",
        },
        {
            "id": "sample_weight_length",
            "step": "fit",
            "method_id": "models.regularized.weighted_pls",
            "y": y.tolist(),
            "sample_weight": np.ones(ROWS - 1).tolist(),
            "mentions": "sample_weight",
        },
        {
            "id": "labels_length",
            "step": "fit",
            "method_id": "models.classification.pls_lda",
            "labels": labels[:-1].tolist(),
            "mentions": "labels",
        },
        {
            "id": "fold_ids_length",
            "step": "fit",
            "method_id": "selection.stability",
            "params": {"top_k": 3},
            "y": y.tolist(),
            "fold_ids": (np.arange(ROWS - 2) % 3).tolist(),
            "mentions": "fold_ids",
        },
        {
            # PLS fitted with two components; the header claims one.
            "id": "n4me_recorded_param_mutated",
            "step": "import",
            "method_id": "models.pls.pls_regression",
            "n4me_base64": _b64(mutate_param(state, "n_components", 1)),
            "control_base64": _b64(state),
            "control_predict": pls.predict(X_test).tolist(),
            "mentions": "n_components",
        },
        {
            # Output views must hold exactly X rows x the operation's width:
            # narrower or wider views are refused before anything is written.
            "id": "output_view_shape",
            "step": "output_view",
            "method_id": "models.pls.pls_regression",
            "n4me_base64": _b64(state),
            "transform_cols": 2,
            "predict_cols": 1,
            "mentions": "output view",
        },
        {
            "id": "training_rows_export_without_opt_in",
            "step": "export",
            "method_id": "models.pls.kernel",
            "params": {"n_components": 2},
            "y": y.tolist(),
            "contains_training_rows": True,
            "export_size_with_opt_in": len(kernel.to_n4me(allow_training_rows=True)),
            "mentions": "training rows",
        },
        {
            # A refit on one class fails; the two-class model keeps predicting.
            "id": "classifier_failed_refit",
            "step": "refit",
            "method_id": "models.classification.pls_lda",
            "params": {"n_components": 2},
            "labels": labels.tolist(),
            "refit_labels": np.zeros(ROWS, dtype=np.int64).tolist(),
            "classes": lda.classes_.tolist(),
            "predict_labels": lda.predict(X_test).tolist(),
            "mentions": "two classes",
        },
    ]
    doc = {
        "abi": ".".join(map(str, n4m.abi_version())),
        "x_train": X.tolist(),
        "x_test": X_test.tolist(),
        "cases": cases,
    }
    OUTPUT.write_text(json.dumps(doc, indent=1) + "\n", encoding="utf-8")
    R_OUTPUT.write_text(render_r(doc), encoding="utf-8")
    print(
        f"wrote {OUTPUT.relative_to(REPO)} and {R_OUTPUT.relative_to(REPO)} "
        f"({len(cases)} cases)"
    )


def r_vector(values) -> str:
    return "c(" + ", ".join(repr(float(v)) for v in values) + ")"


def r_matrix(rows) -> str:
    arr = np.asarray(rows, dtype=float)
    return f"matrix({r_vector(arr.ravel())}, nrow = {arr.shape[0]}, byrow = TRUE)"


def r_value(value) -> str:
    if isinstance(value, bool):
        return "TRUE" if value else "FALSE"
    if isinstance(value, str):
        return json.dumps(value)
    if isinstance(value, dict):
        return (
            "list(" + ", ".join(f"{k} = {r_value(v)}" for k, v in value.items()) + ")"
        )
    if isinstance(value, list):
        return (
            r_matrix(value) if value and isinstance(value[0], list) else r_vector(value)
        )
    return f"{value}L" if isinstance(value, int) else repr(float(value))


def render_r(doc: dict) -> str:
    cases = []
    for case in doc["cases"]:
        fields = []
        for key, value in case.items():
            if key.endswith("_base64"):
                fields.append(f'    {key[:-7]}_hex = "{base64.b64decode(value).hex()}"')
            else:
                fields.append(f"    {key} = {r_value(value)}")
        cases.append("   list(\n" + ",\n".join(fields) + ")")
    return "\n".join(
        [
            "# Generated by scripts/generate_estimator_negative_fixture.py from the Python binding.",
            f"# Do not edit. N4ME states written by n4m ABI {doc['abi']}.",
            "estimator_roles_negative <- list(",
            f"  x_train = {r_matrix(doc['x_train'])},",
            f"  x_test = {r_matrix(doc['x_test'])},",
            "  cases = list(",
            ",\n".join(cases),
            "  )",
            ")",
            "",
        ]
    )


if __name__ == "__main__":
    main()
