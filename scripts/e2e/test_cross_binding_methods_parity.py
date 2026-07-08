from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest


SCRIPT = Path(__file__).with_name("cross_binding_methods_parity.py")
SPEC = importlib.util.spec_from_file_location("cross_binding_methods_parity", SCRIPT)
assert SPEC is not None
assert SPEC.loader is not None
gate = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(gate)


def _row(backend: str, pred_path: Path, *, kind: str = "pls4all_binding") -> dict[str, str]:
    return {
        "algorithm": "pls",
        "backend": backend,
        "language": backend.split("_", 1)[0],
        "kind": kind,
        "n": "20",
        "p": "8",
        "seed_base": "1234567890",
        "prediction_seed": "1234567890",
        "ok": "True",
        "binding_parity_ok": "True" if kind != "external" else "",
        "binding_parity_max_diff": "0.0" if kind != "external" else "",
        "reference_parity_ok": "True",
        "reference_parity_rmse_rel": "4.0e-16",
        "reference_parity_note": "",
        "predictions_path": str(pred_path),
    }


def test_validate_orchestrator_rows_requires_all_release_backends(tmp_path: Path) -> None:
    pred_path = tmp_path / "predictions.npy"
    np.save(pred_path, np.array([[1.0], [2.0], [3.0]], dtype=np.float64))
    rows = [
        _row("cpp", pred_path),
        _row("python_tier1", pred_path),
        _row("r_tier1", pred_path),
        _row("ref_python_scikit_learn", pred_path, kind="external"),
    ]

    parity_rows, prediction_rows = gate._validate_orchestrator_rows(rows)

    assert [row["backend"] for row in parity_rows] == list(gate.REQUIRED_BACKENDS)
    assert all(row["ok"] is True for row in parity_rows)
    assert all(row["reference_parity_ok"] is True for row in parity_rows)
    assert len(prediction_rows) == len(gate.REQUIRED_BACKENDS)
    assert all(row["sha256"] for row in prediction_rows)


def test_validate_orchestrator_rows_fails_when_backend_is_missing(tmp_path: Path) -> None:
    pred_path = tmp_path / "predictions.npy"
    np.save(pred_path, np.array([1.0], dtype=np.float64))
    rows = [
        _row("cpp", pred_path),
        _row("python_tier1", pred_path),
        _row("ref_python_scikit_learn", pred_path, kind="external"),
    ]

    with pytest.raises(RuntimeError, match="missing backends"):
        gate._validate_orchestrator_rows(rows)


def test_validate_orchestrator_rows_fails_on_false_parity(tmp_path: Path) -> None:
    pred_path = tmp_path / "predictions.npy"
    np.save(pred_path, np.array([1.0], dtype=np.float64))
    rows = [
        _row("cpp", pred_path),
        _row("python_tier1", pred_path),
        _row("r_tier1", pred_path),
        _row("ref_python_scikit_learn", pred_path, kind="external"),
    ]
    rows[1]["reference_parity_ok"] = "False"

    with pytest.raises(RuntimeError, match="reference_parity_ok=False"):
        gate._validate_orchestrator_rows(rows)


def test_base_env_isolates_r_gate_library(tmp_path: Path) -> None:
    r_lib = tmp_path / "r-lib"

    env = gate._base_env(r_lib)

    assert env["N4M_R_GATE_LIB"] == str(r_lib)
    assert env["R_LIBS"] == str(r_lib)
    assert env["R_LIBS_USER"] == str(r_lib)
    assert str(gate.REPO / "build/dev-release/cpp/src") in env.get("LD_LIBRARY_PATH", "")


def test_build_wasm_orchestrator_fixture_uses_cross_binding_dataset(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repo = tmp_path / "repo"
    data_dir = repo / "benchmarks/cross_binding/data"
    data_dir.mkdir(parents=True)
    csv_path = data_dir / "data_20x8_seed1234567890.csv"
    X = np.arange(160, dtype=np.float64).reshape(20, 8) / 10.0
    y = (X[:, 0] - 0.25 * X[:, 1]).reshape(20, 1)
    np.savetxt(
        csv_path,
        np.hstack([X, y]),
        delimiter=",",
        header=",".join([f"x{i}" for i in range(8)] + ["y"]),
        comments="",
    )
    pred_path = tmp_path / "cpp.npy"
    predictions = (y + 0.125).astype(np.float64)
    np.save(pred_path, predictions)
    rows = [
        _row("cpp", pred_path),
        _row("python_tier1", pred_path),
        _row("r_tier1", pred_path),
        _row("ref_python_scikit_learn", pred_path, kind="external"),
    ]
    prediction_rows = [
        {"backend": "cpp", "path": str(pred_path), "sha256": "cpp-sha", "shape": [20, 1]},
        {"backend": "python_tier1", "path": str(pred_path), "sha256": "py-sha", "shape": [20, 1]},
        {"backend": "r_tier1", "path": str(pred_path), "sha256": "r-sha", "shape": [20, 1]},
    ]
    monkeypatch.setattr(gate, "REPO", repo)
    monkeypatch.setattr(
        gate,
        "_native_pls_fixture_arrays",
        lambda X_arg, Y_arg, n_components: {
            "version": "test-native",
            "coefficients": np.arange(8, dtype=np.float64).reshape(8, 1),
            "x_mean": X_arg.mean(axis=0),
            "y_mean": Y_arg.mean(axis=0),
            "predictions": predictions,
        },
    )

    fixture_path = gate._build_wasm_orchestrator_fixture(
        rows,
        prediction_rows,
        tmp_path,
        n_components=3,
    )
    fixture = json.loads(fixture_path.read_text(encoding="utf-8"))

    assert fixture["schema"] == "n4a.methods.wasm_orchestrator_fixture.v1"
    assert fixture["status"] == "pass"
    assert fixture["dataset_csv"] == str(csv_path)
    assert fixture["reference_backend"] == "cpp"
    assert fixture["reference_predictions_path"] == str(pred_path)
    assert fixture["cpp_native_predictions_rmse_rel"] == 0.0
    assert fixture["tolerances"]["binding_parity_max_diff"] == gate.STRICT_TOLERANCES["binding_parity_max_diff"]
    assert fixture["prediction_digests"]["cpp"]["sha256"] == "cpp-sha"
    assert fixture["n"] == 20
    assert fixture["p"] == 8
    assert fixture["q"] == 1
    assert len(fixture["X"]) == 160
    assert len(fixture["Y"]) == 20
    assert fixture["predictions"] == [float(value) for value in predictions.ravel()]
