from __future__ import annotations

import importlib.util
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
        "backend": backend,
        "language": backend.split("_", 1)[0],
        "kind": kind,
        "algorithm": "pls",
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
