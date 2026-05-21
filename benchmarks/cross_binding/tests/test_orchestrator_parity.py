from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from benchmarks.cross_binding import orchestrator
from benchmarks.parity_timing._comparator import binding_parity, reference_parity


def _record(tmp_path: Path, backend: str, kind: str,
             predictions: np.ndarray, *, seed: int = 123,
             algorithm: str = "pcr") -> dict:
    pred_path = tmp_path / f"{backend}_{seed}.npy"
    np.save(pred_path, np.asarray(predictions, dtype=np.float64),
            allow_pickle=False)
    return {
        "algorithm": algorithm,
        "backend": backend,
        "language": "Python",
        "tier": "test",
        "kind": kind,
        "n": 200,
        "p": 50,
        "threads": 1,
        "libp4a_build": "blas-omp",
        "seed_base": seed,
        "prediction_seed": seed,
        "ok": True,
        "predictions_path": str(pred_path),
    }


def _write_oracle_metadata(path: Path, *, seed: int = 123,
                           algorithm: str = "pcr",
                           backend: str = "ref_python_scikit_learn") -> None:
    orchestrator.prediction_metadata_path(path).write_text(
        json.dumps({
            "algorithm": algorithm,
            "backend": backend,
            "n": 200,
            "p": 50,
            "prediction_seed": seed,
        }) + "\n"
    )


def test_external_rows_are_reference_checked_not_binding_checked(
        tmp_path, monkeypatch):
    monkeypatch.setattr(orchestrator, "ORACLES_DIR", tmp_path / "oracles")
    orchestrator.ORACLES_DIR.mkdir()

    ref = np.array([1.0, 2.0, 3.0])
    records = [
        _record(tmp_path, "cpp", "pls4all_core", ref),
        _record(tmp_path, "ref_python_scikit_learn", "external", ref),
        _record(tmp_path, "ref_r_pls", "external", ref),
    ]

    orchestrator.compute_parity(records)

    external = records[2]
    assert external["binding_parity_ok"] is None
    assert external["binding_parity_note"] == "not_applicable"
    assert external["reference_parity_ok"] is True
    assert external["parity_ref"] == ""

    oracle = orchestrator.reference_oracle_path(
        "pcr", "ref_python_scikit_learn", 200, 50, 123)
    assert oracle.exists()


def test_pls4all_only_rows_use_stored_reference_oracle(tmp_path, monkeypatch):
    monkeypatch.setattr(orchestrator, "ORACLES_DIR", tmp_path / "oracles")
    orchestrator.ORACLES_DIR.mkdir()

    ref = np.array([1.0, 2.0, 3.0])
    oracle = orchestrator.reference_oracle_path(
        "pcr", "ref_python_scikit_learn", 200, 50, 123)
    np.save(oracle, ref, allow_pickle=False)
    _write_oracle_metadata(oracle)

    records = [
        _record(tmp_path, "cpp", "pls4all_core", ref),
        _record(tmp_path, "python_tier1", "pls4all_binding", ref),
    ]

    orchestrator.compute_parity(records)

    for row in records:
        assert row["binding_parity_ok"] is True
        assert row["reference_parity_ok"] is True
        assert row["reference_oracle_path"] == str(oracle)


def test_reference_gate_clamps_loose_method_tolerance(
        tmp_path, monkeypatch):
    monkeypatch.setattr(orchestrator, "ORACLES_DIR", tmp_path / "oracles")
    orchestrator.ORACLES_DIR.mkdir()

    oracle = orchestrator.reference_oracle_path(
        "bagging_pls", "ref_python_scikit_learn", 200, 50, 123)
    np.save(oracle, np.array([1.0, 1.0]), allow_pickle=False)
    _write_oracle_metadata(
        oracle, algorithm="bagging_pls",
        backend="ref_python_scikit_learn")

    records = [
        _record(tmp_path, "cpp", "pls4all_core",
                np.array([2.0, 1.0]), algorithm="bagging_pls"),
    ]

    orchestrator.compute_parity(records)

    assert records[0]["reference_parity_ok"] is False
    assert records[0]["reference_parity_tolerance"] == 1e-3


def test_stale_stored_reference_oracle_metadata_is_a_gate_failure(
        tmp_path, monkeypatch):
    monkeypatch.setattr(orchestrator, "ORACLES_DIR", tmp_path / "oracles")
    orchestrator.ORACLES_DIR.mkdir()

    oracle = orchestrator.reference_oracle_path(
        "pcr", "ref_python_scikit_learn", 200, 50, 123)
    np.save(oracle, np.array([1.0, 2.0, 3.0]), allow_pickle=False)
    _write_oracle_metadata(oracle, seed=999)

    records = [
        _record(tmp_path, "cpp", "pls4all_core",
                np.array([1.0, 2.0, 3.0])),
    ]

    orchestrator.compute_parity(records)

    assert records[0]["binding_parity_ok"] is True
    assert records[0]["reference_parity_ok"] is False
    assert "metadata mismatch" in records[0]["reference_parity_note"]


def test_stabilize_prediction_path_includes_prediction_seed(
        tmp_path, monkeypatch):
    monkeypatch.setattr(orchestrator, "PREDS_DIR", tmp_path / "preds")
    orchestrator.PREDS_DIR.mkdir()

    record = _record(
        tmp_path,
        "cpp",
        "pls4all_core",
        np.array([1.0, 2.0, 3.0]),
        seed=123,
    )

    orchestrator.stabilize_prediction_path(record)

    path = Path(record["predictions_path"])
    assert path.name == "pcr_cpp_200x50_t1_blas-omp_seed123.npy"
    assert path.exists()
    assert np.load(path).tolist() == [1.0, 2.0, 3.0]


def test_missing_stored_reference_oracle_is_a_gate_failure(
        tmp_path, monkeypatch):
    monkeypatch.setattr(orchestrator, "ORACLES_DIR", tmp_path / "oracles")
    orchestrator.ORACLES_DIR.mkdir()

    records = [
        _record(tmp_path, "cpp", "pls4all_core",
                np.array([1.0, 2.0, 3.0])),
    ]

    orchestrator.compute_parity(records)

    assert records[0]["binding_parity_ok"] is True
    assert records[0]["reference_parity_ok"] is False
    assert records[0]["reference_parity_note"].startswith(
        "reference oracle missing")


def test_binding_gate_does_not_fall_back_to_external_reference(
        tmp_path, monkeypatch):
    monkeypatch.setattr(orchestrator, "ORACLES_DIR", tmp_path / "oracles")
    orchestrator.ORACLES_DIR.mkdir()

    ref = np.array([1.0, 2.0, 3.0])
    records = [
        _record(tmp_path, "matlab_tier1", "pls4all_binding", ref),
        _record(tmp_path, "ref_python_scikit_learn", "external", ref),
    ]

    orchestrator.compute_parity(records)

    binding = records[0]
    assert binding["binding_parity_ref"] == ""
    assert binding["binding_parity_ok"] is False
    assert binding["binding_parity_note"] == "binding reference missing"
    assert binding["reference_parity_ok"] is True


def test_comparator_allows_vector_shape_quirk_only() -> None:
    vector = np.array([1.0, 2.0, 3.0])
    column = np.array([[1.0], [2.0], [3.0]])
    flat_matrix = np.array([1.0, 2.0, 3.0, 4.0])
    matrix = np.array([[1.0, 2.0], [3.0, 4.0]])
    row = np.array([[1.0, 2.0, 3.0, 4.0]])

    assert binding_parity(vector, column).ok is True
    assert reference_parity(vector, column, tolerance=1e-6).ok is True
    assert binding_parity(flat_matrix, matrix).ok is True
    assert reference_parity(matrix, flat_matrix, tolerance=1e-6).ok is True

    binding = binding_parity(matrix, row)
    reference = reference_parity(matrix, row, tolerance=1e-6)

    assert binding.ok is False
    assert reference.ok is False
    assert "shape mismatch" in binding.note
    assert "shape mismatch" in reference.note
