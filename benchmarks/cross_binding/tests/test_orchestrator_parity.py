from __future__ import annotations

from pathlib import Path

import numpy as np

from benchmarks.cross_binding import orchestrator


def _record(tmp_path: Path, backend: str, kind: str,
             predictions: np.ndarray, *, seed: int = 123) -> dict:
    pred_path = tmp_path / f"{backend}_{seed}.npy"
    np.save(pred_path, np.asarray(predictions, dtype=np.float64),
            allow_pickle=False)
    return {
        "algorithm": "pcr",
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

    records = [
        _record(tmp_path, "cpp", "pls4all_core", ref),
        _record(tmp_path, "python_tier1", "pls4all_binding", ref),
    ]

    orchestrator.compute_parity(records)

    for row in records:
        assert row["binding_parity_ok"] is True
        assert row["reference_parity_ok"] is True
        assert row["reference_oracle_path"] == str(oracle)


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
