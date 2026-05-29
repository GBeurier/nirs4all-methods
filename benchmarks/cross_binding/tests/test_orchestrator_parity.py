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
        _record(tmp_path, "cpp", "n4m_core", ref),
        _record(tmp_path, "ref_python_scikit_learn", "external", ref),
        _record(tmp_path, "ref_r_pls", "external", ref),
    ]

    orchestrator.compute_parity(records)

    # The canonical donor reference is compared to itself -> "self", ok=True.
    canonical = records[1]
    assert canonical["is_canonical_reference"] is True
    assert canonical["reference_parity_ok"] is True
    assert canonical["reference_parity_note"] == "self"

    # A SECONDARY external library (R pls vs the sklearn oracle) is an
    # informational cross-check: it records a divergence but never fails as if
    # it were an n4m bug.
    external = records[2]
    assert external["binding_parity_ok"] is None
    assert external["binding_parity_note"] == "not_applicable"
    assert external["reference_parity_ok"] is None
    assert external["reference_parity_note"] == "cross_check"
    assert external["parity_ref"] == ""

    oracle = orchestrator.reference_oracle_path(
        "pcr", "ref_python_scikit_learn", 200, 50, 123)
    assert oracle.exists()


def test_selection_jaccard_metric():
    # Selectors are gated on set overlap, not relative RMSE.
    assert orchestrator._selection_jaccard([1, 0, 1, 0], [1, 0, 1, 0]) == 1.0
    assert orchestrator._selection_jaccard([1, 0, 0, 0], [0, 1, 0, 0]) == 0.0
    assert orchestrator._selection_jaccard([1, 1, 0, 0], [1, 0, 0, 0]) == 0.5
    assert orchestrator._selection_jaccard([0, 0, 0], [0, 0, 0]) == 1.0  # both empty
    # 2-D (1,p) masks (the shape selectors actually emit) ravel the same way.
    assert orchestrator._selection_jaccard([[1, 0, 1]], [[1, 0, 1]]) == 1.0


def test_selection_divergence_allowlist_is_justified():
    # Every allowlisted selector divergence must carry a written reason, so the
    # allowlist cannot grow silently with bare entries.
    for algo, reason in orchestrator.SELECTION_DIVERGENCE_ALLOWLIST.items():
        assert isinstance(reason, str) and len(reason) > 20, algo
    assert "t2_select" in orchestrator.SELECTION_DIVERGENCE_ALLOWLIST


def test_pls4all_only_rows_use_stored_reference_oracle(tmp_path, monkeypatch):
    monkeypatch.setattr(orchestrator, "ORACLES_DIR", tmp_path / "oracles")
    orchestrator.ORACLES_DIR.mkdir()

    ref = np.array([1.0, 2.0, 3.0])
    oracle = orchestrator.reference_oracle_path(
        "pcr", "ref_python_scikit_learn", 200, 50, 123)
    np.save(oracle, ref, allow_pickle=False)

    records = [
        _record(tmp_path, "cpp", "n4m_core", ref),
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
        _record(tmp_path, "cpp", "n4m_core",
                np.array([1.0, 2.0, 3.0])),
    ]

    orchestrator.compute_parity(records)

    assert records[0]["binding_parity_ok"] is True
    assert records[0]["reference_parity_ok"] is False
    assert records[0]["reference_parity_note"].startswith(
        "reference oracle missing")
