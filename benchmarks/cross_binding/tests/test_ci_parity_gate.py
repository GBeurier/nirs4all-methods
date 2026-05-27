"""Unit tests for the cross-binding parity CI gate logic.

These exercise the pure classification + exit-code logic with synthetic
backend records, so they run anywhere (no R/Octave/libn4m needed). The
live cross-language run is exercised by the `cross-binding-parity.yml`
workflow, which installs the toolchains and passes `--require`.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from benchmarks.cross_binding import ci_parity_gate as gate


def _rec_ok(tmp_path: Path, backend: str, preds: np.ndarray) -> dict:
    p = tmp_path / f"{backend}.npy"
    np.save(p, np.asarray(preds, dtype=np.float64), allow_pickle=False)
    return {"backend": backend, "ok": True, "predictions_path": str(p)}


def test_matching_binding_passes(tmp_path):
    cpp = np.array([1.0, 2.0, 3.0])
    rec = _rec_ok(tmp_path, "r_tier1", cpp.copy())
    status, rmse_rel, _ = gate.evaluate_binding(rec, cpp, tol=1e-12)
    assert status == gate.PASS
    assert rmse_rel == 0.0


def test_diverging_binding_fails(tmp_path):
    cpp = np.array([1.0, 2.0, 3.0])
    rec = _rec_ok(tmp_path, "r_tier1", cpp + 1e-3)
    status, rmse_rel, _ = gate.evaluate_binding(rec, cpp, tol=1e-12)
    assert status == gate.FAIL
    assert rmse_rel > 1e-12


def test_absent_driver_is_skip_not_fail(tmp_path):
    rec = {"backend": "matlab_tier1", "ok": False,
           "reason": "command not found: octave"}
    status, _, _ = gate.evaluate_binding(rec, np.array([1.0]), tol=1e-12)
    assert status == gate.SKIP

    rec2 = {"backend": "r_tier1", "ok": False,
            "reason": "script not found: bench_r_tier1.R"}
    assert gate.evaluate_binding(rec2, np.array([1.0]), tol=1e-12)[0] == gate.SKIP


def test_runtime_error_is_fail_not_skip():
    rec = {"backend": "r_tier1", "ok": False, "reason": "non-json output"}
    status, _, _ = gate.evaluate_binding(rec, np.array([1.0]), tol=1e-12)
    assert status == gate.FAIL


def test_ran_but_no_predictions_is_fail(tmp_path):
    rec = {"backend": "r_tier1", "ok": True,
           "predictions_path": str(tmp_path / "missing.npy")}
    status, _, _ = gate.evaluate_binding(rec, np.array([1.0]), tol=1e-12)
    assert status == gate.FAIL


def test_exit_code_clean_pass():
    v = [gate.Verdict("python_tier1", "Python", "raw", "pls 60x30",
                      gate.PASS, 0.0, "")]
    assert gate.gate_exit_code(v, required_langs=["python"]) == 0


def test_exit_code_fails_on_any_fail():
    v = [gate.Verdict("r_tier1", "R", "raw", "pls 60x30", gate.FAIL, 1e-2, "")]
    assert gate.gate_exit_code(v, required_langs=[]) == 1


def test_exit_code_fails_when_required_language_skipped():
    # R skipped (toolchain absent) but R is required -> hard fail.
    v = [
        gate.Verdict("python_tier1", "Python", "raw", "pls 60x30",
                     gate.PASS, 0.0, ""),
        gate.Verdict("r_tier1", "R", "raw", "pls 60x30", gate.SKIP, None,
                     "command not found: Rscript"),
    ]
    assert gate.gate_exit_code(v, required_langs=["python", "r"]) == 1
    # ...but tolerated when R is not required (local dev without R).
    assert gate.gate_exit_code(v, required_langs=["python"]) == 0


def test_exit_code_fails_when_required_language_partially_skips():
    # A required language must have EVERY cell PASS: one PASS + one SKIP fails.
    v = [
        gate.Verdict("r_tier1", "R", "raw", "pls 60x30", gate.PASS, 0.0, ""),
        gate.Verdict("r_tier1", "R", "raw", "pcr 60x30", gate.SKIP, None,
                     "command not found: Rscript"),
    ]
    assert gate.gate_exit_code(v, required_langs=["r"]) == 1


def test_exit_code_requires_language_to_be_present():
    # Requiring a language that never produced a verdict fails the gate.
    v = [gate.Verdict("python_tier1", "Python", "raw", "pls 60x30",
                      gate.PASS, 0.0, "")]
    assert gate.gate_exit_code(v, required_langs=["octave"]) == 1
