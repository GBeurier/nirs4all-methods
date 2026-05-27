"""Unit tests for the golden-snapshot generator's pure logic.

Exercises snapshot construction, the strict golden comparison, and the
committed pilot snapshots' shape — no libn4m needed. The live regenerate/check
is exercised by cross-binding-parity.yml.
"""
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[3]
SNAP_DIR = REPO / "parity" / "snapshots" / "current"


def _load():
    # Unique module name: parity/comparator/run.py and parity/generators/run.py
    # share a basename, so a plain `import run` would collide in the session.
    path = REPO / "parity" / "generators" / "run.py"
    spec = importlib.util.spec_from_file_location("n4m_snapshot_gen_run", path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod  # so @dataclass can resolve cls.__module__
    spec.loader.exec_module(mod)
    return mod


gen = _load()


def test_scenario_id_and_path():
    sid = gen.scenario_id("pls", 60, 30, 7)
    assert sid == "pls_60x30_s7"
    assert gen.snapshot_path("pls", sid).name == "pls_60x30_s7.json"


def test_build_snapshot_has_required_provenance():
    snap = gen.build_snapshot("pls", 6, 3, 2, 1, np.arange(6.0))
    for key in ("schema", "source_kind", "method", "scenario", "seed", "n", "p",
                "n_components", "abi", "provenance", "output_key",
                "output_len", "output"):
        assert key in snap, f"missing {key}"
    assert snap["source_kind"] == "n4m_self"
    assert snap["output_len"] == 6
    assert snap["output"] == list(np.arange(6.0))
    # Binary identity must NOT be committed (it churns every rebuild).
    assert "lib_sha256" not in snap and "lib_bytes" not in snap


def test_compare_output_matches():
    snap = {"output": [1.0, 2.0, 3.0]}
    ok, _ = gen.compare_output(np.array([1.0, 2.0, 3.0]), snap)
    assert ok


def test_compare_output_detects_drift():
    snap = {"output": [1.0, 2.0, 3.0]}
    ok, note = gen.compare_output(np.array([1.0, 2.0, 3.0 + 1e-3]), snap)
    assert not ok and "rmse_rel" in note


def test_compare_output_detects_shape_mismatch():
    snap = {"output": [1.0, 2.0, 3.0]}
    ok, note = gen.compare_output(np.array([1.0, 2.0]), snap)
    assert not ok and "shape drift" in note


def test_gate_exit_code():
    V = gen.SnapVerdict
    assert gen.gate_exit_code([V("m", "s", gen.PASS, "")]) == 0
    assert gen.gate_exit_code([V("m", "s", gen.SKIP, "")]) == 0
    assert gen.gate_exit_code([V("m", "s", gen.FAIL, "")]) == 1


def test_process_run_failure_with_resolved_build_is_fail(monkeypatch):
    # Codex blocking fix: a resolved-build run failure must FAIL, not SKIP.
    import types
    fake = types.ModuleType("_n4m_runner")
    fake.run_n4m = lambda *a, **k: (False, None, "boom")
    monkeypatch.setitem(sys.modules, "_n4m_runner", fake)
    v = gen.process("pls", 6, 3, 2, 1, write=False, lib_dir="/resolved")
    assert v.status == gen.FAIL
    assert gen.gate_exit_code([v]) == 1


def test_committed_pilot_snapshots_are_wellformed():
    files = sorted(SNAP_DIR.glob("*/*.json"))
    assert files, "pilot snapshots missing — run `make snapshot`"
    for f in files:
        snap = json.loads(f.read_text())
        assert snap["schema"] == "snapshot-v1"
        assert snap["source_kind"] == "n4m_self"
        assert snap["output_len"] == len(snap["output"])
        assert {"seed", "n", "p", "abi", "provenance"} <= set(snap)
