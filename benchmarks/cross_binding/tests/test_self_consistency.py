"""Unit tests for the self-consistency comparator's pure logic.

Exercises the invariant checks, determinism comparison, and gate exit code
with synthetic arrays — no libn4m needed. The live AOM/POP run is exercised by
the cross-binding-parity.yml workflow (which builds libn4m).
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[3]


def _load():
    path = REPO / "parity" / "comparator" / "self_consistency.py"
    spec = importlib.util.spec_from_file_location("n4m_self_consistency", path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod  # so @dataclass can resolve cls.__module__
    spec.loader.exec_module(mod)
    return mod


sc = _load()


def _spec(expected_len, invariants=("finite", "nonempty", "len")):
    return sc.SelfConsistencySpec("m", 8, 4, 2, 1, expected_len, invariants, "note")


def test_invariants_all_pass():
    arr = np.arange(8.0)
    res = sc.check_invariants(arr, _spec(expected_len=8))
    assert all(ok for _, ok, _ in res)


def test_invariant_finite_fails_on_nan():
    arr = np.array([1.0, np.nan, 3.0])
    res = dict((n, ok) for n, ok, _ in sc.check_invariants(arr, _spec(3)))
    assert res["finite"] is False


def test_invariant_len_mismatch_fails():
    arr = np.arange(5.0)
    res = dict((n, ok) for n, ok, _ in sc.check_invariants(arr, _spec(expected_len=8)))
    assert res["len"] is False


def test_invariant_nonempty_fails_on_empty():
    res = dict((n, ok) for n, ok, _ in sc.check_invariants(np.array([]), _spec(0)))
    assert res["nonempty"] is False


def test_determinism_identical_ok():
    a = np.array([1.0, 2.0, 3.0])
    ok, _ = sc.check_determinism(a, a.copy())
    assert ok


def test_determinism_detects_drift():
    ok, note = sc.check_determinism(np.array([1.0, 2.0]), np.array([1.0, 2.0 + 1e-9]))
    assert not ok and "nondeterministic" in note


def test_determinism_detects_shape_drift():
    ok, note = sc.check_determinism(np.array([1.0, 2.0]), np.array([1.0]))
    assert not ok and "shape drift" in note


def test_gate_exit_code():
    V = sc.MethodVerdict
    assert sc.gate_exit_code([V("a", sc.PASS, "ok", [], "")]) == 0
    assert sc.gate_exit_code([V("a", sc.SKIP, "no build", [], "")]) == 0
    assert sc.gate_exit_code([V("a", sc.PASS, "ok", [], ""),
                              V("b", sc.FAIL, "bad", [], "")]) == 1


def test_specs_cover_the_differentiators():
    ids = {s.method for s in sc.SPECS}
    assert {"aom_pls", "pop_pls"} <= ids


def test_run_failure_with_resolved_build_is_fail_not_skip(monkeypatch):
    # Codex blocking fix: once a lib dir is resolved, a crashed run must FAIL
    # (so CI can't pass on a broken AOM/POP), not SKIP.
    import types
    fake = types.ModuleType("_n4m_runner")
    fake.run_n4m = lambda *a, **k: (False, None, "boom")
    monkeypatch.setitem(sys.modules, "_n4m_runner", fake)
    v = sc.evaluate_spec(sc.SPECS[0], lib_dir="/some/resolved/build")
    assert v.status == sc.FAIL
    assert sc.gate_exit_code([v]) == 1
