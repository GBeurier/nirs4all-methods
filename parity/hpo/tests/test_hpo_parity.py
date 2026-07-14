# SPDX-License-Identifier: CECILL-2.1
"""Pytest wrapper for the HPO cross-binding parity gate.

Skips when no libn4m build is available (N4M_LIB_PATH unset). With a build, every
HpoSpec must reproduce its golden trace, Sobol must match scipy, and the native
pruner verdicts must match the pure-Python reference rules.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE.parent))

pytestmark = pytest.mark.skipif(
    "N4M_LIB_PATH" not in os.environ, reason="no libn4m build (set N4M_LIB_PATH)"
)


def _specs():
    import specs

    return specs.REGISTRY


@pytest.mark.parametrize("spec", _specs(), ids=lambda s: s.id)
def test_golden_trace_stable(spec):
    import run_native
    from comparators import compare_traces, golden_path, load_golden

    golden_dir = _HERE.parent / "golden"
    gp = golden_path(golden_dir, spec.id)
    assert gp.exists(), f"missing golden {gp.name} — run parity/hpo/run.py --update"
    trace = run_native.run_spec(spec)
    errs = compare_traces(trace, load_golden(golden_dir, spec.id), tol=spec.tol)
    assert not errs, f"{spec.id}: {errs[:3]}"


def test_sobol_matches_scipy():
    import numpy as np
    import run_native
    import specs
    from references import sobol_reference

    spec = specs.by_id("sobol_unit3")
    trace = run_native.run_spec(spec)
    got = np.array([[rec["params"][p.name] for p in spec.space] for rec in trace])
    ref = sobol_reference(len(spec.space), spec.n_trials)
    assert np.array_equal(got, ref)


@pytest.mark.parametrize(
    "spec_id", ["median_prune", "asha_prune", "hyperband_prune", "racing_parallel"]
)
def test_pruner_decisions_match_reference(spec_id):
    import run_native
    import specs
    from run import _reference_pruned

    spec = specs.by_id(spec_id)
    trace = run_native.run_spec(spec)
    native = [rec.get("pruned_at", -1) for rec in trace]
    assert native == _reference_pruned(spec, trace)


def test_failed_trials_keep_structured_host_errors():
    import run_native
    import specs

    trace = run_native.run_spec(specs.by_id("random_failed_trials"))
    assert [rec["id"] for rec in trace if rec["status"] == 3] == [2, 5]
    for rec in trace:
        if rec["status"] == 3:
            assert rec["error"]["code"] == "OBJECTIVE_EVALUATION_FAILED"
            assert rec["error"]["details"]["trial_id"] == rec["id"]
            assert "score" not in rec


@pytest.mark.parametrize("spec", _specs(), ids=lambda s: s.id)
def test_rich_trace_sequences_cover_every_native_event(spec):
    import run_native

    trace = run_native.run_spec(spec)
    events: list[int] = []
    for rec in trace:
        events.append(rec["asked_at"])
        events.extend(item["sequence"] for item in rec.get("intermediates", []))
        if rec["status"] != 0:
            events.append(rec["terminal_at"])
    assert sorted(events) == list(range(len(events)))


def test_parallel_racing_schedule_is_really_interleaved():
    import run_native
    import specs

    trace = run_native.run_spec(specs.by_id("racing_parallel"))
    assert max(rec["asked_at"] for rec in trace[:4]) < min(
        event["sequence"] for rec in trace[:4] for event in rec["intermediates"]
    )
    first_step_ids = [
        rec["id"]
        for sequence in range(4, 8)
        for rec in trace[:4]
        for event in rec["intermediates"]
        if event["sequence"] == sequence
    ]
    assert first_step_ids == [2, 0, 3, 1]
