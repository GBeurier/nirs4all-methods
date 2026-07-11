# SPDX-License-Identifier: CECILL-2.1
"""Comparators for HPO parity traces.

- ``compare_traces`` — the cross-binding gate: two StudyTraces must agree on the
  ordered proposed parameters (and scores / prune decisions), exactly for
  deterministic samplers or within an absolute tolerance for float axes.
- helpers to load/dump the golden JSON.
"""
from __future__ import annotations

import json
from pathlib import Path


class TraceMismatch(AssertionError):
    pass


def _cmp_value(a, b, tol: float) -> bool:
    if isinstance(a, str) or isinstance(b, str):
        return a == b
    if isinstance(a, bool) or isinstance(b, bool):
        return bool(a) == bool(b)
    if tol <= 0.0:
        return a == b  # exact bit-for-bit
    return abs(float(a) - float(b)) <= tol


def compare_traces(actual: list[dict], golden: list[dict], tol: float = 0.0) -> list[str]:
    """Return a list of human-readable mismatch strings (empty == parity)."""
    errs: list[str] = []
    if len(actual) != len(golden):
        errs.append(f"trace length {len(actual)} != golden {len(golden)}")
        return errs
    for i, (a, g) in enumerate(zip(actual, golden)):
        ap, gp = a.get("params", {}), g.get("params", {})
        if set(ap) != set(gp):
            errs.append(f"[{i}] param names {sorted(ap)} != {sorted(gp)}")
            continue
        for name in gp:
            if not _cmp_value(ap[name], gp[name], tol):
                errs.append(f"[{i}] param {name!r}: {ap[name]!r} != golden {gp[name]!r}")
        if "score" in g and not _cmp_value(a.get("score"), g["score"], tol):
            errs.append(f"[{i}] score {a.get('score')!r} != golden {g['score']!r}")
        if "pruned_at" in g and a.get("pruned_at") != g["pruned_at"]:
            errs.append(f"[{i}] pruned_at {a.get('pruned_at')!r} != golden {g['pruned_at']!r}")
        if "status" in g and a.get("status") != g["status"]:
            errs.append(f"[{i}] status {a.get('status')!r} != golden {g['status']!r}")
    return errs


def golden_path(golden_dir: Path, spec_id: str) -> Path:
    return golden_dir / f"{spec_id}.json"


def load_golden(golden_dir: Path, spec_id: str) -> list[dict]:
    return json.loads(golden_path(golden_dir, spec_id).read_text())


def dump_golden(golden_dir: Path, spec_id: str, trace: list[dict]) -> None:
    golden_dir.mkdir(parents=True, exist_ok=True)
    golden_path(golden_dir, spec_id).write_text(json.dumps(trace, indent=2, sort_keys=True) + "\n")
