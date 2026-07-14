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
        return isinstance(a, str) and isinstance(b, str) and a == b
    if isinstance(a, bool) or isinstance(b, bool):
        return isinstance(a, bool) and isinstance(b, bool) and a == b
    if a is None or b is None:
        return a is b
    if tol <= 0.0:
        return a == b  # exact bit-for-bit
    return abs(float(a) - float(b)) <= tol


def _compare_node(actual, golden, path: str, tol: float, errs: list[str]) -> None:
    if isinstance(golden, dict):
        if not isinstance(actual, dict):
            errs.append(f"{path}: {type(actual).__name__} != golden object")
            return
        if set(actual) != set(golden):
            errs.append(f"{path}: keys {sorted(actual)} != golden {sorted(golden)}")
            return
        for key, expected in golden.items():
            _compare_node(actual[key], expected, f"{path}.{key}", tol, errs)
        return
    if isinstance(golden, list):
        if not isinstance(actual, list):
            errs.append(f"{path}: {type(actual).__name__} != golden array")
            return
        if len(actual) != len(golden):
            errs.append(f"{path}: length {len(actual)} != golden {len(golden)}")
            return
        for index, (value, expected) in enumerate(zip(actual, golden)):
            _compare_node(value, expected, f"{path}[{index}]", tol, errs)
        return
    if not _cmp_value(actual, golden, tol):
        errs.append(f"{path}: {actual!r} != golden {golden!r}")


def compare_traces(
    actual: list[dict], golden: list[dict], tol: float = 0.0
) -> list[str]:
    """Return full-structure trace mismatches (empty means exact parity)."""
    errs: list[str] = []
    _compare_node(actual, golden, "trace", tol, errs)
    return errs


def golden_path(golden_dir: Path, spec_id: str) -> Path:
    return golden_dir / f"{spec_id}.json"


def load_golden(golden_dir: Path, spec_id: str) -> list[dict]:
    return json.loads(golden_path(golden_dir, spec_id).read_text())


def dump_golden(golden_dir: Path, spec_id: str, trace: list[dict]) -> None:
    golden_dir.mkdir(parents=True, exist_ok=True)
    golden_path(golden_dir, spec_id).write_text(
        json.dumps(trace, indent=2, sort_keys=True) + "\n"
    )
