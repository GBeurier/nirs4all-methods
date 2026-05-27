#!/usr/bin/env python3
"""Self-consistency comparator for methods with no external reference.

Some methods are this project's scientific differentiators (AOM-PLS, POP-PLS)
and have no external library to check against. Their correctness gate is
**self-consistency**, not numerical parity:

  - determinism  — same seed → bit-identical output across independent runs
                   (catches unseeded RNG / hash-order / nondeterministic
                   selection or CV tie-breaking).
  - invariants   — structural properties the output must always satisfy
                   (finite, non-empty, expected element count).

PHASE-C-MIN SCOPE (catalog-free pilot). In full Phase C these specs come from
each method's `self_consistency` block in `catalog/methods/<id>.yaml`. The
catalog is still M2-auto-discovered stubs (no `status:`/`self_consistency`),
so this file carries a SMALL, EXPLICIT, hand-curated spec set for the
differentiator methods. It is a temporary bridge — when the catalog gains real
`self_consistency` blocks, this registry is replaced by a catalog read. Do NOT
grow it into a parallel catalog.

Wired to `make parity-paper-only` (METHOD=all|<id>). Reuses the deterministic
n4m runner (`parity/_n4m_runner`) — no new FFI/runner. SKIP is reserved for the
TOTAL ABSENCE of a libn4m build (local convenience). Once a build is resolved
or passed via --lib-dir, a method that fails to run, is nondeterministic, or
violates an invariant is a FAIL — so CI cannot pass on a crashed AOM/POP run.
"""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "parity"))
RESULTS = REPO / "parity" / "results" / "self_consistency.json"

PASS, FAIL, SKIP = "PASS", "FAIL", "SKIP"


@dataclass(frozen=True)
class SelfConsistencySpec:
    method: str
    n: int
    p: int
    nc: int
    seed: int
    expected_len: int  # element count of the raveled output
    invariants: tuple[str, ...]
    note: str


# Hand-curated, temporary, catalog-free (see module docstring).
SPECS: tuple[SelfConsistencySpec, ...] = (
    SelfConsistencySpec(
        "aom_pls", n=80, p=40, nc=3, seed=20240517, expected_len=80,
        invariants=("finite", "nonempty", "len"),
        note="AOM-PLS global adaptive operator selection (paper_only differentiator)"),
    SelfConsistencySpec(
        "pop_pls", n=80, p=40, nc=3, seed=20240517, expected_len=80,
        invariants=("finite", "nonempty", "len"),
        note="POP-PLS per-component adaptive operator selection (paper_only differentiator)"),
    SelfConsistencySpec(
        "aom_preprocess", n=80, p=40, nc=3, seed=20240517, expected_len=80 * 40,
        invariants=("finite", "nonempty", "len"),
        note="AOM preprocessing primitive — transformed (n x p) output"),
)


def check_invariants(arr: np.ndarray, spec: SelfConsistencySpec) -> list[tuple[str, bool, str]]:
    """Evaluate a spec's declared invariants. Pure — unit-testable."""
    a = np.asarray(arr)
    results: list[tuple[str, bool, str]] = []
    for name in spec.invariants:
        if name == "finite":
            ok = bool(np.all(np.isfinite(a)))
            results.append((name, ok, "" if ok else "non-finite values present"))
        elif name == "nonempty":
            ok = a.size > 0
            results.append((name, ok, "" if ok else "empty output"))
        elif name == "len":
            ok = a.size == spec.expected_len
            results.append((name, ok, "" if ok else f"size {a.size} != expected {spec.expected_len}"))
        else:  # pragma: no cover - guards against a typo'd invariant name
            results.append((name, False, f"unknown invariant {name!r}"))
    return results


def check_determinism(a: np.ndarray, b: np.ndarray) -> tuple[bool, str]:
    """Two independent same-seed runs must be bit-identical. Pure."""
    a = np.asarray(a)
    b = np.asarray(b)
    if a.shape != b.shape:
        return False, f"shape drift between runs ({a.shape} vs {b.shape})"
    if not np.array_equal(a, b):
        return False, f"nondeterministic: max|Δ|={float(np.max(np.abs(a - b))):.3e}"
    return True, ""


@dataclass
class MethodVerdict:
    method: str
    status: str
    determinism: str
    invariants: list[tuple[str, bool, str]]
    note: str


def evaluate_spec(spec: SelfConsistencySpec, lib_dir: str) -> MethodVerdict:
    """Evaluate one spec against a RESOLVED libn4m build. A run failure here is
    a real FAIL (the build exists; the method must run) — only the total
    absence of a build is a SKIP, handled in `run()` before this is called."""
    from _n4m_runner import run_n4m

    ok_a, preds_a, why_a = run_n4m(spec.method, spec.n, spec.p, spec.nc,
                                   spec.seed, lib_dir=lib_dir, name="sc_a")
    if not ok_a:
        return MethodVerdict(spec.method, FAIL, f"run failed: {why_a}", [], spec.note)
    ok_b, preds_b, why_b = run_n4m(spec.method, spec.n, spec.p, spec.nc,
                                   spec.seed, lib_dir=lib_dir, name="sc_b")
    if not ok_b:
        return MethodVerdict(spec.method, FAIL, f"second run failed: {why_b}", [], spec.note)

    det_ok, det_note = check_determinism(preds_a, preds_b)
    inv = check_invariants(preds_a, spec)
    status = PASS if det_ok and all(ok for _, ok, _ in inv) else FAIL
    return MethodVerdict(spec.method, status,
                         "ok" if det_ok else det_note, inv, spec.note)


def gate_exit_code(verdicts: list[MethodVerdict]) -> int:
    """1 if any spec FAILED; 0 otherwise (SKIP — absent build — is tolerated)."""
    return 1 if any(v.status == FAIL for v in verdicts) else 0


def run(method: str, lib_dir: str | None) -> tuple[list[MethodVerdict], int]:
    specs = SPECS if method in ("all", "", None) else tuple(s for s in SPECS if s.method == method)
    if not specs:
        print(f"no self-consistency spec for method {method!r}; "
              f"known: {', '.join(s.method for s in SPECS)}", file=sys.stderr)
        return [], 2
    if lib_dir is None:
        from _n4m_runner import discover_lib_dir
        lib_dir = discover_lib_dir()
    if lib_dir is None:
        print("no libn4m build found — build one (e.g. cmake --preset blas-omp) "
              "or pass --lib-dir; skipping.", file=sys.stderr)
        return [MethodVerdict(s.method, SKIP, "no libn4m build", [], s.note) for s in specs], 0
    return [evaluate_spec(s, lib_dir) for s in specs], 0


def render(verdicts: list[MethodVerdict], code: int) -> str:
    icon = {PASS: "✅", FAIL: "❌", SKIP: "⚠️ "}
    lines = ["", "Self-consistency (paper-only differentiators)", "=" * 64]
    for v in verdicts:
        lines.append(f"{icon.get(v.status, '?')} {v.status:4} {v.method:16} "
                     f"determinism={v.determinism}")
        for name, ok, why in v.invariants:
            lines.append(f"      {'·' if ok else '✗'} {name}: {'ok' if ok else why}")
    lines.append("=" * 64)
    return "\n".join(lines)


def _to_json(verdicts: list[MethodVerdict]) -> dict:
    return {
        "schema": "self-consistency-v1",
        "source_kind": "n4m_self",
        "methods": {
            v.method: {
                "status": v.status,
                "determinism": v.determinism,
                "invariants": {name: {"ok": ok, "note": why} for name, ok, why in v.invariants},
                "note": v.note,
            } for v in verdicts
        },
        "totals": {
            "methods": len(verdicts),
            "pass": sum(v.status == PASS for v in verdicts),
            "fail": sum(v.status == FAIL for v in verdicts),
            "skip": sum(v.status == SKIP for v in verdicts),
        },
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--method", default="all", help="method id or 'all'")
    ap.add_argument("--lib-dir", default=None, help="libn4m build dir (auto-discovered if omitted)")
    ap.add_argument("--out", type=Path, default=RESULTS, help="verdict JSON output path")
    args = ap.parse_args(argv)

    verdicts, code = run(args.method, args.lib_dir)
    if code == 2:
        return 2
    print(render(verdicts, code))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(_to_json(verdicts), indent=2) + "\n")
    exit_code = gate_exit_code(verdicts)
    print("PAPER-ONLY GATE PASS" if exit_code == 0 else "PAPER-ONLY GATE FAIL")
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
