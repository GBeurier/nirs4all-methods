#!/usr/bin/env python3
"""Golden-snapshot generator (Phase-C-min pilot).

Produces and verifies `parity/snapshots/current/<method>/<scenario>.json` —
the scenario-based layout that full Phase C (C4) will migrate the 202 legacy
`parity/fixtures/*.json` into. This pilot establishes the layout + provenance
for a few methods WITHOUT the full migration, the Docker generators, or the
catalog coupling (all deferred).

`source_kind` is **n4m_self**: these snapshots capture n4m's OWN output, so they
are a GOLDEN REGRESSION gate (does the current build still produce the same
numbers as the committed snapshot?), NOT external-correctness validation. The
latter needs the deferred Docker reference generators.

Two modes (Codex review):
  --check (default)  regenerate in-memory, compare to the committed snapshot,
                     FAIL on drift or missing snapshot. Read-only gate.
  --write            regenerate and overwrite the committed snapshots.

A method whose libn4m build is absent is SKIPped (visible), not failed.
Wired to `make snapshot METHOD=<id|all>` (which passes --write).
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
SNAP_DIR = REPO / "parity" / "snapshots" / "current"

# Pilot scenarios: (method, n, p, n_components, seed). Cheap, deterministic.
SCENARIOS: tuple[tuple[str, int, int, int, int], ...] = (
    ("pls", 60, 30, 3, 20240517),
    ("pcr", 60, 30, 3, 20240517),
    ("aom_pls", 80, 40, 3, 20240517),
    ("pop_pls", 80, 40, 3, 20240517),
)

TOOL_VERSION = "phase-c-min-1"
PASS, FAIL, SKIP = "PASS", "FAIL", "SKIP"


def scenario_id(method: str, n: int, p: int, seed: int) -> str:
    return f"{method}_{n}x{p}_s{seed}"


def snapshot_path(method: str, sid: str) -> Path:
    return SNAP_DIR / method / f"{sid}.json"


def build_snapshot(method: str, n: int, p: int, nc: int, seed: int,
                   output: np.ndarray) -> dict:
    """The committed golden snapshot. Deliberately carries only STABLE fields:
    the output, the seed/dims, the ABI version (from the header), and tool
    provenance. Binary identity (lib sha256/bytes) is intentionally NOT stored —
    it changes on every rebuild/preset/host and would dirty every snapshot on a
    routine `make snapshot` even when the output is identical. It is reported on
    the --write console instead (runtime info, not golden data)."""
    from _n4m_runner import abi_version

    out = np.asarray(output, dtype=np.float64).ravel()
    return {
        "schema": "snapshot-v1",
        "source_kind": "n4m_self",
        "method": method,
        "scenario": scenario_id(method, n, p, seed),
        "seed": seed,
        "n": n,
        "p": p,
        "n_components": nc,
        "abi": abi_version(),
        "provenance": {
            "generated_by": "parity/generators/run.py",
            "tool_version": TOOL_VERSION,
            "command": f"run.py --method {method} --write",
        },
        "output_key": "predictions",
        "output_len": int(out.size),
        "output": out.tolist(),
    }


def compare_output(fresh: np.ndarray, snapshot: dict, tol: float = 1e-12) -> tuple[bool, str]:
    """Strict golden compare of a fresh run vs a committed snapshot. Pure."""
    from benchmarks.parity_timing._comparator import rms, rmse

    ref = np.asarray(snapshot.get("output", []), dtype=np.float64)
    cur = np.asarray(fresh, dtype=np.float64).ravel()
    if cur.shape != ref.shape:
        return False, f"shape drift ({cur.shape} vs snapshot {ref.shape})"
    ref_rms = rms(ref)
    rel = rmse(cur, ref) / ref_rms if ref_rms > 0 else rmse(cur, ref)
    return (rel <= tol), f"rmse_rel={rel:.2e} (tol {tol:.0e})"


@dataclass
class SnapVerdict:
    method: str
    scenario: str
    status: str
    note: str


def process(method: str, n: int, p: int, nc: int, seed: int, *,
            write: bool, lib_dir: str) -> SnapVerdict:
    """Run/check one scenario against a RESOLVED libn4m build. A run failure
    here is a FAIL (the build exists) — only the total absence of a build is a
    SKIP, handled in `main()` before this is called."""
    from _n4m_runner import run_n4m

    sid = scenario_id(method, n, p, seed)
    ok, preds, why = run_n4m(method, n, p, nc, seed, lib_dir=lib_dir, name="snap")
    if not ok:
        return SnapVerdict(method, sid, FAIL, f"run failed: {why}")
    path = snapshot_path(method, sid)
    if write:
        snap = build_snapshot(method, n, p, nc, seed, preds)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(snap, indent=2) + "\n")
        return SnapVerdict(method, sid, PASS, f"wrote {path.relative_to(REPO)}")
    # check mode
    if not path.exists():
        return SnapVerdict(method, sid, FAIL, f"missing snapshot {path.relative_to(REPO)} (run --write)")
    snap = json.loads(path.read_text())
    match, note = compare_output(preds, snap)
    return SnapVerdict(method, sid, PASS if match else FAIL, note)


def selected_scenarios(method: str) -> list[tuple]:
    if method in ("all", "", None):
        return list(SCENARIOS)
    return [s for s in SCENARIOS if s[0] == method]


def gate_exit_code(verdicts: list[SnapVerdict]) -> int:
    return 1 if any(v.status == FAIL for v in verdicts) else 0


def render(verdicts: list[SnapVerdict], write: bool) -> str:
    icon = {PASS: "✅", FAIL: "❌", SKIP: "⚠️ "}
    mode = "write" if write else "check"
    lines = ["", f"Golden snapshots (n4m_self, {mode} mode)", "=" * 60]
    for v in verdicts:
        lines.append(f"{icon.get(v.status, '?')} {v.status:4} {v.scenario:22} {v.note}")
    lines.append("=" * 60)
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--method", default="all", help="method id or 'all'")
    ap.add_argument("--ref", default="n4m_self",
                    help="reference kind (only n4m_self is wired in the pilot)")
    mode = ap.add_mutually_exclusive_group()
    mode.add_argument("--write", action="store_true", help="regenerate + overwrite snapshots")
    mode.add_argument("--check", action="store_true", help="compare to committed snapshots (default)")
    ap.add_argument("--lib-dir", default=None, help="libn4m build dir (auto-discovered if omitted)")
    args = ap.parse_args(argv)

    if args.ref != "n4m_self":
        print(f"only --ref n4m_self is wired in the Phase-C-min pilot (got {args.ref!r}); "
              "external Docker reference generators are deferred.", file=sys.stderr)
        return 2

    scenarios = selected_scenarios(args.method)
    if not scenarios:
        print(f"no scenario for method {args.method!r}; known: "
              f"{', '.join(sorted({s[0] for s in SCENARIOS}))}", file=sys.stderr)
        return 2

    lib_dir = args.lib_dir
    if lib_dir is None:
        from _n4m_runner import discover_lib_dir
        lib_dir = discover_lib_dir()

    write = bool(args.write)  # check is the default when neither flag is given
    if lib_dir is None:
        # No build at all → SKIP (local convenience). A RESOLVED build that then
        # fails to run is a FAIL inside process(), so CI cannot pass on breakage.
        print("no libn4m build found — build one or pass --lib-dir; skipping.", file=sys.stderr)
        verdicts = [SnapVerdict(m, scenario_id(m, n, p, seed), SKIP, "no libn4m build")
                    for (m, n, p, nc, seed) in scenarios]
    else:
        if write:
            from _n4m_runner import lib_identity
            ident = lib_identity(lib_dir)
            print(f"libn4m: {lib_dir} sha256={ident['lib_sha256']}")  # runtime provenance
        verdicts = [process(m, n, p, nc, seed, write=write, lib_dir=lib_dir)
                    for (m, n, p, nc, seed) in scenarios]
    print(render(verdicts, write))
    return gate_exit_code(verdicts)


if __name__ == "__main__":
    raise SystemExit(main())
