#!/usr/bin/env python3
"""Parity comparator — the unified parity-gate summary (Phase C, minimal).

Reads the cross-binding result CSVs (`benchmarks/cross_binding/results/
full_matrix.csv` + `dashboard_refresh_*.csv`), REUSES the dashboard's verdict
classification (`docs/_extras/build_landing.build_payload`, so the gate and the
dashboard can never disagree), and writes a method-level parity summary to
`parity/results/latest.json`:

  * **reference parity** — n4m (C++ + bindings) vs each method's canonical
    external reference (sklearn / R-pls / ropls / mixOmics / ikpls / nirs4all …):
    verdict histogram + divergence δ.
  * **binding parity** — each binding tier (python_tier1/2, R, MATLAB, …) vs
    the C++ core: verdict histogram.

A "scenario" here is `(method, n, p, seed, backend)` — see
`parity/SCENARIOS_MIN.md`. The full scenario/snapshot/Docker rebuild (plan
Phase C, C1-C20) is deferred; this gate is CSV-backed and proves the parity
that the existing orchestrator + donor pipelines already produce.

By default this is a SUMMARY (exit 0): it surfaces the full verdict histogram —
including the pre-existing `error` cells (method×backend combos that genuinely
fail, e.g. a method not runnable in a given external library) and `divergent`
cells (recorded, often-expected implementation differences). The authoritative
regression gates remain the pytest suites (`test_donor_binding_specs.py`, the
orchestrator). Pass `--strict` to exit non-zero on any `error` verdict once an
error-free baseline / per-combo allowlist exists.
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "docs" / "_extras"))

DEFAULT_RESULTS = REPO / "benchmarks" / "cross_binding" / "results"
DEFAULT_OUT = REPO / "parity" / "results" / "latest.json"


def build_summary(results_dir: Path) -> dict:
    """Method-level parity summary, derived from the dashboard payload."""
    from build_landing import build_payload  # local import: needs no libn4m

    payload = build_payload(results_dir)["payload"]
    methods = payload["method_scores"]
    ref_tot: Counter = Counter()
    bind_tot: Counter = Counter()
    for sc in methods.values():
        ref_tot.update(sc["reference"])
        bind_tot.update(sc["binding"])
    return {
        "generated_at": payload["generated_at"],
        "host": payload.get("host", {}),
        "scenario_model": "(method, n, p, seed, backend) — see parity/SCENARIOS_MIN.md",
        "methods": methods,
        "totals": {
            "methods": len(methods),
            "reference": dict(ref_tot),
            "binding": dict(bind_tot),
        },
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--results", type=Path, default=DEFAULT_RESULTS)
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--method", default="all", help="one dashboard method id, or 'all'")
    ap.add_argument("--strict", action="store_true",
                    help="exit non-zero on any `error` verdict (off by default)")
    args = ap.parse_args(argv)

    summary = build_summary(args.results)
    if args.method and args.method != "all":
        if args.method not in summary["methods"]:
            print(f"unknown method: {args.method}", file=sys.stderr)
            return 2
        summary["methods"] = {args.method: summary["methods"][args.method]}
        summary["totals"]["methods"] = 1

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(summary, indent=2))

    t = summary["totals"]
    print(f"parity summary → {args.out}")
    print(f"  methods:            {t['methods']}")
    print(f"  reference verdicts: {dict(sorted(t['reference'].items()))}")
    print(f"  binding verdicts:   {dict(sorted(t['binding'].items()))}")
    errors = t["reference"].get("error", 0) + t["binding"].get("error", 0)
    if errors:
        msg = f"{errors} error verdict(s) (method×backend combos that fail)"
        if args.strict:
            print(f"  FAIL (--strict): {msg}", file=sys.stderr)
            return 1
        print(f"  note: {msg} — informational (use --strict to gate)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
