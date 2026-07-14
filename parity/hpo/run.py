#!/usr/bin/env python3
# SPDX-License-Identifier: CECILL-2.1
"""HPO cross-binding parity gate (Track Q).

Runs every HpoSpec through the native libn4m optimizer and checks three things:

  1. golden stability  — the study trace matches the committed golden
                         (``golden/<id>.json``). This is the cross-binding
                         contract: R / MATLAB-Octave / JS-WASM bindings must
                         reproduce the same golden byte-for-byte.
  2. Sobol Tier-A       — the Sobol cell equals scipy.stats.qmc.Sobol(scramble=False).
  3. pruner decisions   — native prune verdicts match an independent pure-Python
                         reimplementation of the documented rule.

It also executes the versioned 9 x 5 compatibility contract: all 45 native
sampler-pruner compositions must replay exactly, survive a public checkpoint,
and preserve their ordered event tape; every declared transverse refusal must
return its stable C ABI status code.

Usage:
  N4M_LIB_PATH=.../libn4m.so.2.2.0 python run.py            # check (CI gate)
  N4M_LIB_PATH=.../libn4m.so.2.2.0 python run.py --update    # regenerate goldens
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))

import numpy as np  # noqa: E402

import run_native  # noqa: E402
import specs  # noqa: E402
from comparators import compare_traces, dump_golden, golden_path, load_golden  # noqa: E402
from compatibility import (  # noqa: E402
    CompatibilityContractError,
    load_contract as load_compatibility_contract,
    run_compatibility_gate,
)
from references import (  # noqa: E402
    asha_should_prune,
    hyperband_should_prune,
    median_should_prune,
    racing_should_prune,
    sobol_reference,
)
from trace_contract import validate_conformance_pack, validate_study_trace  # noqa: E402

GOLDEN = _HERE / "golden"
CONTRACTS = _HERE / "contracts"


def _reference_pruned(spec, trace: list[dict]) -> list[int]:
    order = [rec["id"] for rec in trace]
    eta = spec.reduction_factor or 3
    history: dict[int, list[tuple[int, float]]] = {}
    pruned_by_id = {rec["id"]: -1 for rec in trace}
    events = [
        (event["sequence"], rec["id"], event)
        for rec in trace
        for event in rec.get("intermediates", ())
    ]
    events.sort(key=lambda item: item[0])
    maximize = spec.direction == "maximize"
    for _sequence, tid, event in events:
        step, score = event["step"], event["score"]
        history.setdefault(tid, []).append((step, score))
        if spec.pruner == "median":
            decision = median_should_prune(
                history, tid, step, score, spec.n_startup_trials, maximize
            )
        elif spec.pruner == "asha":
            decision = asha_should_prune(history, tid, step, score, eta, maximize)
        elif spec.pruner == "hyperband":
            decision = hyperband_should_prune(
                order,
                history,
                tid,
                step,
                score,
                eta,
                spec.max_resource,
                maximize,
            )
        elif spec.pruner == "racing":
            decision = racing_should_prune(history, tid, order=order, maximize=maximize)
        else:
            decision = False
        if decision:
            pruned_by_id[tid] = step

    return [pruned_by_id[rec["id"]] for rec in trace]


def main() -> int:
    ap = argparse.ArgumentParser(description="HPO cross-binding parity gate")
    ap.add_argument("--update", action="store_true", help="regenerate golden traces")
    ap.add_argument("--only", default="", help="run a single spec id")
    args = ap.parse_args()

    validate_conformance_pack(CONTRACTS / "study_trace_conformance_pack.v1.json")
    lifecycle = json.loads((GOLDEN / "lifecycle_states.v1.json").read_text())
    validate_study_trace(lifecycle)
    print("OK   lifecycle_states StudyTrace v1 semantic contract")
    compatibility_contract = load_compatibility_contract()
    print("OK   sampler_pruner compatibility contract v1 (45 declared cells)")

    if "N4M_LIB_PATH" not in os.environ:
        print("SKIP: N4M_LIB_PATH not set (no libn4m build)", file=sys.stderr)
        return 0

    registry = [s for s in specs.REGISTRY if not args.only or s.id == args.only]
    failures: list[str] = []

    try:
        reports, refusal_probes = run_compatibility_gate(compatibility_contract)
    except CompatibilityContractError as error:
        failures.append(f"sampler-pruner compatibility: {error}")
    else:
        print(
            "OK   sampler_pruner compatibility "
            f"{len(reports)}/45 cells + {refusal_probes} typed refusal probes"
        )

    for spec in registry:
        trace = run_native.run_spec(spec)

        # (2) Sobol Tier-A external reference.
        if spec.sampler == "sobol":
            dim = len(spec.space)
            ref = sobol_reference(dim, spec.n_trials)
            got = np.array(
                [[rec["params"][p.name] for p in spec.space] for rec in trace]
            )
            if not np.array_equal(got, ref):
                failures.append(
                    f"{spec.id}: Sobol != scipy (max |Δ|={np.max(np.abs(got - ref)):.2e})"
                )
            else:
                print(f"OK   {spec.id:16s} Sobol Tier-A == scipy")

        # (3) pruner decision cross-check against the pure-Python rule.
        if spec.pruner != "none" and spec.intermediate:
            native = [rec.get("pruned_at", -1) for rec in trace]
            ref = _reference_pruned(spec, trace)
            if native != ref:
                bad = [
                    (i, native[i], ref[i])
                    for i in range(len(ref))
                    if native[i] != ref[i]
                ]
                failures.append(
                    f"{spec.id}: pruner decisions differ from reference at {bad}"
                )
            else:
                npr = sum(1 for x in native if x >= 0)
                print(
                    f"OK   {spec.id:16s} pruner decisions == reference ({npr} pruned)"
                )

        # (1) golden stability / cross-binding contract.
        gp = golden_path(GOLDEN, spec.id)
        if args.update:
            dump_golden(GOLDEN, spec.id, trace)
            print(f"WROTE {spec.id:16s} -> {gp.relative_to(_HERE.parents[1])}")
        elif not gp.exists():
            failures.append(f"{spec.id}: no golden ({gp.name}); run with --update")
        else:
            errs = compare_traces(trace, load_golden(GOLDEN, spec.id), tol=spec.tol)
            if errs:
                failures.append(
                    f"{spec.id}: {len(errs)} golden mismatch(es); first: {errs[0]}"
                )
            else:
                print(f"OK   {spec.id:16s} golden trace stable ({len(trace)} trials)")

    if failures:
        print("\nFAILURES:", file=sys.stderr)
        for f in failures:
            print("  ✗ " + f, file=sys.stderr)
        return 1
    print(
        f"\nHPO parity gate: {len(registry)} selected specs + 45 compatibility cells OK"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
