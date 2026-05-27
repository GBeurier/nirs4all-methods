#!/usr/bin/env python3
"""Donor-operator REFERENCE timing pipeline (dashboard Layer 3).

Layers 1-2 time the donor ops through the C++ tiers and the Python bindings.
This times the external real-library reference — **nirs4all**, the universal
donor baseline (every donor op was donated from it) — in-process, and records
the live n4m↔nirs4all cross-parity. Output fills the previously-empty
`nirs4all` column for donor methods (donors named `ref_python_nirs4all` as
their reference but no nirs4all rows existed).

Per the Codex design ruling:
  * one universal `ref_python_nirs4all` reference (scipy/sklearn/R deferred);
  * stochastic augmenters + seeded ops are timing-only (nirs4all RNG ≠ n4m's);
  * deterministic ops get value cross-parity at rmse_rel <= 1e-6 (exact for
    masks/indices); known impl differences are allowlisted as expected
    divergences (`donor_ops._N4_EXPECTED_DIVERGENCE`).

The reference spec + invocation + cross-parity classification live in
`donor_ops.py` (`reference_for`, `run_reference`), validated against the live
library. Output: `dashboard_refresh_*_donor_reference.csv` (gitignored), merged
by `build_landing.py` into the `nirs4all` column.
"""
from __future__ import annotations

import argparse
import csv
import os
import sys
import traceback
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE / "scripts"))
sys.path.insert(0, str(REPO / "bindings/python/src"))

import donor_ops as D  # noqa: E402
from _common import _safe_version  # noqa: E402
from bench_donor_binding_timing import FIELDNAMES, time_op  # noqa: E402

DEFAULT_OUT = REPO / "benchmarks/cross_binding/results/dashboard_refresh_2026_05_27_donor_reference.csv"
DEFAULT_SIZES = ["250x50", "50x250"]
DEFAULT_SEED_BASE = 1_234_567_890
LIBP4A_BUILD = "blas-omp"
REF_TOL = 1e-6


def _rmse(a: np.ndarray, b: np.ndarray) -> tuple[float, float]:
    """Absolute and relative RMSE of `a` (n4m) vs `b` (reference)."""
    if a.shape != b.shape:
        return float("inf"), float("inf")
    abs_rmse = float(np.sqrt(np.mean((a - b) ** 2))) if a.size else 0.0
    denom = float(np.sqrt(np.mean(b ** 2))) or 1.0
    return abs_rmse, abs_rmse / denom


def run_cell(spec: D.DonorOpSpec, n: int, p: int, seed_base: int, max_runs: int,
             versions: str) -> dict:
    ref = D.reference_for(spec)
    inp = D.build_inputs(spec, n, p, seed_base)
    row = {k: "" for k in FIELDNAMES}
    row.update(
        algorithm=spec.dashboard_id, backend="ref_python_nirs4all", language="Python",
        tier="nirs4all", kind="external", n=n, p=p, threads=1,
        libp4a_build=LIBP4A_BUILD, seed_base=seed_base, prediction_seed=seed_base,
        reference_kind="nirs4all_sanctioned", is_canonical_reference="True",
        reference_parity_ref="ref_python_nirs4all", reference_parity_tolerance=REF_TOL,
        versions_json=versions, ok="False",
    )
    try:
        ref_vec = D.run_reference(spec, ref, inp, seed_base)  # also the parity vector
        stats = time_op(lambda: D.run_reference(spec, ref, inp, seed_base), max_runs, seed_base)
        row.update({k: stats[k] for k in stats})
        row["ok"] = "True"
    except Exception as exc:
        row["reason"] = f"nirs4all reference failed: {type(exc).__name__}: {exc}"
        return row

    # Live n4m↔nirs4all cross-parity (n4m idiomatic output is the n4m side;
    # it equals the raw + C++ output for the binding-validated ops).
    if not ref.value_parity:
        row.update(reference_parity_rmse_abs="", reference_parity_rmse_rel="",
                   reference_parity_ok="", reference_parity_note=f"timing-only: {ref.note}")
        return row
    try:
        n4m_vec = D.run_idiom(spec, inp, seed_base)
    except Exception as exc:
        row["reference_parity_note"] = f"n4m side failed: {type(exc).__name__}: {exc}"
        return row
    if spec.parity in ("mask", "indices"):
        ok = n4m_vec.shape == ref_vec.shape and np.array_equal(n4m_vec, ref_vec)
        abs_rmse = 0.0 if ok else 1.0
        rel = abs_rmse
    else:
        abs_rmse, rel = _rmse(n4m_vec, ref_vec)
        ok = rel <= ref.tol_rel
    expected = D._N4_EXPECTED_DIVERGENCE.get(spec.bench_id)
    if ok:
        note = "n4m matches nirs4all"
    elif expected:
        note = f"expected divergence: {expected}"
    else:
        note = "UNEXPECTED divergence vs nirs4all"
    row.update(reference_parity_rmse_abs=abs_rmse, reference_parity_rmse_rel=rel,
               reference_parity_ok=str(ok), reference_parity_note=note)
    return row


def cell_key(row: dict) -> tuple:
    return (row["algorithm"], row["backend"], row.get("libp4a_build", ""),
            str(row["n"]), str(row["p"]), str(row.get("threads", 1)))


def load_existing(path: Path) -> dict[tuple, dict]:
    if not path.exists():
        return {}
    with path.open(newline="") as f:
        return {cell_key(r): r for r in csv.DictReader(f)
                if str(r.get("ok")).lower() == "true" or (r.get("reason") or "").strip()}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out-csv", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--sizes", nargs="+", default=DEFAULT_SIZES)
    ap.add_argument("--runs", type=int, default=200)
    ap.add_argument("--seed", type=int, default=DEFAULT_SEED_BASE)
    ap.add_argument("--methods", nargs="*")
    ap.add_argument("--resume-existing", action="store_true")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    for var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "BLIS_NUM_THREADS"):
        os.environ.setdefault(var, "1")

    try:
        import nirs4all
    except ImportError:
        print("nirs4all not importable; set PYTHONPATH to the nirs4all checkout", file=sys.stderr)
        return 1
    import n4m
    # `timing_schema` must be adaptive-v1 so these live-timing rows outrank the
    # parity-only `ref_python_nirs4all` rows the fixture generator emits for the
    # ~90 nirs4all-backed donor methods (build_landing dedups on schema first).
    versions = (f'{{"reference_library":"nirs4all","reference_version":"{getattr(nirs4all,"__version__","?")}",'
                f'"n4m":"{n4m.version()}","numpy":"{_safe_version("numpy")}","timing_schema":"adaptive-v1"}}')

    sizes = [tuple(int(x) for x in s.lower().split("x")) for s in args.sizes]
    specs = [s for s in D.all_specs() if s.on_dashboard]
    if args.methods:
        specs = [s for s in specs if s.bench_id in set(args.methods)]

    out = {} if args.force else load_existing(args.out_csv)
    timed = skipped = failed = 0
    for spec in specs:
        for n, p in sizes:
            key = (spec.dashboard_id, "ref_python_nirs4all", LIBP4A_BUILD, str(n), str(p), "1")
            if not args.force and key in out:
                skipped += 1
                continue
            try:
                row = run_cell(spec, n, p, args.seed, args.runs, versions)
            except Exception:
                failed += 1
                print(f"  CELL-FAIL {spec.bench_id} {n}x{p}:\n{traceback.format_exc()}", file=sys.stderr)
                continue
            out[cell_key(row)] = row
            ms = row.get("reported_ms") or row.get("reason", "—")
            print(f"  {spec.dashboard_id:30s} {n}x{p}: nirs4all={ms} "
                  f"parity={row.get('reference_parity_note', '')[:48]}")
            timed += 1

    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    with args.out_csv.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows(out.values())
    print(f"\nWrote {len(out)} nirs4all reference rows (timed={timed}, skipped={skipped}, "
          f"failed={failed}) to {args.out_csv}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
