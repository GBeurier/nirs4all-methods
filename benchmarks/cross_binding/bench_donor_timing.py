#!/usr/bin/env python3
"""Donor-operator timing pipeline.

The cross-binding orchestrator times the PLS-family methods; the donor
operators (augmentation / preprocessing / filters / splitters) are only
parity-validated by `n4m_tests` and shipped to the dashboard with blank
timing by ``parity/scripts/append_n4m_tests_to_dashboard.py``. This script
fills the C++ (``cpp`` backend) timing for every donor method the
``n4m_donor_bench`` tool can run, while preserving the parity verdicts that
the fixture suite already established.

Design (mirrors the orchestrator's resume contract):
  * The C++ timing primitive is ``bench/cpp/n4m_donor_bench`` — one adaptive
    timing JSON record per (method, size) cell.
  * For each cell we take the matching ``cpp`` row from the fixtures CSV
    (so all parity / divergence / reference columns stay exactly as the
    1e-12 fixture gate recorded them) and fill in only the timing columns.
  * ``--resume-existing`` loads the output CSV and skips any cell already
    timed (``ok`` and a non-empty ``reported_ms``), so re-runs only do the
    missing/failed cells. ``--force`` re-times everything.

The output is a ``dashboard_refresh_*.csv`` delta; ``build_landing.py`` merges
it ahead of the parity-only fixture rows (same adaptive-v1 schema, newer file
wins the per-cell dedup), so the matrix shows real ms for these methods.
"""
from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
DEFAULT_BENCH = REPO / "build/blas-omp/bench/cpp/n4m_donor_bench"

# The dashboard donor method names come from the n4m_tests test-group names
# (via append_n4m_tests_to_dashboard.py), which sometimes differ from the C
# ABI op id the bench tool exposes. Map dashboard-name -> bench-op where they
# diverge; identical names need no entry. Dashboard rows with no timeable ABI
# op (e.g. the `aug_phase17` bundle group, or the v2-deferred *_simplify_stub
# ops) are left parity-only.
DASHBOARD_TO_BENCH = {
    # augmenter test-group name -> ABI op id (where they diverge)
    "aug_edge_curvature": "aug_edge_curve",
    "aug_edge_artifacts_combined": "aug_edge_artifacts",
    "aug_random_x_operation": "aug_random_x_op",
    "aug_spline_smoothing": "aug_spline_smooth",
    "aug_spline_x_perturbations": "aug_spline_x_perturb",
    "aug_spline_y_perturbations": "aug_spline_y_perturb",
    # wavelength/spectral augmenters are listed without the `aug_` prefix in
    # the fixtures (their n4m_tests group names drop it)
    "band_mask": "aug_band_mask",
    "band_perturb": "aug_band_perturb",
    "channel_dropout": "aug_channel_dropout",
    "gauss_jitter": "aug_gauss_jitter",
    "local_clip": "aug_local_clip",
    "local_warp": "aug_local_warp",
    "magnitude_warp": "aug_magnitude_warp",
    "unsharp_mask": "aug_unsharp_mask",
    "wavelength_shift": "aug_wavelength_shift",
    "wavelength_stretch": "aug_wavelength_stretch",
    # FCK static transformer: n4m_tests group name is "fck"
    "fck": "pp_fck_static",
}
DEFAULT_FIXTURES = REPO / "benchmarks/cross_binding/results/dashboard_refresh_2026_05_22_n4m_fixtures.csv"
DEFAULT_OUT = REPO / "benchmarks/cross_binding/results/dashboard_refresh_2026_05_26_donor_timing.csv"

# Timing columns the bench JSON supplies; everything else is copied verbatim
# from the fixture row so parity/divergence/reference stay authoritative.
TIMING_FIELDS = (
    "reported_ms", "median_ms", "sample_median_ms", "mean_ms", "min_ms",
    "max_ms", "n_runs", "total_runs", "max_runs", "warmup_ms",
    "warmup_included", "timing_statistic", "timing_decision",
)


def bench_methods(bench_bin: Path) -> set[str]:
    out = subprocess.run([str(bench_bin), "--list"], capture_output=True, text=True, check=True)
    return {line.strip() for line in out.stdout.splitlines() if line.strip()}


def run_cell(bench_bin: Path, method: str, n: int, p: int, runs: int, seed: int) -> dict:
    proc = subprocess.run(
        [str(bench_bin), "--method", method, "--n", str(n), "--p", str(p),
         "--runs", str(runs), "--seed", str(seed)],
        capture_output=True, text=True, timeout=600,
    )
    last = proc.stdout.strip().splitlines()[-1] if proc.stdout.strip() else "{}"
    return json.loads(last)


def cell_key(row: dict) -> tuple:
    return (row.get("algorithm"), row.get("backend"), row.get("libp4a_build", ""),
            str(row.get("n")), str(row.get("p")), str(row.get("threads")))


def load_existing(path: Path) -> dict[tuple, dict]:
    """Previously-timed rows keyed by cell, so a resume run preserves them
    (and never rewrites the output down to only the freshly-timed cells)."""
    if not path.exists():
        return {}
    existing: dict[tuple, dict] = {}
    with path.open(newline="") as f:
        for row in csv.DictReader(f):
            if str(row.get("ok")).lower() == "true" and (row.get("reported_ms") or "").strip():
                existing[cell_key(row)] = row
    return existing


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--bench-bin", type=Path, default=DEFAULT_BENCH)
    ap.add_argument("--fixtures-csv", type=Path, default=DEFAULT_FIXTURES)
    ap.add_argument("--out-csv", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--sizes", nargs="+", default=["250x50", "50x250"],
                    help="dashboard cell sizes as NxP")
    ap.add_argument("--runs", type=int, default=200,
                    help="upper bound on timed reps; the bench tool picks the "
                         "actual count adaptively (more for fast ops, fewer for "
                         "slow), so this is just the cap for ultra-fast ops")
    ap.add_argument("--seed", type=int, default=1_234_567_890)
    ap.add_argument("--resume-existing", action="store_true",
                    help="skip cells already timed in --out-csv")
    ap.add_argument("--force", action="store_true",
                    help="re-time every cell even if already present")
    args = ap.parse_args()

    if not args.bench_bin.exists():
        print(f"bench tool not built at {args.bench_bin}; "
              f"`cmake --preset blas-omp -DN4M_BUILD_BENCH=ON && "
              f"cmake --build --preset blas-omp --target n4m_donor_bench`",
              file=sys.stderr)
        return 1

    timeable = bench_methods(args.bench_bin)
    sizes = [tuple(int(x) for x in s.lower().split("x")) for s in args.sizes]

    with args.fixtures_csv.open(newline="") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames or []
        fixture_rows = list(reader)

    # Fixture-driven: iterate the cpp parity rows (the dashboard's donor
    # cells) and time each via its mapped ABI op. The output row keeps the
    # dashboard `algorithm` name; only the timing columns are filled.
    requested_sizes = {(str(n), str(p)) for n, p in sizes}
    cells: list[tuple[dict, str]] = []  # (fixture row, bench op id)
    skipped_unmapped: set[str] = set()
    for r in fixture_rows:
        if r.get("backend") != "cpp":
            continue
        if (str(r.get("n")), str(r.get("p"))) not in requested_sizes:
            continue
        method = r["algorithm"]
        bench_op = DASHBOARD_TO_BENCH.get(method, method)
        if bench_op not in timeable:
            skipped_unmapped.add(method)
            continue
        cells.append((r, bench_op))

    # Resume: keep previously-timed cells; --force re-times everything.
    out: dict[tuple, dict] = {} if args.force else load_existing(args.out_csv)

    timed = skipped = failed = 0
    for base, bench_op in cells:
        n, p = int(base["n"]), int(base["p"])
        key = cell_key(base)
        if key in out and not args.force:
            skipped += 1
            continue
        rec = run_cell(args.bench_bin, bench_op, n, p, args.runs, args.seed)
        if not rec.get("ok"):
            failed += 1
            print(f"  FAIL {base['algorithm']} ({bench_op}) {n}x{p}: "
                  f"{rec.get('reason', 'unknown')}", file=sys.stderr)
            continue
        row = dict(base)
        for col in TIMING_FIELDS:
            if col in rec:
                row[col] = rec[col]
        out[key] = row
        timed += 1
        print(f"  timed {base['algorithm']:30s} {n}x{p}: {rec['reported_ms']:.4f} ms "
              f"({rec['timing_statistic']}, {rec['n_runs']} runs)")

    if skipped_unmapped:
        print(f"  (no ABI op for {len(skipped_unmapped)} dashboard method(s), "
              f"left parity-only: {', '.join(sorted(skipped_unmapped))})", file=sys.stderr)

    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    with args.out_csv.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(out.values())
    print(f"\nWrote {len(out)} timed cpp rows "
          f"(timed={timed}, skipped={skipped}, failed={failed}) to {args.out_csv}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
