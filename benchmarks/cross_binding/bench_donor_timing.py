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
# C++ build tiers → the n4m_donor_bench binary linked against that libn4m
# build. Each tier becomes its own dashboard cpp column (build_landing maps
# dev-release→native, blas-on→blas, omp-on→omp, blas-omp→blas+omp). Only the
# tiers whose binary is actually built get timed.
CPP_BUILDS = ["dev-release", "blas-on", "omp-on", "blas-omp"]


def bench_bin_for(build: str) -> Path:
    return REPO / f"build/{build}/bench/cpp/n4m_donor_bench"


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
            done = (str(row.get("ok")).lower() == "true"
                    and ((row.get("reported_ms") or "").strip()
                         or row.get("timing_decision") == "build_insensitive"))
            if done:
                existing[cell_key(row)] = row
    return existing


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--builds", nargs="+", default=CPP_BUILDS,
                    help="cpp build tiers to time (each becomes a cpp column); "
                         "tiers whose n4m_donor_bench binary is missing are skipped")
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

    # Discover which build tiers are actually built.
    builds = [(b, bench_bin_for(b)) for b in args.builds if bench_bin_for(b).exists()]
    if not builds:
        print(f"no n4m_donor_bench binary found for any of {args.builds}; build with "
              f"`cmake --preset <tier> -DN4M_BUILD_BENCH=ON && "
              f"cmake --build --preset <tier> --target n4m_donor_bench`", file=sys.stderr)
        return 1
    print(f"timing cpp build tiers: {', '.join(b for b, _ in builds)}", file=sys.stderr)

    timeable = bench_methods(builds[0][1])  # op list is identical across builds
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

    # Phase-17 augmenters are rolled into the fixture's single `aug_phase17`
    # cpp row; give each its individual cpp timing by cloning that base row
    # (inheriting its correctness verdict). donor_ops is the source of truth
    # for which ops are on-dashboard but absent from the fixture.
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import donor_ops as _D
    fixture_cpp_algos = {r["algorithm"] for r in fixture_rows if r.get("backend") == "cpp"}
    phase17_base = {(str(r["n"]), str(r["p"])): r for r in fixture_rows
                    if r.get("backend") == "cpp" and r["algorithm"] == "aug_phase17"}
    for spec in _D.all_specs():
        if not spec.on_dashboard or spec.dashboard_id in fixture_cpp_algos:
            continue
        bench_op = DASHBOARD_TO_BENCH.get(spec.dashboard_id, spec.dashboard_id)
        if bench_op not in timeable:
            skipped_unmapped.add(spec.dashboard_id)
            continue
        for sn, sp in requested_sizes:
            base = phase17_base.get((sn, sp))
            if base is None:
                continue
            row = dict(base)
            row["algorithm"] = spec.dashboard_id
            row["reference_parity_note"] = "inherited from aug_phase17 fixture"
            cells.append((row, bench_op))

    # Resume: keep previously-timed cells; --force re-times everything.
    out: dict[tuple, dict] = {} if args.force else load_existing(args.out_csv)

    # Sentinel: an elementwise op (no BLAS/OMP code path) runs identically
    # across tiers, so timing blas-only / omp-only is redundant. We always
    # time the `native` baseline and the `blas+omp` production build; if they
    # match within noise the op is build-insensitive and the blas-only /
    # omp-only cells get a "≡ native" sentinel (timing_decision=
    # build_insensitive, no ms) instead of a repeated number. Anchor tiers
    # always get a real timing; only the two middle tiers can be sentineled.
    INSENSITIVE_REL = 0.20  # |blas+omp − native| / native below this ⇒ no effect
    builds_map = dict(builds)
    anchors = [b for b in ("dev-release", "blas-omp") if b in builds_map]
    middles = [b for b in ("blas-on", "omp-on") if b in builds_map]

    def emit(build, base, n, p, key, rec):
        row = dict(base); row["libp4a_build"] = build
        for col in TIMING_FIELDS:
            if col in rec:
                row[col] = rec[col]
        out[key] = row

    def sentinel(build, base):
        row = dict(base); row["libp4a_build"] = build
        for col in TIMING_FIELDS:
            row[col] = ""
        row["ok"] = "True"
        row["timing_decision"] = "build_insensitive"
        out[cell_key(row)] = row

    timed = skipped = failed = sentineled = 0
    for base, bench_op in cells:
        n, p = int(base["n"]), int(base["p"])
        recs: dict[str, dict] = {}
        for build in anchors:
            row = dict(base); row["libp4a_build"] = build
            key = cell_key(row)
            if key in out and not args.force:
                skipped += 1; recs[build] = out[key]; continue
            rec = run_cell(builds_map[build], bench_op, n, p, args.runs, args.seed)
            if not rec.get("ok"):
                failed += 1
                print(f"  FAIL {base['algorithm']} ({bench_op}) {n}x{p} [{build}]: "
                      f"{rec.get('reason', 'unknown')}", file=sys.stderr)
                continue
            emit(build, base, n, p, key, rec); recs[build] = rec; timed += 1
        # decide build-sensitivity from the two anchors
        nat = recs.get("dev-release", {}).get("reported_ms")
        bo = recs.get("blas-omp", {}).get("reported_ms")
        try:
            insensitive = (nat and bo and abs(float(bo) - float(nat)) / float(nat) < INSENSITIVE_REL)
        except (TypeError, ValueError):
            insensitive = False
        for build in middles:
            row = dict(base); row["libp4a_build"] = build
            key = cell_key(row)
            if key in out and not args.force:
                skipped += 1; continue
            if insensitive:
                sentinel(build, base); sentineled += 1
            else:
                rec = run_cell(builds_map[build], bench_op, n, p, args.runs, args.seed)
                if not rec.get("ok"):
                    failed += 1; continue
                emit(build, base, n, p, key, rec); timed += 1
        print(f"  {base['algorithm']:28s} {n}x{p}: native={nat} blas+omp={bo}"
              f"{' (blas/omp ≡ native)' if insensitive else ''}")

    if skipped_unmapped:
        print(f"  (no ABI op for {len(skipped_unmapped)} dashboard method(s), "
              f"left parity-only: {', '.join(sorted(skipped_unmapped))})", file=sys.stderr)

    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    with args.out_csv.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(out.values())
    print(f"\nWrote {len(out)} cpp rows "
          f"(timed={timed}, sentinel={sentineled}, skipped={skipped}, "
          f"failed={failed}) to {args.out_csv}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
