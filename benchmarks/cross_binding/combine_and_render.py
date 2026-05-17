#!/usr/bin/env python3
"""Merge multiple cross_binding CSVs and re-render the docs page.

Each input CSV has the same schema (orchestrator output). We just
concatenate the rows (dedup headers), feed the combined CSV to
render_docs.render(), and clean up the temporary merge file.
"""
from __future__ import annotations

import argparse
import csv
import sys
import tempfile
from pathlib import Path

import render_docs  # same directory


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csvs", nargs="+", required=True,
                     help="Input CSV files (in display order)")
    ap.add_argument("--out", required=True,
                     help="Output markdown path")
    args = ap.parse_args()

    # Merge: keep header from the first, append rows from the rest.
    rows = []
    header = None
    for csv_path in args.csvs:
        p = Path(csv_path)
        if not p.exists():
            print(f"WARN: missing {p}, skipping", file=sys.stderr)
            continue
        with p.open() as f:
            r = csv.reader(f)
            h = next(r)
            if header is None:
                header = h
            for row in r:
                rows.append(row)

    if not rows:
        sys.exit("No rows merged — nothing to render.")

    # Write the merged CSV to a tempfile and feed render_docs.
    with tempfile.NamedTemporaryFile(
            "w", newline="", suffix=".csv", delete=False) as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(rows)
        merged = Path(f.name)

    try:
        out = Path(args.out)
        # Main page: 1 thread only (single-thread is the clean,
        # apples-to-apples comparison; externals are mostly single-
        # threaded anyway).
        render_docs.render(
            merged, out, only_threads=[1],
            page_title="Cross-binding benchmark — parity + timing (1 thread)",
            page_intro=(
                "Single-thread numbers — the cleanest cross-language "
                "comparison. **Most external libraries don't honour "
                "OMP_NUM_THREADS at the algorithm level** (sklearn, "
                "pls::plsr, plsregress, ikpls all run the PLS algo "
                "serially even when BLAS is multi-threaded), so a "
                "multi-thread comparison would compare pls4all's "
                "OpenMP path against everyone else's single-threaded "
                "loop. That's a real, useful number — see the "
                "[thread sweep page](cross_binding_threads.md) — but "
                "this main page sticks to 1 thread for honest "
                "apples-to-apples timing."
            ),
        )
        print(f"Main page → {out}")

        # Secondary page: thread sweep (1 / 3 / 10) across pls4all
        # backends.
        threads_out = out.with_name("cross_binding_threads.md")
        render_docs.render(
            merged, threads_out,
            only_threads=None,  # all
            page_title="Cross-binding benchmark — thread sweep",
            page_intro=(
                "Same matrix as the [main page](cross_binding.md), "
                "but with thread counts 1, 3 and 10 shown in separate "
                "per-(algorithm, thread) sections. **External "
                "libraries (`sklearn`, `pls`, `ropls`, `mixOmics`, "
                "`plsregress`, `ikpls`) typically don't accelerate "
                "their inner loops with thread count** — only their "
                "linked BLAS does, and that helps only when the algo "
                "is GEMM-bound. pls4all bindings use OpenMP at the "
                "C kernel level on top of the BLAS, so multi-thread "
                "wins are visible here."
            ),
        )
        print(f"Threads page → {threads_out}")
        print(f"Merged {len(rows)} rows from {len(args.csvs)} CSVs")
    finally:
        merged.unlink(missing_ok=True)


if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    main()
