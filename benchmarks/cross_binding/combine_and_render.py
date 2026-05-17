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
        render_docs.render(merged, Path(args.out))
        print(f"Merged {len(rows)} rows from {len(args.csvs)} CSVs "
               f"→ {args.out}")
    finally:
        merged.unlink(missing_ok=True)


if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    main()
