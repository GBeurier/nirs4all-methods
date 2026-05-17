#!/usr/bin/env python3
"""Render the cross-binding timing+parity CSV into a sphinx-compatible
Markdown page (docs/source/benchmarks/cross_binding.md).

Layout: one section per (algo, threads) with a pivot table whose rows
are dataset sizes and columns are backends. Each cell shows
`<median_ms> <parity_icon>` where parity_icon is ✓ / ⚠ / ✗ / ⏳ / —.

Footnote: collected versions per backend.
"""
from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
import platform
from collections import defaultdict
from pathlib import Path

ICON_OK = "✓"
ICON_WARN = "⚠"
ICON_FAIL = "✗"
ICON_TIMEOUT = "⏳"
ICON_NA = "—"


def parity_icon(row) -> str:
    if not _is_true(row.get("ok")):
        reason = (row.get("reason") or "").lower()
        if "timeout" in reason:
            return ICON_TIMEOUT
        return ICON_NA
    if _is_true(row.get("parity_ok")):
        return ICON_OK
    try:
        d = float(row.get("parity_max_diff", "nan"))
    except (TypeError, ValueError):
        return ICON_WARN
    if d != d:  # NaN
        return ICON_WARN
    if d < 1e-3:
        return ICON_WARN
    return ICON_FAIL


def _is_true(v) -> bool:
    return str(v).lower() == "true"


def fmt_ms(ms_str: str) -> str:
    try:
        f = float(ms_str)
    except (TypeError, ValueError):
        return "—"
    if f >= 1000:
        return f"{f / 1000:.1f}s"
    if f >= 10:
        return f"{f:.1f}"
    return f"{f:.2f}"


def render(csv_path: Path, out_path: Path) -> None:
    rows = list(csv.DictReader(open(csv_path)))

    # Header info.
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out = []
    out.append("# Cross-binding benchmark — parity + timing\n")
    out.append("> **Plaquette de pub** : pour chaque algorithme, on prouve que "
                "pls4all produit *exactement* le même résultat que la référence "
                "externe (parity), souvent plus vite (timing).\n")
    out.append(f"_Generated: {dt.datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')} UTC_  ")
    out.append(f"_Host: {platform.platform()}_  ")
    out.append(f"_CSV: `{csv_path.name}` ({len(rows)} cells)_\n")
    out.append("\n## Legend\n")
    out.append(f"- {ICON_OK}  parity OK (max abs diff ≤ tolerance, default 1e-6)")
    out.append(f"- {ICON_WARN}  parity within 1e-3 but > tolerance (algorithmic drift)")
    out.append(f"- {ICON_FAIL}  parity mismatch (typically different preprocessing convention)")
    out.append(f"- {ICON_TIMEOUT}  cell timed out (300s wall-clock)")
    out.append(f"- {ICON_NA}  not implemented / skipped\n")
    out.append("Each cell: `<median_ms> <icon>`. Median is over 5 runs "
                "(warmup discarded).\n")

    # Group by (algorithm, threads).
    groups = defaultdict(list)  # (algo, threads) → list[row]
    for r in rows:
        if not r.get("algorithm"):
            continue
        groups[(r["algorithm"], int(r["threads"]))].append(r)

    # Backends present, in display order = first-seen order.
    seen_backends: list[str] = []
    for r in rows:
        b = r.get("backend")
        if b and b not in seen_backends:
            seen_backends.append(b)

    for (algo, threads), grp in sorted(groups.items()):
        out.append(f"\n## {algo}  —  {threads} thread{'s' if threads > 1 else ''}\n")
        # Pivot rows: sizes; columns: backends.
        sizes = sorted({(int(r["n"]), int(r["p"])) for r in grp})
        backends = [b for b in seen_backends if any(r["backend"] == b for r in grp)]
        header = "| size | " + " | ".join(backends) + " |"
        sep = "|---" + ("|---" * len(backends)) + "|"
        out.append(header)
        out.append(sep)
        for (n, p) in sizes:
            cells = []
            for b in backends:
                match = [r for r in grp if r["backend"] == b
                          and int(r["n"]) == n and int(r["p"]) == p]
                if not match:
                    cells.append(ICON_NA)
                else:
                    r = match[0]
                    cells.append(f"{fmt_ms(r.get('median_ms', ''))} "
                                  f"{parity_icon(r)}")
            out.append(f"| {n}×{p} | " + " | ".join(cells) + " |")
        out.append("")

    # Versions footnote.
    out.append("\n## Backend versions\n")
    versions_seen = {}
    for r in rows:
        b = r.get("backend")
        if not b or b in versions_seen:
            continue
        try:
            v = json.loads(r.get("versions_json", "{}"))
        except Exception:
            v = {}
        if v:
            versions_seen[b] = v
    out.append("| backend | versions |")
    out.append("|---|---|")
    for b in seen_backends:
        v = versions_seen.get(b, {})
        if not v:
            out.append(f"| {b} | — |")
            continue
        items = "; ".join(f"`{k}={v[k]}`" for k in v if v[k])
        out.append(f"| {b} | {items} |")

    out.append("\n## Methodology\n")
    out.append("- Reference: `cpp` cell at 1 thread (libp4a via ctypes), "
                "or `python_tier1` when `cpp` is unavailable for an algorithm")
    out.append("- Parity tolerance: 1e-6 (per-algo overrides possible)")
    out.append("- All backends read the **same** orchestrator-generated CSV "
                "(`benchmarks/cross_binding/data/data_<n>x<p>_seed<seed>.csv`) "
                "so input data is bit-identical across languages")
    out.append("- 5 runs per cell, first discarded as warmup, median reported")
    out.append("- Per-cell timeout: 300 s")
    out.append("- Thread control via `OMP_NUM_THREADS = OPENBLAS_NUM_THREADS = "
                "MKL_NUM_THREADS = BLIS_NUM_THREADS` set in the subprocess env, "
                "plus `Context.num_threads` for Python pls4all and "
                "`maxNumCompThreads()` for Octave")
    out.append("- pls4all libp4a build: `build/blas-omp/cpp/src/libp4a.so` "
                "(BLAS + OpenMP enabled)\n")

    out_path.write_text("\n".join(out))
    print(f"Rendered → {out_path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True, help="full_matrix.csv path")
    ap.add_argument("--out", required=True,
                     help="Output markdown path (e.g. docs/source/benchmarks/cross_binding.md)")
    args = ap.parse_args()
    render(Path(args.csv), Path(args.out))


if __name__ == "__main__":
    main()
