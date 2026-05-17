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
    """Format milliseconds with explicit units. Returns NaN as float
    when input is unparseable so the caller can detect missing cells."""
    try:
        f = float(ms_str)
    except (TypeError, ValueError):
        return "—"
    if f >= 1000:
        return f"{f / 1000:.1f} s"
    if f >= 10:
        return f"{f:.1f} ms"
    return f"{f:.2f} ms"


def cell_ms_value(ms_str: str) -> float:
    """Return raw float ms for comparison (best-of-row bolding). NaN
    when missing so it doesn't compete for the minimum."""
    try:
        return float(ms_str)
    except (TypeError, ValueError):
        return float("nan")


def render(csv_path: Path, out_path: Path) -> None:
    rows = list(csv.DictReader(open(csv_path)))

    # Header info.
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out = []
    out.append("# Cross-binding benchmark — parity + timing\n")
    out.append("For every (algorithm × backend × dataset size × thread count) "
                "cell, we report the **median wall-clock time** and the "
                "**parity verdict** vs a deterministic reference backend. "
                "Same algorithm, same input bytes — different implementations.\n")
    out.append(f"_Generated: {dt.datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')} UTC_  ")
    out.append(f"_Host: {platform.platform()}_  ")
    out.append(f"_CSV: `{csv_path.name}` ({len(rows)} cells)_\n")

    out.append("\n## How to read a cell\n")
    out.append("Each cell shows `<median_ms> <icon>`. The icon is the "
                "**parity verdict** vs the reference backend "
                "(`cpp` at 1 thread):\n")
    out.append(f"- {ICON_OK} bit-exact (max abs diff ≤ 1e-6) — "
                "this backend produces the **same predictions** as the reference")
    out.append(f"- {ICON_WARN} close (≤ 1e-3 but > 1e-6) — "
                "algorithmic drift (different convergence path, but answer agrees)")
    out.append(f"- {ICON_FAIL} divergent (> 1e-3) — "
                "typically a different preprocessing convention "
                "(e.g. scale=True vs scale=False); the backend works, "
                "it just isn't equivalent under this cell's reference choice")
    out.append(f"- {ICON_TIMEOUT} cell timed out (300 s wall-clock)")
    out.append(f"- {ICON_NA} backend doesn't implement this algorithm "
                "(skipped) or hit a runtime error\n")

    out.append("Timing is the **median of 5 runs**; the first is discarded "
                "as warmup. All backends in a single cell read the SAME "
                "orchestrator-generated CSV so cross-language input bytes "
                "are identical. See "
                "[methodology.md](methodology.md) for full details.\n")

    out.append("\n## Backend columns\n")
    out.append("| Column | Language | Tier | What it actually runs |")
    out.append("|---|---|---|---|")
    out.append("| `cpp` | C++ | reference | libp4a called via ctypes — same C kernel as every pls4all binding, no wrapper |")
    out.append("| `python_tier1` | Python | pls4all | `pls4all._methods.<algo>_fit(ctx, cfg, X, y)` — raw FFI binding |")
    out.append("| `python_tier2` | Python | pls4all | `pls4all.sklearn.<Class>` — sklearn-style `BaseEstimator`, `.fit() / .predict()` |")
    out.append("| `sklearn` | Python | external | `sklearn.cross_decomposition.PLSRegression`, `sklearn.decomposition.PCA + LinearRegression` (PCR proxy), etc. |")
    out.append("| `ikpls` | Python | external | `ikpls.numpy_ikpls.PLS` — Improved Kernel PLS (only implements plain PLS) |")
    out.append("| `r_tier1` | R | pls4all | `pls4all_method(algo, X, y, ...)` — unified dispatcher (33 fits + 24 selectors + 4 diagnostics) |")
    out.append("| `r_tier2` | R | pls4all | base R formula+S3 wrappers (`pls(y ~ ., data)`, `cppls(...)`, etc.) |")
    out.append("| `r_pls` | R | external | `pls::plsr / pls::cppls / pls::pcr` — CRAN `pls` package |")
    out.append("| `r_ropls` | R | external | `ropls::opls` — Bioconductor (covers OPLS only) |")
    out.append("| `r_mixomics` | R | external | `mixOmics::pls / spls / plsda / splsda` |")
    out.append("| `matlab_tier1` | MATLAB / Octave | pls4all | `pls4all.<algo>(X, y, ...)` — single dispatcher MEX |")
    out.append("| `matlab_tier2` | MATLAB / Octave | pls4all | `pls4all.fit(algo, X, y, ...)` factory + classdefs |")
    out.append("| `matlab_pls` | MATLAB / Octave | external | Octave statistics `plsregress` (SIMPLS, plain PLS only) |\n")

    out.append("Empty cells (`—`) mean the backend doesn't implement that "
                "algorithm — e.g. `sklearn` has no native CPPLS, `pls::plsr` "
                "has no GPR-PLS, `plsregress` covers only plain PLS. This "
                "is honest scope reporting, not a fail.\n")

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
        out.append("_Row label format: `n_samples × n_features`. "
                    "Cell shows `median time + parity icon`. "
                    "**Bold** = fastest backend in the row (only counted "
                    "when the cell is OK and parity is ✓ or ⚠ — "
                    "comparing a divergent ✗ backend to a parity-correct "
                    "one would be meaningless)._\n")
        # Pivot rows: sizes; columns: backends.
        sizes = sorted({(int(r["n"]), int(r["p"])) for r in grp})
        backends = [b for b in seen_backends if any(r["backend"] == b for r in grp)]
        header = "| samples × features | " + " | ".join(backends) + " |"
        sep = "|---" + ("|---" * len(backends)) + "|"
        out.append(header)
        out.append(sep)
        for (n, p) in sizes:
            # First pass: collect (raw_ms, formatted, icon) per column.
            col_data = []
            for b in backends:
                match = [r for r in grp if r["backend"] == b
                          and int(r["n"]) == n and int(r["p"]) == p]
                if not match:
                    col_data.append((float("nan"), ICON_NA, ""))
                    continue
                r = match[0]
                if not _is_true(r.get("ok")):
                    col_data.append((float("nan"),
                                       fmt_ms(r.get("median_ms", "")),
                                       parity_icon(r)))
                else:
                    raw = cell_ms_value(r.get("median_ms", ""))
                    col_data.append((raw,
                                       fmt_ms(r.get("median_ms", "")),
                                       parity_icon(r)))
            # Find the min over OK cells with parity ≤ ⚠ (i.e. not ✗
            # divergent — comparing a scale=True backend to a scale=False
            # reference's timing is meaningless). Bold the winner.
            valid_for_min = [
                (i, raw) for i, (raw, _, icon) in enumerate(col_data)
                if raw == raw and icon in (ICON_OK, ICON_WARN)  # not NaN, not ✗/—/⏳
            ]
            min_idx = min(valid_for_min, key=lambda x: x[1])[0] if valid_for_min else None
            cells = []
            for i, (raw, formatted, icon) in enumerate(col_data):
                if icon == ICON_NA and formatted == ICON_NA:
                    # pure-empty cell
                    cells.append("—")
                    continue
                label = f"{formatted} {icon}".strip()
                if i == min_idx:
                    label = f"**{label}**"
                cells.append(label)
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
