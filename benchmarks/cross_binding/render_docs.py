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
import os
import platform
import re
import subprocess
from collections import defaultdict
from pathlib import Path

ICON_OK = "✓"
ICON_WARN = "⚠"
ICON_FAIL = "✗"
ICON_TIMEOUT = "⏳"
ICON_NA = "—"

# Map orchestrator backend names → "owner.language[.variant]" display
# labels. `pls4all.*` are us, everything else is an external library
# (with the language as the prefix). The detailed definition for each
# column lives at the bottom of the rendered doc (Backend definitions
# section).
# pls4all bindings keep the `pls4all.` prefix; external libraries get
# their actual package name verbatim (no `R.` / `python.` prefix) so
# the column header reads like the real import path you'd type.
BACKEND_DISPLAY: dict[str, str] = {
    # pls4all (us)
    "cpp":          "pls4all.cpp",
    "python_tier1": "pls4all.python",
    "python_tier2": "pls4all.sklearn",
    "r_tier1":      "pls4all.R",
    "r_tier2":      "pls4all.R.formula",
    "matlab_tier1": "pls4all.matlab",
    "matlab_tier2": "pls4all.matlab.classdef",
    # external libraries — their real package name
    "sklearn":      "sklearn",
    "ikpls":        "ikpls",
    "r_pls":        "pls",         # CRAN `pls` package
    "r_ropls":      "ropls",       # Bioconductor
    "r_mixomics":   "mixOmics",    # Bioconductor
    "matlab_pls":   "plsregress",  # Octave statistics
}

# Per-backend long description for the "Backend definitions" section.
BACKEND_LONG: dict[str, tuple[str, str, str]] = {
    # name → (Language, Tier, What it runs)
    "cpp":          ("C++",          "pls4all reference", "libp4a called via ctypes — same C kernel as every pls4all binding, no high-level wrapper"),
    "python_tier1": ("Python",       "pls4all raw",       "`pls4all._methods.<algo>_fit(ctx, cfg, X, y, …)` — direct FFI binding"),
    "python_tier2": ("Python",       "pls4all idiomatic", "`pls4all.sklearn.<Class>` — sklearn-style BaseEstimator with `.fit() / .predict()`"),
    "sklearn":      ("Python",       "external",          "`sklearn.cross_decomposition.PLSRegression`, `sklearn.decomposition.PCA + LinearRegression / Ridge / GaussianProcessRegressor` (proxies)"),
    "ikpls":        ("Python",       "external",          "`ikpls.numpy_ikpls.PLS` — Improved Kernel PLS (covers plain PLS only)"),
    "r_tier1":      ("R",            "pls4all raw",       "`pls4all_method(algo, X, y, ...)` — unified dispatcher (33 fits + 24 selectors + 4 diagnostics)"),
    "r_tier2":      ("R",            "pls4all idiomatic", "`pls(y ~ ., data)`, `cppls(...)`, `sparse_pls(...)`, … — base R formula+S3 wrappers"),
    "r_pls":        ("R",            "external",          "CRAN `pls` package — `pls::plsr / pls::cppls / pls::pcr`"),
    "r_ropls":      ("R",            "external",          "Bioconductor `ropls` — `ropls::opls` (covers OPLS only)"),
    "r_mixomics":   ("R",            "external",          "Bioconductor `mixOmics` — `pls / spls / plsda / splsda`"),
    "matlab_tier1": ("MATLAB/Octave","pls4all raw",       "`pls4all.<algo>(X, y, ...)` — single dispatcher MEX"),
    "matlab_tier2": ("MATLAB/Octave","pls4all idiomatic", "`pls4all.fit(algo, X, y, ...)` factory + per-algorithm classdefs"),
    "matlab_pls":   ("MATLAB/Octave","external",          "Octave statistics `plsregress` (SIMPLS, plain PLS only)"),
}


def disp(b: str) -> str:
    return BACKEND_DISPLAY.get(b, b)


def host_cpu_info() -> dict:
    """Best-effort CPU / OS info for the doc header. Reads /proc/cpuinfo
    where available (Linux), falls back to platform.* otherwise."""
    info = {
        "os": f"{platform.system()} {platform.release()} ({platform.machine()})",
        "python": platform.python_version(),
        "cpu_model": platform.processor() or "unknown",
        "cpu_count_logical": os.cpu_count() or "?",
        "cpu_count_physical": "?",
        "ram_gb": "?",
    }
    cpuinfo_path = Path("/proc/cpuinfo")
    if cpuinfo_path.exists():
        try:
            text = cpuinfo_path.read_text()
            # model name (first occurrence)
            m = re.search(r"^model name\s*:\s*(.+)$", text, re.M)
            if m:
                info["cpu_model"] = m.group(1).strip()
            # physical core count: unique (physical id, core id) pairs
            pairs = set(re.findall(r"physical id\s*:\s*(\d+).*?core id\s*:\s*(\d+)",
                                     text, re.S))
            if pairs:
                info["cpu_count_physical"] = len(pairs)
            # CPU MHz (first)
            mhz = re.search(r"^cpu MHz\s*:\s*([\d.]+)$", text, re.M)
            if mhz:
                info["cpu_mhz"] = f"{float(mhz.group(1)):.0f}"
            # cache size (first)
            cache = re.search(r"^cache size\s*:\s*(.+)$", text, re.M)
            if cache:
                info["cpu_cache"] = cache.group(1).strip()
        except Exception:
            pass
    # /proc/meminfo for RAM
    meminfo_path = Path("/proc/meminfo")
    if meminfo_path.exists():
        try:
            m = re.search(r"^MemTotal:\s*(\d+)\s*kB",
                            meminfo_path.read_text(), re.M)
            if m:
                info["ram_gb"] = f"{int(m.group(1)) / 1024 / 1024:.1f}"
        except Exception:
            pass
    # uname -r for kernel
    try:
        info["kernel"] = subprocess.check_output(
            ["uname", "-r"], timeout=2).decode().strip()
    except Exception:
        info["kernel"] = platform.release()
    return info


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
    info = host_cpu_info()
    out.append(f"_Generated: {dt.datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')} UTC_  ")
    out.append(f"_CSV: `{csv_path.name}` ({len(rows)} cells)_\n")
    out.append("\n## Host\n")
    out.append("| | |")
    out.append("|---|---|")
    out.append(f"| **CPU**            | {info['cpu_model']} |")
    out.append(f"| **Cores**          | {info['cpu_count_physical']} physical / {info['cpu_count_logical']} logical |")
    if info.get("cpu_mhz"):
        out.append(f"| **Frequency**  | {info['cpu_mhz']} MHz nominal |")
    if info.get("cpu_cache"):
        out.append(f"| **L3 cache**   | {info['cpu_cache']} |")
    out.append(f"| **RAM**            | {info['ram_gb']} GB |")
    out.append(f"| **OS / kernel**    | {info['os']} · kernel {info['kernel']} |")
    out.append(f"| **Python**         | {info['python']} |\n")

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

    out.append("\nColumn names: anything starting with `pls4all.` is "
                "one of this project's bindings; the others use their "
                "real package name (`sklearn`, `ikpls`, `pls`, `ropls`, "
                "`mixOmics`, `plsregress`). **Every algorithm table "
                "keeps every column** — `—` cells are intentional and "
                "show *where coverage is still missing*, either on "
                "our side (an algorithm not yet wrapped in a tier-2 "
                "class) or on the external side (e.g. `sklearn` doesn't "
                "implement CPPLS / OPLS, `plsregress` only does plain "
                "PLS, `ikpls` only does plain PLS). Full per-column "
                "description: see [Backend definitions]"
                "(#backend-definitions) at the bottom of this page.\n")

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
        # Pivot rows: sizes; columns: ALL backends (whether they cover
        # this algo or not). Empty `—` cells are intentional: they show
        # the user where coverage is still missing (the "chemin à
        # parcourir" — what we / external libs would need to add to
        # make this matrix dense).
        sizes = sorted({(int(r["n"]), int(r["p"])) for r in grp})
        backends = list(seen_backends)
        header = "| samples × features | " + " | ".join(disp(b) for b in backends) + " |"
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

    # Backend definitions + versions footnote.
    out.append("\n## Backend definitions\n")
    out.append("Each column in the per-algorithm tables above is one of "
                "the entries below. Columns are named `owner.language"
                "[.variant]`: `pls4all.*` are this project's own "
                "bindings; everything else is an external library "
                "shipped by its own maintainers.\n")
    out.append("| Column | Language | Tier | What it actually runs |")
    out.append("|---|---|---|---|")
    for b in seen_backends:
        if b not in BACKEND_LONG:
            continue
        lang, tier, what = BACKEND_LONG[b]
        out.append(f"| `{disp(b)}` | {lang} | {tier} | {what} |")
    out.append("")

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
    out.append("\n## Versions per backend\n")
    out.append("| Column | Versions |")
    out.append("|---|---|")
    for b in seen_backends:
        v = versions_seen.get(b, {})
        if not v:
            out.append(f"| `{disp(b)}` | — |")
            continue
        items = "; ".join(f"`{k}={v[k]}`" for k in v if v[k])
        out.append(f"| `{disp(b)}` | {items} |")

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
