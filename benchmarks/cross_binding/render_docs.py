#!/usr/bin/env python3
"""Render the cross-binding timing+parity CSV into a sphinx-compatible
Markdown page (docs/source/benchmarks/cross_binding.md).

Layout: one section per (algo, threads) with a pivot table whose rows
are dataset sizes and columns are backends. Each cell shows
`<reported_ms> <gate_icon>` where the gate is reference parity for
C++/external columns and binding parity for internal bindings.

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
import sys
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
    "registry_pls4all": "pls4all.registry",
    "cpp":          "pls4all.cpp",
    "python_tier1": "pls4all.python",
    "python_tier2": "pls4all.sklearn",
    "r_tier1":      "pls4all.R",
    "r_tier2":      "pls4all.R.formula",
    "r_pls_compat": "pls4all.R.pls_compat",
    "r_mdatools_compat": "pls4all.R.mdatools_compat",
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
    "registry_pls4all": ("Python",       "pls4all canonical", "`benchmarks.parity_timing.registry.MethodSpec.pls4all_fn` — the canonical per-method pls4all entry point"),
    "cpp":          ("C++",          "pls4all native", "libp4a called via ctypes — same C kernel as every pls4all binding, no high-level wrapper"),
    "python_tier1": ("Python",       "pls4all raw",       "`pls4all._methods.<algo>_fit(ctx, cfg, X, y, …)` — direct FFI binding"),
    "python_tier2": ("Python",       "pls4all idiomatic", "`pls4all.sklearn.<Class>` — sklearn-style BaseEstimator with `.fit() / .predict()`"),
    "sklearn":      ("Python",       "external",          "`sklearn.cross_decomposition.PLSRegression`, `sklearn.decomposition.PCA + LinearRegression / Ridge / GaussianProcessRegressor` (proxies)"),
    "ikpls":        ("Python",       "external",          "`ikpls.numpy_ikpls.PLS` — Improved Kernel PLS (covers plain PLS only)"),
    "r_tier1":      ("R",            "pls4all raw",       "`pls4all_method(algo, X, y, ...)` — unified dispatcher (33 fits + 24 selectors + 4 diagnostics)"),
    "r_tier2":      ("R",            "pls4all idiomatic", "`pls(y ~ ., data)`, `cppls(...)`, `sparse_pls(...)`, … — base R formula+S3 wrappers"),
    "r_pls_compat": ("R",            "pls4all pls-compatible", "`plsr()` / `pcr()` for PLS/PCR/CPPLS, formula-built dispatcher path for the rest of the method matrix"),
    "r_mdatools_compat": ("R",       "pls4all mdatools-compatible", "`pls(x, y, ...)` for PLS/PCR/CPPLS, matrix dispatcher path for the rest of the method matrix"),
    "r_pls":        ("R",            "external",          "CRAN `pls` package — `pls::plsr / pls::cppls / pls::pcr`"),
    "r_ropls":      ("R",            "external",          "Bioconductor `ropls` — `ropls::opls` (covers OPLS only)"),
    "r_mixomics":   ("R",            "external",          "Bioconductor `mixOmics` — `pls / spls / plsda / splsda`"),
    "matlab_tier1": ("MATLAB/Octave","pls4all raw",       "`pls4all.<algo>(X, y, ...)` — single dispatcher MEX"),
    "matlab_tier2": ("MATLAB/Octave","pls4all idiomatic", "`pls4all.fit(algo, X, y, ...)` factory + per-algorithm classdefs"),
    "matlab_pls":   ("MATLAB/Octave","external",          "Octave statistics `plsregress` (SIMPLS, plain PLS only)"),
}

REF_DISPLAY_OVERRIDE = {
    "ref_matlab_libpls": "libPLS",
}


def disp(b: str, build: str = "blas-omp") -> str:
    """Display label for a (backend, libp4a_build) pair. For the cpp
    backend the libp4a build is part of the column identity so we can
    surface the BLAS / OpenMP / CUDA acceleration tiers side-by-side."""
    if b == "cpp":
        # Map libp4a build → suffix that says what's enabled.
        suffix = {
            "dev-release": "native",    # no BLAS, no OMP
            "blas-on":     "blas",
            "omp-on":      "omp",
            "blas-omp":    "blas+omp",
            "cuda-on":     "cuda",
        }.get(build, build)
        return f"pls4all.cpp.{suffix}"
    if b in REF_DISPLAY_OVERRIDE:
        return REF_DISPLAY_OVERRIDE[b]
    if b.startswith("ref_"):
        return "ref." + b[len("ref_"):]
    return BACKEND_DISPLAY.get(b, b)


def column_key(row: dict) -> str:
    """Canonical column identity. cpp splits per-libp4a-build; everything
    else collapses to the bare backend name."""
    b = row.get("backend", "")
    if b == "cpp":
        return f"cpp::{row.get('libp4a_build', 'blas-omp')}"
    return b


def column_disp(key: str) -> str:
    """Reverse of column_key for display."""
    if key.startswith("cpp::"):
        return disp("cpp", key[len("cpp::"):])
    return disp(key)


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
    """Legacy single-icon (binding consistency only). Used by older
    pages that haven't migrated to the two-icon dual-gate layout."""
    if not _is_true(row.get("ok")):
        reason = (row.get("reason") or "").lower()
        if "timeout" in reason:
            return ICON_TIMEOUT
        return ICON_NA
    # Prefer the explicit binding_parity_ok column; fall back to the
    # legacy alias parity_ok for old CSVs.
    parity_ok = row.get("binding_parity_ok", row.get("parity_ok"))
    if _is_true(parity_ok):
        return ICON_OK
    diff_val = row.get("binding_parity_max_diff",
                        row.get("parity_max_diff", "nan"))
    try:
        d = float(diff_val)
    except (TypeError, ValueError):
        return ICON_WARN
    if d != d:  # NaN
        return ICON_WARN
    if d < 1e-3:
        return ICON_WARN
    return ICON_FAIL


def binding_parity_icon(row) -> str:
    """Gate 1: did this binding produce the same numbers as cpp?

    Same logic as `parity_icon` (it's the binding-consistency gate),
    kept under an explicit name so the two-gate renderer reads cleanly.
    """
    return parity_icon(row)


def reference_parity_icon(row) -> str:
    """Gate 2: does our C kernel match the canonical external reference?

    Returns:
      ICON_OK      — within `method.rmse_rel_tol`.
      ICON_WARN    — finite RMSE near tolerance, or NaN with shape info.
      ICON_FAIL    — RMSE well above tolerance.
      ICON_NA / "—" — the cell didn't run, or the method is paper_only
                      (no executable reference exists).
      ICON_TIMEOUT — backend timed out (carried over from `ok=False`).
    """
    if not _is_true(row.get("ok")):
        reason = (row.get("reason") or "").lower()
        if "timeout" in reason:
            return ICON_TIMEOUT
        return ICON_NA
    if (row.get("reference_kind") or "").lower() == "paper_only":
        return ICON_NA
    ref_ok = row.get("reference_parity_ok")
    if ref_ok in (None, "", "None"):
        return ICON_NA
    if _is_true(ref_ok):
        return ICON_OK
    rel = row.get("reference_parity_rmse_rel", "nan")
    try:
        d = float(rel)
    except (TypeError, ValueError):
        return ICON_WARN
    if d != d:  # NaN
        return ICON_WARN
    tol = reference_tolerance(row)
    if tol is None:
        return ICON_WARN
    # within 10× the per-method tolerance → WARN (close but no cigar)
    return ICON_WARN if d < 10 * tol else ICON_FAIL


def uses_reference_gate(row) -> bool:
    """Whether this row's visible/static verdict is reference parity."""
    if row.get("backend") == "cpp":
        return True
    return not (row.get("kind") or "").lower().startswith("pls4all")


def dual_parity_label(row) -> str:
    """Compact one-icon label for the visible/static gate.

    C++ and external libraries display reference parity. Internal
    pls4all bindings display binding parity.
    """
    if uses_reference_gate(row):
        return reference_parity_icon(row)
    return binding_parity_icon(row)


def effective_parity_icon(row) -> str:
    """Verdict to use for filtering/bolding a timing in static tables.

    C++ and external libraries are judged by gate 2 (reference parity).
    Internal bindings are judged by gate 1 (binding parity), so only the
    meaningful gate drives bolding/filtering.
    """
    return (
        reference_parity_icon(row)
        if uses_reference_gate(row)
        else binding_parity_icon(row)
    )


def _is_true(v) -> bool:
    return str(v).lower() == "true"


def _float_or_none(v) -> float | None:
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    if f != f:
        return None
    return f


_METHOD_TOLERANCES: dict[str, float] | None = None


def _method_tolerances() -> dict[str, float]:
    global _METHOD_TOLERANCES
    if _METHOD_TOLERANCES is not None:
        return _METHOD_TOLERANCES

    root = Path(__file__).resolve().parents[2]
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    try:
        from benchmarks.parity_timing.registry import METHODS
        _METHOD_TOLERANCES = {
            m.name: float(m.rmse_rel_tol) for m in METHODS
        }
        return _METHOD_TOLERANCES
    except Exception:
        pass

    lockfile = root / "benchmarks/parity_timing/truth_sources.lock.json"
    try:
        data = json.loads(lockfile.read_text())
        _METHOD_TOLERANCES = {
            e["name"]: float(e["rmse_rel_tol"])
            for e in data.get("methods", [])
            if "name" in e and "rmse_rel_tol" in e
        }
    except Exception:
        _METHOD_TOLERANCES = {}
    return _METHOD_TOLERANCES


def reference_tolerance(row: dict) -> float | None:
    for key in (
        "reference_parity_tolerance",
        "reference_parity_tol",
        "rmse_rel_tol",
        "tolerance",
    ):
        f = _float_or_none(row.get(key))
        if f is not None and f > 0:
            return f
    return _method_tolerances().get(row.get("algorithm") or row.get("algo"))


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


def render(csv_path: Path, out_path: Path,
            only_threads: list[int] | None = None,
            page_title: str = "Cross-binding benchmark — parity + timing",
            page_intro: str | None = None) -> None:
    """Render the matrix as a Markdown page.

    `only_threads`: when given, keep only rows whose `threads` value is
    in this list. Used to split the main page (1 thread only) from a
    secondary thread-sweep page (1, 3, 10 threads visible together).
    """
    rows = list(csv.DictReader(open(csv_path)))
    if only_threads is not None:
        keep = set(only_threads)
        rows = [r for r in rows if int(r.get("threads", "0")) in keep]
    run_counts = sorted({int(float(r.get("total_runs") or r.get("n_runs") or 0))
                         for r in rows if r.get("total_runs") or r.get("n_runs")})
    if not run_counts:
        run_text = "the recorded number of timed runs"
    elif len(run_counts) == 1:
        n_runs = run_counts[0]
        run_text = f"{n_runs} timed run" + ("" if n_runs == 1 else "s")
    else:
        run_text = " or ".join(
            [", ".join(str(x) for x in run_counts[:-1]), str(run_counts[-1])]
        )
        run_text = f"{run_text} timed runs"

    # Header info.
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out = []
    out.append(f"# {page_title}\n")
    if page_intro is None:
        page_intro = (
            "For every (algorithm × backend × dataset size × thread "
            "count) cell, we report the **adaptive wall-clock time** "
            "and one visible **parity verdict**. C++/external columns "
            "show reference parity; internal bindings show binding "
            "parity. Every backend reads the same orchestrator-generated "
            "CSV dataset."
        )
    out.append(page_intro + "\n")
    info = host_cpu_info()
    generated_at = dt.datetime.now(dt.UTC).strftime("%Y-%m-%d %H:%M:%S")
    out.append(f"_Generated: {generated_at} UTC_")
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
    out.append("Each cell shows `<reported_ms> <gate>`. C++ and external "
                "library cells use Gate 2, the reference parity against "
                "the method's canonical oracle. Internal binding cells "
                "use Gate 1, the binding parity against the native C++ "
                "baseline:\n")
    out.append(f"- {ICON_OK} **exact/pass** — gate-specific tolerance passed")
    out.append(f"- {ICON_WARN} **drift** — finite mismatch, below 10× the "
                "method's gate-2 tolerance or below the binding drift band")
    out.append(f"- {ICON_FAIL} **divergent** — outside the gate's tolerance band")
    out.append(f"- {ICON_TIMEOUT} cell hit the 24 h wall-clock guard")
    out.append(f"- `—` see *Why a cell is empty* below\n")

    out.append("**Bold** = fastest cell on the row, **counted only among "
                f"cells whose relevant gate is {ICON_OK}**. For internal "
                "bindings that is Gate 1; for C++ and external libraries "
                "it is Gate 2. Drift / divergent / empty cells never carry "
                "the bold.\n")

    out.append("Timing uses the adaptive protocol: run #1 is the "
                "warmstart; if run #1 exceeds 5 min it is reported and "
                "the cell stops. Otherwise run #2 is the first scored "
                "sample and the warmstart is excluded. A 2-run cell "
                "reports run #2 alone; a 3-run cell reports the mean of "
                "runs #2-#3; 10/20/40-run cells report the median of "
                "runs after the warmstart. All backends in a single cell "
                "read the same orchestrator-generated CSV dataset. See "
                "[methodology.md](methodology.md) for the full details.\n")

    out.append("\n## Why a cell is empty (`—`)\n")
    out.append("An empty cell means the backend **did not produce a "
                "timing for that (algorithm, size) combination**. Four "
                "distinct reasons, all reported as `—` in the matrix:")
    out.append("")
    out.append("1. **External library doesn't implement the algorithm.** "
                "Example: `sklearn` has no native CPPLS or sparse SIMPLS; "
                "`plsregress` (Octave) only does plain PLS; "
                "`ikpls` is plain PLS only; "
                "`ropls` only does OPLS; "
                "`mixOmics` covers PLS / sparse PLS / PLS-DA. "
                "Filling this column requires the external maintainer "
                "to add the algorithm — out of our control.")
    out.append("2. **pls4all wrapper missing for that algorithm.** "
                "Example: `pls4all.sklearn` (tier 2) doesn't yet ship "
                "a class for `continuum_regression` or `kernel_pls`; "
                "`pls4all.R.formula` doesn't have a formula wrapper for "
                "every algorithm yet. This is the *chemin à parcourir* "
                "on our side — visible TODO.")
    out.append("3. **Bench script missing the dispatch case.** "
                "A few cells can fail because a binding bench script "
                "doesn't yet route a specific algorithm to its underlying "
                "call. Pure tooling gap, no library impact.")
    out.append("4. **Run too slow / OOM / crashed.** The cell hit the "
                "24 h guard (`⏳`) or raised a runtime error.")
    out.append("")
    out.append("Each `—` represents one of these four reasons. The CSV "
                "(`results/full_matrix.csv`) carries a `reason` column "
                "that tells you exactly which one for any given cell.\n")

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

    # Column keys present. Order them deterministically:
    #   1. pls4all.cpp.* in libp4a-build order (ref → blas → omp →
    #      blas+omp → cuda) — left-most, since they're the reference.
    #   2. pls4all.python / pls4all.sklearn
    #   3. pls4all.R / pls4all.R.formula
    #   4. pls4all.matlab / pls4all.matlab.classdef
    #   5. externals: sklearn / ikpls / pls / ropls / mixOmics / plsregress
    #   6. anything else (future-proof fallback) in first-seen order
    CANONICAL_BUILD_ORDER = ["dev-release", "blas-on", "omp-on",
                              "blas-omp", "cuda-on"]
    CANONICAL_BACKEND_ORDER = [
        "registry_pls4all",
        "python_tier1", "python_tier2",
        "r_tier1", "r_tier2", "r_pls_compat", "r_mdatools_compat",
        "matlab_tier1", "matlab_tier2",
        "sklearn", "ikpls",
        "r_pls", "r_ropls", "r_mixomics",
        "matlab_pls",
    ]
    present_keys = set()
    for r in rows:
        k = column_key(r)
        if k:
            present_keys.add(k)
    seen_keys: list[str] = []
    # 1. cpp tiers in build order.
    for build in CANONICAL_BUILD_ORDER:
        k = f"cpp::{build}"
        if k in present_keys:
            seen_keys.append(k)
    # 2-5. other backends in canonical order.
    for b in CANONICAL_BACKEND_ORDER:
        if b in present_keys:
            seen_keys.append(b)
    # 6. fallback for anything we didn't anticipate.
    for r in rows:
        k = column_key(r)
        if k and k not in seen_keys:
            seen_keys.append(k)

    for (algo, threads), grp in sorted(groups.items()):
        out.append(f"\n## {algo}  —  {threads} thread{'s' if threads > 1 else ''}\n")
        # Pivot rows: sizes; columns: ALL backend column-keys (whether
        # they cover this algo or not). Empty `—` cells are intentional
        # — they show *where coverage is still missing*.
        sizes = sorted({(int(r["n"]), int(r["p"])) for r in grp})
        backends = list(seen_keys)
        header = "| samples × features | " + " | ".join(column_disp(b) for b in backends) + " |"
        sep = "|---" + ("|---" * len(backends)) + "|"
        out.append(header)
        out.append(sep)
        for (n, p) in sizes:
            # First pass: collect (raw_ms, formatted, icon) per column.
            col_data = []
            for b in backends:
                match = [r for r in grp if column_key(r) == b
                          and int(r["n"]) == n and int(r["p"]) == p]
                if not match:
                    col_data.append((float("nan"), ICON_NA, "", ICON_NA))
                    continue
                r = match[0]
                if not _is_true(r.get("ok")):
                    col_data.append((float("nan"),
                                       fmt_ms(r.get("reported_ms") or r.get("median_ms", "")),
                                       dual_parity_label(r),
                                       effective_parity_icon(r)))
                else:
                    raw = cell_ms_value(r.get("reported_ms") or r.get("median_ms", ""))
                    col_data.append((raw,
                                       fmt_ms(r.get("reported_ms") or r.get("median_ms", "")),
                                       dual_parity_label(r),
                                       effective_parity_icon(r)))
            # Bold the fastest backend in the row, BUT only among
            # cells with strict parity ✓ (bit-exact vs reference). A
            # ⚠ cell (algorithmic drift within 1e-3) isn't strictly
            # producing the same predictions, so its timing isn't an
            # apples-to-apples win — same reason ✗ cells are excluded.
            valid_for_min = [
                (i, raw) for i, (raw, _, _, effective_icon) in enumerate(col_data)
                if raw == raw and effective_icon == ICON_OK
            ]
            min_idx = min(valid_for_min, key=lambda x: x[1])[0] if valid_for_min else None
            cells = []
            for i, (raw, formatted, icon, effective_icon) in enumerate(col_data):
                if not icon and formatted == ICON_NA:
                    # pure-empty cell
                    cells.append("—")
                    continue
                label = icon if formatted == ICON_NA else f"{formatted} {icon}".strip()
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
    # Backend defs: distinguish the cpp libp4a tiers as separate rows.
    CPP_TIER_DESC = {
        "dev-release": ("C++", "pls4all native scalar",
                          "libp4a built with `PLS4ALL_WITH_BLAS=OFF, OPENMP=OFF` — pure scalar native C++ loops, no acceleration. Used as one C++ implementation column; binding parity still uses cpp @ blas-omp when available."),
        "blas-on":     ("C++", "pls4all + BLAS",
                          "libp4a built with `PLS4ALL_WITH_BLAS=ON` only — links system BLAS (OpenBLAS in this env), benefits from BLAS thread parallelism."),
        "omp-on":      ("C++", "pls4all + OpenMP",
                          "libp4a built with `PLS4ALL_WITH_OPENMP=ON` only — OpenMP parallelism in the C kernel loops, no BLAS."),
        "blas-omp":    ("C++", "pls4all + BLAS + OpenMP",
                          "libp4a built with both `PLS4ALL_WITH_BLAS=ON` and `PLS4ALL_WITH_OPENMP=ON` — the recommended production config."),
        "cuda-on":     ("C++", "pls4all + CUDA",
                          "libp4a built with `PLS4ALL_WITH_CUDA=ON` — GEMM kernels offloaded to GPU via cuBLAS. Overhead-dominated at small matrix sizes; wins at large ones."),
    }
    for key in seen_keys:
        if key.startswith("cpp::"):
            build = key[len("cpp::"):]
            if build in CPP_TIER_DESC:
                lang, tier, what = CPP_TIER_DESC[build]
                out.append(f"| `{column_disp(key)}` | {lang} | {tier} | {what} |")
        elif key in BACKEND_LONG:
            lang, tier, what = BACKEND_LONG[key]
            out.append(f"| `{column_disp(key)}` | {lang} | {tier} | {what} |")
        elif key.startswith("ref_"):
            out.append(f"| `{column_disp(key)}` | external | reference | registry-declared external reference library |")
    out.append("")

    # Versions: collect by (backend, libp4a_build) so we can show the
    # libp4a version per cpp tier.
    versions_seen = {}
    for r in rows:
        key = column_key(r)
        if not key or key in versions_seen:
            continue
        try:
            v = json.loads(r.get("versions_json", "{}"))
        except Exception:
            v = {}
        if v:
            versions_seen[key] = v
    out.append("\n## Versions per backend\n")
    out.append("| Column | Versions |")
    out.append("|---|---|")
    for key in seen_keys:
        v = versions_seen.get(key, {})
        if not v:
            out.append(f"| `{column_disp(key)}` | — |")
            continue
        items = "; ".join(f"`{k}={v[k]}`" for k in v if v[k])
        out.append(f"| `{column_disp(key)}` | {items} |")

    out.append("\n## Methodology\n")
    out.append("- Gate 1 reference: `cpp` cell at 1 thread (libp4a via "
                "ctypes), or `python_tier1` when `cpp` is unavailable "
                "for an algorithm")
    out.append("- Gate 2 reference: the registry-declared canonical "
                "external reference for the method")
    out.append("- Gate 1 tolerance: 1e-6 max-abs-diff")
    out.append("- Gate 2 release tolerance: strict rmse_rel <= 1e-3 "
                "(<= 1e-6 is displayed as exact). `MethodSpec.rmse_rel_tol` "
                "is only a diagnostic/variant budget and must not turn a "
                "strict reference divergence into a passing release gate.")
    out.append("- All backends read the same orchestrator-generated CSV dataset "
                "(`benchmarks/cross_binding/data/data_<n>x<p>_seed<seed>.csv`)")
    out.append("- Adaptive timing: warmstart run #1; if there is any "
               "subsequent run, run #1 is excluded from the score. "
               "2 total runs report run #2 alone; 3 total runs report "
               "the mean of runs #2-#3; 10/20/40 total runs report the "
               "median after the warmstart.")
    out.append("- Per-cell timeout guard: 24 h")
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
