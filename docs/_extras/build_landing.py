"""Build the JSON payload that powers the interactive landing dashboard.

Reads the canonical `benchmarks/cross_binding/results/full_matrix.csv`,
normalises rows to a dashboard-friendly shape and
emits a compact JSON blob. The blob is injected into
`docs/_templates/landing.html` at sphinx-build time via the
`bench_data_json` html_context variable populated by `conf.py:setup()`.

Same column-naming and parity-icon conventions as `render_docs.py` so
the dashboard agrees with the per-page markdown tables.
"""
from __future__ import annotations

import csv
import datetime as dt
import json
import os
import platform
import re
import subprocess
from collections import defaultdict
from pathlib import Path
from typing import Any


# Same canonical orderings & display labels as render_docs.py — kept in
# sync deliberately so dashboard column ids match the markdown tables.
BACKEND_DISPLAY: dict[str, str] = {
    "registry_pls4all": "pls4all.registry",
    "cpp":          "pls4all.cpp",     # gets a build suffix
    "python_tier1": "pls4all.python",
    "python_tier2": "pls4all.sklearn",
    "r_tier1":      "pls4all.R",
    "r_tier2":      "pls4all.R.formula",
    "matlab_tier1": "pls4all.matlab",
    "matlab_tier2": "pls4all.matlab.classdef",
    "sklearn":      "sklearn",
    "ikpls":        "ikpls",
    "r_pls":        "pls",
    "r_ropls":      "ropls",
    "r_mixomics":   "mixOmics",
    "matlab_pls":   "plsregress",
}

CPP_BUILD_SUFFIX = {
    "dev-release": "ref",
    "blas-on":     "blas",
    "omp-on":      "omp",
    "blas-omp":    "blas+omp",
    "cuda-on":     "cuda",
}

CPP_TIER_DESC = {
    "dev-release": ("pls4all reference (single-thread)",
                    "libp4a built with `PLS4ALL_WITH_BLAS=OFF, OPENMP=OFF` — pure scalar reference loops, no acceleration. The parity baseline."),
    "blas-on":     ("pls4all + BLAS",
                    "libp4a built with `PLS4ALL_WITH_BLAS=ON` only — links system BLAS (OpenBLAS), benefits from BLAS thread parallelism."),
    "omp-on":      ("pls4all + OpenMP",
                    "libp4a built with `PLS4ALL_WITH_OPENMP=ON` only — OpenMP in C kernel loops, no BLAS."),
    "blas-omp":    ("pls4all + BLAS + OpenMP",
                    "libp4a built with both `BLAS=ON` and `OPENMP=ON` — the recommended production config."),
    "cuda-on":     ("pls4all + CUDA",
                    "libp4a built with `PLS4ALL_WITH_CUDA=ON` — GEMM kernels offloaded to GPU via cuBLAS. Overhead-dominated at small matrices; wins at large ones."),
}

BACKEND_LONG: dict[str, tuple[str, str, str]] = {
    "python_tier1": ("Python",        "pls4all raw",        "`pls4all._methods.<algo>_fit(ctx, cfg, X, y, …)` — direct FFI binding"),
    "registry_pls4all": ("Python",    "pls4all bench reference",
                                       "`benchmarks.parity_timing.registry.MethodSpec.pls4all_fn` — the canonical pls4all invocation defined by the benchmark suite for each algorithm (algorithm + solver + deflation + …). This is the column used as the parity reference against sklearn / pls / ropls / plsregress."),
    "python_tier2": ("Python",        "pls4all idiomatic",  "`pls4all.sklearn.<Class>` — sklearn-style BaseEstimator with `.fit()/.predict()`"),
    "sklearn":      ("Python",        "external",           "`sklearn.cross_decomposition.PLSRegression`, `sklearn.decomposition.PCA` + LinearRegression / Ridge / GaussianProcessRegressor (proxies)"),
    "ikpls":        ("Python",        "external",           "`ikpls.numpy_ikpls.PLS` — Improved Kernel PLS (plain PLS only)"),
    "r_tier1":      ("R",             "pls4all raw",        "`pls4all_method(algo, X, y, ...)` — unified dispatcher (33 fits + 24 selectors + 4 diagnostics)"),
    "r_tier2":      ("R",             "pls4all idiomatic",  "`pls(y ~ ., data)`, `cppls(...)`, `sparse_pls(...)`, … — base R formula+S3 wrappers"),
    "r_pls":        ("R",             "external",           "CRAN `pls` package — `pls::plsr / pls::cppls / pls::pcr`"),
    "r_ropls":      ("R",             "external",           "Bioconductor `ropls` — `ropls::opls` (OPLS only)"),
    "r_mixomics":   ("R",             "external",           "Bioconductor `mixOmics` — `pls / spls / plsda / splsda`"),
    "matlab_tier1": ("MATLAB/Octave", "pls4all raw",        "`pls4all.<algo>(X, y, ...)` — single dispatcher MEX"),
    "matlab_tier2": ("MATLAB/Octave", "pls4all idiomatic",  "`pls4all.fit(algo, X, y, ...)` factory + per-algorithm classdefs"),
    "matlab_pls":   ("MATLAB/Octave", "external",           "Octave statistics `plsregress` (SIMPLS, plain PLS only)"),
}

# Column ordering — matches render_docs.py canonical order.
CPP_BUILD_ORDER = ["dev-release", "blas-on", "omp-on", "blas-omp", "cuda-on"]
BACKEND_ORDER = [
    "registry_pls4all",
    "python_tier1", "python_tier2",
    "r_tier1", "r_tier2",
    "matlab_tier1", "matlab_tier2",
    "sklearn", "ikpls",
    "r_pls", "r_ropls", "r_mixomics",
    "matlab_pls",
]

GROUP_LABELS = {
    "cpp":     "pls4all · C++ (libp4a)",
    "python":  "pls4all · Python",
    "r":       "pls4all · R",
    "matlab":  "pls4all · MATLAB/Octave",
    "ext-py":  "external · Python",
    "ext-r":   "external · R",
    "ext-ml":  "external · MATLAB/Octave",
}

BACKEND_GROUP = {
    "python_tier1": "python", "python_tier2": "python",
    "registry_pls4all": "python",
    "r_tier1": "r", "r_tier2": "r",
    "matlab_tier1": "matlab", "matlab_tier2": "matlab",
    "sklearn": "ext-py", "ikpls": "ext-py",
    "r_pls": "ext-r", "r_ropls": "ext-r", "r_mixomics": "ext-r",
    "matlab_pls": "ext-ml",
}

# Algorithm taxonomy — used by the dashboard for the "group" filter and
# the per-algo group label. Order in ALGO_GROUPS_ORDER drives display
# order in the dropdown.
ALGO_GROUPS_ORDER = [
    "core", "ensemble", "sparse", "robust", "nonlinear",
    "multi-block", "calibration-transfer", "classification",
    "missing", "regularized", "other",
]
ALGO_GROUP_LABELS = {
    "core":                 "Core PLS",
    "ensemble":             "Ensemble",
    "sparse":               "Sparse",
    "robust":               "Robust / weighted",
    "nonlinear":            "Nonlinear / local",
    "multi-block":          "Multi-block / cross-modal",
    "calibration-transfer": "Calibration transfer",
    "classification":       "Classification & GLM",
    "missing":              "Missing data",
    "regularized":          "Regularized",
    "other":                "Other",
}
ALGO_GROUP = {
    "pls":                  "core",
    "cppls":                "core",
    "sparse_simpls":        "sparse",
    "fused_sparse_pls":     "sparse",
    "sparse_pls_da":        "sparse",
    "group_sparse_pls":     "sparse",
    "bagging_pls":          "ensemble",
    "boosting_pls":         "ensemble",
    "robust_pls":           "robust",
    "weighted_pls":         "robust",
    "kernel_pls":           "nonlinear",
    "lw_pls":               "nonlinear",
    "gpr_pls":              "nonlinear",
    "continuum_regression": "nonlinear",
    "o2pls":                "multi-block",
    "mir_pls":              "multi-block",
    "ds":                   "calibration-transfer",
    "pds":                  "calibration-transfer",
    "ecr":                  "calibration-transfer",
    "pls_lda":              "classification",
    "pls_logistic":         "classification",
    "pls_qda":              "classification",
    "pls_glm":              "classification",
    "pls_cox":              "classification",
    "missing_aware_nipals": "missing",
    "ridge_pls":            "regularized",
}


def column_id(backend: str, build: str) -> str:
    """Same column identity used in the markdown — `pls4all.cpp.<suffix>`
    for the C++ tiers, the display label otherwise."""
    if backend == "cpp":
        suffix = CPP_BUILD_SUFFIX.get(build, build)
        return f"pls4all.cpp.{suffix}"
    if backend.startswith("ref_"):
        return "ref." + backend[len("ref_"):]
    return BACKEND_DISPLAY.get(backend, backend)


def is_true(v: Any) -> bool:
    return str(v).lower() == "true"


def parity_code(row: dict) -> str:
    """Legacy binding-consistency parity code (gate 1 only).

    exact | drift | divergent | error | not_run | not_available.

    Distinguishes hard runtime errors (ModuleNotFoundError, ImportError,
    crash) from libraries that do not expose an algorithm so the dashboard
    can show the right icon instead of collapsing every failure to a dash.
    Reads `binding_parity_*` first, falls back to legacy `parity_*`.
    """
    if not is_true(row.get("ok")):
        reason = (row.get("reason") or "").lower()
        if "timeout" in reason:
            return "not_run"
        if ("not implemented by" in reason or
                "unsupported algo" in reason or
                "unsupported algorithm" in reason):
            return "not_available"
        if any(k in reason for k in (
                "modulenotfounderror", "importerror", "notimplemented",
                "error", "exception", "crash", "traceback", "segfault")):
            return "error"
        return "not_run"
    parity_ok = row.get("binding_parity_ok", row.get("parity_ok"))
    if is_true(parity_ok):
        return "exact"
    diff_val = row.get("binding_parity_max_diff",
                        row.get("parity_max_diff", "nan"))
    try:
        d = float(diff_val)
    except (TypeError, ValueError):
        return "drift"
    if d != d:           # NaN
        return "drift"
    if d < 1e-3:
        return "drift"
    return "divergent"


def reference_parity_code(row: dict) -> str:
    """Gate-2 parity classification.

    Values:
      exact          — within `method.rmse_rel_tol`.
      drift          — finite but above tolerance, < 10× tolerance.
      divergent      — > 10× tolerance.
      not_available  — `reference_kind=paper_only` (no executable truth).
      not_run        — backend didn't produce a prediction at all.
      error          — runtime error path (delegated to gate 1's
                       classifier, kept here so the two columns line up).
    """
    if not is_true(row.get("ok")):
        # Defer to the same error/timeout categorisation as gate 1.
        return parity_code(row)
    kind = (row.get("reference_kind") or "").lower()
    if kind == "paper_only":
        return "not_available"
    ref_ok = row.get("reference_parity_ok")
    if ref_ok in (None, "", "None"):
        return "not_available"
    if is_true(ref_ok):
        return "exact"
    rel = row.get("reference_parity_rmse_rel", "nan")
    try:
        d = float(rel)
    except (TypeError, ValueError):
        return "drift"
    if d != d:
        return "drift"
    if d < 10:
        return "drift"
    return "divergent"


def fmt_ms(ms_str: str) -> str:
    try:
        f = float(ms_str)
    except (TypeError, ValueError):
        return "—"
    if f >= 1000:
        return f"{f / 1000:.1f} s"
    if f >= 10:
        return f"{f:.1f} ms"
    return f"{f:.2f} ms"


def host_info() -> dict:
    info = {
        "os": f"{platform.system()} {platform.release()} ({platform.machine()})",
        "python": platform.python_version(),
        "cpu_model": platform.processor() or "unknown",
        "cpu_count_logical": os.cpu_count() or "?",
        "cpu_count_physical": "?",
        "ram_gb": "?",
        "kernel": platform.release(),
        "cpu_mhz": None,
        "cpu_cache": None,
    }
    cpuinfo = Path("/proc/cpuinfo")
    if cpuinfo.exists():
        try:
            text = cpuinfo.read_text()
            m = re.search(r"^model name\s*:\s*(.+)$", text, re.M)
            if m:
                info["cpu_model"] = m.group(1).strip()
            pairs = set(re.findall(r"physical id\s*:\s*(\d+).*?core id\s*:\s*(\d+)",
                                    text, re.S))
            if pairs:
                info["cpu_count_physical"] = len(pairs)
            mhz = re.search(r"^cpu MHz\s*:\s*([\d.]+)$", text, re.M)
            if mhz:
                info["cpu_mhz"] = f"{float(mhz.group(1)):.0f}"
            cache = re.search(r"^cache size\s*:\s*(.+)$", text, re.M)
            if cache:
                info["cpu_cache"] = cache.group(1).strip()
        except Exception:
            pass
    meminfo = Path("/proc/meminfo")
    if meminfo.exists():
        try:
            m = re.search(r"^MemTotal:\s*(\d+)\s*kB", meminfo.read_text(), re.M)
            if m:
                info["ram_gb"] = f"{int(m.group(1)) / 1024 / 1024:.1f}"
        except Exception:
            pass
    try:
        info["kernel"] = subprocess.check_output(
            ["uname", "-r"], timeout=2).decode().strip()
    except Exception:
        pass
    return info


def build_payload(results_dir: Path) -> dict:
    """Read canonical cross-binding CSVs and build the dashboard payload."""
    rows_in: list[dict] = []
    full_matrix = results_dir / "full_matrix.csv"
    csv_paths = [full_matrix] if full_matrix.exists() else []
    generated_at = dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    if csv_paths:
        latest_source = max(p.stat().st_mtime for p in csv_paths)
        generated_at = dt.datetime.fromtimestamp(
            latest_source, dt.timezone.utc
        ).strftime("%Y-%m-%d %H:%M UTC")
    for p in csv_paths:
        rows_in.extend(csv.DictReader(p.open()))

    # Dedupe: same (algo, backend, build, n, p, threads) — keep last seen.
    seen: dict[tuple, dict] = {}
    for r in rows_in:
        if not r.get("algorithm"):
            continue
        try:
            n, p_, t = int(r["n"]), int(r["p"]), int(r["threads"])
        except (ValueError, KeyError):
            continue
        key = (r["algorithm"], r["backend"], r.get("libp4a_build", ""), n, p_, t)
        seen[key] = r

    # Collect columns actually present.
    present: set[str] = set()
    for r in seen.values():
        present.add(column_id(r["backend"], r.get("libp4a_build", "")))

    # Registry-declared external reference libraries. Infer the language
    # from the slug prefix (`ref.python_*`, `ref.r_*`, `ref.matlab_*`,
    # `ref.julia_*`, …) so the language band groups them correctly.
    REF_LANG_PREFIX = {
        "python":  ("Python",        "ext-py"),
        "r":       ("R",             "ext-r"),
        "matlab":  ("MATLAB/Octave", "ext-ml"),
        "octave":  ("MATLAB/Octave", "ext-ml"),
        "julia":   ("Julia",         "ext-jl"),
        "rust":    ("Rust",          "ext-rs"),
        "go":      ("Go",            "ext-go"),
        "js":      ("JavaScript",    "ext-js"),
        "java":    ("Java",          "ext-java"),
        "cpp":     ("C++",           "ext-cpp"),
    }
    def _ref_lang(cid: str) -> tuple[str, str]:
        rest = cid[len("ref."):]
        for prefix, (lang, group) in REF_LANG_PREFIX.items():
            if rest.startswith(prefix + "_") or rest == prefix:
                return lang, group
        return "external", "ext-py"

    def _ref_short(cid: str, lang_prefix: str) -> str:
        """Compact a `ref.python_nirs4all_operators_models_sklearn_lwpls`
        slug down to something readable in narrow columns: drop the
        language prefix, then collapse known package prefixes."""
        rest = cid[len("ref."):]
        if rest.startswith(lang_prefix + "_"):
            rest = rest[len(lang_prefix) + 1:]
        elif rest == lang_prefix:
            rest = lang_prefix
        # Last segment is usually the most informative method/package name.
        parts = rest.split("_")
        if len(parts) >= 2:
            return f"{parts[0]}/{parts[-1]}"
        return rest

    # Build raw column entries first, then sort by language to get
    # contiguous language bands.
    raw_cols: list[dict] = []
    for build in CPP_BUILD_ORDER:
        cid = f"pls4all.cpp.{CPP_BUILD_SUFFIX[build]}"
        if cid in present:
            tier, what = CPP_TIER_DESC[build]
            raw_cols.append({
                "id": cid, "label": cid, "short": cid[len("pls4all.cpp."):],
                "group": "cpp", "tier": tier, "lang": "C++",
                "what": what, "build": build, "kind": "pls4all",
            })
    # Hand-tuned overrides for the displayed `short` label when the
    # default "strip pls4all. prefix" heuristic produces internal-jargon
    # names. Keys are the canonical backend slug from BACKEND_DISPLAY.
    SHORT_OVERRIDES = {
        # "registry" came from the python file name (registry.py) and
        # collided semantically with "canonical PLS" in the PLS lit
        # (CPPLS / Indahl 2009). "bench-ref" is unambiguous: this column
        # is the parity reference invocation of pls4all used by the
        # benchmark suite for each algorithm.
        "registry_pls4all": "bench-ref",
    }
    for be in BACKEND_ORDER:
        cid = BACKEND_DISPLAY.get(be, be)
        if cid in present:
            lang, tier, what = BACKEND_LONG[be]
            is_pls4all = cid.startswith("pls4all.")
            short = SHORT_OVERRIDES.get(be) or (cid[len("pls4all."):] if is_pls4all else cid)
            raw_cols.append({
                "id": cid, "label": cid, "short": short,
                "group": BACKEND_GROUP[be], "tier": tier, "lang": lang,
                "what": what, "build": "",
                "kind": "pls4all" if is_pls4all else "external",
            })
    for cid in sorted(c for c in present if c.startswith("ref.")):
        lang, group = _ref_lang(cid)
        # Map the lang back to the prefix used in the cid.
        lang_to_prefix = {v[0]: k for k, v in REF_LANG_PREFIX.items()}
        prefix = lang_to_prefix.get(lang, "")
        short = _ref_short(cid, prefix) if prefix else cid[len("ref."):]
        raw_cols.append({
            "id": cid, "label": cid, "short": short,
            "group": group, "tier": "registry reference", "lang": lang,
            "what": f"Registry-declared external reference library ({lang}).",
            "build": "", "kind": "external",
        })

    # Canonical language order for contiguous band rendering: C++,
    # Python, R, MATLAB/Octave, then other languages alphabetically.
    LANG_ORDER = ["C++", "Python", "R", "MATLAB/Octave",
                  "Julia", "Rust", "Go", "JavaScript", "Java", "external"]
    def _lang_rank(c: dict) -> int:
        try:
            return LANG_ORDER.index(c["lang"])
        except ValueError:
            return len(LANG_ORDER)
    # Stable sort: language rank → kind (pls4all first) → original order.
    columns = sorted(raw_cols, key=lambda c: (_lang_rank(c),
                                                0 if c["kind"] == "pls4all" else 1))

    # Build rows: one per (algo, n, p, threads). Cells keyed by column id.
    # We always emit a row for every (algo, size, threads) triple that
    # appears in the CSVs — even if every backend in that row failed —
    # so the dashboard surfaces the "this algo is unimplemented
    # everywhere" case (e.g. ds, pds, pls_cox, weighted_pls).
    pivot: dict[tuple, dict] = defaultdict(dict)
    for r in seen.values():
        n = int(r["n"]); p_ = int(r["p"]); t = int(r["threads"])
        rkey = (r["algorithm"], n, p_, t)
        if rkey not in pivot:
            pivot[rkey] = {}        # ensure row exists even with no cells
        cid = column_id(r["backend"], r.get("libp4a_build", ""))
        verdict = parity_code(r)
        ref_verdict = reference_parity_code(r)
        try:
            ms = float(r["median_ms"]) if r.get("median_ms") else None
        except ValueError:
            ms = None
        ok = is_true(r.get("ok"))
        # `parity` keeps the legacy gate-1 verdict (binding consistency
        # vs cpp). `reference_parity` is the new gate-2 verdict (kernel
        # vs the method's canonical external reference). The dashboard
        # renders both icons per cell.
        cell = {"parity": verdict, "reference_parity": ref_verdict,
                "ok": ok}
        if ms is not None and ms == ms:   # not NaN
            cell["ms"] = round(ms, 3)
            cell["fmt"] = fmt_ms(r["median_ms"])
        kind = (r.get("reference_kind") or "").strip()
        if kind:
            cell["reference_kind"] = kind
        if is_true(r.get("is_canonical_reference")):
            cell["is_canonical_reference"] = True
        # Capture the failure reason so the dashboard tooltip can show
        # "ModuleNotFoundError: foo" instead of a silent dash.
        reason = (r.get("reason") or "").strip()
        if reason and verdict in ("error", "not_run", "not_available",
                                  "divergent", "drift"):
            cell["reason"] = reason[:200]
        # Keep every explicit verdict emitted by the orchestrator so the
        # dashboard can distinguish "not run" from "not available in lib".
        if ms is not None or verdict in (
                "not_run", "not_available", "divergent", "drift", "error"):
            pivot[rkey][cid] = cell

    rows_out = []
    for (algo, n, p_, t), cells in sorted(pivot.items(),
                                            key=lambda x: (x[0][0],
                                                            x[0][1] * x[0][2],
                                                            x[0][3])):
        rows_out.append({
            "algo": algo, "n": n, "p": p_, "threads": t,
            "cells": cells,
        })

    # Per-column versions (best available).
    versions: dict[str, str] = {}
    for r in seen.values():
        cid = column_id(r["backend"], r.get("libp4a_build", ""))
        if cid in versions:
            continue
        try:
            v = json.loads(r.get("versions_json") or "{}")
        except Exception:
            v = {}
        if v:
            items = "; ".join(f"{k}={v[k]}" for k in v if v[k])
            if items:
                versions[cid] = items

    # Presets: which columns to show by default. Each preset's column
    # list is filtered against `all_cols` so ghost ids that don't exist
    # in the current data never make it into state.cols (a ghost id
    # would blank the table — fixed). Empty presets are dropped.
    all_cols = [c["id"] for c in columns]
    cpp_blas_omp = "pls4all.cpp.blas+omp" if "pls4all.cpp.blas+omp" in all_cols else None
    cpp_ref = "pls4all.cpp.ref" if "pls4all.cpp.ref" in all_cols else None

    by_group = defaultdict(list)
    for c in columns:
        by_group[c["group"]].append(c["id"])
    pls4all_groups = ("cpp", "python", "r", "matlab")

    def by_lang(lang_value: str) -> list:
        """All columns (pls4all bindings + externals) for a host language,
        preserving the canonical column order."""
        return [c["id"] for c in columns if c["lang"] == lang_value]

    headline_candidates = [
        cpp_blas_omp, "pls4all.python", "pls4all.sklearn",
        "pls4all.R", "pls4all.matlab",
        # add the canonical registry column too — it's the new "pls4all"
        # entry point in the registry-driven runs
        "pls4all.registry",
        "sklearn", "pls", "plsregress",
    ]
    headline = [c for c in headline_candidates if c and c in all_cols]
    cpp_tiers = [c for c in all_cols if c.startswith("pls4all.cpp.")]
    pls4all_only = [c for c in all_cols if c.startswith("pls4all.")]
    externals = [c for c in all_cols if not c.startswith("pls4all.")]
    thread_sweep = [c for c in [cpp_blas_omp, "pls4all.python",
                                  "pls4all.sklearn", "pls4all.registry",
                                  "sklearn"] if c and c in all_cols]
    # API-parity view — same algorithm, same idiomatic API in each
    # language. Compares pls4all's "mimicking" bindings (sklearn-style,
    # R-formula + S3, MATLAB classdef) against the canonical external
    # libraries users would otherwise reach for. Raw low-level bindings
    # (pls4all.python ctypes, pls4all.R dispatcher, pls4all.matlab MEX)
    # are intentionally excluded — they're the wrong API to compare
    # against sklearn / pls / plsregress.
    api_parity_candidates = [
        cpp_blas_omp,
        # Python: pls4all sklearn-style binding vs canonical externals
        "pls4all.sklearn",
        "sklearn", "ikpls", "ref.python_scikit_learn", "ref.python_ikpls",
        # R: pls4all formula + S3 binding vs canonical R externals
        "pls4all.R.formula",
        "pls", "mixOmics", "ropls",
        "ref.r_pls", "ref.r_mixomics", "ref.r_ropls",
        # MATLAB: pls4all classdef binding vs the stats-toolbox baseline
        "pls4all.matlab.classdef",
        "plsregress", "ref.matlab_libpls",
    ]
    api_parity = [c for c in api_parity_candidates if c and c in all_cols]

    # tier-1 = raw low-level dispatcher bindings (ctypes, R dispatcher,
    # MEX) — the columns named like their host language root.
    tier_1 = [c for c in (
        "pls4all.python", "pls4all.R", "pls4all.matlab"
    ) if c in all_cols]
    # tier-2 = idiomatic per-language wrappers built on top of the
    # tier-1 dispatch (sklearn-style estimators, R formula + S3, MATLAB
    # classdef). These are the bindings users will actually write code
    # against, so the dashboard wants a one-click view that lines them
    # up against each other.
    tier_2 = [c for c in (
        "pls4all.sklearn", "pls4all.R.formula", "pls4all.matlab.classdef"
    ) if c in all_cols]
    # registry/bench-ref is the single canonical pls4all invocation
    # used by the benchmark suite as parity baseline; expose it as its
    # own preset so users can see only what's parity-anchored.
    bench_ref = [c for c in ("pls4all.registry",) if c in all_cols]

    # Per-language sets (pls4all bindings + externals for that host language)
    py_all = by_lang("Python")
    r_all  = by_lang("R")
    m_all  = by_lang("MATLAB/Octave")

    # Native baseline tiers — compare the parity-reference scalar build
    # against the production BLAS+OpenMP one, plus the registry entry
    # point. This is the "how much speed do the C++ optimisations buy?"
    # view, which is what the benchmark is really *for*.
    native_baseline = [c for c in (
        cpp_ref, cpp_blas_omp, "pls4all.registry",
    ) if c and c in all_cols]

    raw_presets = {
        # Top-level overviews — listed first so the preset rail leads
        # with the curated comparisons before the bulkier "all of X"
        # presets.
        "headline":     {"label": "Headline",       "cols": headline},
        "api-parity":   {"label": "API parity",     "cols": api_parity},
        "tier-2":       {"label": "Tier-2 idiomatic","cols": [cpp_blas_omp, *tier_2] if cpp_blas_omp else tier_2},
        "tier-1":       {"label": "Tier-1 raw",      "cols": [cpp_blas_omp, *tier_1] if cpp_blas_omp else tier_1},
        "bench-ref":    {"label": "Bench reference","cols": bench_ref},
        "native":       {"label": "C++ native",     "cols": native_baseline},
        "cpp-tiers":    {"label": "C++ all tiers",  "cols": cpp_tiers},
        # Language-scoped views — keep the C++ baseline included on each
        # so users always see the anchor next to the language slice.
        "python":       {"label": "Python (all)",   "cols": ([cpp_blas_omp] if cpp_blas_omp else []) + py_all},
        "r":            {"label": "R (all)",        "cols": ([cpp_blas_omp] if cpp_blas_omp else []) + r_all},
        "matlab":       {"label": "MATLAB (all)",   "cols": ([cpp_blas_omp] if cpp_blas_omp else []) + m_all},
        # Roll-up sets — keep at the end since they're the broadest.
        "pls4all":      {"label": "pls4all only",   "cols": pls4all_only},
        "externals":    {"label": "Externals",      "cols": externals},
        "thread-sweep": {"label": "Thread sweep",   "cols": thread_sweep},
        "all":          {"label": "All",            "cols": all_cols},
    }
    # Drop presets that would resolve to zero columns in the current data.
    presets = {k: v for k, v in raw_presets.items() if v["cols"]}

    # Algo → group mapping (only for algos actually present in the data).
    present_algos = sorted({r["algo"] for r in rows_out})
    algo_to_group = {a: ALGO_GROUP.get(a, "other") for a in present_algos}
    # Groups actually present, in canonical order.
    present_group_keys = [g for g in ALGO_GROUPS_ORDER
                            if any(algo_to_group[a] == g for a in present_algos)]
    algo_groups = [
        {"key": g, "label": ALGO_GROUP_LABELS[g],
         "algos": [a for a in present_algos if algo_to_group[a] == g]}
        for g in present_group_keys
    ]
    # Languages actually present in the columns.
    present_langs = []
    seen_lang = set()
    for c in columns:
        if c.get("lang") and c["lang"] not in seen_lang:
            seen_lang.add(c["lang"])
            present_langs.append(c["lang"])

    payload = {
        "generated_at": generated_at,
        "host":         host_info(),
        "columns":      columns,
        "presets":      presets,
        "rows":         rows_out,
        "versions":     versions,
        "algo_to_group":algo_to_group,
        "algo_groups":  algo_groups,
        "languages":    present_langs,
        "stats": {
            "algos":    len(present_algos),
            "backends": len(columns),
            "groups":   len(algo_groups),
            "sizes":    len({(r["n"], r["p"]) for r in rows_out}),
            "threads":  sorted({r["threads"] for r in rows_out}),
            "rows":     len(rows_out),
            "cells":    sum(len(r["cells"]) for r in rows_out),
            "ok":       sum(1 for r in rows_out for c in r["cells"].values()
                            if c.get("parity") == "exact"),
        },
    }
    return {
        "payload":      payload,
        "json":         json.dumps(payload, separators=(",", ":")),
        "generated_at": payload["generated_at"],
    }


def main() -> None:
    """CLI: write the payload to docs/_static/bench-data.json for inspection."""
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--results",
                     default="benchmarks/cross_binding/results",
                     help="directory containing full_matrix.csv")
    ap.add_argument("--out",
                     default="docs/_static/bench-data.json",
                     help="output path for the JSON payload")
    args = ap.parse_args()
    p = build_payload(Path(args.results))
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(p["payload"], indent=2))
    s = p["payload"]["stats"]
    print(f"wrote {out} · {s['cells']} cells / {s['rows']} rows / "
            f"{s['backends']} cols / {s['algos']} algos / "
            f"{s['ok']} exact ✓ ({s['ok']/s['cells']*100:.0f}%)")


if __name__ == "__main__":
    main()
