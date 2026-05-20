"""Build the JSON payload that powers the interactive landing dashboard.

Reads the canonical `benchmarks/cross_binding/results/full_matrix.csv`
plus optional `dashboard_refresh_*.csv` deltas, normalises rows to a
dashboard-friendly shape and emits a compact JSON blob. The blob is injected into
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
import sys
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
    "r_pls_compat": "pls4all.R.pls_compat",
    "r_mdatools_compat": "pls4all.R.mdatools_compat",
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
    "dev-release": "native",
    "blas-on":     "blas",
    "omp-on":      "omp",
    "blas-omp":    "blas+omp",
    "cuda-on":     "cuda",
}

CPP_TIER_DESC = {
    "dev-release": ("pls4all native scalar",
                    "libp4a built with `PLS4ALL_WITH_BLAS=OFF, OPENMP=OFF` — pure scalar native C++ loops, no acceleration. Used as one C++ implementation column; binding parity still uses cpp @ blas-omp when available."),
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
    "r_pls_compat": ("R",             "pls4all pls-compatible", "`plsr()` / `pcr()` for PLS/PCR/CPPLS, formula-built dispatcher path for the rest of the method matrix"),
    "r_mdatools_compat": ("R",        "pls4all mdatools-compatible", "`pls(x, y, ...)` for PLS/PCR/CPPLS, matrix dispatcher path for the rest of the method matrix"),
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
    "r_tier1", "r_tier2", "r_pls_compat", "r_mdatools_compat",
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
    "r_pls_compat": "r", "r_mdatools_compat": "r",
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
    "missing", "regularized", "adaptive", "other",
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
    "adaptive":             "Adaptive operators",
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
    "aom_preprocess":       "adaptive",
    "aom_pls":              "adaptive",
    "pop_pls":              "adaptive",
}

SELECTOR_ALGOS = {
    "vip_select", "coefficient_select", "selectivity_ratio_select",
    "spa_select", "cars_select", "ga_select", "pso_select",
    "vissa_select", "iriv_select", "irf_select", "shaving_select",
    "bve_select", "rep_select", "ipw_select", "st_select",
    "t2_select", "wvc_select", "wvc_threshold_select",
    "emcuve_select", "randomization_select", "bipls_select",
    "sipls_select", "interval_select", "stability_select",
    "uve_select", "random_frog_select", "scars_select",
    "vip_spa_select",
}

MATLAB_TIER2_SUPPORTED = {
    "pls", "pls_simpls", "simpls", "pcr", "sparse_simpls", "cppls",
    "ecr", "weighted_pls", "robust_pls", "ridge_pls",
    "continuum_regression", "recursive_pls", "n_pls",
    "kernel_pls", "kernel_pls_rbf", "o2pls", "mb_pls",
    "pls_glm", "mir_pls", "missing_aware_nipals",
    "bagging_pls", "boosting_pls",
}

FIXED_EXTERNAL_SUPPORT = {
    "ikpls": {"pls"},
    "r_pls": {"pls", "pcr", "cppls"},
    "r_ropls": {"opls"},
    "matlab_pls": {"pls"},
}


# Registry-driven `ref_*` backends that ARE the same library as one of
# the fixed legacy externals. We collapse the two into a single
# dashboard column so the user doesn't see "sklearn" AND
# "ref.python_scikit_learn" side by side. The fixed-bench script
# provides the timing column; the registry-driven `ref_*` row provides
# the gate-2 parity verdict (which the orchestrator's compute_parity
# uses internally — the row never appears in the dashboard).
REF_ALIAS_TO_LEGACY = {
    "ref_python_scikit_learn":  "sklearn",
    "ref_python_ikpls":         "ikpls",
    "ref_r_pls":                "r_pls",
    "ref_r_ropls":              "r_ropls",
    "ref_r_mixomics":           "r_mixomics",
}

# Display labels for registry-driven `ref_*` backends that don't have a
# legacy equivalent. Drop the noisy "ref." prefix and just use the
# library name verbatim.
REF_DISPLAY_OVERRIDE = {
    "ref_python_diplslib":          "diplslib",
    "ref_python_tensorly":          "tensorly",
    "ref_python_auswahl":           "auswahl",
    "ref_python_pyswarms":          "pyswarms",
    "ref_python_onpls":             "onpls",
    "ref_python_nirs4all":                                  "nirs4all",
    "ref_python_nirs4all_operators_models_sklearn_aom_pls": "nirs4all",
    "ref_python_nirs4all_operators_models_sklearn_mbpls":   "nirs4all",
    "ref_python_nirs4all_operators_models_sklearn_lwpls":   "nirs4all",
    "ref_python_nirs4all_bench_aom_v0_aompls":              "nirs4all",
    "ref_r_spls":                   "spls",
    "ref_r_chemometrics":           "chemometrics",
    "ref_r_jico":                   "JICO",
    "ref_r_kernlab_pls":            "kernlab",
    "ref_r_omicspls":               "OmicsPLS",
    "ref_r_mdatools":               "mdatools",
    "ref_r_multiblock":             "multiblock",
    "ref_r_mboost":                 "mboost",
    "ref_r_plsrglm":                "plsRglm",
    "ref_r_plsrcox":                "plsRcox",
    "ref_r_base":                   "stats (R)",
    "ref_r_softimpute":             "softImpute",
    "ref_r_sgpls":                  "sgPLS",
    "ref_r_plsvarsel":              "plsVarSel",
    "ref_r_enpls":                  "enpls",
    "ref_r_pls_stats":              "pls (R, stats)",
    "ref_matlab_libpls":            "libPLS",
}


def column_id(backend: str, build: str) -> str:
    """Same column identity used in the markdown — `pls4all.cpp.<suffix>`
    for the C++ tiers, the display label otherwise.

    Critical: `ref_*` rows that are the SAME library as a fixed legacy
    external collapse onto the legacy column (so the dashboard shows ONE
    column per library, not two). Standalone `ref_*` rows (libraries
    that have no legacy bench script) get a display label override so
    the column shows up as e.g. "OmicsPLS" rather than "ref.r_omicspls".
    """
    if backend == "cpp":
        suffix = CPP_BUILD_SUFFIX.get(build, build)
        return f"pls4all.cpp.{suffix}"
    if backend.startswith("ref_"):
        legacy = REF_ALIAS_TO_LEGACY.get(backend)
        if legacy is not None:
            return BACKEND_DISPLAY.get(legacy, legacy)
        return REF_DISPLAY_OVERRIDE.get(backend, backend[len("ref_"):])
    return BACKEND_DISPLAY.get(backend, backend)


def collapsed_backend(backend: str) -> str:
    """When a `ref_*` row is the same library as a legacy fixed external,
    return the legacy backend id. Used to decide whether a `ref_*` row
    should be merged into an existing legacy cell or kept as its own
    standalone column."""
    return REF_ALIAS_TO_LEGACY.get(backend, backend)


def is_true(v: Any) -> bool:
    return str(v).lower() == "true"


def _float_or_none(v: Any) -> float | None:
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    if f != f:
        return None
    return f


_METHOD_TOLERANCES: dict[str, float] | None = None
STRICT_REFERENCE_EXACT_TOL = 1e-6
STRICT_REFERENCE_DRIFT_TOL = 1e-3
REFERENCE_ABS_EXACT_TOL = 1e-9


def _method_tolerances() -> dict[str, float]:
    """Resolve MethodSpec.rmse_rel_tol without making the dashboard depend
    on a generated CSV column that older benchmark runs did not emit."""
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
    """Per-method gate-2 tolerance, preferring explicit run output."""
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


def _quality_from_tol(tol: float | None) -> str:
    """Tolerance-band classifier (kept aligned with registry._truth_source_quality)."""
    if tol is None:
        return "unknown"
    if tol <= 1e-6:
        return "strict"
    if tol < 1e-1:
        return "relaxed"
    return "qualitative"


def _strict_reference_code(rel: float | None,
                           abs_rmse: float | None = None) -> str | None:
    """Release-policy reference severity independent of MethodSpec tolerances."""
    if rel is None:
        return None
    if abs_rmse is not None and abs_rmse <= REFERENCE_ABS_EXACT_TOL:
        return "exact"
    if rel <= STRICT_REFERENCE_EXACT_TOL:
        return "exact"
    if rel <= STRICT_REFERENCE_DRIFT_TOL:
        return "drift"
    return "divergent"


def _format_divergence(value: float | None) -> str:
    if value is None or value != value:
        return ""
    if value == 0:
        return "0"
    av = abs(value)
    if av < 1e-3 or av >= 100:
        return f"{value:.0e}"
    if av < 0.1:
        return f"{value:.1e}"
    return f"{value:.2f}"


def _has_comparison_value(value: Any) -> bool:
    return _float_or_none(value) is not None


def divergence_fields(row: dict) -> dict:
    """Build per-cell divergence fields for the dashboard.

    Source rules:
      - rows with reference-gate metadata    → reference gate (vs MethodSpec ref)
      - legacy rows without ref metadata     → binding gate fallback (vs C++)
      - is_canonical_reference=True          → basis="self", no numeric value

    Returns a dict ready for cell merge:
        {"divergence": float|None, "divergence_fmt": str|None,
         "divergence_basis": "binding"|"reference"|"self",
         "divergence_quality": "strict"|"drift"|"divergent"|"binding"|"unknown"}
    Empty dict if the row can't supply a value.
    """
    kind = (row.get("kind") or "").strip().lower()
    # `divergence_status` carries the gate verdict for cells that have a
    # measured timing but no comparison number — so the dashboard can
    # render `✓ / ≈ / ✗ / ⊘` instead of a flat `—`. The CSV genuinely
    # lacks cross-library parity for non-canonical externals (e.g.
    # ikpls run as a fixed external on pls), so the gate verdict is
    # the most we can honestly say about those cells.
    parity_note = (row.get("reference_parity_note")
                    or row.get("binding_parity_note") or "").strip()
    if is_true(row.get("is_canonical_reference")):
        return {
            "divergence": None,
            "divergence_fmt": "source",
            "divergence_basis": "self",
            "divergence_quality": "self",
            "divergence_status": "source",
        }
    has_reference_gate = any(
        row.get(k) not in (None, "", "None")
        for k in (
            "reference_parity_ok",
            "reference_parity_rmse_rel",
            "reference_parity_note",
            "reference_kind",
        )
    )
    if kind in ("pls4all_binding", "pls4all_core", "external") and has_reference_gate:
        # The release view is reference-first: if a row has an executable
        # external comparison, show that number even for bindings. Gate 1
        # binding exactness remains available through `binding_parity`, but
        # it must not mask a gate-2 reference failure as numeric zero.
        ref_diff = _float_or_none(row.get("reference_parity_rmse_rel"))
        ref_abs = _float_or_none(row.get("reference_parity_rmse_abs"))
        ref_status = _strict_reference_code(ref_diff, ref_abs)
        if ref_status is not None:
            out = {
                "divergence": ref_diff,
                "divergence_fmt": _format_divergence(ref_diff),
                "divergence_basis": "reference",
                "divergence_quality": (
                    "strict" if ref_status == "exact" else ref_status
                ),
                "divergence_status": ref_status,
            }
            if parity_note:
                out["divergence_note"] = parity_note[:160]
            return out
        status = _diff_status(
            None,
            gate="reference",
            parity_ok=row.get("reference_parity_ok"),
            ok=is_true(row.get("ok")),
            note=parity_note,
        )
        out = {
            "divergence": None,
            "divergence_fmt": "—",
            "divergence_basis": "reference",
            "divergence_quality": "unknown",
            "divergence_status": status,
        }
        if parity_note:
            out["divergence_note"] = parity_note[:160]
        return out
    if kind == "pls4all_binding":
        diff = _float_or_none(row.get("binding_parity_max_diff")
                              or row.get("parity_max_diff"))
        status = _diff_status(
            diff,
            gate="binding",
            parity_ok=row.get("binding_parity_ok") or row.get("parity_ok"),
            note=parity_note,
        )
        out = {
            "divergence": diff,
            "divergence_fmt": _format_divergence(diff) if diff is not None else "—",
            "divergence_basis": "binding",
            "divergence_quality": "binding",
            "divergence_status": status,
        }
        if parity_note:
            out["divergence_note"] = parity_note[:160]
        return out
    if kind in ("pls4all_core", "external"):
        # rmse_rel only — the tooltip says "rmse_rel" so falling back to
        # rmse_abs (different units, very different magnitude scale)
        # would be a silent lie about what the number means.
        diff = _float_or_none(row.get("reference_parity_rmse_rel"))
        status = _diff_status(
            diff,
            gate="reference",
            parity_ok=row.get("reference_parity_ok"),
            ok=is_true(row.get("ok")),
            note=parity_note,
        )
        out = {
            "divergence": diff,
            "divergence_fmt": _format_divergence(diff) if diff is not None else "—",
            "divergence_basis": "reference",
            "divergence_quality": "unknown",
            "divergence_status": status,
        }
        if parity_note:
            out["divergence_note"] = parity_note[:160]
        return out
    return {}


def _diff_status(diff: float | None, *, gate: str,
                  parity_ok=None, ok: bool = True,
                  note: str = "") -> str:
    """Non-numeric fallback for the divergence cell.

    When `diff` is finite, the renderer prefers the numeric value. When
    it is None / NaN, this function returns a short string the renderer
    can map to a glyph:

      "exact"     → gate passed (✓)
      "drift"     → gate passed within drift band (≈)
      "divergent" → gate failed (✗)
      "no_compare"→ row ran (`ok=True`) but no comparison was recorded
                    against this gate (very common for non-canonical
                    externals — they have a timing but no parity)
      "not_run"   → row didn't run at all
    """
    if diff is not None and diff == diff:
        # numeric — caller will use the value, status is informational
        return "ok"
    if not ok:
        return "not_run"
    if is_true(parity_ok):
        return "exact"
    if parity_ok in (None, "", "None"):
        return "no_compare"
    note_l = (note or "").lower()
    if any(marker in note_l for marker in (
            "reference oracle missing",
            "not_applicable",
            "predictions missing",
            "missing",
    )):
        return "no_compare"
    if any(marker in note_l for marker in (
            "shape mismatch",
            "empty predictions",
    )):
        return "divergent"
    return "no_compare"


def parity_code(row: dict) -> str:
    """Legacy binding-consistency parity code (gate 1 only).

    exact | drift | divergent | error | deferred | not_run | not_available.

    Distinguishes hard runtime errors (ModuleNotFoundError, ImportError,
    crash) from libraries that do not expose an algorithm so the dashboard
    can show the right icon instead of collapsing every failure to a dash.
    Reads `binding_parity_*` first, falls back to legacy `parity_*`.
    """
    if not is_true(row.get("ok")):
        if (row.get("timing_statistic") or "").strip().lower() == "deferred":
            return "deferred"
        algo = row.get("algorithm") or row.get("algo") or ""
        backend = row.get("backend") or ""
        supported = FIXED_EXTERNAL_SUPPORT.get(backend)
        if supported is not None and algo not in supported:
            return "not_available"
        if backend == "r_tier2" and algo in SELECTOR_ALGOS:
            return "not_available"
        if backend == "matlab_tier2" and algo not in MATLAB_TIER2_SUPPORTED:
            return "not_available"
        reason = (row.get("reason") or "").lower()
        unavailable_markers = (
            "not implemented by",
            "not implemented",
            "notimplemented",
            "no sklearn estimator",
            "no formula wrapper",
            "no pls4all.fit classdef entry",
            "formula api limitation",
            "only accepts 1-d y",
            "unsupported algo",
            "unsupported algorithm",
            "unsupported method",
            "not available",
            "does not expose",
        )
        if any(k in reason for k in unavailable_markers):
            return "not_available"
        if "timeout" in reason:
            return "error"
        if any(k in reason for k in (
                "modulenotfounderror", "importerror",
                "error", "exception", "crash", "traceback", "segfault")):
            return "error"
        return "error"
    parity_ok = row.get("binding_parity_ok", row.get("parity_ok"))
    diff_val = row.get("binding_parity_max_diff",
                        row.get("parity_max_diff", "nan"))
    if parity_ok in (None, "", "None") and not _has_comparison_value(diff_val):
        return "not_available"
    if is_true(parity_ok):
        return "exact"
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
    """Gate-2 parity classification under the strict release policy.

    Values:
      exact          — rmse_abs ≤ 1e-9 or rmse_rel ≤ 1e-6.
      drift          — finite rmse_rel ≤ 1e-3 and not exact.
      divergent      — finite rmse_rel > 1e-3.
      not_available  — `reference_kind=paper_only` (no executable truth).
      deferred       — intentionally skipped in the CRAN-fast dashboard build;
                       catch-up benchmark jobs fill these cells later.
      not_run        — legacy value only; new builds classify missing
                       implementations as not_available and runtime
                       failures as error.
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
    rel = row.get("reference_parity_rmse_rel", "nan")
    abs_rmse = row.get("reference_parity_rmse_abs", "nan")
    note_l = (row.get("reference_parity_note") or "").lower()
    d = _float_or_none(rel)
    a = _float_or_none(abs_rmse)
    if d is None:
        if any(marker in note_l for marker in (
                "shape mismatch",
                "empty predictions",
        )):
            return "divergent"
        return "not_available"
    return _strict_reference_code(d, a) or "not_available"


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
    refresh_paths = sorted(results_dir.glob("dashboard_refresh_*.csv"))
    csv_paths = ([full_matrix] if full_matrix.exists() else []) + refresh_paths
    generated_at = dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    if csv_paths:
        latest_source = max(p.stat().st_mtime for p in csv_paths)
        generated_at = dt.datetime.fromtimestamp(
            latest_source, dt.timezone.utc
        ).strftime("%Y-%m-%d %H:%M UTC")
    for source_index, path in enumerate(csv_paths):
        source_mtime = path.stat().st_mtime
        for row in csv.DictReader(path.open()):
            row["_source_index"] = source_index
            row["_source_mtime"] = source_mtime
            rows_in.append(row)

    def _timing_schema(row: dict) -> str:
        try:
            versions = json.loads(row.get("versions_json") or "{}")
        except json.JSONDecodeError:
            return ""
        return str(versions.get("timing_schema") or "")

    def _schema_rank(schema: str) -> int:
        return {"adaptive-v1": 3, "warmup-v3": 2, "warmup-v2": 1}.get(schema, 0)

    def _row_rank(row: dict) -> tuple:
        return (
            _schema_rank(_timing_schema(row)),
            float(row.get("_source_mtime") or 0.0),
            int(row.get("_source_index") or 0),
        )

    # Dedupe: same (algo, backend, build, n, p, threads). Prefer the current
    # Prefer the current adaptive timing schema over legacy warmup/cold-run
    # CSV rows, then newest file.
    seen: dict[tuple, dict] = {}
    for r in rows_in:
        if not r.get("algorithm"):
            continue
        try:
            n, p_, t = int(r["n"]), int(r["p"]), int(r["threads"])
        except (ValueError, KeyError):
            continue
        key = (r["algorithm"], r["backend"], r.get("libp4a_build", ""), n, p_, t)
        if key not in seen or _row_rank(r) >= _row_rank(seen[key]):
            seen[key] = r

    # Non-C++ dashboard columns are unique even when the CSV contains one
    # row per libp4a build sweep. Prefer the production `blas-omp` row so
    # stale dev/blas/omp duplicate rows cannot keep an old gate verdict.
    preferred: dict[tuple, dict] = {}
    for r in seen.values():
        be = r.get("backend", "")
        if be == "cpp":
            key = (
                r["algorithm"], be, r.get("libp4a_build", ""),
                int(r["n"]), int(r["p"]), int(r["threads"]),
            )
        else:
            key = (
                r["algorithm"], be,
                int(r["n"]), int(r["p"]), int(r["threads"]),
            )
        old = preferred.get(key)
        if old is None:
            preferred[key] = r
            continue
        old_build = old.get("libp4a_build", "")
        new_build = r.get("libp4a_build", "")
        if new_build == "blas-omp" and old_build != "blas-omp":
            preferred[key] = r
    seen = {
        (r["algorithm"], r["backend"], r.get("libp4a_build", ""),
         int(r["n"]), int(r["p"]), int(r["threads"])): r
        for r in preferred.values()
    }

    # Collect columns actually present. Track the backend(s) that
    # contributed to each column id so the column metadata builder can
    # tell when a registry-driven `ref_*` row collapsed onto a legacy
    # column vs when it stands alone.
    present: set[str] = set()
    backends_per_cid: dict[str, set[str]] = defaultdict(set)
    for r in seen.values():
        cid = column_id(r["backend"], r.get("libp4a_build", ""))
        present.add(cid)
        backends_per_cid[cid].add(r["backend"])

    # Externally declared benchmark libraries. Infer the language
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
    # Standalone `ref_*` columns: registry-declared external libraries
    # that don't have a fixed legacy bench script. Find them by looking
    # for cids whose only contributing backend is a `ref_*` row AND that
    # cid isn't already registered above.
    already_registered = {c["id"] for c in raw_cols}
    for cid in sorted(present):
        if cid in already_registered:
            continue
        contributors = backends_per_cid.get(cid, set())
        # Skip "natural" columns that just happen to not be in present
        # for this run (shouldn't happen but defensive).
        if not contributors:
            continue
        ref_only = all(b.startswith("ref_") for b in contributors)
        if not ref_only:
            continue
        # Infer the language from the original `ref_<lang>_…` backend
        # name; the cid itself may now be a generic library name like
        # "OmicsPLS" so we can't rely on it.
        sample_backend = next(iter(contributors))
        body = sample_backend[len("ref_"):]
        lang = "external"
        group = "ext-py"
        for prefix, (l, g) in REF_LANG_PREFIX.items():
            if body.startswith(prefix + "_") or body == prefix:
                lang, group = l, g
                break
        raw_cols.append({
            "id": cid, "label": cid, "short": cid,
            "group": group, "tier": "external", "lang": lang,
            "what": f"External library benchmark backend ({lang}).",
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
            raw_ms = r.get("reported_ms") or r.get("median_ms")
            ms = float(raw_ms) if raw_ms else None
        except ValueError:
            ms = None
        ok = is_true(r.get("ok"))
        # `parity` keeps the legacy gate-1 verdict (binding consistency
        # vs cpp). `reference_parity` is the new gate-2 verdict (kernel
        # vs the method's canonical external reference). The dashboard
        # renders both icons per cell.
        cell = {"parity": verdict, "reference_parity": ref_verdict,
                "binding_parity": verdict, "ok": ok,
                "_source_is_ref": r["backend"].startswith("ref_"),
                "_rank": _row_rank(r)}
        # Divergence fields — feed the dashboard's "divergence δ" viz.
        # Source rule depends on `kind`: binding-gate for pls4all_binding,
        # reference-gate for pls4all_core/external, basis="self" for the
        # canonical reference row. See divergence_fields() docstring.
        cell.update(divergence_fields(r))
        if ms is not None and ms == ms:   # not NaN
            cell["ms"] = round(ms, 3)
            cell["fmt"] = fmt_ms(raw_ms)
        if r.get("timing_statistic"):
            cell["timing_statistic"] = r["timing_statistic"]
        if r.get("n_runs"):
            cell["n_runs"] = int(float(r["n_runs"]))
        if r.get("total_runs"):
            cell["total_runs"] = int(float(r["total_runs"]))
        if r.get("warmup_ms"):
            cell["warmup_ms"] = round(float(r["warmup_ms"]), 3)
        tol = reference_tolerance(r)
        if tol is not None:
            cell["reference_parity_tolerance"] = tol
        kind = (r.get("reference_kind") or "").strip()
        if kind:
            cell["reference_kind"] = kind
        if is_true(r.get("is_canonical_reference")):
            cell["is_canonical_reference"] = True
        # Capture the failure reason so the dashboard tooltip can show
        # "ModuleNotFoundError: foo" instead of a silent dash.
        reason = (r.get("reason") or "").strip()
        if reason and verdict in ("error", "deferred", "not_run", "not_available",
                                  "divergent", "drift"):
            cell["reason"] = reason[:200]
        # Keep every explicit verdict emitted by the orchestrator so the
        # dashboard can distinguish "not run" from "not available in lib".
        if ms is not None or verdict in (
                "deferred", "not_run", "not_available", "divergent", "drift", "error"):
            existing = pivot[rkey].get(cid)
            if existing is None:
                pivot[rkey][cid] = cell
            else:
                # `ref_*` row collided onto the same column as its
                # legacy equivalent (e.g. ref_python_scikit_learn →
                # sklearn). Merge: keep the legacy bench's timing (it's
                # what the user expects to see), but pull in the
                # canonical-reference flag + reference-parity verdict
                # from whichever row carries them. The legacy bench
                # script never sets is_canonical_reference (the
                # orchestrator's compute_parity sets it on the `ref_*`
                # row by design).
                incoming_is_ref = bool(cell.get("_source_is_ref"))
                existing_is_ref = bool(existing.get("_source_is_ref"))
                incoming_ok = bool(cell.get("ok"))
                existing_ok = bool(existing.get("ok"))
                incoming_rank = cell.get("_rank", (0, 0.0, 0))
                existing_rank = existing.get("_rank", (0, 0.0, 0))
                incoming_ref_comparison = (
                    incoming_is_ref
                    and incoming_ok
                    and cell.get("divergence_basis") in ("reference", "self")
                    and cell.get("divergence_status") not in (
                        "no_compare", "not_run")
                )

                # Timing is an atomic part of the selected source. Prefer
                # the legacy fixed-bench timing when both rows succeeded,
                # but do not keep an ok=false legacy cell if the canonical
                # ref_* alias actually ran.
                should_take_timing = (
                    cell.get("ms") is not None
                    and (
                        existing.get("ms") is None
                        or (incoming_ok and not existing_ok)
                        or (existing_is_ref and not incoming_is_ref and incoming_ok)
                        or (incoming_is_ref and existing_is_ref
                            and incoming_rank >= existing_rank)
                    )
                )
                if should_take_timing:
                    existing["ms"] = cell["ms"]
                    existing["fmt"] = cell["fmt"]

                should_take_status = incoming_ok and not existing_ok
                if should_take_status:
                    existing["ok"] = cell["ok"]
                    existing["parity"] = cell["parity"]
                    existing["binding_parity"] = cell["binding_parity"]
                    existing.pop("reason", None)
                if cell.get("is_canonical_reference"):
                    existing["is_canonical_reference"] = True
                    # The alias row is the actual canonical reference for
                    # this column. Its gate-2 verdict and status must win
                    # over a failed legacy row that collapsed to the same
                    # dashboard column.
                    if incoming_ok or not existing_ok:
                        existing["ok"] = cell["ok"]
                        existing["parity"] = cell["parity"]
                        existing["binding_parity"] = cell["binding_parity"]
                        if incoming_ok:
                            existing.pop("reason", None)
                    # When the ref_* alias collapses onto a legacy column
                    # (e.g. ref_python_scikit_learn → sklearn), the
                    # divergence value the user wants is the ref-gate
                    # self-comparison from the alias row, not the legacy
                    # row's binding diff. Only copy if the alias actually
                    # ran — a failed canonical ref must not falsely tag
                    # the merged cell as "source" (basis=self).
                    if incoming_ok:
                        for k in ("divergence", "divergence_fmt",
                                  "divergence_basis", "divergence_quality",
                                  "divergence_status", "divergence_note"):
                            if k in cell:
                                existing[k] = cell[k]
                # Even when the incoming ref_* alias is NOT the canonical
                # reference (e.g. `ref_python_ikpls` for `pls`, where the
                # canonical ref is sklearn), it may still carry a finite
                # `reference_parity_rmse_rel`. The legacy fixed bench
                # never computes that, so the existing cell starts with
                # `divergence=None`. Upgrade the merged cell to the
                # alias's number rather than leaving "—" for the user.
                elif incoming_ok and cell.get("divergence") is not None:
                    existing_div = existing.get("divergence")
                    existing_is_finite = (existing_div is not None
                                           and existing_div == existing_div)
                    if not existing_is_finite:
                        for k in ("divergence", "divergence_fmt",
                                  "divergence_basis", "divergence_quality",
                                  "divergence_status", "divergence_note"):
                            if k in cell:
                                existing[k] = cell[k]
                elif incoming_ref_comparison and not existing_is_ref:
                    # A fixed legacy external row (e.g. `r_pls`) may have
                    # timing only while its collapsed `ref_*` alias carries
                    # the actual reference-gate verdict. Keep the legacy
                    # timing, but let the ref alias own the gate metadata.
                    for k in ("divergence", "divergence_fmt",
                              "divergence_basis", "divergence_quality",
                              "divergence_status", "divergence_note"):
                        if k in cell:
                            existing[k] = cell[k]
                if cell.get("reference_kind") and not existing.get("reference_kind"):
                    existing["reference_kind"] = cell["reference_kind"]
                if cell.get("reference_parity_tolerance") is not None:
                    existing["reference_parity_tolerance"] = cell[
                        "reference_parity_tolerance"]
                # Prefer the gate-2 verdict from whichever side actually
                # computed it. If the incoming row is the canonical
                # reference alias itself, its self-comparison verdict is
                # the correct global gate-2 verdict for that column.
                if (cell.get("is_canonical_reference")
                        or incoming_ref_comparison
                        or existing.get("reference_parity") in (None, "self", "not_available")):
                    existing["reference_parity"] = cell["reference_parity"]
                if reason and not existing.get("ok"):
                    existing["reason"] = reason[:200]

    # Synthetic leading "Reference" column: for each row, find the cell
    # marked `is_canonical_reference=True` (set on the canonical
    # external lib's row by the orchestrator's compute_parity) and
    # mirror it into a special `__reference` cell. The renderer pins
    # this column to the front of the table and colours it by the
    # canonical lib's language. If no canonical reference is wired
    # (paper_only methods), the cell stays empty.
    REF_COL_ID = "__reference"
    columns_by_id = {c["id"]: c for c in raw_cols}
    for (algo, n, p_, t), cells in pivot.items():
        canonical_cid = None
        for cid, c in cells.items():
            if c.get("is_canonical_reference"):
                canonical_cid = cid
                break
        if canonical_cid is None:
            continue
        src = cells[canonical_cid]
        col_meta = columns_by_id.get(canonical_cid, {})
        mirror = {
            "parity": src.get("parity"),
            "reference_parity": src.get("reference_parity"),
            "ok": src.get("ok"),
            "ref_of": col_meta.get("short") or col_meta.get("label") or canonical_cid,
            "ref_lang": col_meta.get("lang", "external"),
            "is_canonical_reference": True,
        }
        if "ms" in src:
            mirror["ms"] = src["ms"]
            mirror["fmt"] = src["fmt"]
        if "reference_kind" in src:
            mirror["reference_kind"] = src["reference_kind"]
        # Carry divergence info onto the synthetic reference column so the
        # divergence-δ viz can render "source" instead of "—".
        for k in ("divergence", "divergence_fmt",
                  "divergence_basis", "divergence_quality",
                  "divergence_status", "divergence_note"):
            if k in src:
                mirror[k] = src[k]
        cells[REF_COL_ID] = mirror

    for cells in pivot.values():
        for c in cells.values():
            c.pop("_source_is_ref", None)
            c.pop("_rank", None)

    rows_out = []
    for (algo, n, p_, t), cells in sorted(pivot.items(),
                                            key=lambda x: (x[0][0],
                                                            x[0][1] * x[0][2],
                                                            x[0][3])):
        rows_out.append({
            "algo": algo, "n": n, "p": p_, "threads": t,
            "cells": cells,
        })

    # Register the synthetic Reference column at the very front so the
    # renderer pins it before any backend column. `kind="reference"`
    # tells the JS layer to apply the special per-row language-tint +
    # leading-column styling. Prepended to `columns` (after the sort)
    # rather than to `raw_cols` because language-rank sorting would
    # otherwise scatter it back into the C++ band.
    if any(REF_COL_ID in r["cells"] for r in rows_out):
        columns = [{
            "id": REF_COL_ID,
            "label": "canonical reference",
            "short": "reference",
            "group": "reference",
            "tier": "per-method canonical reference",
            "lang": "external",   # per-row override via cell.ref_lang
            "what": "The library that defines the ground truth for each "
                    "method (sklearn for `pls`/`pcr`, ropls for `opls`, "
                    "mixOmics for `*_da`, …). The cell value mirrors the "
                    "canonical library's timing for that row and is "
                    "language-tinted accordingly.",
            "build": "",
            "kind": "reference",
        }] + columns

    # Per-column versions (best available).
    versions: dict[str, str] = {}
    version_rows: dict[str, dict] = {}
    for r in seen.values():
        cid = column_id(r["backend"], r.get("libp4a_build", ""))
        old = version_rows.get(cid)
        if old is not None and _row_rank(old) > _row_rank(r):
            continue
        version_rows[cid] = r
    for cid, r in version_rows.items():
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
    selectable_cols = [c for c in all_cols if c != REF_COL_ID]
    cpp_blas_omp = "pls4all.cpp.blas+omp" if "pls4all.cpp.blas+omp" in all_cols else None
    cpp_native = "pls4all.cpp.native" if "pls4all.cpp.native" in all_cols else None

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
    pls4all_only = [c for c in selectable_cols if c.startswith("pls4all.")]
    externals = [c for c in selectable_cols if not c.startswith("pls4all.")]
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
        "sklearn", "ikpls", "nirs4all",
        "ref.python_scikit_learn", "ref.python_ikpls",
        # R: pls4all formula + S3 binding vs canonical R externals
        "pls4all.R.formula",
        "pls4all.R.pls_compat", "pls4all.R.mdatools_compat",
        "pls", "mixOmics", "ropls",
        "ref.r_pls", "ref.r_mixomics", "ref.r_ropls",
        # MATLAB: pls4all classdef binding vs the stats-toolbox baseline
        "pls4all.matlab.classdef",
        "plsregress", "libPLS",
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
        "pls4all.sklearn", "pls4all.R.formula",
        "pls4all.R.pls_compat", "pls4all.R.mdatools_compat",
        "pls4all.matlab.classdef"
    ) if c in all_cols]
    # registry/bench-ref is the single canonical pls4all invocation
    # used by the benchmark suite as parity baseline; expose it as its
    # own preset so users can see only what's parity-anchored.
    bench_ref = [c for c in ("pls4all.registry",) if c in all_cols]

    # Per-language sets (pls4all bindings + externals for that host language)
    py_all = by_lang("Python")
    r_all  = by_lang("R")
    m_all  = by_lang("MATLAB/Octave")

    # Native baseline tiers — compare the native scalar build
    # against the production BLAS+OpenMP one, plus the registry entry
    # point. This is the "how much speed do the C++ optimisations buy?"
    # view, which is what the benchmark is really *for*.
    native_baseline = [c for c in (
        cpp_native, cpp_blas_omp, "pls4all.registry",
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
        "all":          {"label": "All",            "cols": selectable_cols},
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

    columns_by_id = {c["id"]: c for c in columns}

    def _cell_effective_parity(cid: str, cell: dict) -> str | None:
        if cid == REF_COL_ID:
            return None
        column = columns_by_id.get(cid, {})
        kind = (column.get("kind") or "").lower()
        column_id = column.get("id") or cid
        if kind in {"external", "reference"} or column_id.startswith("pls4all.cpp."):
            return cell.get("reference_parity") or cell.get("parity")
        return cell.get("binding_parity") or cell.get("parity")

    timed_cells = [
        (cid, c)
        for r in rows_out
        for cid, c in r["cells"].items()
        if cid != REF_COL_ID
        and c.get("ok")
        and isinstance(c.get("ms"), (int, float))
    ]

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
            "cells":    len(timed_cells),
            "ok":       sum(1 for cid, c in timed_cells
                            if _cell_effective_parity(cid, c) == "exact"),
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
