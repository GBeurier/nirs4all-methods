"""Parity gate runner.

For each method in `registry.METHODS`, runs the configured cell once and
compares pls4all's predictions to (a) a Python reference and (b) an R
reference, when available. When no external reference exists, the method
is gated against a NumPy mirror and explicitly flagged in the output so
the README can highlight the limitation.

Outputs:
- benchmarks/results/parity_gate/parity.csv  — one row per (method, ref)
- benchmarks/results/parity_gate/summary.md  — Markdown table for README

Timing benchmarks are tracked separately by benchmarks/runners/matrix.py
and were deliberately excluded from this runner per the user spec (CPU
load).
"""

from __future__ import annotations

import argparse
import csv
import math
import platform
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[2]
BINDING_SRC = REPO_ROOT / "bindings" / "python" / "src"
sys.path.insert(0, str(BINDING_SRC))

import pls4all  # noqa: E402

from .registry import (  # noqa: E402
    METHODS,
    MethodSpec,
    iter_reference_factories,
    reference_id,
)


@dataclass
class ParityRow:
    method: str
    description: str
    reference_lang: str
    reference_lib: str
    reference_version: str
    reference_notes: str
    # Stable cross-binding column id for this reference (`ref.<id>`).
    # `None` for rows that never instantiate an adapter — paper-only,
    # `no_python_reference`, `no_r_reference`, `reference_unavailable`,
    # or `reference_error`. Downstream tooling (docs, lockfile) joins on
    # this id rather than the more brittle `reference_lib`.
    reference_id: str | None
    n_samples: int
    n_features: int
    params: dict
    parity_pass: bool
    rmse_abs: float
    rmse_rel: float
    tolerance: float
    status: str


def _make_dataset(n: int, p: int, seed: int = 11, n_targets: int = 1,
                  n_classes: int = 0
                   ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray,
                              np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    k = min(5, n - 1, p)
    T = rng.standard_normal((n, k))
    W = rng.standard_normal((p, k))
    X = T @ W.T + 0.05 * rng.standard_normal((n, p))
    B = rng.standard_normal((k, n_targets))
    Y = T @ B + 0.02 * rng.standard_normal((n, n_targets))
    if n_targets == 1:
        Y = Y.reshape(-1, 1)
    X_target = X.copy()
    X_target[:, 0] *= 1.1
    X_target[:, 1] += 0.05
    # Deterministic positive sample weights for §26 weighted-PLS variants.
    sample_weights = rng.uniform(0.5, 2.0, size=n)
    # Deterministic class labels for §7 sparse PLS-DA.
    if n_classes >= 2:
        labels = rng.integers(0, n_classes, size=n).astype(np.int32)
    else:
        labels = np.zeros(n, dtype=np.int32)
    # Deterministic group assignment for §7 group / fused sparse PLS.
    # Group features into n_groups consecutive blocks.
    n_groups = max(1, min(5, p // 3))
    group_assignment = (np.arange(p) // max(1, p // n_groups)).astype(np.int32)
    group_assignment = np.minimum(group_assignment, n_groups - 1)
    return X, Y, X_target, sample_weights, labels, group_assignment


def _rmse(a: np.ndarray, b: np.ndarray) -> float:
    diff = a - b
    return float(math.sqrt(float(np.mean(diff * diff))))


def _resolve_extras(method: MethodSpec, X_target: np.ndarray,
                    sample_weights: np.ndarray,
                    labels: np.ndarray,
                    group_assignment: np.ndarray) -> dict:
    extras: dict = {}
    if method.needs_x_target:
        extras["X_target"] = X_target
    if method.needs_sample_weights:
        extras["sample_weights"] = sample_weights
    if method.needs_labels:
        extras["y_labels"] = labels
    if method.needs_group_assignment:
        extras["group_assignment"] = group_assignment
    return extras


def _run_one_reference(method: MethodSpec, factory, X, Y, X_target,
                        sample_weights, labels, group_assignment,
                        pls4all_pred: np.ndarray | None) -> ParityRow:
    extras = _resolve_extras(method, X_target, sample_weights, labels,
                              group_assignment)
    n_samples = method.cell_params["n_samples"]
    n_features = method.cell_params["n_features"]
    try:
        reference = factory(**method.cell_params)
        if reference is None:
            return ParityRow(
                method=method.name, description=method.description,
                reference_lang="?", reference_lib="(unavailable)",
                reference_version="-",
                reference_notes=method.notes,
                reference_id=None,
                n_samples=n_samples, n_features=n_features,
                params=method.cell_params,
                parity_pass=False,
                rmse_abs=float("nan"), rmse_rel=float("nan"),
                tolerance=method.rmse_rel_tol,
                status="reference_unavailable",
            )
        reference.fit(X, Y, **extras)
        ref_pred = reference.predict(X)
    except Exception as exc:
        return ParityRow(
            method=method.name, description=method.description,
            reference_lang="?", reference_lib="?",
            reference_version="?",
            reference_notes=method.notes,
            reference_id=None,
            n_samples=n_samples, n_features=n_features,
            params=method.cell_params,
            parity_pass=False,
            rmse_abs=float("nan"), rmse_rel=float("nan"),
            tolerance=method.rmse_rel_tol,
            status=f"reference_error:{type(exc).__name__}:{exc}",
        )

    rid = reference_id(reference.library_name, reference.language)

    if pls4all_pred is None:
        return ParityRow(
            method=method.name, description=method.description,
            reference_lang=reference.language,
            reference_lib=reference.library_name,
            reference_version=reference.library_version,
            reference_notes=reference.notes or method.notes,
            reference_id=rid,
            n_samples=n_samples, n_features=n_features,
            params=method.cell_params,
            parity_pass=False,
            rmse_abs=float("nan"), rmse_rel=float("nan"),
            tolerance=method.rmse_rel_tol,
            status="pls4all_failed",
        )

    if ref_pred.shape != pls4all_pred.shape:
        return ParityRow(
            method=method.name, description=method.description,
            reference_lang=reference.language,
            reference_lib=reference.library_name,
            reference_version=reference.library_version,
            reference_notes=reference.notes or method.notes,
            reference_id=rid,
            n_samples=n_samples, n_features=n_features,
            params=method.cell_params,
            parity_pass=False,
            rmse_abs=float("nan"), rmse_rel=float("nan"),
            tolerance=method.rmse_rel_tol,
            status=(f"shape_mismatch:{pls4all_pred.shape}_vs_"
                    f"{ref_pred.shape}"),
        )

    rmse_abs = _rmse(pls4all_pred, ref_pred)
    ref_rmse = math.sqrt(float(np.mean(ref_pred * ref_pred))) or 1.0
    rmse_rel = rmse_abs / ref_rmse
    parity_pass = bool(rmse_rel <= method.rmse_rel_tol or rmse_abs <= 1e-9)
    return ParityRow(
        method=method.name, description=method.description,
        reference_lang=reference.language,
        reference_lib=reference.library_name,
        reference_version=reference.library_version,
        reference_notes=reference.notes or method.notes,
        reference_id=rid,
        n_samples=n_samples, n_features=n_features,
        params=method.cell_params,
        parity_pass=parity_pass,
        rmse_abs=rmse_abs, rmse_rel=rmse_rel,
        tolerance=method.rmse_rel_tol,
        status="ok" if parity_pass else "parity_failed",
    )


def _run_method(method: MethodSpec) -> list[ParityRow]:
    n = method.cell_params["n_samples"]
    p = method.cell_params["n_features"]

    # Paper-only methods have no widely installable external reference and
    # are documented with their canonical citation. We still smoke-fit the
    # pls4all side to confirm the entry point is callable.
    if method.paper_only:
        n_targets = int(method.cell_params.get("n_targets", 1))
        n_classes = int(method.cell_params.get("n_classes", 0))
        X, Y, X_target, sample_weights, labels, groups = _make_dataset(
            n, p, n_targets=n_targets, n_classes=n_classes)
        extras = _resolve_extras(method, X_target, sample_weights, labels,
                                  groups)
        pls4all_ok = True
        try:
            with pls4all.Context() as ctx, pls4all.Config() as cfg:
                result = method.pls4all_fn(
                    ctx, cfg, X, Y, **method.cell_params, **extras)
                result.matrix(method.prediction_key)
                result.close()
        except Exception as exc:
            print(f"  [{method.name}] pls4all_error: {exc}",
                  file=sys.stderr)
            pls4all_ok = False
        return [ParityRow(
            method=method.name, description=method.description,
            reference_lang="paper", reference_lib="paper-only",
            reference_version="-",
            reference_notes=method.paper_only,
            reference_id=None,
            n_samples=n, n_features=p, params=method.cell_params,
            parity_pass=pls4all_ok,
            rmse_abs=float("nan"), rmse_rel=float("nan"),
            tolerance=method.rmse_rel_tol,
            status="paper_only" if pls4all_ok else "pls4all_failed",
        )]

    n_targets = int(method.cell_params.get("n_targets", 1))
    n_classes = int(method.cell_params.get("n_classes", 0))
    X, Y, X_target, sample_weights, labels, groups = _make_dataset(
        n, p, n_targets=n_targets, n_classes=n_classes)
    extras = _resolve_extras(method, X_target, sample_weights, labels,
                              groups)

    pls4all_pred: np.ndarray | None = None
    try:
        with pls4all.Context() as ctx, pls4all.Config() as cfg:
            result = method.pls4all_fn(ctx, cfg, X, Y, **method.cell_params,
                                        **extras)
            pls4all_pred = result.matrix(method.prediction_key)
            result.close()
    except Exception as exc:
        print(f"  [{method.name}] pls4all_error: {exc}", file=sys.stderr)

    rows: list[ParityRow] = []
    factories = list(iter_reference_factories(method))
    has_python = method.python_reference is not None
    has_r = method.r_reference is not None

    if not has_python:
        rows.append(ParityRow(
            method=method.name, description=method.description,
            reference_lang="python", reference_lib="(none)",
            reference_version="-",
            reference_notes="No Python reference registered for this method.",
            reference_id=None,
            n_samples=n, n_features=p, params=method.cell_params,
            parity_pass=False, rmse_abs=float("nan"), rmse_rel=float("nan"),
            tolerance=method.rmse_rel_tol,
            status="no_python_reference",
        ))
    elif method.python_reference is not None:
        rows.append(_run_one_reference(method, method.python_reference,
                                        X, Y, X_target, sample_weights,
                                        labels, groups, pls4all_pred))

    if not has_r:
        rows.append(ParityRow(
            method=method.name, description=method.description,
            reference_lang="R", reference_lib="(none)",
            reference_version="-",
            reference_notes=(method.notes or
                             "No R reference available for this method."),
            reference_id=None,
            n_samples=n, n_features=p, params=method.cell_params,
            parity_pass=False, rmse_abs=float("nan"), rmse_rel=float("nan"),
            tolerance=method.rmse_rel_tol,
            status="no_r_reference",
        ))
    elif method.r_reference is not None:
        rows.append(_run_one_reference(method, method.r_reference,
                                        X, Y, X_target, sample_weights,
                                        labels, groups, pls4all_pred))

    for role, factory in factories:
        if ((role == "python" and factory is method.python_reference) or
                (role == "r" and factory is method.r_reference)):
            continue
        rows.append(_run_one_reference(method, factory,
                                        X, Y, X_target, sample_weights,
                                        labels, groups, pls4all_pred))

    return rows


def run(methods: list[str] | None = None) -> list[ParityRow]:
    selected = [m for m in METHODS if methods is None or m.name in methods]
    rows: list[ParityRow] = []
    for method in selected:
        print(f"=== {method.name}: {method.description} ===")
        for row in _run_method(method):
            tag = "PASS" if row.parity_pass else "FAIL"
            print(f"  {tag} [{row.reference_lang}/{row.reference_lib}] "
                  f"rmse_rel={row.rmse_rel:.3e} status={row.status}")
            rows.append(row)
    return rows


def _write_csv(path: Path, rows: list[ParityRow]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "method", "description",
        "reference_lang", "reference_lib", "reference_version",
        "reference_id",
        "n_samples", "n_features", "params",
        "parity_pass", "rmse_abs", "rmse_rel", "tolerance",
        "status", "reference_notes",
    ]
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({
                "method": row.method,
                "description": row.description,
                "reference_lang": row.reference_lang,
                "reference_lib": row.reference_lib,
                "reference_version": row.reference_version,
                "reference_id": row.reference_id or "",
                "n_samples": row.n_samples,
                "n_features": row.n_features,
                "params": str(row.params),
                "parity_pass": row.parity_pass,
                "rmse_abs": (f"{row.rmse_abs:.6e}"
                             if math.isfinite(row.rmse_abs) else "nan"),
                "rmse_rel": (f"{row.rmse_rel:.6e}"
                             if math.isfinite(row.rmse_rel) else "nan"),
                "tolerance": f"{row.tolerance:.3e}",
                "status": row.status,
                "reference_notes": row.reference_notes.replace("\n", " "),
            })


def _parity_quality(row: ParityRow) -> str:
    """Classify the strength of a passing parity row:
      - "tight"    : rmse_rel < 0.3 * tolerance (high confidence)
      - "moderate" : 0.3-0.6 of tolerance (real agreement)
      - "loose"    : 0.6-0.9 of tolerance (algorithmic divergence likely)
      - "weak"     : >= 0.9 * tolerance (passes barely; widened tolerance —
                     usually stochastic-RNG divergence or algorithmic
                     convention mismatch; treat as smoke check, not bit
                     parity)
      - "—"        : non-passing / non-ok statuses
    """
    if row.status != "ok" or not row.parity_pass:
        return "—"
    if not (math.isfinite(row.rmse_rel) and math.isfinite(row.tolerance)
            and row.tolerance > 0):
        return "—"
    ratio = row.rmse_rel / row.tolerance
    if ratio < 0.3:
        return "tight"
    if ratio < 0.6:
        return "moderate"
    if ratio < 0.9:
        return "loose"
    return "weak"


def _write_summary(path: Path, rows: list[ParityRow]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    by_method: dict[str, list[ParityRow]] = {}
    for row in rows:
        by_method.setdefault(row.method, []).append(row)
    # Count parity-quality buckets across all `ok` rows.
    qualities = [(r.method, r.reference_lib, _parity_quality(r),
                  r.rmse_rel, r.tolerance)
                 for r in rows if r.status == "ok"]
    weak = [(m, lib, rr, tol) for (m, lib, q, rr, tol) in qualities
            if q == "weak"]
    by_quality: dict[str, int] = {}
    for _, _, q, _, _ in qualities:
        by_quality[q] = by_quality.get(q, 0) + 1
    lines = [
        "# Parity gate report",
        "",
        f"Host: `{platform.platform()}`",
        f"pls4all: `{pls4all.version()}`",
        f"Python: `{sys.version.split()[0]}`",
        f"NumPy: `{np.__version__}`",
        "",
        "## Pass quality breakdown",
        "",
        ("| Quality | Count | Definition |"),
        ("|---------|------:|------------|"),
        (f"| tight    | {by_quality.get('tight', 0)} | "
         "`rmse_rel < 30% of tolerance` — high-confidence parity. |"),
        (f"| moderate | {by_quality.get('moderate', 0)} | "
         "30-60% of tolerance — real agreement. |"),
        (f"| loose    | {by_quality.get('loose', 0)} | "
         "60-90% of tolerance — algorithmic divergence likely; passes with margin. |"),
        (f"| **weak** | **{by_quality.get('weak', 0)}** | "
         "**>= 90% of tolerance** — passes barely; tolerance widened to "
         "accept stochastic-RNG or algorithmic-convention divergence. "
         "**Treat as smoke check, not bit parity.** |"),
        "",
    ]
    if weak:
        lines += [
            "### Weak-parity passes (read with caution)",
            "",
            ("These pass the gate but rely on widened tolerances. Their "
             "external reference and the pls4all implementation use the "
             "same algorithm family but differ in RNG, convention or "
             "parameter mapping; do not read a green check as bit-exact "
             "agreement."),
            "",
            ("| Method | Reference | rmse_rel | tolerance | margin |"),
            ("|--------|-----------|---------:|----------:|-------:|"),
        ]
        for m, lib, rr, tol in weak:
            margin = (1.0 - rr / tol) * 100.0 if tol > 0 else 0.0
            lines.append(
                f"| `{m}` | `{lib}` | {rr:.3e} | {tol:.2e} | {margin:.1f}% |"
            )
        lines.append("")
    lines += [
        "## All rows",
        "",
        "Each method is compared against a Python reference and an R "
        "reference. Methods without a widely installable external "
        "reference are flagged `paper-only` or `(none)` in the lib column.",
        "",
        ("| Method | Description | Reference (lang / lib / version) | "
         "Parity | Quality | RMSE rel | Tolerance | Status |"),
        ("|--------|-------------|----------------------------------|"
         "--------|---------|----------|-----------|--------|"),
    ]
    for method_name, method_rows in by_method.items():
        for row in method_rows:
            parity = "✓" if row.parity_pass else "✗"
            rmse = (f"{row.rmse_rel:.3e}"
                    if math.isfinite(row.rmse_rel) else "—")
            ref = (f"{row.reference_lang} / `{row.reference_lib}` "
                   f"{row.reference_version}")
            quality = _parity_quality(row)
            lines.append(
                f"| `{row.method}` | {row.description} | {ref} | "
                f"{parity} | {quality} | {rmse} | {row.tolerance:.0e} | "
                f"{row.status} |"
            )
    note_rows = [row for row in rows if row.reference_notes
                  and row.reference_lib == "numpy-mirror"]
    if note_rows:
        lines += ["", "## Reference caveats", ""]
        seen = set()
        for row in note_rows:
            key = (row.method, row.reference_lib)
            if key in seen:
                continue
            seen.add(key)
            lines.append(f"- **`{row.method}`** "
                          f"({row.reference_lang} / `{row.reference_lib}`): "
                          f"{row.reference_notes}")
    lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


@dataclass
class TimingRow:
    method: str
    n_samples: int
    n_features: int
    pls4all_ms: float  # binding wall-clock, median of `repeats` runs
    pls4all_min_ms: float
    python_ref_ms: float
    python_ref_lib: str
    r_ref_ms: float
    r_ref_lib: str
    repeats: int
    status: str  # "ok" / "no_python_reference" / "no_r_reference" / "error:<...>"


def _measure_wall_ms(fn, repeats: int) -> tuple[float, float, str]:
    """Run `fn` `repeats` times; return (median_ms, min_ms, status).
    A single exception is recorded once and returned as (nan, nan, 'error:...').
    """
    import time
    samples_ms: list[float] = []
    try:
        for _ in range(repeats):
            t0 = time.perf_counter()
            fn()
            samples_ms.append((time.perf_counter() - t0) * 1000.0)
    except Exception as exc:
        return (float("nan"), float("nan"),
                f"error:{type(exc).__name__}:{str(exc)[:80]}")
    s = sorted(samples_ms)
    median = s[len(s) // 2]
    return (median, s[0], "ok")


def time_methods(methods: list[str] | None = None,
                  repeats: int = 3,
                  *, inproc_r: bool = True) -> list[TimingRow]:
    """For each MethodSpec, time three modes:
      1. pls4all binding call (`pls4all_fn`)
      2. Python reference fit+predict
      3. R reference fit+predict
    Returns one row per method. Skips paper_only methods (no external timing).

    `inproc_r`: when True, routes R adapters through rpy2 in-process via
    `RAdapter.use_inproc(True)` so the per-call Rscript subprocess
    startup (~1 s on this host) does not dominate the wall-clock. Falls
    back to subprocess automatically if rpy2 isn't importable.
    """
    from .registry import RAdapter, _ensure_inproc_r
    if inproc_r:
        if _ensure_inproc_r() is not None:
            RAdapter.use_inproc(True)
            print("[timing] R adapters running INPROC via rpy2")
        else:
            print("[timing] rpy2 unavailable — R adapters stay on Rscript subprocess",
                  file=sys.stderr)
    selected = [m for m in METHODS if methods is None or m.name in methods]
    rows: list[TimingRow] = []
    for method in selected:
        if method.paper_only:
            continue  # no external to compare against
        n = method.cell_params["n_samples"]
        p = method.cell_params["n_features"]
        n_targets = int(method.cell_params.get("n_targets", 1))
        n_classes = int(method.cell_params.get("n_classes", 0))
        X, Y, X_target, sw, lbl, grp = _make_dataset(
            n, p, n_targets=n_targets, n_classes=n_classes)
        extras = _resolve_extras(method, X_target, sw, lbl, grp)

        # --- 1. pls4all binding -----------------------------------------
        def _run_pls4all():
            with pls4all.Context() as ctx, pls4all.Config() as cfg:
                result = method.pls4all_fn(
                    ctx, cfg, X, Y, **method.cell_params, **extras)
                result.matrix(method.prediction_key)
                result.close()

        n4m_median, n4m_min, n4m_status = _measure_wall_ms(_run_pls4all, repeats)

        # --- 2. Python reference ----------------------------------------
        py_ms, py_lib = float("nan"), "(none)"
        if method.python_reference is not None:
            try:
                ref = method.python_reference(**method.cell_params)
                py_lib = ref.library_name

                def _run_py():
                    ref.fit(X, Y, **extras)
                    ref.predict(X)

                py_ms, _, py_status = _measure_wall_ms(_run_py, repeats)
                if py_status != "ok":
                    py_ms = float("nan")
            except Exception:
                py_ms = float("nan")
                py_lib = "(error)"

        # --- 3. R reference ---------------------------------------------
        r_ms, r_lib = float("nan"), "(none)"
        if method.r_reference is not None:
            try:
                ref = method.r_reference(**method.cell_params)
                r_lib = ref.library_name

                def _run_r():
                    ref.fit(X, Y, **extras)
                    ref.predict(X)

                r_ms, _, r_status = _measure_wall_ms(_run_r, repeats)
                if r_status != "ok":
                    r_ms = float("nan")
            except Exception:
                r_ms = float("nan")
                r_lib = "(error)"

        # Extra references are reported through the parity CSV. The timing
        # CSV intentionally preserves the legacy one-Python/one-R schema.

        row = TimingRow(
            method=method.name, n_samples=n, n_features=p,
            pls4all_ms=n4m_median, pls4all_min_ms=n4m_min,
            python_ref_ms=py_ms, python_ref_lib=py_lib,
            r_ref_ms=r_ms, r_ref_lib=r_lib,
            repeats=repeats, status=n4m_status,
        )
        rows.append(row)
        print(f"=== {method.name}: pls4all={n4m_median:.1f}ms "
              f"py[{py_lib}]={py_ms:.1f}ms r[{r_lib}]={r_ms:.1f}ms ===")
    return rows


def _write_timings_csv(path: Path, rows: list[TimingRow]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["method", "n_samples", "n_features",
                  "pls4all_ms", "pls4all_min_ms",
                  "python_ref_ms", "python_ref_lib",
                  "r_ref_ms", "r_ref_lib",
                  "binding_overhead_vs_py_pct",
                  "binding_overhead_vs_r_pct",
                  "repeats", "status"]
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for r in rows:
            py_pct = ((r.pls4all_ms / r.python_ref_ms - 1.0) * 100.0
                       if math.isfinite(r.pls4all_ms)
                       and math.isfinite(r.python_ref_ms)
                       and r.python_ref_ms > 0 else float("nan"))
            r_pct = ((r.pls4all_ms / r.r_ref_ms - 1.0) * 100.0
                      if math.isfinite(r.pls4all_ms)
                      and math.isfinite(r.r_ref_ms)
                      and r.r_ref_ms > 0 else float("nan"))
            writer.writerow({
                "method": r.method,
                "n_samples": r.n_samples,
                "n_features": r.n_features,
                "pls4all_ms": (f"{r.pls4all_ms:.3f}"
                                if math.isfinite(r.pls4all_ms) else "nan"),
                "pls4all_min_ms": (f"{r.pls4all_min_ms:.3f}"
                                    if math.isfinite(r.pls4all_min_ms) else "nan"),
                "python_ref_ms": (f"{r.python_ref_ms:.3f}"
                                   if math.isfinite(r.python_ref_ms) else "nan"),
                "python_ref_lib": r.python_ref_lib,
                "r_ref_ms": (f"{r.r_ref_ms:.3f}"
                              if math.isfinite(r.r_ref_ms) else "nan"),
                "r_ref_lib": r.r_ref_lib,
                "binding_overhead_vs_py_pct": (f"{py_pct:+.1f}"
                                                if math.isfinite(py_pct) else "nan"),
                "binding_overhead_vs_r_pct": (f"{r_pct:+.1f}"
                                               if math.isfinite(r_pct) else "nan"),
                "repeats": r.repeats,
                "status": r.status,
            })


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="parity-gate", description=__doc__)
    parser.add_argument("--methods", nargs="*", default=None,
                        help="Subset of method names (default: all).")
    parser.add_argument("--output-dir", type=Path,
                        default=REPO_ROOT / "benchmarks" / "results" /
                        "parity_gate",
                        help="Output directory for CSV + summary.")
    parser.add_argument("--with-timings", action="store_true",
                        help="Also collect wall-clock timings for "
                             "pls4all binding, Python ref and R ref. "
                             "Writes timings.csv to output-dir.")
    parser.add_argument("--timing-repeats", type=int, default=3,
                        help="Per-mode repeat count for --with-timings "
                             "(median of N runs).")
    parser.add_argument("--r-subprocess", action="store_true",
                        help="Force Rscript subprocess for timings even "
                             "when rpy2 is available. Default: inproc R "
                             "via rpy2 (fairer; avoids the per-call "
                             "Rscript startup overhead).")
    args = parser.parse_args(argv)
    rows = run(methods=args.methods)
    _write_csv(args.output_dir / "parity.csv", rows)
    _write_summary(args.output_dir / "summary.md", rows)
    if args.with_timings:
        print("\n--- timing pass ---")
        trows = time_methods(methods=args.methods,
                              repeats=args.timing_repeats,
                              inproc_r=not args.r_subprocess)
        _write_timings_csv(args.output_dir / "timings.csv", trows)
        print(f"\nWrote {args.output_dir / 'timings.csv'} ({len(trows)} rows)")
    print(f"\nWrote {args.output_dir / 'parity.csv'} ({len(rows)} rows)")
    print(f"Wrote {args.output_dir / 'summary.md'}")
    failures = [r for r in rows if not r.parity_pass and
                r.status not in {"no_r_reference", "no_python_reference",
                                  "paper_only", "reference_unavailable"}]
    if failures:
        print(f"\n{len(failures)} parity failure(s):", file=sys.stderr)
        for f in failures:
            print(f"  {f.method} [{f.reference_lang}/{f.reference_lib}]: "
                  f"{f.status}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
