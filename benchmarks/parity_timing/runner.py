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

from .registry import METHODS, MethodSpec  # noqa: E402


@dataclass
class ParityRow:
    method: str
    description: str
    reference_lang: str
    reference_lib: str
    reference_version: str
    reference_notes: str
    n_samples: int
    n_features: int
    params: dict
    parity_pass: bool
    rmse_abs: float
    rmse_rel: float
    tolerance: float
    status: str


def _make_dataset(n: int, p: int, seed: int = 11
                   ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    k = min(5, n - 1, p)
    T = rng.standard_normal((n, k))
    W = rng.standard_normal((p, k))
    X = T @ W.T + 0.05 * rng.standard_normal((n, p))
    b = rng.standard_normal(k)
    y = T @ b + 0.02 * rng.standard_normal(n)
    Y = y.reshape(-1, 1)
    X_target = X.copy()
    X_target[:, 0] *= 1.1
    X_target[:, 1] += 0.05
    return X, Y, X_target


def _rmse(a: np.ndarray, b: np.ndarray) -> float:
    diff = a - b
    return float(math.sqrt(float(np.mean(diff * diff))))


def _resolve_extras(method: MethodSpec, X_target: np.ndarray) -> dict:
    return {"X_target": X_target} if method.needs_x_target else {}


def _run_one_reference(method: MethodSpec, factory, X, Y, X_target,
                        pls4all_pred: np.ndarray | None) -> ParityRow:
    extras = _resolve_extras(method, X_target)
    n_samples = method.cell_params["n_samples"]
    n_features = method.cell_params["n_features"]
    try:
        reference = factory(**method.cell_params)
        reference.fit(X, Y, **extras)
        ref_pred = reference.predict(X)
    except Exception as exc:
        return ParityRow(
            method=method.name, description=method.description,
            reference_lang="?", reference_lib="?",
            reference_version="?",
            reference_notes=method.notes,
            n_samples=n_samples, n_features=n_features,
            params=method.cell_params,
            parity_pass=False,
            rmse_abs=float("nan"), rmse_rel=float("nan"),
            tolerance=method.rmse_rel_tol,
            status=f"reference_error:{type(exc).__name__}:{exc}",
        )

    if pls4all_pred is None:
        return ParityRow(
            method=method.name, description=method.description,
            reference_lang=reference.language,
            reference_lib=reference.library_name,
            reference_version=reference.library_version,
            reference_notes=reference.notes or method.notes,
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
    X, Y, X_target = _make_dataset(n, p)
    extras = _resolve_extras(method, X_target)

    pls4all_pred: np.ndarray | None = None
    try:
        with pls4all.Context() as ctx, pls4all.Config() as cfg:
            result = method.pls4all_fn(ctx, cfg, X, Y, **method.cell_params,
                                        **extras)
            pls4all_pred = result.matrix("predictions")
            result.close()
    except Exception as exc:
        print(f"  [{method.name}] pls4all_error: {exc}", file=sys.stderr)

    rows: list[ParityRow] = []
    if method.python_reference is None:
        rows.append(ParityRow(
            method=method.name, description=method.description,
            reference_lang="python", reference_lib="(none)",
            reference_version="-",
            reference_notes="No Python reference registered for this method.",
            n_samples=n, n_features=p, params=method.cell_params,
            parity_pass=False, rmse_abs=float("nan"), rmse_rel=float("nan"),
            tolerance=method.rmse_rel_tol,
            status="no_python_reference",
        ))
    else:
        rows.append(_run_one_reference(method, method.python_reference,
                                        X, Y, X_target, pls4all_pred))

    if method.r_reference is None:
        rows.append(ParityRow(
            method=method.name, description=method.description,
            reference_lang="R", reference_lib="(none)",
            reference_version="-",
            reference_notes=(method.notes or
                             "No R reference available for this method."),
            n_samples=n, n_features=p, params=method.cell_params,
            parity_pass=False, rmse_abs=float("nan"), rmse_rel=float("nan"),
            tolerance=method.rmse_rel_tol,
            status="no_r_reference",
        ))
    else:
        rows.append(_run_one_reference(method, method.r_reference,
                                        X, Y, X_target, pls4all_pred))

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


def _write_summary(path: Path, rows: list[ParityRow]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    by_method: dict[str, list[ParityRow]] = {}
    for row in rows:
        by_method.setdefault(row.method, []).append(row)
    lines = [
        "# Parity gate report",
        "",
        f"Host: `{platform.platform()}`",
        f"pls4all: `{pls4all.version()}`",
        f"Python: `{sys.version.split()[0]}`",
        f"NumPy: `{np.__version__}`",
        "",
        "Each method is compared against a Python reference and an R "
        "reference. Methods without a widely installable external "
        "reference are flagged `numpy-mirror` in the lib column.",
        "",
        ("| Method | Description | Reference (lang / lib / version) | "
         "Parity | RMSE rel | Tolerance | Status |"),
        ("|--------|-------------|----------------------------------|"
         "--------|----------|-----------|--------|"),
    ]
    for method_name, method_rows in by_method.items():
        for row in method_rows:
            parity = "✓" if row.parity_pass else "✗"
            rmse = (f"{row.rmse_rel:.3e}"
                    if math.isfinite(row.rmse_rel) else "—")
            ref = (f"{row.reference_lang} / `{row.reference_lib}` "
                   f"{row.reference_version}")
            lines.append(
                f"| `{row.method}` | {row.description} | {ref} | "
                f"{parity} | {rmse} | {row.tolerance:.0e} | {row.status} |"
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


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="parity-gate", description=__doc__)
    parser.add_argument("--methods", nargs="*", default=None,
                        help="Subset of method names (default: all).")
    parser.add_argument("--output-dir", type=Path,
                        default=REPO_ROOT / "benchmarks" / "results" /
                        "parity_gate",
                        help="Output directory for CSV + summary.")
    args = parser.parse_args(argv)
    rows = run(methods=args.methods)
    _write_csv(args.output_dir / "parity.csv", rows)
    _write_summary(args.output_dir / "summary.md", rows)
    print(f"\nWrote {args.output_dir / 'parity.csv'} ({len(rows)} rows)")
    print(f"Wrote {args.output_dir / 'summary.md'}")
    failures = [r for r in rows if not r.parity_pass and
                r.status not in {"no_r_reference", "no_python_reference"}]
    if failures:
        print(f"\n{len(failures)} parity failure(s):", file=sys.stderr)
        for f in failures:
            print(f"  {f.method} [{f.reference_lang}/{f.reference_lib}]: "
                  f"{f.status}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
