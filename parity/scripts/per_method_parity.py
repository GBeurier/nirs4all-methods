#!/usr/bin/env python3
"""Per-method parity tester for nirs4all-methods.

Runs a single method's pls4all implementation against its registry-declared
external reference(s) and prints the RMSE divergence at multiple dataset sizes.

Usage:
    PLS4ALL_LIB_PATH=/path/to/libn4m.so python per_method_parity.py <method_name> [--sizes NxP,...]

Examples:
    python per_method_parity.py pls
    python per_method_parity.py ridge_pls --sizes 100x50,500x500,500x2500
    python per_method_parity.py di_pls --reference ref_python_diplslib
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "bindings/python/src"))

if "PLS4ALL_LIB_PATH" not in os.environ:
    default_lib = REPO / "build/dev-release/cpp/src/libn4m.so"
    if default_lib.exists():
        os.environ["PLS4ALL_LIB_PATH"] = str(default_lib)

import pls4all  # noqa: E402

from benchmarks.parity_timing.registry import (  # noqa: E402
    get_method,
    iter_reference_factories,
    reference_id,
)


def _resolved_references_with_factories(method):
    """Like resolved_references_for_method, but keeps the factory callable."""
    out = []
    seen = set()
    for role, factory in iter_reference_factories(method):
        try:
            probe = factory(**method.cell_params)
        except Exception as exc:  # noqa: BLE001
            out.append({"id": f"<factory-error:{role}>", "factory": None,
                        "library": role, "language": role, "error": str(exc)})
            continue
        if probe is None:
            continue
        rid = reference_id(probe.library_name, probe.language)
        if rid in seen:
            continue
        seen.add(rid)
        out.append({
            "id": rid,
            "role": role,
            "factory": factory,
            "library": probe.library_name,
            "language": probe.language,
            "version": probe.library_version,
        })
    return out


def gen_dataset(n: int, p: int, seed: int,
                n_targets: int = 1) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Match orchestrator.gen_dataset_csv generator exactly.

    For ``n_targets > 1`` we fall back to the runner.py latent-factor
    generator so methods that require multi-column Y (O2PLS / MB-PLS /
    canonical / PLS-SVD …) have a usable target matrix.
    """
    if n_targets > 1:
        rng = np.random.default_rng(seed)
        k = min(5, n - 1, p)
        T = rng.standard_normal((n, k))
        W = rng.standard_normal((p, k))
        X = T @ W.T + 0.05 * rng.standard_normal((n, p))
        B = rng.standard_normal((k, n_targets))
        Y = T @ B + 0.02 * rng.standard_normal((n, n_targets))
        X_target = X.copy()
        X_target[:, 0] *= 1.1
        if p > 1:
            X_target[:, 1] += 0.05
        return X.astype(np.float64), Y.astype(np.float64), X_target.astype(np.float64)
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n, p)).astype(np.float64)
    y = (2.0 * X[:, min(2, p - 1)]
          - X[:, min(5, p - 1)]
          + 0.5 * X[:, min(8, p - 1)]
          + 0.05 * rng.standard_normal(n)).astype(np.float64)
    X_target = X + 0.01 * rng.standard_normal(X.shape)
    return X, y.reshape(-1, 1), X_target


def gen_labels(y: np.ndarray, n_classes: int, seed: int) -> np.ndarray:
    """Deterministic class label vector for label-using methods. Mirrors
    ``benchmarks.parity_timing.runner._make_dataset`` semantics: uses a
    seeded ``Generator.integers`` call so labels are reproducible and
    every class has a non-trivial probability of being present."""
    rng = np.random.default_rng(seed + 7919)
    n = int(y.shape[0])
    return rng.integers(0, max(2, int(n_classes)), size=n).astype(np.int32)


def gen_group_assignment(p: int) -> np.ndarray:
    """Deterministic group assignment matching ``runner._make_dataset``:
    features in ``n_groups`` consecutive blocks (n_groups = min(5, p // 3))."""
    n_groups = max(1, min(5, p // 3))
    group_assignment = (np.arange(p) // max(1, p // n_groups)).astype(np.int32)
    return np.minimum(group_assignment, n_groups - 1)


def call_pls4all(method, n_samples: int, n_features: int, seed: int) -> np.ndarray:
    n_targets = int(method.cell_params.get("n_targets", 1) or 1)
    X, y, X_target = gen_dataset(n_samples, n_features, seed, n_targets=n_targets)
    ctx = pls4all.Context()
    cfg = pls4all.Config()
    kwargs = dict(method.cell_params)
    n_classes = int(kwargs.get("n_classes", 0) or 0)
    kwargs.pop("n_samples", None)
    kwargs.pop("n_features", None)
    if method.needs_x_target:
        kwargs["X_target"] = X_target
    if getattr(method, "needs_sample_weights", False):
        kwargs["sample_weights"] = np.abs(y.reshape(-1)) + 0.5
    if getattr(method, "needs_labels", False):
        kwargs["y_labels"] = gen_labels(y, n_classes, seed)
    if getattr(method, "needs_group_assignment", False):
        kwargs["group_assignment"] = gen_group_assignment(n_features)
    res = method.pls4all_fn(ctx, cfg, X, y, **kwargs)
    prediction_key = getattr(method, "prediction_key", "predictions") or "predictions"
    # MethodResult-style wrappers expose .matrix(<name>).
    if hasattr(res, "matrix"):
        for name in (prediction_key, "predictions"):
            try:
                return np.asarray(res.matrix(name), dtype=np.float64)
            except (KeyError, RuntimeError):
                continue
    if hasattr(res, "predictions"):
        return np.asarray(res.predictions, dtype=np.float64)
    if hasattr(res, "predict"):
        return np.asarray(res.predict(X), dtype=np.float64)
    return np.asarray(res, dtype=np.float64)


def call_reference(ref_factory, method, n_samples: int, n_features: int, seed: int) -> np.ndarray:
    n_targets = int(method.cell_params.get("n_targets", 1) or 1)
    X, y, X_target = gen_dataset(n_samples, n_features, seed, n_targets=n_targets)
    kwargs = dict(method.cell_params)
    n_classes = int(kwargs.get("n_classes", 0) or 0)
    kwargs.pop("n_samples", None)
    kwargs.pop("n_features", None)
    if method.needs_x_target:
        kwargs["X_target"] = X_target
    ref = ref_factory(**kwargs)
    if ref is None:
        return None
    fit_kwargs = {}
    if getattr(method, "needs_sample_weights", False):
        fit_kwargs["sample_weights"] = np.abs(y.reshape(-1)) + 0.5
    if getattr(method, "needs_labels", False):
        fit_kwargs["y_labels"] = gen_labels(y, n_classes, seed)
    if getattr(method, "needs_group_assignment", False):
        fit_kwargs["group_assignment"] = gen_group_assignment(n_features)
    if method.needs_x_target:
        ref.fit(X, y, X_target=X_target, **fit_kwargs)
    else:
        ref.fit(X, y, **fit_kwargs)
    return np.asarray(ref.predict(X), dtype=np.float64)


def rmse(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=np.float64).reshape(-1)
    b = np.asarray(b, dtype=np.float64).reshape(-1)
    return float(np.sqrt(np.mean((a - b) ** 2)))


def max_abs(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=np.float64).reshape(-1)
    b = np.asarray(b, dtype=np.float64).reshape(-1)
    return float(np.max(np.abs(a - b)))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("method", help="Method name as in registry")
    ap.add_argument("--sizes", default="100x50,500x500,500x2500",
                    help="Comma list of NxP pairs (default: 100x50,500x500,500x2500)")
    ap.add_argument("--seed", type=int, default=1_234_567_890,
                    help="Dataset seed (default matches orchestrator)")
    ap.add_argument("--reference", default=None,
                    help="If set, only run this reference id (e.g. ref_python_scikit_learn)")
    ap.add_argument("--tol", type=float, default=1e-6,
                    help="Failure threshold (default 1e-6)")
    args = ap.parse_args(argv)

    try:
        method = get_method(args.method)
    except KeyError:
        print(f"ERROR: unknown method '{args.method}'", file=sys.stderr)
        return 2

    refs = _resolved_references_with_factories(method)
    if args.reference:
        wanted = args.reference.replace("ref_", "")
        refs = [r for r in refs if r["id"] == wanted]
    if not refs:
        all_refs = _resolved_references_with_factories(method)
        print(f"ERROR: no references resolved for '{args.method}' (try one of: "
              f"{[r['id'] for r in all_refs]})", file=sys.stderr)
        return 2

    sizes = []
    for cell in args.sizes.split(","):
        n_str, p_str = cell.lower().strip().split("x")
        sizes.append((int(n_str), int(p_str)))

    print(f"\n=== {args.method} parity ===")
    print(f"sizes={sizes}  seed={args.seed}  tol={args.tol:.1e}")
    print(f"references: {[r['id'] for r in refs]}\n")

    overall_pass = True

    for n, p in sizes:
        print(f"--- {n}x{p} ---")
        try:
            t0 = time.perf_counter()
            preds_p4a = call_pls4all(method, n, p, args.seed)
            t_p4a = time.perf_counter() - t0
            print(f"  pls4all                       : ok ({t_p4a:.2f}s, shape={preds_p4a.shape})")
        except Exception as exc:  # noqa: BLE001
            print(f"  pls4all                       : ERROR — {type(exc).__name__}: {exc}")
            overall_pass = False
            continue

        for ref in refs:
            ref_id = ref["id"]
            ref_factory = ref["factory"]
            try:
                t0 = time.perf_counter()
                preds_ref = call_reference(ref_factory, method, n, p, args.seed)
                t_ref = time.perf_counter() - t0
            except Exception as exc:  # noqa: BLE001
                print(f"  {ref_id:32s} : ERROR — {type(exc).__name__}: {exc}")
                continue
            if preds_ref is None:
                print(f"  {ref_id:32s} : skipped (factory returned None)")
                continue
            if preds_ref.shape != preds_p4a.shape:
                try:
                    preds_ref_aligned = preds_ref.reshape(preds_p4a.shape)
                except ValueError:
                    print(f"  {ref_id:32s} : SHAPE MISMATCH p4a={preds_p4a.shape} ref={preds_ref.shape}")
                    overall_pass = False
                    continue
            else:
                preds_ref_aligned = preds_ref
            d_max = max_abs(preds_p4a, preds_ref_aligned)
            d_rmse = rmse(preds_p4a, preds_ref_aligned)
            verdict = "PASS" if d_max < args.tol else "FAIL"
            if verdict == "FAIL":
                overall_pass = False
            print(f"  {ref_id:32s} : max_abs={d_max:.3e}  rmse={d_rmse:.3e}  ({t_ref:.2f}s)  [{verdict}]")
        print()

    return 0 if overall_pass else 1


if __name__ == "__main__":
    sys.exit(main())
