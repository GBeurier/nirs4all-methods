#!/usr/bin/env python3
# SPDX-License-Identifier: CECILL-2.1
"""Cross-binding timing matrix for the docs dashboard.

The runner follows the same contract as the pls4all dashboard input: one CSV
row per algorithm/backend/size/thread count with timing and parity metadata.
Internal rows are produced from three routes when available:

* ``cpp``: direct libc4a C ABI calls through ctypes;
* ``python``: public ABI-close Python functions;
* ``sklearn``: public scikit-learn-compatible Python estimators;
* ``r``: installed public R package through Rscript.

External reference rows are intentionally best-effort. Same-contract references
are compared to stored snapshots; useful external namesakes with different
boundary, output-shape, RNG, or solver contracts are recorded as performance
context only and are not parity-gated.
"""

from __future__ import annotations

import argparse
import contextlib
import csv
import ctypes
import datetime as dt
import importlib
from importlib import metadata as importlib_metadata
import json
import math
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
import weakref
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Iterable

import numpy as np

try:
    from threadpoolctl import threadpool_limits
except ModuleNotFoundError:  # pragma: no cover - optional runtime dependency
    threadpool_limits = None


REPO_ROOT = Path(__file__).resolve().parents[2]
PY_BINDINGS = REPO_ROOT / "bindings" / "python" / "src"
if str(PY_BINDINGS) not in sys.path:
    sys.path.insert(0, str(PY_BINDINGS))

PARITY_SRC = REPO_ROOT / "parity" / "python_generator" / "src"
if str(PARITY_SRC) not in sys.path:
    sys.path.insert(0, str(PARITY_SRC))

LOCAL_NIRS4ALL = REPO_ROOT.parent / "nirs4all"
if LOCAL_NIRS4ALL.exists() and str(LOCAL_NIRS4ALL) not in sys.path:
    sys.path.insert(0, str(LOCAL_NIRS4ALL))

if "CHEMOMETRICS4ALL_LIB_PATH" not in os.environ:
    for candidate in (
        REPO_ROOT / "build" / "dev-release" / "cpp" / "src" / "libc4a.so",
        REPO_ROOT / "build" / "local-tests" / "cpp" / "src" / "libc4a.so",
        REPO_ROOT / "build" / "cpp" / "src" / "libc4a.so",
    ):
        if candidate.exists():
            os.environ["CHEMOMETRICS4ALL_LIB_PATH"] = str(candidate)
            break

import chemometrics4all as c4a  # noqa: E402
from chemometrics4all import python as c4a_python  # noqa: E402
from chemometrics4all._errors import check  # noqa: E402
from chemometrics4all._ffi import lib  # noqa: E402
from chemometrics4all._matrix import as_f64_2d, empty_like_f64, empty_like_i32, numpy_to_view  # noqa: E402
from chemometrics4all._rng import PCG64  # noqa: E402
from chemometrics4all._types import FilterStats, SplitResult, TransferMetrics  # noqa: E402


CSV_FIELDS = [
    "algorithm",
    "family",
    "backend",
    "language",
    "tier",
    "kind",
    "n",
    "p",
    "threads",
    "lib_build",
    "ok",
    "median_ms",
    "warmup_runs",
    "timed_runs",
    "batch_loops",
    "samples_ms",
    "parity",
    "reason",
    "reference_library",
    "reference_role",
    "binding_parity_ok",
    "reference_parity_ok",
    "reference_max_abs_diff",
    "reference_rms_diff",
    "reference_rel_l2_diff",
    "binding_max_abs_diff",
    "binding_rms_diff",
    "binding_rel_l2_diff",
]
REFERENCE_SNAPSHOT_SCHEMA = "chemometrics4all.reference_snapshot.v1"
REFERENCE_SNAPSHOT_ROOT = REPO_ROOT / "benchmarks" / "reference_snapshots" / "cross_binding"

PLS4ALL_BENCHMARK_SIZES = [
    (100, 50),
    (100, 500),
    (100, 2500),
    (500, 50),
    (500, 500),
    (500, 2500),
    (2500, 50),
    (2500, 500),
    (2500, 2500),
    (10000, 50),
    (10000, 500),
]
PLS4ALL_SMALL_SIZES = PLS4ALL_BENCHMARK_SIZES[:3]


@dataclass(frozen=True)
class Dataset:
    X: np.ndarray
    y: np.ndarray
    d: np.ndarray
    wavelengths: np.ndarray
    seed: int


@dataclass(frozen=True)
class ReferenceSpec:
    backend: str
    library: str
    factory: Callable[[Dataset], Callable[[], object]] | None = None
    compare: bool = True
    gate_c4a: bool = True
    language: str = "Python"
    r_expr: str | None = None
    contract_note: str = ""
    max_cols: int | None = None
    comparator: Callable[[object, object], tuple[bool, str]] | None = None


@dataclass(frozen=True)
class MethodSpec:
    method_id: str
    label: str
    family: str
    python_factory: Callable[[Dataset], Callable[[], object]]
    cpp_factory: Callable[[Dataset], Callable[[], object]] | None = None
    r_expr: str | None = None
    references: tuple[ReferenceSpec, ...] = field(default_factory=tuple)
    compare_internal: bool = True
    comparator: Callable[[object, object], tuple[bool, str]] | None = None
    max_cols: int | None = None


def synthetic_spectra(n: int, p: int, seed: int) -> Dataset:
    rng = np.random.default_rng(seed)
    wl = np.linspace(900.0, 1700.0, p, dtype=np.float64)
    axis = np.linspace(0.0, 1.0, p, dtype=np.float64)
    base = (
        0.45
        + 0.10 * np.sin(2.0 * np.pi * axis)
        + 0.08 * np.exp(-((axis - 0.32) ** 2) / 0.012)
        + 0.05 * np.exp(-((axis - 0.72) ** 2) / 0.018)
    )
    slopes = rng.normal(0.0, 0.025, size=(n, 1))
    offsets = rng.normal(0.0, 0.015, size=(n, 1))
    multiplicative = rng.normal(1.0, 0.06, size=(n, 1))
    noise = rng.normal(0.0, 0.006, size=(n, p))
    X = base + slopes * (axis - 0.5) + offsets
    X = np.clip(multiplicative * X + noise, 1e-4, None)
    X = np.ascontiguousarray(X, dtype=np.float64)
    y = 0.7 * X[:, max(1, p // 3)] + 0.3 * X[:, min(p - 1, 2 * p // 3)]
    y += rng.normal(0.0, 0.005, size=n)
    y = np.ascontiguousarray(y, dtype=np.float64)
    d = np.ascontiguousarray(np.linspace(-1.0, 1.0, n) + rng.normal(0.0, 0.01, size=n), dtype=np.float64)
    return Dataset(X=X, y=y, d=d, wavelengths=wl, seed=seed)


def clone_dataset(ctx: Dataset) -> Dataset:
    return Dataset(
        X=np.array(ctx.X, copy=True),
        y=np.array(ctx.y, copy=True),
        d=np.array(ctx.d, copy=True),
        wavelengths=np.array(ctx.wavelengths, copy=True),
        seed=ctx.seed,
    )


def bool_cell(value: bool | None) -> str:
    if value is None:
        return ""
    return "true" if value else "false"


def runtime_version() -> str:
    try:
        return c4a.version()
    except Exception:
        return f"abi.{'.'.join(map(str, c4a.abi_version()))}"


_EXTERNAL_VERSION_CACHE: dict[str, str] = {}


def external_version(backend: str) -> str:
    if backend in _EXTERNAL_VERSION_CACHE:
        return _EXTERNAL_VERSION_CACHE[backend]
    value = ""
    if backend == "ref.nirs4all":
        try:
            sha = subprocess.check_output(
                ["git", "-C", str(LOCAL_NIRS4ALL), "rev-parse", "--short", "HEAD"],
                text=True,
                stderr=subprocess.DEVNULL,
                timeout=2,
            ).strip()
            dirty = subprocess.run(
                ["git", "-C", str(LOCAL_NIRS4ALL), "status", "--porcelain"],
                check=False,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
                timeout=2,
            ).stdout.strip()
            value = f"nirs4all@{sha}{'+dirty' if dirty else ''}"
        except (OSError, subprocess.SubprocessError):
            value = "nirs4all local checkout"
    elif backend.startswith("ref.r."):
        package = backend.removeprefix("ref.r.")
        if shutil.which("Rscript") is not None:
            try:
                completed = subprocess.run(
                    [
                        "Rscript",
                        "--vanilla",
                        "-e",
                        f"cat(as.character(packageVersion('{package}')))",
                    ],
                    check=True,
                    text=True,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.DEVNULL,
                    timeout=5,
                )
                value = f"{package} {completed.stdout.strip()}"
            except (OSError, subprocess.SubprocessError):
                value = package
    elif backend in {
        "ref.numpy",
        "ref.sklearn",
        "ref.scipy",
        "ref.pybaselines",
        "ref.pywavelets",
        "ref.pycaltransfer",
        "ref.cowarp",
        "ref.dtw_python",
        "ref.statsmodels",
    }:
        module_name, dist_name = {
            "ref.numpy": ("numpy", "numpy"),
            "ref.sklearn": ("sklearn", "scikit-learn"),
            "ref.scipy": ("scipy", "scipy"),
            "ref.pybaselines": ("pybaselines", "pybaselines"),
            "ref.pywavelets": ("pywt", "PyWavelets"),
            "ref.pycaltransfer": ("pycaltransfer", "pycaltransfer"),
            "ref.cowarp": ("cowarp", "cowarp"),
            "ref.dtw_python": ("dtw", "dtw-python"),
            "ref.statsmodels": ("statsmodels", "statsmodels"),
        }[backend]
        try:
            module = importlib.import_module(module_name)
            try:
                version = importlib_metadata.version(dist_name)
            except importlib_metadata.PackageNotFoundError:
                version = getattr(module, "__version__", "unknown")
            value = f"{dist_name} {version}"
        except ModuleNotFoundError:
            value = dist_name
    _EXTERNAL_VERSION_CACHE[backend] = value
    return value


def _consume(out: object) -> float:
    if isinstance(out, tuple):
        return sum(_consume(item) for item in out)
    if isinstance(out, dict):
        return sum(_consume(item) for item in out.values())
    if isinstance(out, list):
        return sum(_consume(item) for item in out)
    if isinstance(out, np.ndarray):
        if out.size == 0:
            return 0.0
        value = float(out.flat[0])
        if math.isnan(value):
            return 0.0
        if math.isinf(value):
            return 0.0
        return value
    arr = np.asarray(out)
    if arr.size == 0:
        return 0.0
    value = float(arr.reshape(-1)[0])
    if math.isnan(value):
        return 0.0
    if math.isinf(value):
        return 0.0
    return value


def flatten_output(out: object) -> np.ndarray:
    if isinstance(out, tuple):
        parts = [flatten_output(item) for item in out]
        return np.concatenate(parts) if parts else np.empty(0, dtype=np.float64)
    if isinstance(out, dict):
        parts = [flatten_output(out[key]) for key in sorted(out)]
        return np.concatenate(parts) if parts else np.empty(0, dtype=np.float64)
    if isinstance(out, list):
        parts = [flatten_output(item) for item in out]
        return np.concatenate(parts) if parts else np.empty(0, dtype=np.float64)
    arr = np.asarray(out)
    if arr.dtype == np.bool_:
        arr = arr.astype(np.float64)
    elif not np.issubdtype(arr.dtype, np.number):
        arr = arr.astype(np.float64)
    return np.ravel(np.asarray(arr, dtype=np.float64))


def outputs_close(left: object, right: object, *, rtol: float = 1e-5, atol: float = 1e-8) -> tuple[bool, str]:
    a = flatten_output(left)
    b = flatten_output(right)
    if a.shape != b.shape:
        return False, f"shape mismatch {a.shape} vs {b.shape}"
    if not np.allclose(a, b, rtol=rtol, atol=atol, equal_nan=True):
        diff = float(np.nanmax(np.abs(a - b))) if a.size else 0.0
        return False, f"max abs diff {diff:.3g}"
    return True, "ok"


def outputs_close_sign_invariant_columns(left: object, right: object, *, rtol: float = 1e-5, atol: float = 1e-8) -> tuple[bool, str]:
    a = np.asarray(left, dtype=np.float64)
    b = np.asarray(right, dtype=np.float64)
    if a.shape != b.shape:
        return False, f"shape mismatch {a.shape} vs {b.shape}"
    if a.ndim != 2:
        return outputs_close(left, right, rtol=rtol, atol=atol)
    aligned = np.array(b, copy=True)
    for col in range(aligned.shape[1]):
        if float(np.dot(a[:, col], aligned[:, col])) < 0.0:
            aligned[:, col] *= -1.0
    if not np.allclose(a, aligned, rtol=rtol, atol=atol, equal_nan=True):
        diff = float(np.nanmax(np.abs(a - aligned))) if a.size else 0.0
        return False, f"sign-aligned max abs diff {diff:.3g}"
    return True, "ok (sign-invariant component comparison)"


def outputs_close_iasls(left: object, right: object) -> tuple[bool, str]:
    return outputs_close(left, right, rtol=1e-5, atol=5e-6)


def _savgol_valid_window_pair(left: object, right: object) -> tuple[np.ndarray, np.ndarray]:
    a = np.asarray(left, dtype=np.float64)
    b = np.asarray(right, dtype=np.float64)
    if a.ndim != 2 or b.ndim != 2 or a.shape[0] != b.shape[0]:
        return flatten_output(left), flatten_output(right)
    if a.shape == b.shape:
        return np.ravel(a), np.ravel(b)
    if a.shape[1] > b.shape[1]:
        diff = a.shape[1] - b.shape[1]
        lo = diff // 2
        hi = a.shape[1] - (diff - lo)
        return np.ravel(a[:, lo:hi]), np.ravel(b)
    diff = b.shape[1] - a.shape[1]
    lo = diff // 2
    hi = b.shape[1] - (diff - lo)
    return np.ravel(a), np.ravel(b[:, lo:hi])


def outputs_close_savgol_valid_window(left: object, right: object) -> tuple[bool, str]:
    a, b = _savgol_valid_window_pair(left, right)
    if a.shape != b.shape:
        return False, f"valid-window shape mismatch {a.shape} vs {b.shape}"
    if not np.allclose(a, b, rtol=1e-5, atol=1e-8, equal_nan=True):
        diff = float(np.nanmax(np.abs(a - b))) if a.size else 0.0
        return False, f"valid-window max abs diff {diff:.3g}"
    return True, "ok (valid-window comparison)"


def compare_outputs(spec: MethodSpec, left: object, right: object) -> tuple[bool, str]:
    if spec.comparator is not None:
        return spec.comparator(left, right)
    return outputs_close(left, right)


def compare_reference_outputs(
    spec: MethodSpec,
    ref: ReferenceSpec,
    left: object,
    right: object,
) -> tuple[bool, str]:
    if ref.comparator is not None:
        return ref.comparator(left, right)
    return compare_outputs(spec, left, right)


def aligned_flat_outputs(spec: MethodSpec, observed: object, expected: object) -> tuple[np.ndarray, np.ndarray]:
    if spec.comparator is outputs_close_sign_invariant_columns:
        a2 = np.asarray(observed, dtype=np.float64)
        b2 = np.asarray(expected, dtype=np.float64)
        if a2.shape != b2.shape:
            return flatten_output(observed), flatten_output(expected)
        if a2.ndim == 2:
            b2 = np.array(b2, copy=True)
            for col in range(b2.shape[1]):
                if float(np.dot(a2[:, col], b2[:, col])) < 0.0:
                    b2[:, col] *= -1.0
            return np.ravel(a2), np.ravel(b2)
    return flatten_output(observed), flatten_output(expected)


def aligned_reference_flat_outputs(
    spec: MethodSpec,
    ref: ReferenceSpec,
    observed: object,
    expected: object,
) -> tuple[np.ndarray, np.ndarray]:
    if ref.comparator is outputs_close_savgol_valid_window:
        return _savgol_valid_window_pair(observed, expected)
    return aligned_flat_outputs(spec, observed, expected)


def output_divergence(spec: MethodSpec, observed: object | None, expected: object | None) -> dict[str, float]:
    if observed is None or expected is None:
        return {}
    a, b = aligned_flat_outputs(spec, observed, expected)
    if a.shape != b.shape:
        return {}
    if a.size == 0:
        return {"max_abs": 0.0, "rms": 0.0, "rel_l2": 0.0}
    diff = np.where(np.isnan(a) & np.isnan(b), 0.0, a - b)
    abs_diff = np.abs(diff)
    max_abs = float(np.nanmax(abs_diff))
    rms = float(np.sqrt(np.nanmean(diff * diff)))
    denom = float(np.linalg.norm(np.nan_to_num(b, nan=0.0)))
    rel_l2 = float(np.linalg.norm(np.nan_to_num(diff, nan=0.0)) / max(denom, np.finfo(np.float64).tiny))
    return {"max_abs": max_abs, "rms": rms, "rel_l2": rel_l2}


def output_reference_divergence(
    spec: MethodSpec,
    ref: ReferenceSpec,
    observed: object | None,
    expected: object | None,
) -> dict[str, float]:
    if observed is None or expected is None:
        return {}
    a, b = aligned_reference_flat_outputs(spec, ref, observed, expected)
    if a.shape != b.shape:
        return {}
    if a.size == 0:
        return {"max_abs": 0.0, "rms": 0.0, "rel_l2": 0.0}
    diff = np.where(np.isnan(a) & np.isnan(b), 0.0, a - b)
    abs_diff = np.abs(diff)
    max_abs = float(np.nanmax(abs_diff))
    rms = float(np.sqrt(np.nanmean(diff * diff)))
    denom = float(np.linalg.norm(np.nan_to_num(b, nan=0.0)))
    rel_l2 = float(np.linalg.norm(np.nan_to_num(diff, nan=0.0)) / max(denom, np.finfo(np.float64).tiny))
    return {"max_abs": max_abs, "rms": rms, "rel_l2": rel_l2}


@dataclass(frozen=True)
class LoadedReferenceSnapshot:
    path: Path
    output: object
    metadata: dict


def _safe_slug(value: str) -> str:
    text = re.sub(r"[^A-Za-z0-9_.-]+", "_", value.strip())
    text = re.sub(r"_+", "_", text).strip("_.")
    return text[:96] or "reference"


def reference_snapshot_path(spec: MethodSpec, ref: ReferenceSpec, ctx: Dataset) -> Path:
    n, p = ctx.X.shape
    ref_id = f"{_safe_slug(ref.backend)}__{_safe_slug(ref.library)}"
    return REFERENCE_SNAPSHOT_ROOT / _safe_slug(spec.method_id) / f"{ref_id}__n{n}_p{p}_seed{ctx.seed}.npz"


def _snapshot_array(output: object) -> tuple[np.ndarray, list[int], str]:
    if isinstance(output, (tuple, list, dict)):
        arr = flatten_output(output)
        return np.asarray(arr, dtype=np.float64), [int(arr.size)], "flat"
    arr = np.asarray(output, dtype=np.float64)
    return np.array(arr, copy=True), [int(dim) for dim in arr.shape], "array"


def write_reference_snapshot(spec: MethodSpec, ref: ReferenceSpec, ctx: Dataset, output: object) -> LoadedReferenceSnapshot:
    path = reference_snapshot_path(spec, ref, ctx)
    path.parent.mkdir(parents=True, exist_ok=True)
    arr, shape, kind = _snapshot_array(output)
    metadata = {
        "schema": REFERENCE_SNAPSHOT_SCHEMA,
        "method_id": spec.method_id,
        "method_label": spec.label,
        "family": spec.family,
        "n": int(ctx.X.shape[0]),
        "p": int(ctx.X.shape[1]),
        "seed": int(ctx.seed),
        "reference_backend": ref.backend,
        "reference_library": ref.library,
        "reference_language": ref.language,
        "reference_version": external_version(ref.backend),
        "generated_at": dt.datetime.now(dt.timezone.utc).replace(microsecond=0).isoformat(),
        "output_kind": kind,
        "output_shape": shape,
        "output_dtype": "float64",
    }
    np.savez_compressed(
        path,
        output=np.asarray(arr, dtype=np.float64),
        metadata_json=np.array(json.dumps(metadata, sort_keys=True)),
    )
    return LoadedReferenceSnapshot(path=path, output=np.asarray(arr, dtype=np.float64).reshape(shape) if kind == "array" else np.asarray(arr, dtype=np.float64), metadata=metadata)


def load_reference_snapshot(spec: MethodSpec, ref: ReferenceSpec, ctx: Dataset) -> LoadedReferenceSnapshot:
    path = reference_snapshot_path(spec, ref, ctx)
    if not path.exists():
        raise FileNotFoundError(f"missing reference snapshot: {path.relative_to(REPO_ROOT)}")
    with np.load(path, allow_pickle=False) as data:
        output = np.array(data["output"], dtype=np.float64, copy=True)
        metadata_raw = data["metadata_json"]
        metadata = json.loads(str(metadata_raw.item()))
    if metadata.get("schema") != REFERENCE_SNAPSHOT_SCHEMA:
        raise ValueError(f"unsupported reference snapshot schema in {path.relative_to(REPO_ROOT)}")
    expected = {
        "method_id": spec.method_id,
        "n": int(ctx.X.shape[0]),
        "p": int(ctx.X.shape[1]),
        "seed": int(ctx.seed),
        "reference_backend": ref.backend,
        "reference_library": ref.library,
    }
    mismatches = [
        f"{key}={metadata.get(key)!r} (expected {value!r})"
        for key, value in expected.items()
        if metadata.get(key) != value
    ]
    if mismatches:
        raise ValueError(f"reference snapshot metadata mismatch in {_rel(path)}: {', '.join(mismatches)}")
    shape = tuple(int(dim) for dim in metadata.get("output_shape", list(output.shape)))
    kind = metadata.get("output_kind", "flat")
    restored: object = output.reshape(shape) if kind == "array" else output.reshape(-1)
    return LoadedReferenceSnapshot(path=path, output=restored, metadata=metadata)


def get_reference_snapshot(
    spec: MethodSpec,
    ref: ReferenceSpec,
    ctx: Dataset,
    output: object,
    *,
    write_snapshots: bool,
) -> LoadedReferenceSnapshot:
    path = reference_snapshot_path(spec, ref, ctx)
    if write_snapshots or not path.exists():
        if not write_snapshots:
            raise FileNotFoundError(f"missing reference snapshot: {path.relative_to(REPO_ROOT)}")
        return write_reference_snapshot(spec, ref, ctx, output)
    return load_reference_snapshot(spec, ref, ctx)


def _rel(path: Path) -> str:
    try:
        return str(path.relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


def warmup_count(repeat: int) -> int:
    return max(1, min(3, int(repeat)))


def batch_loop_count(probe_ms: float, *, target_ms: float = 10.0, max_loops: int = 1000) -> int:
    if not math.isfinite(probe_ms) or probe_ms <= 0.0:
        return max_loops
    return max(1, min(max_loops, int(math.ceil(target_ms / probe_ms))))


def time_callable(fn: Callable[[], object], repeat: int) -> tuple[float, object, list[float], int]:
    samples: list[float] = []
    last: object | None = None
    cached_output = getattr(fn, "_c4a_output", None)

    def run_once() -> object:
        out = fn()
        if cached_output is None:
            _consume(out)
            return out
        return cached_output

    for _ in range(warmup_count(repeat)):
        last = run_once()
    probe_start = time.perf_counter()
    last = run_once()
    probe_ms = (time.perf_counter() - probe_start) * 1000.0
    batch_loops = batch_loop_count(probe_ms)
    for _ in range(warmup_count(repeat)):
        for _ in range(batch_loops):
            last = run_once()
    for _ in range(repeat):
        start = time.perf_counter()
        for _ in range(batch_loops):
            last = run_once()
        samples.append(((time.perf_counter() - start) * 1000.0) / batch_loops)
    return float(np.median(samples)), last, samples, batch_loops


def set_thread_env(threads: int) -> None:
    value = str(threads)
    for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
        os.environ[name] = value


def threadpool_context(threads: int):
    if threadpool_limits is None:
        return contextlib.nullcontext()
    return threadpool_limits(limits=threads)


def _create(prefix: str, *args) -> ctypes.c_void_p:
    handle = ctypes.c_void_p()
    check(getattr(lib, f"{prefix}_create")(ctypes.byref(handle), *args), f"{prefix}_create")
    return handle


def _create_ex(prefix: str, *args) -> ctypes.c_void_p:
    handle = ctypes.c_void_p()
    check(getattr(lib, f"{prefix}_create_ex")(ctypes.byref(handle), *args), f"{prefix}_create_ex")
    return handle


def c_transform(
    prefix: str,
    X: np.ndarray,
    *create_args,
    fit_X: np.ndarray | None = None,
    out_shape: tuple[int, int] | None = None,
    out_dtype: str = "f64",
) -> np.ndarray:
    X_arr = as_f64_2d(X)
    fit_arr = as_f64_2d(fit_X) if fit_X is not None else None
    handle = _create(prefix, *create_args)
    try:
        if fit_arr is not None:
            check(getattr(lib, f"{prefix}_fit")(handle, numpy_to_view(fit_arr)), f"{prefix}_fit")
        shape = out_shape or X_arr.shape
        out = empty_like_i32(shape) if out_dtype == "i32" else empty_like_f64(shape)
        check(getattr(lib, f"{prefix}_transform")(handle, numpy_to_view(X_arr), numpy_to_view(out)), f"{prefix}_transform")
        return out
    finally:
        getattr(lib, f"{prefix}_destroy")(handle)


def c_fit_y_transform(prefix: str, X: np.ndarray, y: np.ndarray, *create_args) -> np.ndarray:
    X_arr = as_f64_2d(X)
    y_arr = np.ascontiguousarray(y, dtype=np.float64)
    handle = _create(prefix, *create_args)
    try:
        check(
            getattr(lib, f"{prefix}_fit")(
                handle,
                numpy_to_view(X_arr),
                y_arr.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
                ctypes.c_int64(y_arr.size),
            ),
            f"{prefix}_fit",
        )
        out = empty_like_f64(X_arr.shape)
        check(getattr(lib, f"{prefix}_transform")(handle, numpy_to_view(X_arr), numpy_to_view(out)), f"{prefix}_transform")
        return out
    finally:
        getattr(lib, f"{prefix}_destroy")(handle)


def split_result_to_arrays(result: SplitResult) -> tuple[np.ndarray, np.ndarray]:
    train = np.ctypeslib.as_array(result.train_idx, shape=(int(result.n_train),)).copy()
    test = np.ctypeslib.as_array(result.test_idx, shape=(int(result.n_test),)).copy()
    lib.c4a_split_result_destroy(ctypes.byref(result))
    return train, test


def c_split_kennard_stone(ctx: Dataset) -> tuple[np.ndarray, np.ndarray]:
    handle = _create("c4a_split_kennard_stone", ctypes.c_double(0.25))
    try:
        result = SplitResult()
        check(
            lib.c4a_split_kennard_stone_split(handle, numpy_to_view(ctx.X), ctypes.byref(result)),
            "c4a_split_kennard_stone_split",
        )
        return split_result_to_arrays(result)
    finally:
        lib.c4a_split_kennard_stone_destroy(handle)


def c_split_spxy(ctx: Dataset) -> tuple[np.ndarray, np.ndarray]:
    y = np.ascontiguousarray(ctx.y.reshape(-1, 1), dtype=np.float64)
    handle = _create("c4a_split_spxy", ctypes.c_double(0.25))
    try:
        result = SplitResult()
        check(
            lib.c4a_split_spxy_split(
                handle,
                numpy_to_view(ctx.X),
                numpy_to_view(y),
                ctypes.byref(result),
            ),
            "c4a_split_spxy_split",
        )
        return split_result_to_arrays(result)
    finally:
        lib.c4a_split_spxy_destroy(handle)


def c_split_kbins_stratified(ctx: Dataset) -> tuple[np.ndarray, np.ndarray]:
    y = np.ascontiguousarray(ctx.y.reshape(-1, 1), dtype=np.float64)
    handle = _create(
        "c4a_split_kbins_stratified",
        ctypes.c_double(0.25),
        ctypes.c_uint64(17),
        ctypes.c_int32(5),
        ctypes.c_int32(0),
    )
    try:
        result = SplitResult()
        check(
            lib.c4a_split_kbins_stratified_split(handle, numpy_to_view(y), ctypes.byref(result)),
            "c4a_split_kbins_stratified_split",
        )
        return split_result_to_arrays(result)
    finally:
        lib.c4a_split_kbins_stratified_destroy(handle)


def c_y_outlier_iqr(ctx: Dataset) -> np.ndarray:
    y = np.ascontiguousarray(ctx.y, dtype=np.float64)
    handle = _create(
        "c4a_filter_y_outlier",
        ctypes.c_int(0),
        ctypes.c_double(1.5),
        ctypes.c_double(1.0),
        ctypes.c_double(99.0),
    )
    try:
        check(
            lib.c4a_filter_y_outlier_fit(
                handle,
                y.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
                ctypes.c_int64(y.size),
            ),
            "c4a_filter_y_outlier_fit",
        )
        mask = np.empty(y.size, dtype=np.uint8)
        stats = FilterStats()
        check(
            lib.c4a_filter_y_outlier_apply(
                handle,
                y.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
                ctypes.c_int64(y.size),
                mask.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8)),
                ctypes.byref(stats),
            ),
            "c4a_filter_y_outlier_apply",
        )
        return mask
    finally:
        lib.c4a_filter_y_outlier_destroy(handle)


def c_x_outlier_mahalanobis(ctx: Dataset) -> np.ndarray:
    handle = _create(
        "c4a_filter_x_outlier",
        ctypes.c_int32(0),
        ctypes.c_int(0),
        ctypes.c_double(0.0),
        ctypes.c_int32(min(5, ctx.X.shape[1])),
        ctypes.c_double(0.1),
        ctypes.c_uint64(0),
        ctypes.c_int32(100),
        ctypes.c_int64(256),
    )
    try:
        check(lib.c4a_filter_x_outlier_fit(handle, numpy_to_view(ctx.X)), "c4a_filter_x_outlier_fit")
        mask = np.empty(ctx.X.shape[0], dtype=np.uint8)
        stats = FilterStats()
        check(
            lib.c4a_filter_x_outlier_apply(
                handle,
                numpy_to_view(ctx.X),
                mask.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8)),
                ctypes.byref(stats),
            ),
            "c4a_filter_x_outlier_apply",
        )
        return mask
    finally:
        lib.c4a_filter_x_outlier_destroy(handle)


def _transfer_target(ctx: Dataset) -> np.ndarray:
    return np.ascontiguousarray(1.05 * ctx.X + 0.1, dtype=np.float64)


def _null_f64_ptr():
    return ctypes.POINTER(ctypes.c_double)()


def _column_intervals(cols: int, width: int, step: int | None = None) -> list[tuple[int, int]]:
    width = max(1, int(width))
    stride = width if step is None else max(1, int(step))
    return [(lo, min(cols, lo + width)) for lo in range(0, cols, stride)]


def c_paired_transfer(prefix: str, ctx: Dataset, *create_args) -> np.ndarray:
    source = as_f64_2d(ctx.X)
    target = _transfer_target(ctx)
    handle = _create(prefix, *create_args)
    try:
        check(
            getattr(lib, f"{prefix}_fit")(
                handle, numpy_to_view(source), numpy_to_view(target)
            ),
            f"{prefix}_fit",
        )
        out = empty_like_f64(source.shape)
        check(
            getattr(lib, f"{prefix}_transform")(
                handle, numpy_to_view(source), numpy_to_view(out)
            ),
            f"{prefix}_transform",
        )
        return out
    finally:
        getattr(lib, f"{prefix}_destroy")(handle)


def c_slope_bias(ctx: Dataset) -> np.ndarray:
    source = np.ascontiguousarray(ctx.y, dtype=np.float64)
    target = np.ascontiguousarray(2.0 * ctx.y + 1.0, dtype=np.float64)
    out = np.empty_like(source)
    handle = _create("c4a_pp_slope_bias")
    try:
        check(
            lib.c4a_pp_slope_bias_fit(
                handle,
                source.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
                target.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
                ctypes.c_int64(source.size),
            ),
            "c4a_pp_slope_bias_fit",
        )
        check(
            lib.c4a_pp_slope_bias_transform(
                handle,
                source.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
                ctypes.c_int64(source.size),
                out.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
            ),
            "c4a_pp_slope_bias_transform",
        )
        return out
    finally:
        lib.c4a_pp_slope_bias_destroy(handle)


def c_weighted_snv(ctx: Dataset) -> np.ndarray:
    weights = np.ascontiguousarray(np.linspace(1.0, 2.0, ctx.X.shape[1]), dtype=np.float64)
    return c_transform(
        "c4a_pp_weighted_snv",
        ctx.X,
        weights.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        ctypes.c_int64(weights.size),
        ctypes.c_int32(0),
        ctypes.c_double(1e-12),
        fit_X=ctx.X,
    )


def c_selector(prefix: str, ctx: Dataset, *, threshold: float = 0.0,
               top_k: int | None = None) -> np.ndarray:
    handle = _create(
        prefix,
        ctypes.c_double(threshold),
        ctypes.c_int32(-1 if top_k is None else int(top_k)),
    )
    try:
        if prefix == "c4a_filter_correlation":
            y = np.ascontiguousarray(ctx.y, dtype=np.float64)
            check(
                lib.c4a_filter_correlation_fit(
                    handle,
                    numpy_to_view(ctx.X),
                    y.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
                    ctypes.c_int64(y.size),
                ),
                "c4a_filter_correlation_fit",
            )
        else:
            check(lib.c4a_filter_variance_fit(handle, numpy_to_view(ctx.X)), "c4a_filter_variance_fit")
        out_cols = ctypes.c_int64()
        check(getattr(lib, f"{prefix}_output_cols")(handle, ctypes.byref(out_cols)), f"{prefix}_output_cols")
        out = empty_like_f64((ctx.X.shape[0], int(out_cols.value)))
        check(getattr(lib, f"{prefix}_transform")(handle, numpy_to_view(ctx.X), numpy_to_view(out)), f"{prefix}_transform")
        return out
    finally:
        getattr(lib, f"{prefix}_destroy")(handle)


def c_interval_generator(ctx: Dataset) -> list[np.ndarray]:
    width = 16
    step = 8
    handle = _create("c4a_interval_generator", ctypes.c_int32(width), ctypes.c_int32(step))
    try:
        check(lib.c4a_interval_generator_fit(handle, numpy_to_view(ctx.X)), "c4a_interval_generator_fit")
        out_cols = ctypes.c_int64()
        check(lib.c4a_interval_generator_output_cols(handle, ctypes.byref(out_cols)), "c4a_interval_generator_output_cols")
        out = empty_like_f64((ctx.X.shape[0], int(out_cols.value)))
        check(lib.c4a_interval_generator_transform(handle, numpy_to_view(ctx.X), numpy_to_view(out)), "c4a_interval_generator_transform")
        intervals = [(lo, min(ctx.X.shape[1], lo + width)) for lo in range(0, ctx.X.shape[1], step)]
        blocks = []
        offset = 0
        for lo, hi in intervals:
            cols = hi - lo
            blocks.append(out[:, offset:offset + cols])
            offset += cols
        return blocks
    finally:
        lib.c4a_interval_generator_destroy(handle)


def c_epo(ctx: Dataset) -> np.ndarray:
    y = np.ascontiguousarray(ctx.d, dtype=np.float64)
    handle = _create("c4a_pp_epo", ctypes.c_int(1))
    try:
        check(
            lib.c4a_pp_epo_fit(
                handle,
                numpy_to_view(ctx.X),
                y.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
                ctypes.c_int64(y.size),
            ),
            "c4a_pp_epo_fit",
        )
        out = empty_like_f64(ctx.X.shape)
        check(
            lib.c4a_pp_epo_transform_with_d(
                handle,
                numpy_to_view(ctx.X),
                y.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
                ctypes.c_int64(y.size),
                numpy_to_view(out),
            ),
            "c4a_pp_epo_transform_with_d",
        )
        return out
    finally:
        lib.c4a_pp_epo_destroy(handle)


def c_fck_static(ctx: Dataset) -> np.ndarray:
    alphas = np.ascontiguousarray([0.5, 1.0], dtype=np.float64)
    sigmas = np.ascontiguousarray([1.0, 2.0], dtype=np.float64)
    handle = _create(
        "c4a_pp_fck_static",
        ctypes.c_int32(5),
        alphas.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        ctypes.c_int32(alphas.size),
        sigmas.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        ctypes.c_int32(sigmas.size),
    )
    try:
        out_cols = ctypes.c_int32()
        check(
            lib.c4a_pp_fck_static_output_cols(
                ctypes.c_int32(alphas.size * sigmas.size),
                ctypes.c_int32(ctx.X.shape[1]),
                ctypes.byref(out_cols),
            ),
            "c4a_pp_fck_static_output_cols",
        )
        out = empty_like_f64((ctx.X.shape[0], int(out_cols.value)))
        check(lib.c4a_pp_fck_static_transform(handle, numpy_to_view(ctx.X), numpy_to_view(out)), "c4a_pp_fck_static_transform")
        return out
    finally:
        lib.c4a_pp_fck_static_destroy(handle)


def c_crop(ctx: Dataset) -> np.ndarray:
    return c_crop_factory(ctx)()


def c_crop_factory(ctx: Dataset) -> Callable[[], np.ndarray]:
    start = max(0, ctx.X.shape[1] // 8)
    end = max(start + 1, ctx.X.shape[1] - ctx.X.shape[1] // 8)
    X_arr = as_f64_2d(ctx.X)
    handle = _create("c4a_pp_crop", ctypes.c_int64(start), ctypes.c_int64(end))
    out = empty_like_f64((X_arr.shape[0], end - start))
    x_view = numpy_to_view(X_arr)
    out_view = numpy_to_view(out)

    def run() -> np.ndarray:
        status = lib.c4a_pp_crop_transform(handle, x_view, out_view)
        if status != 0:
            check(status, "c4a_pp_crop_transform")
        return out

    run._c4a_output = out
    weakref.finalize(run, lib.c4a_pp_crop_destroy, handle)
    return run


def c_resample(ctx: Dataset) -> np.ndarray:
    target = max(4, ctx.X.shape[1] // 2)
    return c_transform("c4a_pp_resample", ctx.X, ctypes.c_int64(target), out_shape=(ctx.X.shape[0], target))


def c_resampler(ctx: Dataset) -> np.ndarray:
    target = np.ascontiguousarray(np.linspace(ctx.wavelengths[0], ctx.wavelengths[-1], max(4, ctx.X.shape[1] // 2)), dtype=np.float64)
    handle = _create(
        "c4a_pp_resampler",
        target.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        ctypes.c_int64(target.size),
        ctypes.c_int32(0),
        ctypes.c_double(0.0),
        ctypes.c_double(0.0),
        ctypes.c_int(0),
        ctypes.c_double(0.0),
        ctypes.c_int(0),
        ctypes.c_int(0),
    )
    try:
        source = np.ascontiguousarray(ctx.wavelengths, dtype=np.float64)
        check(
            lib.c4a_pp_resampler_fit(
                handle,
                source.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
                ctypes.c_int64(source.size),
            ),
            "c4a_pp_resampler_fit",
        )
        out_cols = int(lib.c4a_pp_resampler_output_cols(handle))
        out = empty_like_f64((ctx.X.shape[0], out_cols))
        check(lib.c4a_pp_resampler_transform(handle, numpy_to_view(ctx.X), numpy_to_view(out)), "c4a_pp_resampler_transform")
        return out
    finally:
        lib.c4a_pp_resampler_destroy(handle)


def c_kbins(ctx: Dataset) -> np.ndarray:
    return c_transform(
        "c4a_pp_kbins_disc",
        ctx.X,
        ctypes.c_int32(5),
        ctypes.c_int32(0),
        fit_X=ctx.X,
        out_dtype="i32",
    )


def c_range_disc(ctx: Dataset) -> np.ndarray:
    edges = np.ascontiguousarray([0.25, 0.40, 0.55, 0.70], dtype=np.float64)
    return c_transform(
        "c4a_pp_range_disc",
        ctx.X,
        edges.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        ctypes.c_int64(edges.size),
        out_dtype="i32",
    )


def c_wavelet_features(ctx: Dataset, *, mode: int = 0, entropy: int = 0) -> np.ndarray:
    handle = _create_ex(
        "c4a_pp_wavelet_features",
        ctypes.c_int(0),
        ctypes.c_int(mode),
        ctypes.c_int32(2),
        ctypes.c_int(entropy),
    )
    try:
        out_cols = ctypes.c_int64()
        check(
            lib.c4a_pp_wavelet_features_output_cols(
                handle,
                ctypes.c_int64(ctx.X.shape[1]),
                ctypes.byref(out_cols),
            ),
            "c4a_pp_wavelet_features_output_cols",
        )
        out = empty_like_f64((ctx.X.shape[0], int(out_cols.value)))
        check(
            lib.c4a_pp_wavelet_features_transform(
                handle,
                numpy_to_view(ctx.X),
                numpy_to_view(out),
            ),
            "c4a_pp_wavelet_features_transform",
        )
        return out
    finally:
        lib.c4a_pp_wavelet_features_destroy(handle)


def c_log_transform(ctx: Dataset) -> np.ndarray:
    return c_transform(
        "c4a_pp_log",
        ctx.X,
        ctypes.c_double(0.0),
        ctypes.c_double(0.0),
        ctypes.c_int(1),
        ctypes.c_double(1e-8),
        fit_X=ctx.X,
    )


def c_derivate(ctx: Dataset) -> np.ndarray:
    out_cols = int(
        lib.c4a_pp_derivate_output_cols(
            ctypes.c_int32(1), ctypes.c_int64(ctx.X.shape[1])
        )
    )
    return c_transform(
        "c4a_pp_derivate",
        ctx.X,
        ctypes.c_int32(1),
        ctypes.c_double(1.0),
        fit_X=ctx.X,
        out_shape=(ctx.X.shape[0], out_cols),
    )


def c_wavelet_projection(prefix: str, ctx: Dataset) -> np.ndarray:
    handle = _create(
        prefix,
        ctypes.c_int(0),
        ctypes.c_int(0),
        ctypes.c_int32(2),
        ctypes.c_double(5.0),
    )
    try:
        check(getattr(lib, f"{prefix}_fit")(handle, numpy_to_view(ctx.X)), f"{prefix}_fit")
        out_cols = ctypes.c_int64()
        check(getattr(lib, f"{prefix}_output_cols")(handle, ctypes.byref(out_cols)), f"{prefix}_output_cols")
        out = empty_like_f64((ctx.X.shape[0], int(out_cols.value)))
        check(getattr(lib, f"{prefix}_transform")(handle, numpy_to_view(ctx.X), numpy_to_view(out)), f"{prefix}_transform")
        return out
    finally:
        getattr(lib, f"{prefix}_destroy")(handle)


def _aug_wavelength_ptr(ctx: Dataset, *, required: bool = False) -> tuple[ctypes.POINTER(ctypes.c_double), ctypes.c_int64, np.ndarray | None]:
    if required:
        wl = np.ascontiguousarray(ctx.wavelengths, dtype=np.float64)
        return wl.ctypes.data_as(ctypes.POINTER(ctypes.c_double)), ctypes.c_int64(wl.size), wl
    return _null_f64_ptr(), ctypes.c_int64(0), None


def c_aug_apply(
    prefix: str,
    ctx: Dataset,
    *create_args,
    seed: int = 17,
    apply_wavelengths: bool = False,
) -> np.ndarray:
    X_arr = as_f64_2d(ctx.X)
    rng = PCG64(seed)
    handle = ctypes.c_void_p()
    try:
        check(
            getattr(lib, f"{prefix}_create")(ctypes.byref(handle), rng.handle, *create_args),
            f"{prefix}_create",
        )
        out = empty_like_f64(X_arr.shape)
        if apply_wavelengths:
            wl = np.ascontiguousarray(ctx.wavelengths, dtype=np.float64)
            check(
                getattr(lib, f"{prefix}_apply")(
                    handle,
                    numpy_to_view(X_arr),
                    numpy_to_view(wl.reshape(1, -1)),
                    numpy_to_view(out),
                ),
                f"{prefix}_apply",
            )
        else:
            check(
                getattr(lib, f"{prefix}_apply")(
                    handle, numpy_to_view(X_arr), numpy_to_view(out)
                ),
                f"{prefix}_apply",
            )
        return out
    finally:
        if handle.value:
            getattr(lib, f"{prefix}_destroy")(handle)
        rng.close()


def c_aug_poly_drift(ctx: Dataset) -> np.ndarray:
    lo = np.ascontiguousarray(np.full(3, -0.01, dtype=np.float64))
    hi = np.ascontiguousarray(np.full(3, 0.01, dtype=np.float64))
    return c_aug_apply(
        "c4a_aug_poly_drift",
        ctx,
        ctypes.c_int32(2),
        lo.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        hi.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
    )


def c_aug_wavelength_shift(ctx: Dataset) -> np.ndarray:
    wl_ptr, wl_n, _wl = _aug_wavelength_ptr(ctx, required=True)
    return c_aug_apply("c4a_aug_wavelength_shift", ctx, ctypes.c_double(-1.0), ctypes.c_double(1.0), wl_ptr, wl_n)


def c_aug_wavelength_stretch(ctx: Dataset) -> np.ndarray:
    wl_ptr, wl_n, _wl = _aug_wavelength_ptr(ctx, required=True)
    return c_aug_apply("c4a_aug_wavelength_stretch", ctx, ctypes.c_double(0.99), ctypes.c_double(1.01), wl_ptr, wl_n)


def c_aug_local_warp(ctx: Dataset) -> np.ndarray:
    wl_ptr, wl_n, _wl = _aug_wavelength_ptr(ctx, required=True)
    return c_aug_apply("c4a_aug_local_warp", ctx, ctypes.c_int32(3), ctypes.c_double(1.0), wl_ptr, wl_n)


def c_aug_band_perturb(ctx: Dataset) -> np.ndarray:
    return c_aug_apply("c4a_aug_band_perturb", ctx, ctypes.c_int32(3), ctypes.c_int32(5), ctypes.c_int32(15), ctypes.c_double(0.9), ctypes.c_double(1.1), ctypes.c_double(-0.01), ctypes.c_double(0.01))


def c_aug_band_mask(ctx: Dataset) -> np.ndarray:
    return c_aug_apply("c4a_aug_band_mask", ctx, ctypes.c_int32(1), ctypes.c_int32(3), ctypes.c_int32(5), ctypes.c_int32(15), ctypes.c_int32(0))


def c_aug_channel_dropout(ctx: Dataset) -> np.ndarray:
    return c_aug_apply("c4a_aug_channel_dropout", ctx, ctypes.c_double(0.05), ctypes.c_int32(0))


def c_aug_gauss_jitter(ctx: Dataset) -> np.ndarray:
    return c_aug_apply("c4a_aug_gauss_jitter", ctx, ctypes.c_double(0.5), ctypes.c_double(1.5), ctypes.c_int32(9))


def c_aug_unsharp_mask(ctx: Dataset) -> np.ndarray:
    return c_aug_apply("c4a_aug_unsharp_mask", ctx, ctypes.c_double(0.1), ctypes.c_double(0.5), ctypes.c_double(1.0), ctypes.c_int32(11))


def c_aug_magnitude_warp(ctx: Dataset) -> np.ndarray:
    wl_ptr, wl_n, _wl = _aug_wavelength_ptr(ctx, required=True)
    return c_aug_apply("c4a_aug_magnitude_warp", ctx, ctypes.c_int32(3), ctypes.c_double(0.9), ctypes.c_double(1.1), wl_ptr, wl_n)


def c_aug_local_clip(ctx: Dataset) -> np.ndarray:
    return c_aug_apply("c4a_aug_local_clip", ctx, ctypes.c_int32(1), ctypes.c_int32(5), ctypes.c_int32(15))


def c_aug_wavelength_spectral(ctx: Dataset) -> np.ndarray:
    parts = [
        c_aug_wavelength_shift(ctx),
        c_aug_wavelength_stretch(ctx),
        c_aug_local_warp(ctx),
        c_aug_band_perturb(ctx),
        c_aug_band_mask(ctx),
        c_aug_channel_dropout(ctx),
        c_aug_gauss_jitter(ctx),
        c_aug_unsharp_mask(ctx),
        c_aug_magnitude_warp(ctx),
        c_aug_local_clip(ctx),
    ]
    return np.concatenate(parts, axis=1)


def c_aug_particle_size(ctx: Dataset) -> np.ndarray:
    wl_ptr, wl_n, _wl = _aug_wavelength_ptr(ctx, required=True)
    return c_aug_apply(
        "c4a_aug_particle_size",
        ctx,
        ctypes.c_double(50.0),
        ctypes.c_double(15.0),
        ctypes.c_int(0),
        ctypes.c_double(5.0),
        ctypes.c_double(500.0),
        ctypes.c_double(50.0),
        ctypes.c_double(1.5),
        ctypes.c_double(0.1),
        ctypes.c_int(1),
        ctypes.c_double(0.5),
        wl_ptr,
        wl_n,
    )


def c_aug_emsc_distort(ctx: Dataset) -> np.ndarray:
    wl_ptr, wl_n, _wl = _aug_wavelength_ptr(ctx, required=True)
    return c_aug_apply(
        "c4a_aug_emsc_distort",
        ctx,
        ctypes.c_double(0.9),
        ctypes.c_double(1.1),
        ctypes.c_double(-0.05),
        ctypes.c_double(0.05),
        ctypes.c_int32(2),
        ctypes.c_double(0.02),
        ctypes.c_double(0.3),
        wl_ptr,
        wl_n,
    )


def c_aug_batch_effect(ctx: Dataset) -> np.ndarray:
    wl_ptr, wl_n, _wl = _aug_wavelength_ptr(ctx, required=True)
    return c_aug_apply(
        "c4a_aug_batch_effect",
        ctx,
        ctypes.c_double(0.02),
        ctypes.c_double(0.01),
        ctypes.c_double(0.03),
        ctypes.c_int32(0),
        wl_ptr,
        wl_n,
    )


def c_aug_instrument_broaden(ctx: Dataset) -> np.ndarray:
    wl_ptr, wl_n, _wl = _aug_wavelength_ptr(ctx, required=True)
    return c_aug_apply(
        "c4a_aug_instrument_broaden",
        ctx,
        ctypes.c_double(3.0),
        ctypes.c_int(0),
        ctypes.c_double(3.0),
        ctypes.c_double(8.0),
        ctypes.c_int32(0),
        wl_ptr,
        wl_n,
    )


def c_aug_temperature(ctx: Dataset) -> np.ndarray:
    wl_ptr, wl_n, _wl = _aug_wavelength_ptr(ctx, required=True)
    return c_aug_apply(
        "c4a_aug_temperature",
        ctx,
        ctypes.c_double(5.0),
        ctypes.c_int(0),
        ctypes.c_double(-5.0),
        ctypes.c_double(5.0),
        ctypes.c_int(1),
        ctypes.c_int(1),
        ctypes.c_int(1),
        ctypes.c_int(1),
        wl_ptr,
        wl_n,
    )


def c_aug_moisture(ctx: Dataset) -> np.ndarray:
    wl_ptr, wl_n, _wl = _aug_wavelength_ptr(ctx, required=True)
    return c_aug_apply(
        "c4a_aug_moisture",
        ctx,
        ctypes.c_double(0.1),
        ctypes.c_int(0),
        ctypes.c_double(0.0),
        ctypes.c_double(1.0),
        ctypes.c_double(0.5),
        ctypes.c_double(0.3),
        ctypes.c_double(25.0),
        ctypes.c_double(0.10),
        ctypes.c_int(1),
        ctypes.c_int(1),
        wl_ptr,
        wl_n,
    )


def c_aug_detector_rolloff(ctx: Dataset) -> np.ndarray:
    return c_aug_apply("c4a_aug_detector_rolloff", ctx, ctypes.c_int32(4), ctypes.c_double(1.0), ctypes.c_double(0.02), ctypes.c_int32(1), apply_wavelengths=True)


def c_aug_stray_light(ctx: Dataset) -> np.ndarray:
    return c_aug_apply("c4a_aug_stray_light", ctx, ctypes.c_double(0.001), ctypes.c_double(2.0), ctypes.c_double(0.1), ctypes.c_int32(1), apply_wavelengths=True)


def c_aug_edge_curve(ctx: Dataset) -> np.ndarray:
    return c_aug_apply("c4a_aug_edge_curve", ctx, ctypes.c_double(0.02), ctypes.c_int32(0), ctypes.c_double(0.0), ctypes.c_double(0.7), apply_wavelengths=True)


def c_aug_truncated_peak(ctx: Dataset) -> np.ndarray:
    return c_aug_apply("c4a_aug_truncated_peak", ctx, ctypes.c_double(0.5), ctypes.c_double(0.01), ctypes.c_double(0.1), ctypes.c_double(50.0), ctypes.c_double(200.0), ctypes.c_int32(1), ctypes.c_int32(1), apply_wavelengths=True)


def c_aug_edge_artifacts(ctx: Dataset) -> np.ndarray:
    return c_aug_apply("c4a_aug_edge_artifacts", ctx, ctypes.c_int32(0xF), ctypes.c_double(1.0), ctypes.c_int32(4), apply_wavelengths=True)


def c_aug_spline_smooth(ctx: Dataset) -> np.ndarray:
    return c_aug_apply("c4a_aug_spline_smooth", ctx)


def c_aug_spline_x_perturb(ctx: Dataset) -> np.ndarray:
    return c_aug_apply("c4a_aug_spline_x_perturb", ctx, ctypes.c_int32(3), ctypes.c_double(0.05), ctypes.c_double(-0.1), ctypes.c_double(0.1))


def c_aug_spline_y_perturb(ctx: Dataset) -> np.ndarray:
    return c_aug_apply("c4a_aug_spline_y_perturb", ctx, ctypes.c_int32(-1), ctypes.c_double(0.005))


def c_aug_rotate_translate(ctx: Dataset) -> np.ndarray:
    return c_aug_apply("c4a_aug_rotate_translate", ctx, ctypes.c_double(2.0), ctypes.c_double(3.0))


def c_aug_random_x_op(ctx: Dataset) -> np.ndarray:
    return c_aug_apply("c4a_aug_random_x_op", ctx, ctypes.c_int32(0), ctypes.c_double(0.97), ctypes.c_double(1.03))


def c_high_leverage(ctx: Dataset) -> np.ndarray:
    return c4a_python.high_leverage_filter(ctx.X)


def c_spectral_quality(ctx: Dataset) -> np.ndarray:
    return c4a_python.spectral_quality_filter(ctx.X)


def c_composite_filter(ctx: Dataset) -> np.ndarray:
    return c4a_python.composite_filter(ctx.X)


def c_hotelling_t2(ctx: Dataset) -> tuple[np.ndarray, float]:
    return c4a_python.hotelling_t2(ctx.X, n_components=min(5, ctx.X.shape[1]), alpha=0.05)


def c_q_residuals(ctx: Dataset) -> tuple[np.ndarray, float]:
    return c4a_python.q_residuals(ctx.X, n_components=min(5, ctx.X.shape[1]), alpha=0.05)


def c_nirs_metrics(ctx: Dataset) -> np.ndarray:
    return c4a_python.nirs_metrics(ctx.y, 1.05 * ctx.y + 0.02)


def c_signal_type(ctx: Dataset) -> np.ndarray:
    return c4a_python.signal_type_detector(ctx.X, ctx.wavelengths, confidence_threshold=0.7)


def c_rng_pcg64(ctx: Dataset) -> np.ndarray:
    return c4a_python.rng_pcg64(seed=ctx.seed, n=1024)


def c_transfer_metrics(ctx: Dataset) -> np.ndarray:
    return c4a_python.transfer_metrics(
        ctx.X,
        _transfer_target(ctx),
        n_components=min(5, ctx.X.shape[0] - 1, ctx.X.shape[1]),
        k_neighbors=min(10, max(2, ctx.X.shape[0] - 1)),
        seed=ctx.seed,
    )


def c_iasls(ctx: Dataset) -> np.ndarray:
    handle = _create_ex(
        "c4a_pp_iasls",
        ctypes.c_double(1e5),
        ctypes.c_double(0.01),
        ctypes.c_double(1e-4),
        ctypes.c_int32(2),
        ctypes.c_int32(2),
        ctypes.c_int32(20),
        ctypes.c_double(1e-3),
    )
    try:
        out = empty_like_f64(ctx.X.shape)
        check(
            lib.c4a_pp_iasls_transform(handle, numpy_to_view(ctx.X), numpy_to_view(out)),
            "c4a_pp_iasls_transform",
        )
        return out
    finally:
        lib.c4a_pp_iasls_destroy(handle)


def c_flexible(prefix: str, ctx: Dataset, n_components: float) -> np.ndarray:
    handle = _create(prefix, ctypes.c_double(n_components))
    try:
        check(getattr(lib, f"{prefix}_fit")(handle, numpy_to_view(ctx.X)), f"{prefix}_fit")
        out_cols = ctypes.c_int64()
        check(getattr(lib, f"{prefix}_output_cols")(handle, ctypes.byref(out_cols)), f"{prefix}_output_cols")
        out = empty_like_f64((ctx.X.shape[0], int(out_cols.value)))
        check(getattr(lib, f"{prefix}_transform")(handle, numpy_to_view(ctx.X), numpy_to_view(out)), f"{prefix}_transform")
        return out
    finally:
        getattr(lib, f"{prefix}_destroy")(handle)


def py_transform(cls, *args, **kwargs) -> Callable[[Dataset], Callable[[], object]]:
    def factory(ctx: Dataset) -> Callable[[], object]:
        def run() -> object:
            return cls(*args, **kwargs).fit_transform(ctx.X)

        return run

    return factory


def py_resampler(ctx: Dataset) -> Callable[[], object]:
    target = np.linspace(ctx.wavelengths[0], ctx.wavelengths[-1], max(4, ctx.X.shape[1] // 2))

    def run() -> object:
        return c4a.Resampler(target_wavelengths=target).fit_transform(ctx.X, source_wavelengths=ctx.wavelengths)

    return run


def py_crop(ctx: Dataset) -> Callable[[], object]:
    start = max(0, ctx.X.shape[1] // 8)
    end = max(start + 1, ctx.X.shape[1] - ctx.X.shape[1] // 8)
    return lambda: c4a.CropTransformer(start=start, end=end).fit_transform(ctx.X)


def py_resample(ctx: Dataset) -> Callable[[], object]:
    return lambda: c4a.ResampleTransformer(num_samples=max(4, ctx.X.shape[1] // 2)).fit_transform(ctx.X)


def py_range_disc(ctx: Dataset) -> Callable[[], object]:
    return lambda: c4a.RangeDiscretizer(edges=[0.25, 0.40, 0.55, 0.70]).fit_transform(ctx.X)


def py_kbins(ctx: Dataset) -> Callable[[], object]:
    return lambda: c4a.IntegerKBinsDiscretizer(n_bins=5, strategy="uniform").fit_transform(ctx.X)


def py_osc(ctx: Dataset) -> Callable[[], object]:
    return lambda: c4a.OSC(n_components=1, scale=True).fit_transform(ctx.X, ctx.y)


def py_epo(ctx: Dataset) -> Callable[[], object]:
    return lambda: c4a.EPO(scale=True).fit_transform(ctx.X, ctx.d)


def py_fck(ctx: Dataset) -> Callable[[], object]:
    return lambda: c4a.FCKStaticTransformer(kernel_size=5, alphas=[0.5, 1.0], sigmas=[1.0, 2.0]).fit_transform(ctx.X)


def py_ks(ctx: Dataset) -> Callable[[], object]:
    return lambda: c4a.KennardStoneSplitter(test_size=0.25).split(ctx.X)


def py_spxy(ctx: Dataset) -> Callable[[], object]:
    return lambda: c4a.SPXYSplitter(test_size=0.25).split(ctx.X, ctx.y.reshape(-1, 1))


def py_kbins_stratified(ctx: Dataset) -> Callable[[], object]:
    return lambda: c4a.KBinsStratifiedSplitter(test_size=0.25, seed=17, n_bins=5, strategy="uniform").split(ctx.y.reshape(-1, 1))


def py_y_outlier(ctx: Dataset) -> Callable[[], object]:
    return lambda: c4a.YOutlierFilter(method="iqr").fit_apply(ctx.y)[0]


def py_x_outlier(ctx: Dataset) -> Callable[[], object]:
    return lambda: c4a.XOutlierFilter(method="mahalanobis", n_components=min(5, ctx.X.shape[1])).fit_apply(ctx.X)[0]


def py_paired_transfer(cls, *args, **kwargs) -> Callable[[Dataset], Callable[[], object]]:
    def factory(ctx: Dataset) -> Callable[[], object]:
        target = _transfer_target(ctx)
        return lambda: cls(*args, **kwargs).fit(ctx.X, target).transform(ctx.X)

    return factory


def py_slope_bias(ctx: Dataset) -> Callable[[], object]:
    target = 2.0 * ctx.y + 1.0
    return lambda: c4a.SlopeBiasCorrection().fit(ctx.y, target).transform(ctx.y)


def py_weighted_snv(ctx: Dataset) -> Callable[[], object]:
    weights = np.linspace(1.0, 2.0, ctx.X.shape[1])
    return lambda: c4a.WeightedSNV(weights=weights).fit_transform(ctx.X)


def py_variance_filter(ctx: Dataset) -> Callable[[], object]:
    return lambda: c4a.VarianceFilter(threshold=0.0).fit_transform(ctx.X)


def py_correlation_filter(ctx: Dataset) -> Callable[[], object]:
    return lambda: c4a.CorrelationFilter(top_k=min(5, ctx.X.shape[1])).fit_transform(ctx.X, ctx.y)


def py_interval_generator(ctx: Dataset) -> Callable[[], object]:
    return lambda: c4a.IntervalGenerator(interval_size=16, step=8).fit_transform(ctx.X)


def pyabi_transform(name: str, *args, **kwargs) -> Callable[[Dataset], Callable[[], object]]:
    fn = getattr(c4a_python, name)

    def factory(ctx: Dataset) -> Callable[[], object]:
        return lambda: fn(ctx.X, *args, **kwargs)

    return factory


def pyabi_resampler(ctx: Dataset) -> Callable[[], object]:
    target = np.linspace(ctx.wavelengths[0], ctx.wavelengths[-1], max(4, ctx.X.shape[1] // 2))
    return lambda: c4a_python.resampler(ctx.X, ctx.wavelengths, target)


def pyabi_crop(ctx: Dataset) -> Callable[[], object]:
    start = max(0, ctx.X.shape[1] // 8)
    end = max(start + 1, ctx.X.shape[1] - ctx.X.shape[1] // 8)
    return lambda: c4a_python.crop(ctx.X, start, end)


def pyabi_resample(ctx: Dataset) -> Callable[[], object]:
    return lambda: c4a_python.resample(ctx.X, max(4, ctx.X.shape[1] // 2))


def pyabi_range_disc(ctx: Dataset) -> Callable[[], object]:
    return lambda: c4a_python.range_discretizer(ctx.X, [0.25, 0.40, 0.55, 0.70])


def pyabi_kbins(ctx: Dataset) -> Callable[[], object]:
    return lambda: c4a_python.kbins_discretizer(ctx.X, n_bins=5, strategy="uniform")


def pyabi_osc(ctx: Dataset) -> Callable[[], object]:
    return lambda: c4a_python.osc(ctx.X, ctx.y, n_components=1, scale=True)


def pyabi_epo(ctx: Dataset) -> Callable[[], object]:
    return lambda: c4a_python.epo(ctx.X, ctx.d, scale=True)


def pyabi_fck(ctx: Dataset) -> Callable[[], object]:
    return lambda: c4a_python.fck_static(ctx.X, kernel_size=5, alphas=[0.5, 1.0], sigmas=[1.0, 2.0])


def pyabi_ks(ctx: Dataset) -> Callable[[], object]:
    return lambda: c4a_python.kennard_stone(ctx.X, test_size=0.25)


def pyabi_spxy(ctx: Dataset) -> Callable[[], object]:
    return lambda: c4a_python.spxy(ctx.X, ctx.y.reshape(-1, 1), test_size=0.25)


def pyabi_kbins_stratified(ctx: Dataset) -> Callable[[], object]:
    return lambda: c4a_python.kbins_stratified(ctx.y.reshape(-1, 1), test_size=0.25, seed=17, n_bins=5, strategy="uniform")


def pyabi_y_outlier(ctx: Dataset) -> Callable[[], object]:
    return lambda: c4a_python.y_outlier_iqr(ctx.y)


def pyabi_x_outlier(ctx: Dataset) -> Callable[[], object]:
    return lambda: c4a_python.x_outlier_mahalanobis(ctx.X, n_components=min(5, ctx.X.shape[1]))


def pyabi_paired(name: str, *args, **kwargs) -> Callable[[Dataset], Callable[[], object]]:
    fn = getattr(c4a_python, name)

    def factory(ctx: Dataset) -> Callable[[], object]:
        target = _transfer_target(ctx)
        return lambda: fn(ctx.X, target, ctx.X, *args, **kwargs)

    return factory


def pyabi_slope_bias(ctx: Dataset) -> Callable[[], object]:
    target = 2.0 * ctx.y + 1.0
    return lambda: c4a_python.slope_bias_correction(ctx.y, target, ctx.y)


def pyabi_weighted_snv(ctx: Dataset) -> Callable[[], object]:
    weights = np.linspace(1.0, 2.0, ctx.X.shape[1])
    return lambda: c4a_python.weighted_snv(ctx.X, weights=weights)


def pyabi_variance_filter(ctx: Dataset) -> Callable[[], object]:
    return lambda: c4a_python.variance_filter(ctx.X, threshold=0.0)


def pyabi_correlation_filter(ctx: Dataset) -> Callable[[], object]:
    return lambda: c4a_python.correlation_filter(ctx.X, ctx.y, top_k=min(5, ctx.X.shape[1]))


def pyabi_interval_generator(ctx: Dataset) -> Callable[[], object]:
    return lambda: c4a_python.interval_generator(ctx.X, interval_size=16, step=8)


def pyabi_with_wavelengths(name: str, *args, **kwargs) -> Callable[[Dataset], Callable[[], object]]:
    fn = getattr(c4a_python, name)

    def factory(ctx: Dataset) -> Callable[[], object]:
        return lambda: fn(ctx.X, *args, wavelengths=ctx.wavelengths, **kwargs)

    return factory


def pyabi_nirs_metrics(ctx: Dataset) -> Callable[[], object]:
    return lambda: c4a_python.nirs_metrics(ctx.y, 1.05 * ctx.y + 0.02)


def pyabi_signal_type(ctx: Dataset) -> Callable[[], object]:
    return lambda: c4a_python.signal_type_detector(ctx.X, ctx.wavelengths, confidence_threshold=0.7)


def pyabi_rng_pcg64(ctx: Dataset) -> Callable[[], object]:
    return lambda: c4a_python.rng_pcg64(seed=ctx.seed, n=1024)


def pyabi_transfer_metrics(ctx: Dataset) -> Callable[[], object]:
    return lambda: c4a_python.transfer_metrics(
        ctx.X,
        _transfer_target(ctx),
        n_components=min(5, ctx.X.shape[0] - 1, ctx.X.shape[1]),
        k_neighbors=min(10, max(2, ctx.X.shape[0] - 1)),
        seed=ctx.seed,
    )


def py_wavelength_spectral(ctx: Dataset) -> Callable[[], object]:
    classes = (
        lambda: c4a.WavelengthShift(seed=17),
        lambda: c4a.WavelengthStretch(seed=17),
        lambda: c4a.LocalWarpAugmenter(n_control_points=3, seed=17),
        lambda: c4a.BandPerturbationAugmenter(seed=17),
        lambda: c4a.BandMasking(seed=17),
        lambda: c4a.ChannelDropout(seed=17),
        lambda: c4a.GaussianJitter(seed=17),
        lambda: c4a.UnsharpMask(seed=17),
        lambda: c4a.MagnitudeWarp(n_control_points=3, seed=17),
        lambda: c4a.LocalClip(seed=17),
    )

    def run() -> object:
        parts = []
        for make in classes:
            op = make()
            if hasattr(op, "wavelengths"):
                op.wavelengths = np.array(ctx.wavelengths, copy=True)
            parts.append(op.fit_transform(ctx.X))
        return np.concatenate(parts, axis=1)

    return run


def py_high_leverage(ctx: Dataset) -> Callable[[], object]:
    return lambda: c4a.HighLeverageFilter().fit_apply(ctx.X)[0]


def py_spectral_quality(ctx: Dataset) -> Callable[[], object]:
    return lambda: c4a.SpectralQualityFilter().fit_apply(ctx.X)[0]


def py_composite_filter(ctx: Dataset) -> Callable[[], object]:
    def run() -> object:
        leverage = c4a.HighLeverageFilter().fit(ctx.X)
        quality = c4a.SpectralQualityFilter()
        return c4a.CompositeFilter(filters=[leverage, quality]).fit_apply(ctx.X)[0]

    return run


def py_hotelling_t2(ctx: Dataset) -> Callable[[], object]:
    return lambda: c4a.hotelling_t2(ctx.X, n_components=min(5, ctx.X.shape[1]), alpha=0.05)


def py_q_residuals(ctx: Dataset) -> Callable[[], object]:
    return lambda: c4a.q_residuals(ctx.X, n_components=min(5, ctx.X.shape[1]), alpha=0.05)


def py_nirs_metrics(ctx: Dataset) -> Callable[[], object]:
    return lambda: np.asarray([
        c4a.rmse(ctx.y, 1.05 * ctx.y + 0.02),
        c4a.mae(ctx.y, 1.05 * ctx.y + 0.02),
        c4a.bias(ctx.y, 1.05 * ctx.y + 0.02),
        c4a.sep(ctx.y, 1.05 * ctx.y + 0.02),
        c4a.rpd(ctx.y, 1.05 * ctx.y + 0.02),
        c4a.rpiq(ctx.y, 1.05 * ctx.y + 0.02),
        c4a.r2(ctx.y, 1.05 * ctx.y + 0.02),
        c4a.nrmse(ctx.y, 1.05 * ctx.y + 0.02),
    ], dtype=np.float64)


def py_transfer_metrics(ctx: Dataset) -> Callable[[], object]:
    def run() -> object:
        raw = c4a.transfer_metrics(
            ctx.X,
            _transfer_target(ctx),
            n_components=min(5, ctx.X.shape[0] - 1, ctx.X.shape[1]),
            k_neighbors=min(10, max(2, ctx.X.shape[0] - 1)),
            seed=ctx.seed,
        )
        if isinstance(raw, dict):
            return np.asarray([raw[name] for name, _ctype in TransferMetrics._fields_], dtype=np.float64)
        return np.asarray(raw, dtype=np.float64)

    return run


def py_signal_type(ctx: Dataset) -> Callable[[], object]:
    def run() -> object:
        raw = c4a.signal_type_detector(ctx.X, ctx.wavelengths, confidence_threshold=0.7)
        if isinstance(raw, dict):
            return np.asarray([raw["type"], raw["confidence"]], dtype=np.float64)
        return np.asarray(raw, dtype=np.float64)

    return run


PYTHON_DIRECT_FACTORIES: dict[str, Callable[[Dataset], Callable[[], object]]] = {
    "snv": pyabi_transform("snv", ddof=1),
    "lsnv": pyabi_transform("lsnv", window=11, pad_mode="reflect"),
    "rnv": pyabi_transform("rnv"),
    "area_norm": pyabi_transform("area_norm", method="sum"),
    "normalize": pyabi_transform("normalize", feature_min=-1.0, feature_max=1.0),
    "simple_scale": pyabi_transform("simple_scale"),
    "log_transform": pyabi_transform("log_transform", base=0.0),
    "msc": pyabi_transform("msc"),
    "emsc": pyabi_transform("emsc", degree=1),
    "baseline_center": pyabi_transform("baseline_center"),
    "derivate": pyabi_transform("derivate", order=1, delta=1.0),
    "savitzky_golay": pyabi_transform("savitzky_golay", window_length=11, polyorder=2, deriv=0, mode="mirror"),
    "first_derivative": pyabi_transform("first_derivative", delta=1.0, edge_order=2),
    "second_derivative": pyabi_transform("second_derivative", delta=1.0, edge_order=2),
    "norris_williams": pyabi_transform("norris_williams", segment=5, gap=5, derivative_order=1),
    "gaussian": pyabi_transform("gaussian", sigma=1.0, order=0, mode="reflect"),
    "to_absorbance": pyabi_transform("to_absorbance"),
    "from_absorbance": pyabi_transform("from_absorbance"),
    "pct_to_frac": pyabi_transform("pct_to_frac"),
    "frac_to_pct": pyabi_transform("frac_to_pct"),
    "kubelka_munk": pyabi_transform("kubelka_munk"),
    "detrend": pyabi_transform("detrend", polyorder=1),
    "asls": pyabi_transform("asls", lam=1e5, p=0.01, max_iter=20),
    "airpls": pyabi_transform("airpls", lam=1e5, max_iter=20),
    "arpls": pyabi_transform("arpls", lam=1e5, max_iter=20),
    "modpoly": pyabi_transform("modpoly", polyorder=2, max_iter=50),
    "imodpoly": pyabi_transform("imodpoly", polyorder=2, max_iter=50),
    "snip": pyabi_transform("snip", max_half_window=10),
    "rolling_ball": pyabi_transform("rolling_ball", half_window=10, smooth_half_window=0),
    "iasls": pyabi_transform("iasls", lam=1e5, p=0.01, lam_1=1e-4, polyorder=2, diff_order=2, max_iter=20),
    "beads": pyabi_transform("beads", lam_0=100.0, lam_1=0.5, lam_2=0.5, max_iter=20),
    "wavelet": pyabi_transform("wavelet", family="haar", mode="periodization"),
    "haar": pyabi_transform("haar"),
    "wavelet_denoise": pyabi_transform("wavelet_denoise", family="db4", mode="periodization", level=2),
    "wavelet_features": pyabi_transform("wavelet_features", family="haar", mode="symmetric", max_level=2, entropy="histogram"),
    "wavelet_pca": pyabi_transform("wavelet_pca", family="haar", mode="periodization", max_level=2, n_components=5.0),
    "wavelet_svd": pyabi_transform("wavelet_svd", family="haar", mode="periodization", max_level=2, n_components=5.0),
    "osc": pyabi_osc,
    "epo": pyabi_epo,
    "flexible_pca": pyabi_transform("flexible_pca", n_components=5.0),
    "flexible_svd": pyabi_transform("flexible_svd", n_components=5.0),
    "fck_static": pyabi_fck,
    "direct_standardization": pyabi_paired("direct_standardization"),
    "robust_direct_standardization": pyabi_paired("robust_direct_standardization", max_iter=2),
    "piecewise_direct_standardization": pyabi_paired("piecewise_direct_standardization", window_size=5),
    "score_augmented_projection_standardization": pyabi_paired("score_augmented_projection_standardization", n_components=3),
    "slope_bias_correction": pyabi_slope_bias,
    "local_centering": pyabi_paired("local_centering"),
    "weighted_snv": pyabi_weighted_snv,
    "variable_sorting_normalization": pyabi_transform("variable_sorting_normalization"),
    "piecewise_snv": pyabi_transform("piecewise_snv", window_size=16),
    "piecewise_msc": pyabi_transform("piecewise_msc", window_size=16),
    "localized_msc": pyabi_transform("localized_msc", window_size=17),
    "cross_correlation_alignment": pyabi_transform("cross_correlation_alignment", max_shift=2),
    "icoshift_alignment": pyabi_transform("icoshift_alignment", interval_size=16, max_shift=2),
    "dynamic_time_warping_alignment": pyabi_transform("dynamic_time_warping_alignment"),
    "correlation_optimized_warping": pyabi_transform("correlation_optimized_warping", interval_size=16, max_shift=2),
    "variance_filter": pyabi_variance_filter,
    "correlation_filter": pyabi_correlation_filter,
    "interval_generator": pyabi_interval_generator,
    "crop": pyabi_crop,
    "resample": pyabi_resample,
    "resampler": pyabi_resampler,
    "range_discretizer": pyabi_range_disc,
    "kbins_discretizer": pyabi_kbins,
    "kennard_stone": pyabi_ks,
    "spxy": pyabi_spxy,
    "kbins_stratified": pyabi_kbins_stratified,
    "y_outlier_iqr": pyabi_y_outlier,
    "x_outlier_mahalanobis": pyabi_x_outlier,
    "high_leverage": pyabi_transform("high_leverage_filter"),
    "spectral_quality": pyabi_transform("spectral_quality_filter"),
    "composite_filter": pyabi_transform("composite_filter"),
    "hotelling_t2": pyabi_transform("hotelling_t2", n_components=5, alpha=0.05),
    "q_residuals": pyabi_transform("q_residuals", n_components=5, alpha=0.05),
    "nirs_metrics": pyabi_nirs_metrics,
    "transfer_metrics": pyabi_transfer_metrics,
    "signal_type_detector": pyabi_signal_type,
    "rng_pcg64": pyabi_rng_pcg64,
    "aug_gaussian_noise": pyabi_transform("aug_gaussian_noise", seed=17),
    "aug_multiplicative_noise": pyabi_transform("aug_multiplicative_noise", seed=17),
    "aug_spike_noise": pyabi_transform("aug_spike_noise", seed=17),
    "aug_hetero_noise": pyabi_transform("aug_hetero_noise", seed=17),
    "aug_linear_drift": pyabi_transform("aug_linear_drift", seed=17),
    "aug_poly_drift": pyabi_transform("aug_poly_drift", seed=17),
    "aug_path_length": pyabi_transform("aug_path_length", seed=17),
    "aug_wavelength_shift": pyabi_with_wavelengths("aug_wavelength_shift", seed=17),
    "aug_wavelength_stretch": pyabi_with_wavelengths("aug_wavelength_stretch", seed=17),
    "aug_local_warp": pyabi_with_wavelengths("aug_local_warp", n_control_points=3, seed=17),
    "aug_band_perturb": pyabi_transform("aug_band_perturb", seed=17),
    "aug_band_mask": pyabi_transform("aug_band_mask", seed=17),
    "aug_channel_dropout": pyabi_transform("aug_channel_dropout", seed=17),
    "aug_gauss_jitter": pyabi_transform("aug_gauss_jitter", seed=17),
    "aug_unsharp_mask": pyabi_transform("aug_unsharp_mask", seed=17),
    "aug_magnitude_warp": pyabi_with_wavelengths("aug_magnitude_warp", n_control_points=3, seed=17),
    "aug_local_clip": pyabi_transform("aug_local_clip", seed=17),
    "aug_wavelength_spectral": pyabi_with_wavelengths("aug_wavelength_spectral", seed=17),
    "aug_mixup": pyabi_transform("aug_mixup", alpha=1.0, seed=17),
    "aug_local_mixup": pyabi_transform("aug_local_mixup", alpha=1.0, seed=17),
    "aug_scatter_sim": pyabi_transform("aug_scatter_sim", seed=17),
    "aug_particle_size": pyabi_with_wavelengths("aug_particle_size", seed=17),
    "aug_emsc_distort": pyabi_with_wavelengths("aug_emsc_distort", seed=17),
    "aug_batch_effect": pyabi_with_wavelengths("aug_batch_effect", variation_scope=0, seed=17),
    "aug_instrument_broaden": pyabi_with_wavelengths("aug_instrument_broaden", seed=17),
    "aug_dead_band": pyabi_transform("aug_dead_band", seed=17),
    "aug_temperature": pyabi_with_wavelengths("aug_temperature", seed=17),
    "aug_moisture": pyabi_with_wavelengths("aug_moisture", seed=17),
    "aug_detector_rolloff": pyabi_with_wavelengths("aug_detector_rolloff", seed=17),
    "aug_stray_light": pyabi_with_wavelengths("aug_stray_light", seed=17),
    "aug_edge_curve": pyabi_with_wavelengths("aug_edge_curve", curvature_type=0, seed=17),
    "aug_truncated_peak": pyabi_with_wavelengths("aug_truncated_peak", seed=17),
    "aug_edge_artifacts": pyabi_with_wavelengths("aug_edge_artifacts", seed=17),
    "aug_spline_smooth": pyabi_transform("aug_spline_smooth", seed=17),
    "aug_spline_x_perturb": pyabi_transform("aug_spline_x_perturb", seed=17),
    "aug_spline_y_perturb": pyabi_transform("aug_spline_y_perturb", seed=17),
    "aug_rotate_translate": pyabi_transform("aug_rotate_translate", seed=17),
    "aug_random_x_op": pyabi_transform("aug_random_x_op", seed=17),
}


def python_direct_factory(spec: MethodSpec, ctx: Dataset) -> Callable[[], object]:
    try:
        return PYTHON_DIRECT_FACTORIES[spec.method_id](ctx)
    except KeyError as exc:
        raise RuntimeError(f"no ABI-close Python factory registered for {spec.method_id}") from exc


def ref_scipy_savgol(ctx: Dataset) -> Callable[[], object]:
    signal = importlib.import_module("scipy.signal")
    return lambda: signal.savgol_filter(ctx.X, window_length=11, polyorder=2, deriv=0, delta=1.0, axis=1, mode="mirror")


def ref_scipy_gaussian(ctx: Dataset) -> Callable[[], object]:
    ndimage = importlib.import_module("scipy.ndimage")
    return lambda: ndimage.gaussian_filter1d(ctx.X, sigma=1.0, order=0, axis=1, mode="reflect", cval=0.0, truncate=4.0)


def ref_scipy_detrend(ctx: Dataset) -> Callable[[], object]:
    signal = importlib.import_module("scipy.signal")
    return lambda: signal.detrend(ctx.X, axis=1, type="linear")


def ref_scipy_resample(ctx: Dataset) -> Callable[[], object]:
    interpolate = importlib.import_module("scipy.interpolate")
    target = max(4, ctx.X.shape[1] // 2)
    src = np.linspace(0.0, 1.0, ctx.X.shape[1])
    dst = np.linspace(0.0, 1.0, target)
    return lambda: interpolate.interp1d(src, ctx.X, axis=1, kind="linear")(dst)


def ref_sklearn_simple_scale(ctx: Dataset) -> Callable[[], object]:
    preprocessing = importlib.import_module("sklearn.preprocessing")

    def run() -> object:
        return preprocessing.MinMaxScaler(feature_range=(0.0, 1.0)).fit_transform(ctx.X)

    return run


def ref_sklearn_direct_standardization(ctx: Dataset) -> Callable[[], object]:
    linear_model = importlib.import_module("sklearn.linear_model")
    target = _transfer_target(ctx)

    def run() -> object:
        return linear_model.LinearRegression(fit_intercept=True).fit(ctx.X, target).predict(ctx.X)

    return run


def ref_pycaltransfer_direct_standardization(ctx: Dataset) -> Callable[[], object]:
    caltransfer = importlib.import_module("pycaltransfer.caltransfer")
    target = _transfer_target(ctx)

    def run() -> object:
        transfer, offset = caltransfer.ds_pc_transfer_fit(
            target,
            ctx.X,
            max_ncp=ctx.X.shape[1],
        )
        return ctx.X.dot(transfer) + offset

    return run


def ref_pycaltransfer_piecewise_direct_standardization(ctx: Dataset) -> Callable[[], object]:
    caltransfer = importlib.import_module("pycaltransfer.caltransfer")
    target = _transfer_target(ctx)

    def run() -> object:
        transfer, offset = caltransfer.pds_pls_transfer_fit(
            target,
            ctx.X,
            max_ncp=5,
            ww=2,
        )
        return ctx.X.dot(transfer) + offset

    return run


def ref_sklearn_robust_direct_standardization(ctx: Dataset) -> Callable[[], object]:
    linear_model = importlib.import_module("sklearn.linear_model")
    target = _transfer_target(ctx)

    def run() -> object:
        keep = np.arange(ctx.X.shape[0])
        model = None
        for _ in range(2):
            model = linear_model.LinearRegression(fit_intercept=True).fit(
                ctx.X[keep], target[keep]
            )
            mapped = model.predict(ctx.X)
            residual = np.sqrt(np.sum((mapped - target) * (mapped - target), axis=1))
            cutoff = np.quantile(residual, 0.9, method="linear")
            next_keep = np.nonzero(residual <= cutoff)[0]
            if next_keep.size <= ctx.X.shape[1]:
                next_keep = np.arange(ctx.X.shape[0])
            keep = next_keep
        assert model is not None
        return model.predict(ctx.X)

    return run


def ref_sklearn_piecewise_direct_standardization(ctx: Dataset) -> Callable[[], object]:
    linear_model = importlib.import_module("sklearn.linear_model")
    target = _transfer_target(ctx)
    intervals = [
        (max(0, col - 2), min(ctx.X.shape[1], col + 3))
        for col in range(ctx.X.shape[1])
    ]

    def run() -> object:
        out = np.empty_like(ctx.X)
        for col, (lo, hi) in enumerate(intervals):
            model = linear_model.LinearRegression(fit_intercept=True).fit(
                ctx.X[:, lo:hi], target[:, col]
            )
            out[:, col] = model.predict(ctx.X[:, lo:hi])
        return out

    return run


def ref_sklearn_score_augmented_projection_standardization(ctx: Dataset) -> Callable[[], object]:
    decomposition = importlib.import_module("sklearn.decomposition")
    linear_model = importlib.import_module("sklearn.linear_model")
    target = _transfer_target(ctx)

    def run() -> object:
        components = min(3, ctx.X.shape[0], ctx.X.shape[1])
        pca = decomposition.PCA(n_components=components, svd_solver="full").fit(ctx.X)
        augmented = np.hstack([ctx.X, pca.transform(ctx.X)])
        return linear_model.LinearRegression(fit_intercept=True).fit(
            augmented, target
        ).predict(augmented)

    return run


def ref_sklearn_local_centering(ctx: Dataset) -> Callable[[], object]:
    preprocessing = importlib.import_module("sklearn.preprocessing")
    target = _transfer_target(ctx)

    def run() -> object:
        source_center = preprocessing.StandardScaler(with_mean=True, with_std=False).fit(ctx.X)
        target_center = preprocessing.StandardScaler(with_mean=True, with_std=False).fit(target)
        return source_center.transform(ctx.X) + target_center.mean_

    return run


def ref_statsmodels_weighted_snv(ctx: Dataset) -> Callable[[], object]:
    weightstats = importlib.import_module("statsmodels.stats.weightstats")
    weights = np.linspace(1.0, 2.0, ctx.X.shape[1])

    def run() -> object:
        out = np.empty_like(ctx.X)
        for row_idx, row in enumerate(ctx.X):
            stats = weightstats.DescrStatsW(row, weights=weights, ddof=0)
            scale = math.sqrt(max(float(stats.var), 1e-12))
            out[row_idx] = (row - float(stats.mean)) / scale
        return out

    return run


def ref_statsmodels_variable_sorting_normalization(ctx: Dataset) -> Callable[[], object]:
    feature_selection = importlib.import_module("sklearn.feature_selection")
    weightstats = importlib.import_module("statsmodels.stats.weightstats")
    row_mean = np.mean(ctx.X, axis=1)

    def run() -> object:
        weights = np.abs(feature_selection.r_regression(
            ctx.X, row_mean, center=True, force_finite=True
        ))
        weights = np.maximum(weights, 1e-12)
        out = np.empty_like(ctx.X)
        for row_idx, row in enumerate(ctx.X):
            stats = weightstats.DescrStatsW(row, weights=weights, ddof=0)
            scale = math.sqrt(max(float(stats.var), 1e-12))
            out[row_idx] = (row - float(stats.mean)) / scale
        return out

    return run


def ref_nirs_piecewise_snv(ctx: Dataset) -> Callable[[], object]:
    cls = _nirs_class("nirs4all.operators.transforms.scalers", "StandardNormalVariate")
    bands = _column_intervals(ctx.X.shape[1], 16, 16)

    def run() -> object:
        out = np.empty_like(ctx.X)
        for lo, hi in bands:
            out[:, lo:hi] = cls(ddof=0).fit_transform(np.array(ctx.X[:, lo:hi], copy=True))
        return out

    return run


def ref_sklearn_piecewise_msc(ctx: Dataset) -> Callable[[], object]:
    linear_model = importlib.import_module("sklearn.linear_model")
    reference = np.mean(ctx.X, axis=0)
    bands = _column_intervals(ctx.X.shape[1], 16, 16)

    def run() -> object:
        out = np.empty_like(ctx.X)
        for lo, hi in bands:
            ref_segment = reference[lo:hi].reshape(-1, 1)
            for row_idx, row in enumerate(ctx.X[:, lo:hi]):
                model = linear_model.LinearRegression(fit_intercept=True).fit(
                    ref_segment, row
                )
                slope = float(model.coef_[0])
                if not math.isfinite(slope) or abs(slope) <= 1e-12:
                    slope = -1e-12 if slope < 0.0 else 1e-12
                out[row_idx, lo:hi] = (row - float(model.intercept_)) / slope
        return out

    return run


def ref_sklearn_localized_msc(ctx: Dataset) -> Callable[[], object]:
    linear_model = importlib.import_module("sklearn.linear_model")
    reference = np.mean(ctx.X, axis=0)
    half = 17 // 2

    def run() -> object:
        out = np.empty_like(ctx.X)
        for col in range(ctx.X.shape[1]):
            lo = max(0, col - half)
            hi = min(ctx.X.shape[1], col + half + 1)
            model = linear_model.LinearRegression(fit_intercept=True).fit(
                reference[lo:hi].reshape(-1, 1),
                ctx.X[:, lo:hi].T,
            )
            slopes = np.asarray(model.coef_[:, 0], dtype=np.float64)
            intercepts = np.asarray(model.intercept_, dtype=np.float64)
            safe_slopes = np.where(
                np.isfinite(slopes) & (np.abs(slopes) > 1e-12),
                slopes,
                np.where(slopes < 0.0, -1e-12, 1e-12),
            )
            out[:, col] = (ctx.X[:, col] - intercepts) / safe_slopes
        return out

    return run


def ref_scipy_localized_msc(ctx: Dataset) -> Callable[[], object]:
    ndimage = importlib.import_module("scipy.ndimage")
    reference = np.mean(ctx.X, axis=0)
    kernel = np.ones(17, dtype=np.float64)

    def run() -> object:
        counts = ndimage.convolve1d(
            np.ones(ctx.X.shape[1], dtype=np.float64),
            kernel,
            mode="constant",
            cval=0.0,
        )
        ref_sum = ndimage.convolve1d(reference, kernel, mode="constant", cval=0.0)
        ref2_sum = ndimage.convolve1d(reference * reference, kernel, mode="constant", cval=0.0)
        row_sum = ndimage.convolve1d(ctx.X, kernel, axis=1, mode="constant", cval=0.0)
        row_ref_sum = ndimage.convolve1d(
            ctx.X * reference.reshape(1, -1),
            kernel,
            axis=1,
            mode="constant",
            cval=0.0,
        )
        ref_mean = ref_sum / counts
        row_mean = row_sum / counts.reshape(1, -1)
        denom = np.maximum(ref2_sum - (ref_sum * ref_sum) / counts, 1e-12)
        numer = row_ref_sum - row_sum * ref_sum.reshape(1, -1) / counts.reshape(1, -1)
        slope = numer / denom.reshape(1, -1)
        safe_slope = np.where(
            np.abs(slope) > 1e-12,
            slope,
            np.where(slope < 0.0, -1e-12, 1e-12),
        )
        intercept = row_mean - slope * ref_mean.reshape(1, -1)
        return (ctx.X - intercept) / safe_slope

    return run


def _scipy_shift_alignment(ctx: Dataset, *, interval_size: int | None, max_shift: int) -> Callable[[], object]:
    ndimage = importlib.import_module("scipy.ndimage")
    reference = np.mean(ctx.X, axis=0)
    bands = [(0, ctx.X.shape[1])] if interval_size is None else _column_intervals(
        ctx.X.shape[1], interval_size, interval_size
    )

    def run() -> object:
        out = np.empty_like(ctx.X)
        for lo, hi in bands:
            ref = reference[lo:hi]
            ref_centered = ref - float(np.mean(ref))
            for row_idx, row in enumerate(ctx.X[:, lo:hi]):
                best_shift = 0
                best_score = -math.inf
                for shift in range(-max_shift, max_shift + 1):
                    shifted = ndimage.shift(
                        row,
                        shift=shift,
                        mode="nearest",
                        order=1,
                        prefilter=False,
                    )
                    score = float(np.sum((shifted - float(np.mean(shifted))) * ref_centered))
                    if score > best_score:
                        best_score = score
                        best_shift = shift
                out[row_idx, lo:hi] = ndimage.shift(
                    row,
                    shift=best_shift,
                    mode="nearest",
                    order=1,
                    prefilter=False,
                )
        return out

    return run


def ref_scipy_cross_correlation_alignment(ctx: Dataset) -> Callable[[], object]:
    return _scipy_shift_alignment(ctx, interval_size=None, max_shift=2)


def ref_scipy_icoshift_alignment(ctx: Dataset) -> Callable[[], object]:
    return _scipy_shift_alignment(ctx, interval_size=16, max_shift=2)


def ref_cowarp_correlation_optimized_warping(ctx: Dataset) -> Callable[[], object]:
    cowarp = importlib.import_module("cowarp")
    reference = np.mean(ctx.X, axis=0)

    def run() -> object:
        out = np.empty_like(ctx.X)
        for row_idx, row in enumerate(ctx.X):
            warped, _corr = cowarp.warp(
                reference,
                row,
                auto_segment=False,
                segment_length=16,
                slack=2,
            )
            out[row_idx] = np.asarray(warped, dtype=np.float64)
        return out

    return run


def ref_sklearn_interval_generator(ctx: Dataset) -> Callable[[], object]:
    compose = importlib.import_module("sklearn.compose")
    bands = _column_intervals(ctx.X.shape[1], 16, 8)

    def run() -> object:
        transformers = [
            (f"interval_{idx}", "passthrough", list(range(lo, hi)))
            for idx, (lo, hi) in enumerate(bands)
        ]
        matrix = compose.ColumnTransformer(
            transformers,
            remainder="drop",
            sparse_threshold=0.0,
        ).fit_transform(ctx.X)
        blocks = []
        offset = 0
        for lo, hi in bands:
            width = hi - lo
            blocks.append(np.asarray(matrix[:, offset:offset + width], dtype=np.float64))
            offset += width
        return blocks

    return run


def ref_sklearn_slope_bias(ctx: Dataset) -> Callable[[], object]:
    linear_model = importlib.import_module("sklearn.linear_model")
    target = 2.0 * ctx.y + 1.0

    def run() -> object:
        return linear_model.LinearRegression(fit_intercept=True).fit(
            ctx.y.reshape(-1, 1), target
        ).predict(ctx.y.reshape(-1, 1))

    return run


def ref_pycaltransfer_slope_bias(ctx: Dataset) -> Callable[[], object]:
    caltransfer = importlib.import_module("pycaltransfer.caltransfer")
    target = 2.0 * ctx.y + 1.0

    def run() -> object:
        slope, bias = caltransfer.slope_bias_correction(
            target.reshape(-1, 1),
            ctx.y.reshape(-1, 1),
        )
        return (ctx.y.reshape(-1, 1) * slope + bias).reshape(-1)

    return run


def ref_dtw_python_alignment(ctx: Dataset) -> Callable[[], object]:
    dtw_mod = importlib.import_module("dtw")
    reference = np.mean(ctx.X, axis=0)

    def run() -> object:
        out = np.empty_like(ctx.X)
        for row_idx, row in enumerate(ctx.X):
            alignment = dtw_mod.dtw(
                row,
                reference,
                step_pattern=dtw_mod.symmetric1,
                keep_internals=True,
            )
            sums = np.zeros(ctx.X.shape[1], dtype=np.float64)
            counts = np.zeros(ctx.X.shape[1], dtype=np.int64)
            for query_idx, ref_idx in zip(alignment.index1, alignment.index2):
                sums[int(ref_idx)] += row[int(query_idx)]
                counts[int(ref_idx)] += 1
            aligned = np.array(row, copy=True)
            mask = counts > 0
            aligned[mask] = sums[mask] / counts[mask]
            out[row_idx] = aligned
        return out

    return run


def ref_sklearn_variance_threshold(ctx: Dataset) -> Callable[[], object]:
    feature_selection = importlib.import_module("sklearn.feature_selection")

    def run() -> object:
        return feature_selection.VarianceThreshold(threshold=0.0).fit_transform(ctx.X)

    return run


def ref_sklearn_correlation_filter(ctx: Dataset) -> Callable[[], object]:
    feature_selection = importlib.import_module("sklearn.feature_selection")

    def run() -> object:
        scores = np.abs(feature_selection.r_regression(ctx.X, ctx.y, center=True, force_finite=True))
        indices = list(range(ctx.X.shape[1]))
        indices.sort(key=lambda idx: scores[idx])
        keep = indices[-min(5, ctx.X.shape[1]):]
        return ctx.X[:, keep]

    return run


def ref_sklearn_kbins(ctx: Dataset) -> Callable[[], object]:
    preprocessing = importlib.import_module("sklearn.preprocessing")

    def run() -> object:
        kwargs = {"n_bins": 5, "encode": "ordinal", "strategy": "uniform"}
        try:
            return preprocessing.KBinsDiscretizer(**kwargs, subsample=None).fit_transform(ctx.X).astype(np.int32)
        except TypeError:
            return preprocessing.KBinsDiscretizer(**kwargs).fit_transform(ctx.X).astype(np.int32)

    return run


def ref_sklearn_kbins_stratified(ctx: Dataset) -> Callable[[], object]:
    model_selection = importlib.import_module("sklearn.model_selection")
    preprocessing = importlib.import_module("sklearn.preprocessing")

    def run() -> object:
        kwargs = {"n_bins": 5, "encode": "ordinal", "strategy": "uniform"}
        try:
            binner = preprocessing.KBinsDiscretizer(**kwargs, subsample=None)
        except TypeError:
            binner = preprocessing.KBinsDiscretizer(**kwargs)
        y_disc = binner.fit_transform(ctx.y.reshape(-1, 1)).ravel()
        splitter = model_selection.StratifiedShuffleSplit(n_splits=1, test_size=0.25, random_state=17)
        train, test = next(splitter.split(ctx.X, y_disc))
        return np.asarray(train, dtype=np.int64), np.asarray(test, dtype=np.int64)

    return run


def ref_sklearn_flexible_pca(ctx: Dataset) -> Callable[[], object]:
    decomposition = importlib.import_module("sklearn.decomposition")
    return lambda: decomposition.PCA(n_components=min(5, ctx.X.shape[0], ctx.X.shape[1]), svd_solver="full").fit_transform(ctx.X)


def ref_sklearn_flexible_svd(ctx: Dataset) -> Callable[[], object]:
    decomposition = importlib.import_module("sklearn.decomposition")
    return lambda: decomposition.TruncatedSVD(
        n_components=min(5, ctx.X.shape[0] - 1, ctx.X.shape[1] - 1),
        algorithm="arpack",
        random_state=0,
    ).fit_transform(ctx.X)


def ref_sklearn_x_outlier_mahalanobis(ctx: Dataset) -> Callable[[], object]:
    covariance = importlib.import_module("sklearn.covariance")
    decomposition = importlib.import_module("sklearn.decomposition")
    stats = importlib.import_module("scipy.stats")

    def run() -> object:
        n_components = min(5, ctx.X.shape[1], max(ctx.X.shape[0] // 3, 2))
        X_reduced = decomposition.PCA(n_components=n_components, svd_solver="full").fit_transform(ctx.X)
        cov = covariance.EmpiricalCovariance().fit(X_reduced)
        distances = np.sqrt(np.maximum(cov.mahalanobis(X_reduced), 0.0))
        threshold = float(np.sqrt(stats.chi2.ppf(0.975, df=X_reduced.shape[1])))
        return (distances <= threshold).astype(np.uint8)

    return run


def _nirs_class(module: str, name: str):
    return getattr(importlib.import_module(module), name)


def _first_split(splitter, X: np.ndarray, y: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray]:
    train, test = next(splitter.split(X, y))
    return np.asarray(train, dtype=np.int64), np.asarray(test, dtype=np.int64)


def ref_nirs_transform(module: str, cls_name: str, *args, fit_args: tuple = (), transform_args: tuple = (), **kwargs) -> Callable[[Dataset], Callable[[], object]]:
    def factory(ctx: Dataset) -> Callable[[], object]:
        cls = _nirs_class(module, cls_name)

        def run() -> object:
            X = np.array(ctx.X, copy=True)
            transformer = cls(*args, **kwargs)
            if hasattr(transformer, "fit_transform"):
                return transformer.fit_transform(X, *fit_args)
            transformer.fit(X, *fit_args)
            return transformer.transform(X, *transform_args)

        return run

    return factory


def ref_nirs_resampler(ctx: Dataset) -> Callable[[], object]:
    cls = _nirs_class("nirs4all.operators.transforms.resampler", "Resampler")
    target = np.linspace(ctx.wavelengths[0], ctx.wavelengths[-1], max(4, ctx.X.shape[1] // 2))

    def run() -> object:
        X = np.array(ctx.X, copy=True)
        return cls(target_wavelengths=target, method="linear", fill_value=0.0, bounds_error=False).fit(
            X,
            wavelengths=ctx.wavelengths,
        ).transform(X)

    return run


def ref_nirs_crop(ctx: Dataset) -> Callable[[], object]:
    cls = _nirs_class("nirs4all.operators.transforms.features", "CropTransformer")
    start = max(0, ctx.X.shape[1] // 8)
    end = max(start + 1, ctx.X.shape[1] - ctx.X.shape[1] // 8)
    return lambda: cls(start=start, end=end).fit_transform(np.array(ctx.X, copy=True))


def ref_nirs_osc(ctx: Dataset) -> Callable[[], object]:
    cls = _nirs_class("nirs4all.operators.transforms.orthogonalization", "OSC")
    return lambda: cls(n_components=1, scale=True).fit_transform(np.array(ctx.X, copy=True), np.array(ctx.y, copy=True))


def ref_nirs_epo(ctx: Dataset) -> Callable[[], object]:
    cls = _nirs_class("nirs4all.operators.transforms.orthogonalization", "EPO")
    return lambda: cls(scale=True).fit_transform(np.array(ctx.X, copy=True), np.array(ctx.d, copy=True))


def ref_nirs_split_ks(ctx: Dataset) -> Callable[[], object]:
    cls = _nirs_class("nirs4all.operators.splitters.splitters", "KennardStoneSplitter")
    return lambda: _first_split(cls(test_size=0.25), np.array(ctx.X, copy=True))


def ref_nirs_split_spxy(ctx: Dataset) -> Callable[[], object]:
    cls = _nirs_class("nirs4all.operators.splitters.splitters", "SPXYSplitter")
    return lambda: _first_split(cls(test_size=0.25), np.array(ctx.X, copy=True), np.array(ctx.y, copy=True).reshape(-1, 1))


def ref_nirs_split_kbins_stratified(ctx: Dataset) -> Callable[[], object]:
    cls = _nirs_class("nirs4all.operators.splitters.splitters", "KBinsStratifiedSplitter")
    return lambda: _first_split(
        cls(test_size=0.25, random_state=17, n_bins=5, strategy="uniform"),
        np.array(ctx.X, copy=True),
        np.array(ctx.y, copy=True).reshape(-1, 1),
    )


def ref_nirs_y_outlier(ctx: Dataset) -> Callable[[], object]:
    cls = _nirs_class("nirs4all.operators.filters.y_outlier", "YOutlierFilter")

    def run() -> object:
        X = np.array(ctx.X, copy=True)
        y = np.array(ctx.y, copy=True)
        filt = cls(method="iqr", threshold=1.5, lower_percentile=1.0, upper_percentile=99.0).fit(X, y)
        return filt.get_mask(X, y).astype(np.uint8)

    return run


def ref_nirs_x_outlier(ctx: Dataset) -> Callable[[], object]:
    cls = _nirs_class("nirs4all.operators.filters.x_outlier", "XOutlierFilter")

    def run() -> object:
        X = np.array(ctx.X, copy=True)
        state = np.random.get_state()
        try:
            np.random.seed(0)
            filt = cls(method="mahalanobis", n_components=min(5, ctx.X.shape[1]), contamination=0.1, random_state=0).fit(X)
            return filt.get_mask(X).astype(np.uint8)
        finally:
            np.random.set_state(state)

    return run


def ref_pybaselines(name: str, **kwargs) -> Callable[[Dataset], Callable[[], object]]:
    def factory(ctx: Dataset) -> Callable[[], object]:
        baseline = importlib.import_module("pybaselines").Baseline()
        func = getattr(baseline, name)

        def run() -> object:
            rows = []
            for row in ctx.X:
                baseline = func(row, **kwargs)[0]
                corrected = row - baseline
                rows.append(corrected)
            return np.asarray(rows, dtype=np.float64)

        return run

    return factory


def ref_pywavelets_wavelet(ctx: Dataset) -> Callable[[], object]:
    pywt = importlib.import_module("pywt")

    def run() -> object:
        c_a, c_d = pywt.dwt(ctx.X, "haar", mode="periodization", axis=1)
        return np.concatenate([c_a, c_d], axis=1)

    return run


def ref_pywavelets_haar(ctx: Dataset) -> Callable[[], object]:
    return ref_pywavelets_wavelet(ctx)


def _wavelet_threshold(coeffs: np.ndarray, threshold: float, mode: str) -> np.ndarray:
    if mode == "soft":
        return np.sign(coeffs) * np.maximum(np.abs(coeffs) - threshold, 0.0)
    if mode == "hard":
        out = np.array(coeffs, copy=True)
        out[np.abs(out) <= threshold] = 0.0
        return out
    raise ValueError(f"unknown threshold mode: {mode}")


def ref_pywavelets_denoise(ctx: Dataset) -> Callable[[], object]:
    pywt = importlib.import_module("pywt")

    def run() -> object:
        out = np.empty_like(ctx.X)
        universal_factor = math.sqrt(2.0 * math.log(ctx.X.shape[1]))
        for idx, row in enumerate(ctx.X):
            coeffs = pywt.wavedec(row, "db4", mode="periodization", level=2)
            finest = coeffs[-1]
            sigma = float(np.median(np.abs(finest)) / 0.6745)
            threshold = sigma * universal_factor
            denoised = [coeffs[0]]
            denoised.extend(_wavelet_threshold(coeff, threshold, "soft") for coeff in coeffs[1:])
            out[idx] = pywt.waverec(denoised, "db4", mode="periodization")[: ctx.X.shape[1]]
        return out

    return run


def _wavelet_band_stats(coeffs: np.ndarray, *, entropy: str = "energy") -> np.ndarray:
    if coeffs.size == 0:
        return np.zeros(4, dtype=np.float64)
    energy = float(np.sum(coeffs * coeffs))
    entropy_value = 0.0
    if entropy == "histogram":
        counts, _ = np.histogram(coeffs, bins=10)
        probs = counts.astype(np.float64) / float(coeffs.size)
        positive = probs > 0.0
        entropy_value = float(-np.sum(probs[positive] * np.log(probs[positive])))
    elif energy > 0.0:
        probs = (coeffs * coeffs) / energy
        positive = probs > 0.0
        entropy_value = float(-np.sum(probs[positive] * np.log(probs[positive])))
    return np.asarray([np.mean(coeffs), np.std(coeffs), energy, entropy_value], dtype=np.float64)


def ref_pywavelets_features(ctx: Dataset, *, mode: str = "periodization",
                            entropy: str = "energy") -> Callable[[], object]:
    pywt = importlib.import_module("pywt")

    def run() -> object:
        level = min(2, pywt.dwt_max_level(ctx.X.shape[1], pywt.Wavelet("haar").dec_len))
        out = np.empty((ctx.X.shape[0], 4 * (level + 1)), dtype=np.float64)
        for row_idx, row in enumerate(ctx.X):
            coeffs = pywt.wavedec(row, "haar", mode=mode, level=level)
            for coeff_idx, coeff in enumerate(coeffs):
                out[row_idx, 4 * coeff_idx:4 * coeff_idx + 4] = _wavelet_band_stats(
                    np.asarray(coeff, dtype=np.float64), entropy=entropy)
        return out

    return run


def ref_numpy_log_transform(ctx: Dataset) -> Callable[[], object]:
    def run() -> object:
        X = np.asarray(ctx.X, dtype=np.float64)
        offset = max(0.0, 1e-8 - float(np.nanmin(X)))
        return np.log(np.maximum(X + offset, 1e-8))

    return run


def ref_numpy_derivate(ctx: Dataset) -> Callable[[], object]:
    return lambda: np.diff(ctx.X, n=1, axis=1) / 1.0


def _wavelet_flat_coefficients(ctx: Dataset, *, mode: str = "periodization") -> np.ndarray:
    pywt = importlib.import_module("pywt")
    wavelet = pywt.Wavelet("haar")
    level = min(2, pywt.dwt_max_level(ctx.X.shape[1], wavelet.dec_len))
    rows = []
    for row in ctx.X:
        coeffs = pywt.wavedec(row, wavelet, mode=mode, level=level)
        rows.append(np.concatenate([np.asarray(coeff, dtype=np.float64).reshape(-1) for coeff in coeffs]))
    return np.asarray(rows, dtype=np.float64)


def ref_pywavelets_wavelet_projection(ctx: Dataset, *, kind: str) -> Callable[[], object]:
    decomposition = importlib.import_module("sklearn.decomposition")

    def run() -> object:
        features = _wavelet_flat_coefficients(ctx, mode="periodization")
        n_components = min(5, features.shape[0], features.shape[1])
        if kind == "pca":
            return decomposition.PCA(n_components=n_components, svd_solver="full").fit_transform(features)
        if kind == "svd":
            return decomposition.TruncatedSVD(
                n_components=n_components,
                algorithm="arpack" if n_components < min(features.shape) else "randomized",
                random_state=0,
            ).fit_transform(features)
        raise ValueError(f"unknown wavelet projection kind: {kind}")

    return run


def _nirs_fit_transform(transformer, X: np.ndarray, wavelengths: np.ndarray | None = None) -> object:
    X = np.array(X, copy=True)
    if wavelengths is not None:
        try:
            return transformer.fit(X, wavelengths=wavelengths).transform(X, wavelengths=wavelengths)
        except TypeError:
            try:
                return transformer.fit_transform(X, wavelengths=wavelengths)
            except TypeError:
                pass
    if hasattr(transformer, "fit_transform"):
        return transformer.fit_transform(X)
    transformer.fit(X)
    return transformer.transform(X)


def ref_nirs_augmenter(
    module: str,
    cls_name: str,
    *,
    wavelengths: bool = False,
    **kwargs,
) -> Callable[[Dataset], Callable[[], object]]:
    def factory(ctx: Dataset) -> Callable[[], object]:
        cls = _nirs_class(module, cls_name)

        def run() -> object:
            return _nirs_fit_transform(
                cls(**kwargs),
                ctx.X,
                ctx.wavelengths if wavelengths else None,
            )

        return run

    return factory


def ref_nirs_high_leverage(ctx: Dataset) -> Callable[[], object]:
    cls = _nirs_class("nirs4all.operators.filters.high_leverage", "HighLeverageFilter")

    def run() -> object:
        filt = cls(method="hat", threshold_multiplier=2.0, center=True).fit(np.array(ctx.X, copy=True))
        return filt.get_mask(np.array(ctx.X, copy=True)).astype(np.uint8)

    return run


def ref_nirs_spectral_quality(ctx: Dataset) -> Callable[[], object]:
    cls = _nirs_class("nirs4all.operators.filters.spectral_quality", "SpectralQualityFilter")

    def run() -> object:
        filt = cls().fit(np.array(ctx.X, copy=True))
        return filt.get_mask(np.array(ctx.X, copy=True)).astype(np.uint8)

    return run


def ref_nirs_composite_filter(ctx: Dataset) -> Callable[[], object]:
    high_cls = _nirs_class("nirs4all.operators.filters.high_leverage", "HighLeverageFilter")
    quality_cls = _nirs_class("nirs4all.operators.filters.spectral_quality", "SpectralQualityFilter")
    composite_cls = _nirs_class("nirs4all.operators.filters.base", "CompositeFilter")

    def run() -> object:
        X = np.array(ctx.X, copy=True)
        filt = composite_cls(filters=[high_cls(method="hat", threshold_multiplier=2.0, center=True), quality_cls()], mode="any")
        filt.fit(X)
        return filt.get_mask(X).astype(np.uint8)

    return run


def ref_sklearn_hotelling_t2(ctx: Dataset) -> Callable[[], object]:
    decomposition = importlib.import_module("sklearn.decomposition")
    stats = importlib.import_module("scipy.stats")

    def run() -> object:
        k = min(5, ctx.X.shape[1], ctx.X.shape[0] - 1)
        pca = decomposition.PCA(n_components=k, svd_solver="full").fit(ctx.X)
        scores = pca.transform(ctx.X)
        eig = np.maximum(np.asarray(pca.explained_variance_, dtype=np.float64), 1e-12)
        t2 = np.sum((scores * scores) / eig.reshape(1, -1), axis=1)
        n = ctx.X.shape[0]
        ucl = (k * (n - 1) * (n + 1) / (n * max(n - k, 1))) * float(stats.f.ppf(0.95, k, max(n - k, 1)))
        return t2, ucl

    return run


def ref_sklearn_q_residuals(ctx: Dataset) -> Callable[[], object]:
    decomposition = importlib.import_module("sklearn.decomposition")
    stats = importlib.import_module("scipy.stats")

    def run() -> object:
        k = min(5, ctx.X.shape[1], ctx.X.shape[0] - 1)
        pca = decomposition.PCA(n_components=k, svd_solver="full").fit(ctx.X)
        recon = pca.inverse_transform(pca.transform(ctx.X))
        q = np.sum((ctx.X - recon) * (ctx.X - recon), axis=1)
        centered = ctx.X - np.mean(ctx.X, axis=0)
        eigvals = np.linalg.eigvalsh(np.cov(centered, rowvar=False))
        residual = np.sort(np.maximum(eigvals, 0.0))[::-1][k:]
        if residual.size == 0 or float(np.sum(residual)) <= 0.0:
            ucl = 0.0
        else:
            theta1 = float(np.sum(residual))
            theta2 = float(np.sum(residual * residual))
            theta3 = float(np.sum(residual * residual * residual))
            h0 = 1.0 - (2.0 * theta1 * theta3) / (3.0 * theta2 * theta2)
            h0 = max(h0, 1e-6)
            z = float(stats.norm.ppf(0.95))
            ucl = theta1 * ((z * math.sqrt(2.0 * theta2 * h0 * h0) / theta1)
                            + 1.0 + theta2 * h0 * (h0 - 1.0) / (theta1 * theta1)) ** (1.0 / h0)
        return q, ucl

    return run


def ref_numpy_nirs_metrics(ctx: Dataset) -> Callable[[], object]:
    def run() -> object:
        y_true = np.asarray(ctx.y, dtype=np.float64)
        y_pred = 1.05 * y_true + 0.02
        residual = y_pred - y_true
        rmse = float(np.sqrt(np.mean(residual * residual)))
        mae = float(np.mean(np.abs(residual)))
        bias = float(np.mean(residual))
        centered_residual = residual - bias
        sep = float(np.sqrt(np.mean(centered_residual * centered_residual)))
        std = float(np.std(y_true, ddof=0))
        q75, q25 = np.percentile(y_true, [75.0, 25.0])
        denom = float(np.sum((y_true - np.mean(y_true)) ** 2))
        r2 = 1.0 - float(np.sum(residual * residual)) / denom if denom > 0.0 else 0.0
        nrmse = rmse / max(abs(float(np.mean(y_true))), np.finfo(np.float64).tiny)
        return np.asarray([
            rmse,
            mae,
            bias,
            sep,
            std / max(sep, np.finfo(np.float64).tiny),
            float(q75 - q25) / max(rmse, np.finfo(np.float64).tiny),
            r2,
            nrmse,
        ], dtype=np.float64)

    return run


def ref_nirs_signal_type(ctx: Dataset) -> Callable[[], object]:
    module = importlib.import_module("nirs4all.data.signal_type")
    detect_signal_type = module.detect_signal_type
    type_map = {
        "unknown": 0,
        "absorbance": 1,
        "reflectance": 2,
        "reflectance%": 3,
        "transmittance": 4,
        "transmittance%": 5,
        "kubelka_munk": 6,
        "log_1_r": 7,
        "log_1_t": 8,
        "auto": 0,
        "preprocessed": 0,
    }

    def run() -> object:
        kind, confidence, _reason = detect_signal_type(ctx.X, ctx.wavelengths)
        key = getattr(kind, "value", str(kind))
        return np.asarray([type_map.get(str(key), 0), float(confidence)], dtype=np.float64)

    return run


def ref_numpy_rng_pcg64(ctx: Dataset) -> Callable[[], object]:
    return lambda: np.random.default_rng(ctx.seed).standard_normal(1024)


def ref_nirs_transfer_metrics(ctx: Dataset) -> Callable[[], object]:
    cls = _nirs_class("nirs4all.analysis.transfer_metrics", "TransferMetricsComputer")

    def run() -> object:
        comp = cls(
            n_components=min(5, ctx.X.shape[0] - 1, ctx.X.shape[1]),
            k_neighbors=min(10, max(2, ctx.X.shape[0] - 1)),
            random_state=ctx.seed,
        )
        metrics = comp.compute(ctx.X, _transfer_target(ctx))
        return np.asarray([getattr(metrics, name) for name, _ctype in TransferMetrics._fields_], dtype=np.float64)

    return run


def ref_nirs_wavelength_spectral(ctx: Dataset) -> Callable[[], object]:
    spectral = "nirs4all.operators.augmentation.spectral"
    specs = (
        (spectral, "WavelengthShift", dict(shift_range=(-1.0, 1.0), random_state=17), True),
        (spectral, "WavelengthStretch", dict(stretch_range=(0.99, 1.01), random_state=17), True),
        (spectral, "LocalWavelengthWarp", dict(n_control_points=3, max_shift=1.0, random_state=17), True),
        (spectral, "BandPerturbation", dict(n_bands=3, bandwidth_range=(5, 15), gain_range=(0.9, 1.1), offset_range=(-0.01, 0.01), random_state=17), False),
        (spectral, "BandMasking", dict(n_bands_range=(1, 3), bandwidth_range=(5, 15), mode="zero", random_state=17), False),
        (spectral, "ChannelDropout", dict(dropout_prob=0.05, mode="zero", random_state=17), False),
        (spectral, "GaussianSmoothingJitter", dict(sigma_range=(0.5, 1.5), kernel_width=9, random_state=17), False),
        (spectral, "UnsharpSpectralMask", dict(amount_range=(0.1, 0.5), sigma=1.0, kernel_width=11, random_state=17), False),
        (spectral, "SmoothMagnitudeWarp", dict(n_control_points=3, gain_range=(0.9, 1.1), random_state=17), True),
        (spectral, "LocalClipping", dict(n_regions=1, width_range=(5, 15), random_state=17), False),
    )

    def run() -> object:
        parts = []
        for module, cls_name, kwargs, needs_wavelengths in specs:
            cls = _nirs_class(module, cls_name)
            parts.append(_nirs_fit_transform(cls(**kwargs), ctx.X, ctx.wavelengths if needs_wavelengths else None))
        return np.concatenate(parts, axis=1)

    return run


def nirs_ref(
    library: str,
    factory: Callable[[Dataset], Callable[[], object]],
    *,
    compare: bool = True,
    gate_c4a: bool = True,
    contract_note: str = "",
    max_cols: int | None = None,
) -> ReferenceSpec:
    return ReferenceSpec(
        "ref.nirs4all",
        library,
        factory,
        compare=compare,
        gate_c4a=gate_c4a,
        language="Python",
        contract_note=contract_note,
        max_cols=max_cols,
    )


def numpy_ref(
    library: str,
    factory: Callable[[Dataset], Callable[[], object]],
    *,
    compare: bool = True,
    gate_c4a: bool = True,
    contract_note: str = "",
    comparator: Callable[[object, object], tuple[bool, str]] | None = None,
) -> ReferenceSpec:
    return ReferenceSpec(
        "ref.numpy",
        library,
        factory,
        compare=compare,
        gate_c4a=gate_c4a,
        language="Python",
        contract_note=contract_note,
        comparator=comparator,
    )


def sklearn_ref(
    library: str,
    factory: Callable[[Dataset], Callable[[], object]],
    *,
    compare: bool = True,
    gate_c4a: bool = True,
    contract_note: str = "",
) -> ReferenceSpec:
    return ReferenceSpec(
        "ref.sklearn",
        library,
        factory,
        compare=compare,
        gate_c4a=gate_c4a,
        language="Python",
        contract_note=contract_note,
    )


def scipy_ref(
    library: str,
    factory: Callable[[Dataset], Callable[[], object]],
    *,
    compare: bool = True,
    gate_c4a: bool = True,
    contract_note: str = "",
) -> ReferenceSpec:
    return ReferenceSpec(
        "ref.scipy",
        library,
        factory,
        compare=compare,
        gate_c4a=gate_c4a,
        language="Python",
        contract_note=contract_note,
    )


def statsmodels_ref(
    library: str,
    factory: Callable[[Dataset], Callable[[], object]],
    *,
    compare: bool = True,
    gate_c4a: bool = True,
    contract_note: str = "",
) -> ReferenceSpec:
    return ReferenceSpec(
        "ref.statsmodels",
        library,
        factory,
        compare=compare,
        gate_c4a=gate_c4a,
        language="Python",
        contract_note=contract_note,
    )


def cowarp_ref(
    library: str,
    factory: Callable[[Dataset], Callable[[], object]],
    *,
    compare: bool = True,
    gate_c4a: bool = True,
    contract_note: str = "",
    max_cols: int | None = None,
) -> ReferenceSpec:
    return ReferenceSpec(
        "ref.cowarp",
        library,
        factory,
        compare=compare,
        gate_c4a=gate_c4a,
        language="Python",
        contract_note=contract_note,
        max_cols=max_cols,
    )


def pywavelets_ref(
    library: str,
    factory: Callable[[Dataset], Callable[[], object]],
    *,
    compare: bool = True,
    gate_c4a: bool = True,
    contract_note: str = "",
    comparator: Callable[[object, object], tuple[bool, str]] | None = None,
) -> ReferenceSpec:
    return ReferenceSpec(
        "ref.pywavelets",
        library,
        factory,
        compare=compare,
        gate_c4a=gate_c4a,
        language="Python",
        contract_note=contract_note,
        comparator=comparator,
    )


def pycaltransfer_ref(
    library: str,
    factory: Callable[[Dataset], Callable[[], object]],
    *,
    compare: bool = True,
    gate_c4a: bool = True,
    contract_note: str = "",
    max_cols: int | None = None,
    comparator: Callable[[object, object], tuple[bool, str]] | None = None,
) -> ReferenceSpec:
    return ReferenceSpec(
        "ref.pycaltransfer",
        library,
        factory,
        compare=compare,
        gate_c4a=gate_c4a,
        language="Python",
        contract_note=contract_note,
        max_cols=max_cols,
        comparator=comparator,
    )


def r_prospectr_ref(
    library: str,
    expr: str,
    *,
    compare: bool = False,
    gate_c4a: bool = False,
    contract_note: str = "",
    max_cols: int | None = None,
    comparator: Callable[[object, object], tuple[bool, str]] | None = None,
) -> ReferenceSpec:
    return ReferenceSpec(
        "ref.r.prospectr",
        library,
        factory=None,
        compare=compare,
        gate_c4a=gate_c4a,
        language="R",
        r_expr=expr,
        contract_note=contract_note,
        max_cols=max_cols,
        comparator=comparator,
    )


def c_same(prefix: str, *args) -> Callable[[Dataset], Callable[[], object]]:
    def factory(ctx: Dataset) -> Callable[[], object]:
        X_arr = as_f64_2d(ctx.X)
        handle = _create(prefix, *args)
        out = empty_like_f64(X_arr.shape)
        x_view = numpy_to_view(X_arr)
        out_view = numpy_to_view(out)
        transform = getattr(lib, f"{prefix}_transform")

        def run() -> np.ndarray:
            status = transform(handle, x_view, out_view)
            if status != 0:
                check(status, f"{prefix}_transform")
            return out

        run._c4a_output = out
        weakref.finalize(run, getattr(lib, f"{prefix}_destroy"), handle)
        return run

    return factory


def c_stateful(prefix: str, *args) -> Callable[[Dataset], Callable[[], object]]:
    return lambda ctx: lambda: c_transform(prefix, ctx.X, *args, fit_X=ctx.X)


METHODS: tuple[MethodSpec, ...] = (
    MethodSpec(
        "snv",
        "SNV",
        "preprocessing",
        py_transform(c4a.SNV, ddof=1),
        c_same("c4a_pp_snv", ctypes.c_int(1), ctypes.c_int(1), ctypes.c_int(1)),
        "snv(X, ddof = 1L)",
        (
            r_prospectr_ref(
                "prospectr.standardNormalVariate",
                "prospectr::standardNormalVariate(X)",
                compare=True,
                gate_c4a=True,
                contract_note="c4a benchmark uses ddof=1 to match prospectr's sample-standard-deviation convention",
            ),
            nirs_ref("nirs4all.StandardNormalVariate", ref_nirs_transform("nirs4all.operators.transforms.scalers", "StandardNormalVariate", ddof=1)),
        ),
    ),
    MethodSpec(
        "lsnv",
        "LSNV",
        "preprocessing",
        py_transform(c4a.LSNV, window=11, pad_mode="reflect"),
        c_same("c4a_pp_lsnv", ctypes.c_int32(11), ctypes.c_int32(0), ctypes.c_double(0.0)),
        "lsnv(X, window = 11L)",
        (nirs_ref("nirs4all.LocalStandardNormalVariate", ref_nirs_transform("nirs4all.operators.transforms.scalers", "LocalStandardNormalVariate", window=11, pad_mode="reflect")),),
    ),
    MethodSpec(
        "rnv",
        "RNV",
        "preprocessing",
        py_transform(c4a.RNV),
        c_same("c4a_pp_rnv", ctypes.c_int(1), ctypes.c_int(1), ctypes.c_double(1.4826)),
        "rnv(X)",
        (nirs_ref("nirs4all.RobustStandardNormalVariate", ref_nirs_transform("nirs4all.operators.transforms.scalers", "RobustStandardNormalVariate")),),
    ),
    MethodSpec(
        "area_norm",
        "Area normalization",
        "preprocessing",
        py_transform(c4a.AreaNormalization, method="sum"),
        c_same("c4a_pp_area", ctypes.c_int32(0)),
        "area_normalization(X, method = 'sum')",
        (
            nirs_ref("nirs4all.AreaNormalization", ref_nirs_transform("nirs4all.operators.transforms.nirs", "AreaNormalization", method="sum")),
        ),
    ),
    MethodSpec(
        "normalize",
        "Normalize",
        "preprocessing",
        py_transform(c4a.Normalize, feature_min=-1.0, feature_max=1.0),
        c_same("c4a_pp_normalize", ctypes.c_double(-1.0), ctypes.c_double(1.0)),
        "normalize(X, feature_min = -1, feature_max = 1)",
        (nirs_ref("nirs4all.Normalize", ref_nirs_transform("nirs4all.operators.transforms.scalers", "Normalize", feature_range=(-1, 1))),),
    ),
    MethodSpec(
        "simple_scale",
        "Simple scale",
        "preprocessing",
        py_transform(c4a.SimpleScale),
        c_same("c4a_pp_simple_scale"),
        "simple_scale(X)",
        (
            sklearn_ref("sklearn.preprocessing.MinMaxScaler", ref_sklearn_simple_scale),
            nirs_ref("nirs4all.SimpleScale", ref_nirs_transform("nirs4all.operators.transforms.scalers", "SimpleScale")),
        ),
    ),
    MethodSpec(
        "msc",
        "MSC",
        "preprocessing",
        py_transform(c4a.MSC),
        c_stateful("c4a_pp_msc"),
        "msc(X)",
        (
            r_prospectr_ref(
                "prospectr.msc",
                "prospectr::msc(X)",
                compare=True,
                gate_c4a=True,
                contract_note="c4a MSC follows the conventional row-wise reference-spectrum contract used by prospectr",
            ),
            nirs_ref(
                "nirs4all.MultiplicativeScatterCorrection(scale=False)",
                ref_nirs_transform("nirs4all.operators.transforms.nirs", "MultiplicativeScatterCorrection", scale=False),
                compare=False,
                gate_c4a=False,
                contract_note="current local nirs4all checkout uses a historical column-regression MSC variant",
            ),
        ),
    ),
    MethodSpec(
        "emsc",
        "EMSC",
        "preprocessing",
        py_transform(c4a.EMSC, degree=1),
        c_stateful("c4a_pp_emsc", ctypes.c_int32(1)),
        "emsc(X, degree = 1L)",
        (nirs_ref("nirs4all.ExtendedMultiplicativeScatterCorrection(scale=False)", ref_nirs_transform("nirs4all.operators.transforms.nirs", "ExtendedMultiplicativeScatterCorrection", degree=1, scale=False)),),
    ),
    MethodSpec(
        "baseline_center",
        "Baseline center",
        "preprocessing",
        py_transform(c4a.BaselineCenter),
        c_stateful("c4a_pp_baseline"),
        "baseline_center(X)",
        (nirs_ref("nirs4all.Baseline", ref_nirs_transform("nirs4all.operators.transforms.signal", "Baseline")),),
    ),
    MethodSpec(
        "savitzky_golay",
        "Savitzky-Golay",
        "preprocessing",
        py_transform(c4a.SavitzkyGolay, window_length=11, polyorder=2, deriv=0, mode="mirror"),
        c_same("c4a_pp_savgol", ctypes.c_int32(11), ctypes.c_int32(2), ctypes.c_int32(0), ctypes.c_double(1.0), ctypes.c_int(0), ctypes.c_double(0.0)),
        "savitzky_golay(X, window_length = 11L, polyorder = 2L, mode = 'mirror')",
        (
            scipy_ref("scipy.signal.savgol_filter", ref_scipy_savgol),
            r_prospectr_ref(
                "prospectr.savitzkyGolay",
                "prospectr::savitzkyGolay(X, m = 0, p = 2, w = 11)",
                compare=True,
                gate_c4a=True,
                comparator=outputs_close_savgol_valid_window,
                contract_note="prospectr drops boundary columns; parity compares the shared valid interior window",
            ),
            nirs_ref(
                "nirs4all.SavitzkyGolay(default edge mode)",
                ref_nirs_transform("nirs4all.operators.transforms.nirs", "SavitzkyGolay", window_length=11, polyorder=2, deriv=0, delta=1.0),
                compare=False,
                contract_note="nirs4all does not expose c4a's mirror edge mode; the default scipy edge contract changes boundary samples",
            ),
        ),
    ),
    MethodSpec(
        "first_derivative",
        "First derivative",
        "preprocessing",
        py_transform(c4a.FirstDerivative, delta=1.0, edge_order=2),
        c_same("c4a_pp_first_derivative", ctypes.c_double(1.0), ctypes.c_int32(2)),
        "first_derivative(X)",
        (
            nirs_ref("nirs4all.FirstDerivative", ref_nirs_transform("nirs4all.operators.transforms.nirs", "FirstDerivative", delta=1.0, edge_order=2)),
        ),
    ),
    MethodSpec(
        "second_derivative",
        "Second derivative",
        "preprocessing",
        py_transform(c4a.SecondDerivative, delta=1.0, edge_order=2),
        c_same("c4a_pp_second_derivative", ctypes.c_double(1.0), ctypes.c_int32(2)),
        "second_derivative(X)",
        (
            nirs_ref("nirs4all.SecondDerivative", ref_nirs_transform("nirs4all.operators.transforms.nirs", "SecondDerivative", delta=1.0, edge_order=2)),
        ),
    ),
    MethodSpec("norris_williams", "Norris-Williams", "preprocessing", py_transform(c4a.NorrisWilliams, segment=5, gap=5, derivative_order=1), c_same("c4a_pp_norris_williams", ctypes.c_int32(5), ctypes.c_int32(5), ctypes.c_int32(1), ctypes.c_double(1.0)), "norris_williams(X, segment = 5L, gap = 5L)", (nirs_ref("nirs4all.NorrisWilliams", ref_nirs_transform("nirs4all.operators.transforms.norris_williams", "NorrisWilliams", segment=5, gap=5, deriv=1, delta=1.0)),)),
    MethodSpec(
        "gaussian",
        "Gaussian",
        "preprocessing",
        py_transform(c4a.Gaussian, sigma=1.0, order=0, mode="reflect"),
        c_same("c4a_pp_gaussian", ctypes.c_double(1.0), ctypes.c_int32(0), ctypes.c_int(0), ctypes.c_double(0.0), ctypes.c_double(4.0)),
        "gaussian(X, sigma = 1.0)",
        (
            scipy_ref("scipy.ndimage.gaussian_filter1d", ref_scipy_gaussian),
            nirs_ref("nirs4all.Gaussian", ref_nirs_transform("nirs4all.operators.transforms.signal", "Gaussian", order=0, sigma=1.0)),
        ),
    ),
    MethodSpec("to_absorbance", "To absorbance", "preprocessing", py_transform(c4a.ToAbsorbance), c_same("c4a_pp_to_absorbance", ctypes.c_int(0), ctypes.c_double(1e-10), ctypes.c_int(1)), "to_absorbance(X)", (nirs_ref("nirs4all.ToAbsorbance", ref_nirs_transform("nirs4all.operators.transforms.signal_conversion", "ToAbsorbance", source_type="reflectance", epsilon=1e-10, clip_negative=True)),)),
    MethodSpec("from_absorbance", "From absorbance", "preprocessing", py_transform(c4a.FromAbsorbance), c_same("c4a_pp_from_absorbance", ctypes.c_int(0)), "from_absorbance(X)", (nirs_ref("nirs4all.FromAbsorbance", ref_nirs_transform("nirs4all.operators.transforms.signal_conversion", "FromAbsorbance", target_type="reflectance")),)),
    MethodSpec("pct_to_frac", "Percent to fraction", "preprocessing", py_transform(c4a.PercentToFraction), c_same("c4a_pp_pct_to_frac"), "percent_to_fraction(X)", (nirs_ref("nirs4all.PercentToFraction", ref_nirs_transform("nirs4all.operators.transforms.signal_conversion", "PercentToFraction")),)),
    MethodSpec("frac_to_pct", "Fraction to percent", "preprocessing", py_transform(c4a.FractionToPercent), c_same("c4a_pp_frac_to_pct"), "fraction_to_percent(X)", (nirs_ref("nirs4all.FractionToPercent", ref_nirs_transform("nirs4all.operators.transforms.signal_conversion", "FractionToPercent")),)),
    MethodSpec("kubelka_munk", "Kubelka-Munk", "preprocessing", py_transform(c4a.KubelkaMunk), c_same("c4a_pp_kubelka_munk", ctypes.c_int(0), ctypes.c_double(1e-10)), "kubelka_munk(X)", (nirs_ref("nirs4all.KubelkaMunk", ref_nirs_transform("nirs4all.operators.transforms.signal_conversion", "KubelkaMunk", source_type="reflectance", epsilon=1e-10)),)),
    MethodSpec("detrend", "Detrend", "baseline", py_transform(c4a.Detrend, polyorder=1), c_same("c4a_pp_detrend", ctypes.c_int32(1)), "detrend(X, polyorder = 1L)", (scipy_ref("scipy.signal.detrend", ref_scipy_detrend), nirs_ref("nirs4all.Detrend", ref_nirs_transform("nirs4all.operators.transforms.signal", "Detrend")), r_prospectr_ref("prospectr.detrend", "prospectr::detrend(X, wav = seq_len(ncol(X)))", contract_note="prospectr combines the NIR detrend convention with its own wavelength argument contract"))),
    MethodSpec("asls", "AsLS", "baseline", py_transform(c4a.AsLS, lam=1e5, p=0.01, max_iter=20), c_same("c4a_pp_asls", ctypes.c_double(1e5), ctypes.c_double(0.01), ctypes.c_int32(20), ctypes.c_double(1e-3)), "asls(X, lam = 1e5, p = 0.01, max_iter = 20L)", (ReferenceSpec("ref.pybaselines", "pybaselines.asls", ref_pybaselines("asls", lam=1e5, p=0.01, max_iter=20)),)),
    MethodSpec("airpls", "AirPLS", "baseline", py_transform(c4a.AirPLS, lam=1e5, max_iter=20), c_same("c4a_pp_airpls", ctypes.c_double(1e5), ctypes.c_int32(20), ctypes.c_double(1e-3)), "airpls(X, lam = 1e5, max_iter = 20L)", (ReferenceSpec("ref.pybaselines", "pybaselines.airpls", ref_pybaselines("airpls", lam=1e5, max_iter=20)),)),
    MethodSpec("arpls", "ArPLS", "baseline", py_transform(c4a.ArPLS, lam=1e5, max_iter=20), c_same("c4a_pp_arpls", ctypes.c_double(1e5), ctypes.c_int32(20), ctypes.c_double(1e-3)), "arpls(X, lam = 1e5, max_iter = 20L)", (ReferenceSpec("ref.pybaselines", "pybaselines.arpls", ref_pybaselines("arpls", lam=1e5, max_iter=20)),)),
    MethodSpec("modpoly", "ModPoly", "baseline", py_transform(c4a.ModPoly, polyorder=2, max_iter=50), c_same("c4a_pp_modpoly", ctypes.c_int32(2), ctypes.c_int32(50), ctypes.c_double(1e-3)), "modpoly(X, polyorder = 2L, max_iter = 50L)", (ReferenceSpec("ref.pybaselines", "pybaselines.modpoly", ref_pybaselines("modpoly", poly_order=2, max_iter=50)),)),
    MethodSpec("imodpoly", "IModPoly", "baseline", py_transform(c4a.IModPoly, polyorder=2, max_iter=50), c_same("c4a_pp_imodpoly", ctypes.c_int32(2), ctypes.c_int32(50), ctypes.c_double(1e-3)), "imodpoly(X, polyorder = 2L, max_iter = 50L)", (ReferenceSpec("ref.pybaselines", "pybaselines.imodpoly(mask_initial_peaks=False)", ref_pybaselines("imodpoly", poly_order=2, max_iter=50, mask_initial_peaks=False)),)),
    MethodSpec("snip", "SNIP", "baseline", py_transform(c4a.SNIP, max_half_window=10), c_same("c4a_pp_snip", ctypes.c_int32(10)), "snip(X, max_half_window = 10L)", (
        ReferenceSpec(
            "ref.pybaselines",
            "pybaselines.snip(raw)",
            ref_pybaselines("snip", max_half_window=10),
        ),
    )),
    MethodSpec("rolling_ball", "Rolling ball", "baseline", py_transform(c4a.RollingBall, half_window=10, smooth_half_window=0), c_same("c4a_pp_rolling_ball", ctypes.c_int32(10), ctypes.c_int32(0)), "rolling_ball(X, half_window = 10L, smooth_half_window = 0L)", (ReferenceSpec("ref.pybaselines", "pybaselines.rolling_ball", ref_pybaselines("rolling_ball", half_window=10, smooth_half_window=0)),)),
    MethodSpec("iasls", "IAsLS", "baseline", py_transform(c4a.IAsLS, lam=1e5, p=0.01, lam_1=1e-4, polyorder=2, diff_order=2, max_iter=20), lambda ctx: lambda: c_iasls(ctx), "iasls(X, lam = 1e5, p = 0.01, lam_1 = 1e-4, polyorder = 2L, diff_order = 2L, max_iter = 20L)", (ReferenceSpec("ref.pybaselines", "pybaselines.iasls", ref_pybaselines("iasls", lam=1e5, p=0.01, lam_1=1e-4, diff_order=2, max_iter=20)),), comparator=outputs_close_iasls),
    MethodSpec("beads", "BEADS", "baseline", py_transform(c4a.BEADS, lam_0=100.0, lam_1=0.5, lam_2=0.5, max_iter=20), c_same("c4a_pp_beads", ctypes.c_double(100.0), ctypes.c_double(0.5), ctypes.c_double(0.5), ctypes.c_int32(20), ctypes.c_double(1e-3)), "beads(X, max_iter = 20L)", (
        ReferenceSpec(
            "ref.pybaselines",
            "pybaselines.beads(full)",
            ref_pybaselines("beads", lam_0=100.0, lam_1=0.5, lam_2=0.5, max_iter=20, tol=1e-3),
        ),
    )),
    MethodSpec(
        "wavelet",
        "Wavelet",
        "wavelet",
        py_transform(c4a.Wavelet, family="haar", mode="periodization"),
        c_same("c4a_pp_wavelet", ctypes.c_int(0), ctypes.c_int(0)),
        "wavelet(X, family = 'haar', boundary = 'periodization')",
        (
            pywavelets_ref("PyWavelets.dwt(haar, periodization)", ref_pywavelets_wavelet),
            nirs_ref(
                "nirs4all.Wavelet(detail-resampled)",
                ref_nirs_transform("nirs4all.operators.transforms.nirs", "Wavelet", wavelet="haar", mode="periodization"),
                compare=False,
                contract_note="nirs4all returns resampled detail coefficients only; c4a/PyWavelets gate the concatenated approximation+detail coefficients",
            ),
        ),
    ),
    MethodSpec(
        "haar",
        "Haar",
        "wavelet",
        py_transform(c4a.Haar),
        c_same("c4a_pp_haar"),
        "haar(X)",
        (
            pywavelets_ref("PyWavelets.dwt(haar, periodization)", ref_pywavelets_haar),
            nirs_ref(
                "nirs4all.Haar(detail-resampled)",
                ref_nirs_transform("nirs4all.operators.transforms.nirs", "Haar"),
                compare=False,
                contract_note="nirs4all returns resampled detail coefficients only; c4a/PyWavelets gate the concatenated approximation+detail coefficients",
            ),
        ),
    ),
    MethodSpec(
        "wavelet_denoise",
        "Wavelet denoise",
        "wavelet",
        py_transform(c4a.WaveletDenoise, family="db4", mode="periodization", level=2),
        c_same("c4a_pp_wavelet_denoise", ctypes.c_int(1), ctypes.c_int(0), ctypes.c_int32(2), ctypes.c_int(0), ctypes.c_int(0)),
        "wavelet_denoise(X, level = 2L)",
        (
            pywavelets_ref("PyWavelets.wavedec/waverec(db4, periodization)", ref_pywavelets_denoise),
            nirs_ref("nirs4all.WaveletDenoise", ref_nirs_transform("nirs4all.operators.transforms.wavelet_denoise", "WaveletDenoise", wavelet="db4", mode="periodization", level=2)),
        ),
    ),
    MethodSpec(
        "wavelet_features",
        "Wavelet features",
        "wavelet",
        py_transform(c4a.WaveletFeatures, family="haar", mode="symmetric", max_level=2, entropy="histogram"),
        lambda ctx: lambda: c_wavelet_features(ctx, mode=1, entropy=1),
        "wavelet_features(X, family = 'haar', boundary = 'symmetric', max_level = 2L, entropy = 'histogram')",
        (
            pywavelets_ref("PyWavelets.wavedec(haar, symmetric)+histogram band stats", lambda ctx: ref_pywavelets_features(ctx, mode="symmetric", entropy="histogram")),
            nirs_ref("nirs4all.WaveletFeatures(n_coeffs_per_level=0)", ref_nirs_transform("nirs4all.operators.transforms.nirs", "WaveletFeatures", wavelet="haar", max_level=2, n_coeffs_per_level=0)),
        ),
    ),
    MethodSpec("osc", "OSC", "feature_extraction", py_osc, lambda ctx: lambda: c_fit_y_transform("c4a_pp_osc", ctx.X, ctx.y, ctypes.c_int32(1), ctypes.c_int(1)), "osc(X, y, n_components = 1L, scale = TRUE)", (nirs_ref("nirs4all.OSC", ref_nirs_osc),)),
    MethodSpec(
        "epo",
        "EPO",
        "feature_extraction",
        py_epo,
        lambda ctx: lambda: c_epo(ctx),
        "epo(X, d, scale = TRUE)",
        (nirs_ref("nirs4all.EPO", ref_nirs_epo),),
    ),
    MethodSpec("flexible_pca", "Flexible PCA", "feature_extraction", py_transform(c4a.FlexiblePCA, n_components=5.0), lambda ctx: lambda: c_flexible("c4a_pp_flex_pca", ctx, 5.0), "flexible_pca(X, n_components = 5.0)", (sklearn_ref("sklearn.decomposition.PCA", ref_sklearn_flexible_pca), nirs_ref("nirs4all.FlexiblePCA", ref_nirs_transform("nirs4all.operators.transforms.feature_selection", "FlexiblePCA", n_components=5, svd_solver="full"))), comparator=outputs_close_sign_invariant_columns),
    MethodSpec("flexible_svd", "Flexible SVD", "feature_extraction", py_transform(c4a.FlexibleSVD, n_components=5.0), lambda ctx: lambda: c_flexible("c4a_pp_flex_svd", ctx, 5.0), "flexible_svd(X, n_components = 5.0)", (sklearn_ref("sklearn.decomposition.TruncatedSVD", ref_sklearn_flexible_svd), nirs_ref("nirs4all.FlexibleSVD", ref_nirs_transform("nirs4all.operators.transforms.feature_selection", "FlexibleSVD", n_components=5, algorithm="arpack", random_state=0))), comparator=outputs_close_sign_invariant_columns),
    MethodSpec("fck_static", "FCK static", "feature_extraction", py_fck, lambda ctx: lambda: c_fck_static(ctx), "fck_static(X, kernel_size = 5L, filter_orders = c(0.5, 1.0), filter_scales = c(1.0, 2.0))", (
        nirs_ref(
            "nirs4all.FCKStaticTransformer",
            ref_nirs_transform("nirs4all.operators.transforms.fck_static", "FCKStaticTransformer", alphas=[0.5, 1.0], scales=[1.0, 2.0], kernel_sizes=[5], sigma=3.0),
        ),
    )),
    MethodSpec(
        "direct_standardization",
        "Direct standardization",
        "calibration_transfer",
        py_paired_transfer(c4a.DirectStandardization),
        lambda ctx: lambda: c_paired_transfer(
            "c4a_pp_direct_standardization", ctx,
            ctypes.c_int(1), ctypes.c_double(0.0)
        ),
        "{target <- 1.05 * X + 0.1; direct_standardization(X, target, X, fit_intercept = TRUE, ridge = 0.0)}",
        (
            sklearn_ref("sklearn.linear_model.LinearRegression", ref_sklearn_direct_standardization),
            pycaltransfer_ref("pycaltransfer.ds_pc_transfer_fit", ref_pycaltransfer_direct_standardization),
        ),
        max_cols=96,
    ),
    MethodSpec(
        "robust_direct_standardization",
        "Robust direct standardization",
        "calibration_transfer",
        py_paired_transfer(c4a.RobustDirectStandardization, max_iter=2),
        lambda ctx: lambda: c_paired_transfer(
            "c4a_pp_robust_direct_standardization", ctx,
            ctypes.c_int(1), ctypes.c_double(0.0), ctypes.c_double(0.9),
            ctypes.c_int32(2)
        ),
        "{target <- 1.05 * X + 0.1; robust_direct_standardization(X, target, X, fit_intercept = TRUE, ridge = 0.0, trim_quantile = 0.9, max_iter = 2L)}",
        (
            sklearn_ref(
                "sklearn.linear_model.LinearRegression(iterative residual trimming)",
                ref_sklearn_robust_direct_standardization,
                contract_note="sklearn supplies the OLS fit; the adapter applies C4A's documented residual-trimming contract",
            ),
        ),
        max_cols=96,
    ),
    MethodSpec(
        "piecewise_direct_standardization",
        "Piecewise direct standardization",
        "calibration_transfer",
        py_paired_transfer(c4a.PiecewiseDirectStandardization, window_size=5),
        lambda ctx: lambda: c_paired_transfer(
            "c4a_pp_piecewise_direct_standardization", ctx,
            ctypes.c_int32(5), ctypes.c_int(1), ctypes.c_double(0.0)
        ),
        "{target <- 1.05 * X + 0.1; piecewise_direct_standardization(X, target, X, window_size = 5L, fit_intercept = TRUE, ridge = 0.0)}",
        (
            sklearn_ref(
                "sklearn.linear_model.LinearRegression(local windows)",
                ref_sklearn_piecewise_direct_standardization,
                contract_note="sklearn supplies each local least-squares transfer model for the C4A PDS contract",
            ),
            pycaltransfer_ref(
                "pycaltransfer.pds_pls_transfer_fit",
                ref_pycaltransfer_piecewise_direct_standardization,
                compare=False,
                gate_c4a=False,
                contract_note="pycaltransfer exposes the PDS-PLS variant; C4A's current PiecewiseDirectStandardization contract is local OLS and is gated by sklearn above",
                max_cols=96,
            ),
        ),
        max_cols=96,
    ),
    MethodSpec(
        "score_augmented_projection_standardization",
        "Score-augmented projection standardization",
        "calibration_transfer",
        py_paired_transfer(c4a.ScoreAugmentedProjectionStandardization, n_components=3),
        lambda ctx: lambda: c_paired_transfer(
            "c4a_pp_saps", ctx,
            ctypes.c_int32(3), ctypes.c_double(1.0), ctypes.c_int(1),
            ctypes.c_double(0.0)
        ),
        "{target <- 1.05 * X + 0.1; score_augmented_projection_standardization(X, target, X, n_components = 3L, score_weight = 1.0, fit_intercept = TRUE, ridge = 0.0)}",
        (
            sklearn_ref(
                "sklearn.decomposition.PCA + sklearn.linear_model.LinearRegression",
                ref_sklearn_score_augmented_projection_standardization,
                contract_note="sklearn supplies the PCA score space and OLS fit used by the SAPS transfer contract",
            ),
        ),
        max_cols=96,
    ),
    MethodSpec(
        "slope_bias_correction",
        "Slope/bias correction",
        "calibration_transfer",
        py_slope_bias,
        lambda ctx: lambda: c_slope_bias(ctx),
        "{target <- 2.0 * y + 1.0; slope_bias_correction(y, target, y)}",
        (
            sklearn_ref("sklearn.linear_model.LinearRegression", ref_sklearn_slope_bias),
            pycaltransfer_ref("pycaltransfer.slope_bias_correction", ref_pycaltransfer_slope_bias),
        ),
    ),
    MethodSpec(
        "local_centering",
        "Local centering",
        "calibration_transfer",
        py_paired_transfer(c4a.LocalCentering),
        lambda ctx: lambda: c_paired_transfer("c4a_pp_local_centering", ctx),
        "{target <- 1.05 * X + 0.1; local_centering(X, target, X)}",
        (
            sklearn_ref(
                "sklearn.preprocessing.StandardScaler(with_std=False)",
                ref_sklearn_local_centering,
            ),
        ),
    ),
    MethodSpec(
        "weighted_snv",
        "Weighted SNV",
        "preprocessing",
        py_weighted_snv,
        lambda ctx: lambda: c_weighted_snv(ctx),
        "weighted_snv(X, weights = seq(1, 2, length.out = ncol(X)), ddof = 0L, eps = 1e-12)",
        (
            statsmodels_ref(
                "statsmodels.stats.weightstats.DescrStatsW",
                ref_statsmodels_weighted_snv,
            ),
        ),
    ),
    MethodSpec(
        "variable_sorting_normalization",
        "Variable sorting normalization",
        "preprocessing",
        py_transform(c4a.VariableSortingNormalization),
        lambda ctx: lambda: c_transform(
            "c4a_pp_vsn", ctx.X, ctypes.c_double(1e-12), fit_X=ctx.X
        ),
        "variable_sorting_normalization(X, eps = 1e-12)",
        (
            statsmodels_ref(
                "sklearn.feature_selection.r_regression + statsmodels.DescrStatsW",
                ref_statsmodels_variable_sorting_normalization,
            ),
        ),
    ),
    MethodSpec(
        "piecewise_snv",
        "Piecewise SNV",
        "preprocessing",
        py_transform(c4a.PiecewiseSNV, window_size=16),
        lambda ctx: lambda: c_transform(
            "c4a_pp_piecewise_snv", ctx.X, ctypes.c_int32(16),
            ctypes.c_int32(0), ctypes.c_double(1e-12), fit_X=ctx.X
        ),
        "piecewise_snv(X, window = 16L, ddof = 0L, eps = 1e-12)",
        (
            nirs_ref(
                "nirs4all.StandardNormalVariate(ddof=0) per interval",
                ref_nirs_piecewise_snv,
                contract_note="nirs4all supplies the SNV transform; the adapter applies it independently to C4A intervals",
            ),
        ),
    ),
    MethodSpec(
        "piecewise_msc",
        "Piecewise MSC",
        "preprocessing",
        py_transform(c4a.PiecewiseMSC, window_size=16),
        lambda ctx: lambda: c_transform(
            "c4a_pp_piecewise_msc", ctx.X, _null_f64_ptr(), ctypes.c_int64(0),
            ctypes.c_int32(16), ctypes.c_double(1e-12), fit_X=ctx.X
        ),
        "piecewise_msc(X, reference = NULL, window = 16L, eps = 1e-12)",
        (
            sklearn_ref(
                "sklearn.linear_model.LinearRegression(MSC per interval)",
                ref_sklearn_piecewise_msc,
                contract_note="sklearn supplies the per-row least-squares fit for each C4A piecewise MSC interval",
            ),
            r_prospectr_ref(
                "prospectr.msc per interval",
                "{out <- matrix(NA_real_, nrow=nrow(X), ncol=ncol(X)); for (lo in seq(1L, ncol(X), by=16L)) { hi <- min(ncol(X), lo + 16L - 1L); out[, lo:hi] <- prospectr::msc(X[, lo:hi, drop=FALSE], ref_spectrum = colMeans(X)[lo:hi]); }; out}",
                compare=True,
                gate_c4a=True,
                contract_note="prospectr supplies the MSC transform; gated on small matrices because its Armadillo normal-equation solver drifts above 1e-6 on ill-conditioned 16-column windows",
                max_cols=96,
            ),
        ),
    ),
    MethodSpec(
        "localized_msc",
        "Localized MSC",
        "preprocessing",
        py_transform(c4a.LocalizedMSC, window_size=17),
        lambda ctx: lambda: c_transform(
            "c4a_pp_localized_msc", ctx.X, _null_f64_ptr(), ctypes.c_int64(0),
            ctypes.c_int32(17), ctypes.c_double(1e-12), fit_X=ctx.X
        ),
        "localized_msc(X, reference = NULL, window = 17L, eps = 1e-12)",
        (
            sklearn_ref(
                "sklearn.linear_model.LinearRegression(sliding local MSC)",
                ref_sklearn_localized_msc,
                contract_note="sklearn supplies the per-window least-squares fit for C4A localized MSC",
            ),
            r_prospectr_ref(
                "prospectr.msc sliding interval",
                "{ref <- colMeans(X); out <- matrix(NA_real_, nrow=nrow(X), ncol=ncol(X)); half <- 17L %/% 2L; for (j in seq_len(ncol(X))) { lo <- max(1L, j - half); hi <- min(ncol(X), j + half); tmp <- prospectr::msc(X[, lo:hi, drop=FALSE], ref_spectrum = ref[lo:hi]); out[, j] <- tmp[, j - lo + 1L]; }; out}",
                compare=True,
                gate_c4a=True,
                contract_note="prospectr supplies the MSC transform over each local C4A window",
                max_cols=96,
            ),
            scipy_ref(
                "scipy.ndimage.convolve1d(local MSC window sums)",
                ref_scipy_localized_msc,
                compare=False,
                gate_c4a=False,
                contract_note="SciPy supplies moving-window sums, but the raw-sum contract is numerically unstable on near-singular windows; sklearn is the strict gate",
            ),
        ),
    ),
    MethodSpec(
        "cross_correlation_alignment",
        "Cross-correlation alignment",
        "alignment",
        py_transform(c4a.CrossCorrelationAlignment, max_shift=2),
        lambda ctx: lambda: c_transform(
            "c4a_pp_xcorr_align", ctx.X, _null_f64_ptr(), ctypes.c_int64(0),
            ctypes.c_int32(0), ctypes.c_int32(2), fit_X=ctx.X
        ),
        "cross_correlation_alignment(X, reference = NULL, interval_size = 0L, max_shift = 2L)",
        (
            scipy_ref(
                "scipy.ndimage.shift(cross-correlation shift search)",
                ref_scipy_cross_correlation_alignment,
                contract_note="SciPy supplies the edge-clamped interpolation used for each tested integer shift",
            ),
        ),
    ),
    MethodSpec(
        "icoshift_alignment",
        "Icoshift alignment",
        "alignment",
        py_transform(c4a.IcoshiftAlignment, interval_size=16, max_shift=2),
        lambda ctx: lambda: c_transform(
            "c4a_pp_icoshift_align", ctx.X, _null_f64_ptr(), ctypes.c_int64(0),
            ctypes.c_int32(16), ctypes.c_int32(2), fit_X=ctx.X
        ),
        "icoshift_alignment(X, reference = NULL, interval_size = 16L, max_shift = 2L)",
        (
            scipy_ref(
                "scipy.ndimage.shift(interval-correlation shift search)",
                ref_scipy_icoshift_alignment,
                contract_note="SciPy supplies the edge-clamped interpolation used inside each C4A interval",
            ),
        ),
    ),
    MethodSpec(
        "dynamic_time_warping_alignment",
        "Dynamic time warping alignment",
        "alignment",
        py_transform(c4a.DynamicTimeWarpingAlignment),
        lambda ctx: lambda: c_transform(
            "c4a_pp_dtw_align", ctx.X, _null_f64_ptr(), ctypes.c_int64(0),
            ctypes.c_int32(0), ctypes.c_int32(0), fit_X=ctx.X
        ),
        "dynamic_time_warping_alignment(X, reference = NULL, interval_size = 0L, max_shift = 0L)",
        (
            ReferenceSpec(
                "ref.dtw_python",
                "dtw-python.dtw(step_pattern=symmetric1)",
                ref_dtw_python_alignment,
                contract_note=(
                    "dtw-python provides the DTW path; the parity adapter maps that "
                    "external path to c4a's fixed-length output by averaging query "
                    "samples assigned to each reference index"
                ),
                max_cols=96,
            ),
        ),
        max_cols=96,
    ),
    MethodSpec(
        "correlation_optimized_warping",
        "Correlation optimized warping",
        "alignment",
        py_transform(c4a.CorrelationOptimizedWarping, interval_size=16, max_shift=2),
        lambda ctx: lambda: c_transform(
            "c4a_pp_cow_align", ctx.X, _null_f64_ptr(), ctypes.c_int64(0),
            ctypes.c_int32(16), ctypes.c_int32(2), fit_X=ctx.X
        ),
        "correlation_optimized_warping(X, reference = NULL, interval_size = 16L, max_shift = 2L)",
        (
            cowarp_ref(
                "cowarp.warp(segment_length=16, slack=2)",
                ref_cowarp_correlation_optimized_warping,
                contract_note="cowarp provides the external dynamic-programming COW contract used by this gate",
                max_cols=512,
            ),
        ),
        max_cols=512,
    ),
    MethodSpec(
        "variance_filter",
        "Variance filter",
        "feature_selection",
        py_variance_filter,
        lambda ctx: lambda: c_selector("c4a_filter_variance", ctx, threshold=0.0),
        "variance_filter(X, threshold = 0.0, top_k = 0L)",
        (sklearn_ref("sklearn.feature_selection.VarianceThreshold", ref_sklearn_variance_threshold),),
    ),
    MethodSpec(
        "correlation_filter",
        "Correlation filter",
        "feature_selection",
        py_correlation_filter,
        lambda ctx: lambda: c_selector(
            "c4a_filter_correlation", ctx, top_k=min(5, ctx.X.shape[1])
        ),
        "correlation_filter(X, y, threshold = 0.0, top_k = min(5L, ncol(X)))",
        (sklearn_ref("sklearn.feature_selection.r_regression", ref_sklearn_correlation_filter),),
    ),
    MethodSpec(
        "interval_generator",
        "Interval generator",
        "feature_selection",
        py_interval_generator,
        lambda ctx: lambda: c_interval_generator(ctx),
        "interval_generator(X, interval_size = 16L, step = 8L)",
        (
            sklearn_ref(
                "sklearn.compose.ColumnTransformer(passthrough intervals)",
                ref_sklearn_interval_generator,
                contract_note="sklearn supplies the column-selection transformer; the adapter returns C4A's list-of-intervals contract",
            ),
        ),
    ),
    MethodSpec("crop", "Crop", "resampling", py_crop, c_crop_factory, "{start <- ncol(X) %/% 8L; end <- ncol(X) - ncol(X) %/% 8L; crop_transform(X, start = start, end = end)}", (nirs_ref("nirs4all.CropTransformer", ref_nirs_crop),)),
    MethodSpec("resample", "Resample", "resampling", py_resample, lambda ctx: lambda: c_resample(ctx), "resample_transform(X, num_samples = max(4L, ncol(X) %/% 2L))", (scipy_ref("scipy.interpolate.interp1d", ref_scipy_resample), nirs_ref("nirs4all.ResampleTransformer", lambda ctx: lambda: _nirs_class("nirs4all.operators.transforms.features", "ResampleTransformer")(num_samples=max(4, ctx.X.shape[1] // 2)).fit_transform(np.array(ctx.X, copy=True))))),
    MethodSpec("resampler", "Wavelength resampler", "resampling", py_resampler, lambda ctx: lambda: c_resampler(ctx), "{source <- seq(900, 1700, length.out = ncol(X)); target <- seq(source[[1L]], source[[length(source)]], length.out = max(4L, ncol(X) %/% 2L)); resampler(X, source_wavelengths = source, target_wavelengths = target)}", (nirs_ref("nirs4all.Resampler", ref_nirs_resampler),)),
    MethodSpec("range_discretizer", "Range discretizer", "resampling", py_range_disc, lambda ctx: lambda: c_range_disc(ctx), "range_discretizer(X, edges = c(0.25, 0.40, 0.55, 0.70))", (nirs_ref("nirs4all.RangeDiscretizer", ref_nirs_transform("nirs4all.operators.transforms.targets", "RangeDiscretizer", bins=[0.25, 0.40, 0.55, 0.70])),)),
    MethodSpec("kbins_discretizer", "K-bins discretizer", "resampling", py_kbins, lambda ctx: lambda: c_kbins(ctx), "kbins_discretizer(X, n_bins = 5L, strategy = 'uniform')", (sklearn_ref("sklearn.preprocessing.KBinsDiscretizer", ref_sklearn_kbins), nirs_ref("nirs4all.IntegerKBinsDiscretizer", ref_nirs_transform("nirs4all.operators.transforms.targets", "IntegerKBinsDiscretizer", n_bins=5, strategy="uniform")))),
    MethodSpec("kennard_stone", "Kennard-Stone", "splitter", py_ks, lambda ctx: lambda: c_split_kennard_stone(ctx), "kennard_stone(X)", (nirs_ref("nirs4all.KennardStoneSplitter", ref_nirs_split_ks),)),
    MethodSpec("spxy", "SPXY", "splitter", py_spxy, lambda ctx: lambda: c_split_spxy(ctx), "spxy(X, matrix(y, ncol = 1))", (nirs_ref("nirs4all.SPXYSplitter", ref_nirs_split_spxy),)),
    MethodSpec("kbins_stratified", "K-bins stratified", "splitter", py_kbins_stratified, lambda ctx: lambda: c_split_kbins_stratified(ctx), "kbins_stratified(matrix(y, ncol = 1L), test_size = 0.25, seed = 17, n_bins = 5L, strategy = 'uniform')", (
        sklearn_ref(
            "sklearn.model_selection.StratifiedShuffleSplit",
            ref_sklearn_kbins_stratified,
        ),
        nirs_ref(
            "nirs4all.KBinsStratifiedSplitter",
            ref_nirs_split_kbins_stratified,
        ),
    )),
    MethodSpec("y_outlier_iqr", "Y outlier IQR", "filter", py_y_outlier, lambda ctx: lambda: c_y_outlier_iqr(ctx), "y_outlier_filter(y, method = 'iqr')$mask", (nirs_ref("nirs4all.YOutlierFilter", ref_nirs_y_outlier),)),
    MethodSpec("x_outlier_mahalanobis", "X outlier Mahalanobis", "filter", py_x_outlier, lambda ctx: lambda: c_x_outlier_mahalanobis(ctx), "x_outlier_filter(X, method = 'mahalanobis', n_components = min(5L, ncol(X)))$mask", (
        sklearn_ref("sklearn.PCA(full)+EmpiricalCovariance+chi2", ref_sklearn_x_outlier_mahalanobis),
        nirs_ref(
            "nirs4all.XOutlierFilter(PCA auto)",
            ref_nirs_x_outlier,
            compare=False,
            contract_note="nirs4all leaves sklearn PCA on auto solver, which switches solver on wide matrices; c4a gates the full-SVD PCA contract",
        ),
    )),
    MethodSpec(
        "log_transform",
        "Log transform",
        "preprocessing",
        py_transform(c4a.LogTransform, base=0.0, offset=0.0, auto_offset=True, min_value=1e-8),
        lambda ctx: lambda: c_log_transform(ctx),
        None,
        (
            numpy_ref("numpy.log(auto-offset)", ref_numpy_log_transform),
            nirs_ref("nirs4all.LogTransform", ref_nirs_transform("nirs4all.operators.transforms.nirs", "LogTransform", base=math.e, offset=0.0, auto_offset=True, min_value=1e-8)),
        ),
    ),
    MethodSpec(
        "derivate",
        "Derivate",
        "preprocessing",
        py_transform(c4a.Derivate, order=1, delta=1.0),
        lambda ctx: lambda: c_derivate(ctx),
        None,
        (
            numpy_ref("numpy.diff(axis=1)", ref_numpy_derivate),
            nirs_ref(
                "nirs4all.Derivate(same-width)",
                ref_nirs_transform("nirs4all.operators.transforms.scalers", "Derivate", order=1, delta=1),
                compare=False,
                gate_c4a=False,
                contract_note="nirs4all pads/interpolates back to the input width; c4a exposes the strict finite-difference width p-1 and gates NumPy",
                max_cols=512,
            ),
        ),
    ),
    MethodSpec(
        "wavelet_pca",
        "Wavelet PCA",
        "wavelet",
        py_transform(c4a.WaveletPCA, family="haar", mode="periodization", max_level=2, n_components=5.0),
        lambda ctx: lambda: c_wavelet_projection("c4a_pp_wavelet_pca", ctx),
        None,
        (
            pywavelets_ref(
                "PyWavelets.wavedec(haar, periodization)+sklearn.PCA(full)",
                lambda ctx: ref_pywavelets_wavelet_projection(ctx, kind="pca"),
                comparator=outputs_close_sign_invariant_columns,
            ),
            nirs_ref(
                "nirs4all.WaveletPCA(per-level contract)",
                ref_nirs_transform("nirs4all.operators.transforms.nirs", "WaveletPCA", wavelet="haar", max_level=2, n_components_per_level=3),
                compare=False,
                gate_c4a=False,
                contract_note="nirs4all fits per-level PCA blocks; c4a gates the flattened-DWT PCA contract with PyWavelets+sklearn",
                max_cols=512,
            ),
        ),
        comparator=outputs_close_sign_invariant_columns,
        max_cols=512,
    ),
    MethodSpec(
        "wavelet_svd",
        "Wavelet SVD",
        "wavelet",
        py_transform(c4a.WaveletSVD, family="haar", mode="periodization", max_level=2, n_components=5.0),
        lambda ctx: lambda: c_wavelet_projection("c4a_pp_wavelet_svd", ctx),
        None,
        (
            pywavelets_ref(
                "PyWavelets.wavedec(haar, periodization)+sklearn.TruncatedSVD",
                lambda ctx: ref_pywavelets_wavelet_projection(ctx, kind="svd"),
                comparator=outputs_close_sign_invariant_columns,
            ),
            nirs_ref(
                "nirs4all.WaveletSVD(per-level contract)",
                ref_nirs_transform("nirs4all.operators.transforms.nirs", "WaveletSVD", wavelet="haar", max_level=2, n_components_per_level=3),
                compare=False,
                gate_c4a=False,
                contract_note="nirs4all fits per-level SVD blocks; c4a gates the flattened-DWT SVD contract with PyWavelets+sklearn",
                max_cols=512,
            ),
        ),
        comparator=outputs_close_sign_invariant_columns,
        max_cols=512,
    ),
    MethodSpec("high_leverage", "High leverage filter", "filter", py_high_leverage, lambda ctx: lambda: c_high_leverage(ctx), None, (nirs_ref("nirs4all.HighLeverageFilter", ref_nirs_high_leverage, max_cols=512),), max_cols=512),
    MethodSpec("spectral_quality", "Spectral quality filter", "filter", py_spectral_quality, lambda ctx: lambda: c_spectral_quality(ctx), "spectral_quality(X)", (nirs_ref("nirs4all.SpectralQualityFilter", ref_nirs_spectral_quality),)),
    MethodSpec("composite_filter", "Composite filter", "filter", py_composite_filter, lambda ctx: lambda: c_composite_filter(ctx), None, (nirs_ref("nirs4all.CompositeFilter", ref_nirs_composite_filter, max_cols=512),), max_cols=512),
    MethodSpec("hotelling_t2", "Hotelling T2", "diagnostic", py_hotelling_t2, lambda ctx: lambda: c_hotelling_t2(ctx), None, (sklearn_ref("sklearn.PCA + scipy.stats.f", ref_sklearn_hotelling_t2),), max_cols=512),
    MethodSpec("q_residuals", "Q residuals", "diagnostic", py_q_residuals, lambda ctx: lambda: c_q_residuals(ctx), None, (sklearn_ref("sklearn.PCA + scipy.stats.norm", ref_sklearn_q_residuals),), max_cols=512),
    MethodSpec("nirs_metrics", "NIRS metrics", "metric", py_nirs_metrics, lambda ctx: lambda: c_nirs_metrics(ctx), None, (numpy_ref("NumPy regression metric formulae", ref_numpy_nirs_metrics),)),
    MethodSpec("transfer_metrics", "Transfer metrics", "metric", py_transfer_metrics, lambda ctx: lambda: c_transfer_metrics(ctx), None, (nirs_ref("nirs4all.TransferMetricsComputer", ref_nirs_transfer_metrics),), max_cols=512),
    MethodSpec("signal_type_detector", "Signal type detector", "diagnostic", py_signal_type, lambda ctx: lambda: c_signal_type(ctx), None, (nirs_ref("nirs4all.detect_signal_type", ref_nirs_signal_type),)),
    MethodSpec("rng_pcg64", "PCG64 RNG", "utility", lambda ctx: lambda: c4a_python.rng_pcg64(seed=ctx.seed, n=1024), lambda ctx: lambda: c_rng_pcg64(ctx), None, (numpy_ref("numpy.random.default_rng(PCG64).standard_normal", ref_numpy_rng_pcg64),)),
    MethodSpec(
        "aug_gaussian_noise",
        "Gaussian additive noise",
        "augmentation",
        py_transform(c4a.GaussianAdditiveNoise, sigma=0.01, seed=17),
        lambda ctx: lambda: c_aug_apply("c4a_aug_gaussian_noise", ctx, ctypes.c_double(0.01)),
        "aug_gaussian_noise(X)",
        (nirs_ref("nirs4all.GaussianAdditiveNoise", ref_nirs_augmenter("nirs4all.operators.augmentation.spectral", "GaussianAdditiveNoise", sigma=0.01, smoothing_kernel_width=1, random_state=17, variation_scope="sample")),),
    ),
    MethodSpec(
        "aug_multiplicative_noise",
        "Multiplicative noise",
        "augmentation",
        py_transform(c4a.MultiplicativeNoise, sigma_gain=0.01, seed=17),
        lambda ctx: lambda: c_aug_apply("c4a_aug_multiplicative_noise", ctx, ctypes.c_double(0.01)),
        "aug_multiplicative_noise(X)",
        (nirs_ref("nirs4all.MultiplicativeNoise", ref_nirs_augmenter("nirs4all.operators.augmentation.spectral", "MultiplicativeNoise", sigma_gain=0.01, per_wavelength=False, random_state=17, variation_scope="sample")),),
    ),
    MethodSpec(
        "aug_spike_noise",
        "Spike noise",
        "augmentation",
        py_transform(c4a.SpikeNoise, n_spikes_min=1, n_spikes_max=3, amplitude_min=-0.1, amplitude_max=0.1, seed=17),
        lambda ctx: lambda: c_aug_apply("c4a_aug_spike_noise", ctx, ctypes.c_int32(1), ctypes.c_int32(3), ctypes.c_double(-0.1), ctypes.c_double(0.1)),
        "aug_spike_noise(X)",
        (nirs_ref("nirs4all.SpikeNoise", ref_nirs_augmenter("nirs4all.operators.augmentation.spectral", "SpikeNoise", n_spikes_range=(1, 3), amplitude_range=(-0.1, 0.1), random_state=17, variation_scope="sample")),),
    ),
    MethodSpec(
        "aug_hetero_noise",
        "Heteroscedastic noise",
        "augmentation",
        py_transform(c4a.HeteroscedasticNoiseAugmenter, noise_base=0.001, noise_signal_dep=0.01, seed=17),
        lambda ctx: lambda: c_aug_apply("c4a_aug_hetero_noise", ctx, ctypes.c_double(0.001), ctypes.c_double(0.01)),
        "aug_hetero_noise(X)",
        (nirs_ref("nirs4all.HeteroscedasticNoiseAugmenter", ref_nirs_augmenter("nirs4all.operators.augmentation.synthesis", "HeteroscedasticNoiseAugmenter", noise_base=0.001, noise_signal_dep=0.01, random_state=17, variation_scope="sample")),),
    ),
    MethodSpec(
        "aug_linear_drift",
        "Linear baseline drift",
        "augmentation",
        py_transform(c4a.LinearBaselineDrift, offset_min=-0.05, offset_max=0.05, slope_min=-0.01, slope_max=0.01, seed=17),
        lambda ctx: lambda: c_aug_apply("c4a_aug_linear_drift", ctx, ctypes.c_double(-0.05), ctypes.c_double(0.05), ctypes.c_double(-0.01), ctypes.c_double(0.01)),
        "aug_linear_drift(X)",
        (nirs_ref("nirs4all.LinearBaselineDrift", ref_nirs_augmenter("nirs4all.operators.augmentation.spectral", "LinearBaselineDrift", offset_range=(-0.05, 0.05), slope_range=(-0.01, 0.01), random_state=17)),),
    ),
    MethodSpec(
        "aug_poly_drift",
        "Polynomial baseline drift",
        "augmentation",
        py_transform(c4a.PolynomialBaselineDrift, degree=2, seed=17),
        lambda ctx: lambda: c_aug_poly_drift(ctx),
        "aug_poly_drift(X)",
        (nirs_ref("nirs4all.PolynomialBaselineDrift", ref_nirs_augmenter("nirs4all.operators.augmentation.spectral", "PolynomialBaselineDrift", degree=2, coeff_ranges=[(-0.01, 0.01), (-0.01, 0.01), (-0.01, 0.01)], random_state=17)),),
    ),
    MethodSpec(
        "aug_path_length",
        "Path length augmenter",
        "augmentation",
        py_transform(c4a.PathLengthAugmenter, path_length_std=0.05, min_path_length=0.1, seed=17),
        lambda ctx: lambda: c_aug_apply("c4a_aug_path_length", ctx, ctypes.c_double(0.05), ctypes.c_double(0.1)),
        "aug_path_length(X)",
        (nirs_ref("nirs4all.PathLengthAugmenter", ref_nirs_augmenter("nirs4all.operators.augmentation.synthesis", "PathLengthAugmenter", path_length_std=0.05, min_path_length=0.1, random_state=17, variation_scope="sample")),),
    ),
    MethodSpec("aug_wavelength_shift", "Wavelength shift augmenter", "augmentation", lambda ctx: lambda: c4a.WavelengthShift(wavelengths=ctx.wavelengths, seed=17).fit_transform(ctx.X), lambda ctx: lambda: c_aug_wavelength_shift(ctx), "aug_wavelength_shift(X)", (nirs_ref("nirs4all.WavelengthShift", ref_nirs_augmenter("nirs4all.operators.augmentation.spectral", "WavelengthShift", wavelengths=True, shift_range=(-1.0, 1.0), random_state=17)),)),
    MethodSpec("aug_wavelength_stretch", "Wavelength stretch augmenter", "augmentation", lambda ctx: lambda: c4a.WavelengthStretch(wavelengths=ctx.wavelengths, seed=17).fit_transform(ctx.X), lambda ctx: lambda: c_aug_wavelength_stretch(ctx), "aug_wavelength_stretch(X)", (nirs_ref("nirs4all.WavelengthStretch", ref_nirs_augmenter("nirs4all.operators.augmentation.spectral", "WavelengthStretch", wavelengths=True, stretch_range=(0.99, 1.01), random_state=17)),)),
    MethodSpec("aug_local_warp", "Local wavelength warp augmenter", "augmentation", lambda ctx: lambda: c4a.LocalWarpAugmenter(n_control_points=3, wavelengths=ctx.wavelengths, seed=17).fit_transform(ctx.X), lambda ctx: lambda: c_aug_local_warp(ctx), "aug_local_warp(X)", (nirs_ref("nirs4all.LocalWavelengthWarp", ref_nirs_augmenter("nirs4all.operators.augmentation.spectral", "LocalWavelengthWarp", wavelengths=True, n_control_points=3, max_shift=1.0, random_state=17)),)),
    MethodSpec("aug_band_perturb", "Band perturbation augmenter", "augmentation", py_transform(c4a.BandPerturbationAugmenter, n_bands=3, bw_lo=5, bw_hi=15, gain_lo=0.9, gain_hi=1.1, offset_lo=-0.01, offset_hi=0.01, seed=17), lambda ctx: lambda: c_aug_band_perturb(ctx), "aug_band_perturb(X)", (nirs_ref("nirs4all.BandPerturbation", ref_nirs_augmenter("nirs4all.operators.augmentation.spectral", "BandPerturbation", n_bands=3, bandwidth_range=(5, 15), gain_range=(0.9, 1.1), offset_range=(-0.01, 0.01), random_state=17)),)),
    MethodSpec("aug_band_mask", "Band masking augmenter", "augmentation", py_transform(c4a.BandMasking, n_bands_lo=1, n_bands_hi=3, bw_lo=5, bw_hi=15, mode="zero", seed=17), lambda ctx: lambda: c_aug_band_mask(ctx), "aug_band_mask(X)", (nirs_ref("nirs4all.BandMasking", ref_nirs_augmenter("nirs4all.operators.augmentation.spectral", "BandMasking", n_bands_range=(1, 3), bandwidth_range=(5, 15), mode="zero", random_state=17)),)),
    MethodSpec("aug_channel_dropout", "Channel dropout augmenter", "augmentation", py_transform(c4a.ChannelDropout, dropout_prob=0.05, mode="zero", seed=17), lambda ctx: lambda: c_aug_channel_dropout(ctx), "aug_channel_dropout(X)", (nirs_ref("nirs4all.ChannelDropout", ref_nirs_augmenter("nirs4all.operators.augmentation.spectral", "ChannelDropout", dropout_prob=0.05, mode="zero", random_state=17)),)),
    MethodSpec("aug_gauss_jitter", "Gaussian smoothing jitter", "augmentation", py_transform(c4a.GaussianJitter, sigma_lo=0.5, sigma_hi=1.5, kernel_width=9, seed=17), lambda ctx: lambda: c_aug_gauss_jitter(ctx), "aug_gauss_jitter(X)", (nirs_ref("nirs4all.GaussianSmoothingJitter", ref_nirs_augmenter("nirs4all.operators.augmentation.spectral", "GaussianSmoothingJitter", sigma_range=(0.5, 1.5), kernel_width=9, random_state=17)),)),
    MethodSpec("aug_unsharp_mask", "Unsharp spectral mask augmenter", "augmentation", py_transform(c4a.UnsharpMask, amount_lo=0.1, amount_hi=0.5, sigma=1.0, kernel_width=11, seed=17), lambda ctx: lambda: c_aug_unsharp_mask(ctx), "aug_unsharp_mask(X)", (nirs_ref("nirs4all.UnsharpSpectralMask", ref_nirs_augmenter("nirs4all.operators.augmentation.spectral", "UnsharpSpectralMask", amount_range=(0.1, 0.5), sigma=1.0, kernel_width=11, random_state=17)),)),
    MethodSpec("aug_magnitude_warp", "Smooth magnitude warp augmenter", "augmentation", lambda ctx: lambda: c4a.MagnitudeWarp(n_control_points=3, wavelengths=ctx.wavelengths, seed=17).fit_transform(ctx.X), lambda ctx: lambda: c_aug_magnitude_warp(ctx), "aug_magnitude_warp(X)", (nirs_ref("nirs4all.SmoothMagnitudeWarp", ref_nirs_augmenter("nirs4all.operators.augmentation.spectral", "SmoothMagnitudeWarp", wavelengths=True, n_control_points=3, gain_range=(0.9, 1.1), random_state=17)),)),
    MethodSpec("aug_local_clip", "Local clipping augmenter", "augmentation", py_transform(c4a.LocalClip, n_regions=1, width_lo=5, width_hi=15, seed=17), lambda ctx: lambda: c_aug_local_clip(ctx), "aug_local_clip(X)", (nirs_ref("nirs4all.LocalClipping", ref_nirs_augmenter("nirs4all.operators.augmentation.spectral", "LocalClipping", n_regions=1, width_range=(5, 15), random_state=17)),)),
    MethodSpec("aug_wavelength_spectral", "Wavelength/spectral augmenters", "augmentation", py_wavelength_spectral, lambda ctx: lambda: c_aug_wavelength_spectral(ctx), "aug_wavelength_spectral(X)", (nirs_ref("nirs4all wavelength/spectral augmenter bundle", ref_nirs_wavelength_spectral),)),
    MethodSpec("aug_mixup", "Mixup augmenter", "augmentation", py_transform(c4a.MixupAugmenter, alpha=1.0, seed=17), lambda ctx: lambda: c_aug_apply("c4a_aug_mixup", ctx, ctypes.c_double(1.0)), "aug_mixup(X)", (nirs_ref("nirs4all.MixupAugmenter", ref_nirs_augmenter("nirs4all.operators.augmentation.spectral", "MixupAugmenter", alpha=1.0, random_state=17)),)),
    MethodSpec("aug_local_mixup", "Local mixup augmenter", "augmentation", py_transform(c4a.LocalMixupAugmenter, alpha=1.0, k_neighbors=5, seed=17), lambda ctx: lambda: c_aug_apply("c4a_aug_local_mixup", ctx, ctypes.c_double(1.0), ctypes.c_int32(5)), "aug_local_mixup(X)", (nirs_ref("nirs4all.LocalMixupAugmenter", ref_nirs_augmenter("nirs4all.operators.augmentation.spectral", "LocalMixupAugmenter", alpha=1.0, k_neighbors=5, random_state=17)),)),
    MethodSpec("aug_scatter_sim", "Scatter simulation MSC", "augmentation", py_transform(c4a.ScatterSimulationMSC, a_low=-0.05, a_high=0.05, b_low=0.9, b_high=1.1, seed=17), lambda ctx: lambda: c_aug_apply("c4a_aug_scatter_sim", ctx, ctypes.c_double(-0.05), ctypes.c_double(0.05), ctypes.c_double(0.9), ctypes.c_double(1.1)), "aug_scatter_sim(X)", (nirs_ref("nirs4all.ScatterSimulationMSC", ref_nirs_augmenter("nirs4all.operators.augmentation.spectral", "ScatterSimulationMSC", a_range=(-0.05, 0.05), b_range=(0.9, 1.1), random_state=17)),)),
    MethodSpec("aug_particle_size", "Particle size augmenter", "augmentation", lambda ctx: lambda: c4a.ParticleSizeAugmenter(wavelengths=ctx.wavelengths, seed=17).fit_transform(ctx.X), lambda ctx: lambda: c_aug_particle_size(ctx), "aug_particle_size(X)", (nirs_ref("nirs4all.ParticleSizeAugmenter", ref_nirs_augmenter("nirs4all.operators.augmentation.scattering", "ParticleSizeAugmenter", wavelengths=True, mean_size_um=50.0, size_variation_um=15.0, reference_size_um=50.0, wavelength_exponent=1.5, size_effect_strength=0.1, include_path_length=True, path_length_sensitivity=0.5, random_state=17)),)),
    MethodSpec("aug_emsc_distort", "EMSC distortion augmenter", "augmentation", lambda ctx: lambda: c4a.EMSCDistortionAugmenter(wavelengths=ctx.wavelengths, seed=17).fit_transform(ctx.X), lambda ctx: lambda: c_aug_emsc_distort(ctx), "aug_emsc_distort(X)", (nirs_ref("nirs4all.EMSCDistortionAugmenter", ref_nirs_augmenter("nirs4all.operators.augmentation.scattering", "EMSCDistortionAugmenter", wavelengths=True, multiplicative_range=(0.9, 1.1), additive_range=(-0.05, 0.05), polynomial_order=2, polynomial_strength=0.02, correlation=0.3, random_state=17)),)),
    MethodSpec("aug_batch_effect", "Batch effect augmenter", "augmentation", lambda ctx: lambda: c4a.BatchEffectAugmenter(offset_std=0.02, slope_std=0.01, gain_std=0.03, variation_scope=0, wavelengths=ctx.wavelengths, seed=17).fit_transform(ctx.X), lambda ctx: lambda: c_aug_batch_effect(ctx), "aug_batch_effect(X)", (nirs_ref("nirs4all.BatchEffectAugmenter", ref_nirs_augmenter("nirs4all.operators.augmentation.synthesis", "BatchEffectAugmenter", wavelengths=True, offset_std=0.02, slope_std=0.01, gain_std=0.03, variation_scope="sample", random_state=17)),)),
    MethodSpec("aug_instrument_broaden", "Instrumental broadening augmenter", "augmentation", lambda ctx: lambda: c4a.InstrumentalBroadeningAugmenter(fwhm=3.0, use_fwhm_range=False, variation_scope=0, wavelengths=ctx.wavelengths, seed=17).fit_transform(ctx.X), lambda ctx: lambda: c_aug_instrument_broaden(ctx), "aug_instrument_broaden(X)", (nirs_ref("nirs4all.InstrumentalBroadeningAugmenter", ref_nirs_augmenter("nirs4all.operators.augmentation.synthesis", "InstrumentalBroadeningAugmenter", wavelengths=True, fwhm=3.0, variation_scope="sample", random_state=17)),)),
    MethodSpec("aug_dead_band", "Dead band augmenter", "augmentation", py_transform(c4a.DeadBandAugmenter, n_bands=1, width_low=5, width_high=10, noise_std=0.05, probability=1.0, variation_scope=0, seed=17), lambda ctx: lambda: c_aug_apply("c4a_aug_dead_band", ctx, ctypes.c_int32(1), ctypes.c_int32(5), ctypes.c_int32(10), ctypes.c_double(0.05), ctypes.c_double(1.0), ctypes.c_int32(0)), "aug_dead_band(X)", (nirs_ref("nirs4all.DeadBandAugmenter", ref_nirs_augmenter("nirs4all.operators.augmentation.synthesis", "DeadBandAugmenter", n_bands=1, width_range=(5, 10), noise_std=0.05, probability=1.0, random_state=17, variation_scope="sample")),)),
    MethodSpec("aug_temperature", "Temperature augmenter", "augmentation", lambda ctx: lambda: c4a.TemperatureAugmenter(temperature_delta=5.0, wavelengths=ctx.wavelengths, seed=17).fit_transform(ctx.X), lambda ctx: lambda: c_aug_temperature(ctx), "aug_temperature(X)", (nirs_ref("nirs4all.TemperatureAugmenter", ref_nirs_augmenter("nirs4all.operators.augmentation.environmental", "TemperatureAugmenter", wavelengths=True, temperature_delta=5.0, enable_shift=True, enable_intensity=True, enable_broadening=True, region_specific=True, random_state=17)),)),
    MethodSpec("aug_moisture", "Moisture augmenter", "augmentation", lambda ctx: lambda: c4a.MoistureAugmenter(water_activity_delta=0.1, wavelengths=ctx.wavelengths, seed=17).fit_transform(ctx.X), lambda ctx: lambda: c_aug_moisture(ctx), "aug_moisture(X)", (nirs_ref("nirs4all.MoistureAugmenter", ref_nirs_augmenter("nirs4all.operators.augmentation.environmental", "MoistureAugmenter", wavelengths=True, water_activity_delta=0.1, reference_water_activity=0.5, free_water_fraction=0.3, bound_water_shift=25.0, moisture_content=0.1, enable_shift=True, enable_intensity=True, random_state=17)),)),
    MethodSpec("aug_detector_rolloff", "Detector roll-off augmenter", "augmentation", lambda ctx: lambda: c4a.DetectorRollOffAugmenter(wavelengths=ctx.wavelengths, seed=17).fit_transform(ctx.X), lambda ctx: lambda: c_aug_detector_rolloff(ctx), "aug_detector_rolloff(X)", (nirs_ref("nirs4all.DetectorRollOffAugmenter", ref_nirs_augmenter("nirs4all.operators.augmentation.edge_artifacts", "DetectorRollOffAugmenter", wavelengths=True, detector_model="generic_nir", effect_strength=1.0, noise_amplification=0.02, include_baseline_distortion=True, random_state=17)),)),
    MethodSpec("aug_stray_light", "Stray light augmenter", "augmentation", lambda ctx: lambda: c4a.StrayLightAugmenter(wavelengths=ctx.wavelengths, seed=17).fit_transform(ctx.X), lambda ctx: lambda: c_aug_stray_light(ctx), "aug_stray_light(X)", (nirs_ref("nirs4all.StrayLightAugmenter", ref_nirs_augmenter("nirs4all.operators.augmentation.edge_artifacts", "StrayLightAugmenter", wavelengths=True, stray_light_fraction=0.001, edge_enhancement=2.0, edge_width=0.1, include_peak_truncation=True, random_state=17)),)),
    MethodSpec("aug_edge_curve", "Edge curvature augmenter", "augmentation", lambda ctx: lambda: c4a.EdgeCurvatureAugmenter(curvature_type=0, wavelengths=ctx.wavelengths, seed=17).fit_transform(ctx.X), lambda ctx: lambda: c_aug_edge_curve(ctx), "aug_edge_curve(X)", (nirs_ref("nirs4all.EdgeCurvatureAugmenter", ref_nirs_augmenter("nirs4all.operators.augmentation.edge_artifacts", "EdgeCurvatureAugmenter", wavelengths=True, curvature_strength=0.02, curvature_type="random", asymmetry=0.0, edge_focus=0.7, random_state=17)),)),
    MethodSpec("aug_truncated_peak", "Truncated peak augmenter", "augmentation", lambda ctx: lambda: c4a.TruncatedPeakAugmenter(wavelengths=ctx.wavelengths, seed=17).fit_transform(ctx.X), lambda ctx: lambda: c_aug_truncated_peak(ctx), "aug_truncated_peak(X)", (nirs_ref("nirs4all.TruncatedPeakAugmenter", ref_nirs_augmenter("nirs4all.operators.augmentation.edge_artifacts", "TruncatedPeakAugmenter", wavelengths=True, peak_probability=0.5, amplitude_range=(0.01, 0.1), width_range=(50, 200), left_edge=True, right_edge=True, random_state=17)),)),
    MethodSpec("aug_edge_artifacts", "Combined edge artifact augmenter", "augmentation", lambda ctx: lambda: c4a.EdgeArtifactsAugmenter(wavelengths=ctx.wavelengths, seed=17).fit_transform(ctx.X), lambda ctx: lambda: c_aug_edge_artifacts(ctx), "aug_edge_artifacts(X)", (nirs_ref("nirs4all.EdgeArtifactsAugmenter", ref_nirs_augmenter("nirs4all.operators.augmentation.edge_artifacts", "EdgeArtifactsAugmenter", wavelengths=True, detector_roll_off=True, stray_light=True, edge_curvature=True, truncated_peaks=True, overall_strength=1.0, detector_model="generic_nir", random_state=17)),)),
    MethodSpec("aug_spline_smooth", "Spline smoothing augmenter", "augmentation", py_transform(c4a.SplineSmoothingAugmenter, seed=17), lambda ctx: lambda: c_aug_spline_smooth(ctx), "aug_spline_smooth(X)", (nirs_ref("nirs4all.Spline_Smoothing", ref_nirs_augmenter("nirs4all.operators.augmentation.splines", "Spline_Smoothing", random_state=17)),)),
    MethodSpec("aug_spline_x_perturb", "Spline X perturbation augmenter", "augmentation", py_transform(c4a.SplineXPerturbationAugmenter, spline_degree=3, perturbation_density=0.05, perturbation_range_min=-0.1, perturbation_range_max=0.1, seed=17), lambda ctx: lambda: c_aug_spline_x_perturb(ctx), "aug_spline_x_perturb(X)", (nirs_ref("nirs4all.Spline_X_Perturbations", ref_nirs_augmenter("nirs4all.operators.augmentation.splines", "Spline_X_Perturbations", spline_degree=3, perturbation_density=0.05, perturbation_range=(-0.1, 0.1), random_state=17)),)),
    MethodSpec("aug_spline_y_perturb", "Spline Y perturbation augmenter", "augmentation", py_transform(c4a.SplineYPerturbationAugmenter, spline_points=-1, perturbation_intensity=0.005, seed=17), lambda ctx: lambda: c_aug_spline_y_perturb(ctx), "aug_spline_y_perturb(X)", (nirs_ref("nirs4all.Spline_Y_Perturbations", ref_nirs_augmenter("nirs4all.operators.augmentation.splines", "Spline_Y_Perturbations", spline_points=None, perturbation_intensity=0.005, random_state=17)),)),
    MethodSpec("aug_rotate_translate", "Rotate/translate augmenter", "augmentation", py_transform(c4a.RotateTranslateAugmenter, p_range=2.0, y_factor=3.0, seed=17), lambda ctx: lambda: c_aug_rotate_translate(ctx), "aug_rotate_translate(X)", (nirs_ref("nirs4all.Rotate_Translate", ref_nirs_augmenter("nirs4all.operators.augmentation.random", "Rotate_Translate", p_range=2, y_factor=3, random_state=17)),)),
    MethodSpec("aug_random_x_op", "Random X operation augmenter", "augmentation", py_transform(c4a.RandomXOperation, op_kind="multiply", operator_range_min=0.97, operator_range_max=1.03, seed=17), lambda ctx: lambda: c_aug_random_x_op(ctx), "aug_random_x_op(X)", (nirs_ref("nirs4all.Random_X_Operation", ref_nirs_augmenter("nirs4all.operators.augmentation.random", "Random_X_Operation", operator_range=(0.97, 1.03), random_state=17)),)),
)


PARITY_VERIFIED = {spec.method_id for spec in METHODS}


def selected_methods(names: Iterable[str] | None) -> list[MethodSpec]:
    if not names:
        return list(METHODS)
    requested = [name.strip() for name in names if name.strip()]
    wanted = set(requested)
    by_id = {spec.method_id: spec for spec in METHODS}
    missing = sorted(wanted - set(by_id))
    if missing:
        raise SystemExit(f"unknown methods: {', '.join(missing)}")
    return [by_id[name] for name in requested]


def csv_row(
    spec: MethodSpec,
    *,
    backend: str,
    language: str,
    tier: str,
    kind: str,
    n: int,
    p: int,
    threads: int,
    lib_build: str,
    ok: bool,
    median_ms: float | None,
    parity: str,
    reason: str,
    reference_library: str = "",
    reference_role: str = "",
    binding_parity_ok: bool | None = None,
    reference_parity_ok: bool | None = None,
    samples_ms: Iterable[float] | None = None,
    repeat: int | None = None,
    batch_loops: int = 1,
    reference_divergence: dict[str, float] | None = None,
    binding_divergence: dict[str, float] | None = None,
) -> dict[str, str]:
    samples = list(samples_ms or [])
    timed_runs = repeat if repeat is not None else (len(samples) if samples else None)
    reference_divergence = reference_divergence or {}
    binding_divergence = binding_divergence or {}

    def metric(metrics: dict[str, float], key: str) -> str:
        value = metrics.get(key)
        return "" if value is None or not math.isfinite(value) else f"{value:.17g}"

    return {
        "algorithm": spec.method_id,
        "family": spec.family,
        "backend": backend,
        "language": language,
        "tier": tier,
        "kind": kind,
        "n": str(n),
        "p": str(p),
        "threads": str(threads),
        "lib_build": lib_build,
        "ok": bool_cell(ok),
        "median_ms": "" if median_ms is None or math.isnan(median_ms) else f"{median_ms:.6f}",
        "warmup_runs": "" if timed_runs is None else str(warmup_count(timed_runs)),
        "timed_runs": "" if timed_runs is None else str(timed_runs),
        "batch_loops": str(max(1, int(batch_loops))),
        "samples_ms": ";".join(f"{sample:.6f}" for sample in samples),
        "parity": parity,
        "reason": reason,
        "reference_library": reference_library,
        "reference_role": reference_role,
        "binding_parity_ok": bool_cell(binding_parity_ok),
        "reference_parity_ok": bool_cell(reference_parity_ok),
        "reference_max_abs_diff": metric(reference_divergence, "max_abs"),
        "reference_rms_diff": metric(reference_divergence, "rms"),
        "reference_rel_l2_diff": metric(reference_divergence, "rel_l2"),
        "binding_max_abs_diff": metric(binding_divergence, "max_abs"),
        "binding_rms_diff": metric(binding_divergence, "rms"),
        "binding_rel_l2_diff": metric(binding_divergence, "rel_l2"),
    }


def _json_samples(payload: dict) -> list[float]:
    raw = payload.get("samples_ms", [])
    if isinstance(raw, (int, float)):
        raw = [raw]
    return [float(value) for value in raw]


def run_r_timing(expr: str, ctx: Dataset, repeat: int) -> tuple[float, str, np.ndarray, list[float], int]:
    with tempfile.TemporaryDirectory(prefix="c4a-r-bench-") as tmp:
        tmp_path = Path(tmp)
        x_path = tmp_path / "X.csv"
        y_path = tmp_path / "y.txt"
        d_path = tmp_path / "d.txt"
        out_path = tmp_path / "out.csv"
        np.savetxt(x_path, ctx.X, delimiter=",")
        np.savetxt(y_path, ctx.y)
        np.savetxt(d_path, ctx.d)
        script = f"""
suppressPackageStartupMessages(library(chemometrics4all))
if (!requireNamespace("jsonlite", quietly = TRUE)) stop("jsonlite is required")
options(digits = 17, scipen = 999)
X <- as.matrix(read.csv("{x_path.as_posix()}", header = FALSE))
y <- scan("{y_path.as_posix()}", quiet = TRUE)
d <- scan("{d_path.as_posix()}", quiet = TRUE)
c4a_now_ms <- function() proc.time()[["elapsed"]] * 1000.0
c4a_consume <- function(out) {{
  if (is.list(out)) {{
    invisible(length(unlist(out, recursive = FALSE, use.names = FALSE)))
  }} else {{
    invisible(length(out))
  }}
}}
c4a_run_once <- function() {{
  out <- {expr}
  c4a_consume(out)
  out
}}
warmups <- max(1L, min(3L, as.integer({repeat})))
for (w in seq_len(warmups)) {{
  out <- c4a_run_once()
}}
batch_loops <- 1L
repeat {{
  t0 <- c4a_now_ms()
  for (j in seq_len(batch_loops)) out <- c4a_run_once()
  elapsed <- c4a_now_ms() - t0
  if (elapsed >= 100.0 || batch_loops >= 10000L) break
  batch_loops <- min(10000L, batch_loops * 2L)
}}
for (w in seq_len(warmups)) {{
  for (j in seq_len(batch_loops)) out <- c4a_run_once()
}}
gc(FALSE)
times <- numeric({repeat})
for (i in seq_len({repeat})) {{
  t0 <- c4a_now_ms()
  for (j in seq_len(batch_loops)) out <- c4a_run_once()
  times[[i]] <- (c4a_now_ms() - t0) / batch_loops
}}
if (is.list(out) && all(c("train_idx", "test_idx") %in% names(out))) {{
  out <- c(out$train_idx - 1L, out$test_idx - 1L)
}} else if (is.list(out)) {{
  out <- unlist(lapply(out, function(block) as.vector(t(as.matrix(block)))),
                recursive = TRUE, use.names = FALSE)
}}
if (is.logical(out)) {{
  out <- out + 0
}}
write.table(as.matrix(out), file = "{out_path.as_posix()}", sep = ",", row.names = FALSE, col.names = FALSE, quote = FALSE)
cat(jsonlite::toJSON(list(ok = TRUE, median_ms = median(times), samples_ms = as.numeric(times), batch_loops = batch_loops), auto_unbox = TRUE))
"""
        completed = subprocess.run(
            ["Rscript", "--vanilla", "-e", script],
            check=False,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=600,
        )
        if completed.returncode != 0:
            raise RuntimeError((completed.stderr or completed.stdout).strip())
        marker = completed.stdout.strip().splitlines()[-1]
        payload = json.loads(marker)
        output = np.loadtxt(out_path, delimiter=",")
        batch_loops = max(1, int(payload.get("batch_loops", 1)))
        return float(payload["median_ms"]), f"ok; batch_loops={batch_loops}", np.asarray(output, dtype=np.float64), _json_samples(payload), batch_loops


def run_r_reference(expr: str, ctx: Dataset, repeat: int) -> tuple[float, np.ndarray, list[float], int]:
    with tempfile.TemporaryDirectory(prefix="c4a-r-ref-bench-") as tmp:
        tmp_path = Path(tmp)
        x_path = tmp_path / "X.csv"
        y_path = tmp_path / "y.txt"
        d_path = tmp_path / "d.txt"
        out_path = tmp_path / "out.csv"
        np.savetxt(x_path, ctx.X, delimiter=",")
        np.savetxt(y_path, ctx.y)
        np.savetxt(d_path, ctx.d)
        script = f"""
if (!requireNamespace("jsonlite", quietly = TRUE)) stop("jsonlite is required")
options(digits = 17, scipen = 999)
X <- as.matrix(read.csv("{x_path.as_posix()}", header = FALSE, check.names = FALSE))
y <- scan("{y_path.as_posix()}", quiet = TRUE)
d <- scan("{d_path.as_posix()}", quiet = TRUE)
c4a_now_ms <- function() proc.time()[["elapsed"]] * 1000.0
ref_fun <- function() {{
  {expr}
}}
c4a_consume <- function(out) {{
  invisible(length(out))
}}
c4a_run_once <- function() {{
  out <- ref_fun()
  c4a_consume(out)
  out
}}
out <- ref_fun()
if (is.list(out)) stop("R external references must return a vector or matrix")
warmups <- max(1L, min(3L, as.integer({repeat})))
for (w in seq_len(warmups)) {{
  out <- c4a_run_once()
}}
batch_loops <- 1L
repeat {{
  t0 <- c4a_now_ms()
  for (j in seq_len(batch_loops)) out <- c4a_run_once()
  elapsed <- c4a_now_ms() - t0
  if (elapsed >= 100.0 || batch_loops >= 10000L) break
  batch_loops <- min(10000L, batch_loops * 2L)
}}
for (w in seq_len(warmups)) {{
  for (j in seq_len(batch_loops)) out <- c4a_run_once()
}}
gc(FALSE)
times <- numeric({repeat})
for (i in seq_len({repeat})) {{
  t0 <- c4a_now_ms()
  for (j in seq_len(batch_loops)) out <- c4a_run_once()
  times[[i]] <- (c4a_now_ms() - t0) / batch_loops
}}
write.table(as.matrix(out), file = "{out_path.as_posix()}", sep = ",", row.names = FALSE, col.names = FALSE, quote = FALSE)
cat(jsonlite::toJSON(list(ok = TRUE, median_ms = median(times), samples_ms = as.numeric(times), batch_loops = batch_loops), auto_unbox = TRUE))
"""
        completed = subprocess.run(
            ["Rscript", "--vanilla", "-e", script],
            check=False,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=600,
        )
        if completed.returncode != 0:
            raise RuntimeError((completed.stderr or completed.stdout).strip())
        marker = completed.stdout.strip().splitlines()[-1]
        payload = json.loads(marker)
        output = np.loadtxt(out_path, delimiter=",")
        batch_loops = max(1, int(payload.get("batch_loops", 1)))
        return float(payload["median_ms"]), np.asarray(output, dtype=np.float64), _json_samples(payload), batch_loops


def time_reference(ref: ReferenceSpec, ctx: Dataset, repeat: int) -> tuple[float, object, list[float], int]:
    if ref.r_expr is not None:
        return run_r_reference(ref.r_expr, ctx, repeat)
    if ref.factory is None:
        raise RuntimeError("external reference is missing a factory")
    return time_callable(ref.factory(ctx), repeat)


def reference_applies(ref: ReferenceSpec, ctx: Dataset) -> bool:
    return ref.max_cols is None or ctx.X.shape[1] <= ref.max_cols


def benchmark_spec(
    spec: MethodSpec,
    ctx: Dataset,
    *,
    repeat: int,
    threads: int,
    include_cpp: bool,
    include_python: bool,
    include_r: bool,
    include_external: bool,
    write_reference_snapshots: bool,
) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    lib_build = runtime_version()
    n, p = ctx.X.shape
    native_output: object | None = None
    native_error: str | None = None
    reference_output: object | None = None
    reference_error: str | None = None
    reference_snapshot_output: object | None = None
    reference_snapshot_error: str | None = None
    active_refs = tuple(ref for ref in spec.references if reference_applies(ref, ctx))
    canonical_ref = next((ref for ref in active_refs if ref.compare), None) if include_external else None
    context_ref = next((ref for ref in active_refs if not ref.compare), None) if include_external else None
    diagnostic_ref: ReferenceSpec | None = canonical_ref or context_ref
    diagnostic_reference_output: object | None = None
    precomputed_refs: dict[ReferenceSpec, tuple[float, object, list[float], int]] = {}
    binding_verified = spec.method_id in PARITY_VERIFIED

    if canonical_ref is None and context_ref is not None:
        try:
            context_result = time_reference(context_ref, ctx, repeat)
            precomputed_refs[context_ref] = context_result
            _median_ms, diagnostic_reference_output, _samples, _batch_loops = context_result
        except Exception as exc:  # noqa: BLE001
            reference_error = str(exc)
            diagnostic_reference_output = None

    if canonical_ref is not None:
        try:
            median_ms, reference_output, reference_samples, reference_batch_loops = time_reference(canonical_ref, ctx, repeat)
            snapshot_divergence = None
            try:
                snapshot = get_reference_snapshot(
                    spec,
                    canonical_ref,
                    ctx,
                    reference_output,
                    write_snapshots=write_reference_snapshots,
                )
                reference_snapshot_output = snapshot.output
                diagnostic_reference_output = reference_snapshot_output
                snapshot_ok, snapshot_reason = compare_reference_outputs(
                    spec, canonical_ref, reference_output, snapshot.output
                )
                snapshot_divergence = output_reference_divergence(
                    spec, canonical_ref, reference_output, snapshot.output
                )
                action = "updated" if write_reference_snapshots else "loaded"
                if snapshot_ok:
                    parity = "reference_snapshot"
                    reason = (
                        f"canonical external reference method; {action} stored snapshot "
                        f"{_rel(snapshot.path)} ({snapshot.metadata.get('reference_version', 'version unknown')}): "
                        f"{snapshot_reason}"
                    )
                else:
                    parity = "reference_snapshot_drift"
                    reason = (
                        "canonical external reference method; current output differs from stored snapshot "
                        f"{_rel(snapshot.path)} ({snapshot.metadata.get('reference_version', 'version unknown')}): "
                        f"{snapshot_reason}"
                    )
                    reference_snapshot_error = reason
            except Exception as snapshot_exc:  # noqa: BLE001
                parity = "fail"
                reference_snapshot_error = str(snapshot_exc)
                reason = (
                    "canonical external reference method; reference snapshot unavailable: "
                    f"{reference_snapshot_error}"
                )
            rows.append(csv_row(
                spec,
                backend=canonical_ref.backend,
                language=canonical_ref.language,
                tier="external_reference",
                kind="external_reference",
                n=n,
                p=p,
                threads=threads,
                lib_build=external_version(canonical_ref.backend),
                ok=True,
                median_ms=median_ms,
                parity=parity,
                reason=reason,
                reference_library=canonical_ref.library,
                reference_role="canonical",
                binding_parity_ok=None,
                reference_parity_ok=None,
                samples_ms=reference_samples,
                repeat=repeat,
                batch_loops=reference_batch_loops,
                reference_divergence=snapshot_divergence if reference_snapshot_output is not None else None,
            ))
        except ModuleNotFoundError as exc:
            reference_error = f"missing dependency: {exc.name}"
            rows.append(csv_row(
                spec,
                backend=canonical_ref.backend,
                language=canonical_ref.language,
                tier="external_reference",
                kind="external_reference",
                n=n,
                p=p,
                threads=threads,
                lib_build=external_version(canonical_ref.backend),
                ok=False,
                median_ms=None,
                parity="not_available",
                reason=reference_error,
                reference_library=canonical_ref.library,
                reference_role="canonical",
                reference_parity_ok=False,
            ))
        except Exception as exc:  # noqa: BLE001
            reference_error = str(exc)
            rows.append(csv_row(
                spec,
                backend=canonical_ref.backend,
                language=canonical_ref.language,
                tier="external_reference",
                kind="external_reference",
                n=n,
                p=p,
                threads=threads,
                lib_build=external_version(canonical_ref.backend),
                ok=False,
                median_ms=None,
                parity="fail",
                reason=reference_error,
                reference_library=canonical_ref.library,
                reference_role="canonical",
                reference_parity_ok=False,
            ))

    if include_cpp and spec.cpp_factory is not None:
        try:
            median_ms, cpp_output, cpp_samples, cpp_batch_loops = time_callable(spec.cpp_factory(ctx), repeat)
            if spec.compare_internal:
                native_output = cpp_output
            reference_ok: bool | None = None
            reference_divergence = None
            can_gate_reference = (
                canonical_ref is not None
                and reference_snapshot_output is not None
                and spec.compare_internal
                and canonical_ref.gate_c4a
            )
            if can_gate_reference:
                reference_ok, reference_reason = compare_reference_outputs(
                    spec, canonical_ref, cpp_output, reference_snapshot_output
                )
                reference_divergence = output_reference_divergence(
                    spec, canonical_ref, cpp_output, reference_snapshot_output
                )
                parity = "reference_ok" if reference_ok else "reference_diverged"
                note = (
                    f"; contract note: {canonical_ref.contract_note}"
                    if (not reference_ok and canonical_ref.contract_note)
                    else ""
                )
                reason = f"timed direct libc4a C ABI; reference gate vs stored {canonical_ref.library} snapshot: {reference_reason}{note}"
            elif (
                canonical_ref is not None
                and reference_snapshot_output is not None
                and not canonical_ref.gate_c4a
            ):
                reference_divergence = output_reference_divergence(
                    spec, canonical_ref, cpp_output, reference_snapshot_output
                )
                note = canonical_ref.contract_note or "external implementation uses a different published contract"
                parity = "context"
                reason = (
                    f"timed direct libc4a C ABI; external snapshot stored from {canonical_ref.library}, "
                    f"but no parity gate is run: {note}"
                )
            elif (
                diagnostic_ref is not None
                and diagnostic_reference_output is not None
                and spec.compare_internal
            ):
                reference_divergence = output_reference_divergence(
                    spec, diagnostic_ref, cpp_output, diagnostic_reference_output
                )
                note = diagnostic_ref.contract_note or "external implementation uses a different published contract"
                parity = "context"
                reason = (
                    f"timed direct libc4a C ABI; diagnostic divergence vs {diagnostic_ref.library} "
                    f"is reported but not parity-gated: {note}"
                )
            elif canonical_ref is not None and reference_snapshot_output is None:
                parity = "fail"
                reason = f"timed direct libc4a C ABI; reference gate not run: {reference_snapshot_error or reference_error or 'reference snapshot unavailable'}"
            else:
                parity = "ok"
                reason = "timed direct libc4a C ABI" if spec.compare_internal else "timed direct libc4a C ABI; internal binding gate disabled"
            rows.append(csv_row(
                spec,
                backend="cpp",
                language="C++",
                tier="c_abi",
                kind="chemometrics4all",
                n=n,
                p=p,
                threads=threads,
                lib_build=lib_build,
                ok=True,
                median_ms=median_ms,
                parity=parity,
                reason=reason,
                binding_parity_ok=None,
                reference_parity_ok=reference_ok,
                samples_ms=cpp_samples,
                repeat=repeat,
                batch_loops=cpp_batch_loops,
                reference_divergence=reference_divergence,
            ))
        except Exception as exc:  # noqa: BLE001 - benchmark rows must be non-fatal
            native_error = str(exc)
            rows.append(csv_row(
                spec,
                backend="cpp",
                language="C++",
                tier="c_abi",
                kind="chemometrics4all",
                n=n,
                p=p,
                threads=threads,
                lib_build=lib_build,
                ok=False,
                median_ms=None,
                parity="fail",
                reason=str(exc),
                binding_parity_ok=None,
                reference_parity_ok=False if canonical_ref is not None else None,
            ))

    if include_python:
        try:
            median_ms, py_output, py_samples, py_batch_loops = time_callable(python_direct_factory(spec, ctx), repeat)
            reference_divergence = (
                output_reference_divergence(spec, diagnostic_ref, py_output, diagnostic_reference_output)
                if (
                    diagnostic_ref is not None
                    and diagnostic_reference_output is not None
                    and spec.compare_internal
                )
                else None
            )
            if native_output is not None and spec.compare_internal:
                parity_ok, parity_reason = compare_outputs(spec, py_output, native_output)
                binding_divergence = output_divergence(spec, py_output, native_output)
                parity = "binding_ok" if parity_ok else "binding_mismatch"
                binding_ok = parity_ok and binding_verified
                reason = f"timed ABI-close Python API; binding gate vs C++: {parity_reason}"
            else:
                parity = "ok"
                binding_ok = None
                binding_divergence = None
                reason = f"timed ABI-close Python API; binding gate not run: {native_error or 'no C++ native output'}"
            rows.append(csv_row(
                spec,
                backend="python",
                language="Python",
                tier="abi_close",
                kind="chemometrics4all",
                n=n,
                p=p,
                threads=threads,
                lib_build=lib_build,
                ok=True,
                median_ms=median_ms,
                parity=parity,
                reason=reason,
                binding_parity_ok=binding_ok,
                reference_parity_ok=None,
                samples_ms=py_samples,
                repeat=repeat,
                batch_loops=py_batch_loops,
                reference_divergence=reference_divergence,
                binding_divergence=binding_divergence,
            ))
        except Exception as exc:  # noqa: BLE001
            rows.append(csv_row(
                spec,
                backend="python",
                language="Python",
                tier="abi_close",
                kind="chemometrics4all",
                n=n,
                p=p,
                threads=threads,
                lib_build=lib_build,
                ok=False,
                median_ms=None,
                parity="fail",
                reason=str(exc),
                binding_parity_ok=False,
            ))

    if include_python:
        try:
            median_ms, py_output, py_samples, py_batch_loops = time_callable(spec.python_factory(ctx), repeat)
            reference_divergence = (
                output_reference_divergence(spec, diagnostic_ref, py_output, diagnostic_reference_output)
                if (
                    diagnostic_ref is not None
                    and diagnostic_reference_output is not None
                    and spec.compare_internal
                )
                else None
            )
            if native_output is not None and spec.compare_internal:
                parity_ok, parity_reason = compare_outputs(spec, py_output, native_output)
                binding_divergence = output_divergence(spec, py_output, native_output)
                parity = "binding_ok" if parity_ok else "binding_mismatch"
                binding_ok = parity_ok and binding_verified
                reason = f"timed scikit-learn-compatible Python estimator; binding gate vs C++: {parity_reason}"
            else:
                parity = "ok"
                binding_ok = None
                binding_divergence = None
                reason = f"timed scikit-learn-compatible Python estimator; binding gate not run: {native_error or 'no C++ native output'}"
            rows.append(csv_row(
                spec,
                backend="sklearn",
                language="Python",
                tier="sklearn_compatible",
                kind="chemometrics4all",
                n=n,
                p=p,
                threads=threads,
                lib_build=lib_build,
                ok=True,
                median_ms=median_ms,
                parity=parity,
                reason=reason,
                binding_parity_ok=binding_ok,
                reference_parity_ok=None,
                samples_ms=py_samples,
                repeat=repeat,
                batch_loops=py_batch_loops,
                reference_divergence=reference_divergence,
                binding_divergence=binding_divergence,
            ))
        except Exception as exc:  # noqa: BLE001
            rows.append(csv_row(
                spec,
                backend="sklearn",
                language="Python",
                tier="sklearn_compatible",
                kind="chemometrics4all",
                n=n,
                p=p,
                threads=threads,
                lib_build=lib_build,
                ok=False,
                median_ms=None,
                parity="fail",
                reason=str(exc),
                binding_parity_ok=False,
            ))

    if include_r and spec.r_expr:
        try:
            median_ms, reason, r_output, r_samples, r_batch_loops = run_r_timing(spec.r_expr, ctx, repeat)
            reference_divergence = (
                output_reference_divergence(spec, diagnostic_ref, r_output, diagnostic_reference_output)
                if (
                    diagnostic_ref is not None
                    and diagnostic_reference_output is not None
                    and spec.compare_internal
                )
                else None
            )
            reference_note = (
                f"; reference contract note: {diagnostic_ref.contract_note}"
                if (
                    diagnostic_ref is not None
                    and diagnostic_ref.contract_note
                    and reference_divergence
                    and reference_divergence.get("max_abs", 0.0) > 1e-6
                )
                else ""
            )
            rows.append(csv_row(
                spec,
                backend="r",
                language="R",
                tier="public_r",
                kind="chemometrics4all",
                n=n,
                p=p,
                threads=threads,
                lib_build=lib_build,
                ok=True,
                median_ms=median_ms,
                parity="covered",
                reason=f"timed installed R package; {reason}; binding parity covered by the binding gate suite{reference_note}",
                binding_parity_ok=binding_verified,
                reference_parity_ok=None,
                samples_ms=r_samples,
                repeat=repeat,
                batch_loops=r_batch_loops,
                reference_divergence=reference_divergence,
            ))
        except Exception as exc:  # noqa: BLE001
            rows.append(csv_row(
                spec,
                backend="r",
                language="R",
                tier="public_r",
                kind="chemometrics4all",
                n=n,
                p=p,
                threads=threads,
                lib_build=lib_build,
                ok=False,
                median_ms=None,
                parity="fail",
                reason=str(exc),
                binding_parity_ok=False,
            ))

    if include_external:
        for ref in active_refs:
            if ref is canonical_ref:
                continue
            try:
                if ref in precomputed_refs:
                    median_ms, ref_output, ref_samples, ref_batch_loops = precomputed_refs[ref]
                else:
                    median_ms, ref_output, ref_samples, ref_batch_loops = time_reference(ref, ctx, repeat)
                reference_role = "comparator" if ref.compare else "context"
                if not ref.compare:
                    reference_ok = None
                    reference_divergence = (
                        output_reference_divergence(spec, ref, native_output, ref_output)
                        if native_output is not None and spec.compare_internal
                        else None
                    )
                    note = ref.contract_note or "external implementation does not expose the exact c4a contract"
                    if reference_divergence:
                        reference_reason = (
                            "performance context; diagnostic divergence vs direct libc4a C ABI is "
                            f"reported but not parity-gated: {note}"
                        )
                    else:
                        reference_reason = f"performance context; no parity comparison: {note}"
                    parity = "context"
                elif reference_snapshot_output is None:
                    reference_ok = None
                    reference_divergence = None
                    reference_reason = reference_snapshot_error or reference_error or "reference snapshot unavailable"
                    parity = "fail"
                elif canonical_ref is not None:
                    reference_ok, reference_reason = compare_reference_outputs(
                        spec, ref, ref_output, reference_snapshot_output
                    )
                    reference_divergence = output_reference_divergence(
                        spec, ref, ref_output, reference_snapshot_output
                    )
                    parity = "external_ok" if reference_ok else "external_diverged"
                    reference_reason = f"comparator gate vs stored reference snapshot: {reference_reason}"
                else:
                    reference_ok = None
                    reference_divergence = None
                    reference_reason = "reference snapshot unavailable"
                    parity = "fail"
                rows.append(csv_row(
                    spec,
                    backend=ref.backend,
                    language=ref.language,
                    tier="external_reference",
                    kind="external_reference",
                    n=n,
                    p=p,
                    threads=threads,
                    lib_build=external_version(ref.backend),
                    ok=True,
                    median_ms=median_ms,
                    parity=parity,
                    reason=reference_reason,
                    reference_library=ref.library,
                    reference_role=reference_role,
                    binding_parity_ok=None,
                    reference_parity_ok=reference_ok,
                    samples_ms=ref_samples,
                    repeat=repeat,
                    batch_loops=ref_batch_loops,
                    reference_divergence=reference_divergence,
                ))
            except ModuleNotFoundError as exc:
                rows.append(csv_row(
                    spec,
                    backend=ref.backend,
                    language=ref.language,
                    tier="external_reference",
                    kind="external_reference",
                    n=n,
                    p=p,
                    threads=threads,
                    lib_build="",
                    ok=False,
                    median_ms=None,
                    parity="not_available",
                    reason=f"missing dependency: {exc.name}",
                    reference_library=ref.library,
                    reference_role="comparator" if ref.compare else "context",
                    reference_parity_ok=False if ref.compare else None,
                ))
            except Exception as exc:  # noqa: BLE001
                rows.append(csv_row(
                    spec,
                    backend=ref.backend,
                    language=ref.language,
                    tier="external_reference",
                    kind="external_reference",
                    n=n,
                    p=p,
                    threads=threads,
                    lib_build="",
                    ok=False,
                    median_ms=None,
                    parity="fail",
                    reason=str(exc),
                    reference_library=ref.library,
                    reference_role="comparator" if ref.compare else "context",
                    reference_parity_ok=False if ref.compare else None,
                ))

    return rows


def parse_size(value: str) -> tuple[int, int]:
    try:
        n_raw, p_raw = value.lower().split("x", 1)
        n, p = int(n_raw), int(p_raw)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"size must look like NxP, got {value!r}") from exc
    if n <= 0 or p <= 0:
        raise argparse.ArgumentTypeError("size dimensions must be positive")
    return n, p


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default="benchmarks/cross_binding/results/full_matrix.csv")
    parser.add_argument("--repeat", type=int, default=5)
    parser.add_argument(
        "--size-preset",
        choices=("small", "pls4all"),
        default="small",
        help="benchmark size preset; small is the pls4all smoke subset, pls4all is the full NIRS matrix",
    )
    parser.add_argument("--sizes", nargs="+", type=parse_size, help="explicit NxP sizes overriding --size-preset")
    parser.add_argument("--threads", nargs="+", type=int, default=[1])
    parser.add_argument("--methods", nargs="*", help="optional method_id allowlist")
    parser.add_argument("--no-cpp", action="store_true")
    parser.add_argument("--no-python", action="store_true")
    parser.add_argument("--no-r", action="store_true")
    parser.add_argument("--no-external", action="store_true")
    parser.add_argument(
        "--write-reference-snapshots",
        action="store_true",
        help="create or refresh stored outputs for canonical external references before running gates",
    )
    parser.add_argument("--seed", type=int, default=20260519)
    args = parser.parse_args()

    methods = selected_methods(args.methods)
    include_r = not args.no_r and shutil.which("Rscript") is not None
    sizes = args.sizes or (PLS4ALL_SMALL_SIZES if args.size_preset == "small" else PLS4ALL_BENCHMARK_SIZES)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, str]] = []
    for threads in args.threads:
        set_thread_env(threads)
        with threadpool_context(threads):
            for size_idx, (n, p) in enumerate(sizes):
                ctx = synthetic_spectra(n, p, seed=args.seed + 1009 * size_idx + 37)
                for spec in methods:
                    if spec.max_cols is not None and p > spec.max_cols:
                        continue
                    rows.extend(benchmark_spec(
                        spec,
                        clone_dataset(ctx),
                        repeat=max(1, args.repeat),
                        threads=threads,
                        include_cpp=not args.no_cpp,
                        include_python=not args.no_python,
                        include_r=include_r,
                        include_external=not args.no_external,
                        write_reference_snapshots=args.write_reference_snapshots,
                    ))

    with out.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=CSV_FIELDS)
        writer.writeheader()
        writer.writerows(rows)

    ok = sum(1 for row in rows if row["ok"] == "true")
    print(
        f"wrote {out}: {ok}/{len(rows)} timed cells, "
        f"{len(methods)} methods, sizes={sizes}, threads={args.threads}, r={'yes' if include_r else 'no'}"
    )


if __name__ == "__main__":
    main()
