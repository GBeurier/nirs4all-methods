#!/usr/bin/env python3
# SPDX-License-Identifier: CECILL-2.1
"""Cross-binding timing matrix for the docs dashboard.

The runner follows the same contract as the pls4all dashboard input: one CSV
row per algorithm/backend/size/thread count with timing and parity metadata.
Internal rows are produced from three routes when available:

* ``cpp``: direct libc4a C ABI calls through ctypes;
* ``python``: public sklearn-style Python bindings;
* ``r``: installed public R package through Rscript.

External reference rows are intentionally best-effort. They are included only
when the corresponding third-party package is importable and the output can be
compared to chemometrics4all on the generated dataset.
"""

from __future__ import annotations

import argparse
import contextlib
import csv
import ctypes
import datetime as dt
import importlib
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
from chemometrics4all._errors import check  # noqa: E402
from chemometrics4all._ffi import lib  # noqa: E402
from chemometrics4all._matrix import as_f64_2d, empty_like_f64, empty_like_i32, numpy_to_view  # noqa: E402
from chemometrics4all._types import FilterStats, SplitResult  # noqa: E402


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
    language: str = "Python"
    r_expr: str | None = None


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
    elif backend in {"ref.sklearn", "ref.scipy", "ref.numpy", "ref.pybaselines", "ref.pywavelets"}:
        module_name = {
            "ref.sklearn": "sklearn",
            "ref.scipy": "scipy",
            "ref.numpy": "numpy",
            "ref.pybaselines": "pybaselines",
            "ref.pywavelets": "pywt",
        }[backend]
        try:
            module = importlib.import_module(module_name)
            value = f"{module_name} {getattr(module, '__version__', 'unknown')}"
        except ModuleNotFoundError:
            value = module_name
    elif backend == "ref.r.base":
        value = "R base"
    elif backend == "ref.r.stats":
        value = "R stats"
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


def compare_outputs(spec: MethodSpec, left: object, right: object) -> tuple[bool, str]:
    if spec.comparator is not None:
        return spec.comparator(left, right)
    return outputs_close(left, right)


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


def ref_numpy_snv(ctx: Dataset) -> Callable[[], object]:
    return lambda: (ctx.X - ctx.X.mean(axis=1, keepdims=True)) / np.maximum(ctx.X.std(axis=1, ddof=0, keepdims=True), 1e-12)


def ref_numpy_rnv(ctx: Dataset) -> Callable[[], object]:
    def run() -> object:
        med = np.median(ctx.X, axis=1, keepdims=True)
        mad = np.median(np.abs(ctx.X - med), axis=1, keepdims=True) * 1.4826
        return (ctx.X - med) / np.maximum(mad, 1e-12)

    return run


def ref_numpy_area(ctx: Dataset) -> Callable[[], object]:
    return lambda: ctx.X / np.maximum(np.sum(ctx.X, axis=1, keepdims=True), 1e-12)


def ref_numpy_normalize(ctx: Dataset) -> Callable[[], object]:
    def run() -> object:
        norm = np.sqrt(np.sum(ctx.X * ctx.X, axis=0, keepdims=True))
        return ctx.X / np.maximum(norm, 1e-12)

    return run


def ref_numpy_simple_scale(ctx: Dataset) -> Callable[[], object]:
    def run() -> object:
        mn = ctx.X.min(axis=0, keepdims=True)
        mx = ctx.X.max(axis=0, keepdims=True)
        return (ctx.X - mn) / np.maximum(mx - mn, 1e-12)

    return run


def ref_numpy_to_abs(ctx: Dataset) -> Callable[[], object]:
    return lambda: -np.log10(np.maximum(ctx.X, 1e-10))


def ref_numpy_from_abs(ctx: Dataset) -> Callable[[], object]:
    return lambda: np.power(10.0, -ctx.X)


def ref_numpy_pct_to_frac(ctx: Dataset) -> Callable[[], object]:
    return lambda: ctx.X / 100.0


def ref_numpy_frac_to_pct(ctx: Dataset) -> Callable[[], object]:
    return lambda: ctx.X * 100.0


def ref_numpy_km(ctx: Dataset) -> Callable[[], object]:
    return lambda: ((1.0 - np.maximum(ctx.X, 1e-10)) ** 2) / (2.0 * np.maximum(ctx.X, 1e-10))


def ref_numpy_first(ctx: Dataset) -> Callable[[], object]:
    return lambda: np.gradient(ctx.X, 1.0, axis=1, edge_order=2)


def ref_numpy_second(ctx: Dataset) -> Callable[[], object]:
    return lambda: np.gradient(np.gradient(ctx.X, 1.0, axis=1, edge_order=2), 1.0, axis=1, edge_order=2)


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


def r_base(expr: str, *, compare: bool = True) -> ReferenceSpec:
    return ReferenceSpec("ref.r.base", "R base", compare=compare, language="R", r_expr=expr)


def r_stats(expr: str, *, compare: bool = True) -> ReferenceSpec:
    return ReferenceSpec("ref.r.stats", "R stats", compare=compare, language="R", r_expr=expr)


def nirs_ref(library: str, factory: Callable[[Dataset], Callable[[], object]], *, compare: bool = True) -> ReferenceSpec:
    return ReferenceSpec("ref.nirs4all", library, factory, compare=compare, language="Python")


def sklearn_ref(library: str, factory: Callable[[Dataset], Callable[[], object]], *, compare: bool = True) -> ReferenceSpec:
    return ReferenceSpec("ref.sklearn", library, factory, compare=compare, language="Python")


def scipy_ref(library: str, factory: Callable[[Dataset], Callable[[], object]], *, compare: bool = True) -> ReferenceSpec:
    return ReferenceSpec("ref.scipy", library, factory, compare=compare, language="Python")


def pywavelets_ref(library: str, factory: Callable[[Dataset], Callable[[], object]], *, compare: bool = True) -> ReferenceSpec:
    return ReferenceSpec("ref.pywavelets", library, factory, compare=compare, language="Python")


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


R_REF_SNV = '{m <- rowMeans(X); xc <- sweep(X, 1L, m, "-"); s <- sqrt(rowMeans(xc * xc)); sweep(xc, 1L, ifelse(s == 0, 1, s), "/")}'
R_REF_RNV = '{med <- apply(X, 1L, median); xc <- sweep(X, 1L, med, "-"); mad <- apply(abs(xc), 1L, median) * 1.4826; sweep(xc, 1L, ifelse(mad == 0, 1, mad), "/")}'
R_REF_AREA_SUM = '{area <- rowSums(X); area <- ifelse(abs(area) < 1e-10, 1, area); sweep(X, 1L, area, "/")}'
R_REF_NORMALIZE_L2 = '{norms <- sqrt(colSums(X * X)); sweep(X, 2L, norms, "/")}'
R_REF_SIMPLE_SCALE = '{mn <- apply(X, 2L, min); mx <- apply(X, 2L, max); sweep(sweep(X, 2L, mn, "-"), 2L, mx - mn, "/")}'
R_REF_TO_ABS = '-log10(pmax(X, 1e-10))'
R_REF_FROM_ABS = '10^(-X)'
R_REF_PCT_TO_FRAC = 'X / 100'
R_REF_FRAC_TO_PCT = 'X * 100'
R_REF_KM = '{x <- pmax(X, 1e-10); ((1 - x)^2) / (2 * x)}'
R_REF_DETREND = '{x <- seq_len(ncol(X)); design <- cbind(1, x); t(apply(X, 1L, function(row) stats::lm.fit(design, row)$residuals))}'


METHODS: tuple[MethodSpec, ...] = (
    MethodSpec(
        "snv",
        "SNV",
        "preprocessing",
        py_transform(c4a.SNV),
        c_same("c4a_pp_snv", ctypes.c_int(1), ctypes.c_int(1), ctypes.c_int(0)),
        "snv(X)",
        (ReferenceSpec("ref.numpy", "numpy", ref_numpy_snv), r_base(R_REF_SNV)),
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
        (ReferenceSpec("ref.numpy", "numpy", ref_numpy_rnv), r_base(R_REF_RNV)),
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
            ReferenceSpec("ref.numpy", "numpy", ref_numpy_area),
            r_base(R_REF_AREA_SUM),
        ),
    ),
    MethodSpec(
        "normalize",
        "Normalize",
        "preprocessing",
        py_transform(c4a.Normalize, feature_min=-1.0, feature_max=1.0),
        c_same("c4a_pp_normalize", ctypes.c_double(-1.0), ctypes.c_double(1.0)),
        "normalize(X, feature_min = -1, feature_max = 1)",
        (ReferenceSpec("ref.numpy", "numpy", ref_numpy_normalize), r_base(R_REF_NORMALIZE_L2)),
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
            ReferenceSpec("ref.numpy", "numpy", ref_numpy_simple_scale),
            r_base(R_REF_SIMPLE_SCALE),
        ),
    ),
    MethodSpec(
        "msc",
        "MSC",
        "preprocessing",
        py_transform(c4a.MSC),
        c_stateful("c4a_pp_msc"),
        "msc(X)",
        (nirs_ref("nirs4all.MultiplicativeScatterCorrection(scale=False)", ref_nirs_transform("nirs4all.operators.transforms.nirs", "MultiplicativeScatterCorrection", scale=False)),),
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
            nirs_ref("nirs4all.SavitzkyGolay", ref_nirs_transform("nirs4all.operators.transforms.nirs", "SavitzkyGolay", window_length=11, polyorder=2, deriv=0, delta=1.0), compare=False),
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
            ReferenceSpec("ref.numpy", "numpy.gradient", ref_numpy_first),
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
            ReferenceSpec("ref.numpy", "numpy.gradient", ref_numpy_second),
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
            nirs_ref("nirs4all.Gaussian", ref_nirs_transform("nirs4all.operators.transforms.signal", "Gaussian", order=0, sigma=1.0)),
            scipy_ref("scipy.ndimage.gaussian_filter1d", ref_scipy_gaussian),
        ),
    ),
    MethodSpec("to_absorbance", "To absorbance", "preprocessing", py_transform(c4a.ToAbsorbance), c_same("c4a_pp_to_absorbance", ctypes.c_int(0), ctypes.c_double(1e-10), ctypes.c_int(1)), "to_absorbance(X)", (ReferenceSpec("ref.numpy", "numpy", ref_numpy_to_abs), r_base(R_REF_TO_ABS))),
    MethodSpec("from_absorbance", "From absorbance", "preprocessing", py_transform(c4a.FromAbsorbance), c_same("c4a_pp_from_absorbance", ctypes.c_int(0)), "from_absorbance(X)", (ReferenceSpec("ref.numpy", "numpy", ref_numpy_from_abs), r_base(R_REF_FROM_ABS))),
    MethodSpec("pct_to_frac", "Percent to fraction", "preprocessing", py_transform(c4a.PercentToFraction), c_same("c4a_pp_pct_to_frac"), "percent_to_fraction(X)", (ReferenceSpec("ref.numpy", "numpy", ref_numpy_pct_to_frac), r_base(R_REF_PCT_TO_FRAC))),
    MethodSpec("frac_to_pct", "Fraction to percent", "preprocessing", py_transform(c4a.FractionToPercent), c_same("c4a_pp_frac_to_pct"), "fraction_to_percent(X)", (ReferenceSpec("ref.numpy", "numpy", ref_numpy_frac_to_pct), r_base(R_REF_FRAC_TO_PCT))),
    MethodSpec("kubelka_munk", "Kubelka-Munk", "preprocessing", py_transform(c4a.KubelkaMunk), c_same("c4a_pp_kubelka_munk", ctypes.c_int(0), ctypes.c_double(1e-10)), "kubelka_munk(X)", (ReferenceSpec("ref.numpy", "numpy", ref_numpy_km), r_base(R_REF_KM))),
    MethodSpec("detrend", "Detrend", "baseline", py_transform(c4a.Detrend, polyorder=1), c_same("c4a_pp_detrend", ctypes.c_int32(1)), "detrend(X, polyorder = 1L)", (nirs_ref("nirs4all.Detrend", ref_nirs_transform("nirs4all.operators.transforms.signal", "Detrend")), scipy_ref("scipy.signal.detrend", ref_scipy_detrend), r_stats(R_REF_DETREND))),
    MethodSpec("asls", "AsLS", "baseline", py_transform(c4a.AsLS, lam=1e5, p=0.01, max_iter=20), c_same("c4a_pp_asls", ctypes.c_double(1e5), ctypes.c_double(0.01), ctypes.c_int32(20), ctypes.c_double(1e-3)), "asls(X, lam = 1e5, p = 0.01, max_iter = 20L)", (ReferenceSpec("ref.pybaselines", "pybaselines.asls", ref_pybaselines("asls", lam=1e5, p=0.01, max_iter=20)),)),
    MethodSpec("airpls", "AirPLS", "baseline", py_transform(c4a.AirPLS, lam=1e5, max_iter=20), c_same("c4a_pp_airpls", ctypes.c_double(1e5), ctypes.c_int32(20), ctypes.c_double(1e-3)), "airpls(X, lam = 1e5, max_iter = 20L)", (ReferenceSpec("ref.pybaselines", "pybaselines.airpls", ref_pybaselines("airpls", lam=1e5, max_iter=20)),)),
    MethodSpec("arpls", "ArPLS", "baseline", py_transform(c4a.ArPLS, lam=1e5, max_iter=20), c_same("c4a_pp_arpls", ctypes.c_double(1e5), ctypes.c_int32(20), ctypes.c_double(1e-3)), "arpls(X, lam = 1e5, max_iter = 20L)", (ReferenceSpec("ref.pybaselines", "pybaselines.arpls", ref_pybaselines("arpls", lam=1e5, max_iter=20)),)),
    MethodSpec("modpoly", "ModPoly", "baseline", py_transform(c4a.ModPoly, polyorder=2, max_iter=50), c_same("c4a_pp_modpoly", ctypes.c_int32(2), ctypes.c_int32(50), ctypes.c_double(1e-3)), "modpoly(X, polyorder = 2L, max_iter = 50L)", (ReferenceSpec("ref.pybaselines", "pybaselines.modpoly", ref_pybaselines("modpoly", poly_order=2, max_iter=50)),)),
    MethodSpec("imodpoly", "IModPoly", "baseline", py_transform(c4a.IModPoly, polyorder=2, max_iter=50), c_same("c4a_pp_imodpoly", ctypes.c_int32(2), ctypes.c_int32(50), ctypes.c_double(1e-3)), "imodpoly(X, polyorder = 2L, max_iter = 50L)", (ReferenceSpec("ref.pybaselines", "pybaselines.imodpoly(mask_initial_peaks=False)", ref_pybaselines("imodpoly", poly_order=2, max_iter=50, mask_initial_peaks=False), compare=False),)),
    MethodSpec("snip", "SNIP", "baseline", py_transform(c4a.SNIP, max_half_window=10), c_same("c4a_pp_snip", ctypes.c_int32(10)), "snip(X, max_half_window = 10L)", (ReferenceSpec("ref.pybaselines", "pybaselines.snip", ref_pybaselines("snip", max_half_window=10), compare=False),)),
    MethodSpec("rolling_ball", "Rolling ball", "baseline", py_transform(c4a.RollingBall, half_window=10, smooth_half_window=0), c_same("c4a_pp_rolling_ball", ctypes.c_int32(10), ctypes.c_int32(0)), "rolling_ball(X, half_window = 10L, smooth_half_window = 0L)", (ReferenceSpec("ref.pybaselines", "pybaselines.rolling_ball", ref_pybaselines("rolling_ball", half_window=10, smooth_half_window=0)),)),
    MethodSpec("iasls", "IAsLS", "baseline", py_transform(c4a.IAsLS, lam=1e5, p=0.01, lam_1=1e-4, polyorder=2, diff_order=2, max_iter=20), lambda ctx: lambda: c_iasls(ctx), "iasls(X, lam = 1e5, p = 0.01, lam_1 = 1e-4, polyorder = 2L, diff_order = 2L, max_iter = 20L)", (ReferenceSpec("ref.pybaselines", "pybaselines.iasls", ref_pybaselines("iasls", lam=1e5, p=0.01, lam_1=1e-4, diff_order=2, max_iter=20)),), comparator=outputs_close_iasls),
    MethodSpec("beads", "BEADS", "baseline", py_transform(c4a.BEADS, lam_0=100.0, lam_1=0.5, lam_2=0.5, max_iter=20), c_same("c4a_pp_beads", ctypes.c_double(100.0), ctypes.c_double(0.5), ctypes.c_double(0.5), ctypes.c_int32(20), ctypes.c_double(1e-3)), "beads(X, max_iter = 20L)", (ReferenceSpec("ref.pybaselines", "pybaselines.beads(full)", ref_pybaselines("beads", lam_0=100.0, lam_1=0.5, lam_2=0.5, max_iter=20), compare=False),)),
    MethodSpec(
        "wavelet",
        "Wavelet",
        "wavelet",
        py_transform(c4a.Wavelet, family="haar", mode="periodization"),
        c_same("c4a_pp_wavelet", ctypes.c_int(0), ctypes.c_int(0)),
        "wavelet(X, family = 'haar', boundary = 'periodization')",
        (
            pywavelets_ref("PyWavelets.dwt(haar, periodization)", ref_pywavelets_wavelet),
            nirs_ref("nirs4all.Wavelet", ref_nirs_transform("nirs4all.operators.transforms.nirs", "Wavelet", wavelet="haar", mode="periodization"), compare=False),
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
            nirs_ref("nirs4all.Haar", ref_nirs_transform("nirs4all.operators.transforms.nirs", "Haar"), compare=False),
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
    MethodSpec("flexible_pca", "Flexible PCA", "feature_extraction", py_transform(c4a.FlexiblePCA, n_components=5.0), lambda ctx: lambda: c_flexible("c4a_pp_flex_pca", ctx, 5.0), "flexible_pca(X, n_components = 5.0)", (nirs_ref("nirs4all.FlexiblePCA", ref_nirs_transform("nirs4all.operators.transforms.feature_selection", "FlexiblePCA", n_components=5, svd_solver="full")), sklearn_ref("sklearn.decomposition.PCA", ref_sklearn_flexible_pca)), comparator=outputs_close_sign_invariant_columns),
    MethodSpec("flexible_svd", "Flexible SVD", "feature_extraction", py_transform(c4a.FlexibleSVD, n_components=5.0), lambda ctx: lambda: c_flexible("c4a_pp_flex_svd", ctx, 5.0), "flexible_svd(X, n_components = 5.0)", (nirs_ref("nirs4all.FlexibleSVD", ref_nirs_transform("nirs4all.operators.transforms.feature_selection", "FlexibleSVD", n_components=5, algorithm="arpack", random_state=0)), sklearn_ref("sklearn.decomposition.TruncatedSVD", ref_sklearn_flexible_svd)), comparator=outputs_close_sign_invariant_columns),
    MethodSpec("fck_static", "FCK static", "feature_extraction", py_fck, lambda ctx: lambda: c_fck_static(ctx), "fck_static(X, kernel_size = 5L, filter_orders = c(0.5, 1.0), filter_scales = c(1.0, 2.0))", (nirs_ref("nirs4all.FCKStaticTransformer", ref_nirs_transform("nirs4all.operators.transforms.fck_static", "FCKStaticTransformer", alphas=[0.5, 1.0], scales=[1.0, 2.0], kernel_sizes=[5], sigma=3.0), compare=False),)),
    MethodSpec("crop", "Crop", "resampling", py_crop, c_crop_factory, "{start <- ncol(X) %/% 8L; end <- ncol(X) - ncol(X) %/% 8L; crop_transform(X, start = start, end = end)}", (nirs_ref("nirs4all.CropTransformer", ref_nirs_crop),)),
    MethodSpec("resample", "Resample", "resampling", py_resample, lambda ctx: lambda: c_resample(ctx), "resample_transform(X, num_samples = max(4L, ncol(X) %/% 2L))", (nirs_ref("nirs4all.ResampleTransformer", lambda ctx: lambda: _nirs_class("nirs4all.operators.transforms.features", "ResampleTransformer")(num_samples=max(4, ctx.X.shape[1] // 2)).fit_transform(np.array(ctx.X, copy=True))), scipy_ref("scipy.interpolate.interp1d", ref_scipy_resample))),
    MethodSpec("resampler", "Wavelength resampler", "resampling", py_resampler, lambda ctx: lambda: c_resampler(ctx), "{source <- seq(900, 1700, length.out = ncol(X)); target <- seq(source[[1L]], source[[length(source)]], length.out = max(4L, ncol(X) %/% 2L)); resampler(X, source_wavelengths = source, target_wavelengths = target)}", (nirs_ref("nirs4all.Resampler", ref_nirs_resampler),)),
    MethodSpec("range_discretizer", "Range discretizer", "resampling", py_range_disc, lambda ctx: lambda: c_range_disc(ctx), "range_discretizer(X, edges = c(0.25, 0.40, 0.55, 0.70))", (nirs_ref("nirs4all.RangeDiscretizer", ref_nirs_transform("nirs4all.operators.transforms.targets", "RangeDiscretizer", bins=[0.25, 0.40, 0.55, 0.70])),)),
    MethodSpec("kbins_discretizer", "K-bins discretizer", "resampling", py_kbins, lambda ctx: lambda: c_kbins(ctx), "kbins_discretizer(X, n_bins = 5L, strategy = 'uniform')", (nirs_ref("nirs4all.IntegerKBinsDiscretizer", ref_nirs_transform("nirs4all.operators.transforms.targets", "IntegerKBinsDiscretizer", n_bins=5, strategy="uniform")), sklearn_ref("sklearn.preprocessing.KBinsDiscretizer", ref_sklearn_kbins))),
    MethodSpec("kennard_stone", "Kennard-Stone", "splitter", py_ks, lambda ctx: lambda: c_split_kennard_stone(ctx), "kennard_stone(X)", (nirs_ref("nirs4all.KennardStoneSplitter", ref_nirs_split_ks),)),
    MethodSpec("spxy", "SPXY", "splitter", py_spxy, lambda ctx: lambda: c_split_spxy(ctx), "spxy(X, matrix(y, ncol = 1))", (nirs_ref("nirs4all.SPXYSplitter", ref_nirs_split_spxy),)),
    MethodSpec("kbins_stratified", "K-bins stratified", "splitter", py_kbins_stratified, lambda ctx: lambda: c_split_kbins_stratified(ctx), "kbins_stratified(matrix(y, ncol = 1L), test_size = 0.25, seed = 17, n_bins = 5L, strategy = 'uniform')", (nirs_ref("nirs4all.KBinsStratifiedSplitter", ref_nirs_split_kbins_stratified, compare=False), sklearn_ref("sklearn.model_selection.StratifiedShuffleSplit", ref_sklearn_kbins_stratified, compare=False))),
    MethodSpec("y_outlier_iqr", "Y outlier IQR", "filter", py_y_outlier, lambda ctx: lambda: c_y_outlier_iqr(ctx), "y_outlier_filter(y, method = 'iqr')$mask", (nirs_ref("nirs4all.YOutlierFilter", ref_nirs_y_outlier),)),
    MethodSpec("x_outlier_mahalanobis", "X outlier Mahalanobis", "filter", py_x_outlier, lambda ctx: lambda: c_x_outlier_mahalanobis(ctx), "x_outlier_filter(X, method = 'mahalanobis', n_components = min(5L, ncol(X)))$mask", (sklearn_ref("sklearn.PCA(full)+EmpiricalCovariance+chi2", ref_sklearn_x_outlier_mahalanobis), nirs_ref("nirs4all.XOutlierFilter", ref_nirs_x_outlier, compare=False))),
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
) -> dict[str, str]:
    samples = list(samples_ms or [])
    timed_runs = repeat if repeat is not None else (len(samples) if samples else None)
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
    }


def _json_samples(payload: dict) -> list[float]:
    raw = payload.get("samples_ms", [])
    if isinstance(raw, (int, float)):
        raw = [raw]
    return [float(value) for value in raw]


def run_r_timing(expr: str, ctx: Dataset, repeat: int) -> tuple[float, str, list[float], int]:
    with tempfile.TemporaryDirectory(prefix="c4a-r-bench-") as tmp:
        tmp_path = Path(tmp)
        x_path = tmp_path / "X.csv"
        y_path = tmp_path / "y.txt"
        d_path = tmp_path / "d.txt"
        np.savetxt(x_path, ctx.X, delimiter=",")
        np.savetxt(y_path, ctx.y)
        np.savetxt(d_path, ctx.d)
        script = f"""
suppressPackageStartupMessages(library(chemometrics4all))
if (!requireNamespace("jsonlite", quietly = TRUE)) stop("jsonlite is required")
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
cat(jsonlite::toJSON(list(ok = TRUE, median_ms = median(times), samples_ms = as.numeric(times), batch_loops = batch_loops), auto_unbox = TRUE))
"""
        completed = subprocess.run(
            ["Rscript", "--vanilla", "-e", script],
            check=False,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=180,
        )
        if completed.returncode != 0:
            raise RuntimeError((completed.stderr or completed.stdout).strip())
        marker = completed.stdout.strip().splitlines()[-1]
        payload = json.loads(marker)
        batch_loops = max(1, int(payload.get("batch_loops", 1)))
        return float(payload["median_ms"]), f"ok; batch_loops={batch_loops}", _json_samples(payload), batch_loops


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
write.table(as.matrix(out), file = "{out_path.as_posix()}", sep = ",", row.names = FALSE, col.names = FALSE)
cat(jsonlite::toJSON(list(ok = TRUE, median_ms = median(times), samples_ms = as.numeric(times), batch_loops = batch_loops), auto_unbox = TRUE))
"""
        completed = subprocess.run(
            ["Rscript", "--vanilla", "-e", script],
            check=False,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=180,
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
    canonical_ref = spec.references[0] if include_external and spec.references else None
    binding_verified = spec.method_id in PARITY_VERIFIED

    if canonical_ref is not None:
        try:
            median_ms, reference_output, reference_samples, reference_batch_loops = time_reference(canonical_ref, ctx, repeat)
            if canonical_ref.compare:
                try:
                    snapshot = get_reference_snapshot(
                        spec,
                        canonical_ref,
                        ctx,
                        reference_output,
                        write_snapshots=write_reference_snapshots,
                    )
                    reference_snapshot_output = snapshot.output
                    snapshot_ok, snapshot_reason = compare_outputs(spec, reference_output, snapshot.output)
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
                            f"canonical external reference method; current output differs from stored snapshot "
                            f"{_rel(snapshot.path)} ({snapshot.metadata.get('reference_version', 'version unknown')}): "
                            f"{snapshot_reason}"
                        )
                        reference_snapshot_error = reason
                except Exception as snapshot_exc:  # noqa: BLE001
                    parity = "reference_snapshot_missing"
                    reference_snapshot_error = str(snapshot_exc)
                    reason = f"canonical external reference method; reference snapshot unavailable: {reference_snapshot_error}"
            else:
                parity = "timing_only"
                reason = (
                    "canonical external reference method; reference gate disabled because "
                    "the external contract is not equivalent for exact parity"
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
            if canonical_ref is not None and reference_snapshot_output is not None and canonical_ref.compare and spec.compare_internal:
                reference_ok, reference_reason = compare_outputs(spec, cpp_output, reference_snapshot_output)
                parity = "reference_ok" if reference_ok else "reference_diverged"
                reason = f"timed direct libc4a C ABI; reference gate vs stored {canonical_ref.library} snapshot: {reference_reason}"
            elif canonical_ref is not None and reference_output is not None and not canonical_ref.compare:
                parity = "reference_not_compared"
                reason = f"timed direct libc4a C ABI; reference gate not run: {canonical_ref.library} is timing-only"
            elif canonical_ref is not None and canonical_ref.compare and reference_snapshot_output is None:
                parity = "reference_not_compared"
                reason = f"timed direct libc4a C ABI; reference gate not run: {reference_snapshot_error or reference_error or 'reference snapshot unavailable'}"
            else:
                parity = "ok"
                reason = "timed direct libc4a C ABI" if spec.compare_internal else "timed direct libc4a C ABI; timing-only ABI contract"
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
            median_ms, py_output, py_samples, py_batch_loops = time_callable(spec.python_factory(ctx), repeat)
            if native_output is not None and spec.compare_internal:
                parity_ok, parity_reason = compare_outputs(spec, py_output, native_output)
                parity = "binding_ok" if parity_ok else "binding_mismatch"
                binding_ok = parity_ok and binding_verified
                reason = f"timed public Python binding; binding gate vs C++: {parity_reason}"
            else:
                parity = "ok"
                binding_ok = None
                reason = f"timed public Python binding; binding gate not run: {native_error or 'no C++ native output'}"
            rows.append(csv_row(
                spec,
                backend="python",
                language="Python",
                tier="sklearn_style",
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
            ))
        except Exception as exc:  # noqa: BLE001
            rows.append(csv_row(
                spec,
                backend="python",
                language="Python",
                tier="sklearn_style",
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
            median_ms, reason, r_samples, r_batch_loops = run_r_timing(spec.r_expr, ctx, repeat)
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
                reason=f"timed installed R package; {reason}; binding parity covered by the binding gate suite",
                binding_parity_ok=binding_verified,
                reference_parity_ok=None,
                samples_ms=r_samples,
                repeat=repeat,
                batch_loops=r_batch_loops,
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
        for idx, ref in enumerate(spec.references):
            if idx == 0:
                continue
            try:
                median_ms, ref_output, ref_samples, ref_batch_loops = time_reference(ref, ctx, repeat)
                if not ref.compare:
                    reference_ok, reference_reason = None, "timing-only external reference; exact parity contract not equivalent"
                    parity = "timing_only"
                elif reference_snapshot_output is None:
                    reference_ok = None
                    reference_reason = reference_snapshot_error or reference_error or "reference snapshot unavailable"
                    parity = "external_not_compared"
                elif canonical_ref is None or canonical_ref.compare:
                    reference_ok, reference_reason = compare_outputs(spec, ref_output, reference_snapshot_output)
                    parity = "external_ok" if reference_ok else "external_diverged"
                    reference_reason = f"comparator gate vs stored reference snapshot: {reference_reason}"
                else:
                    reference_ok, reference_reason = None, "timing-only external reference; canonical reference is not exact-comparable"
                    parity = "timing_only"
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
                    reference_role="comparator",
                    binding_parity_ok=None,
                    reference_parity_ok=reference_ok,
                    samples_ms=ref_samples,
                    repeat=repeat,
                    batch_loops=ref_batch_loops,
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
                    reference_role="comparator",
                    reference_parity_ok=False,
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
                    reference_role="comparator",
                    reference_parity_ok=False,
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
