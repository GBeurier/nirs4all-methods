#!/usr/bin/env python3
"""Cross-binding × threads × sizes × algorithms × libp4a-build
timing + parity orchestrator.

For each (algo, backend, n, p, threads, libp4a_build) cell:
  - Generate deterministic dataset (per run, seed = seed_base + run_index)
  - Spawn the backend's bench script with threading env vars set
  - Collect timing (median/min/max), predictions path (.npy), versions
  - Post-hoc: compute max_abs_diff vs the reference cell's predictions

Result CSV: rows = cells, columns include parity verdict + speedup vs
the per-language external. Result Markdown: pivot table per (algo, threads)
with a parity icon next to each timing.

Usage:
  # smoke (PLS at 500×200, 1 thread)
  python orchestrator.py --algorithms pls --sizes 500x200 --threads 1

  # complete canonical method/reference matrix
  python orchestrator.py --algorithms all --registry-cells \
    --threads 1 3 10 --reference-backends all
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Iterable

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))
DATA_DIR = HERE / "data"
PREDS_DIR = HERE / "data" / ".predictions"
ORACLES_DIR = HERE / "data" / ".reference_oracles"
X_TARGET_DIR = HERE / "data" / ".x_target"
RESULTS_DIR = HERE / "results"
SCRIPTS_DIR = HERE / "scripts"
for d in (DATA_DIR, PREDS_DIR, ORACLES_DIR, X_TARGET_DIR,
          RESULTS_DIR, SCRIPTS_DIR):
    d.mkdir(parents=True, exist_ok=True)

from benchmarks.parity_timing.registry import (  # noqa: E402
    canonical_reference_for_method,
    get_method,
    method_names,
    resolved_references_for_method,
)

# Default base seed. uint32-safe so it round-trips losslessly through
# R/Octave doubles and accepts as sklearn random_state. +1 per timed run
# inside each cell, up to the adaptive protocol's max-run cap.
DEFAULT_SEED_BASE = 1_234_567_890
DEFAULT_MAX_RUNS = 40

# 11 sizes (10000×2500 skipped per user decision).
DEFAULT_SIZES = [
    (100, 50), (100, 500), (100, 2500),
    (500, 50), (500, 500), (500, 2500),
    (2500, 50), (2500, 500), (2500, 2500),
    (10000, 50), (10000, 500),
]

# Per-cell wall-clock guard (seconds). Slow-cell runtime is controlled by
# adaptive early-stop rules, not by a short timeout.
DEFAULT_TIMEOUT_S = 24 * 60 * 60
TIMING_SCHEMA = "adaptive-v1"

# (name, script, language, tier, kind)
#   kind ∈ {"pls4all_core", "pls4all_binding", "external"}
#   For pls4all_core / pls4all_binding entries we sweep both libp4a builds
#   when --libp4a-build=both; externals only get one row.
BACKENDS = [
    ("registry_pls4all", "bench_registry_pls4all.py", "Python", "canonical", "pls4all_binding"),
    ("cpp",          "bench_cpp.py",          "C++",    "direct",   "pls4all_core"),
    ("python_tier1", "bench_python_tier1.py", "Python", "tier 1",   "pls4all_binding"),
    ("python_tier2", "bench_python_tier2.py", "Python", "tier 2",   "pls4all_binding"),
    ("sklearn",      "bench_sklearn.py",      "Python", "external", "external"),
    ("ikpls",        "bench_python_ikpls.py", "Python", "external", "external"),
    ("r_tier1",      "bench_r_tier1.R",       "R",      "tier 1",   "pls4all_binding"),
    ("r_tier2",      "bench_r_tier2.R",       "R",      "tier 2",   "pls4all_binding"),
    ("r_pls_compat", "bench_r_pls_compat.R",  "R",      "pls compat", "pls4all_binding"),
    ("r_mdatools_compat", "bench_r_mdatools_compat.R", "R", "mdatools compat", "pls4all_binding"),
    ("r_pls",        "bench_r_pls.R",         "R",      "external", "external"),
    ("r_ropls",      "bench_r_ropls.R",       "R",      "external", "external"),
    ("r_mixomics",   "bench_r_mixomics.R",    "R",      "external", "external"),
    ("matlab_tier1", "bench_matlab_tier1.m",  "MATLAB", "tier 1",   "pls4all_binding"),
    ("matlab_tier2", "bench_matlab_tier2.m",  "MATLAB", "tier 2",   "pls4all_binding"),
    ("matlab_pls",   "bench_matlab_pls.m",    "MATLAB", "external", "external"),
]

REFERENCE_SCRIPT_OVERRIDES = {
    # These reference libraries already have fixed external benchmark
    # drivers that run warmup + timed samples inside one R process. The
    # generic registry reference driver would fork Rscript for every
    # prediction when rpy2 is unavailable, turning the reference columns
    # into R startup timings rather than library timings.
    "r_pls": "bench_r_pls.R",
    "r_ropls": "bench_r_ropls.R",
    "r_mixomics": "bench_r_mixomics.R",
}


def registry_reference_backends_for(algo: str) -> list[tuple[str, str, str, str, str]]:
    """External reference backends declared by the canonical MethodSpec."""
    try:
        method = get_method(algo)
    except KeyError:
        return []
    out = []
    for ref in resolved_references_for_method(method):
        lang = ref["language"]
        display_lang = "Python" if lang == "python" else lang
        script = REFERENCE_SCRIPT_OVERRIDES.get(
            ref["id"], "bench_registry_reference.py")
        if ref["id"] == "r_pls" and algo not in {"pls", "pcr", "cppls"}:
            script = "bench_registry_reference.py"
        out.append((f"ref_{ref['id']}", script,
                    display_lang, ref["library"], "external"))
    return out

# libp4a build → libpath. Used for the cpp variant sweep: each build
# corresponds to a distinct OpenMP / BLAS / CUDA tier of the C kernel.
LIBP4A_BUILDS = {
    "dev-release": str(REPO / "build/dev-release/cpp/src"),  # native scalar, no BLAS, no OMP
    "blas-on":     str(REPO / "build/blas-on/cpp/src"),       # BLAS only
    "omp-on":      str(REPO / "build/omp-on/cpp/src"),        # OMP only
    "blas-omp":    str(REPO / "build/blas-omp/cpp/src"),      # BLAS + OMP
    "cuda-on":     str(REPO / "build/cuda-on/cpp/src"),       # BLAS + CUDA
}

# Conda env paths (R + Octave live here).
PLS4ALL_R_ENV = "/home/delete/miniconda3/envs/pls4all_r"


def gen_dataset_csv(n: int, p: int, seed: int) -> Path:
    """Generate and cache deterministic dataset CSV under data/.

    R/MATLAB scripts read the CSV; Python scripts regenerate inline
    via _common.make_dataset (same generator). Cached by (n, p, seed)."""
    csv_path = DATA_DIR / f"data_{n}x{p}_seed{seed}.csv"
    if csv_path.exists():
        return csv_path
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n, p)).astype(np.float64)
    y = (2.0 * X[:, min(2, p - 1)]
          - X[:, min(5, p - 1)]
          + 0.5 * X[:, min(8, p - 1)]
          + 0.05 * rng.standard_normal(n)).astype(np.float64)
    cols = [f"x{i}" for i in range(p)] + ["y"]
    arr = np.hstack([X, y.reshape(-1, 1)])
    # %.17g preserves IEEE-double round-trip.
    np.savetxt(csv_path, arr, delimiter=",", header=",".join(cols),
                comments="", fmt="%.17g")
    return csv_path


def gen_x_target_csv(n: int, p: int, seed: int) -> Path:
    """Generate the NumPy/PCG X_target sidecar used by R/MATLAB cells."""
    path = X_TARGET_DIR / f"xtarget_{n}x{p}_seed{seed}.csv"
    if path.exists():
        return path
    csv_path = gen_dataset_csv(n, p, seed)
    arr = np.loadtxt(csv_path, delimiter=",", skiprows=1, dtype=np.float64)
    X = np.ascontiguousarray(arr[:, :-1])
    rng = np.random.default_rng(seed)
    X_target = X + 0.01 * rng.standard_normal(X.shape)
    np.savetxt(path, X_target, delimiter=",", comments="", fmt="%.17g")
    return path


def predictions_path(algo: str, backend: str, n: int, p: int,
                      threads: int, build: str) -> Path:
    return PREDS_DIR / f"{algo}_{backend}_{n}x{p}_t{threads}_{build}.npy"


def reference_oracle_path(algo: str, backend: str, n: int, p: int,
                          prediction_seed: int) -> Path:
    return ORACLES_DIR / f"{algo}_{backend}_{n}x{p}_seed{prediction_seed}.npy"


def prediction_metadata_path(pred_path: Path) -> Path:
    return pred_path.with_suffix(".json")


def record_prediction_seed(record: dict) -> int:
    value = record.get("prediction_seed")
    if value not in (None, ""):
        return int(value)
    pred_path = record.get("predictions_path")
    if pred_path:
        meta_path = prediction_metadata_path(Path(pred_path))
        if meta_path.exists():
            try:
                meta = json.loads(meta_path.read_text())
                value = meta.get("prediction_seed")
                if value not in (None, ""):
                    return int(value)
            except (OSError, json.JSONDecodeError, TypeError, ValueError):
                pass
    seed_base = int(record.get("seed_base") or DEFAULT_SEED_BASE)
    measured_runs = record.get("n_runs")
    try:
        measured_runs_int = int(measured_runs)
    except (TypeError, ValueError):
        measured_runs_int = 1
    if record_timing_schema(record) == TIMING_SCHEMA:
        return seed_base + max(1, measured_runs_int) - 1
    # Legacy CSVs did not record the exact last timed seed. Use seed_base as
    # the safest value for one-shot runs. For warmup-discarded rows, bench
    # scripts stored n_runs=input_runs-1, so the last seed is base+n_runs.
    if measured_runs_int > 1:
        return seed_base + measured_runs_int
    return seed_base


def write_prediction_metadata(record: dict) -> None:
    pred_path = record.get("predictions_path")
    if not record.get("ok") or not pred_path:
        return
    path = Path(pred_path)
    if not path.exists():
        return
    meta = {
        "algorithm": record.get("algorithm"),
        "backend": record.get("backend"),
        "kind": record.get("kind"),
        "n": record.get("n"),
        "p": record.get("p"),
        "threads": record.get("threads"),
        "libp4a_build": record.get("libp4a_build"),
        "seed_base": record.get("seed_base"),
        "prediction_seed": record_prediction_seed(record),
        "n_runs": record.get("n_runs"),
        "total_runs": record.get("total_runs"),
        "max_runs": record.get("max_runs"),
        "warmup_ms": record.get("warmup_ms"),
        "timing_statistic": record.get("timing_statistic"),
        "versions": record.get("versions") or {},
    }
    meta_path = prediction_metadata_path(path)
    tmp = meta_path.with_name(f"{meta_path.name}.tmp.{os.getpid()}")
    tmp.write_text(json.dumps(meta, sort_keys=True, indent=2) + "\n")
    os.replace(tmp, meta_path)


def snapshot_reference_oracle(record: dict) -> None:
    """Freeze the canonical external prediction vector for later gates.

    The regular predictions cache is per-run scratch. This snapshot is the
    oracle used by --only-pls4all runs; it changes only when the canonical
    reference backend is run again, typically after updating that library.
    """
    if not record.get("ok") or record.get("kind") != "external":
        return
    pred_path = record.get("predictions_path")
    if not pred_path or not Path(pred_path).exists():
        return
    try:
        method = get_method(record["algorithm"])
        canonical = canonical_reference_for_method(method)
    except KeyError:
        return
    if canonical is None:
        return
    canonical_backend = f"ref_{canonical['id']}"
    if record.get("backend") != canonical_backend:
        return

    prediction_seed = record_prediction_seed(record)
    dst = reference_oracle_path(
        record["algorithm"], canonical_backend,
        int(record["n"]), int(record["p"]), prediction_seed)
    tmp_oracle = dst.with_name(f"{dst.name}.tmp.{os.getpid()}")
    shutil.copy2(Path(pred_path), tmp_oracle)
    os.replace(tmp_oracle, dst)
    meta = {
        "algorithm": record.get("algorithm"),
        "backend": canonical_backend,
        "n": record.get("n"),
        "p": record.get("p"),
        "seed_base": record.get("seed_base"),
        "prediction_seed": prediction_seed,
        "n_runs": record.get("n_runs"),
        "total_runs": record.get("total_runs"),
        "max_runs": record.get("max_runs"),
        "warmup_ms": record.get("warmup_ms"),
        "timing_statistic": record.get("timing_statistic"),
        "source_predictions_path": str(pred_path),
        "versions": record.get("versions") or {},
    }
    meta_path = prediction_metadata_path(dst)
    tmp_meta = meta_path.with_name(f"{meta_path.name}.tmp.{os.getpid()}")
    tmp_meta.write_text(json.dumps(meta, sort_keys=True, indent=2) + "\n")
    os.replace(tmp_meta, meta_path)


def load_prediction_array(path: Path):
    try:
        return np.load(path), ""
    except Exception as exc:
        return None, f"load error: {exc}"


def load_reference_oracle(algo: str, canonical_backend: str, n: int, p: int,
                          prediction_seed: int, reference_ref: dict | None):
    """Load the canonical external prediction vector for one dataset seed."""
    if reference_ref is not None:
        ref_seed = record_prediction_seed(reference_ref)
        rref_path = reference_ref.get("predictions_path")
        if ref_seed == prediction_seed and rref_path:
            path = Path(rref_path)
            if path.exists():
                arr, note = load_prediction_array(path)
                if arr is not None:
                    return arr, str(path), "current"
                return None, str(path), note

    path = reference_oracle_path(algo, canonical_backend, n, p,
                                 prediction_seed)
    if path.exists():
        arr, note = load_prediction_array(path)
        if arr is not None:
            return arr, str(path), "stored"
        return None, str(path), note
    return None, str(path), "missing"


def _params_to_json(params: dict) -> str:
    """Serialise the orchestrator-side `adapted_params` dict so R/MATLAB
    receive the exact same param shape Python uses. Numpy arrays become
    plain lists; numpy scalars become Python scalars. Keys that only make
    sense to the Python registry layer (n_samples, n_features) are dropped
    so the R dispatcher's per-algo whitelists don't reject them."""
    out = {}
    for key, value in params.items():
        if key in {"n_samples", "n_features"}:
            continue
        if hasattr(value, "tolist"):
            out[key] = value.tolist()
        elif isinstance(value, (np.integer,)):
            out[key] = int(value)
        elif isinstance(value, (np.floating,)):
            out[key] = float(value)
        else:
            out[key] = value
    return json.dumps(out)


def _subprocess_failure_reason(out: subprocess.CompletedProcess) -> str:
    """Return the most actionable short failure line from stdout/stderr.

    R and Octave often end with generic stack frames such as
    "Execution halted" or "bench_*.m at line ..."; the actual gate reason
    is usually one or two lines above.
    """
    lines = []
    for text in (out.stderr, out.stdout):
        lines.extend((text or "").strip().splitlines())
    for line in reversed(lines):
        clean = line.strip()
        low = clean.lower()
        if not clean:
            continue
        if clean == "Execution halted":
            continue
        if low.startswith("calls:"):
            continue
        if low == "error: called from":
            continue
        if low.startswith("called from"):
            continue
        if " at line " in low and " column " in low:
            continue
        if low.startswith("error: ignoring const execution_exception"):
            continue
        return clean[:200]
    return "exit != 0"


def run_backend(name: str, script: str, language: str, tier: str,
                 kind: str, algo: str, n: int, p: int, n_components: int,
                 n_runs: int, threads: int, libp4a_build: str,
                 seed_base: int, timeout: int) -> dict:
    """Run one backend on one cell. Returns the parsed JSON record
    augmented with cell metadata."""
    script_path = SCRIPTS_DIR / script
    if not script_path.exists():
        return {"backend": name, "ok": False,
                 "reason": f"script not found: {script}"}

    # Per-cell env: thread caps + libp4a build choice + seed base.
    env = os.environ.copy()
    env["OMP_NUM_THREADS"]      = str(threads)
    env["OPENBLAS_NUM_THREADS"] = str(threads)
    env["MKL_NUM_THREADS"]      = str(threads)
    env["BLIS_NUM_THREADS"]     = str(threads)
    env["BENCH_THREADS"]        = str(threads)
    env["BENCH_SEED_BASE"]      = str(seed_base)
    env["BENCH_CELL_TIMEOUT"]   = str(timeout)
    # Externals don't care about libp4a build; pls4all backends do.
    lib_dir = LIBP4A_BUILDS[libp4a_build]
    env["PLS4ALL_LIB_DIR"] = lib_dir
    # PLS4ALL_LIB_PATH is the env var actually honoured by
    # `pls4all._ffi._load_library` to override library discovery. Without
    # it the per-build sweep silently runs against whichever libp4a the
    # importer finds first (auditwheel sibling / pip-installed wheel /
    # build/dev-release), making the build column degenerate.
    env["PLS4ALL_LIB_PATH"] = os.path.join(lib_dir, "libp4a.so")
    env["BENCH_LIBP4A_BUILD"] = libp4a_build
    env["LD_LIBRARY_PATH"] = ":".join([
        lib_dir,
        f"{PLS4ALL_R_ENV}/lib",
        f"{PLS4ALL_R_ENV}/lib/octave/10.3.0",
        env.get("LD_LIBRARY_PATH", ""),
    ])
    env["PATH"] = ":".join([
        f"{PLS4ALL_R_ENV}/bin",
        env.get("PATH", ""),
    ])
    env.setdefault("OCTAVE_HOME", PLS4ALL_R_ENV)
    env["PYTHONPATH"] = ":".join([
        str(REPO),
        str(REPO / "bindings/python/src"),
        env.get("PYTHONPATH", ""),
    ])
    if script == "bench_registry_reference.py" and name.startswith("ref_"):
        env["BENCH_REFERENCE_ID"] = name[len("ref_"):]

    # R / MATLAB scripts route through `pls4all_method` (R) or the
    # +pls4all package (MATLAB). Both need the per-algo params that
    # `adapted_params` computes in Python. Serialise to JSON env var so
    # the R/MATLAB script gets the exact same param shape we use on the
    # Python side.
    if script.endswith(".R") or script.endswith(".m"):
        try:
            from benchmarks.cross_binding.scripts.bench_registry_common import (
                adapted_params, load_method)
            method = load_method(algo)
            params = adapted_params(method, n, p, n_components)
            env["BENCH_R_PARAMS_JSON"] = _params_to_json(params)
            env["BENCH_MATLAB_PARAMS_JSON"] = env["BENCH_R_PARAMS_JSON"]
            if method.needs_labels:
                env["BENCH_R_NEEDS_LABELS"] = "1"
            if method.needs_sample_weights:
                env["BENCH_R_NEEDS_SAMPLE_WEIGHTS"] = "1"
            if method.needs_group_assignment:
                env["BENCH_R_NEEDS_GROUP_ASSIGNMENT"] = "1"
            if method.needs_x_target:
                # R/MATLAB read per-seed NumPy/PCG sidecars so their
                # X_target matches the Python registry exactly.
                env["BENCH_R_NEEDS_X_TARGET"] = "1"
                env["BENCH_R_X_TARGET_DIR"] = str(X_TARGET_DIR)
            # Tell R/MATLAB which field of the result struct matches the
            # registry's parity reference — otherwise classifiers return
            # integer labels where the registry expects `decision_scores`,
            # producing silent parity failures.
            env["BENCH_PREDICTION_KEY"] = method.prediction_key or "predictions"
        except Exception as exc:  # pragma: no cover
            # If the registry import fails we still launch the cell — the
            # bench script will fall back to its own dispatch logic or
            # surface a clearer error than a hard crash here.
            print(f"WARN: could not build BENCH_R_PARAMS_JSON for "
                  f"{algo}: {exc!r}", flush=True)

    pred_path = predictions_path(algo, name, n, p, threads, libp4a_build)
    # Common 8-arg contract used by all scripts (Python + R via positional
    # args; Octave reads BENCH_* env vars instead).
    argv8 = [algo, str(DATA_DIR), str(n), str(p), str(n_components),
              str(n_runs), str(seed_base), str(pred_path)]

    if script.endswith(".py"):
        cmd = [sys.executable, str(script_path), *argv8]
    elif script.endswith(".R"):
        rscript = shutil.which("Rscript") or f"{PLS4ALL_R_ENV}/bin/Rscript"
        cmd = [rscript, str(script_path), *argv8]
    elif script.endswith(".m"):
        oct_bin = shutil.which("octave") or f"{PLS4ALL_R_ENV}/bin/octave"
        env.update({
            "BENCH_ALGO": algo,
            "BENCH_CSV_DIR": str(DATA_DIR),
            "BENCH_N": str(n),
            "BENCH_P": str(p),
            "BENCH_NCOMP": str(n_components),
            "BENCH_NRUNS": str(n_runs),
            "BENCH_PRED_PATH": str(pred_path),
        })
        m_name = Path(script).stem
        cmd = [oct_bin, "--no-gui", "--no-history",
                "--eval",
                f"addpath(genpath('{REPO}/bindings/matlab')); "
                f"addpath('{SCRIPTS_DIR}'); {m_name}"]
    else:
        return {"backend": name, "ok": False,
                 "reason": f"unsupported extension: {script}"}

    t0 = time.perf_counter()
    try:
        out = subprocess.run(cmd, env=env, capture_output=True, text=True,
                              timeout=timeout)
    except FileNotFoundError:
        return {"backend": name, "ok": False,
                 "reason": f"command not found: {cmd[0]}",
                 "subprocess_s": 0.0}
    except subprocess.TimeoutExpired:
        return {"backend": name, "ok": False, "reason": "timeout",
                 "subprocess_s": timeout}
    elapsed = time.perf_counter() - t0
    if out.returncode != 0:
        reason = _subprocess_failure_reason(out)
        return {"backend": name, "ok": False, "reason": reason,
                 "subprocess_s": elapsed,
                 "stdout_tail": out.stdout.strip().splitlines()[-3:],
                 "stderr_tail": out.stderr.strip().splitlines()[-3:]}
    last = out.stdout.strip().splitlines()[-1]
    try:
        rec = json.loads(last)
    except json.JSONDecodeError:
        return {"backend": name, "ok": False,
                 "reason": "non-json output", "tail": last[:200]}
    versions = rec.get("versions")
    if not isinstance(versions, dict):
        versions = {}
    versions["timing_schema"] = TIMING_SCHEMA
    rec["versions"] = versions
    rec["backend"] = name
    rec["subprocess_s"] = elapsed
    rec.setdefault("ok", True)
    if rec.get("reported_ms") in (None, "") and rec.get("median_ms") not in (None, ""):
        rec["reported_ms"] = rec.get("median_ms")
    return rec


def compute_parity(records: list[dict]) -> None:
    """In-place augment each record with TWO parity gates.

    Gate 1 — **binding consistency** (`binding_parity_*` columns):
    every pls4all binding should bit-exactly match our C kernel (cpp).
    Reference per `(algo, n, p)`: cpp @ 1 thread @ blas-omp if
    available, fall back to python_tier1 @ 1 thread @ blas-omp, then
    first OK record. Tolerance: `max|pred - ref| ≤ 1e-6`.

    Gate 2 — **external-reference validity** (`reference_parity_*`
    columns): cpp should agree with the canonical external reference
    (sklearn / ropls / mixOmics / ikpls / R::pls / …) declared by the
    method's `MethodSpec`. Reference per `(algo, n, p, prediction_seed)`:
    the current `ref_<canonical_id>` row when it is part of the run,
    otherwise the frozen oracle snapshot written under
    `data/.reference_oracles/`. Those snapshots change only when the
    canonical reference backend is run again.
    Tolerance: `||pred - ref||_RMS / ||ref||_RMS ≤ method.rmse_rel_tol`
    (with `rmse_abs ≤ 1e-9` escape near zero). Methods with
    `reference_kind == "paper_only"` show `—` (NaN + ok=None) — no
    executable truth to compare against.

    Legacy `parity_*` columns are kept for compatibility. For pls4all
    rows they mirror Gate 1; for external rows they are explicitly
    not-applicable so old renderers do not show an external library as a
    binding failure.
    """
    # Lazy imports — keep the orchestrator startup cheap when callers
    # only need the cell scheduler (e.g. unit tests).
    from benchmarks.parity_timing._comparator import (binding_parity,
                                                       reference_parity)
    from benchmarks.parity_timing.registry import (
        canonical_reference_for_method, get_method, reference_kind)

    for r in records:
        snapshot_reference_oracle(r)

    # Group records by the actual dataset seed. Binding/reference parity only
    # makes sense when every vector came from the same generated dataset.
    ok_groups: dict[tuple, list[dict]] = {}
    for r in records:
        if r.get("ok"):
            key = (r["algorithm"], r["n"], r["p"], record_prediction_seed(r))
            ok_groups.setdefault(key, []).append(r)

    for (algo, n, p, _prediction_seed), group in ok_groups.items():
        # ---- Resolve gate-1 reference (cpp @ 1 thread @ blas-omp) ----
        binding_ref = None
        for cand_name in ("cpp", "python_tier1"):
            for r in group:
                if (r["backend"] == cand_name and r.get("threads") == 1
                        and r.get("libp4a_build") == "blas-omp"):
                    binding_ref = r
                    break
            if binding_ref:
                break
        if binding_ref is None:
            binding_ref = group[0]

        binding_ref_pred = None
        bref_path = binding_ref.get("predictions_path")
        if bref_path and Path(bref_path).exists():
            try:
                binding_ref_pred = np.load(bref_path)
            except Exception:
                binding_ref_pred = None

        # ---- Resolve gate-2 reference (canonical external) -----------
        try:
            method = get_method(algo)
            canonical = canonical_reference_for_method(method)
            ref_kind = reference_kind(method)
        except KeyError:
            canonical, ref_kind = None, "external"

        reference_ref = None
        canonical_backend = ""
        if canonical is not None:
            canonical_backend = f"ref_{canonical['id']}"
            for r in group:
                if r["backend"] == canonical_backend:
                    reference_ref = r
                    break
        rel_tol = float(getattr(method, "rmse_rel_tol", 5e-2)) \
            if canonical is not None else float("nan")

        # ---- Populate both gates per cell ---------------------------
        for r in group:
            # Legacy/back-compat fields keep mirroring the binding gate
            # so the existing renderer doesn't fall over.
            r["binding_parity_ref"] = binding_ref["backend"]
            r["reference_parity_ref"] = (
                canonical_backend if canonical is not None else "")
            r["reference_parity_tolerance"] = rel_tol
            r["reference_oracle_path"] = ""
            r["reference_kind"] = ref_kind
            r["is_canonical_reference"] = (
                r.get("backend") == r["reference_parity_ref"]
                and r["reference_parity_ref"] != "")

            pp = r.get("predictions_path")
            r_pred = None
            if pp and Path(pp).exists():
                r_pred, load_note = load_prediction_array(Path(pp))
                if r_pred is None:
                    r["binding_parity_note"] = load_note
                    r["reference_parity_note"] = load_note

            # Gate 1: binding consistency vs cpp. External libraries are
            # compared by Gate 2 only; they are not pls4all bindings.
            is_pls4all = r.get("kind") in {"pls4all_core", "pls4all_binding"}
            if not is_pls4all:
                r["binding_parity_max_diff"] = float("nan")
                r["binding_parity_ok"] = None
                r["binding_parity_note"] = "not_applicable"
            elif r_pred is None or binding_ref_pred is None:
                r["binding_parity_max_diff"] = float("nan")
                r["binding_parity_ok"] = False
                r["binding_parity_note"] = (
                    r.get("binding_parity_note") or "predictions missing")
            else:
                bres = binding_parity(r_pred, binding_ref_pred,
                                       tolerance=r.get("parity_tolerance",
                                                        1e-6))
                r["binding_parity_max_diff"] = bres.max_abs_diff
                r["binding_parity_ok"] = bres.ok
                r["binding_parity_note"] = bres.note

            # Legacy mirrors.
            r["parity_ref"] = r["binding_parity_ref"] if is_pls4all else ""
            r["parity_max_diff"] = r["binding_parity_max_diff"]
            r["parity_ok"] = r["binding_parity_ok"]
            r["parity_note"] = r.get("binding_parity_note", "")

            # Gate 2: external-reference validity.
            if ref_kind == "paper_only":
                r["reference_parity_rmse_abs"] = float("nan")
                r["reference_parity_rmse_rel"] = float("nan")
                r["reference_parity_ok"] = None
                r["reference_parity_note"] = "paper_only"
            elif r["is_canonical_reference"]:
                # The canonical reference compared to itself — trivially OK.
                r["reference_parity_rmse_abs"] = 0.0
                r["reference_parity_rmse_rel"] = 0.0
                r["reference_parity_ok"] = True
                r["reference_parity_note"] = "self"
            elif r_pred is None:
                r["reference_parity_rmse_abs"] = float("nan")
                r["reference_parity_rmse_rel"] = float("nan")
                r["reference_parity_ok"] = False
                r["reference_parity_note"] = (
                    r.get("reference_parity_note", "predictions missing"))
            else:
                prediction_seed = record_prediction_seed(r)
                reference_ref_pred, oracle_path, oracle_status = (
                    load_reference_oracle(algo, canonical_backend, n, p,
                                          prediction_seed, reference_ref))
                r["reference_oracle_path"] = oracle_path
                if reference_ref_pred is None:
                    r["reference_parity_rmse_abs"] = float("nan")
                    r["reference_parity_rmse_rel"] = float("nan")
                    r["reference_parity_ok"] = False
                    if oracle_status == "missing":
                        r["reference_parity_note"] = (
                            "reference oracle missing; run canonical "
                            "reference backend")
                    else:
                        r["reference_parity_note"] = oracle_status
                    continue
                rres = reference_parity(r_pred, reference_ref_pred,
                                          tolerance=rel_tol)
                r["reference_parity_rmse_abs"] = rres.rmse_abs
                r["reference_parity_rmse_rel"] = rres.rmse_rel
                r["reference_parity_ok"] = rres.ok
                r["reference_parity_note"] = rres.note


def parse_sizes(args_sizes) -> list[tuple[int, int]]:
    if not args_sizes:
        return DEFAULT_SIZES
    out = []
    for s in args_sizes:
        n, p = s.split("x")
        out.append((int(n), int(p)))
    return out


CSV_COLUMNS = [
    "algorithm", "backend", "language", "tier", "kind",
    "n", "p", "threads", "libp4a_build", "seed_base", "prediction_seed",
    "ok", "reported_ms", "median_ms", "sample_median_ms", "mean_ms",
    "min_ms", "max_ms", "n_runs", "total_runs", "max_runs", "warmup_ms",
    "warmup_included", "timing_statistic", "timing_decision",
    # Legacy alias columns — mirror the binding-consistency gate so the
    # existing dashboard renderer still works during the migration.
    "parity_ref", "parity_max_diff", "parity_ok", "parity_note",
    # Gate 1: binding consistency (every pls4all binding vs cpp).
    "binding_parity_ref", "binding_parity_max_diff",
    "binding_parity_ok", "binding_parity_note",
    # Gate 2: external-reference validity (cpp vs canonical reference,
    # tolerance = method.rmse_rel_tol).
    "reference_parity_ref", "reference_parity_rmse_abs",
    "reference_parity_rmse_rel", "reference_parity_ok",
    "reference_parity_note", "reference_parity_tolerance",
    "reference_oracle_path", "reference_kind",
    "is_canonical_reference",
    "subprocess_s", "predictions_path",
    "versions_json", "reason",
]


def _to_bool(value):
    if isinstance(value, bool):
        return value
    if value in (None, ""):
        return None
    return str(value).strip().lower() == "true"


def _to_int(value):
    if value in (None, ""):
        return None
    return int(value)


def _to_float(value):
    if value in (None, ""):
        return None
    try:
        return float(value)
    except ValueError:
        return None


def coerce_csv_record(row: dict) -> dict:
    out = dict(row)
    for name in ("n", "p", "threads", "seed_base", "prediction_seed",
                 "n_runs", "total_runs", "max_runs"):
        out[name] = _to_int(out.get(name))
    for name in ("ok", "parity_ok", "binding_parity_ok",
                 "reference_parity_ok", "is_canonical_reference",
                 "warmup_included"):
        out[name] = _to_bool(out.get(name))
    for name in ("reported_ms", "median_ms", "sample_median_ms",
                 "mean_ms", "min_ms", "max_ms", "warmup_ms",
                 "parity_max_diff", "binding_parity_max_diff",
                 "reference_parity_rmse_abs", "reference_parity_rmse_rel",
                 "reference_parity_tolerance", "subprocess_s"):
        out[name] = _to_float(out.get(name))
    try:
        out["versions"] = json.loads(out.get("versions_json") or "{}")
    except json.JSONDecodeError:
        out["versions"] = {}
    return out


def load_existing_records(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with path.open(newline="") as f:
        return [coerce_csv_record(r) for r in csv.DictReader(f)]


def load_defer_rules(path: str | None) -> list[dict]:
    """Load wildcard cell rules used to defer expensive benchmark cells.

    CSV columns are optional among: algorithm, backend, language, tier, kind,
    n, p, threads, libp4a_build, reason. Empty values and "*" are wildcards.
    """
    if not path:
        return []
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"defer rules file not found: {p}")
    with p.open(newline="") as f:
        return [dict(r) for r in csv.DictReader(f)]


def _rule_matches_value(rule_value, actual) -> bool:
    value = str(rule_value or "").strip()
    if not value or value == "*":
        return True
    return value == str(actual)


def defer_reason(rules: list[dict], *, algorithm: str, backend: str,
                 language: str, tier: str, kind: str, n: int, p: int,
                 threads: int, libp4a_build: str) -> str | None:
    cell = {
        "algorithm": algorithm,
        "backend": backend,
        "language": language,
        "tier": tier,
        "kind": kind,
        "n": n,
        "p": p,
        "threads": threads,
        "libp4a_build": libp4a_build,
    }
    for rule in rules:
        if all(_rule_matches_value(rule.get(key), value)
               for key, value in cell.items()):
            return (rule.get("reason") or
                    "deferred slow cell for CRAN-fast benchmark")
    return None


def slot_key(record: dict) -> tuple:
    return (record.get("algorithm"), record.get("backend"),
            int(record.get("n")), int(record.get("p")),
            int(record.get("threads")), record.get("libp4a_build"))


def strict_key(record: dict) -> tuple:
    sb = record.get("seed_base")
    nr = record.get("max_runs") if record_timing_schema(record) == TIMING_SCHEMA else record.get("n_runs")
    return (*slot_key(record),
            int(sb) if sb is not None else None,
            int(nr) if nr is not None else None)


def planned_keys(algo: str, backend_name: str, n: int, p: int,
                 threads: int, build: str, seed_base: int,
                 n_runs: int) -> tuple[tuple, tuple]:
    slot = (algo, backend_name, n, p, threads, build)
    # Adaptive rows emit n_runs = actual timed samples and max_runs = the
    # planning cap. Resume strictness therefore keys on the cap.
    strict = (*slot, seed_base, n_runs)
    return slot, strict


def record_timing_schema(record: dict) -> str | None:
    versions = record.get("versions")
    if isinstance(versions, dict):
        value = versions.get("timing_schema")
        return str(value) if value else None
    return None


def write_records(csv_out: Path, records: list[dict]) -> None:
    with csv_out.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(CSV_COLUMNS)
        for r in records:
            w.writerow([
                r.get("algorithm"), r.get("backend"), r.get("language"),
                r.get("tier"), r.get("kind"),
                r.get("n"), r.get("p"), r.get("threads"),
                r.get("libp4a_build"), r.get("seed_base"),
                r.get("prediction_seed") or record_prediction_seed(r),
                r.get("ok"),
                r.get("reported_ms") if r.get("reported_ms") not in (None, "") else r.get("median_ms"),
                r.get("median_ms"),
                r.get("sample_median_ms"),
                r.get("mean_ms"),
                r.get("min_ms"), r.get("max_ms"), r.get("n_runs"),
                r.get("total_runs"), r.get("max_runs"), r.get("warmup_ms"),
                r.get("warmup_included"),
                r.get("timing_statistic"),
                r.get("timing_decision"),
                # Legacy alias columns (mirror gate 1).
                r.get("parity_ref"), r.get("parity_max_diff"),
                r.get("parity_ok"), r.get("parity_note", ""),
                # Gate 1: binding consistency.
                r.get("binding_parity_ref"),
                r.get("binding_parity_max_diff"),
                r.get("binding_parity_ok"),
                r.get("binding_parity_note", ""),
                # Gate 2: external-reference validity.
                r.get("reference_parity_ref", ""),
                r.get("reference_parity_rmse_abs"),
                r.get("reference_parity_rmse_rel"),
                r.get("reference_parity_ok"),
                r.get("reference_parity_note", ""),
                r.get("reference_parity_tolerance"),
                r.get("reference_oracle_path", ""),
                r.get("reference_kind", ""),
                r.get("is_canonical_reference"),
                r.get("subprocess_s"), r.get("predictions_path"),
                json.dumps(r.get("versions") or {}), r.get("reason", ""),
            ])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--algorithms", nargs="+", default=["pls"],
                         help="Algorithm names (space-separated), or 'all' "
                              "for the canonical registry")
    parser.add_argument("--algorithms-file",
                         help="File with one algorithm per line")
    parser.add_argument("--sizes", nargs="*",
                         help="Dataset sizes 'NxP' (default: 11-size matrix)")
    parser.add_argument("--threads", nargs="+", type=int, default=[1, 3, 10],
                         help="Thread counts to sweep")
    parser.add_argument("--libp4a-build",
                         choices=["dev-release", "blas-on", "omp-on",
                                  "blas-omp", "cuda-on", "both",
                                  "all-cpu", "all"],
                         default="blas-omp",
                         help="libp4a build for pls4all backends. 'both' = "
                              "dev-release + blas-omp; 'all-cpu' = native + "
                              "blas + omp + blasomp; 'all' = the full "
                              "5-build sweep including cuda-on.")
    parser.add_argument("--n-runs", type=int, default=DEFAULT_MAX_RUNS,
                         help="Maximum timed runs per cell under the adaptive protocol")
    parser.add_argument("--n-components", type=int, default=5)
    parser.add_argument("--seed-base", type=int, default=DEFAULT_SEED_BASE)
    parser.add_argument("--only-pls4all", action="store_true",
                         help="Skip external backends entirely")
    parser.add_argument("--canonical-pls4all-only", action="store_true",
                         help="Run only the registry-driven pls4all backend "
                              "plus whichever external references are selected. "
                              "Useful for complete method-catalog sweeps.")
    parser.add_argument("--registry-cells", action="store_true",
                         help="Use each canonical MethodSpec's own "
                              "(n_samples, n_features) cell instead of the "
                              "global --sizes sweep. This is the complete "
                              "method-catalog mode.")
    parser.add_argument("--reference-backends",
                         choices=["none", "fixed", "registry", "all"],
                         default="all",
                         help="External references to include: legacy fixed "
                              "scripts, registry-driven references, both, or "
                              "none.")
    parser.add_argument("--only", nargs="*",
                         help="Restrict to a subset of backends by name")
    parser.add_argument("--timeout", type=int, default=DEFAULT_TIMEOUT_S)
    parser.add_argument("--out-csv", default=None,
                         help="Output CSV path (default: results/full_matrix.csv)")
    parser.add_argument("--resume-existing", action="store_true",
                         help="Load --out-csv when it exists and skip cells "
                              "already computed with the same seed/max-runs.")
    parser.add_argument("--force", action="store_true",
                         help="Recompute planned cells even when present in "
                              "--out-csv.")
    parser.add_argument("--rerun-failed", action="store_true",
                         help="With --resume-existing, rerun existing failed "
                              "cells instead of preserving their recorded "
                              "failure.")
    parser.add_argument("--defer-cells-file",
                         help="CSV wildcard rules for cells to mark as "
                              "deferred instead of executing. Used for "
                              "CRAN-fast benchmark publication while slow "
                              "cells run later.")
    parser.add_argument("--flush-each-cell", action="store_true",
                         help="Rewrite --out-csv after every completed cell "
                              "so long sharded runs are resumable and "
                              "observable before the final parity pass.")
    args = parser.parse_args()
    if args.registry_cells:
        os.environ["BENCH_REGISTRY_CELLS"] = "1"
    defer_rules = load_defer_rules(args.defer_cells_file)

    algos = list(args.algorithms)
    if args.algorithms_file:
        with open(args.algorithms_file) as f:
            algos.extend(l.strip() for l in f if l.strip())
    if algos == ["all"]:
        algos = method_names()

    sizes = parse_sizes(args.sizes)
    threads = list(args.threads)
    if args.libp4a_build == "both":
        builds = ["dev-release", "blas-omp"]
    elif args.libp4a_build == "all-cpu":
        builds = ["dev-release", "blas-on", "omp-on", "blas-omp"]
    elif args.libp4a_build == "all":
        builds = ["dev-release", "blas-on", "omp-on", "blas-omp", "cuda-on"]
    else:
        builds = [args.libp4a_build]

    if args.canonical_pls4all_only:
        base_pls4all_backends = [b for b in BACKENDS if b[0] == "registry_pls4all"]
    else:
        base_pls4all_backends = [b for b in BACKENDS if b[4] != "external"]
    fixed_external_backends = [b for b in BACKENDS if b[4] == "external"]

    def backends_for_algo(algo: str):
        backends = list(base_pls4all_backends)
        if not args.only_pls4all and args.reference_backends in {"fixed", "all"}:
            backends.extend(fixed_external_backends)
        if not args.only_pls4all and args.reference_backends in {"registry", "all"}:
            backends.extend(registry_reference_backends_for(algo))
        if args.reference_backends == "none" or args.only_pls4all:
            backends = [b for b in backends if b[4] != "external"]
        if args.only:
            wanted = set(args.only)
            backends = [b for b in backends if b[0] in wanted]
        return backends

    def sizes_for_algo(algo: str):
        if not args.registry_cells:
            return sizes
        try:
            spec = get_method(algo)
        except KeyError:
            return sizes
        return [(int(spec.cell_params["n_samples"]),
                 int(spec.cell_params["n_features"]))]

    csv_out = Path(args.out_csv or RESULTS_DIR / "full_matrix.csv")
    existing_records = load_existing_records(csv_out) if args.resume_existing else []
    existing_by_slot: dict[tuple, dict] = {}
    for row in existing_records:
        try:
            existing_by_slot[slot_key(row)] = row
        except (TypeError, ValueError):
            continue
    planned_slots: set[tuple] = set()
    records: list[dict] = []
    ran_records: list[dict] = []
    skipped_existing = 0

    total = 0
    for algo in algos:
        for _n, _p in sizes_for_algo(algo):
            for _thr in threads:
                for _backend in backends_for_algo(algo):
                    kind = _backend[4]
                    total += len(builds) if kind != "external" else 1
    size_mode = "registry-cells" if args.registry_cells else f"{len(sizes)} sizes"
    print(f"# {len(algos)} algorithms × {size_mode} × "
           f"{len(threads)} threads × dynamic backends "
           f"(builds={builds}) ≈ {total} cells")
    print(f"# seed_base={args.seed_base} timeout={args.timeout}s "
           f"adaptive_max_runs={args.n_runs}")
    if defer_rules:
        print(f"# defer_rules={args.defer_cells_file} ({len(defer_rules)} rules)")

    cell_count = 0
    for algo in algos:
        algo_sizes = sizes_for_algo(algo)
        algo_backends = backends_for_algo(algo)
        for n, p in algo_sizes:
            # Pre-generate CSV cache for the adaptive seed window. Python,
            # R and MATLAB all read these files so each backend sees the
            # exact same input bits for any run count selected at runtime.
            last_seed = args.seed_base + args.n_runs - 1
            gen_dataset_csv(n, p, last_seed)
            for i in range(args.n_runs):
                gen_dataset_csv(n, p, args.seed_base + i)
            try:
                method = get_method(algo)
            except KeyError:
                method = None
            if method is not None and method.needs_x_target:
                for i in range(args.n_runs):
                    gen_x_target_csv(n, p, args.seed_base + i)

            for thr in threads:
                for backend in algo_backends:
                    name, script, lang, tier, kind = backend
                    # pls4all bindings sweep over `builds`; externals
                    # always use blas-omp (most realistic config for
                    # an external user on a modern install).
                    if kind == "external":
                        per_backend_builds = ["blas-omp"]
                    else:
                        per_backend_builds = list(builds)
                    for build in per_backend_builds:
                        cell_count += 1
                        prefix = (f"[{cell_count}] {algo:<22s} "
                                  f"{n:>5d}×{p:<4d} t={thr:<2d} "
                                  f"{build:<11s} {name:<14s}: ")
                        print(prefix, end="", flush=True)
                        slot, strict = planned_keys(
                            algo, name, n, p, thr, build,
                            args.seed_base, args.n_runs)
                        # Accept the raw input count for current rows. The
                        # timing_schema guard keeps legacy cold-run CSV rows
                        # from being reused under --resume-existing.
                        strict_alt = (*slot, args.seed_base, args.n_runs)
                        planned_slots.add(slot)
                        existing = existing_by_slot.get(slot)
                        if (args.resume_existing and not args.force
                                and existing is not None
                                and record_timing_schema(existing) == TIMING_SCHEMA
                                and strict_key(existing) in (strict, strict_alt)
                                and (existing.get("ok") or not args.rerun_failed)):
                            records.append(existing)
                            skipped_existing += 1
                            status = "ok" if existing.get("ok") else "failed"
                            print(f"SKIP existing {status}")
                            continue

                        reason = defer_reason(
                            defer_rules,
                            algorithm=algo, backend=name, language=lang,
                            tier=tier, kind=kind, n=n, p=p, threads=thr,
                            libp4a_build=build)
                        if reason:
                            rec = {
                                "backend": name,
                                "ok": False,
                                "skipped": True,
                                "reason": reason,
                                "algorithm": algo,
                                "n": n, "p": p,
                                "threads": thr,
                                "libp4a_build": build,
                                "language": lang,
                                "tier": tier,
                                "kind": kind,
                                "seed_base": args.seed_base,
                                "prediction_seed": args.seed_base,
                                "n_runs": 0,
                                "total_runs": 0,
                                "max_runs": args.n_runs,
                                "timing_statistic": "deferred",
                                "timing_decision": "deferred",
                                "subprocess_s": 0.0,
                                "versions": {"timing_schema": TIMING_SCHEMA},
                            }
                            records.append(rec)
                            print(f"DEFER — {reason[:100]}")
                            if args.flush_each_cell:
                                write_records(csv_out, records)
                            continue

                        rec = run_backend(name, script, lang, tier, kind,
                                            algo, n, p, args.n_components,
                                            args.n_runs, thr, build,
                                            args.seed_base, args.timeout)
                        rec.update({
                            "algorithm": algo,
                            "n": n, "p": p,
                            "threads": thr,
                            "libp4a_build": build,
                            "language": lang,
                            "tier": tier,
                            "kind": kind,
                            "seed_base": args.seed_base,
                        })
                        rec["prediction_seed"] = record_prediction_seed(rec)
                        write_prediction_metadata(rec)
                        snapshot_reference_oracle(rec)
                        records.append(rec)
                        ran_records.append(rec)
                        if rec.get("ok"):
                            score = rec.get("reported_ms", rec.get("median_ms", 0))
                            stat = rec.get("timing_statistic", "time")
                            runs_done = rec.get("n_runs", "?")
                            print(f"{stat}={float(score):.2f}ms n={runs_done}")
                        else:
                            print(f"FAIL — {rec.get('reason', '?')[:100]}")
                        if args.flush_each_cell:
                            write_records(csv_out, records)

    if existing_records:
        preserved = [r for r in existing_records
                     if slot_key(r) not in planned_slots]
        if preserved:
            print(f"# Preserving {len(preserved)} existing out-of-scope cells")
        records = preserved + records

    # Post-hoc parity computation.
    if ran_records:
        ran_ids = {id(r) for r in ran_records}
        parity_scope = [r for r in records
                        if id(r) in ran_ids
                        or (r.get("predictions_path")
                            and Path(str(r["predictions_path"])).exists())]
        print(f"\n# Computing parity for {len(parity_scope)} records "
              f"({len(ran_records)} newly run)")
        compute_parity(parity_scope)
    else:
        print("\n# No cells ran; preserving existing parity columns")

    # Write CSV.
    write_records(csv_out, records)
    print(f"CSV  → {csv_out}")
    print(f"Predictions cache → {PREDS_DIR}")
    if args.resume_existing:
        print(f"Resume summary: skipped={skipped_existing}, "
              f"ran={len(ran_records)}, total_written={len(records)}")


if __name__ == "__main__":
    main()
