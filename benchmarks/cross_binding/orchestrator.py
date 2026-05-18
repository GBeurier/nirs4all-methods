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
  python orchestrator.py --algorithms pls --sizes 500x200 --threads 1 --n-runs 3

  # complete canonical method/reference matrix
  python orchestrator.py --algorithms all --registry-cells \
    --threads 1 3 10 --n-runs 5 --reference-backends all
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
RESULTS_DIR = HERE / "results"
SCRIPTS_DIR = HERE / "scripts"
for d in (DATA_DIR, PREDS_DIR, RESULTS_DIR, SCRIPTS_DIR):
    d.mkdir(parents=True, exist_ok=True)

from benchmarks.parity_timing.registry import (  # noqa: E402
    get_method,
    method_names,
    resolved_references_for_method,
)

# Default base seed. uint32-safe so it round-trips losslessly through
# R/Octave doubles and accepts as sklearn random_state. +1 per run
# inside each cell, so the 5 seeds used are [B, B+1, B+2, B+3, B+4] —
# still uint32, still 'long int enough' to look intentional and avoid
# collision with other ad-hoc seeds elsewhere in the test suite.
DEFAULT_SEED_BASE = 1_234_567_890

# 11 sizes (10000×2500 skipped per user decision).
DEFAULT_SIZES = [
    (100, 50), (100, 500), (100, 2500),
    (500, 50), (500, 500), (500, 2500),
    (2500, 50), (2500, 500), (2500, 2500),
    (10000, 50), (10000, 500),
]

# Per-cell wall-clock timeout (seconds).
DEFAULT_TIMEOUT_S = 300

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
    ("r_pls",        "bench_r_pls.R",         "R",      "external", "external"),
    ("r_ropls",      "bench_r_ropls.R",       "R",      "external", "external"),
    ("r_mixomics",   "bench_r_mixomics.R",    "R",      "external", "external"),
    ("matlab_tier1", "bench_matlab_tier1.m",  "MATLAB", "tier 1",   "pls4all_binding"),
    ("matlab_tier2", "bench_matlab_tier2.m",  "MATLAB", "tier 2",   "pls4all_binding"),
    ("matlab_pls",   "bench_matlab_pls.m",    "MATLAB", "external", "external"),
]


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
        out.append((f"ref_{ref['id']}", "bench_registry_reference.py",
                    display_lang, ref["library"], "external"))
    return out

# libp4a build → libpath. Used for the cpp variant sweep: each build
# corresponds to a distinct OpenMP / BLAS / CUDA tier of the C kernel.
LIBP4A_BUILDS = {
    "dev-release": str(REPO / "build/dev-release/cpp/src"),  # ref, no BLAS, no OMP
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


def predictions_path(algo: str, backend: str, n: int, p: int,
                      threads: int, build: str) -> Path:
    return PREDS_DIR / f"{algo}_{backend}_{n}x{p}_t{threads}_{build}.npy"


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
                # Not yet implemented for R/MATLAB — would require per-seed
                # .npy sidecars since Python's PCG64 isn't reproducible in
                # other languages. The R/MATLAB script fails the cell with
                # a clear reason when this flag is missing.
                env["BENCH_R_NEEDS_X_TARGET"] = "1"
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
    except subprocess.TimeoutExpired:
        return {"backend": name, "ok": False, "reason": "timeout",
                 "subprocess_s": timeout}
    elapsed = time.perf_counter() - t0
    if out.returncode != 0:
        tail = (out.stderr or out.stdout).strip().splitlines()
        reason = tail[-1][:200] if tail else "exit != 0"
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
    rec["backend"] = name
    rec["subprocess_s"] = elapsed
    rec.setdefault("ok", True)
    return rec


def compute_parity(records: list[dict]) -> None:
    """In-place augment each record with parity_max_diff + parity_ok.

    Reference per (algo, n, p): cpp_blasomp @ 1 thread if available,
    fallback to python_tier1 @ 1 thread, fallback to first OK record.
    Compares the .npy at predictions_path element-wise (max abs diff)."""
    # Group records by (algo, n, p).
    groups: dict[tuple, list[dict]] = {}
    for r in records:
        if not r.get("ok"):
            continue
        groups.setdefault((r["algorithm"], r["n"], r["p"]), []).append(r)

    for key, group in groups.items():
        # Pick the reference: prefer cpp_blasomp @ 1 thread + algo applicable.
        ref = None
        for cand_name in ("cpp", "python_tier1"):
            for r in group:
                if (r["backend"] == cand_name and r.get("threads") == 1
                        and r.get("libp4a_build") == "blas-omp"):
                    ref = r
                    break
            if ref:
                break
        if ref is None:
            ref = group[0]
        ref_pred_path = ref.get("predictions_path")
        if not ref_pred_path or not Path(ref_pred_path).exists():
            for r in group:
                r["parity_ref"] = ref["backend"]
                r["parity_max_diff"] = float("nan")
                r["parity_ok"] = False
                r["parity_note"] = "ref predictions missing"
            continue
        ref_preds = np.load(ref_pred_path)
        for r in group:
            r["parity_ref"] = ref["backend"]
            pp = r.get("predictions_path")
            if not pp or not Path(pp).exists():
                r["parity_max_diff"] = float("nan")
                r["parity_ok"] = False
                r["parity_note"] = "predictions missing"
                continue
            try:
                p = np.load(pp)
            except Exception as e:
                r["parity_max_diff"] = float("nan")
                r["parity_ok"] = False
                r["parity_note"] = f"load error: {e}"
                continue
            if p.shape != ref_preds.shape:
                # Some externals predict only one column for multi-Y; keep
                # the comparison but flag it.
                if p.size == ref_preds.size:
                    p = p.reshape(ref_preds.shape)
                else:
                    r["parity_max_diff"] = float("nan")
                    r["parity_ok"] = False
                    r["parity_note"] = (f"shape mismatch ({p.shape} vs "
                                          f"{ref_preds.shape})")
                    continue
            diff = float(np.max(np.abs(p - ref_preds)))
            r["parity_max_diff"] = diff
            r["parity_ok"] = diff <= r.get("parity_tolerance", 1e-6)


def parse_sizes(args_sizes) -> list[tuple[int, int]]:
    if not args_sizes:
        return DEFAULT_SIZES
    out = []
    for s in args_sizes:
        n, p = s.split("x")
        out.append((int(n), int(p)))
    return out


CSV_COLUMNS = ["algorithm", "backend", "language", "tier", "kind",
               "n", "p", "threads", "libp4a_build", "seed_base",
               "ok", "median_ms", "min_ms", "max_ms", "n_runs",
               "parity_ref", "parity_max_diff", "parity_ok", "parity_note",
               "subprocess_s", "predictions_path",
               "versions_json", "reason"]


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
    for name in ("n", "p", "threads", "seed_base", "n_runs"):
        out[name] = _to_int(out.get(name))
    for name in ("ok", "parity_ok"):
        out[name] = _to_bool(out.get(name))
    for name in ("median_ms", "min_ms", "max_ms",
                 "parity_max_diff", "subprocess_s"):
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


def slot_key(record: dict) -> tuple:
    return (record.get("algorithm"), record.get("backend"),
            int(record.get("n")), int(record.get("p")),
            int(record.get("threads")), record.get("libp4a_build"))


def strict_key(record: dict) -> tuple:
    sb = record.get("seed_base")
    nr = record.get("n_runs")
    return (*slot_key(record),
            int(sb) if sb is not None else None,
            int(nr) if nr is not None else None)


def planned_keys(algo: str, backend_name: str, n: int, p: int,
                 threads: int, build: str, seed_base: int,
                 n_runs: int) -> tuple[tuple, tuple]:
    slot = (algo, backend_name, n, p, threads, build)
    # Bench scripts emit n_runs = number of TIMED samples (warmup discarded
    # when input n_runs >= 3). Resume must match against that value, not
    # the input. See _common.time_runs_seeded.
    measured = n_runs - 1 if n_runs >= 3 else n_runs
    strict = (*slot, seed_base, measured)
    return slot, strict


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
                r.get("ok"), r.get("median_ms"), r.get("min_ms"),
                r.get("max_ms"), r.get("n_runs"),
                r.get("parity_ref"), r.get("parity_max_diff"),
                r.get("parity_ok"), r.get("parity_note", ""),
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
                              "dev-release + blas-omp; 'all-cpu' = ref + "
                              "blas + omp + blasomp; 'all' = the full "
                              "5-build sweep including cuda-on.")
    parser.add_argument("--n-runs", type=int, default=5,
                         help="Runs per cell (first discarded as warmup if >= 3)")
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
                              "already computed with the same seed/n_runs.")
    parser.add_argument("--force", action="store_true",
                         help="Recompute planned cells even when present in "
                              "--out-csv.")
    parser.add_argument("--rerun-failed", action="store_true",
                         help="With --resume-existing, rerun existing failed "
                              "cells instead of preserving their recorded "
                              "failure.")
    args = parser.parse_args()
    if args.registry_cells:
        os.environ["BENCH_REGISTRY_CELLS"] = "1"

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
           f"n_runs={args.n_runs}")

    cell_count = 0
    for algo in algos:
        algo_sizes = sizes_for_algo(algo)
        algo_backends = backends_for_algo(algo)
        for n, p in algo_sizes:
            # Pre-generate CSV cache for the first run's seed (other
            # runs regenerate inline in Python; R/MATLAB always read CSV
            # so they need it. Each run uses seed_base+i but for parity
            # we compare predictions from the LAST run, so we cache the
            # LAST seed's CSV.
            last_seed = args.seed_base + args.n_runs - 1
            gen_dataset_csv(n, p, last_seed)
            # Pre-generate all warmup seeds too, for backends that always
            # reread the CSV per run.
            for i in range(args.n_runs):
                gen_dataset_csv(n, p, args.seed_base + i)

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
                        # Accept both warmup-discarded (n_runs - 1) and raw
                        # input (n_runs) recorded values — bench scripts emit
                        # the timed-sample count on success but legacy/failed
                        # rows may have the input value.
                        strict_alt = (*slot, args.seed_base, args.n_runs)
                        planned_slots.add(slot)
                        existing = existing_by_slot.get(slot)
                        if (args.resume_existing and not args.force
                                and existing is not None
                                and strict_key(existing) in (strict, strict_alt)
                                and (existing.get("ok") or not args.rerun_failed)):
                            records.append(existing)
                            skipped_existing += 1
                            status = "ok" if existing.get("ok") else "failed"
                            print(f"SKIP existing {status}")
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
                        records.append(rec)
                        ran_records.append(rec)
                        if rec.get("ok"):
                            print(f"median={rec.get('median_ms', 0):.2f}ms")
                        else:
                            print(f"FAIL — {rec.get('reason', '?')[:100]}")

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
