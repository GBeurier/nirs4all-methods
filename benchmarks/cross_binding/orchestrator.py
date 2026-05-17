#!/usr/bin/env python3
"""Cross-binding PLS-regression timing harness.

For each (backend, dataset-size) combo, runs an external script that
fits SIMPLS N times and emits a JSON record. The orchestrator
aggregates everything into a Markdown table comparing:

  - C++ direct (libp4a baseline, 1 thread)
  - Python tier 1 (pls4all.cython-style)
  - Python tier 2 (pls4all.sklearn.PLSRegression)
  - R       tier 1 (.Call pls4all_fit)
  - R       tier 2 (pls(formula, data, ncomp))
  - MATLAB  tier 1 (pls4all.pls_fit)
  - MATLAB  tier 2 (pls4all.fit("pls", ...))
  - External Python: sklearn.cross_decomposition.PLSRegression
  - External R:      pls::plsr(method = "simpls")
  - External MATLAB: plsregress (Octave statistics)

Each backend reports {best_ms, median_ms, min_ms, max_ms, n_runs}.
The orchestrator computes speedup ratios relative to the external
implementation of the same language.

Usage:
    python orchestrator.py                                  # all sizes
    python orchestrator.py --sizes 100x50 500x200
    python orchestrator.py --only python_tier1 sklearn      # subset
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
import statistics
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent
DATA_DIR = HERE / "data"
RESULTS_DIR = HERE / "results"
SCRIPTS_DIR = HERE / "scripts"
DATA_DIR.mkdir(exist_ok=True)
RESULTS_DIR.mkdir(exist_ok=True)
SCRIPTS_DIR.mkdir(exist_ok=True)

# (name, runner_callable_or_module, language, tier_label)
BACKENDS = [
    ("cpp",           "bench_cpp.py",          "C++",     "direct"),
    ("python_tier1",  "bench_python_tier1.py", "Python",  "tier 1"),
    ("python_tier2",  "bench_python_tier2.py", "Python",  "tier 2"),
    ("sklearn",       "bench_sklearn.py",      "Python",  "external"),
    ("r_tier1",       "bench_r_tier1.R",       "R",       "tier 1"),
    ("r_tier2",       "bench_r_tier2.R",       "R",       "tier 2"),
    ("r_pls",         "bench_r_pls.R",         "R",       "external"),
    ("matlab_tier1",  "bench_matlab_tier1.m",  "MATLAB",  "tier 1"),
    ("matlab_tier2",  "bench_matlab_tier2.m",  "MATLAB",  "tier 2"),
    ("matlab_pls",    "bench_matlab_pls.m",    "MATLAB",  "external"),
]

DEFAULT_SIZES = [(100, 50), (500, 200), (1000, 500)]


def gen_dataset(n: int, p: int, seed: int = 0) -> Path:
    """Generate a deterministic NumPy-style CSV (n × p X, n × 1 y)."""
    import numpy as np
    out = DATA_DIR / f"data_{n}x{p}_seed{seed}.csv"
    if out.exists():
        return out
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n, p))
    y = (2.0 * X[:, 2] - X[:, 5] + 0.5 * X[:, 8]
          + 0.05 * rng.standard_normal(n))
    # Header row: x0 .. x{p-1}, y
    cols = [f"x{i}" for i in range(p)] + ["y"]
    arr = np.hstack([X, y.reshape(-1, 1)])
    np.savetxt(out, arr, delimiter=",", header=",".join(cols),
                comments="", fmt="%.10g")
    return out


def run_backend(name: str, script: str, csv_path: Path,
                 n_components: int, n_runs: int) -> dict:
    """Run one backend on one dataset; return its parsed JSON record."""
    script_path = SCRIPTS_DIR / script
    if not script_path.exists():
        return {"backend": name, "ok": False,
                 "reason": f"script not found: {script}"}
    env = os.environ.copy()
    env.setdefault("LD_LIBRARY_PATH",
                    str(REPO / "build/dev-release/cpp/src"))
    # Append rather than overwrite so OCTAVE_HOME etc. survive.
    if "OCTAVE_HOME" not in env:
        env["OCTAVE_HOME"] = "/home/delete/miniconda3/envs/pls4all_r"
    env["LD_LIBRARY_PATH"] = ":".join([
        str(REPO / "build/dev-release/cpp/src"),
        "/home/delete/miniconda3/envs/pls4all_r/lib",
        "/home/delete/miniconda3/envs/pls4all_r/lib/octave/10.3.0",
        env.get("LD_LIBRARY_PATH", ""),
    ])
    # Allow Python backends to import pls4all from the source tree
    # without a system-wide install.
    env["PYTHONPATH"] = ":".join([
        str(REPO / "bindings/python/src"),
        env.get("PYTHONPATH", ""),
    ])

    if script.endswith(".py"):
        cmd = [sys.executable, str(script_path), str(csv_path),
                str(n_components), str(n_runs)]
    elif script.endswith(".R"):
        rscript = shutil.which("Rscript") or \
            "/home/delete/miniconda3/envs/pls4all_r/bin/Rscript"
        cmd = [rscript, str(script_path), str(csv_path),
                str(n_components), str(n_runs)]
    elif script.endswith(".m"):
        oct_bin = shutil.which("octave") or \
            "/home/delete/miniconda3/envs/pls4all_r/bin/octave"
        # Octave takes the script as an --eval string; pass args via env.
        env["BENCH_CSV"] = str(csv_path)
        env["BENCH_NCOMP"] = str(n_components)
        env["BENCH_NRUNS"] = str(n_runs)
        m_name = Path(script).stem
        m_dir = str(script_path.parent)
        cmd = [oct_bin, "--no-gui", "--no-history",
                "--eval",
                f"addpath(genpath('{REPO}/bindings/matlab')); "
                f"addpath('{m_dir}'); {m_name}"]
    else:
        return {"backend": name, "ok": False,
                 "reason": f"unsupported extension: {script}"}

    t0 = time.perf_counter()
    try:
        out = subprocess.run(cmd, env=env, capture_output=True, text=True,
                              timeout=600)
    except subprocess.TimeoutExpired:
        return {"backend": name, "ok": False, "reason": "timeout"}
    elapsed = time.perf_counter() - t0
    if out.returncode != 0:
        return {"backend": name, "ok": False, "reason":
                  (out.stderr or out.stdout).strip().splitlines()[-1][:160],
                  "stdout_tail": out.stdout.strip().splitlines()[-3:],
                  "subprocess_s": elapsed}
    # Each backend script prints its JSON record as the LAST line.
    last = out.stdout.strip().splitlines()[-1]
    try:
        rec = json.loads(last)
    except json.JSONDecodeError:
        return {"backend": name, "ok": False,
                 "reason": "non-json output", "tail": last[:160]}
    rec["backend"] = name
    rec["subprocess_s"] = elapsed
    rec.setdefault("ok", True)
    return rec


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sizes", nargs="*",
                         help="dataset sizes 'NxP', e.g. 500x200")
    parser.add_argument("--only", nargs="*",
                         help="restrict to a subset of backends")
    parser.add_argument("--n-components", type=int, default=5)
    parser.add_argument("--n-runs", type=int, default=10)
    parser.add_argument("--no-write", action="store_true")
    args = parser.parse_args()

    sizes = (
        [tuple(int(x) for x in s.split("x")) for s in args.sizes]
        if args.sizes else DEFAULT_SIZES
    )
    if args.only:
        backends = [b for b in BACKENDS if b[0] in args.only]
    else:
        backends = BACKENDS

    all_records: list[dict] = []
    for n, p in sizes:
        csv_path = gen_dataset(n, p)
        print(f"\n=== {n} × {p} (components={args.n_components},"
               f" runs={args.n_runs}) ===")
        for name, script, lang, tier in backends:
            print(f"  {name:14s}: ", end="", flush=True)
            rec = run_backend(name, script, csv_path,
                                args.n_components, args.n_runs)
            rec.update({"n": n, "p": p,
                          "language": lang, "tier": tier})
            all_records.append(rec)
            if rec.get("ok"):
                print(f"median={rec.get('median_ms', float('nan')):.2f}ms"
                       f"  min={rec.get('min_ms', float('nan')):.2f}ms")
            else:
                print(f"FAILED — {rec.get('reason', '?')[:80]}")

    # Aggregate into a CSV + Markdown.
    csv_out = RESULTS_DIR / "timing.csv"
    md_out = RESULTS_DIR / "summary.md"
    with csv_out.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["backend", "language", "tier", "n", "p", "ok",
                      "median_ms", "min_ms", "max_ms", "n_runs", "reason"])
        for r in all_records:
            w.writerow([r.get("backend"), r.get("language"), r.get("tier"),
                          r.get("n"), r.get("p"), r.get("ok"),
                          r.get("median_ms"), r.get("min_ms"),
                          r.get("max_ms"), r.get("n_runs"),
                          r.get("reason", "")])
    print(f"\nCSV written → {csv_out}")

    # Build a comparison markdown — for each size, list backends with
    # median_ms and speedup vs the external implementation in the same
    # language (where applicable).
    by_size: dict[tuple[int, int], list[dict]] = {}
    for r in all_records:
        by_size.setdefault((r["n"], r["p"]), []).append(r)
    with md_out.open("w") as f:
        f.write("# Cross-binding PLS regression timing\n\n")
        for (n, p), recs in by_size.items():
            f.write(f"## n = {n}, p = {p}\n\n")
            f.write("| Backend | Language | Tier | Median (ms) | "
                     "Speedup vs external | OK |\n")
            f.write("|---|---|---|---:|---:|:---:|\n")
            # external per-language baseline
            ext_per_lang = {r["language"]: r for r in recs
                             if r.get("tier") == "external" and r.get("ok")}
            for r in recs:
                med = r.get("median_ms", float("nan"))
                speedup = "—"
                if r.get("ok") and r["language"] in ext_per_lang:
                    ext_med = ext_per_lang[r["language"]]["median_ms"]
                    if ext_med and med:
                        speedup = f"{ext_med / med:.2f}x"
                ok = "✓" if r.get("ok") else "✗"
                med_str = f"{med:.2f}" if r.get("ok") else "—"
                f.write(f"| {r.get('backend')} | {r['language']} | "
                         f"{r['tier']} | {med_str} | {speedup} | {ok} |\n")
            f.write("\n")
    print(f"Markdown → {md_out}")


if __name__ == "__main__":
    main()
