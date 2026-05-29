#!/usr/bin/env python3
"""Parity-reference dependency doctor.

The cross-binding / per-method parity matrix compares the native C++ engine
against one external *donor* per method. A missing donor dependency does not
fail loudly — it silently leaves a *hole* in the divergence/benchmark matrix.
This script probes every reference dependency the registry can use and reports
exactly what is installed, so "missing value" never means "forgot to install".

Layers probed:
  * Python reference packages (in the active interpreter / parity venv)
  * R reference packages (in ``$N4M_R_ENV`` if set, else the PATH ``Rscript``)
  * the Octave + oct2py + bundled libPLS bridge (used by ``ecr`` / ``cars``)

Usage::

    N4M_R_ENV=/path/to/r-env \
      parity/python_generator/.venv/bin/python parity/scripts/check_references.py
    # add --strict to exit non-zero when any REQUIRED dependency is missing
    # add --json for machine-readable output

Exit code: 0 when every REQUIRED dependency is present (or always, without
``--strict``); 1 when ``--strict`` and a required dependency is missing.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
LIBPLS_DIR = REPO / "bindings" / "octave" / "libPLS_1.95"
# Sibling nirs4all source repo — the live donor for aom_pls/aom_preprocess/pop_pls.
# (The ~104 augmentation/preprocessing donor methods use FROZEN fixtures and do
# not need a live nirs4all at parity time.)
NIRS4ALL_SRC = REPO.parent / "nirs4all"

# --- Python reference packages actually imported by benchmarks/parity_timing/registry.py
PY_REQUIRED = [
    "sklearn", "numpy", "scipy",          # core references + NumPy ports
    "auswahl",                            # RandomFrog / IntervalRandomFrog / VIP_SPA / VISSA
    "pyswarms",                           # BinaryPSO selector
    "tensorly",                           # N-PLS PARAFAC reference
    "ikpls",                              # Improved-Kernel PLS cross-check for `pls`
    "diPLSlib",                           # canonical reference for `di_pls`
]
# oct2py: no longer required — the ecr/cars libPLS bridge now uses a one-shot
# `octave-cli --eval` (a persistent oct2py session stalls on conda Octave-10).
# rpy2: unused (the registry shells out to Rscript).
PY_OPTIONAL = ["oct2py", "rpy2"]

# --- R reference packages used by the registry (mbpls intentionally excluded:
#     the PyPI/CRAN mbpls is unused; mb_pls uses the in-tree nirs4all reference)
R_REQUIRED = [
    "plsVarSel", "enpls", "multiblock", "plsRglm", "plsRcox", "chemometrics",
    "JICO", "ropls", "mixOmics", "sgPLS", "mdatools", "mboost", "softImpute",
    "survival", "survAUC", "rms", "permute", "prospectr", "pls", "spls",
    "OmicsPLS", "kernlab", "corpcor", "genalg", "baseline", "EMSC", "wavelets",
    "plsdepot", "mvdalab", "HDANOVA", "multiway",
]
R_OPTIONAL = ["mbpls"]  # broken against sklearn 1.8 + unused; here for completeness


def _r_env_path() -> dict[str, str]:
    env = dict(os.environ)
    r_env = env.get("N4M_R_ENV")
    if r_env:
        env["PATH"] = f"{r_env}/bin:{env.get('PATH', '')}"
    return env


def probe_r(pkgs: list[str]) -> dict[str, bool]:
    env = _r_env_path()
    rscript = shutil.which("Rscript", path=env.get("PATH"))
    if not rscript:
        return {p: False for p in pkgs}
    expr = (
        "pkgs <- c(" + ",".join(repr(p) for p in pkgs) + "); "
        "cat(paste(as.integer(pkgs %in% installed.packages()[,'Package']), collapse=''))"
    )
    try:
        out = subprocess.run([rscript, "--vanilla", "-e", expr],
                             capture_output=True, text=True, env=env, timeout=90)
        bits = out.stdout.strip()
        if out.returncode == 0 and len(bits) >= len(pkgs):
            return {p: bits[i] == "1" for i, p in enumerate(pkgs)}
    except Exception:
        pass
    return {p: False for p in pkgs}


def probe_python(pkgs: list[str]) -> dict[str, bool]:
    out = {}
    for p in pkgs:
        try:
            out[p] = importlib.util.find_spec(p) is not None
        except Exception:
            out[p] = False
    return out


def probe_octave() -> dict[str, bool]:
    env0 = _r_env_path()
    octave = os.environ.get("OCTAVE_EXECUTABLE", "")
    if not octave or not os.path.exists(octave):
        octave = (shutil.which("octave-cli", path=env0.get("PATH"))
                  or shutil.which("octave", path=env0.get("PATH")) or "")
    # Only an existing absolute executable counts (a bogus $OCTAVE_EXECUTABLE or a
    # bare relative name must not be reported as installed).
    octave = octave if (octave and os.path.exists(octave)) else ""
    res = {
        "octave-cli": bool(octave),
        "libPLS_1.95 (in-tree)": LIBPLS_DIR.exists(),
        "oct2py (optional)": importlib.util.find_spec("oct2py") is not None,
    }
    # End-to-end smoke: run a libPLS fn via the SAME one-shot `octave-cli --eval`
    # path the ecr/cars references now use (a persistent oct2py session stalls on
    # conda Octave-10; the one-shot eval runs in ~0.3s). Bounded by a timeout so a
    # broken Octave can never hang the doctor.
    res["octave+libPLS smoke"] = False
    if octave and res["libPLS_1.95 (in-tree)"]:
        prefix = os.path.dirname(os.path.dirname(octave))
        env = dict(os.environ)
        env.setdefault("OCTAVE_EXECUTABLE", octave)
        env.setdefault("OCTAVE_HOME", prefix)  # conda/relocatable Octave needs this
        libdir = os.path.join(prefix, "lib")
        if os.path.isdir(libdir):
            env["LD_LIBRARY_PATH"] = libdir + ":" + env.get("LD_LIBRARY_PATH", "")
        timeout = int(os.environ.get("N4M_OCTAVE_SMOKE_TIMEOUT", "60"))
        script = (f"addpath('{str(LIBPLS_DIR).replace(chr(39), chr(39) * 2)}'); "
                  "x = pretreat(magic(5), 'center'); printf('SMOKE_OK');")
        try:
            out = subprocess.run([octave, "--norc", "--quiet", "--eval", script],
                                 capture_output=True, text=True, env=env,
                                 timeout=timeout)
            res["octave+libPLS smoke"] = "SMOKE_OK" in out.stdout
        except subprocess.TimeoutExpired:
            res["octave+libPLS smoke"] = False
        except Exception:
            res["octave+libPLS smoke"] = False
    return res


def probe_nirs4all() -> dict[str, bool]:
    """The live nirs4all donor (aom_pls / aom_preprocess / pop_pls) needs a
    Python >=3.11 interpreter with nirs4all + its (non-DL) deps. Configure it via
    N4M_NIRS4ALL_PYTHON; otherwise the current interpreter is tried."""
    py = os.environ.get("N4M_NIRS4ALL_PYTHON") or sys.executable
    env = dict(os.environ)
    if NIRS4ALL_SRC.exists():
        env["PYTHONPATH"] = str(NIRS4ALL_SRC) + os.pathsep + env.get("PYTHONPATH", "")
    ok = False
    resolved = py if os.path.exists(py) else (shutil.which(py) or "")
    if resolved:
        try:
            out = subprocess.run([resolved, "-c", "import nirs4all"],
                                 capture_output=True, text=True, env=env, timeout=180)
            ok = out.returncode == 0
        except Exception:
            ok = False
    return {"nirs4all reference (import nirs4all)": ok}


def _fmt(name: str, ok: bool, optional: bool) -> str:
    mark = "OK " if ok else ("opt" if optional else "MISSING")
    glyph = "✓" if ok else ("·" if optional else "✗")
    return f"  {glyph} {mark:7s} {name}"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--strict", action="store_true",
                    help="exit 1 if any REQUIRED dependency is missing")
    ap.add_argument("--require-octave", action="store_true",
                    help="also require the octave+libPLS smoke to pass (ecr/cars donors "
                         "actually run, not just installed). Tune N4M_OCTAVE_SMOKE_TIMEOUT.")
    ap.add_argument("--require-nirs4all", action="store_true",
                    help="also require the live nirs4all donor to import (aom_pls / "
                         "aom_preprocess / pop_pls). Configure N4M_NIRS4ALL_PYTHON (>=3.11).")
    ap.add_argument("--json", action="store_true", help="machine-readable output")
    args = ap.parse_args()

    py = probe_python(PY_REQUIRED + PY_OPTIONAL)
    r = probe_r(R_REQUIRED + R_OPTIONAL)
    oc = probe_octave()
    n4 = probe_nirs4all()

    octave_required = ("octave-cli", "libPLS_1.95 (in-tree)")
    if args.require_octave:
        octave_required = octave_required + ("octave+libPLS smoke",)
    missing_required = (
        [p for p in PY_REQUIRED if not py[p]]
        + [p for p in R_REQUIRED if not r[p]]
        + [k for k in octave_required if not oc[k]]
        + ([k for k, v in n4.items() if not v] if args.require_nirs4all else [])
    )

    if args.json:
        print(json.dumps({"python": py, "r": r, "octave": oc, "nirs4all": n4,
                          "missing_required": missing_required}, indent=2))
    else:
        print(f"Parity reference doctor  (interpreter: {sys.executable})")
        print(f"  N4M_R_ENV = {os.environ.get('N4M_R_ENV', '(unset — using PATH Rscript)')}\n")
        print("Python reference packages:")
        for p in PY_REQUIRED:
            print(_fmt(p, py[p], optional=False))
        for p in PY_OPTIONAL:
            print(_fmt(p, py[p], optional=True))
        print("\nR reference packages:")
        for p in R_REQUIRED:
            print(_fmt(p, r[p], optional=False))
        for p in R_OPTIONAL:
            print(_fmt(p, r[p], optional=True))
        print("\nOctave / libPLS bridge (ecr, cars):")
        for k, v in oc.items():
            optional = (k == "oct2py (optional)") or (
                k == "octave+libPLS smoke" and not args.require_octave)
            print(_fmt(k, v, optional=optional))
        print("\nnirs4all donor (aom_pls / aom_preprocess / pop_pls):")
        print(f"  N4M_NIRS4ALL_PYTHON = "
              f"{os.environ.get('N4M_NIRS4ALL_PYTHON', '(unset — using ' + sys.executable + ')')}")
        for k, v in n4.items():
            print(_fmt(k, v, optional=not args.require_nirs4all))
        print()
        if missing_required:
            print(f"INCOMPLETE — {len(missing_required)} required reference dep(s) missing: "
                  f"{', '.join(missing_required)}")
            print("Run: scripts/setup_parity_references.sh")
        else:
            print("COMPLETE — every required reference dependency is installed. "
                  "The matrix can be filled with no install-caused holes.")

    return 1 if (args.strict and missing_required) else 0


if __name__ == "__main__":
    raise SystemExit(main())
