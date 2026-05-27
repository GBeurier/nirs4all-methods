#!/usr/bin/env python3
"""Cross-binding parity gate — binding predictions vs the C++ raw core.

The library's core promise is **one C ABI, identical numbers across every
shipped binding**. For each `(method, n, p, n_components)` cell this runs the
C++ raw baseline (`bench_cpp.py`) and each language binding driver, then
asserts every binding reproduces the baseline to within strict `rmse_rel <=
1e-12`. Bindings call the *same* libn4m, so agreement is machine-exact; 1e-12
leaves room only for float formatting / accumulation order, never for an
implementation divergence.

It reuses the existing machinery wholesale — `orchestrator.gen_dataset_csv`
(deterministic inputs) + `orchestrator.run_backend` (per-language env,
Rscript/Octave resolution, param marshalling) + the `_comparator` rmse
primitives. It introduces **no** parallel runner and **no** new data.

WHAT THIS PROVES DEPENDS ON WHICH LANGUAGES RAN:
  - The cross-*language* guarantee (R / Octave / JS reproduce the core) is the
    real value, and needs those toolchains + bindings present. Proven locally /
    pre-release at rmse_rel <= 2.5e-15.
  - The `cpp` baseline and `python_tier1` both route through the SAME
    `MethodSpec.pls4all_fn` ctypes path (see bench_python_tier1.py header), so a
    *Python-only* run is NOT a cross-binding proof — it is a harness +
    Python-self-consistency smoke. CI runs exactly that today (R/Octave/JS CI
    builds do not exist yet — see cross-binding-parity.yml), so the CI workflow
    passes `--require python` and is named honestly as a smoke check, not the
    full gate.

Degradation is explicit: a binding whose driver or toolchain is absent is
reported `SKIP` (visible), never silently `PASS`. A `--require`d language must
run and every one of its cells must PASS, else the gate fails.

JS is not yet wired: it has neither a `scripts/bench_js.*` driver nor a
`BINDINGS` entry (the binding is WASM/Emscripten). Wiring it needs both. It is
surfaced as `unwired` so the gap stays honest.

Exit code: 0 iff no binding that ran diverged AND every `--require`d language
ran with all-PASS cells; 1 otherwise.
"""
from __future__ import annotations

import argparse
import os
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

# The C++ raw baseline every binding is compared against.
BASELINE = ("cpp", "bench_cpp.py", "C++", "core")

# (backend, driver script, language label, tier). Raw tier per language: the
# most fundamental "does this binding call the ABI correctly" check. Python's
# idiomatic tier is already gated bit-exact against raw by
# test_donor_binding_specs.py (raw <-> idiomatic), so it is not duplicated here.
BINDINGS = [
    ("python_tier1", "bench_python_tier1.py", "Python", "raw"),
    ("r_tier1", "bench_r_tier1.R", "R", "raw"),
    ("matlab_tier1", "bench_matlab_tier1.m", "Octave", "raw"),
]

# Bindings that exist but have no parity driver yet — reported, not gated.
# Wiring JS needs BOTH a scripts/bench_js.* driver AND a BINDINGS entry below.
UNWIRED = {"js": "needs a scripts/bench_js.* driver + a BINDINGS entry (WASM/Emscripten)"}

# Tiny, cross-language-safe cell set. `pls` and `pcr` are the two methods the
# R `r_pls` reference path also covers, need no x_target/labels sidecars, and
# run in milliseconds. Small n/p keeps cold R/Octave startup the dominant cost.
CELLS = [
    ("pls", 60, 30, 3),
    ("pcr", 60, 30, 3),
]

SEED_BASE = 20240517
PASS, FAIL, SKIP = "PASS", "FAIL", "SKIP"


@dataclass
class Verdict:
    backend: str
    language: str
    tier: str
    cell: str
    status: str
    rmse_rel: float | None
    note: str


def _absent(reason: str) -> bool:
    """A driver/toolchain-absence reason (→ SKIP) vs a real failure (→ FAIL).

    Only the two exact `run_backend` absence reasons qualify: the bench script
    file is missing, or the interpreter (Rscript/Octave) is not on PATH. Any
    other non-OK reason (package not installed, fit error, timeout, non-JSON
    output) is a real FAIL — it must not be silently skipped.
    """
    r = reason.lower()
    return "script not found" in r or "command not found" in r


def evaluate_binding(rec: dict, cpp_pred: np.ndarray, tol: float):
    """Classify one binding record against the C++ baseline predictions.

    Returns (status, rmse_rel, note). Strictly enforces `rmse_rel <= tol`
    (no absolute-rmse escape — bindings calling the same libn4m must match
    relatively). Pure: no subprocess, no I/O beyond the predictions .npy the
    driver already wrote — so it is unit-testable.
    """
    from benchmarks.parity_timing._comparator import rms, rmse

    if not rec.get("ok"):
        reason = str(rec.get("reason", "unknown error"))
        return (SKIP if _absent(reason) else FAIL), None, reason
    pred_path = rec.get("predictions_path")
    if not pred_path or not Path(pred_path).exists():
        return FAIL, None, "ran but emitted no predictions"
    pred = np.asarray(np.load(pred_path), dtype=np.float64)
    ref = np.asarray(cpp_pred, dtype=np.float64)
    if pred.shape != ref.shape:
        if pred.size == ref.size:
            pred = pred.reshape(ref.shape)
        else:
            return FAIL, None, f"shape mismatch ({pred.shape} vs {ref.shape})"
    abs_rmse = rmse(pred, ref)
    ref_rms = rms(ref)
    # Relative when the baseline has scale; fall back to absolute (still strict
    # against the same tol) only when the baseline is essentially zero.
    rmse_rel = abs_rmse / ref_rms if ref_rms > 0 else abs_rmse
    note = f"rmse_rel={rmse_rel:.2e} (tol {tol:.0e})"
    return (PASS if rmse_rel <= tol else FAIL), rmse_rel, note


def gate_exit_code(verdicts: list[Verdict], required_langs: list[str]) -> int:
    """0 iff no binding diverged AND every required language ran with all-PASS.

    A required language must appear in the verdicts and have *every* one of its
    cells PASS — a single SKIP or FAIL (or total absence) of a required
    language fails the gate. Non-required languages may SKIP freely.
    """
    if any(v.status == FAIL for v in verdicts):
        return 1
    by_lang: dict[str, list[str]] = {}
    for v in verdicts:
        by_lang.setdefault(v.language.lower(), []).append(v.status)
    for lang in {x.lower() for x in required_langs}:
        statuses = by_lang.get(lang, [])
        if not statuses or any(s != PASS for s in statuses):
            return 1
    return 0


def run_gate(lib_dir: str, build_key: str, tol: float, threads: int,
             timeout: int, required_langs: list[str]) -> tuple[list[Verdict], int]:
    from benchmarks.cross_binding import orchestrator

    os.environ.setdefault("N4M_R_ENV_NOWARN", "1")
    # Point the orchestrator's build map at the CI/build libn4m directory.
    orchestrator.LIBP4A_BUILDS[build_key] = lib_dir

    verdicts: list[Verdict] = []
    for algo, n, p, nc in CELLS:
        cell = f"{algo} {n}x{p}"
        orchestrator.gen_dataset_csv(n, p, SEED_BASE)
        base = orchestrator.run_backend(
            *BASELINE, "n4m_core", algo, n, p, nc, 1, threads, build_key,
            SEED_BASE, timeout)
        if not base.get("ok") or not base.get("predictions_path"):
            verdicts.append(Verdict("cpp", "C++", "core", cell, FAIL, None,
                                    f"baseline failed: {base.get('reason', '?')}"))
            continue
        cpp_pred = np.load(base["predictions_path"])
        for name, script, lang, tier in BINDINGS:
            rec = orchestrator.run_backend(
                name, script, lang, tier, "pls4all_binding", algo, n, p, nc,
                1, threads, build_key, SEED_BASE, timeout)
            status, rmse_rel, note = evaluate_binding(rec, cpp_pred, tol)
            verdicts.append(Verdict(name, lang, tier, cell, status, rmse_rel, note))

    return verdicts, gate_exit_code(verdicts, required_langs)


def render(verdicts: list[Verdict], required_langs: list[str], code: int) -> str:
    icon = {PASS: "✅", FAIL: "❌", SKIP: "⚠️ "}
    lines = ["", "Cross-binding parity gate (binding vs C++ raw)", "=" * 60]
    for v in verdicts:
        rel = f"{v.rmse_rel:.2e}" if isinstance(v.rmse_rel, float) else "—"
        lines.append(f"{icon.get(v.status, '?')} {v.status:4} "
                     f"{v.backend:14} {v.cell:12} rmse_rel={rel:>9}  {v.note}")
    for lang, why in UNWIRED.items():
        lines.append(f"·   UNWIRED {lang:11} {why}")
    if required_langs:
        lines.append(f"required languages: {', '.join(required_langs)}")
    lines.append("=" * 60)
    lines.append("GATE PASS" if code == 0 else "GATE FAIL")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--lib-dir", required=True,
                    help="directory containing the libn4m.so under test")
    ap.add_argument("--build-key", default="ci",
                    help="orchestrator build label to bind --lib-dir to")
    ap.add_argument("--tol", type=float, default=1e-12,
                    help="rmse_rel tolerance (default 1e-12)")
    ap.add_argument("--threads", type=int, default=1)
    ap.add_argument("--timeout", type=int, default=300,
                    help="per-cell subprocess timeout (s)")
    ap.add_argument("--require", nargs="*", default=[],
                    help="languages that MUST run+pass (else gate fails)")
    args = ap.parse_args(argv)

    lib_dir = str(Path(args.lib_dir).resolve())
    verdicts, code = run_gate(lib_dir, args.build_key, args.tol, args.threads,
                              args.timeout, args.require)
    print(render(verdicts, args.require, code))
    return code


if __name__ == "__main__":
    raise SystemExit(main())
