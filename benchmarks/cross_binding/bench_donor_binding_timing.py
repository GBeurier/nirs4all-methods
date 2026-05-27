#!/usr/bin/env python3
"""Donor-operator BINDING timing pipeline (dashboard Layer 2).

Layer 1 (`bench_donor_timing.py` + `n4m_donor_bench`) times every donor op
through the C ABI across the C++ build tiers. This is the binding-side mirror:
for each timeable donor op it times BOTH Python binding tiers —

  * raw       `n4m.python.<fn>`     → dashboard backend `python_tier1`
  * idiomatic `n4m.sklearn.<Class>` → dashboard backend `python_tier2`

— and records the raw↔idiomatic parity (`binding_parity_*`). The op→call
mappings and the C++-matching deterministic inputs live in `donor_ops.py`
(validated bit-exact against the C++ donor bench via `--dump-output`).

Output is a `dashboard_refresh_*_donor_binding.csv` delta in the shared
46-column schema; `build_landing.py` merges it (cell key
`(algorithm, backend, libp4a_build, n, p, threads)`), so the donor rows gain
`pls4all.python` / `pls4all.sklearn` columns alongside the existing C++ tiers.

Timing reuses the dashboard adaptive protocol (`scripts/_common.adaptive_target`
+ `_stats`), but builds inputs OUTSIDE the timed region so the binding cell
measures the op call (construct + ABI + run + reduce), matching how the C++
tool times `run()` only — not synthetic-input generation.

  # all ops, both dashboard sizes, against the blas-omp build
  N4M_LIB_PATH=build/blas-omp/cpp/src/libn4m.so.1.9.0 \
  PYTHONPATH=bindings/python/src \
  python benchmarks/cross_binding/bench_donor_binding_timing.py --force
"""
from __future__ import annotations

import argparse
import csv
import os
import sys
import time
import traceback
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(HERE))                       # donor_ops
sys.path.insert(0, str(HERE / "scripts"))           # _common
sys.path.insert(0, str(REPO / "bindings/python/src"))  # n4m

import donor_ops as D  # noqa: E402
from _common import adaptive_target, _stats, _safe_version  # noqa: E402

DEFAULT_OUT = REPO / "benchmarks/cross_binding/results/dashboard_refresh_2026_05_27_donor_binding.csv"
DEFAULT_SIZES = ["250x50", "50x250"]
DEFAULT_SEED_BASE = 1_234_567_890
WARMUP_ABORT_MS = 5.0 * 60.0 * 1000.0
LIBP4A_BUILD = "blas-omp"  # binding rows collapse onto the blas-omp column

FIELDNAMES = [
    "algorithm", "backend", "language", "tier", "kind", "n", "p", "threads",
    "libp4a_build", "seed_base", "prediction_seed", "ok", "reported_ms",
    "median_ms", "sample_median_ms", "mean_ms", "min_ms", "max_ms", "n_runs",
    "total_runs", "max_runs", "warmup_ms", "warmup_included", "timing_statistic",
    "timing_decision", "parity_ref", "parity_max_diff", "parity_ok", "parity_note",
    "binding_parity_ref", "binding_parity_max_diff", "binding_parity_ok",
    "binding_parity_note", "reference_parity_ref", "reference_parity_rmse_abs",
    "reference_parity_rmse_rel", "reference_parity_ok", "reference_parity_note",
    "reference_parity_tolerance", "reference_oracle_path", "reference_kind",
    "is_canonical_reference", "subprocess_s", "predictions_path", "versions_json",
    "reason",
]

# raw↔idiomatic must be ~bit-exact (same ABI, same seed); values get a tiny
# float tolerance, discrete outputs (masks / split indices) must match exactly.
VALUES_TOL = 1e-9

TIER = {"python_tier1": "tier 1", "python_tier2": "tier 2"}

# Map the spec reason tokens to descriptive `reason` strings that contain a
# phrase build_landing.parity_code() recognises as `not_available` (an honest
# "this binding doesn't provide the op" glyph) rather than a runtime `error`.
REASON_TEXT = {
    D.NO_RAW: "no raw binding fn — n4m.python does not expose this op",
    D.NO_ESTIMATOR: "no sklearn estimator for this op",
    D.SEMANTIC_MISMATCH: "idiomatic wrapper not available (semantic mismatch)",
    D.BINDING_DEFERRED: "binding deferred — not available",
}


def time_op(run_once, max_runs: int, seed_base: int) -> dict:
    """Adaptive timing of `run_once()` (inputs already built in the closure)."""
    t0 = time.perf_counter()
    run_once()
    warmup_ms = (time.perf_counter() - t0) * 1000.0
    if warmup_ms > WARMUP_ABORT_MS or max_runs < 2:
        return _stats([warmup_ms], statistic="single", warmup_ms=warmup_ms,
                      decision="warmup_only", max_runs=max_runs, total_runs=1,
                      prediction_seed=seed_base, warmup_included=True)
    t0 = time.perf_counter()
    run_once()
    probe_ms = (time.perf_counter() - t0) * 1000.0
    samples = [probe_ms]
    target_total, statistic, decision = adaptive_target(probe_ms, max_runs)
    for _ in range(1, max(1, target_total - 1)):
        t0 = time.perf_counter()
        run_once()
        samples.append((time.perf_counter() - t0) * 1000.0)
    return _stats(samples, statistic=statistic, warmup_ms=warmup_ms,
                  decision=decision, max_runs=max_runs, total_runs=1 + len(samples),
                  prediction_seed=seed_base)


def _blank_row(spec: D.DonorOpSpec, backend: str, n: int, p: int,
               seed_base: int, versions: str) -> dict:
    row = {k: "" for k in FIELDNAMES}
    row.update(
        algorithm=spec.dashboard_id, backend=backend, language="Python",
        tier=TIER[backend], kind="pls4all_binding", n=n, p=p, threads=1,
        libp4a_build=LIBP4A_BUILD, seed_base=seed_base, prediction_seed=seed_base,
        versions_json=versions, ok="False",
    )
    return row


def _parity_diff(spec: D.DonorOpSpec, a: np.ndarray, b: np.ndarray) -> tuple[float, bool]:
    if a.shape != b.shape:
        return float("inf"), False
    diff = float(np.max(np.abs(a - b))) if a.size else 0.0
    ok = diff == 0.0 if spec.parity in ("mask", "indices") else diff <= VALUES_TOL
    return diff, ok


def run_cell(spec: D.DonorOpSpec, n: int, p: int, seed_base: int, max_runs: int,
             versions: str) -> list[dict]:
    """Time both binding tiers for one (op, size); return the two rows."""
    inp = D.build_inputs(spec, n, p, seed_base)
    raw_vec = None

    # ---- tier 1: raw n4m.python -----------------------------------------
    r1 = _blank_row(spec, "python_tier1", n, p, seed_base, versions)
    if spec.raw_fn is None:
        r1["reason"] = REASON_TEXT.get(spec.raw_reason, spec.raw_reason)
    else:
        try:
            raw_vec = D.run_raw(spec, inp, seed_base)  # parity reference vector
            stats = time_op(lambda: D.run_raw(spec, inp, seed_base), max_runs, seed_base)
            r1.update({k: stats[k] for k in stats})
            r1["ok"] = "True"
            # raw tier is the in-binding parity reference.
            r1.update(binding_parity_ref="python_tier1", binding_parity_max_diff=0.0,
                      binding_parity_ok="True", binding_parity_note="raw binding (reference)")
        except Exception as exc:
            r1["reason"] = f"{type(exc).__name__}: {exc}"

    # ---- tier 2: idiomatic n4m.sklearn ----------------------------------
    r2 = _blank_row(spec, "python_tier2", n, p, seed_base, versions)
    if spec.idiom_cls is None:
        r2["reason"] = REASON_TEXT.get(spec.idiom_reason, spec.idiom_reason)
    else:
        try:
            idi_vec = D.run_idiom(spec, inp, seed_base)
            stats = time_op(lambda: D.run_idiom(spec, inp, seed_base), max_runs, seed_base)
            r2.update({k: stats[k] for k in stats})
            r2["ok"] = "True"
            if raw_vec is not None:
                diff, ok = _parity_diff(spec, raw_vec, idi_vec)
                r2.update(binding_parity_ref="python_tier1", binding_parity_max_diff=diff,
                          binding_parity_ok=str(ok),
                          binding_parity_note="idiomatic vs raw binding")
            else:
                r2["binding_parity_note"] = "idiomatic-only (no raw binding tier)"
        except Exception as exc:
            r2["reason"] = f"{type(exc).__name__}: {exc}"
    return [r1, r2]


def cell_key(row: dict) -> tuple:
    return (row["algorithm"], row["backend"], row.get("libp4a_build", ""),
            str(row["n"]), str(row["p"]), str(row.get("threads", 1)))


def load_existing(path: Path) -> dict[tuple, dict]:
    if not path.exists():
        return {}
    out = {}
    with path.open(newline="") as f:
        for row in csv.DictReader(f):
            done = str(row.get("ok")).lower() == "true" or bool((row.get("reason") or "").strip())
            if done:
                out[cell_key(row)] = row
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out-csv", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--sizes", nargs="+", default=DEFAULT_SIZES, help="NxP dashboard cells")
    ap.add_argument("--runs", type=int, default=200, help="adaptive cap for ultra-fast ops")
    ap.add_argument("--seed", type=int, default=DEFAULT_SEED_BASE)
    ap.add_argument("--methods", nargs="*", help="bench-op ids to limit to (default all)")
    ap.add_argument("--resume-existing", action="store_true")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    # Pin threading so timings are reproducible and single-threaded.
    for var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
                "BLIS_NUM_THREADS"):
        os.environ.setdefault(var, "1")

    import n4m  # noqa: E402 — load libn4m (respects N4M_LIB_PATH)
    versions = (f'{{"n4m":"{n4m.version()}","numpy":"{_safe_version("numpy")}",'
                f'"libn4m_build":"{LIBP4A_BUILD}","timing_schema":"adaptive-v1"}}')

    sizes = [tuple(int(x) for x in s.lower().split("x")) for s in args.sizes]
    specs = D.all_specs()
    if args.methods:
        wanted = set(args.methods)
        specs = [s for s in specs if s.bench_id in wanted]
    else:
        # Bundled ops (Phase-17 augmenters folded into the fixture's single
        # `aug_phase17` row) have no individual dashboard row; skip them so we
        # don't synthesise spurious algos. They stay in the registry (valid
        # bench ops, covered by the integrity + dump tests).
        specs = [s for s in specs if s.on_dashboard]

    out = {} if args.force else load_existing(args.out_csv)
    timed = skipped = failed = 0
    for spec in specs:
        for n, p in sizes:
            keys = [(spec.dashboard_id, be, LIBP4A_BUILD, str(n), str(p), "1")
                    for be in ("python_tier1", "python_tier2")]
            if not args.force and all(k in out for k in keys):
                skipped += 1
                continue
            try:
                rows = run_cell(spec, n, p, args.seed, args.runs, versions)
            except Exception:
                failed += 1
                print(f"  CELL-FAIL {spec.bench_id} {n}x{p}:\n{traceback.format_exc()}",
                      file=sys.stderr)
                continue
            for row in rows:
                out[cell_key(row)] = row
            t1 = next((r for r in rows if r["backend"] == "python_tier1"), {})
            t2 = next((r for r in rows if r["backend"] == "python_tier2"), {})
            bp = t2.get("binding_parity_max_diff", "")
            print(f"  {spec.dashboard_id:32s} {n}x{p}: "
                  f"raw={t1.get('reported_ms') or t1.get('reason', '—')} "
                  f"idiom={t2.get('reported_ms') or t2.get('reason', '—')} "
                  f"binding_parity_Δ={bp}")
            timed += 1

    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    with args.out_csv.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows(out.values())
    print(f"\nWrote {len(out)} binding rows (cells timed={timed}, skipped={skipped}, "
          f"failed={failed}) to {args.out_csv}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
