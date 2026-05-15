"""Run pls4all benchmarks and write deterministic + timing CSVs.

Usage:
    python benchmarks/run.py                 # Run and write live CSV + summary
    python benchmarks/run.py --check         # Re-run and assert accuracy CSV matches
    python benchmarks/run.py --repeats N     # Set repeat count (default 5)

Determinism contract:
- Accuracy outputs (selected operator, selected n_components, RMSE delta,
  best-score delta) are gated against tight tolerances and committed into
  benchmarks/results/aom_global_accuracy.csv. Drift fails --check.
- Timing outputs (wall-clock per fit+predict) are platform-dependent and
  written to benchmarks/results/aom_global_timing.csv but NOT gated.
- Both files use UTF-8, LF line endings, fixed column order.

The script exits non-zero on any runner exception or accuracy mismatch.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import fields
from pathlib import Path

from runners import aom_global
from runners._harness import (
    AccuracyRecord,
    TimingRecord,
    format_summary_table,
    host_info,
    round_float,
)


REPO_ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = REPO_ROOT / "benchmarks" / "results"
ACCURACY_CSV = RESULTS_DIR / "aom_global_accuracy.csv"
TIMING_CSV = RESULTS_DIR / "aom_global_timing.csv"
SUMMARY_MD = RESULTS_DIR / "aom_global_summary.md"


def _round_accuracy(records: list[AccuracyRecord]) -> list[AccuracyRecord]:
    out: list[AccuracyRecord] = []
    for r in records:
        rounded = AccuracyRecord(
            benchmark=r.benchmark,
            case=r.case,
            n_samples=r.n_samples,
            n_features=r.n_features,
            n_operators=r.n_operators,
            max_components=r.max_components,
            n_folds=r.n_folds,
            selected_operator_index=r.selected_operator_index,
            selected_operator_index_match=bool(r.selected_operator_index_match),
            selected_n_components=r.selected_n_components,
            selected_n_components_match=bool(r.selected_n_components_match),
            best_score_abs_delta=round_float(r.best_score_abs_delta, 12),
            prediction_rmse_abs_delta=round_float(r.prediction_rmse_abs_delta, 12),
            prediction_rmse_rel_delta=round_float(r.prediction_rmse_rel_delta, 12),
            accuracy_pass=bool(r.accuracy_pass),
        )
        out.append(rounded)
    return out


def _round_timing(records: list[TimingRecord]) -> list[TimingRecord]:
    out: list[TimingRecord] = []
    for r in records:
        rounded = TimingRecord(
            benchmark=r.benchmark,
            case=r.case,
            n_samples=r.n_samples,
            n_features=r.n_features,
            n_operators=r.n_operators,
            max_components=r.max_components,
            repeats=r.repeats,
            pls4all_end_to_end_ms_median=round_float(r.pls4all_end_to_end_ms_median, 3),
            pls4all_end_to_end_ms_min=round_float(r.pls4all_end_to_end_ms_min, 3),
            bench_end_to_end_ms_median=round_float(r.bench_end_to_end_ms_median, 3),
            bench_end_to_end_ms_min=round_float(r.bench_end_to_end_ms_min, 3),
            speedup_median=round_float(r.speedup_median, 3),
        )
        out.append(rounded)
    return out


def _read_csv_text(path: Path) -> str:
    return path.read_text(encoding="utf-8") if path.exists() else ""


def _serialize_records(records: list, record_type) -> str:
    import csv
    from io import StringIO
    buf = StringIO()
    fieldnames = [f.name for f in fields(record_type)]
    writer = csv.DictWriter(buf, fieldnames=fieldnames, lineterminator="\n")
    writer.writeheader()
    for r in records:
        row = {f.name: getattr(r, f.name) for f in fields(record_type)}
        writer.writerow(row)
    return buf.getvalue()


def _write_summary(accuracy: list[AccuracyRecord],
                   timing: list[TimingRecord],
                   metadata: dict) -> None:
    table = format_summary_table(accuracy)
    info = host_info()
    body = [
        "# AOM-PLS global selection benchmark",
        "",
        "Deterministic accuracy comparison between the pls4all public C ABI and",
        "the local `nirs4all/bench/AOM_v0/aompls` oracle. Same data, same folds.",
        "",
        f"- `pls4all` version: `{metadata.get('pls4all_version', 'unknown')}`",
        f"- bench oracle version: `{metadata.get('bench_version', 'unknown')}`",
        "",
        "## Accuracy (gated)",
        "",
        "```",
        table.rstrip(),
        "```",
        "",
        "## Timing (informational, NOT gated)",
        "",
        f"- python: `{info['python']}`",
        f"- platform: `{info['platform']}`",
        f"- processor: `{info['processor']}`",
        f"- machine: `{info['machine']}`",
        f"- logical cores: `{info['cores_logical']}`",
        "",
        "Wall-clock median is the wall time of one full `fit + predict` call;",
        "pls4all's number includes Python list <-> ctypes marshalling overhead.",
        "",
        "| Case | pls4all (ms, median / min) | bench (ms, median / min) | speedup |",
        "|------|----------------------------|--------------------------|---------|",
    ]
    for r in timing:
        body.append(
            f"| {r.case} | "
            f"{r.pls4all_end_to_end_ms_median:.3f} / {r.pls4all_end_to_end_ms_min:.3f} | "
            f"{r.bench_end_to_end_ms_median:.3f} / {r.bench_end_to_end_ms_min:.3f} | "
            f"{r.speedup_median:.3f} |"
        )
    body.append("")
    SUMMARY_MD.parent.mkdir(parents=True, exist_ok=True)
    SUMMARY_MD.write_text("\n".join(body), encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="run-benchmarks",
                                     description=__doc__)
    parser.add_argument("--check", action="store_true",
                        help="Assert committed accuracy CSV matches a fresh run.")
    parser.add_argument("--repeats", type=int, default=5,
                        help="Wall-time repeats per case (default 5).")
    args = parser.parse_args(argv)

    accuracy, timing, metadata = aom_global.run(repeats=args.repeats)
    accuracy = _round_accuracy(accuracy)
    timing = _round_timing(timing)

    # Always fail loudly on accuracy regressions.
    failures = [r for r in accuracy if not r.accuracy_pass]
    if failures:
        print("FAIL: accuracy gate failed:", file=sys.stderr)
        for r in failures:
            print(
                f"  {r.case}: rmse_abs={r.prediction_rmse_abs_delta:.3e} "
                f"best_score_abs={r.best_score_abs_delta:.3e} "
                f"op_match={r.selected_operator_index_match} "
                f"k_match={r.selected_n_components_match}",
                file=sys.stderr,
            )
        return 1

    accuracy_csv = _serialize_records(accuracy, AccuracyRecord)
    timing_csv = _serialize_records(timing, TimingRecord)

    if args.check:
        committed = _read_csv_text(ACCURACY_CSV)
        if committed != accuracy_csv:
            print(
                "FAIL: accuracy CSV differs from the committed copy at "
                f"{ACCURACY_CSV}",
                file=sys.stderr,
            )
            # Useful diff hint for the reader.
            import difflib
            for line in difflib.unified_diff(
                committed.splitlines(keepends=True),
                accuracy_csv.splitlines(keepends=True),
                fromfile="committed", tofile="actual",
            ):
                sys.stderr.write(line)
            return 1
        print(f"OK: accuracy CSV matches {ACCURACY_CSV} ({len(accuracy)} rows)")
        return 0

    ACCURACY_CSV.parent.mkdir(parents=True, exist_ok=True)
    ACCURACY_CSV.write_text(accuracy_csv, encoding="utf-8")
    TIMING_CSV.write_text(timing_csv, encoding="utf-8")
    _write_summary(accuracy, timing, metadata)

    print(f"Wrote {ACCURACY_CSV} ({len(accuracy)} rows)")
    print(f"Wrote {TIMING_CSV} ({len(timing)} rows)")
    print(f"Wrote {SUMMARY_MD}")
    print()
    print(format_summary_table(accuracy))
    return 0


if __name__ == "__main__":
    sys.exit(main())
