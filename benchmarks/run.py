"""Run pls4all benchmarks and write deterministic + timing CSVs.

Usage:
    python benchmarks/run.py                            # run AOM-PLS only
    python benchmarks/run.py --benchmark aom_global     # explicit selector
    python benchmarks/run.py --benchmark pls_regression --scale smoke
    python benchmarks/run.py --benchmark all --scale smoke
    python benchmarks/run.py --check                    # gate the committed accuracy CSV

Each benchmark writes three files under benchmarks/results/<benchmark>/:
- accuracy.csv   (gated by --check; numerical deltas only)
- timing.csv     (informational; platform-dependent)
- summary.md     (informational; regenerated each run)

The runner exits non-zero on any case where `accuracy_pass=False`.
"""

from __future__ import annotations

import argparse
import csv
import difflib
import json
import math
import sys
from dataclasses import fields
from io import StringIO
from pathlib import Path

from runners import aom_global, matrix, pls_regression
from runners._harness import (
    AccuracyRecord,
    TimingRecord,
    format_summary_table,
    host_info,
    round_float,
)


REPO_ROOT = Path(__file__).resolve().parent.parent
RESULTS_ROOT = REPO_ROOT / "benchmarks" / "results"


_BENCHMARKS = {
    "aom_global": {
        "module": aom_global,
        "smoke_supported": False,
        "title": "AOM-PLS global selection",
        "kind": "record",
    },
    "pls_regression": {
        "module": pls_regression,
        "smoke_supported": True,
        "title": "PLS regression",
        "kind": "record",
    },
    "matrix": {
        "module": matrix,
        "smoke_supported": True,
        "title": "Comprehensive performance matrix",
        "kind": "matrix",
    },
}


def _round_accuracy(records: list[AccuracyRecord]) -> list[AccuracyRecord]:
    out: list[AccuracyRecord] = []
    for r in records:
        out.append(AccuracyRecord(
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
        ))
    return out


def _round_timing(records: list[TimingRecord]) -> list[TimingRecord]:
    out: list[TimingRecord] = []
    for r in records:
        out.append(TimingRecord(
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
        ))
    return out


def _serialize_records(records: list, record_type) -> str:
    buf = StringIO()
    fieldnames = [f.name for f in fields(record_type)]
    writer = csv.DictWriter(buf, fieldnames=fieldnames, lineterminator="\n")
    writer.writeheader()
    for r in records:
        row = {f.name: getattr(r, f.name) for f in fields(record_type)}
        writer.writerow(row)
    return buf.getvalue()


def _write_summary(benchmark_id: str, title: str,
                   accuracy: list[AccuracyRecord],
                   timing: list[TimingRecord],
                   metadata: dict, summary_md: Path) -> None:
    table = format_summary_table(accuracy)
    info = host_info()
    meta_lines = [f"- `{k}`: `{v}`" for k, v in metadata.items()]
    body = [
        f"# {title} benchmark",
        "",
        "## Metadata",
        "",
        *meta_lines,
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
        "| Case | pls4all (ms, median / min) | reference (ms, median / min) | speedup |",
        "|------|----------------------------|------------------------------|---------|",
    ]
    for r in timing:
        body.append(
            f"| {r.case} | "
            f"{r.pls4all_end_to_end_ms_median:.3f} / {r.pls4all_end_to_end_ms_min:.3f} | "
            f"{r.bench_end_to_end_ms_median:.3f} / {r.bench_end_to_end_ms_min:.3f} | "
            f"{r.speedup_median:.3f} |"
        )
    body.append("")
    summary_md.parent.mkdir(parents=True, exist_ok=True)
    summary_md.write_text("\n".join(body), encoding="utf-8")


def _run_one(benchmark_id: str, *, scale: str, repeats: int,
             check_only: bool) -> int:
    spec = _BENCHMARKS[benchmark_id]
    if spec["kind"] == "matrix":
        return _run_matrix(benchmark_id, scale=scale, repeats=repeats,
                           check_only=check_only)

    module = spec["module"]
    if scale != "smoke" and not spec["smoke_supported"]:
        # AOM uses its own dataset cases — scale is not parametric.
        scale = "default"
    if spec["smoke_supported"]:
        accuracy, timing, metadata = module.run(scale=scale, repeats=repeats)
    else:
        accuracy, timing, metadata = module.run(repeats=repeats)

    accuracy = _round_accuracy(accuracy)
    timing = _round_timing(timing)

    failures = [r for r in accuracy if not r.accuracy_pass]
    if failures:
        print(f"FAIL [{benchmark_id}]: accuracy gate failed:", file=sys.stderr)
        for r in failures:
            print(
                f"  {r.case}: rmse_abs={r.prediction_rmse_abs_delta:.3e} "
                f"rmse_rel={r.prediction_rmse_rel_delta:.3e} "
                f"op_match={r.selected_operator_index_match} "
                f"k_match={r.selected_n_components_match}",
                file=sys.stderr,
            )
        return 1

    bench_dir = RESULTS_ROOT / benchmark_id
    bench_dir.mkdir(parents=True, exist_ok=True)
    accuracy_csv = _serialize_records(accuracy, AccuracyRecord)
    timing_csv = _serialize_records(timing, TimingRecord)
    accuracy_path = bench_dir / "accuracy.csv"
    timing_path = bench_dir / "timing.csv"
    summary_path = bench_dir / "summary.md"

    if check_only:
        committed = accuracy_path.read_text(encoding="utf-8") if accuracy_path.exists() else ""
        if committed != accuracy_csv:
            print(f"FAIL [{benchmark_id}]: accuracy CSV drift at {accuracy_path}",
                  file=sys.stderr)
            for line in difflib.unified_diff(
                committed.splitlines(keepends=True),
                accuracy_csv.splitlines(keepends=True),
                fromfile="committed", tofile="actual",
            ):
                sys.stderr.write(line)
            return 1
        print(f"OK [{benchmark_id}]: accuracy CSV matches "
              f"{accuracy_path} ({len(accuracy)} rows)")
        return 0

    accuracy_path.write_text(accuracy_csv, encoding="utf-8")
    timing_path.write_text(timing_csv, encoding="utf-8")
    _write_summary(benchmark_id, spec["title"], accuracy, timing, metadata,
                   summary_path)
    print(f"[{benchmark_id}] wrote {accuracy_path} ({len(accuracy)} rows)")
    print(f"[{benchmark_id}] wrote {timing_path} ({len(timing)} rows)")
    print(f"[{benchmark_id}] wrote {summary_path}")
    print(format_summary_table(accuracy))
    return 0


def _write_csv_dict_rows(path: Path, rows: list[dict], fieldnames: list[str]) -> str:
    buf = StringIO()
    writer = csv.DictWriter(buf, fieldnames=fieldnames, lineterminator="\n")
    writer.writeheader()
    for row in rows:
        writer.writerow({k: row.get(k, "") for k in fieldnames})
    payload = buf.getvalue()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(payload, encoding="utf-8")
    return payload


def _matrix_summary(metadata: dict, accuracy: list[dict], timing: list[dict]) -> str:
    info = host_info()
    lines = [
        "# Performance matrix",
        "",
        "Algorithm × dataset × language path. Accuracy compares the",
        "pls4all language paths against scikit-learn; timing is",
        "informational and depends on host CPU pinning + BLAS threads.",
        "",
        "## Metadata",
        "",
        f"- pls4all: `{metadata.get('pls4all_version', 'unknown')}`",
        f"- scale: `{metadata.get('scale', 'unknown')}`",
        f"- repeats: `{metadata.get('repeats', '?')}`",
        f"- R available: `{metadata.get('r_available', False)}`",
        f"- CLI available: `{metadata.get('cli_available', False)}`",
        "",
        "Host:",
        f"- python: `{info['python']}`",
        f"- platform: `{info['platform']}`",
        f"- processor: `{info['processor']}`",
        f"- logical cores: `{info['cores_logical']}`",
        "",
        "Thread pinning environment:",
    ]
    for k, v in metadata.get("env", {}).items():
        lines.append(f"- `{k}`: `{v or '(unset)'}`")
    lines += [
        "",
        "## Timing (ms, median across repeats)",
        "",
        "| Case | Algo | Native C++ | Python | R | sklearn |",
        "|------|------|------------|--------|---|---------|",
    ]
    for row in timing:
        native = f"{row['native_cpp_ms_median']}" if math.isfinite(row['native_cpp_ms_median']) else f"({row['native_cpp_status']})"
        py = f"{row['pls4all_python_ms_median']}" if math.isfinite(row['pls4all_python_ms_median']) else f"({row['pls4all_python_status']})"
        r = f"{row['pls4all_r_ms_median']}" if math.isfinite(row['pls4all_r_ms_median']) else f"({row['pls4all_r_status']})"
        sk = f"{row['sklearn_ms_median']}" if math.isfinite(row['sklearn_ms_median']) else f"({row['sklearn_status']})"
        lines.append(
            f"| {row['case']} | {row['algo']} | {native} | {py} | {r} | {sk} |"
        )
    lines += [
        "",
        "## Accuracy gate",
        "",
        "| Case | Algo | Python pass | R pass | RMSE abs (Python vs sklearn) |",
        "|------|------|-------------|--------|------------------------------|",
    ]
    for row in accuracy:
        rp = row['pls4all_r_pass']
        rp_str = 'n/a' if rp is None else ('y' if rp else 'n')
        py_pass = 'y' if row['pls4all_python_pass'] else 'n'
        lines.append(
            f"| {row['case']} | {row['algo']} | {py_pass} | {rp_str} | "
            f"{row['pls4all_python_rmse_abs']} |"
        )
    return "\n".join(lines) + "\n"


def _run_matrix(benchmark_id: str, *, scale: str, repeats: int,
                check_only: bool) -> int:
    import math  # local to avoid polluting module scope
    spec = _BENCHMARKS[benchmark_id]
    out = spec["module"].run(scale=scale, repeats=repeats)
    timing_rows = out["timing"]
    accuracy_rows = out["accuracy"]
    metadata = out["metadata"]

    failures = [r for r in accuracy_rows if not r["accuracy_pass"]]
    if failures:
        print(f"FAIL [{benchmark_id}]: accuracy gate failed:", file=sys.stderr)
        for r in failures:
            print(
                f"  {r['case']}-{r['algo']}: "
                f"python_pass={r['pls4all_python_pass']} "
                f"python_status={r['pls4all_python_status']} "
                f"r_pass={r['pls4all_r_pass']} r_status={r['pls4all_r_status']}",
                file=sys.stderr,
            )
        return 1

    bench_dir = RESULTS_ROOT / benchmark_id
    bench_dir.mkdir(parents=True, exist_ok=True)

    timing_fields = list(timing_rows[0].keys()) if timing_rows else []
    accuracy_fields = list(accuracy_rows[0].keys()) if accuracy_rows else []
    accuracy_path = bench_dir / "accuracy.csv"
    timing_path = bench_dir / "timing.csv"
    summary_path = bench_dir / "summary.md"
    metadata_path = bench_dir / "metadata.json"

    accuracy_csv = StringIO()
    writer = csv.DictWriter(accuracy_csv, fieldnames=accuracy_fields,
                             lineterminator="\n")
    writer.writeheader()
    for row in accuracy_rows:
        writer.writerow({k: row.get(k, "") for k in accuracy_fields})
    accuracy_payload = accuracy_csv.getvalue()

    if check_only:
        committed = (accuracy_path.read_text(encoding="utf-8")
                     if accuracy_path.exists() else "")
        if committed != accuracy_payload:
            print(f"FAIL [{benchmark_id}]: accuracy CSV drift at {accuracy_path}",
                  file=sys.stderr)
            for line in difflib.unified_diff(
                committed.splitlines(keepends=True),
                accuracy_payload.splitlines(keepends=True),
                fromfile="committed", tofile="actual",
            ):
                sys.stderr.write(line)
            return 1
        print(f"OK [{benchmark_id}]: accuracy CSV matches {accuracy_path}")
        return 0

    accuracy_path.write_text(accuracy_payload, encoding="utf-8")
    _write_csv_dict_rows(timing_path, timing_rows, timing_fields)
    metadata_path.write_text(json.dumps(metadata, indent=2,
                                         sort_keys=True) + "\n",
                              encoding="utf-8")
    summary_path.write_text(_matrix_summary(metadata, accuracy_rows,
                                             timing_rows),
                             encoding="utf-8")
    print(f"[{benchmark_id}] wrote {accuracy_path} ({len(accuracy_rows)} rows)")
    print(f"[{benchmark_id}] wrote {timing_path} ({len(timing_rows)} rows)")
    print(f"[{benchmark_id}] wrote {summary_path}")
    print(f"[{benchmark_id}] wrote {metadata_path}")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="run-benchmarks",
                                     description=__doc__)
    parser.add_argument("--benchmark", default="all",
                        choices=list(_BENCHMARKS.keys()) + ["all"],
                        help="Which benchmark to run (default: all).")
    parser.add_argument("--scale", default="smoke",
                        choices=["smoke", "full"],
                        help="Dataset scale for benchmarks that support it.")
    parser.add_argument("--check", action="store_true",
                        help="Assert committed accuracy CSVs match a fresh run.")
    parser.add_argument("--repeats", type=int, default=1,
                        help="Wall-time repeats per case (default 1 for the "
                             "matrix runner; AOM uses 5 by default).")
    args = parser.parse_args(argv)

    if args.benchmark == "all":
        ids = list(_BENCHMARKS.keys())
    else:
        ids = [args.benchmark]

    exit_code = 0
    for bid in ids:
        rc = _run_one(bid, scale=args.scale, repeats=args.repeats,
                      check_only=args.check)
        if rc != 0 and exit_code == 0:
            exit_code = rc
    return exit_code


if __name__ == "__main__":
    sys.exit(main())
