#!/usr/bin/env python3
"""Summarize AOM candidate train-CV rank vs offline audit/test rank.

**OFFLINE AUDIT TOOL ONLY.** This script reads per-dataset diagnostics JSON
files produced by ``run_aom_staged_real_cohort.py --diagnostics-dir`` that
include an ``audit`` section. It compares the train-CV selected candidate
rank/score against the best audit/test-rank candidate (the oracle) and writes a
CSV and optional Markdown summary.

Production selection is NEVER changed by this script. The audit data in the
diagnostics files was computed after train-CV selection was complete;
``selection_uses_test_set`` is always ``False`` in all input files. Do not use
the audit ranking to route by dataset name, source or id - the audit is a
post-hoc analysis surface only.

Key output columns:

- ``selected_cv_rank`` - rank of the production winner in train-CV space (1 = best by CV).
- ``selected_test_rank`` - rank of the same candidate among all evaluated candidates on
  the held-out test set (1 = best by test RMSEP).
- ``test_rank_delta`` - ``selected_test_rank - 1``; 0 means the CV winner was also
  the test winner; >0 means that many candidates beat the production winner on test.
- ``oracle_gap_ratio`` - ``(selected_test_rmse - oracle_test_rmse) / oracle_test_rmse``;
  the relative RMSEP gap between the production winner and the best-by-test-rank
  candidate. 0 means the CV winner is also the test oracle.
- ``audit_spearman`` - Spearman rank correlation between CV and test rankings across
  all evaluated candidates; near 1 means CV rank order agrees with test rank order.
- ``selected_chain`` / ``oracle_chain`` - compact preprocessing-chain labels for
  the production CV winner and the offline best-by-test candidate.

Files lacking an ``audit`` section (generated before this feature was added)
are counted but not included in the output - rerun
``run_aom_staged_real_cohort.py --diagnostics-dir`` to populate them.
"""

from __future__ import annotations

import argparse
import csv
import glob as _glob
import json
import math
import statistics
from pathlib import Path
from typing import Any


SUMMARY_FIELDS = (
    "dataset_key",
    "database_name",
    "dataset",
    "selected_head",
    "selected_param",
    "selected_chain",
    "oracle_head",
    "oracle_param",
    "oracle_chain",
    "plan",
    "selected_cv_rmse",
    "selected_cv_rank",
    "selected_test_rmse",
    "selected_test_rank",
    "oracle_test_rmse",
    "oracle_cv_rank",
    "n_audit_candidates",
    "test_rank_delta",
    "oracle_gap_ratio",
    "audit_spearman",
    "audit_top1_recall",
    "audit_top3_recall",
    "audit_top5_recall",
    "selection_uses_test_set",
)


def _parse_float(value: Any) -> float | None:
    try:
        f = float(value)
    except (TypeError, ValueError):
        return None
    return None if not math.isfinite(f) else f


def _parse_int(value: Any) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _format_param(value: Any) -> str:
    parsed = _parse_float(value)
    if parsed is None:
        return "" if value in (None, "") else str(value)
    if abs(parsed - round(parsed)) < 1e-12:
        return str(int(round(parsed)))
    return f"{parsed:.6g}"


def _chain_from_candidate(row: dict[str, Any]) -> Any:
    chain = row.get("chain")
    if chain is not None:
        return chain
    chain_json = row.get("chain_json")
    if chain_json:
        try:
            return json.loads(str(chain_json))
        except json.JSONDecodeError:
            return str(chain_json)
    return None


def _chain_label(chain: Any) -> str:
    if chain in (None, ""):
        return ""
    if isinstance(chain, str):
        return chain
    if not isinstance(chain, list):
        return str(chain)
    parts: list[str] = []
    for op in chain:
        if isinstance(op, (list, tuple)) and op:
            name = str(op[0])
            params = op[1] if len(op) > 1 else []
            if params in (None, [], ()):
                parts.append(name)
            elif isinstance(params, (list, tuple)):
                parts.append(
                    name + "(" + ",".join(_format_param(p) for p in params) + ")"
                )
            else:
                parts.append(name + "(" + _format_param(params) + ")")
        else:
            parts.append(str(op))
    return " -> ".join(parts)


def _topk_recall(rank_diag: dict[str, Any], k: int) -> float | str:
    topk = rank_diag.get("topk")
    if not isinstance(topk, list):
        return ""
    for row in topk:
        if not isinstance(row, dict):
            continue
        if _parse_int(row.get("k")) == int(k):
            value = _parse_float(row.get("eval_top_k_recall"))
            return value if value is not None else ""
    return ""


def load_diagnostics_file(path: Path) -> dict[str, Any] | None:
    try:
        return json.loads(path.read_text(encoding="utf-8", errors="replace"))
    except (OSError, json.JSONDecodeError):
        return None


def extract_row(diag: dict[str, Any]) -> dict[str, Any] | None:
    """Extract one rank-audit row from a diagnostics dict.

    Returns ``None`` when the diagnostics lack an ``audit`` section (e.g.
    files generated before audit-rank persistence was added).
    """
    audit = diag.get("audit")
    if not isinstance(audit, dict):
        return None
    selected_cv = audit.get("selected_cv")
    oracle = audit.get("oracle")
    if selected_cv is None and oracle is None:
        return None

    best = diag.get("best") or {}
    selected_cv_dict = selected_cv if isinstance(selected_cv, dict) else {}
    oracle_dict = oracle if isinstance(oracle, dict) else {}

    selected_test_rmse = _parse_float(selected_cv_dict.get("eval_rmse"))
    selected_test_rank = _parse_int(selected_cv_dict.get("eval_rank"))
    selected_cv_rmse = _parse_float(
        selected_cv_dict.get("refit_cv_rmse", best.get("refit_cv_rmse"))
    )
    selected_cv_rank = _parse_int(
        selected_cv_dict.get("cv_rank", best.get("cv_rank"))
    )
    oracle_test_rmse = _parse_float(oracle_dict.get("eval_rmse"))
    oracle_cv_rank = _parse_int(oracle_dict.get("cv_rank"))
    n_candidates = _parse_int(audit.get("n_candidates"))
    selected_param = _parse_float(selected_cv_dict.get("param"))
    oracle_param = _parse_float(oracle_dict.get("param"))
    selected_chain = _chain_label(_chain_from_candidate(selected_cv_dict))
    oracle_chain = _chain_label(_chain_from_candidate(oracle_dict))

    test_rank_delta: int | str = ""
    if selected_test_rank is not None:
        test_rank_delta = int(selected_test_rank) - 1

    oracle_gap_ratio: float | str = ""
    if (
        selected_test_rmse is not None
        and oracle_test_rmse is not None
        and oracle_test_rmse > 0.0
    ):
        oracle_gap_ratio = (selected_test_rmse - oracle_test_rmse) / oracle_test_rmse

    audit_spearman: float | str = ""
    rank_diag = audit.get("audit_rank_diagnostics")
    if isinstance(rank_diag, dict):
        v = _parse_float(rank_diag.get("spearman_rank_correlation"))
        if v is not None:
            audit_spearman = v
    else:
        rank_diag = {}

    return {
        "dataset_key": str(diag.get("dataset_key", "")),
        "database_name": str(diag.get("database_name", "")),
        "dataset": str(diag.get("dataset", "")),
        "selected_head": str(
            diag.get("selected_head", "") or selected_cv_dict.get("head", "")
        ),
        "selected_param": selected_param if selected_param is not None else "",
        "selected_chain": selected_chain,
        "oracle_head": str(oracle_dict.get("head", "")),
        "oracle_param": oracle_param if oracle_param is not None else "",
        "oracle_chain": oracle_chain,
        "plan": str(diag.get("plan", "")),
        "selected_cv_rmse": selected_cv_rmse if selected_cv_rmse is not None else "",
        "selected_cv_rank": selected_cv_rank if selected_cv_rank is not None else "",
        "selected_test_rmse": selected_test_rmse if selected_test_rmse is not None else "",
        "selected_test_rank": selected_test_rank if selected_test_rank is not None else "",
        "oracle_test_rmse": oracle_test_rmse if oracle_test_rmse is not None else "",
        "oracle_cv_rank": oracle_cv_rank if oracle_cv_rank is not None else "",
        "n_audit_candidates": n_candidates if n_candidates is not None else "",
        "test_rank_delta": test_rank_delta,
        "oracle_gap_ratio": oracle_gap_ratio,
        "audit_spearman": audit_spearman,
        "audit_top1_recall": _topk_recall(rank_diag, 1),
        "audit_top3_recall": _topk_recall(rank_diag, 3),
        "audit_top5_recall": _topk_recall(rank_diag, 5),
        "selection_uses_test_set": bool(diag.get("selection_uses_test_set", False)),
    }


def resolve_diagnostics_files(args) -> list[Path]:
    if args.diagnostics_dir and args.diagnostics_glob:
        raise ValueError(
            "--diagnostics-dir and --diagnostics-glob are mutually exclusive"
        )
    paths: list[Path] = []
    if args.diagnostics_dir:
        paths = sorted(Path(args.diagnostics_dir).glob("*.diagnostics.json"))
    elif args.diagnostics_glob:
        paths = sorted(Path(p) for p in _glob.glob(args.diagnostics_glob))
    if not paths:
        raise ValueError(
            "no diagnostics JSON files found; "
            "pass --diagnostics-dir or --diagnostics-glob"
        )
    return paths


def _mismatch_section(audit_rows: list[dict[str, Any]]) -> list[str]:
    """Aggregate mismatch patterns for rows where CV winner != test oracle.

    Post-hoc audit only. Production selection is unchanged. Groups by
    selected_head->oracle_head and selected_chain->oracle_chain.
    """
    mismatch_rows = [
        r
        for r in audit_rows
        if (isinstance(r.get("test_rank_delta"), int) and r["test_rank_delta"] > 0)
        or (
            isinstance(r.get("oracle_gap_ratio"), float)
            and r["oracle_gap_ratio"] > 0
        )
    ]
    if not mismatch_rows:
        return []

    def _stat_fmt(v: int | float) -> str:
        if isinstance(v, int):
            return str(v)
        return str(int(v)) if float(v) == int(v) else f"{v:.4f}"

    def _agg(group: list[dict[str, Any]]) -> tuple[int, str, str, str, str]:
        count = len(group)
        gaps = [
            r["oracle_gap_ratio"]
            for r in group
            if isinstance(r.get("oracle_gap_ratio"), float)
        ]
        ranks = [
            r["selected_test_rank"]
            for r in group
            if isinstance(r.get("selected_test_rank"), int)
        ]
        deltas = [
            r["test_rank_delta"]
            for r in group
            if isinstance(r.get("test_rank_delta"), int)
        ]
        mean_gap = f"{sum(gaps) / len(gaps):.4f}" if gaps else "NA"
        med_gap = f"{statistics.median(gaps):.4f}" if gaps else "NA"
        med_rank = _stat_fmt(statistics.median(ranks)) if ranks else "NA"
        med_delta = _stat_fmt(statistics.median(deltas)) if deltas else "NA"
        return count, mean_gap, med_gap, med_rank, med_delta

    header = (
        "| Pair | Count | Mean gap ratio | Median gap ratio"
        " | Median test rank | Median rank delta |"
    )
    sep = "|---|---:|---:|---:|---:|---:|"

    lines: list[str] = [
        "",
        "## Mismatch patterns (post-hoc audit only)",
        "",
        "> Rows where the train-CV winner was not the test oracle",
        "> (`test_rank_delta > 0` or `oracle_gap_ratio > 0`).",
        "> **Production selection is unchanged** - post-hoc audit surface only.",
        "> Do not use these patterns to route or select by dataset name, source or id.",
        "",
        "### By head pair",
        "",
        header,
        sep,
    ]
    head_groups: dict[str, list[dict[str, Any]]] = {}
    for r in mismatch_rows:
        key = f"{r.get('selected_head', '')} -> {r.get('oracle_head', '')}"
        head_groups.setdefault(key, []).append(r)
    for key in sorted(head_groups):
        count, mean_gap, med_gap, med_rank, med_delta = _agg(head_groups[key])
        lines.append(
            f"| {key} | {count} | {mean_gap} | {med_gap}"
            f" | {med_rank} | {med_delta} |"
        )

    lines += ["", "### By chain pair", "", header, sep]
    chain_groups: dict[str, list[dict[str, Any]]] = {}
    for r in mismatch_rows:
        key = f"{r.get('selected_chain', '')} -> {r.get('oracle_chain', '')}"
        chain_groups.setdefault(key, []).append(r)
    for key in sorted(chain_groups):
        count, mean_gap, med_gap, med_rank, med_delta = _agg(chain_groups[key])
        lines.append(
            f"| {key} | {count} | {mean_gap} | {med_gap}"
            f" | {med_rank} | {med_delta} |"
        )

    lines.append("")
    return lines


def markdown_summary(
    rows: list[dict[str, Any]],
    *,
    top_n: int,
    n_no_audit: int = 0,
    n_input_files: int | None = None,
) -> str:
    lines = [
        "# AOM rank audit summary",
        "",
        "> **Offline audit only.** Production selection uses train-CV;",
        "> `selection_uses_test_set` is `False` for all rows. Do not use",
        "> audit rankings to route or select by dataset name, source or id.",
        "",
    ]
    audit_rows = [r for r in rows if r.get("selected_test_rank") != ""]
    n_total = len(rows) + int(n_no_audit) if n_input_files is None else n_input_files
    if not audit_rows:
        lines.append("*No rows with audit data found.*")
        if n_no_audit:
            lines.append("")
            lines.append(
                f"*{n_no_audit} file(s) lacked an `audit` section "
                f"(pre-feature diagnostics; rerun campaign with "
                f"`--diagnostics-dir` to populate).*"
            )
        return "\n".join(lines) + "\n"

    def _sort_key(r: dict[str, Any]) -> tuple[float, float, str]:
        delta = r.get("test_rank_delta")
        gap = r.get("oracle_gap_ratio")
        return (
            -float(delta) if isinstance(delta, int) else 0.0,
            -float(gap) if isinstance(gap, float) else 0.0,
            str(r.get("dataset_key", "")),
        )

    audit_rows = sorted(audit_rows, key=_sort_key)

    def _fmt(v: Any) -> str:
        if isinstance(v, float):
            return f"{v:.4f}"
        return str(v) if v != "" else "NA"

    lines += [
        f"Rows with audit data: **{len(audit_rows)}** of {n_total} total.",
        "",
        "| Dataset | Selected | Oracle | CV RMSE | Test RMSE | Test rank | Oracle RMSE | Gap ratio | CV/test Spearman |",
        "|---|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in audit_rows[:top_n]:
        selected = (
            f"{row['selected_head']}:{_format_param(row['selected_param'])}"
            f" {row['selected_chain']}"
        ).strip()
        oracle = (
            f"{row['oracle_head']}:{_format_param(row['oracle_param'])}"
            f" {row['oracle_chain']}"
        ).strip()
        lines.append(
            f"| {row['dataset_key']} | {selected} | {oracle}"
            f" | {_fmt(row['selected_cv_rmse'])} | {_fmt(row['selected_test_rmse'])}"
            f" | {_fmt(row['selected_test_rank'])} | {_fmt(row['oracle_test_rmse'])}"
            f" | {_fmt(row['oracle_gap_ratio'])} | {_fmt(row['audit_spearman'])} |"
        )
    lines.append("")
    if n_no_audit > 0:
        lines.append(
            f"*{n_no_audit} file(s) lacked an `audit` section "
            f"(pre-feature diagnostics; rerun campaign with `--diagnostics-dir` "
            f"to populate).*"
        )
    lines.extend(_mismatch_section(audit_rows))
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--diagnostics-dir",
        default="",
        help="Directory containing *.diagnostics.json files",
    )
    parser.add_argument(
        "--diagnostics-glob",
        default="",
        help="Glob pattern for diagnostics JSON files (mutually exclusive with --diagnostics-dir)",
    )
    parser.add_argument("--output", required=True, help="Output CSV path")
    parser.add_argument(
        "--summary-output",
        default="",
        help="Optional Markdown summary output path",
    )
    parser.add_argument(
        "--top-n",
        type=int,
        default=20,
        help="Maximum rows in Markdown table (default: 20)",
    )
    args = parser.parse_args()

    if int(args.top_n) < 1:
        raise ValueError("--top-n must be positive")

    paths = resolve_diagnostics_files(args)
    rows: list[dict[str, Any]] = []
    n_no_audit = 0
    for path in paths:
        diag = load_diagnostics_file(path)
        if diag is None:
            continue
        row = extract_row(diag)
        if row is None:
            n_no_audit += 1
            continue
        rows.append(row)

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=SUMMARY_FIELDS)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in SUMMARY_FIELDS})

    if args.summary_output:
        summary_path = Path(args.summary_output)
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        summary_path.write_text(
            markdown_summary(
                rows,
                top_n=int(args.top_n),
                n_no_audit=n_no_audit,
                n_input_files=len(paths),
            ),
            encoding="utf-8",
        )
        print(args.summary_output)

    if n_no_audit:
        print(
            f"Note: {n_no_audit} file(s) had no audit section "
            "(pre-feature diagnostics; rerun campaign with --diagnostics-dir "
            "to populate)."
        )
    print(f"Wrote {len(rows)} audit row(s) to {output}")
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
