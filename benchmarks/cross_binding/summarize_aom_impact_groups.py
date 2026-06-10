#!/usr/bin/env python3
"""Summarize staged AOM preprocessing impact groups.

This is an offline audit helper for ``run_aom_staged_real_cohort.py
--diagnostics-dir`` outputs. It reads ``impact_groups.csv`` and ranks
preprocessing families/options by train-CV impact diagnostics already produced
by the campaign. It does not fit models and must not be used as a production
dataset-name router.
"""

from __future__ import annotations

import argparse
import csv
import io
import math
import statistics
from collections import Counter, defaultdict
from pathlib import Path


SUMMARY_FIELDS = (
    "group_kind",
    "group",
    "n_rows",
    "n_datasets",
    "dataset_wins",
    "rank1_occurrences",
    "mean_best_rank",
    "median_best_rank",
    "mean_best_score",
    "median_best_score",
    "mean_improvement_vs_identity",
    "selected_head_counts",
    "scale_x_counts",
    "example_datasets",
)


def read_csv(path: Path) -> list[dict[str, str]]:
    text = path.read_text(encoding="utf-8", errors="replace").replace("\x00", "")
    return list(csv.DictReader(io.StringIO(text)))


def parse_float(value) -> float | None:
    try:
        score = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(score):
        return None
    return score


def mean_or_empty(values: list[float]) -> float | str:
    return statistics.fmean(values) if values else ""


def median_or_empty(values: list[float]) -> float | str:
    return statistics.median(values) if values else ""


def counter_label(values: list[str]) -> str:
    counts = Counter(value for value in values if value not in {"", None})
    return ";".join(f"{key}:{counts[key]}" for key in sorted(counts))


def markdown_cell(value: object) -> str:
    return str(value).replace("|", "\\|")


def row_sort_key(row: dict[str, str]) -> tuple[float, float, str]:
    best_rank = parse_float(row.get("best_rank"))
    best_score = parse_float(row.get("best_score"))
    return (
        float("inf") if best_rank is None else best_rank,
        float("inf") if best_score is None else best_score,
        str(row.get("group", "")),
    )


def dataset_winners(rows: list[dict[str, str]]) -> set[tuple[str, str, str]]:
    by_dataset_kind: dict[tuple[str, str], list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        dataset_key = str(row.get("dataset_key", ""))
        group_kind = str(row.get("group_kind", ""))
        if not dataset_key or not group_kind:
            continue
        by_dataset_kind[(dataset_key, group_kind)].append(row)

    winners: set[tuple[str, str, str]] = set()
    for (dataset_key, group_kind), grouped_rows in by_dataset_kind.items():
        best = sorted(grouped_rows, key=row_sort_key)[0]
        winners.add((dataset_key, group_kind, str(best.get("group", ""))))
    return winners


def summarize_impact_rows(rows: list[dict[str, str]]) -> list[dict[str, object]]:
    winners = dataset_winners(rows)
    grouped: dict[tuple[str, str], list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        group_kind = str(row.get("group_kind", ""))
        group = str(row.get("group", ""))
        if not group_kind or not group:
            continue
        grouped[(group_kind, group)].append(row)

    summary = []
    for (group_kind, group), group_rows in grouped.items():
        datasets = sorted({str(row.get("dataset_key", "")) for row in group_rows if row.get("dataset_key")})
        best_ranks = [
            value
            for value in (parse_float(row.get("best_rank")) for row in group_rows)
            if value is not None
        ]
        best_scores = [
            value
            for value in (parse_float(row.get("best_score")) for row in group_rows)
            if value is not None
        ]
        improvements = [
            value
            for value in (
                parse_float(row.get("improvement_vs_identity")) for row in group_rows
            )
            if value is not None
        ]
        dataset_win_count = sum(
            (dataset_key, group_kind, group) in winners for dataset_key in datasets
        )
        rank1_count = sum(
            parse_float(row.get("best_rank")) == 1.0 for row in group_rows
        )
        summary.append(
            {
                "group_kind": group_kind,
                "group": group,
                "n_rows": int(len(group_rows)),
                "n_datasets": int(len(datasets)),
                "dataset_wins": int(dataset_win_count),
                "rank1_occurrences": int(rank1_count),
                "mean_best_rank": mean_or_empty(best_ranks),
                "median_best_rank": median_or_empty(best_ranks),
                "mean_best_score": mean_or_empty(best_scores),
                "median_best_score": median_or_empty(best_scores),
                "mean_improvement_vs_identity": mean_or_empty(improvements),
                "selected_head_counts": counter_label(
                    [str(row.get("selected_head", "")) for row in group_rows]
                ),
                "scale_x_counts": counter_label(
                    [str(row.get("scale_x", "")) for row in group_rows]
                ),
                "example_datasets": ";".join(datasets[:5]),
            }
        )
    summary.sort(
        key=lambda row: (
            str(row["group_kind"]),
            -int(row["dataset_wins"]),
            float(row["mean_best_rank"]) if row["mean_best_rank"] != "" else float("inf"),
            str(row["group"]),
        )
    )
    return summary


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=SUMMARY_FIELDS)
        writer.writeheader()
        writer.writerows({field: row.get(field, "") for field in SUMMARY_FIELDS} for row in rows)


def markdown_summary(rows: list[dict[str, object]], *, top_n: int) -> str:
    lines = ["# AOM impact group summary", ""]
    kinds = sorted({str(row["group_kind"]) for row in rows})
    for kind in kinds:
        subset = [row for row in rows if row["group_kind"] == kind]
        subset.sort(
            key=lambda row: (
                -int(row["dataset_wins"]),
                float(row["mean_best_rank"]) if row["mean_best_rank"] != "" else float("inf"),
                str(row["group"]),
            )
        )
        lines.extend(
            [
                f"## {kind}",
                "",
                "| Group | Datasets | Dataset wins | Rank-1 occurrences | Mean best rank | Mean improvement vs identity |",
                "|---|---:|---:|---:|---:|---:|",
            ]
        )
        for row in subset[:top_n]:
            lines.append(
                "| {group} | {n_datasets} | {dataset_wins} | {rank1_occurrences} | {mean_best_rank} | {mean_improvement_vs_identity} |".format(
                    **{**row, "group": markdown_cell(row["group"])}
                )
            )
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def resolve_impact_groups(args) -> Path:
    if args.impact_groups:
        return Path(args.impact_groups)
    if args.diagnostics_dir:
        return Path(args.diagnostics_dir) / "impact_groups.csv"
    raise ValueError("pass --impact-groups or --diagnostics-dir")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--impact-groups", default="")
    parser.add_argument("--diagnostics-dir", default="")
    parser.add_argument("--output", required=True)
    parser.add_argument("--summary-output", default="")
    parser.add_argument("--top-n", type=int, default=8)
    args = parser.parse_args()

    if int(args.top_n) < 1:
        raise ValueError("--top-n must be positive")
    impact_groups = resolve_impact_groups(args)
    rows = read_csv(impact_groups)
    summary_rows = summarize_impact_rows(rows)
    write_csv(Path(args.output), summary_rows)
    if args.summary_output:
        Path(args.summary_output).parent.mkdir(parents=True, exist_ok=True)
        Path(args.summary_output).write_text(
            markdown_summary(summary_rows, top_n=int(args.top_n)),
            encoding="utf-8",
        )
    print(args.output)
    if args.summary_output:
        print(args.summary_output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
