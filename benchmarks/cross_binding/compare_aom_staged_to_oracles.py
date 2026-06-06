#!/usr/bin/env python3
"""Compare staged AOM/moment results to existing AOM and TabPFN oracle CSVs.

The script is deliberately offline: it does not fit models and it does not
select by dataset name. It normalizes existing benchmark result files, takes the
best reported score per dataset within each source, and writes a joined table.

Default oracle sources are the local paper/lab artifacts:

* AOM-PLS oracle: best row per dataset in paper AOM-PLS seeds012 results.
* AOM-Ridge oracle: best row per dataset in the headline Ridge run.
* TabPFN oracle: best of TabPFN Raw/Opt references when the wide cohort
  reference table is available, with the lab master table as fallback.

Use ``--target`` for one or more CSVs produced by a staged campaign benchmark.
The target score column is auto-detected from common RMSEP/eval/CV names unless
``--target-score-column`` is supplied.
"""

from __future__ import annotations

import argparse
import csv
import io
import math
import statistics
from pathlib import Path
from typing import Iterable


ROOT = Path("/home/delete/nirs4all")
DEFAULT_AOM_PLS = (
    ROOT
    / "nirs4all-aom/benchmarks/runs/scenarios/"
    "paper_aom_aompls_seeds012/results.csv"
)
DEFAULT_AOM_RIDGE = (
    ROOT / "nirs4all-aom/benchmarks/runs/ridge/all54_headline/results.csv"
)
DEFAULT_TABPFN = ROOT / "nirs4all-aom/benchmarks/pls/cohort_regression.csv"
DEFAULT_TABPFN_MASTER = ROOT / "nirs4all-lab/benchmark/results/1_master_results.csv"

SCORE_COLUMNS = (
    "rmsep",
    "RMSEP",
    "eval_rmse",
    "audit_eval_rmse",
    "best_eval_rmse",
    "score_value",
    "best_refit_cv_rmse",
    "refit_cv_rmse",
    "RMSE_MF",
    "RMSECV",
)
LABEL_COLUMNS = (
    "model",
    "canonical_name",
    "variant",
    "result_label",
    "backend",
    "selected_head",
    "plan",
)
WIDE_SCORE_LABELS = {
    "ref_rmse_tabpfn_raw": "TabPFN-Raw",
    "ref_rmse_tabpfn_opt": "TabPFN-opt",
}


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


def first_present(row: dict[str, str], names: Iterable[str]) -> str:
    for name in names:
        value = row.get(name)
        if value not in (None, ""):
            return str(value)
    return ""


def dataset_key(row: dict[str, str]) -> str | None:
    dataset = first_present(row, ("dataset", "dataset_name"))
    database = first_present(row, ("database_name", "dataset_group", "database"))
    if not dataset:
        return None
    if "/" in dataset and not database:
        database, dataset = dataset.split("/", 1)
    if database:
        return f"{database}/{dataset}"
    return dataset


def row_ok(row: dict[str, str]) -> bool:
    status = str(row.get("status", "")).strip().lower()
    return status in {"", "ok", "success", "true", "1"}


def detect_score_column(
    rows: list[dict[str, str]],
    *,
    requested: str | None,
) -> str:
    if requested:
        return requested
    if not rows:
        raise ValueError("cannot detect score column from an empty CSV")
    fieldnames = set(rows[0])
    for name in SCORE_COLUMNS:
        if name in fieldnames:
            return name
    raise ValueError(
        "could not detect score column; pass --target-score-column or "
        "--baseline-score-column"
    )


def label_for(row: dict[str, str], fallback: str) -> str:
    label = first_present(row, LABEL_COLUMNS)
    return label or fallback


def best_by_dataset(
    paths: Iterable[Path],
    *,
    source: str,
    score_column: str | None = None,
    score_columns: tuple[str, ...] = (),
    model_prefixes: tuple[str, ...] = (),
    model_contains: tuple[str, ...] = (),
) -> dict[str, dict[str, object]]:
    best: dict[str, dict[str, object]] = {}
    for path in paths:
        rows = read_csv(path)
        if score_columns:
            columns = tuple(score_columns)
        else:
            columns = (detect_score_column(rows, requested=score_column),)
        for row in rows:
            if not row_ok(row):
                continue
            base_label = label_for(row, source)
            if model_prefixes and not any(
                base_label.startswith(prefix) for prefix in model_prefixes
            ):
                continue
            if model_contains and not any(token in base_label for token in model_contains):
                continue
            key = dataset_key(row)
            if key is None:
                continue
            for column in columns:
                if column not in row:
                    continue
                score = parse_float(row.get(column))
                if score is None:
                    continue
                label = WIDE_SCORE_LABELS.get(column, base_label or column)
                current = best.get(key)
                if current is None or score < float(current["score"]):
                    best[key] = {
                        "dataset_key": key,
                        "source": source,
                        "score": score,
                        "score_column": column,
                        "label": label,
                        "path": str(path),
                    }
    return best


def merge_best_dicts(*sources: dict[str, dict[str, object]]):
    merged: dict[str, dict[str, object]] = {}
    for source in sources:
        for key, row in source.items():
            current = merged.get(key)
            if current is None or float(row["score"]) < float(current["score"]):
                merged[key] = row
    return merged


def tabpfn_best_by_dataset(paths: Iterable[Path]) -> dict[str, dict[str, object]]:
    wide: list[Path] = []
    long: list[Path] = []
    for path in paths:
        rows = read_csv(path)
        fieldnames = set(rows[0]) if rows else set()
        if {"ref_rmse_tabpfn_raw", "ref_rmse_tabpfn_opt"} & fieldnames:
            wide.append(path)
        else:
            long.append(path)
    return merge_best_dicts(
        best_by_dataset(
            wide,
            source="tabpfn_oracle",
            score_columns=("ref_rmse_tabpfn_raw", "ref_rmse_tabpfn_opt"),
        ),
        best_by_dataset(
            long,
            source="tabpfn_oracle",
            score_column="RMSEP",
            model_prefixes=("TabPFN",),
        ),
    )


def merge_sources(
    target: dict[str, dict[str, object]],
    baselines: dict[str, dict[str, dict[str, object]]],
) -> list[dict[str, object]]:
    keys = set(target)
    for rows in baselines.values():
        keys.update(rows)
    merged = []
    for key in sorted(keys):
        row: dict[str, object] = {"dataset_key": key}
        if key in target:
            row["target_score"] = float(target[key]["score"])
            row["target_label"] = str(target[key]["label"])
        else:
            row["target_score"] = ""
            row["target_label"] = ""
        best_name = "target" if key in target else ""
        best_score = float(target[key]["score"]) if key in target else math.inf
        for name, rows in baselines.items():
            entry = rows.get(key)
            if entry is None:
                row[f"{name}_score"] = ""
                row[f"{name}_label"] = ""
                row[f"target_delta_vs_{name}"] = ""
                row[f"target_ratio_vs_{name}"] = ""
                continue
            score = float(entry["score"])
            row[f"{name}_score"] = score
            row[f"{name}_label"] = str(entry["label"])
            if key in target:
                target_score = float(target[key]["score"])
                row[f"target_delta_vs_{name}"] = target_score - score
                row[f"target_ratio_vs_{name}"] = target_score / score
            else:
                row[f"target_delta_vs_{name}"] = ""
                row[f"target_ratio_vs_{name}"] = ""
            if score < best_score:
                best_score = score
                best_name = name
        row["winner"] = best_name
        merged.append(row)
    return merged


def format_score(value: object) -> str:
    score = parse_float(value)
    if score is None:
        return ""
    return f"{score:.6g}"


def format_ratio(value: object) -> str:
    ratio = parse_float(value)
    if ratio is None:
        return ""
    return f"{ratio:.4g}"


def summarize(rows: list[dict[str, object]], baseline_names: list[str]) -> list[str]:
    paired = [row for row in rows if row.get("target_score") != ""]
    lines = [
        "# AOM staged oracle comparison",
        "",
        f"- datasets with target: {len(paired)}",
        f"- datasets in union: {len(rows)}",
    ]
    for name in baseline_names:
        usable = [
            row
            for row in paired
            if row.get(f"{name}_score") not in ("", None)
            and row.get(f"target_ratio_vs_{name}") not in ("", None)
        ]
        if not usable:
            lines.append(f"- {name}: no paired rows")
            continue
        ratios = sorted(float(row[f"target_ratio_vs_{name}"]) for row in usable)
        wins = sum(1 for row in usable if float(row[f"target_delta_vs_{name}"]) < 0)
        median = statistics.median(ratios)
        lines.append(
            f"- {name}: paired={len(usable)}, target_wins={wins}, "
            f"median_ratio={median:.6g}"
        )
    winners: dict[str, int] = {}
    for row in rows:
        winner = str(row.get("winner", ""))
        if winner:
            winners[winner] = winners.get(winner, 0) + 1
    if winners:
        lines.append("")
        lines.append("Winner counts:")
        for name, count in sorted(winners.items(), key=lambda item: (-item[1], item[0])):
            lines.append(f"- {name}: {count}")
    if paired:
        lines.extend(
            [
                "",
                "## Target Paired Rows",
                "",
                "| Dataset | Target | AOM-PLS oracle | ratio | AOM-Ridge oracle | ratio | TabPFN oracle | ratio | Winner |",
                "|---|---:|---:|---:|---:|---:|---:|---:|---|",
            ]
        )
        for row in sorted(paired, key=lambda item: str(item["dataset_key"])):
            lines.append(
                "| "
                + " | ".join(
                    [
                        str(row["dataset_key"]),
                        format_score(row.get("target_score")),
                        format_score(row.get("aom_pls_oracle_score")),
                        format_ratio(row.get("target_ratio_vs_aom_pls_oracle")),
                        format_score(row.get("aom_ridge_oracle_score")),
                        format_ratio(row.get("target_ratio_vs_aom_ridge_oracle")),
                        format_score(row.get("tabpfn_oracle_score")),
                        format_ratio(row.get("target_ratio_vs_tabpfn_oracle")),
                        str(row.get("winner", "")),
                    ]
                )
                + " |"
            )
    return lines


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0]) if rows else ["dataset_key"]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--target",
        action="append",
        default=[],
        help="CSV result file for the staged/moment method under evaluation",
    )
    parser.add_argument("--target-label", default="n4m_staged")
    parser.add_argument("--target-score-column", default=None)
    parser.add_argument(
        "--aom-pls-oracle",
        action="append",
        default=[str(DEFAULT_AOM_PLS)],
        help="AOM-PLS oracle CSV; can be repeated",
    )
    parser.add_argument(
        "--aom-ridge-oracle",
        action="append",
        default=[str(DEFAULT_AOM_RIDGE)],
        help="AOM-Ridge oracle CSV; can be repeated",
    )
    parser.add_argument(
        "--tabpfn-oracle",
        action="append",
        default=[str(DEFAULT_TABPFN), str(DEFAULT_TABPFN_MASTER)],
        help=(
            "TabPFN oracle/master CSV; can be repeated. Wide "
            "ref_rmse_tabpfn_raw/ref_rmse_tabpfn_opt tables and long TabPFN "
            "RMSEP tables are both supported"
        ),
    )
    parser.add_argument(
        "--output",
        default="benchmarks/cross_binding/aom_staged_oracle_comparison.csv",
    )
    parser.add_argument(
        "--summary-output",
        default="benchmarks/cross_binding/aom_staged_oracle_comparison.md",
    )
    args = parser.parse_args()

    target = (
        best_by_dataset(
            [Path(path) for path in args.target],
            source=args.target_label,
            score_column=args.target_score_column,
        )
        if args.target
        else {}
    )
    baselines = {
        "aom_pls_oracle": best_by_dataset(
            [Path(path) for path in args.aom_pls_oracle],
            source="aom_pls_oracle",
            score_column="RMSEP",
            model_contains=("AOM", "POP"),
        ),
        "aom_ridge_oracle": best_by_dataset(
            [Path(path) for path in args.aom_ridge_oracle],
            source="aom_ridge_oracle",
            score_column="rmsep",
            model_prefixes=("AOMRidge",),
        ),
        "tabpfn_oracle": tabpfn_best_by_dataset(
            [Path(path) for path in args.tabpfn_oracle]
        ),
    }
    rows = merge_sources(target, baselines)
    if not rows:
        raise ValueError("no comparable rows found")

    out = Path(args.output)
    write_csv(out, rows)
    summary = summarize(rows, list(baselines))
    summary_out = Path(args.summary_output)
    summary_out.parent.mkdir(parents=True, exist_ok=True)
    summary_out.write_text("\n".join(summary) + "\n", encoding="utf-8")
    print(out)
    print(summary_out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
