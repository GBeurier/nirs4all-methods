#!/usr/bin/env python3
"""Compare staged AOM/moment campaign variants from result CSVs.

This script is an offline audit helper. It does not fit models, does not select
production candidates, and does not use dataset identity as a routing rule. A
dataset key is used only to pair evaluation rows when comparing a variant to a
baseline.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import math
import statistics
from pathlib import Path
from typing import Iterable


CONFIG_COLUMNS = (
    "plan",
    "stages_json",
    "heads",
    "cv",
    "components",
    "ridge_lambdas",
    "max_chains",
    "chain_chunk_size",
    "top_k",
    "refit_top_k",
    "refit_per_head_top_k",
    "moment_policy",
    "split_head_scoring",
    "max_train_samples",
    "max_features",
    "max_train_feature_product",
    "cuda_pls_parallel_folds",
    "cuda_pls_min_device_features",
    "cuda_pls_many_batched",
    "backend_min_cuda_product",
)
ROUTE_COUNTER_COLUMNS = (
    "n_screen_pls_moment_cv_fits",
    "n_screen_pls_moment_host_cv_fits",
    "n_screen_pls_moment_cuda_device_cv_fits",
    "n_screen_pls_moment_cuda_parallel_fold_batches",
    "n_screen_pls_moment_cuda_parallel_fold_jobs",
    "n_screen_pls_moment_cuda_many_batched_batches",
    "n_screen_pls_moment_cuda_many_batched_jobs",
    "n_refit_pls_moment_cv_fits",
    "n_refit_pls_moment_host_cv_fits",
    "n_refit_pls_moment_cuda_device_cv_fits",
    "n_refit_pls_moment_cuda_parallel_fold_batches",
    "n_refit_pls_moment_cuda_parallel_fold_jobs",
    "n_refit_pls_moment_cuda_many_batched_batches",
    "n_refit_pls_moment_cuda_many_batched_jobs",
)
IDENTITY_COLUMN_TOKENS = (
    "dataset",
    "database",
    "source_name",
    "dataset_id",
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


def parse_input_spec(spec: str) -> tuple[str, Path]:
    label, separator, path = str(spec).partition("=")
    if separator:
        label = label.strip()
        path = path.strip()
    else:
        path = label.strip()
        label = Path(path).stem
    if not label:
        raise ValueError(f"input label is empty in {spec!r}")
    if not path:
        raise ValueError(f"input path is empty in {spec!r}")
    return label, Path(path)


def first_present(row: dict[str, str], names: Iterable[str]) -> str:
    for name in names:
        value = row.get(name)
        if value not in (None, ""):
            return str(value)
    return ""


def dataset_key(row: dict[str, str]) -> str | None:
    explicit = first_present(row, ("dataset_key", "key"))
    if explicit:
        return explicit
    dataset = first_present(row, ("dataset", "dataset_name"))
    database = first_present(row, ("database_name", "dataset_group", "database"))
    if not dataset:
        return None
    if "/" in dataset and not database:
        database, dataset = dataset.split("/", 1)
    return f"{database}/{dataset}" if database else dataset


def normalize_status(row: dict[str, str]) -> str:
    return str(row.get("status", "")).strip().lower()


def row_is_status_ok(row: dict[str, str]) -> bool:
    return normalize_status(row) in {"", "ok", "success", "true", "1"}


def row_is_status_skipped(row: dict[str, str]) -> bool:
    return normalize_status(row) in {"skip", "skipped"}


def normalize_config_value(value: str) -> object:
    text = str(value).strip()
    if text == "":
        return ""
    if text.lower() in {"true", "false"}:
        return text.lower() == "true"
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        return text
    return parsed


def config_for_row(row: dict[str, str], source_label: str) -> dict[str, object]:
    config: dict[str, object] = {"source_label": source_label}
    for column in CONFIG_COLUMNS:
        if column not in row:
            continue
        value = row.get(column)
        if value in (None, ""):
            continue
        config[column] = normalize_config_value(str(value))
    if config.get("stages_json"):
        config["plan"] = "custom"
    return config


def config_key_for_row(row: dict[str, str], source_label: str) -> str:
    config = config_for_row(row, source_label)
    return json.dumps(config, sort_keys=True, separators=(",", ":"))


def config_uses_only_campaign_columns(config_key: str) -> bool:
    """Guard helper for tests: config keys must not embed dataset identity."""
    parsed = json.loads(config_key)
    keys = " ".join(str(key).lower() for key in parsed)
    return not any(token in keys for token in IDENTITY_COLUMN_TOKENS)


def variant_id(config_key: str) -> str:
    digest = hashlib.sha1(config_key.encode("utf-8")).hexdigest()
    return "v_" + digest[:12]


def variant_label(config: dict[str, object]) -> str:
    parts = [str(config.get("source_label", "variant"))]
    plan = config.get("plan")
    if plan:
        parts.append(str(plan))
    stages_json = config.get("stages_json")
    if stages_json:
        parts.append("custom_stages")
    heads = config.get("heads")
    if heads:
        parts.append("heads=" + str(heads))
    max_chains = config.get("max_chains")
    if max_chains:
        parts.append("max_chains=" + str(max_chains))
    return " | ".join(parts)


def median_or_empty(values: list[float]) -> float | str:
    return statistics.median(values) if values else ""


def mean_or_empty(values: list[float]) -> float | str:
    return statistics.fmean(values) if values else ""


def best_scores_by_dataset(
    rows: Iterable[dict[str, object]],
    *,
    score_column: str,
) -> dict[str, float]:
    best: dict[str, float] = {}
    for record in rows:
        row = record["row"]
        if not isinstance(row, dict):
            continue
        if not row_is_status_ok(row):
            continue
        key = dataset_key(row)
        score = parse_float(row.get(score_column))
        if key is None or score is None:
            continue
        current = best.get(key)
        if current is None or score < current:
            best[key] = score
    return best


def load_input_rows(input_specs: Iterable[str]) -> list[dict[str, object]]:
    records: list[dict[str, object]] = []
    for spec in input_specs:
        label, path = parse_input_spec(spec)
        for row in read_csv(path):
            records.append(
                {
                    "source_label": label,
                    "source_path": str(path),
                    "row": row,
                }
            )
    return records


def baseline_from_paths(
    paths: Iterable[str],
    *,
    score_column: str,
) -> dict[str, float]:
    records: list[dict[str, object]] = []
    for path_text in paths:
        path = Path(path_text)
        for row in read_csv(path):
            records.append(
                {"source_label": "baseline", "source_path": str(path), "row": row}
            )
    return best_scores_by_dataset(records, score_column=score_column)


def baseline_from_label(
    records: Iterable[dict[str, object]],
    *,
    label: str,
    score_column: str,
) -> dict[str, float]:
    selected = [
        record for record in records if str(record.get("source_label", "")) == label
    ]
    return best_scores_by_dataset(selected, score_column=score_column)


def summarize_variant_groups(
    records: Iterable[dict[str, object]],
    *,
    score_column: str,
    baseline: dict[str, float] | None = None,
) -> list[dict[str, object]]:
    groups: dict[str, dict[str, object]] = {}
    for record in records:
        row = record["row"]
        if not isinstance(row, dict):
            continue
        source_label = str(record.get("source_label", "variant"))
        key = config_key_for_row(row, source_label)
        config = config_for_row(row, source_label)
        group = groups.setdefault(
            key,
            {
                "config": config,
                "config_key": key,
                "source_paths": set(),
                "rows": [],
                "scores": [],
                "fit_seconds": [],
                "n_rows": 0,
                "n_ok": 0,
                "n_skipped": 0,
                "n_error": 0,
                "route_totals": {column: 0.0 for column in ROUTE_COUNTER_COLUMNS},
            },
        )
        group["source_paths"].add(str(record.get("source_path", "")))
        group["rows"].append(record)
        group["n_rows"] += 1

        score = parse_float(row.get(score_column))
        if row_is_status_ok(row) and score is not None:
            group["n_ok"] += 1
            group["scores"].append(score)
            fit_seconds = parse_float(row.get("fit_time_s"))
            if fit_seconds is not None:
                group["fit_seconds"].append(fit_seconds)
        elif row_is_status_skipped(row):
            group["n_skipped"] += 1
        else:
            group["n_error"] += 1

        route_totals = group["route_totals"]
        for column in ROUTE_COUNTER_COLUMNS:
            value = parse_float(row.get(column))
            if value is not None:
                route_totals[column] += value

    summary_rows: list[dict[str, object]] = []
    for key in sorted(groups):
        group = groups[key]
        config = group["config"]
        scores = group["scores"]
        fit_seconds = group["fit_seconds"]
        route_totals = group["route_totals"]
        out: dict[str, object] = {
            "variant_id": variant_id(key),
            "variant_label": variant_label(config),
            "source_label": str(config.get("source_label", "")),
            "source_path": ";".join(sorted(group["source_paths"])),
            "config_key": key,
            "config_json": key,
            "n_rows": group["n_rows"],
            "n_ok": group["n_ok"],
            "n_skipped": group["n_skipped"],
            "n_error": group["n_error"],
            "median_score": median_or_empty(scores),
            "mean_score": mean_or_empty(scores),
            "median_fit_seconds": median_or_empty(fit_seconds),
            "total_fit_seconds": sum(fit_seconds) if fit_seconds else "",
        }
        for column in ROUTE_COUNTER_COLUMNS:
            value = route_totals[column]
            out[f"total_{column}"] = int(value) if value.is_integer() else value

        if baseline:
            add_baseline_comparison(out, group["rows"], score_column, baseline)
        summary_rows.append(out)
    return summary_rows


def add_baseline_comparison(
    out: dict[str, object],
    records: Iterable[dict[str, object]],
    score_column: str,
    baseline: dict[str, float],
) -> None:
    variant_best = best_scores_by_dataset(records, score_column=score_column)
    ratios: list[float] = []
    wins = 0
    losses = 0
    ties = 0
    paired = 0
    for key, score in variant_best.items():
        base_score = baseline.get(key)
        if base_score is None:
            continue
        paired += 1
        if base_score != 0:
            ratios.append(score / base_score)
        delta = score - base_score
        if abs(delta) <= 1e-12:
            ties += 1
        elif delta < 0:
            wins += 1
        else:
            losses += 1
    out["n_paired_baseline"] = paired
    out["wins_vs_baseline"] = wins
    out["losses_vs_baseline"] = losses
    out["ties_vs_baseline"] = ties
    out["median_ratio_vs_baseline"] = median_or_empty(ratios)
    out["mean_ratio_vs_baseline"] = mean_or_empty(ratios)


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if rows:
        fieldnames = list(rows[0])
    else:
        fieldnames = ["variant_id"]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def format_cell(value: object) -> str:
    if value in ("", None):
        return ""
    if isinstance(value, float):
        return f"{value:.6g}"
    return str(value).replace("|", "/")


def write_markdown(path: Path, rows: list[dict[str, object]]) -> None:
    columns = [
        "variant_id",
        "variant_label",
        "n_ok",
        "n_skipped",
        "median_score",
        "median_fit_seconds",
        "n_paired_baseline",
        "wins_vs_baseline",
        "losses_vs_baseline",
        "ties_vs_baseline",
        "median_ratio_vs_baseline",
    ]
    present = [column for column in columns if any(column in row for row in rows)]
    lines = [
        "# AOM staged variant comparison",
        "",
        "Offline audit summary. Dataset keys are used only for paired evaluation.",
        "",
        "|" + "|".join(present) + "|",
        "|" + "|".join("---" for _ in present) + "|",
    ]
    for row in rows:
        cells = [format_cell(row.get(column, "")) for column in present]
        lines.append("|" + "|".join(cells) + "|")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        action="append",
        required=True,
        help=(
            "Staged result CSV, repeated as LABEL=PATH or PATH. The label is "
            "part of the variant configuration; the dataset key is never used "
            "as a configuration field."
        ),
    )
    parser.add_argument(
        "--score-column",
        default="rmsep",
        help="Score column to summarize; lower is better. Default: rmsep",
    )
    parser.add_argument(
        "--baseline",
        action="append",
        default=[],
        help=(
            "Optional baseline CSV. All OK finite rows are aggregated by "
            "dataset key using the best score."
        ),
    )
    parser.add_argument(
        "--baseline-label",
        default="",
        help=(
            "Use OK finite rows from --input entries with this label as the "
            "baseline. Ignored when --baseline is supplied."
        ),
    )
    parser.add_argument(
        "--baseline-score-column",
        default=None,
        help="Baseline score column. Defaults to --score-column.",
    )
    parser.add_argument(
        "--output",
        default="benchmarks/cross_binding/aom_staged_variant_summary.csv",
    )
    parser.add_argument(
        "--summary-output",
        default="",
        help="Optional markdown summary output path.",
    )
    return parser


def main() -> int:
    parser = build_arg_parser()
    args = parser.parse_args()
    score_column = str(args.score_column)
    baseline_score_column = str(args.baseline_score_column or score_column)
    records = load_input_rows(args.input)
    baseline: dict[str, float] | None = None
    if args.baseline:
        baseline = baseline_from_paths(
            args.baseline,
            score_column=baseline_score_column,
        )
    elif args.baseline_label:
        baseline = baseline_from_label(
            records,
            label=str(args.baseline_label),
            score_column=baseline_score_column,
        )
    rows = summarize_variant_groups(
        records,
        score_column=score_column,
        baseline=baseline,
    )
    write_csv(Path(args.output), rows)
    print(args.output)
    if args.summary_output:
        write_markdown(Path(args.summary_output), rows)
        print(args.summary_output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
