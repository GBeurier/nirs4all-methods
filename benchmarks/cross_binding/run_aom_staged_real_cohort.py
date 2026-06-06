#!/usr/bin/env python3
"""Run ``n4m.aom_staged_chain_campaign`` on real NIRS train/test splits.

This is the lightweight production-selection benchmark runner for the staged
AOM/moment workflow. It loads the fixed train/test CSV splits used by the local
NIRS benchmark artifacts, runs train-CV selection only, and evaluates the
selected candidate on the held-out test split through the campaign's audit path.

The output CSV is intentionally compatible with
``compare_aom_staged_to_oracles.py``: its main score column is ``rmsep`` and it
contains ``database_name`` + ``dataset`` keys.
"""

from __future__ import annotations

import argparse
import csv
import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

import n4m


DEFAULT_COHORT = (
    Path("/home/delete/nirs4all")
    / "nirs4all-aom/benchmarks/runs/ridge/diverse11_cohort.csv"
)
DEFAULT_DATA_ROOT = Path("/home/delete/nirs4all/nirs4all-data/regression")
DEFAULT_OUTPUT = Path("benchmarks/cross_binding/aom_staged_real_cohort_results.csv")

RESULT_FIELDS = (
    "schema_version",
    "preset",
    "cohort",
    "dataset_key",
    "database_name",
    "dataset",
    "task",
    "canonical_name",
    "model_class",
    "module",
    "selection",
    "seed",
    "status",
    "error_message",
    "n_train",
    "n_test",
    "n_features",
    "rmsep",
    "audit_oracle_rmse",
    "audit_oracle_head",
    "audit_oracle_param",
    "r2",
    "score_metric",
    "score_value",
    "lower_is_better",
    "fit_time_s",
    "predict_time_s",
    "screen_complete",
    "n_remaining_stage_chunks_total",
    "n_screen_candidates_total",
    "n_screen_split_head_chunks",
    "n_screen_chunk_score_calls",
    "n_ridge_moment_cv_fits",
    "n_ridge_moment_eigen_path_preparations",
    "n_ridge_moment_eigen_path_cv_fits",
    "n_ridge_moment_direct_cv_fits",
    "n_ridge_moment_score_batch_calls",
    "n_ridge_moment_score_batch_jobs",
    "n_refit_candidates",
    "n_screen_pls_moment_cv_fits",
    "n_screen_pls_moment_host_cv_fits",
    "n_screen_pls_moment_cuda_device_cv_fits",
    "n_screen_pls_moment_cuda_parallel_fold_batches",
    "n_screen_pls_moment_cuda_parallel_fold_jobs",
    "n_screen_pls_moment_cuda_many_batched_batches",
    "n_screen_pls_moment_cuda_many_batched_jobs",
    "n_screen_pls_moment_score_batch_calls",
    "n_screen_pls_moment_score_batch_jobs",
    "n_refit_pls_moment_cv_fits",
    "n_refit_pls_moment_host_cv_fits",
    "n_refit_pls_moment_cuda_device_cv_fits",
    "n_refit_pls_moment_cuda_parallel_fold_batches",
    "n_refit_pls_moment_cuda_parallel_fold_jobs",
    "n_refit_pls_moment_cuda_many_batched_batches",
    "n_refit_pls_moment_cuda_many_batched_jobs",
    "n_refit_pls_moment_score_batch_calls",
    "n_refit_pls_moment_score_batch_jobs",
    "selected_head",
    "selected_param",
    "selected_campaign_stage",
    "best_refit_cv_rmse",
    "best_screen_cv_rmse",
    "best_cv_rank",
    "best_screen_rank",
    "rank_spearman",
    "impact_best_score",
    "selection_uses_test_set",
    "plan",
    "stages_json",
    "heads",
    "max_chains",
    "chain_chunk_size",
    "top_k",
    "refit_top_k",
    "refit_per_head_top_k",
    "moment_policy",
    "split_head_scoring",
    "scale_x",
    "scale_x_values",
    "selected_model_config_id",
    "cuda_pls_parallel_folds",
    "cuda_pls_min_device_features",
    "cuda_pls_many_batched",
    "backend_min_cuda_product",
    "checkpoint_dir",
    "library_path",
    "abi",
    "started_at",
    "ended_at",
    "notes",
)

IMPACT_GROUP_FIELDS = (
    "dataset_key",
    "database_name",
    "dataset",
    "selected_head",
    "plan",
    "heads",
    "group_kind",
    "group",
    "n_candidates",
    "best_score",
    "mean_score",
    "median_score",
    "best_rank",
    "mean_rank",
    "improvement_vs_identity",
    "selected_model_config_id",
    "scale_x",
    "selection_uses_test_set",
)

IMPACT_GROUP_KINDS = (
    "by_operator",
    "by_stage_family",
    "by_stage_option",
    "by_head_stage_option",
)

REPORT_COUNTER_KEYS = (
    "screen_complete",
    "n_remaining_stage_chunks_total",
    "n_screen_candidates_total",
    "n_screen_split_head_chunks",
    "n_screen_chunk_score_calls",
    "n_refit_candidates",
    "n_screen_pls_moment_cv_fits",
    "n_screen_pls_moment_host_cv_fits",
    "n_screen_pls_moment_cuda_device_cv_fits",
    "n_screen_pls_moment_cuda_parallel_fold_batches",
    "n_screen_pls_moment_cuda_parallel_fold_jobs",
    "n_screen_pls_moment_cuda_many_batched_batches",
    "n_screen_pls_moment_cuda_many_batched_jobs",
    "n_screen_pls_moment_score_batch_calls",
    "n_screen_pls_moment_score_batch_jobs",
    "n_refit_pls_moment_cv_fits",
    "n_refit_pls_moment_host_cv_fits",
    "n_refit_pls_moment_cuda_device_cv_fits",
    "n_refit_pls_moment_cuda_parallel_fold_batches",
    "n_refit_pls_moment_cuda_parallel_fold_jobs",
    "n_refit_pls_moment_cuda_many_batched_batches",
    "n_refit_pls_moment_cuda_many_batched_jobs",
    "n_refit_pls_moment_score_batch_calls",
    "n_refit_pls_moment_score_batch_jobs",
    "n_ridge_moment_cv_fits",
    "n_ridge_moment_eigen_path_preparations",
    "n_ridge_moment_eigen_path_cv_fits",
    "n_ridge_moment_direct_cv_fits",
    "n_ridge_moment_score_batch_calls",
    "n_ridge_moment_score_batch_jobs",
)


def parse_str_list(text: str) -> tuple[str, ...]:
    values = tuple(part.strip() for part in str(text).split(",") if part.strip())
    if not values:
        raise ValueError("list must contain at least one entry")
    return values


def parse_int_list(text: str) -> tuple[int, ...]:
    values = tuple(int(part.strip()) for part in str(text).split(",") if part.strip())
    if not values:
        raise ValueError("list must contain at least one integer")
    if any(value < 1 for value in values):
        raise ValueError("integer list values must be positive")
    return values


def parse_float_list(text: str) -> tuple[float, ...]:
    values = tuple(float(part.strip()) for part in str(text).split(",") if part.strip())
    if not values:
        raise ValueError("list must contain at least one float")
    return values


def parse_optional_bool_list(text: str) -> tuple[bool | None, ...]:
    values = []
    for part in str(text).split(","):
        token = part.strip().lower()
        if not token:
            continue
        if token in {"none", "null", "default"}:
            values.append(None)
        elif token in {"true", "1", "yes", "on"}:
            values.append(True)
        elif token in {"false", "0", "no", "off"}:
            values.append(False)
        else:
            raise ValueError(f"invalid boolean grid value {part!r}")
    if not values:
        raise ValueError("boolean grid must contain at least one value")
    return tuple(values)


def load_stages_config(args) -> list[Any] | None:
    inline = str(getattr(args, "stages_json", "") or "").strip()
    file_name = str(getattr(args, "stages_json_file", "") or "").strip()
    if inline and file_name:
        raise ValueError("--stages-json and --stages-json-file are mutually exclusive")
    if file_name:
        text = Path(file_name).read_text(encoding="utf-8")
    elif inline:
        text = inline
    else:
        return None

    try:
        value = json.loads(text)
    except json.JSONDecodeError as exc:
        raise ValueError(f"invalid staged campaign JSON: {exc}") from exc
    if not isinstance(value, list) or not value:
        raise ValueError("staged campaign JSON must be a non-empty list")
    for index, item in enumerate(value):
        if not isinstance(item, (str, dict)):
            raise ValueError(
                "each staged campaign JSON item must be a profile string or object; "
                f"got {type(item).__name__} at index {index}"
            )
    return value


def stages_json_label(args) -> str:
    stages = getattr(args, "stages", None)
    if stages is None:
        return ""
    return json.dumps(stages, sort_keys=True, separators=(",", ":"))


def read_semicolon_array(path: Path, *, squeeze: bool = False) -> np.ndarray:
    arr = np.genfromtxt(path, delimiter=";", skip_header=1, dtype=np.float64)
    if arr.ndim == 0:
        arr = arr.reshape(1, 1)
    elif arr.ndim == 1 and not squeeze:
        arr = arr.reshape(1, -1)
    if squeeze and arr.ndim == 2 and arr.shape[1] == 1:
        arr = arr.reshape(-1)
    return np.asarray(arr, dtype=np.float64)


def load_split(data_root: Path, database_name: str, dataset: str):
    path = data_root / database_name / dataset
    required = {
        "X_train": path / "Xtrain.csv",
        "X_test": path / "Xtest.csv",
        "y_train": path / "Ytrain.csv",
        "y_test": path / "Ytest.csv",
    }
    missing = [str(file) for file in required.values() if not file.is_file()]
    if missing:
        raise FileNotFoundError("missing split files: " + ", ".join(missing))
    X_train = read_semicolon_array(required["X_train"])
    X_test = read_semicolon_array(required["X_test"])
    y_train = read_semicolon_array(required["y_train"], squeeze=True)
    y_test = read_semicolon_array(required["y_test"], squeeze=True)
    return X_train, y_train, X_test, y_test, path


def load_cohort(path: Path, *, limit: int | None, datasets: set[str] | None):
    rows = []
    with path.open(newline="", encoding="utf-8", errors="replace") as f:
        for row in csv.DictReader(f):
            status = str(row.get("status", "ok")).strip().lower()
            if status and status != "ok":
                continue
            database = str(row.get("database_name", "")).strip()
            dataset = str(row.get("dataset", "")).strip()
            if not database or not dataset:
                continue
            key = f"{database}/{dataset}"
            if datasets is not None and key not in datasets and dataset not in datasets:
                continue
            rows.append({"database_name": database, "dataset": dataset, "key": key})
            if limit is not None and len(rows) >= int(limit):
                break
    if not rows:
        raise ValueError(f"cohort {path} produced no runnable rows")
    return rows


def load_completed(path: Path) -> set[str]:
    if not path.is_file():
        return set()
    completed = set()
    with path.open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if str(row.get("status", "")).strip().lower() == "ok":
                completed.add(str(row.get("dataset_key", "")))
    return completed


def append_row(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.is_file()
    with path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=RESULT_FIELDS)
        if not exists:
            writer.writeheader()
        writer.writerow({field: row.get(field, "") for field in RESULT_FIELDS})


def safe_dataset_filename(dataset_key: str) -> str:
    safe = "".join(
        char if char.isalnum() or char in {"-", "_", "."} else "_"
        for char in str(dataset_key)
    ).strip("._")
    return safe or "dataset"


def _json_default(value: Any):
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    raise TypeError(f"object of type {type(value).__name__} is not JSON serializable")


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, default=_json_default) + "\n",
        encoding="utf-8",
    )


def _compact_candidate_row(row: Any) -> Any:
    """Strip prediction arrays from a candidate row for compact JSON storage."""
    if not isinstance(row, dict):
        return row
    compact = {}
    for key, value in row.items():
        key_lower = str(key).lower()
        if isinstance(value, np.ndarray):
            continue
        if (
            "prediction" in key_lower
            or key_lower in {"predictions", "y_pred", "yhat", "y_hat"}
        ):
            continue
        compact[key] = value
    return compact


def _compact_audit_payload(audit: dict[str, Any]) -> dict[str, Any]:
    """Build a compact audit summary from the campaign audit report.

    Strips ``eval.rows`` (all-candidate list) and any prediction arrays to
    keep the diagnostics JSON reasonably small. The ``audit_only`` flag is
    preserved so consumers know this data is offline audit only;
    ``selection_uses_test_set`` in the parent diagnostics payload remains
    ``False``.
    """
    compact: dict[str, Any] = {
        "audit_only": bool(audit.get("audit_only", True)),
    }
    n_candidates = audit.get("n_candidates")
    if n_candidates is not None:
        compact["n_candidates"] = int(n_candidates)
    note = audit.get("note")
    if note:
        compact["note"] = str(note)
    eval_report = audit.get("eval")
    if isinstance(eval_report, dict):
        best_cv = eval_report.get("best_cv")
        if best_cv is not None:
            compact["selected_cv"] = _compact_candidate_row(best_cv)
    oracle = audit.get("best_eval")
    if oracle is not None:
        compact["oracle"] = _compact_candidate_row(oracle)
    rank_diag = audit.get("rank_diagnostics")
    if rank_diag is not None:
        compact["audit_rank_diagnostics"] = rank_diag
    return compact


def diagnostics_payload(
    *,
    report: dict[str, Any],
    row: dict[str, Any],
    args,
) -> dict[str, Any]:
    counters = {
        key: report[key]
        for key in REPORT_COUNTER_KEYS
        if key in report
    }
    payload: dict[str, Any] = {
        "dataset_key": row.get("dataset_key", ""),
        "database_name": row.get("database_name", ""),
        "dataset": row.get("dataset", ""),
        "selection_uses_test_set": bool(row.get("selection_uses_test_set", False)),
        "plan": row.get("plan", ""),
        "heads": row.get("heads", ""),
        "scale_x": row.get("scale_x", ""),
        "scale_x_values": row.get("scale_x_values", ""),
        "selected_model_config_id": row.get("selected_model_config_id", ""),
        "selected_head": row.get("selected_head", ""),
        "selected_param": row.get("selected_param", ""),
        "best": report.get("best"),
        "impact": report.get("impact"),
        "rank_diagnostics": report.get("rank_diagnostics"),
        "model_config_summaries": report.get("model_config_summaries"),
        "selected_model_config": report.get("selected_model_config"),
        "route_summary": report.get("route_summary"),
        "counters": counters,
        "runner": {
            "cohort": str(Path(args.cohort_csv).name),
            "cv": int(args.cv),
            "max_chains": (
                None if args.max_chains is None else int(args.max_chains)
            ),
            "top_k": int(args.top_k),
            "refit_top_k": (
                None if args.refit_top_k is None else int(args.refit_top_k)
            ),
            "refit_per_head_top_k": (
                None
                if args.refit_per_head_top_k is None
                else int(args.refit_per_head_top_k)
            ),
            "split_head_scoring": str(args.split_head_scoring),
            "cuda_pls_parallel_folds": bool(args.cuda_pls_parallel_folds),
            "cuda_pls_min_device_features": (
                None
                if args.cuda_pls_min_device_features is None
                else int(args.cuda_pls_min_device_features)
            ),
            "cuda_pls_many_batched": bool(args.cuda_pls_many_batched),
            "backend_min_cuda_product": (
                None
                if args.backend_min_cuda_product is None
                else int(args.backend_min_cuda_product)
            ),
        },
    }
    audit = report.get("audit")
    if isinstance(audit, dict):
        payload["audit"] = _compact_audit_payload(audit)
    return payload


def append_impact_groups(
    path: Path,
    *,
    report: dict[str, Any],
    row: dict[str, Any],
) -> None:
    impact = report.get("impact")
    if not isinstance(impact, dict):
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.is_file()
    with path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=IMPACT_GROUP_FIELDS)
        if not exists:
            writer.writeheader()
        for group_kind in IMPACT_GROUP_KINDS:
            groups = impact.get(group_kind) or ()
            if not isinstance(groups, (list, tuple)):
                continue
            for group in groups:
                if not isinstance(group, dict):
                    continue
                writer.writerow(
                    {
                        "dataset_key": row.get("dataset_key", ""),
                        "database_name": row.get("database_name", ""),
                        "dataset": row.get("dataset", ""),
                        "selected_head": row.get("selected_head", ""),
                        "plan": row.get("plan", ""),
                        "heads": row.get("heads", ""),
                        "group_kind": group_kind,
                        "group": group.get("group", ""),
                        "n_candidates": group.get("n_candidates", ""),
                        "best_score": group.get("best_score", ""),
                        "mean_score": group.get("mean_score", ""),
                        "median_score": group.get("median_score", ""),
                        "best_rank": group.get("best_rank", ""),
                        "mean_rank": group.get("mean_rank", ""),
                        "improvement_vs_identity": group.get(
                            "best_improvement_vs_identity",
                            group.get("improvement_vs_identity", ""),
                        ),
                        "selected_model_config_id": row.get(
                            "selected_model_config_id", ""
                        ),
                        "scale_x": row.get("scale_x", ""),
                        "selection_uses_test_set": row.get(
                            "selection_uses_test_set", ""
                        ),
                    }
                )


def selected_eval_row(report: dict[str, Any]) -> dict[str, Any]:
    audit = report.get("audit") or {}
    eval_report = audit.get("eval") or {}
    row = eval_report.get("best_cv")
    if not isinstance(row, dict):
        raise ValueError("campaign audit report did not contain eval.best_cv")
    return row


def audit_oracle_row(report: dict[str, Any]) -> dict[str, Any]:
    audit = report.get("audit") or {}
    row = audit.get("best_eval")
    if not isinstance(row, dict):
        raise ValueError("campaign audit report did not contain best_eval")
    return row


def _non_negative_int_counter(report: dict[str, Any], key: str) -> int:
    value = report.get(key)
    if isinstance(value, bool):
        raise ValueError(f"Ridge moment route telemetry counter {key} is not an integer")
    if isinstance(value, (int, np.integer)):
        count = int(value)
    elif isinstance(value, (float, np.floating)) and np.isfinite(value) and value.is_integer():
        count = int(value)
    else:
        raise ValueError(f"Ridge moment route telemetry counter {key} is not an integer")
    if count < 0:
        raise ValueError(f"Ridge moment route telemetry counter {key} is negative")
    return count


def validate_ridge_moment_route_telemetry(report: dict[str, Any]) -> None:
    """Validate Ridge moment route counters before persisting campaign rows.

    Older or partial reports may omit these fields, so the guard is active only
    when the total/eigen/direct CV counters are all present.
    """
    required = (
        "n_ridge_moment_cv_fits",
        "n_ridge_moment_eigen_path_cv_fits",
        "n_ridge_moment_direct_cv_fits",
    )
    if not all(key in report for key in required):
        return
    total = _non_negative_int_counter(report, "n_ridge_moment_cv_fits")
    eigen = _non_negative_int_counter(report, "n_ridge_moment_eigen_path_cv_fits")
    direct = _non_negative_int_counter(report, "n_ridge_moment_direct_cv_fits")
    if "n_ridge_moment_eigen_path_preparations" in report:
        _non_negative_int_counter(report, "n_ridge_moment_eigen_path_preparations")
    if eigen + direct != total:
        raise ValueError(
            "Ridge moment route telemetry is inconsistent: "
            f"n_ridge_moment_eigen_path_cv_fits ({eigen}) + "
            f"n_ridge_moment_direct_cv_fits ({direct}) != "
            f"n_ridge_moment_cv_fits ({total})"
        )


def property_skip_reason(args, X_train: np.ndarray) -> str:
    n_train = int(X_train.shape[0])
    n_features = int(X_train.shape[1])
    product = n_train * n_features
    reasons = []
    if args.max_train_samples is not None and n_train > int(args.max_train_samples):
        reasons.append(f"n_train>{int(args.max_train_samples)}")
    if args.max_features is not None and n_features > int(args.max_features):
        reasons.append(f"n_features>{int(args.max_features)}")
    if args.max_train_feature_product is not None and product > int(
        args.max_train_feature_product
    ):
        reasons.append(
            f"n_train_x_n_features>{int(args.max_train_feature_product)}"
        )
    return ";".join(reasons)


def run_one(args, item: dict[str, str]) -> dict[str, Any]:
    started = datetime.now(timezone.utc).isoformat()
    t0 = time.perf_counter()
    key = item["key"]
    scale_x_grid = str(getattr(args, "scale_x_grid", "") or "")
    try:
        X_train, y_train, X_test, y_test, data_path = load_split(
            Path(args.data_root), item["database_name"], item["dataset"]
        )
        skip_reason = property_skip_reason(args, X_train)
        if skip_reason:
            return {
                "schema_version": "0.1.0",
                "preset": "aom_staged_real_cohort",
                "cohort": str(Path(args.cohort_csv).name),
                "dataset_key": key,
                "database_name": item["database_name"],
                "dataset": item["dataset"],
                "task": "regression",
                "canonical_name": "N4M-AOM-staged-chain-campaign",
                "model_class": "aom_staged_chain_campaign",
                "module": "n4m",
                "selection": "exact_cv_refit_train_only",
                "seed": int(args.seed),
                "status": "skipped",
                "error_message": f"property_filter:{skip_reason}",
                "n_train": int(X_train.shape[0]),
                "n_test": int(X_test.shape[0]),
                "n_features": int(X_train.shape[1]),
                "screen_complete": "",
                "selection_uses_test_set": False,
                "plan": str(args.plan),
                "stages_json": stages_json_label(args),
                "heads": str(args.heads),
                "max_chains": (
                    "" if args.max_chains is None else int(args.max_chains)
                ),
                "chain_chunk_size": int(args.chain_chunk_size),
                "top_k": int(args.top_k),
                "refit_top_k": (
                    "" if args.refit_top_k is None else int(args.refit_top_k)
                ),
                "refit_per_head_top_k": (
                    ""
                    if args.refit_per_head_top_k is None
                    else int(args.refit_per_head_top_k)
                ),
                "moment_policy": str(args.moment_policy),
                "split_head_scoring": str(args.split_head_scoring),
                "scale_x": bool(args.scale_x),
                "scale_x_values": scale_x_grid,
                "selected_model_config_id": "",
                "cuda_pls_parallel_folds": bool(args.cuda_pls_parallel_folds),
                "cuda_pls_min_device_features": (
                    ""
                    if args.cuda_pls_min_device_features is None
                    else int(args.cuda_pls_min_device_features)
                ),
                "cuda_pls_many_batched": bool(args.cuda_pls_many_batched),
                "backend_min_cuda_product": (
                    ""
                    if args.backend_min_cuda_product is None
                    else int(args.backend_min_cuda_product)
                ),
                "library_path": n4m.library_path(),
                "abi": ".".join(str(value) for value in n4m.abi_version()),
                "started_at": started,
                "ended_at": datetime.now(timezone.utc).isoformat(),
                "notes": f"data_path={data_path}; property_filter_only",
            }
        checkpoint_dir = None
        if args.checkpoint_dir is not None:
            safe_key = "".join(
                char if char.isalnum() or char in {"-", "_", "."} else "_"
                for char in key
            ).strip("._")
            checkpoint_dir = str(Path(args.checkpoint_dir) / safe_key)
        report = n4m.aom_staged_chain_campaign(
            X_train,
            y_train,
            stages=getattr(args, "stages", None),
            plan=args.plan,
            cv=args.cv,
            ridge_lambdas=parse_float_list(args.ridge_lambdas),
            pls_components=parse_int_list(args.components),
            heads=parse_str_list(args.heads),
            max_chains=args.max_chains,
            chain_chunk_size=args.chain_chunk_size,
            top_k=args.top_k,
            refit_top_k=args.refit_top_k,
            refit_per_head_top_k=args.refit_per_head_top_k,
            checkpoint_dir=checkpoint_dir,
            resume=not bool(args.no_resume),
            max_chunks_per_run=args.max_chunks_per_run,
            scale_x=None if scale_x_grid else args.scale_x,
            scale_x_values=(
                None
                if not scale_x_grid
                else parse_optional_bool_list(scale_x_grid)
            ),
            moment_policy=args.moment_policy,
            split_head_scoring=args.split_head_scoring,
            cuda_pls_parallel_folds=args.cuda_pls_parallel_folds,
            cuda_pls_min_device_features=args.cuda_pls_min_device_features,
            cuda_pls_many_batched=args.cuda_pls_many_batched,
            backend_min_cuda_product=args.backend_min_cuda_product,
            X_audit=X_test,
            y_audit=y_test,
            audit_top_k=None,
        )
        validate_ridge_moment_route_telemetry(report)
        selected = selected_eval_row(report)
        oracle = audit_oracle_row(report)
        elapsed = time.perf_counter() - t0
        rank = report.get("rank_diagnostics") or {}
        impact = report.get("impact") or {}
        best = report["best"]
        selected_config = report.get("selected_model_config") or {}
        selected_scale_x = (
            selected_config.get("scale_x", args.scale_x)
            if isinstance(selected_config, dict)
            else args.scale_x
        )
        row = {
            "schema_version": "0.1.0",
            "preset": "aom_staged_real_cohort",
            "cohort": str(Path(args.cohort_csv).name),
            "dataset_key": key,
            "database_name": item["database_name"],
            "dataset": item["dataset"],
            "task": "regression",
            "canonical_name": "N4M-AOM-staged-chain-campaign",
            "model_class": "aom_staged_chain_campaign",
            "module": "n4m",
            "selection": "exact_cv_refit_train_only",
            "seed": int(args.seed),
            "status": "ok",
            "n_train": int(X_train.shape[0]),
            "n_test": int(X_test.shape[0]),
            "n_features": int(X_train.shape[1]),
            "rmsep": float(selected["eval_rmse"]),
            "audit_oracle_rmse": float(oracle["eval_rmse"]),
            "audit_oracle_head": str(oracle["head"]),
            "audit_oracle_param": float(oracle["param"]),
            "r2": float(selected.get("eval_r2", np.nan)),
            "score_metric": "rmsep",
            "score_value": float(selected["eval_rmse"]),
            "lower_is_better": True,
            "fit_time_s": elapsed,
            "predict_time_s": 0.0,
            "screen_complete": bool(report["screen_complete"]),
            "n_remaining_stage_chunks_total": int(
                report["n_remaining_stage_chunks_total"]
            ),
            "n_screen_candidates_total": int(report["n_screen_candidates_total"]),
            "n_screen_split_head_chunks": int(
                report.get("n_screen_split_head_chunks", 0)
            ),
            "n_screen_chunk_score_calls": int(
                report.get("n_screen_chunk_score_calls", 0)
            ),
            "n_ridge_moment_cv_fits": int(
                report.get("n_ridge_moment_cv_fits", 0)
            ),
            "n_ridge_moment_eigen_path_preparations": int(
                report.get("n_ridge_moment_eigen_path_preparations", 0)
            ),
            "n_ridge_moment_eigen_path_cv_fits": int(
                report.get("n_ridge_moment_eigen_path_cv_fits", 0)
            ),
            "n_ridge_moment_direct_cv_fits": int(
                report.get("n_ridge_moment_direct_cv_fits", 0)
            ),
            "n_ridge_moment_score_batch_calls": int(
                report.get("n_ridge_moment_score_batch_calls", 0)
            ),
            "n_ridge_moment_score_batch_jobs": int(
                report.get("n_ridge_moment_score_batch_jobs", 0)
            ),
            "n_refit_candidates": int(report["n_refit_candidates"]),
            "n_screen_pls_moment_cv_fits": int(
                report.get("n_screen_pls_moment_cv_fits", 0)
            ),
            "n_screen_pls_moment_host_cv_fits": int(
                report.get("n_screen_pls_moment_host_cv_fits", 0)
            ),
            "n_screen_pls_moment_cuda_device_cv_fits": int(
                report.get("n_screen_pls_moment_cuda_device_cv_fits", 0)
            ),
            "n_screen_pls_moment_cuda_parallel_fold_batches": int(
                report.get("n_screen_pls_moment_cuda_parallel_fold_batches", 0)
            ),
            "n_screen_pls_moment_cuda_parallel_fold_jobs": int(
                report.get("n_screen_pls_moment_cuda_parallel_fold_jobs", 0)
            ),
            "n_screen_pls_moment_cuda_many_batched_batches": int(
                report.get("n_screen_pls_moment_cuda_many_batched_batches", 0)
            ),
            "n_screen_pls_moment_cuda_many_batched_jobs": int(
                report.get("n_screen_pls_moment_cuda_many_batched_jobs", 0)
            ),
            "n_screen_pls_moment_score_batch_calls": int(
                report.get("n_screen_pls_moment_score_batch_calls", 0)
            ),
            "n_screen_pls_moment_score_batch_jobs": int(
                report.get("n_screen_pls_moment_score_batch_jobs", 0)
            ),
            "n_refit_pls_moment_cv_fits": int(
                report.get("n_refit_pls_moment_cv_fits", 0)
            ),
            "n_refit_pls_moment_host_cv_fits": int(
                report.get("n_refit_pls_moment_host_cv_fits", 0)
            ),
            "n_refit_pls_moment_cuda_device_cv_fits": int(
                report.get("n_refit_pls_moment_cuda_device_cv_fits", 0)
            ),
            "n_refit_pls_moment_cuda_parallel_fold_batches": int(
                report.get("n_refit_pls_moment_cuda_parallel_fold_batches", 0)
            ),
            "n_refit_pls_moment_cuda_parallel_fold_jobs": int(
                report.get("n_refit_pls_moment_cuda_parallel_fold_jobs", 0)
            ),
            "n_refit_pls_moment_cuda_many_batched_batches": int(
                report.get("n_refit_pls_moment_cuda_many_batched_batches", 0)
            ),
            "n_refit_pls_moment_cuda_many_batched_jobs": int(
                report.get("n_refit_pls_moment_cuda_many_batched_jobs", 0)
            ),
            "n_refit_pls_moment_score_batch_calls": int(
                report.get("n_refit_pls_moment_score_batch_calls", 0)
            ),
            "n_refit_pls_moment_score_batch_jobs": int(
                report.get("n_refit_pls_moment_score_batch_jobs", 0)
            ),
            "selected_head": str(best["head"]),
            "selected_param": float(best["param"]),
            "selected_campaign_stage": str(best.get("campaign_stage", "")),
            "best_refit_cv_rmse": float(best["refit_cv_rmse"]),
            "best_screen_cv_rmse": float(best.get("screen_cv_rmse", np.nan)),
            "best_cv_rank": int(best.get("cv_rank", 0)),
            "best_screen_rank": int(best.get("screen_rank", 0)),
            "rank_spearman": rank.get("spearman_rank_correlation", ""),
            "impact_best_score": impact.get("best_score", ""),
            "selection_uses_test_set": bool(report["selection_uses_test_set"]),
            "plan": str(report["plan"]),
            "stages_json": stages_json_label(args),
            "heads": str(args.heads),
            "max_chains": "" if args.max_chains is None else int(args.max_chains),
            "chain_chunk_size": int(args.chain_chunk_size),
            "top_k": int(args.top_k),
            "refit_top_k": "" if args.refit_top_k is None else int(args.refit_top_k),
            "refit_per_head_top_k": (
                ""
                if args.refit_per_head_top_k is None
                else int(args.refit_per_head_top_k)
            ),
            "moment_policy": str(args.moment_policy),
            "split_head_scoring": str(args.split_head_scoring),
            "scale_x": selected_scale_x,
            "scale_x_values": scale_x_grid,
            "selected_model_config_id": report.get("selected_model_config_id", ""),
            "cuda_pls_parallel_folds": bool(args.cuda_pls_parallel_folds),
            "cuda_pls_min_device_features": (
                ""
                if args.cuda_pls_min_device_features is None
                else int(args.cuda_pls_min_device_features)
            ),
            "cuda_pls_many_batched": bool(args.cuda_pls_many_batched),
            "backend_min_cuda_product": (
                ""
                if args.backend_min_cuda_product is None
                else int(args.backend_min_cuda_product)
            ),
            "checkpoint_dir": "" if checkpoint_dir is None else checkpoint_dir,
            "library_path": n4m.library_path(),
            "abi": ".".join(str(value) for value in n4m.abi_version()),
            "started_at": started,
            "ended_at": datetime.now(timezone.utc).isoformat(),
            "notes": f"data_path={data_path}; audit_oracle_is_test_rank_only",
        }
        diagnostics_dir = str(getattr(args, "diagnostics_dir", "") or "").strip()
        if diagnostics_dir:
            diagnostics_root = Path(diagnostics_dir)
            safe_key = safe_dataset_filename(key)
            write_json(
                diagnostics_root / f"{safe_key}.diagnostics.json",
                diagnostics_payload(report=report, row=row, args=args),
            )
            append_impact_groups(
                diagnostics_root / "impact_groups.csv",
                report=report,
                row=row,
            )
        return row
    except Exception as exc:  # noqa: BLE001 - benchmark rows must record failures.
        return {
            "schema_version": "0.1.0",
            "preset": "aom_staged_real_cohort",
            "cohort": str(Path(args.cohort_csv).name),
            "dataset_key": key,
            "database_name": item["database_name"],
            "dataset": item["dataset"],
            "task": "regression",
            "canonical_name": "N4M-AOM-staged-chain-campaign",
            "model_class": "aom_staged_chain_campaign",
            "module": "n4m",
            "selection": "exact_cv_refit_train_only",
            "seed": int(args.seed),
            "status": "failed",
            "error_message": f"{type(exc).__name__}: {exc}",
            "started_at": started,
            "ended_at": datetime.now(timezone.utc).isoformat(),
        }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cohort-csv", default=str(DEFAULT_COHORT))
    parser.add_argument("--data-root", default=str(DEFAULT_DATA_ROOT))
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--limit", type=int, default=10)
    parser.add_argument("--datasets", default="")
    parser.add_argument("--max-train-samples", type=int, default=None)
    parser.add_argument("--max-features", type=int, default=None)
    parser.add_argument("--max-train-feature-product", type=int, default=None)
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume an existing output CSV instead of replacing it",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--plan", default="compact")
    parser.add_argument(
        "--stages-json",
        default="",
        help=(
            "Inline JSON list of staged campaign profile strings/objects. "
            "Overrides --plan for campaign construction, but the plan label is "
            "still recorded for traceability."
        ),
    )
    parser.add_argument(
        "--stages-json-file",
        default="",
        help="Path to a JSON file with the same list accepted by --stages-json",
    )
    parser.add_argument("--cv", type=int, default=5)
    parser.add_argument("--heads", default="ridge,pls")
    parser.add_argument("--components", default="1,2")
    parser.add_argument("--ridge-lambdas", default="0.1,1.0,10.0")
    parser.add_argument("--max-chains", type=int, default=12)
    parser.add_argument("--chain-chunk-size", type=int, default=6)
    parser.add_argument("--top-k", type=int, default=12)
    parser.add_argument("--refit-top-k", type=int, default=6)
    parser.add_argument("--refit-per-head-top-k", type=int, default=2)
    parser.add_argument("--checkpoint-dir", default=None)
    parser.add_argument("--no-resume", action="store_true")
    parser.add_argument("--max-chunks-per-run", type=int, default=None)
    parser.add_argument("--scale-x", action="store_true")
    parser.add_argument(
        "--scale-x-grid",
        default="",
        help=(
            "Comma-separated optional-bool grid, e.g. false,true. When set, "
            "the staged campaign selects the best value by train exact-CV refit."
        ),
    )
    parser.add_argument("--moment-policy", default="auto")
    parser.add_argument(
        "--split-head-scoring",
        choices=("auto", "off", "force"),
        default="auto",
        help=(
            "Score mixed Ridge+PLS chunks as separate head-homogeneous native "
            "calls so existing Ridge/PLS fast paths can be used without "
            "changing candidate scores. Use off to preserve the single-call "
            "legacy launch shape."
        ),
    )
    parser.add_argument("--cuda-pls-parallel-folds", action="store_true")
    parser.add_argument("--cuda-pls-min-device-features", type=int, default=None)
    parser.add_argument("--cuda-pls-many-batched", action="store_true")
    parser.add_argument("--backend-min-cuda-product", type=int, default=None)
    parser.add_argument(
        "--diagnostics-dir",
        default="",
        help=(
            "Optional directory for per-dataset staged campaign diagnostics. "
            "Writes JSON reports and an aggregate impact_groups.csv for OK rows."
        ),
    )
    args = parser.parse_args()
    if args.scale_x and args.scale_x_grid:
        raise ValueError("--scale-x and --scale-x-grid are mutually exclusive")

    if args.limit is not None and int(args.limit) < 1:
        raise ValueError("--limit must be positive")
    if args.max_chunks_per_run is not None and int(args.max_chunks_per_run) < 1:
        raise ValueError("--max-chunks-per-run must be positive when provided")
    if args.max_train_samples is not None and int(args.max_train_samples) < 1:
        raise ValueError("--max-train-samples must be positive when provided")
    if args.max_features is not None and int(args.max_features) < 1:
        raise ValueError("--max-features must be positive when provided")
    if (
        args.max_train_feature_product is not None
        and int(args.max_train_feature_product) < 1
    ):
        raise ValueError(
            "--max-train-feature-product must be positive when provided"
        )
    args.stages = load_stages_config(args)

    requested = set(parse_str_list(args.datasets)) if args.datasets else None
    items = load_cohort(Path(args.cohort_csv), limit=args.limit, datasets=requested)
    output = Path(args.output)
    if output.exists() and not args.resume:
        output.unlink()
    completed = load_completed(output) if args.resume else set()
    for item in items:
        if item["key"] in completed:
            continue
        row = run_one(args, item)
        append_row(output, row)
        print(json.dumps({
            "dataset_key": row.get("dataset_key"),
            "status": row.get("status"),
            "rmsep": row.get("rmsep"),
            "screen_complete": row.get("screen_complete"),
            "n_refit_candidates": row.get("n_refit_candidates"),
        }))
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
