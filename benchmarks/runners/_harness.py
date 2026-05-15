"""Shared benchmark utilities: deterministic timing, table formatting, dataset
descriptors, accuracy deltas.

Design intent
-------------
- Accuracy outputs (selected operator index, selected n_components, best score
  delta, prediction RMSE delta) are deterministic and gated.
- Timing outputs (wall-clock per fit+predict, end-to-end including ctypes
  marshalling) are platform-dependent — written to a SEPARATE timing CSV that
  is NOT checked under --check.
"""

from __future__ import annotations

import csv
import math
import os
import platform
import statistics
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable, Sequence


@dataclass(frozen=True)
class AccuracyRecord:
    """Deterministic accuracy comparison for one (dataset, algorithm) cell.

    Version metadata is NOT part of this record because the runtime versions
    of `pls4all` and the bench oracle change independently of accuracy. They
    are written to the summary markdown / metadata file instead.
    """

    benchmark: str
    case: str
    n_samples: int
    n_features: int
    n_operators: int
    max_components: int
    n_folds: int
    selected_operator_index: int
    selected_operator_index_match: bool
    selected_n_components: int
    selected_n_components_match: bool
    best_score_abs_delta: float
    prediction_rmse_abs_delta: float
    prediction_rmse_rel_delta: float
    accuracy_pass: bool


@dataclass(frozen=True)
class TimingRecord:
    """Wall-clock timing for one (dataset, algorithm) cell.

    Not subject to --check gating. Useful for tracking trends over time and
    for one-off comparisons with bench.
    """

    benchmark: str
    case: str
    n_samples: int
    n_features: int
    n_operators: int
    max_components: int
    repeats: int
    pls4all_end_to_end_ms_median: float
    pls4all_end_to_end_ms_min: float
    bench_end_to_end_ms_median: float
    bench_end_to_end_ms_min: float
    speedup_median: float


def median_time_ms(thunk, *, repeats: int = 5) -> tuple[float, float]:
    """Run `thunk` ``repeats`` times and return (median, min) wall times in ms.

    `time.perf_counter` is monotonic; the median is robust to single-shot
    noise. The minimum is reported alongside the median so platform noise
    is visible to the reader.
    """
    if repeats < 1:
        raise ValueError("repeats must be >= 1")
    samples: list[float] = []
    for _ in range(repeats):
        start = time.perf_counter()
        thunk()
        samples.append((time.perf_counter() - start) * 1_000.0)
    return float(statistics.median(samples)), float(min(samples))


def write_csv(path: Path, records: Sequence[Any], fieldnames: Iterable[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    field_list = list(fieldnames)
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=field_list)
        writer.writeheader()
        for record in records:
            row = asdict(record) if hasattr(record, "__dataclass_fields__") else dict(record)
            writer.writerow({k: row[k] for k in field_list})


def round_float(value: float, digits: int = 12) -> float:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return value
    return round(float(value), digits)


def host_info() -> dict[str, str]:
    return {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "processor": platform.processor() or "unknown",
        "machine": platform.machine(),
        "cores_logical": str(os.cpu_count() or 0),
    }


def format_summary_table(records: Sequence[AccuracyRecord]) -> str:
    """Render a short ASCII table for the accuracy CSV."""
    if not records:
        return "(no records)\n"
    head = [
        "benchmark", "case", "rmse_abs", "best_score_abs", "op_match",
        "k_match", "pass",
    ]
    rows = [head]
    for r in records:
        rows.append([
            r.benchmark,
            r.case,
            f"{r.prediction_rmse_abs_delta:.3e}",
            f"{r.best_score_abs_delta:.3e}",
            "y" if r.selected_operator_index_match else "n",
            "y" if r.selected_n_components_match else "n",
            "y" if r.accuracy_pass else "n",
        ])
    widths = [max(len(row[i]) for row in rows) for i in range(len(head))]
    sep = "+" + "+".join("-" * (w + 2) for w in widths) + "+"
    out: list[str] = [sep]
    for i, row in enumerate(rows):
        cells = " | ".join(f"{cell:<{widths[j]}}" for j, cell in enumerate(row))
        out.append(f"| {cells} |")
        if i == 0:
            out.append(sep)
    out.append(sep)
    return "\n".join(out) + "\n"
