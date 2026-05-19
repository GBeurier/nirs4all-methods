#!/usr/bin/env python3
# SPDX-License-Identifier: CECILL-2.1
"""Validate committed benchmark dashboard parity metadata.

This guard is intentionally stdlib-only so docs/CI can run it before installing
the benchmark runtime stack. It checks the generated CSV and dashboard JSON for
regressions that would hide reference gates as timing-only dashboard cells.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CSV = REPO_ROOT / "benchmarks" / "cross_binding" / "results" / "full_matrix.csv"
DEFAULT_JSON = REPO_ROOT / "docs" / "_static" / "bench-data.json"
DEFAULT_SNAPSHOT_ROOT = REPO_ROOT / "benchmarks" / "reference_snapshots" / "cross_binding"

FORBIDDEN_RAW_PARITIES = {
    "timing_only",
    "reference_snapshot_missing",
    "reference_not_compared",
    "external_not_compared",
}
FORBIDDEN_DASHBOARD_PARITIES = {"not_compared"}
CANONICAL_SNAPSHOT_PARITIES = {"reference_snapshot", "reference_snapshot_drift"}
CONTEXT_PARITY = "context"


def safe_slug(value: str) -> str:
    text = re.sub(r"[^A-Za-z0-9_.-]+", "_", value.strip())
    text = re.sub(r"_+", "_", text).strip("_.")
    return text[:96] or "reference"


def is_true(value: str | None) -> bool:
    return str(value or "").strip().lower() in {"1", "true", "yes", "y", "pass"}


def load_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def validate_csv(rows: list[dict[str, str]], snapshot_root: Path) -> list[str]:
    errors: list[str] = []
    canonical_refs: set[tuple[str, str, str, str]] = set()

    for idx, row in enumerate(rows, start=2):
        parity = row.get("parity", "").strip()
        if parity in FORBIDDEN_RAW_PARITIES:
            errors.append(f"CSV line {idx}: forbidden raw parity {parity!r}")

        if (
            row.get("kind") == "external_reference"
            and row.get("reference_role") == "canonical"
            and is_true(row.get("ok"))
        ):
            algorithm = row.get("algorithm", "").strip()
            backend = row.get("backend", "").strip()
            reference_library = row.get("reference_library", "").strip()
            n = row.get("n", "").strip()
            p = row.get("p", "").strip()
            canonical_refs.add((algorithm, n, p, row.get("threads", "").strip()))

            if parity not in CANONICAL_SNAPSHOT_PARITIES:
                errors.append(
                    f"CSV line {idx}: canonical reference {algorithm}/{backend} "
                    f"has parity {parity!r}, expected a stored snapshot verdict"
                )

            pattern = (
                f"{safe_slug(backend)}__{safe_slug(reference_library)}"
                f"__n{n}_p{p}_seed*.npz"
            )
            matches = list((snapshot_root / safe_slug(algorithm)).glob(pattern))
            if not matches:
                errors.append(
                    f"CSV line {idx}: missing committed reference snapshot for "
                    f"{algorithm}/{backend}/{reference_library} n={n} p={p}"
                )

        if row.get("kind") == "external_reference" and row.get("reference_role") == "context":
            if parity != CONTEXT_PARITY:
                errors.append(f"CSV line {idx}: context reference has parity {parity!r}")
            if row.get("reference_parity_ok", "").strip():
                errors.append(f"CSV line {idx}: context reference must not carry reference_parity_ok")

    for idx, row in enumerate(rows, start=2):
        if row.get("backend") != "cpp" or not is_true(row.get("ok")):
            continue
        key = (
            row.get("algorithm", "").strip(),
            row.get("n", "").strip(),
            row.get("p", "").strip(),
            row.get("threads", "").strip(),
        )
        if key not in canonical_refs:
            continue
        parity = row.get("parity", "").strip()
        if parity == "ok":
            errors.append(
                f"CSV line {idx}: C++ row {key[0]} n={key[1]} p={key[2]} "
                "has no reference-gate verdict"
            )

    return errors


def validate_json(path: Path, snapshot_root: Path) -> list[str]:
    errors: list[str] = []
    data = json.loads(path.read_text(encoding="utf-8"))
    for row_idx, row in enumerate(data.get("rows", []), start=1):
        algo = row.get("algo", f"row-{row_idx}")
        n = str(row.get("n", "")).strip()
        p = str(row.get("p", "")).strip()
        canonical_refs: set[str] = set()
        for col_id, cell in (row.get("cells") or {}).items():
            parity = str(cell.get("parity") or "").strip()
            raw = str(cell.get("raw_parity") or "").strip()
            if parity in FORBIDDEN_DASHBOARD_PARITIES:
                errors.append(f"JSON row {algo}/{col_id}: forbidden parity {parity!r}")
            if raw in FORBIDDEN_RAW_PARITIES:
                errors.append(f"JSON row {algo}/{col_id}: forbidden raw parity {raw!r}")
            if cell.get("reference_role") == "canonical" and cell.get("ok") is True:
                canonical_refs.add(col_id)
                if raw not in CANONICAL_SNAPSHOT_PARITIES:
                    errors.append(
                        f"JSON row {algo}/{col_id}: canonical reference has raw parity "
                        f"{raw!r}, expected a stored snapshot verdict"
                    )
                reference = str(cell.get("reference") or "").strip()
                pattern = f"{safe_slug(col_id)}__{safe_slug(reference)}__n{n}_p{p}_seed*.npz"
                matches = list((snapshot_root / safe_slug(str(algo))).glob(pattern))
                if not matches:
                    errors.append(
                        f"JSON row {algo}/{col_id}: missing committed reference snapshot "
                        f"for {reference} n={n} p={p}"
                    )
            if cell.get("reference_role") == "context":
                if raw != CONTEXT_PARITY or parity != CONTEXT_PARITY:
                    errors.append(
                        f"JSON row {algo}/{col_id}: context reference has "
                        f"parity={parity!r} raw={raw!r}"
                    )
                if cell.get("reference_parity_ok") is not None:
                    errors.append(
                        f"JSON row {algo}/{col_id}: context reference must not carry reference parity"
                    )
        if canonical_refs:
            for col_id, cell in (row.get("cells") or {}).items():
                if not str(col_id).startswith("chemometrics4all.cpp"):
                    continue
                if cell.get("ok") is True and str(cell.get("raw_parity") or "") == "ok":
                    errors.append(
                        f"JSON row {algo}/{col_id}: C++ cell has no reference-gate verdict"
                    )
    return errors


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", type=Path, default=DEFAULT_CSV)
    parser.add_argument("--json", type=Path, default=DEFAULT_JSON)
    parser.add_argument("--snapshot-root", type=Path, default=DEFAULT_SNAPSHOT_ROOT)
    args = parser.parse_args()

    errors: list[str] = []
    if args.csv.exists():
        errors.extend(validate_csv(load_csv(args.csv), args.snapshot_root))
    if not args.json.exists():
        errors.append(f"missing JSON: {args.json}")
    else:
        errors.extend(validate_json(args.json, args.snapshot_root))

    if errors:
        for error in errors:
            print(f"ERROR: {error}", file=sys.stderr)
        return 1
    print("benchmark dashboard payload guard OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
