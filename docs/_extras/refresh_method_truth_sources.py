#!/usr/bin/env python3
"""Refresh the reviewed parity-reference snapshot for generated method docs.

Run this only in the maintained benchmark-reference environment, where the
optional R and Python reference adapters are available.  The documentation
generator reads the resulting JSON instead of resolving those host-dependent
adapters on every docs build.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
TARGET = ROOT / "docs" / "_extras" / "method_truth_sources.json"
LOCKFILE = ROOT / "benchmarks" / "parity_timing" / "truth_sources.lock.json"


def collect() -> dict[str, dict[str, dict[str, Any]]]:
    """Resolve the registry only for an explicit reviewed snapshot refresh."""
    sys.path.insert(0, str(ROOT))
    from benchmarks.parity_timing.registry import METHODS, truth_source_metadata_for
    from build_methods import canonical_truth_source_cid

    records: dict[str, dict[str, dict[str, Any]]] = {}
    for method in METHODS:
        sources: dict[str, dict[str, Any]] = {}
        for cid, metadata in truth_source_metadata_for(method).items():
            sources[canonical_truth_source_cid(cid)] = metadata
        records[method.name] = sources
    return records


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--write",
        action="store_true",
        help="replace the reviewed snapshot after inspecting the diff",
    )
    args = parser.parse_args()
    if not args.write:
        parser.error("pass --write only from the benchmark-reference environment")

    payload = {
        "schema": 1,
        "source": {
            "registry": "benchmarks/parity_timing/registry.py",
            "lockfile": "benchmarks/parity_timing/truth_sources.lock.json",
            "sha256": hashlib.sha256(LOCKFILE.read_bytes()).hexdigest(),
            "exporter": "docs/_extras/refresh_method_truth_sources.py",
            "description": (
                "Reviewed resolved parity-reference metadata for the generated "
                "method corpus."
            ),
        },
        "methods": collect(),
    }
    TARGET.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    entries = sum(len(sources) for sources in payload["methods"].values())
    print(f"wrote {TARGET}: {len(payload['methods'])} methods, {entries} references")


if __name__ == "__main__":
    main()
