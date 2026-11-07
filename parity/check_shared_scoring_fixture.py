#!/usr/bin/env python3
"""Check the pinned copy locally and, when available, the DAG-owned source."""

import argparse
import hashlib
import json
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dag-ml-repo", type=Path, help="Explicit DAG-ML checkout")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    receipt = json.loads(
        (
            root / "parity/fixtures/shared_scoring_numerical.v1.provenance.json"
        ).read_text()
    )
    expected = receipt["sha256"]
    copied = root / receipt["copy_path"]
    if hashlib.sha256(copied.read_bytes()).hexdigest() != expected:
        print("FAIL: Methods fixture differs from its pinned SHA-256 receipt")
        return 1
    print(f"Pinned Methods fixture: verified SHA-256 {expected}")
    r_copy_path = "bindings/r/n4m/inst/extdata/shared_scoring_numerical.v1.json"
    r_copy = root / r_copy_path
    r_receipt_path = r_copy.with_suffix(".provenance.json")
    if not r_copy.is_file() or not r_receipt_path.is_file():
        print("FAIL: R package-local fixture or provenance receipt is missing")
        return 1
    r_receipt = json.loads(r_receipt_path.read_text())
    provenance_keys = (
        "source_repository",
        "source_commit_at_fixture_creation",
        "source_commit_note",
        "source_path",
        "sha256",
        "hash_scope",
    )
    if (
        any(r_receipt.get(key) != receipt[key] for key in provenance_keys)
        or r_receipt.get("copy_path") != r_copy_path
        or r_receipt.get("copied_from") != receipt["copy_path"]
    ):
        print(
            "FAIL: R package-local fixture provenance differs from the Methods receipt"
        )
        return 1
    if hashlib.sha256(r_copy.read_bytes()).hexdigest() != expected:
        print("FAIL: R package-local fixture differs from its pinned SHA-256 receipt")
        return 1
    print(f"R package-local fixture and provenance: verified SHA-256 {expected}")
    dag = args.dag_ml_repo if args.dag_ml_repo is not None else root.parent / "dag-ml"
    source = dag / receipt["source_path"]
    if not source.is_file():
        print(
            f"DAG source drift check: UNAVAILABLE ({source}); local pinned tests remain usable"
        )
        return 2
    if hashlib.sha256(source.read_bytes()).hexdigest() != expected:
        print(f"FAIL: DAG source drift detected at {source}")
        return 1
    print(f"DAG source drift check: identical ({source})")
    crate_copy = dag / receipt["crate_copy_path"]
    if not crate_copy.is_file():
        print(f"Crate fixture drift check: UNAVAILABLE ({crate_copy})")
        return 2
    if crate_copy.read_bytes() != copied.read_bytes():
        print(f"FAIL: crate-local fixture drift detected at {crate_copy}")
        return 1
    print(f"Crate fixture drift check: identical ({crate_copy}); all four copies match")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
