"""Regenerate or verify the committed parity-reference lockfile.

The lockfile is the host-insensitive structural contract between
`benchmarks/parity_timing/registry.py` and the per-method documentation
that surfaces the 📐 parity-gate truth-source icon.

Usage:

    # Refresh the committed file (run after editing METHODS).
    python -m benchmarks.parity_timing.lockfile --write

    # CI / pre-commit gate: assert the committed file is up-to-date and
    # every MethodSpec declares either a reference or a paper-only citation.
    python -m benchmarks.parity_timing.lockfile --check
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .registry import (
    dump_truth_sources_lockfile,
    truth_source_lockfile_entries,
    validate_methods_have_references,
)

LOCKFILE_PATH = (
    Path(__file__).resolve().parent / "truth_sources.lock.json"
)


def _current_payload() -> dict:
    return {
        "version": 1,
        "description": (
            "Structural snapshot of benchmarks/parity_timing/registry.py's "
            "parity-reference declarations. Host-insensitive: records "
            "only which reference slots are wired (python / r), the "
            "role keys of extra_references, and whether the MethodSpec "
            "is paper_only. Regenerate with "
            "`python -m benchmarks.parity_timing.lockfile`."
        ),
        "methods": truth_source_lockfile_entries(),
    }


def _read_committed(path: Path) -> dict | None:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def cmd_write(path: Path) -> int:
    dump_truth_sources_lockfile(path)
    print(f"wrote {path}")
    return 0


def cmd_check(path: Path) -> int:
    bad = validate_methods_have_references()
    if bad:
        print(
            "MethodSpec(s) without a reference factory or paper_only "
            "citation: " + ", ".join(bad),
            file=sys.stderr,
        )
        return 1
    committed = _read_committed(path)
    if committed is None:
        print(
            f"lockfile {path} does not exist — run "
            "`python -m benchmarks.parity_timing.lockfile --write`",
            file=sys.stderr,
        )
        return 1
    current = _current_payload()
    if committed != current:
        print(
            f"lockfile {path} is out of date — registry changed. Run "
            "`python -m benchmarks.parity_timing.lockfile --write` and "
            "commit.",
            file=sys.stderr,
        )
        return 1
    print(f"lockfile OK ({len(current['methods'])} methods)")
    return 0


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--write", action="store_true",
                   help="Regenerate the committed lockfile from the registry.")
    g.add_argument("--check", action="store_true",
                   help="Verify the committed lockfile matches the registry "
                        "and every MethodSpec declares a reference or "
                        "paper_only citation. Returns non-zero on drift.")
    p.add_argument("--path", default=str(LOCKFILE_PATH),
                   help=f"Lockfile path (default: {LOCKFILE_PATH}).")
    args = p.parse_args(argv)
    path = Path(args.path)
    if args.write:
        return cmd_write(path)
    return cmd_check(path)


if __name__ == "__main__":
    sys.exit(main())
