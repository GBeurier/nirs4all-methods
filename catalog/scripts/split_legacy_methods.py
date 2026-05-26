#!/usr/bin/env python3
"""Split legacy catalog/methods.yaml into catalog/methods/<method_id>.yaml."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from catalog_loader import (
    LEGACY_METHODS,
    METHODS_DIR,
    load_legacy_methods,
    method_filename,
    render_method,
)


def expected_files() -> dict[Path, str]:
    out: dict[Path, str] = {}
    for method in load_legacy_methods(LEGACY_METHODS):
        method_id = method.get("method_id")
        if not isinstance(method_id, str) or not method_id:
            raise SystemExit("method without method_id in catalog/methods.yaml")
        out[METHODS_DIR / method_filename(method_id)] = render_method(method)
    return out


def check_split(files: dict[Path, str]) -> int:
    errors = 0
    for path, content in files.items():
        if not path.exists():
            print(f"missing: {path}", file=sys.stderr)
            errors += 1
            continue
        current = path.read_text(encoding="utf-8")
        if current != content:
            print(f"stale: {path}", file=sys.stderr)
            errors += 1
    expected = {path.name for path in files}
    extra = sorted(
        path for path in METHODS_DIR.glob("*.yaml") if path.name not in expected
    )
    for path in extra:
        print(f"extra: {path}", file=sys.stderr)
        errors += 1
    if errors:
        print(f"FAIL: {errors} split catalog file issue(s)", file=sys.stderr)
        return 1
    print(f"PASS: {len(files)} per-method files are up to date")
    return 0


def write_split(files: dict[Path, str]) -> int:
    METHODS_DIR.mkdir(parents=True, exist_ok=True)
    for path, content in sorted(files.items()):
        path.write_text(content, encoding="utf-8")
    expected = {path.name for path in files}
    pruned = 0
    for path in sorted(METHODS_DIR.glob("*.yaml")):
        if path.name not in expected:
            path.unlink()
            pruned += 1
    suffix = f", pruned {pruned} stale file(s)" if pruned else ""
    print(f"wrote {len(files)} files under {METHODS_DIR}{suffix}")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--check",
        action="store_true",
        help="verify existing split files",
    )
    parser.add_argument("--write", action="store_true", help="write split files")
    args = parser.parse_args(argv)

    files = expected_files()
    if args.check:
        return check_split(files)
    return write_split(files)


if __name__ == "__main__":
    raise SystemExit(main())
