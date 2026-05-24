#!/usr/bin/env python3
"""Migrate per-method catalog YAML files between catalog schema versions."""

from __future__ import annotations

import argparse
import sys

from catalog_loader import (
    CATALOG_VERSION,
    METHODS_DIR,
    load_methods,
    method_filename,
    render_method,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--from", dest="from_version", type=int, required=True)
    parser.add_argument("--to", dest="to_version", type=int, required=True)
    parser.add_argument("--write", action="store_true", help="rewrite files in place")
    args = parser.parse_args(argv)

    if args.from_version != CATALOG_VERSION or args.to_version != CATALOG_VERSION:
        print(
            f"unsupported migration: {args.from_version} -> {args.to_version}; "
            f"only no-op v{CATALOG_VERSION} -> v{CATALOG_VERSION} is implemented",
            file=sys.stderr,
        )
        return 2

    methods = load_methods(prefer_split=True)
    if args.write:
        METHODS_DIR.mkdir(parents=True, exist_ok=True)
        for method in methods:
            path = METHODS_DIR / method_filename(method["method_id"])
            path.write_text(render_method(method), encoding="utf-8")
        print(f"rewrote {len(methods)} v{CATALOG_VERSION} method files")
    else:
        print(
            f"checked {len(methods)} v{CATALOG_VERSION} method files; "
            "no migration needed"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
