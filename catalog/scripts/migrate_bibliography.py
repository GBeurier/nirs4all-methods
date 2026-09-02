#!/usr/bin/env python3
"""Create or verify the lossless bibliography catalog v1 from historical sources.

No network access is used.  The output deliberately leaves structured fields
null when the prior documentation provided only prose, avoiding fabricated
citation metadata while retaining the exact source string and provenance.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from bibliography_catalog import CATALOG_PATH, build_catalog, canonical_json


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=CATALOG_PATH)
    parser.add_argument("--source-revision", help="pin a historical source revision")
    parser.add_argument("--check", action="store_true", help="fail if output is stale")
    args = parser.parse_args(argv)
    rendered = canonical_json(build_catalog(revision=args.source_revision))
    if args.check:
        if (
            not args.output.exists()
            or args.output.read_text(encoding="utf-8") != rendered
        ):
            print(f"FAIL: bibliography migration is stale: {args.output}")
            return 1
        print(f"PASS: bibliography migration is reproducible ({args.output})")
        return 0
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(rendered, encoding="utf-8")
    print(f"WROTE: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
