"""Export the authoritative HPO registry for cross-binding golden-tape runners."""

from __future__ import annotations

import json
import sys
from dataclasses import asdict
from pathlib import Path

from specs import REGISTRY


def main() -> None:
    if len(sys.argv) != 2:
        raise SystemExit("usage: export_specs.py OUTPUT.json")
    Path(sys.argv[1]).write_text(
        json.dumps([asdict(spec) for spec in REGISTRY], ensure_ascii=False),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
