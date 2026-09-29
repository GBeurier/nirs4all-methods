"""Export the 45 native compatibility cells for cross-binding parity runners.

The versioned matrix contract selects the cases. Python/libn4m owns the score
and event tape; each other binding must reproduce it from its own native calls.
Generated files are CI artifacts, not a second committed golden registry.
"""

from __future__ import annotations

import json
import sys
from dataclasses import asdict
from pathlib import Path

from compatibility import _validate_trace, load_contract, make_spec
from export_matlab_fixtures import export as export_matlab_fixtures
from run_native import run_spec


def export(directory: Path) -> None:
    contract = load_contract()
    specs = [make_spec(contract, cell) for cell in contract["cells"]]
    if len(specs) != 45 or len({spec.id for spec in specs}) != 45:
        raise ValueError("compatibility fixture export requires exactly 45 unique cells")
    golden_dir = directory / "golden"
    golden_dir.mkdir(parents=True, exist_ok=True)
    (directory / "specs.json").write_text(
        json.dumps([asdict(spec) for spec in specs], ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    for cell, spec in zip(contract["cells"], specs, strict=True):
        trace = run_spec(spec)
        _validate_trace(contract, cell, trace)
        (golden_dir / f"{spec.id}.json").write_text(
            json.dumps(trace, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
    export_matlab_fixtures(directory, specs=specs, golden_root=golden_dir)


if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise SystemExit("usage: export_compatibility_fixtures.py OUTPUT_DIRECTORY")
    export(Path(sys.argv[1]))
