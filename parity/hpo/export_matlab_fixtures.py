"""Render the canonical HPO specs and goldens as MATLAB/Octave source data.

The CI's Ubuntu 22.04 Octave 6.4 does not provide ``jsondecode``. Keeping JSON
as the authoritative fixture and generating an ordinary ``.m`` function avoids
an Octave-package dependency while giving MATLAB and Octave identical inputs.
"""

from __future__ import annotations

import json
import math
import re
import sys
from dataclasses import asdict
from pathlib import Path

from specs import REGISTRY, HpoSpec

_FIELD = re.compile(r"[A-Za-z][A-Za-z0-9_]*\Z")


def _literal(value: str | float | bool | None) -> str:
    if value is None:
        return "[]"
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, str):
        if any(char in value for char in "\r\n\x00"):
            raise ValueError("MATLAB fixture strings must be single-line text")
        return "'" + value.replace("'", "''") + "'"
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float) and math.isfinite(value):
        return repr(value)
    raise ValueError(f"unsupported MATLAB fixture scalar: {value!r}")


def _emit(lines: list[str], target: str, value: object) -> None:
    if isinstance(value, dict):
        lines.append(f"{target} = struct();")
        for name, item in value.items():
            if not _FIELD.fullmatch(name):
                raise ValueError(f"invalid MATLAB field name: {name!r}")
            _emit(lines, f"{target}.{name}", item)
    elif isinstance(value, (list, tuple)):
        if not value:
            lines.append(f"{target} = [];")
        elif all(isinstance(item, str) for item in value):
            items = ", ".join(_literal(item) for item in value)
            lines.append(f"{target} = {{{items}}};")
        elif all(isinstance(item, (bool, int, float)) for item in value):
            items = ", ".join(_literal(item) for item in value)
            lines.append(f"{target} = [{items}];")
        else:
            lines.append(f"{target} = cell(1, {len(value)});")
            for index, item in enumerate(value, start=1):
                _emit(lines, f"{target}{{{index}}}", item)
    else:
        lines.append(f"{target} = {_literal(value)};")


def export(
    directory: Path,
    *,
    specs: list[HpoSpec] = REGISTRY,
    golden_root: Path | None = None,
) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    if golden_root is None:
        golden_root = Path(__file__).with_name("golden")
    lines = [
        "function [specs, goldens] = n4m_hpo_fixtures()",
        "% Generated from Python HPO specs and native JSON traces; do not edit.",
        f"specs = cell(1, {len(specs)});",
        f"goldens = cell(1, {len(specs)});",
    ]
    for index, spec in enumerate(specs, start=1):
        _emit(lines, f"specs{{{index}}}", asdict(spec))
        golden = json.loads(
            (golden_root / f"{spec.id}.json").read_text(encoding="utf-8")
        )
        _emit(lines, f"goldens{{{index}}}", golden)
    lines.append("end")
    result = directory / "n4m_hpo_fixtures.m"
    result.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return result


if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise SystemExit("usage: export_matlab_fixtures.py OUTPUT_DIRECTORY")
    print(export(Path(sys.argv[1])))
