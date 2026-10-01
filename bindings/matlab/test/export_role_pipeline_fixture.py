"""Render a canonical RolePipeline JSON witness as MATLAB/Octave test data.

Only representation changes: cell arrays retain order and singleton shape,
integers retain their exact bits, and N4ME base64 is decoded without changing
payload bytes. No estimator, oracle, or fitted state is computed here.
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
import math
import re
from pathlib import Path

_FIELD = re.compile(r"[A-Za-z][A-Za-z0-9_]*\Z")


def _integer(value: int) -> str:
    if not -(1 << 63) <= value < (1 << 64):
        raise ValueError("fixture integer is outside int64/uint64")
    if abs(value) <= (1 << 53) - 1:
        return f"int64({value})"
    bits = value % (1 << 64)
    unsigned = (
        f"bitor(bitshift(uint64({bits >> 32}), 32), "
        f"uint64({bits & ((1 << 32) - 1)}))"
    )
    return unsigned if value >= (1 << 63) else f"typecast({unsigned}, 'int64')"


def _literal(value: object) -> str:
    if value is None:
        return "[]"
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, int):
        return _integer(value)
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("fixture numbers must be finite")
        return repr(value)
    if isinstance(value, str):
        if any(ord(char) < 32 or ord(char) > 126 for char in value):
            encoded = value.encode("utf-16-le")
            units = [int.from_bytes(encoded[i:i + 2], "little")
                     for i in range(0, len(encoded), 2)]
            return "char([" + " ".join(map(str, units)) + "])"
        return "'" + value.replace("'", "''") + "'"
    raise ValueError(f"unsupported fixture scalar: {type(value).__name__}")


def _emit(lines: list[str], target: str, value: object) -> None:
    if isinstance(value, dict):
        lines.append(f"{target} = struct();")
        for key, item in value.items():
            if not _FIELD.fullmatch(key):
                raise ValueError(f"invalid MATLAB fixture field: {key!r}")
            _emit(lines, f"{target}.{key}", item)
    elif isinstance(value, bytes):
        values = " ".join(map(str, value))
        lines.append(f"{target} = uint8([{values}]);")
    elif isinstance(value, list):
        lines.append(f"{target} = cell(1, {len(value)});")
        for index, item in enumerate(value, 1):
            _emit(lines, f"{target}{{{index}}}", item)
    else:
        lines.append(f"{target} = {_literal(value)};")


def export(source: Path, output: Path) -> None:
    raw = source.read_bytes()
    fixture = json.loads(raw)
    if not re.fullmatch(r"[0-9]+\.[0-9]+\.[0-9]+", fixture["abi"]):
        raise ValueError("fixture must declare an exact ABI")
    for kind in ("regression", "classification"):
        for state in fixture[kind]["states"]:
            state["n4me_bytes"] = base64.b64decode(state["n4me_base64"], validate=True)
    for case in fixture["cases"]:
        if "states" in case:
            case["state_bytes"] = [base64.b64decode(state, validate=True)
                                   for state in case["states"]]
    lines = [
        "% Generated fixture data; do not edit or use as product code.",
        f"% Source SHA-256: {hashlib.sha256(raw).hexdigest()}",
    ]
    _emit(lines, "role_pipeline_fixture", fixture)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path, help="canonical RolePipeline JSON fixture")
    parser.add_argument("output", type=Path, help="generated .m script (not committed)")
    args = parser.parse_args()
    if args.output.suffix != ".m":
        parser.error("output must end in .m")
    export(args.source, args.output)


if __name__ == "__main__":
    main()
