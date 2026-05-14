"""Generate compact C++ fixture headers from parity JSON files."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any, Sequence


REPO_ROOT = Path(__file__).resolve().parents[4]
FIXTURE_DIR = REPO_ROOT / "parity" / "fixtures"
HEADER_SPECS = {
    "simpls": {
        "out": REPO_ROOT / "cpp" / "tests" / "fixtures" / "simpls_fixtures.hpp",
        "fixtures": (
            "synthetic_simpls_tiny_pls1_v1",
            "synthetic_simpls_small_pls2_v1",
        ),
        "struct": "SimplsFixture",
        "array": "kSimplsFixtures",
    },
    "svd": {
        "out": REPO_ROOT / "cpp" / "tests" / "fixtures" / "svd_fixtures.hpp",
        "fixtures": (
            "synthetic_svd_tiny_pls1_v1",
            "synthetic_svd_small_pls2_v1",
        ),
        "struct": "SvdFixture",
        "array": "kSvdFixtures",
    },
    "power": {
        "out": REPO_ROOT / "cpp" / "tests" / "fixtures" / "power_fixtures.hpp",
        "fixtures": (
            "synthetic_power_tiny_pls1_v1",
            "synthetic_power_small_pls2_v1",
        ),
        "struct": "PowerFixture",
        "array": "kPowerFixtures",
    },
    "pcr": {
        "out": REPO_ROOT / "cpp" / "tests" / "fixtures" / "pcr_fixtures.hpp",
        "fixtures": (
            "synthetic_pcr_tiny_pls1_v1",
            "synthetic_pcr_small_pls2_v1",
        ),
        "struct": "PcrFixture",
        "array": "kPcrFixtures",
    },
    "kernel": {
        "out": REPO_ROOT / "cpp" / "tests" / "fixtures" / "kernel_fixtures.hpp",
        "fixtures": (
            "synthetic_kernel_tiny_pls1_v1",
            "synthetic_kernel_wide_pls2_v1",
        ),
        "struct": "KernelFixture",
        "array": "kKernelFixtures",
    },
    "wide-kernel": {
        "out": REPO_ROOT / "cpp" / "tests" / "fixtures" / "wide_kernel_fixtures.hpp",
        "fixtures": (
            "synthetic_wide_kernel_pls1_v1",
            "synthetic_wide_kernel_pls2_v1",
        ),
        "struct": "WideKernelFixture",
        "array": "kWideKernelFixtures",
    },
    "orthogonal-scores": {
        "out": REPO_ROOT / "cpp" / "tests" / "fixtures" / "orthogonal_scores_fixtures.hpp",
        "fixtures": (
            "synthetic_oscores_tiny_pls1_v1",
            "synthetic_oscores_small_pls2_v1",
        ),
        "struct": "OrthogonalScoresFixture",
        "array": "kOrthogonalScoresFixtures",
    },
}


def _symbol(name: str) -> str:
    return re.sub(r"[^0-9A-Za-z_]", "_", name)


def _shape(shape: list[int]) -> tuple[int, int]:
    if len(shape) == 1:
        return 1, int(shape[0])
    if len(shape) == 2:
        return int(shape[0]), int(shape[1])
    raise ValueError(f"unsupported 3-D fixture field shape: {shape}")


def _format_double(value: float) -> str:
    return f"{float(value):.17g}"


def _emit_array(lines: list[str], name: str, values: list[float]) -> None:
    lines.append(f"inline const double {name}[] = {{")
    row: list[str] = []
    for value in values:
        row.append(_format_double(value))
        if len(row) == 4:
            lines.append(f"    {', '.join(row)},")
            row.clear()
    if row:
        lines.append(f"    {', '.join(row)},")
    lines.append("};")
    lines.append("")


def _matrix_ref(array_name: str, payload: dict[str, Any]) -> str:
    rows, cols = _shape(payload["shape"])
    sign = "true" if payload.get("sign_invariant", False) else "false"
    return (
        f"MatrixRef{{{rows}, {cols}, {array_name}, "
        f"sizeof({array_name}) / sizeof(double), {sign}}}"
    )


def _load_fixture(fixture_id: str, fixture_dir: Path) -> dict[str, Any]:
    with (fixture_dir / f"{fixture_id}.json").open("r", encoding="utf-8") as fh:
        return json.load(fh)


def generate(fixture_ids: Sequence[str],
             fixture_dir: Path,
             out: Path,
             family: str,
             struct_name: str,
             array_name: str) -> None:
    fixtures = [_load_fixture(fid, fixture_dir) for fid in fixture_ids]
    lines = [
        "// SPDX-License-Identifier: CeCILL-2.1",
        f"// Generated mechanically from parity/fixtures/*{family}*.json.",
        "#pragma once",
        "",
        "#include <cstddef>",
        "#include <cstdint>",
        "",
        '#include "phase1_fixtures.hpp"',
        "",
        "namespace pls4all::test::fixtures {",
        "",
        f"struct {struct_name} {{",
        "    const char* id;",
        "    std::int32_t n_components;",
        "    MatrixRef X;",
        "    MatrixRef Y;",
        "    MatrixRef coefficients;",
        "    MatrixRef intercept;",
        "    MatrixRef x_mean;",
        "    MatrixRef x_scale;",
        "    MatrixRef y_mean;",
        "    MatrixRef y_scale;",
        "    MatrixRef weights_w;",
        "    MatrixRef loadings_p;",
        "    MatrixRef y_loadings_q;",
        "    MatrixRef rotations_r;",
        "    MatrixRef scores_t;",
        "    MatrixRef predict_train;",
        "};",
        "",
    ]

    entries: list[str] = []
    fields = (
        ("X", ("data", "X"), "x"),
        ("Y", ("data", "Y"), "y"),
        ("coefficients", ("expected", "coefficients"), "coefficients"),
        ("intercept", ("expected", "intercept"), "intercept"),
        ("x_mean", ("expected", "x_mean"), "x_mean"),
        ("x_scale", ("expected", "x_scale"), "x_scale"),
        ("y_mean", ("expected", "y_mean"), "y_mean"),
        ("y_scale", ("expected", "y_scale"), "y_scale"),
        ("weights_W", ("expected", "weights_W"), "weights_w"),
        ("loadings_P", ("expected", "loadings_P"), "loadings_p"),
        ("y_loadings_Q", ("expected", "y_loadings_Q"), "y_loadings_q"),
        ("rotations_R", ("expected", "rotations_R"), "rotations_r"),
        ("scores_T", ("expected", "scores_T"), "scores_t"),
        ("predict_train", ("expected", "predict_train"), "predict_train"),
    )

    for fixture in fixtures:
        fid = fixture["fixture_id"]
        prefix = _symbol(fid)
        refs: dict[str, str] = {}
        for _public_name, path, suffix in fields:
            payload: dict[str, Any] = fixture
            for key in path:
                payload = payload[key]
            field_array_name = f"{prefix}_{suffix}"
            _emit_array(lines, field_array_name, payload["values"])
            refs[suffix] = _matrix_ref(field_array_name, payload)
        n_components = int(fixture["generator"]["params"]["n_components"])
        entries.append(
            "    {\n"
            f'        "{fid}",\n'
            f"        {n_components},\n"
            f"        {refs['x']},\n"
            f"        {refs['y']},\n"
            f"        {refs['coefficients']},\n"
            f"        {refs['intercept']},\n"
            f"        {refs['x_mean']},\n"
            f"        {refs['x_scale']},\n"
            f"        {refs['y_mean']},\n"
            f"        {refs['y_scale']},\n"
            f"        {refs['weights_w']},\n"
            f"        {refs['loadings_p']},\n"
            f"        {refs['y_loadings_q']},\n"
            f"        {refs['rotations_r']},\n"
            f"        {refs['scores_t']},\n"
            f"        {refs['predict_train']}\n"
            "    }"
        )

    lines.append(f"inline const {struct_name} {array_name}[] = {{")
    lines.append(",\n".join(entries))
    lines.append("};")
    lines.append("")
    lines.append("}  // namespace pls4all::test::fixtures")
    lines.append("")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines), encoding="utf-8", newline="\n")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--family", choices=sorted(HEADER_SPECS), default="simpls",
                        help="Fixture family to emit when fixture ids are omitted.")
    parser.add_argument("--fixture-dir", type=Path, default=FIXTURE_DIR)
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("fixture_ids", nargs="*")
    args = parser.parse_args(argv)
    spec = HEADER_SPECS[args.family]
    out = args.out or spec["out"]
    fixture_ids = args.fixture_ids or list(spec["fixtures"])
    generate(fixture_ids, args.fixture_dir, out, args.family,
             str(spec["struct"]), str(spec["array"]))
    print(f"Wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
