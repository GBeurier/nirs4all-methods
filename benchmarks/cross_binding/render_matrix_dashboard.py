#!/usr/bin/env python3
"""Render the rich cross-binding dashboard as a standalone HTML file.

This command is intentionally a thin adapter around the documentation
dashboard. The interactive matrix in ``docs/_templates/landing.html`` is the
canonical UI; this script lets a single orchestrator CSV use that same shell
without requiring a full Sphinx build.
"""

from __future__ import annotations

import argparse
import html
import importlib.util
import re
import shutil
import tempfile
from pathlib import Path
from types import ModuleType


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_TEMPLATE = REPO_ROOT / "docs/_templates/landing.html"
BUILD_LANDING = REPO_ROOT / "docs/_extras/build_landing.py"
BENCH_DATA_RE = re.compile(r"\{\{\s*bench_data_json\s*\|\s*safe\s*\}\}")
GENERATED_AT_RE = re.compile(r"\{\{\s*generated_at\s*\}\}")


def load_build_landing() -> ModuleType:
    spec = importlib.util.spec_from_file_location("n4m_build_landing", BUILD_LANDING)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"could not load {BUILD_LANDING}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def build_dashboard_json(csv_path: Path) -> dict[str, str]:
    build_landing = load_build_landing()
    with tempfile.TemporaryDirectory(prefix="n4m_dashboard_") as tmp:
        results_dir = Path(tmp)
        shutil.copy2(csv_path, results_dir / "full_matrix.csv")
        payload = build_landing.build_payload(results_dir)
    return {
        "json": payload["json"],
        "generated_at": payload["generated_at"],
    }


def script_safe_json(data: str) -> str:
    return data.replace("&", "\\u0026").replace("<", "\\u003c").replace(">", "\\u003e")


def replace_once(page: str, pattern: re.Pattern[str], value: str, label: str) -> str:
    page, count = pattern.subn(lambda _match: value, page, count=1)
    if count != 1:
        raise RuntimeError(f"expected exactly one {label} placeholder")
    return page


def render(csv_path: Path, out: Path, template_path: Path = DEFAULT_TEMPLATE) -> None:
    if not csv_path.exists():
        raise FileNotFoundError(csv_path)
    if not template_path.exists():
        raise FileNotFoundError(template_path)

    payload = build_dashboard_json(csv_path)
    page = template_path.read_text(encoding="utf-8")
    page = replace_once(
        page,
        BENCH_DATA_RE,
        script_safe_json(payload["json"]),
        "bench_data_json",
    )
    page = replace_once(
        page,
        GENERATED_AT_RE,
        html.escape(payload["generated_at"]),
        "generated_at",
    )

    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(page, encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Render a standalone interactive cross-binding parity dashboard."
    )
    parser.add_argument(
        "--csv",
        required=True,
        type=Path,
        help="orchestrator CSV to render, for example /tmp/n4m_full_parity.csv",
    )
    parser.add_argument(
        "--out",
        required=True,
        type=Path,
        help="output HTML path",
    )
    parser.add_argument(
        "--template",
        default=DEFAULT_TEMPLATE,
        type=Path,
        help="dashboard HTML template; defaults to docs/_templates/landing.html",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    render(args.csv, args.out, args.template)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
