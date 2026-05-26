#!/usr/bin/env python3
"""Render catalog-derived API metadata with per-language Jinja templates."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from catalog_loader import REPO, load_methods

TEMPLATE_DIR = Path(__file__).resolve().parent / "templates"
OUT_DIR = REPO / "build" / "catalog" / "rendered_api"
OUTPUTS = {
    "python": "python_exports.txt",
    "r": "r_namespace.txt",
    "matlab": "matlab_contents.m",
}


def load_jinja():
    try:
        from jinja2 import Environment, FileSystemLoader, StrictUndefined
    except ImportError as exc:  # pragma: no cover - host dependency guard
        raise SystemExit(
            "render_api.py requires jinja2. Install the dev Python "
            "requirements or run inside the supported dev shell."
        ) from exc
    return Environment(
        loader=FileSystemLoader(str(TEMPLATE_DIR)),
        undefined=StrictUndefined,
        trim_blocks=True,
        lstrip_blocks=True,
    )


def render_all() -> dict[str, str]:
    env = load_jinja()
    methods = load_methods(prefer_split=True)
    context = {
        "methods": methods,
        "method_count": len(methods),
    }
    rendered: dict[str, str] = {}
    for language, output_name in OUTPUTS.items():
        template = env.get_template(f"{language}.j2")
        rendered[output_name] = template.render(**context)
        if not rendered[output_name].endswith("\n"):
            rendered[output_name] += "\n"
    return rendered


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--write",
        action="store_true",
        help=f"write rendered files under {OUT_DIR}",
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="compare rendered files with existing outputs",
    )
    args = parser.parse_args(argv)

    rendered = render_all()
    if args.check:
        errors = 0
        for name, content in rendered.items():
            path = OUT_DIR / name
            if not path.exists():
                print(f"missing: {path}", file=sys.stderr)
                errors += 1
            elif path.read_text(encoding="utf-8") != content:
                print(f"stale: {path}", file=sys.stderr)
                errors += 1
        if errors:
            print(f"FAIL: {errors} rendered API file issue(s)", file=sys.stderr)
            return 1
        print(f"PASS: {len(rendered)} rendered API files are up to date")
        return 0

    if args.write:
        OUT_DIR.mkdir(parents=True, exist_ok=True)
        for name, content in rendered.items():
            (OUT_DIR / name).write_text(content, encoding="utf-8")
        print(f"wrote {len(rendered)} files under {OUT_DIR}")
    else:
        for name, content in rendered.items():
            print(f"--- {name} ---")
            print(content, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
