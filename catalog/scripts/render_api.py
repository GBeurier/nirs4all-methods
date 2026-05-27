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


def _abi_version() -> str:
    """Read the current ABI version triple from the canonical header."""
    import re

    hdr = REPO / "cpp" / "include" / "n4m" / "n4m_version.h"
    if not hdr.exists():
        return "unknown"
    txt = hdr.read_text(encoding="utf-8")
    parts = []
    for key in ("MAJOR", "MINOR", "PATCH"):
        m = re.search(rf"N4M_ABI_VERSION_{key}\s+(\d+)", txt)
        if not m:
            return "unknown"
        parts.append(m.group(1))
    return ".".join(parts)


def _real_abi_symbols() -> set[str]:
    """Exported `n4m_*` symbols from the authoritative ABI snapshot (the version
    tag `N4M_1` and comments are not method symbols)."""
    snap = REPO / "cpp" / "abi" / "expected_symbols_linux.txt"
    if not snap.exists():
        return set()
    return {
        ln.strip() for ln in snap.read_text(encoding="utf-8").splitlines()
        if ln.strip() and not ln.startswith("#") and ln.strip().startswith("n4m_")
    }


def render_manifest() -> dict:
    """The raw-binding manifest (bindings/SPEC.md) — a **catalog projection**,
    NOT binding truth. The catalog `abi_symbols` are largely auto-discovered
    guesses, so the manifest also carries an ABI **reconciliation** against the
    authoritative snapshot (`cpp/abi/expected_symbols_linux.txt`): per method,
    which declared symbols are real vs guessed, plus the global set of real
    exported symbols no catalog method claims. This quantifies the Phase-B
    catalog-cleanup debt. Strict raw-class generation is gated on this
    reconciliation reaching ~100 %% (see bindings/SPEC.md); until then the
    manifest is advisory.
    """
    methods = load_methods(prefer_split=True)
    real = _real_abi_symbols()
    out_methods = []
    catalog_syms: set[str] = set()
    for m in methods:
        syms = list(m.get("abi_symbols") or [])
        catalog_syms.update(syms)
        real_syms = [s for s in syms if s in real]
        guessed = [s for s in syms if s not in real]
        out_methods.append({
            "method_id": m.get("method_id"),
            "category": m.get("category"),
            "subcategory": m.get("subcategory"),
            "since_abi": m.get("since_abi"),
            "abi_symbols": syms,
            "real_abi_symbols": real_syms,
            "guessed_abi_symbols": guessed,
        })
    catalog_real = catalog_syms & real
    unmapped = sorted(real - catalog_syms)
    pct = round(100.0 * len(catalog_real) / len(catalog_syms), 1) if catalog_syms else 0.0
    return {
        "abi": _abi_version(),
        "source": "catalog projection (advisory) — see bindings/SPEC.md",
        "method_count": len(methods),
        "reconciliation": {
            "real_abi_symbol_count": len(real),
            "catalog_symbol_count": len(catalog_syms),
            "catalog_real": len(catalog_real),
            "catalog_guessed": len(catalog_syms - real),
            "catalog_coverage_pct": pct,
            "unmapped_real_abi_symbols": unmapped,
            "unmapped_real_count": len(unmapped),
        },
        "methods": out_methods,
    }


def render_all() -> dict[str, str]:
    import json

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
    # The raw-binding manifest is a first-class rendered artifact (JSON, so it
    # rides the same --write/--check path as the per-language symbol lists).
    rendered["raw_manifest.json"] = json.dumps(render_manifest(), indent=2) + "\n"
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
